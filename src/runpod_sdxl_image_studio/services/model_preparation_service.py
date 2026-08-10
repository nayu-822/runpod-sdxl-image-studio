"""Application service for safe remote model selection and preparation."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar
from uuid import UUID

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.database.repositories.model_transfer_repository import (
    ModelTransferRepositoryError,
    ModelTransferRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.rclone.remote_model_catalog import (
    CancelCheck,
    ProcessFinishedCallback,
    ProcessStartedCallback,
    ProgressCallback,
    RemoteModelAdapterError,
    ShutdownCheck,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.model_transfer import (
    ModelTransferErrorCode,
    ModelTransferJob,
    ModelTransferProgress,
    ModelTransferStatus,
    RemoteModelCatalog,
    RemoteModelEntry,
    RemoteModelKind,
    normalize_model_relative_path,
)
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.pod_lifecycle_service import (
    PodLifecycleWorkBlockedError,
)

logger = logging.getLogger(__name__)
_AdmissionResult = TypeVar("_AdmissionResult")


class ModelCatalogPort(Protocol):
    async def list_catalog(self) -> RemoteModelCatalog: ...

    async def download(
        self,
        entry: RemoteModelEntry,
        destination: Path,
        *,
        progress_callback: ProgressCallback | None = None,
        process_started_callback: ProcessStartedCallback | None = None,
        process_finished_callback: ProcessFinishedCallback | None = None,
        cancel_check: CancelCheck | None = None,
        shutdown_check: ShutdownCheck | None = None,
        timeout_seconds: float | None = None,
    ) -> None: ...


class LoraCatalogPort(Protocol):
    def sync_with_capabilities(
        self, file_names: Sequence[str], *, capability_success: bool = True
    ) -> object: ...


class ModelPreparationServiceError(RuntimeError):
    """Safe application error with a stable UI-facing code."""

    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ModelPreparationResult:
    jobs: tuple[ModelTransferJob, ...]
    message: str
    catalog: RemoteModelCatalog | None = None
    missing: tuple[str, ...] = ()


class ModelPreparationService:
    """Keep remote snapshots, local paths, and ComfyUI capabilities separate."""

    def __init__(
        self,
        repository: ModelTransferRepositoryProtocol,
        catalog_adapter: ModelCatalogPort,
        settings: Settings,
        capability_refresh: Callable[[], Awaitable[CapabilityRefreshResult]],
        *,
        lora_catalog_service: LoraCatalogPort | None = None,
        state_changed_callback: Callable[[], None] | None = None,
        work_gate: object | None = None,
    ) -> None:
        self._repository = repository
        self._catalog_adapter = catalog_adapter
        self._settings = settings
        self._capability_refresh = capability_refresh
        self._lora_catalog_service = lora_catalog_service
        self._state_changed_callback = state_changed_callback
        self._work_gate = work_gate
        self._catalog: RemoteModelCatalog | None = None

    async def refresh_catalog(self) -> RemoteModelCatalog:
        if not self._settings.remote_model_enabled:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.NOT_CONFIGURED.value,
                "Google Drive model catalog is disabled",
                retryable=False,
            )
        try:
            catalog = await self._catalog_adapter.list_catalog()
        except RemoteModelAdapterError as exc:
            raise ModelPreparationServiceError(exc.code, str(exc)) from exc
        if not catalog.is_available:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.CATALOG_UNAVAILABLE.value,
                "Google Drive model catalog is unavailable",
            )
        self._catalog = catalog
        return catalog

    def current_catalog(self) -> RemoteModelCatalog | None:
        return self._catalog

    async def prepare_selected(
        self,
        checkpoint: str | None,
        vae: str | None,
        loras: Sequence[str] | None,
        upscaler: str | None,
    ) -> ModelPreparationResult:
        self._ensure_work_allowed()
        catalog = await self.refresh_catalog()
        selections: list[tuple[RemoteModelKind, str]] = []
        for kind, value in (
            (RemoteModelKind.CHECKPOINT, checkpoint),
            (RemoteModelKind.VAE, vae),
            (RemoteModelKind.UPSCALER, upscaler),
        ):
            if value and value.strip():
                selections.append((kind, value))
        selected_loras = tuple(
            dict.fromkeys(item.strip() for item in (loras or ()) if item.strip())
        )
        if len(selected_loras) > self._settings.max_loras:
            raise ModelPreparationServiceError(
                "too_many_loras",
                f"LoRAは最大{self._settings.max_loras}件まで選択できます",
                retryable=False,
            )
        selections.extend((RemoteModelKind.LORA, item) for item in selected_loras)
        if not selections:
            return ModelPreparationResult((), "準備するモデルを選択してください。", catalog)

        resolved_entries: list[RemoteModelEntry] = []
        for kind, relative in selections:
            try:
                entry = catalog.find(kind, relative)
            except ValueError as exc:
                raise ModelPreparationServiceError(
                    ModelTransferErrorCode.INVALID_REMOTE_ENTRY.value,
                    "モデル選択が不正です。",
                    retryable=False,
                ) from exc
            if entry is None:
                raise ModelPreparationServiceError(
                    ModelTransferErrorCode.INVALID_REMOTE_ENTRY.value,
                    "選択したモデルがRemote一覧にありません。再取得してください。",
                    retryable=False,
                )
            resolved_entries.append(entry)

        # Resolve and validate every selection before persisting the first job.
        # This keeps an invalid later selection from leaving an earlier download queued.
        jobs = [await self.prepare_entry(entry) for entry in resolved_entries]
        self._notify_state_changed()
        return ModelPreparationResult(
            tuple(jobs), f"{len(jobs)}件のモデル準備をキューへ登録しました。", catalog
        )

    async def prepare_previous_models(
        self,
        checkpoint: str | None,
        vae: str | None,
        loras: Sequence[str] | None,
        upscaler: str | None = None,
    ) -> ModelPreparationResult:
        """Queue exact restored selections independently and report missing ones."""

        self._ensure_work_allowed()
        catalog = await self.refresh_catalog()
        selections: list[tuple[RemoteModelKind, str]] = []
        for kind, value in (
            (RemoteModelKind.CHECKPOINT, checkpoint),
            (RemoteModelKind.VAE, vae),
            (RemoteModelKind.UPSCALER, upscaler),
        ):
            if value and value.strip():
                selections.append((kind, value.strip()))
        selected_loras = tuple(
            dict.fromkeys(item.strip() for item in (loras or ()) if item.strip())
        )
        if len(selected_loras) > self._settings.max_loras:
            raise ModelPreparationServiceError(
                "too_many_loras",
                f"at most {self._settings.max_loras} LoRAs may be restored",
                retryable=False,
            )
        selections.extend((RemoteModelKind.LORA, item) for item in selected_loras)

        jobs: list[ModelTransferJob] = []
        missing: list[str] = []
        for kind, relative in selections:
            try:
                entry = catalog.find(kind, relative)
            except ValueError:
                entry = None
            if entry is None:
                missing.append(f"{kind.value}:{relative}")
                continue
            try:
                jobs.append(await self.prepare_entry(entry))
            except ModelPreparationServiceError as exc:
                missing.append(f"{kind.value}:{relative} ({exc.code})")
        message = (
            "Some restored models are unavailable; no substitute was selected"
            if missing
            else f"{len(jobs)} restored model selections were queued"
        )
        return ModelPreparationResult(tuple(jobs), message, catalog, tuple(missing))

    async def prepare_entry(self, entry: RemoteModelEntry) -> ModelTransferJob:
        local_path = self.local_path_for(entry)
        local_relative = entry.relative_path
        existing_sha = _matching_local_sha256(local_path, entry)
        if existing_sha is not None:
            try:
                await self._refresh_and_check_visibility(entry)
            except ModelPreparationServiceError:
                raise
            return self._persist_entry(entry, local_relative, existing_sha)
        return self._persist_entry(entry, local_relative, None)

    def _persist_entry(
        self,
        entry: RemoteModelEntry,
        local_relative: str,
        existing_sha: str | None,
    ) -> ModelTransferJob:
        try:
            with self._admission_context():
                job = self._repository.enqueue(entry, local_relative)
                if existing_sha is None:
                    self._notify_state_changed()
                    return job
                try:
                    result = self._repository.mark_already_prepared(job.id, existing_sha)
                    self._notify_state_changed()
                    return result
                except AttributeError:
                    # Keep the service usable with small test fakes that only implement
                    # the original repository protocol.
                    return job
        except PodLifecycleWorkBlockedError as exc:
            raise ModelPreparationServiceError(
                "pod_lifecycle_draining",
                "Pod is preparing to terminate; new work is blocked",
                retryable=False,
            ) from exc
        except ModelTransferRepositoryError as exc:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                "モデル準備ジョブを保存できませんでした。",
            ) from exc

    def local_path_for(self, entry: RemoteModelEntry) -> Path:
        try:
            relative = normalize_model_relative_path(entry.relative_path)
            root = self._root_for_kind(entry.kind).resolve()
            candidate = root.joinpath(*Path(relative).parts)
            resolved_parent = candidate.parent.resolve()
            resolved_parent.relative_to(root)
            if candidate.is_symlink():
                raise ValueError("model destination is a symlink")
            if candidate.exists():
                candidate.resolve(strict=True).relative_to(root)
            return candidate
        except (OSError, ValueError) as exc:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.INVALID_LOCAL_PATH.value,
                "モデル保存先が安全ではありません。",
                retryable=False,
            ) from exc

    async def process_job(
        self,
        job: ModelTransferJob,
        worker_id: str,
        *,
        shutdown_check: ShutdownCheck | None = None,
    ) -> ModelTransferJob:
        if shutdown_check is not None and await _invoke_check(shutdown_check):
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value,
                "モデル準備はアプリケーション終了により中断されました。",
            )
        entry = RemoteModelEntry(
            kind=job.kind,
            relative_path=job.remote_relative_path,
            display_name=job.remote_relative_path,
            size_bytes=job.remote_size_bytes,
            modified_at=job.remote_modified_at,
            remote_hash_algorithm=job.remote_hash_algorithm,
            remote_hash=job.remote_hash,
        )
        final_path = self.local_path_for(entry)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self._assert_contained(final_path, entry.kind)
        temp_path = final_path.parent / f".{job.id}.download"
        temp_path.unlink(missing_ok=True)

        async def on_progress(progress: ModelTransferProgress) -> None:
            try:
                self._repository.update_progress(job.id, worker_id, progress)
            except ModelTransferRepositoryError:
                logger.warning("model transfer progress persistence failed", exc_info=True)
            self._notify_state_changed()

        async def on_started(pid: int) -> None:
            if not self._repository.update_process(job.id, worker_id, pid):
                raise ModelPreparationServiceError(
                    ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                    "モデル転送プロセスを追跡できません。",
                )
            self._notify_state_changed()

        async def on_finished() -> None:
            try:
                self._repository.update_process(job.id, worker_id, None)
            except ModelTransferRepositoryError:
                logger.warning("model transfer process cleanup persistence failed", exc_info=True)

        async def cancel_check() -> bool:
            current = self._repository.get(job.id)
            return current is not None and current.status is ModelTransferStatus.CANCEL_REQUESTED

        try:
            download_kwargs: dict[str, Any] = {
                "progress_callback": on_progress,
                "process_started_callback": on_started,
                "process_finished_callback": on_finished,
                "cancel_check": cancel_check,
                "timeout_seconds": self._settings.remote_model_download_timeout_seconds,
            }
            if shutdown_check is not None:
                download_kwargs["shutdown_check"] = shutdown_check
            await self._catalog_adapter.download(entry, temp_path, **download_kwargs)
            _verify_file(temp_path, entry)
            local_sha256 = _sha256(temp_path)
            self._assert_contained(final_path, entry.kind)
            os.replace(temp_path, final_path)
            visibility = await self._refresh_and_check_visibility(entry)
            del visibility
            try:
                completed = self._repository.mark_completed(job.id, worker_id, local_sha256)
            except ModelTransferRepositoryError as exc:
                with contextlib.suppress(Exception):
                    self._repository.mark_failed(
                        job.id,
                        worker_id,
                        ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                        "モデル実体は保存されましたが、完了状態をDBへ保存できませんでした。",
                        retryable=True,
                    )
                raise ModelPreparationServiceError(
                    ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                    "モデル実体は保存されましたが、完了状態を保存できませんでした。",
                ) from exc
            self._notify_state_changed()
            return completed
        except RemoteModelAdapterError as exc:
            if exc.code == ModelTransferErrorCode.CANCELLED.value:
                raise ModelPreparationServiceError(
                    exc.code, "モデル準備をキャンセルしました。"
                ) from exc
            raise ModelPreparationServiceError(exc.code, "モデル準備に失敗しました。") from exc
        except ModelPreparationServiceError:
            raise
        except (OSError, ValueError) as exc:
            code = (
                ModelTransferErrorCode.SIZE_MISMATCH.value
                if "size" in str(exc).casefold()
                else ModelTransferErrorCode.HASH_MISMATCH.value
                if "hash" in str(exc).casefold()
                else ModelTransferErrorCode.DOWNLOAD_FAILED.value
            )
            raise ModelPreparationServiceError(
                code, "モデルファイルの検証に失敗しました。"
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)

    async def _refresh_and_check_visibility(self, entry: RemoteModelEntry) -> ComfyUICapabilities:
        raw_result = await self._capability_refresh()
        if not raw_result.is_success:
            raise ModelPreparationServiceError(
                "comfyui_capability_refresh_failed",
                "ComfyUIのモデル一覧を更新できませんでした。",
            )
        capabilities = raw_result.capabilities
        if capabilities is None:
            raise ModelPreparationServiceError(
                "comfyui_capability_refresh_failed",
                "ComfyUIのモデル一覧を取得できませんでした。",
            )
        choices = {
            RemoteModelKind.CHECKPOINT: capabilities.checkpoints,
            RemoteModelKind.LORA: capabilities.loras,
            RemoteModelKind.VAE: capabilities.vaes,
            RemoteModelKind.UPSCALER: capabilities.upscale_models,
        }[entry.kind]
        if entry.relative_path not in choices:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.MODEL_NOT_VISIBLE.value,
                "保存したモデルがComfyUIの一覧に表示されません。",
            )
        if entry.kind is RemoteModelKind.LORA and self._lora_catalog_service is not None:
            try:
                self._lora_catalog_service.sync_with_capabilities(
                    capabilities.loras,
                    capability_success=True,
                )
            except Exception:  # noqa: BLE001 - metadata catalog is best effort after visibility
                logger.warning("LoRA metadata catalog sync failed", exc_info=True)
        return capabilities

    def cancel(self, job_id: UUID) -> ModelTransferJob:
        try:
            result = self._run_with_admission(lambda: self._repository.request_cancel(job_id))
        except ModelTransferRepositoryError as exc:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                "モデル準備のキャンセルを保存できませんでした。",
            ) from exc
        return result

    async def retry(self, job_id: UUID) -> ModelTransferJob:
        self._ensure_work_allowed()
        catalog = await self.refresh_catalog()
        current = self._repository.get(job_id)
        if current is None:
            raise ModelPreparationServiceError(
                "model_transfer_not_found", "対象ジョブが見つかりません。"
            )
        entry = catalog.find(current.kind, current.remote_relative_path)
        if entry is None:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.INVALID_REMOTE_ENTRY.value,
                "Remoteモデルが見つかりません。再取得してください。",
                retryable=False,
            )
        try:
            result = self._run_with_admission(
                lambda: self._repository.retry(job_id, entry, entry.relative_path)
            )
        except ModelTransferRepositoryError as exc:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                "モデル準備を再試行できませんでした。",
            ) from exc
        return result

    def list_jobs(self, limit: int = 100) -> tuple[ModelTransferJob, ...]:
        try:
            return self._repository.list_jobs(limit)
        except ModelTransferRepositoryError as exc:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                "モデル準備状況を取得できませんでした。",
            ) from exc

    def reconcile_stateless_restore(self) -> int:
        try:
            count = self._repository.reconcile_stateless_restore()
        except ModelTransferRepositoryError as exc:
            raise ModelPreparationServiceError(
                ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                "モデル準備のstateless復旧に失敗しました。",
            ) from exc
        if count:
            self._notify_state_changed()
        return count

    async def reconcile_files(self) -> int:
        repaired = 0
        for job in self.list_jobs(500):
            if job.status is ModelTransferStatus.COMPLETED:
                continue
            entry = RemoteModelEntry(
                job.kind,
                job.remote_relative_path,
                job.remote_relative_path,
                job.remote_size_bytes,
                job.remote_modified_at,
                job.remote_hash_algorithm,
                job.remote_hash,
            )
            path = self.local_path_for(entry)
            if not path.is_file() or path.is_symlink():
                continue
            try:
                _verify_file(path, entry)
                local_sha256 = _sha256(path)
                await self._refresh_and_check_visibility(entry)
                repaired_job = self._repository.repair_completed(job.id, local_sha256)
                if repaired_job.status is ModelTransferStatus.COMPLETED:
                    repaired += 1
            except (
                OSError,
                ValueError,
                ModelTransferRepositoryError,
                ModelPreparationServiceError,
            ):
                continue
        if repaired:
            self._notify_state_changed()
        return repaired

    def _root_for_kind(self, kind: RemoteModelKind) -> Path:
        return {
            RemoteModelKind.CHECKPOINT: self._settings.checkpoint_dir,
            RemoteModelKind.LORA: self._settings.lora_dir,
            RemoteModelKind.VAE: self._settings.vae_dir,
            RemoteModelKind.UPSCALER: self._settings.upscaler_dir,
        }[kind]

    def _assert_contained(self, path: Path, kind: RemoteModelKind) -> None:
        root = self._root_for_kind(kind).resolve()
        path.parent.resolve().relative_to(root)
        if path.exists():
            path.resolve(strict=True).relative_to(root)

    def _notify_state_changed(self) -> None:
        if self._state_changed_callback is None:
            return
        try:
            self._state_changed_callback()
        except Exception:  # noqa: BLE001 - backup notification is best effort
            logger.warning("model transfer state backup notification failed", exc_info=True)

    def _ensure_work_allowed(self) -> None:
        if self._work_gate is None:
            return
        ensure = getattr(self._work_gate, "ensure_work_allowed", None)
        if callable(ensure):
            try:
                ensure()
            except PodLifecycleWorkBlockedError as exc:
                raise ModelPreparationServiceError(
                    "pod_lifecycle_draining",
                    "Pod is preparing to terminate; new work is blocked",
                    retryable=False,
                ) from exc

    def _admission_context(self) -> AbstractContextManager[object]:
        if self._work_gate is None:
            return nullcontext()
        admit = getattr(self._work_gate, "admit_work", None)
        return admit() if callable(admit) else nullcontext()

    def _run_with_admission(self, action: Callable[[], _AdmissionResult]) -> _AdmissionResult:
        try:
            with self._admission_context():
                result = action()
                self._notify_state_changed()
                return result
        except PodLifecycleWorkBlockedError as exc:
            raise ModelPreparationServiceError(
                "pod_lifecycle_draining",
                "Pod is preparing to terminate; new work is blocked",
                retryable=False,
            ) from exc


def _matching_local_sha256(path: Path, entry: RemoteModelEntry) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        if path.stat().st_size != entry.size_bytes:
            return None
        digest = _sha256(path)
        if entry.remote_hash is None:
            return None
        algorithm = (entry.remote_hash_algorithm or "sha-256").replace("_", "-").casefold()
        if algorithm in {"sha-256", "sha256"} and digest != entry.remote_hash.casefold():
            return None
        if algorithm == "md5" and _hash(path, "md5") != entry.remote_hash.casefold():
            return None
        return digest
    except OSError:
        return None


def _verify_file(path: Path, entry: RemoteModelEntry) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("downloaded model is not a regular file")
    if path.stat().st_size != entry.size_bytes:
        raise ValueError("model size mismatch")
    if entry.remote_hash is None:
        return
    algorithm = (entry.remote_hash_algorithm or "sha-256").replace("_", "-").casefold()
    digest = _sha256(path) if algorithm in {"sha-256", "sha256"} else _hash(path, algorithm)
    if digest != entry.remote_hash.casefold():
        raise ValueError("model hash mismatch")


def _sha256(path: Path) -> str:
    return _hash(path, "sha256")


def _hash(path: Path, algorithm: str) -> str:
    normalized = algorithm.replace("-", "").casefold()
    digest = hashlib.new(normalized)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _invoke_check(callback: ShutdownCheck) -> bool:
    result = callback()
    if hasattr(result, "__await__"):
        result = await result
    return bool(result)


__all__ = [
    "ModelPreparationResult",
    "ModelPreparationService",
    "ModelPreparationServiceError",
]
