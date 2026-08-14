"""Application service for durable single-worker queue operations."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol, TypeVar
from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryError,
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepositoryError,
    UpscaleSettingsRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import (
    BatchSeedStrategy,
    CancellationOutcome,
    GenerationBatch,
    GenerationQueueItem,
    QueueHealthCounts,
    SubmissionState,
)
from runpod_sdxl_image_studio.domain.generation_settings import (
    MAX_SEED,
    RANDOM_SEED,
    GenerationSettings,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSourceKind
from runpod_sdxl_image_studio.services.pod_lifecycle_service import (
    PodLifecycleWorkBlockedError,
)

logger = logging.getLogger(__name__)
_AdmissionResult = TypeVar("_AdmissionResult")


class GenerationCancellationAdapterProtocol(Protocol):
    async def cancel_prompt(self, prompt_id: str) -> CancellationResult: ...


@dataclass(frozen=True)
class CancellationResult:
    """Adapter-level result without leaking ComfyUI response details."""

    requested: bool
    confirmed: bool = False
    outcome: CancellationOutcome = CancellationOutcome.UNAVAILABLE
    message: str = ""

    def __post_init__(self) -> None:
        """Keep the Phase 4 boolean constructor compatible while adding outcome."""

        if self.outcome is CancellationOutcome.UNAVAILABLE and self.confirmed:
            object.__setattr__(self, "outcome", CancellationOutcome.CANCELLED)
        elif self.outcome is CancellationOutcome.CANCELLED and not self.confirmed:
            object.__setattr__(self, "confirmed", True)


class GenerationQueueServiceError(RuntimeError):
    """Safe user-facing queue service boundary error."""


@dataclass(frozen=True)
class QueueEnqueueResult:
    item: GenerationQueueItem
    queue_position: int


@dataclass(frozen=True)
class BatchEnqueueResult:
    batch: GenerationBatch
    items: tuple[GenerationQueueItem, ...]


class GenerationQueueService:
    """Validate settings and delegate durable queue mutations to a repository."""

    def __init__(
        self,
        repository: GenerationDispatchQueueRepositoryProtocol,
        settings: Settings,
        cancellation_adapter: GenerationCancellationAdapterProtocol | None = None,
        *,
        upscale_settings_repository: UpscaleSettingsRepositoryProtocol | None = None,
        lifecycle_gate: object | None = None,
        generation_enqueued_callback: Callable[[], object] | None = None,
        state_changed_callback: Callable[[], None] | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._cancellation_adapter = cancellation_adapter
        self._upscale_settings_repository = upscale_settings_repository
        self._lifecycle_gate = lifecycle_gate
        self._generation_enqueued_callback = generation_enqueued_callback
        self._state_changed_callback = state_changed_callback
        self._wake_callback: Callable[[], None] | None = None

    def set_wake_callback(self, callback: Callable[[], None] | None) -> None:
        self._wake_callback = callback

    def enqueue(
        self,
        settings: GenerationSettings,
        *,
        parent_generation_id: UUID | None = None,
    ) -> QueueEnqueueResult:
        self._ensure_work_allowed()
        resolved = settings.model_copy(update={"seed": _resolve_seed(settings.seed)})
        try:
            snapshot = GenerationSettingsSnapshot.from_settings(resolved)
            item = self._run_with_admission(
                lambda: self._repository.enqueue_single(
                    snapshot,
                    parent_generation_id=parent_generation_id,
                    pending_limit=self._settings.queue_max_pending_jobs,
                )
            )
            return QueueEnqueueResult(item=item, queue_position=item.entry.sequence)
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError(_enqueue_error_message(exc, "生成")) from exc

    def enqueue_batch(
        self,
        settings: GenerationSettings,
        *,
        count: int,
        seed_strategy: BatchSeedStrategy | str,
        start_seed: int | None,
        seed_step: int,
        name: str,
    ) -> BatchEnqueueResult:
        self._ensure_work_allowed()
        strategy = _seed_strategy(seed_strategy)
        self._validate_batch(count, strategy, start_seed, seed_step, name)
        seeds = _batch_seeds(count, strategy, start_seed, seed_step)
        try:
            snapshots = tuple(
                GenerationSettingsSnapshot.from_settings(settings.model_copy(update={"seed": seed}))
                for seed in seeds
            )
            batch, items = self._run_with_admission(
                lambda: self._repository.enqueue_batch(
                    snapshots,
                    name=name,
                    seed_strategy=strategy,
                    start_seed=start_seed,
                    seed_step=seed_step,
                    pending_limit=self._settings.queue_max_pending_jobs,
                )
            )
            return BatchEnqueueResult(batch=batch, items=items)
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError(_enqueue_error_message(exc, "バッチ")) from exc

    def list_jobs(
        self,
        *,
        statuses: Sequence[GenerationStatus] | None = None,
        batch_id: UUID | None = None,
        limit: int = 200,
    ) -> tuple[GenerationQueueItem, ...]:
        try:
            return self._repository.list_queue(
                statuses=statuses,
                batch_id=batch_id,
                limit=min(max(1, limit), 200),
            )
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("キューを取得できませんでした。") from exc

    def get_health_counts(self) -> QueueHealthCounts:
        """Read queue counters without the bounded FIFO materialization path."""

        try:
            return self._repository.get_health_counts()
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("queue health counts could not be read") from exc

    def list_recent_failed(self, limit: int = 100) -> tuple[GenerationQueueItem, ...]:
        """Read only recent failures for the System Health error summary."""

        try:
            return self._repository.list_recent_failed(min(max(1, limit), 100))
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("recent queue failures could not be read") from exc

    def get_job_detail(self, generation_id: UUID) -> GenerationQueueItem | None:
        try:
            return self._repository.get_queue_item(generation_id)
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("ジョブ詳細を取得できませんでした。") from exc

    def get_latest_status_candidate(self) -> GenerationQueueItem | None:
        """Return a bounded reload candidate using the repository's DESC/LIMIT query."""

        try:
            return self._repository.get_latest_status_candidate()
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("最新のジョブ状態を取得できませんでした。") from exc

    async def cancel(self, generation_id: UUID) -> GenerationQueueItem:
        try:
            requested = self._run_mutation(
                lambda: self._repository.request_cancel(generation_id),
                wake=False,
            )
            if requested.generation.status in {
                GenerationStatus.COMPLETED,
                GenerationStatus.FAILED,
                GenerationStatus.CANCELLED,
            }:
                return requested
            if (
                requested.job.prompt_id
                and requested.generation.comfy_prompt_id is not None
                and requested.job.prompt_id != requested.generation.comfy_prompt_id
            ):
                raise GenerationQueueServiceError(
                    "GenerationとJobのprompt IDが一致しないためキャンセルできません。"
                )
            if (
                requested.generation.status is GenerationStatus.PENDING
                and requested.entry.worker_id is None
                and requested.entry.submission_state is SubmissionState.READY
            ):
                return self._run_mutation(
                    lambda: self._repository.mark_cancelled(generation_id),
                    wake=False,
                )
            prompt_id = _shared_prompt_id(requested)
            if not prompt_id:
                return requested
            if self._cancellation_adapter is None:
                raise GenerationQueueServiceError("実行中ジョブのキャンセルAdapterが未設定です。")
            result = await self._cancellation_adapter.cancel_prompt(prompt_id)
            if result.outcome is CancellationOutcome.CANCELLED:
                return self._run_mutation(
                    lambda: self._repository.mark_cancelled(generation_id),
                    wake=False,
                )
            if result.outcome in {
                CancellationOutcome.COMPLETED,
                CancellationOutcome.FAILED,
            }:
                current = self._repository.get_queue_item(generation_id)
                return current or requested
            raise GenerationQueueServiceError(
                result.message or "ComfyUIのキャンセル状態を確認できませんでした。"
            )
        except GenerationQueueServiceError:
            raise
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError("ジョブをキャンセルできませんでした。") from exc

        except Exception as exc:  # noqa: BLE001 - hide adapter details at the UI boundary
            raise GenerationQueueServiceError("キャンセルに失敗しました。") from exc

    def cancel_pending(self, generation_id: UUID) -> GenerationQueueItem | None:
        """Cancel an unstarted Generation without making an async prompt request."""

        try:
            return self._run_mutation(
                lambda: self._cancel_pending_item(generation_id),
                wake=False,
            )
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError(
                "未開始のジョブをキャンセルできませんでした。"
            ) from exc

    def _cancel_pending_item(self, generation_id: UUID) -> GenerationQueueItem | None:
        item = self._repository.get_queue_item(generation_id)
        if item is None:
            return None
        if item.generation.status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }:
            return item
        if (
            item.generation.status is GenerationStatus.PENDING
            and item.entry.worker_id is None
            and item.entry.submission_state is SubmissionState.READY
        ):
            return self._repository.mark_cancelled(generation_id)
        return self._repository.request_cancel(generation_id)

    def link_ambiguous_prompt(self, generation_id: UUID, prompt_id: str) -> GenerationQueueItem:
        """Manually attach a prompt ID after an ambiguous submission."""

        try:
            return self._run_mutation(
                lambda: self._repository.link_ambiguous_prompt(generation_id, prompt_id)
            )
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError("曖昧なprompt状態を手動解決できませんでした") from exc

    def fail_ambiguous_prompt(self, generation_id: UUID) -> GenerationQueueItem:
        """Explicitly mark an ambiguous submission failed when prompt is absent."""

        try:
            return self._run_mutation(lambda: self._repository.fail_ambiguous_prompt(generation_id))
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError(
                "曖昧なprompt状態をfailedへ確定できませんでした"
            ) from exc

    def retry(self, generation_id: UUID) -> QueueEnqueueResult:
        self._ensure_work_allowed()
        try:
            item = self._repository.get_queue_item(generation_id)
            if item is None:
                raise GenerationQueueServiceError("対象ジョブが見つかりません。")
            if item.generation.status not in {GenerationStatus.FAILED, GenerationStatus.CANCELLED}:
                raise GenerationQueueServiceError(
                    "失敗またはキャンセル済みジョブだけ再試行できます。"
                )
            new_item = self._run_with_admission(lambda: self._enqueue_retry_item(item))
            return QueueEnqueueResult(new_item, new_item.entry.sequence)
        except GenerationQueueServiceError:
            raise
        except (
            GenerationDispatchQueueRepositoryError,
            UpscaleSettingsRepositoryError,
            ValueError,
        ) as exc:
            raise GenerationQueueServiceError(
                _enqueue_error_message(exc, "ジョブの再試行")
            ) from exc

    def retry_failed_batch(self, batch_id: UUID) -> BatchEnqueueResult | None:
        self._ensure_work_allowed()
        try:
            source_items = self._repository.list_batch_items(batch_id)
            failed = [
                item for item in source_items if item.generation.status is GenerationStatus.FAILED
            ]
            if not failed:
                return None
            source_batch = next(
                (item.batch for item in source_items if item.batch is not None), None
            )
            if source_batch is None:
                raise GenerationQueueServiceError("対象バッチが見つかりません。")
            snapshots = tuple(item.generation.settings_snapshot for item in failed)
            batch, items = self._run_with_admission(
                lambda: self._repository.enqueue_batch(
                    snapshots,
                    name=f"{source_batch.name} (failed retry)",
                    seed_strategy=source_batch.seed_strategy,
                    start_seed=source_batch.start_seed,
                    seed_step=source_batch.seed_step,
                    retry_of_batch_id=source_batch.id,
                    retry_of_generations=tuple(item.generation.id for item in failed),
                    retry_attempts=tuple(item.generation.retry_attempt + 1 for item in failed),
                    pending_limit=self._settings.queue_max_pending_jobs,
                )
            )
            return BatchEnqueueResult(batch, items)
        except GenerationQueueServiceError:
            raise
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError(
                _enqueue_error_message(exc, "失敗ジョブの再試行")
            ) from exc

    def _ensure_pending_capacity(self, additional: int) -> None:
        if additional < 1:
            raise GenerationQueueServiceError("キュー追加数が不正です。")
        try:
            pending = self._repository.list_queue(
                statuses=(GenerationStatus.PENDING,),
                limit=self._settings.queue_max_pending_jobs + 1,
            )
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("待機中ジョブ数を確認できませんでした。") from exc
        if len(pending) + additional > self._settings.queue_max_pending_jobs:
            raise GenerationQueueServiceError("待機中ジョブ数の上限に達しています。")

    def _ensure_work_allowed(self) -> None:
        if self._lifecycle_gate is None:
            return
        ensure = getattr(self._lifecycle_gate, "ensure_work_allowed", None)
        if callable(ensure):
            try:
                ensure()
            except Exception as exc:  # noqa: BLE001 - keep UI error boundary stable
                if isinstance(exc, GenerationQueueServiceError):
                    raise
                raise GenerationQueueServiceError(str(exc)) from exc

    def _admission_context(self) -> AbstractContextManager[object]:
        if self._lifecycle_gate is None:
            return nullcontext()
        admit = getattr(self._lifecycle_gate, "admit_work", None)
        return admit() if callable(admit) else nullcontext()

    def _run_with_admission(self, action: Callable[[], _AdmissionResult]) -> _AdmissionResult:
        try:
            with self._admission_context():
                result = action()
                self._wake()
                self._notify_generation_enqueued()
                return result
        except GenerationQueueServiceError:
            raise
        except PodLifecycleWorkBlockedError as exc:
            raise GenerationQueueServiceError("新しい処理をキューへ追加できませんでした。") from exc

    def _run_mutation(
        self,
        action: Callable[[], _AdmissionResult],
        *,
        wake: bool = True,
    ) -> _AdmissionResult:
        try:
            with self._admission_context():
                result = action()
                if wake:
                    self._wake()
                self._notify_state_changed()
                return result
        except PodLifecycleWorkBlockedError as exc:
            raise GenerationQueueServiceError(
                "persistent mutation is blocked while Pod termination is draining"
            ) from exc

    def _enqueue_retry_item(self, item: GenerationQueueItem) -> GenerationQueueItem:
        if item.generation.kind.value == "upscale":
            if self._upscale_settings_repository is None:
                raise GenerationQueueServiceError("アップスケール設定Repositoryが未設定です。")
            upscale_snapshot = self._upscale_settings_repository.get_by_generation(
                item.generation.id
            )
            if (upscale_snapshot is None or item.generation.parent_generation_id is None) and (
                upscale_snapshot is None
                or upscale_snapshot.source_kind is not UpscaleSourceKind.METADATA_IMPORT
                or upscale_snapshot.source_import_id is None
            ):
                raise GenerationQueueServiceError("アップスケール設定を復元できません。")
            if upscale_snapshot.source_kind is UpscaleSourceKind.METADATA_IMPORT:
                return self._repository.enqueue_upscale(
                    item.generation.settings_snapshot,
                    upscale_snapshot,
                    parent_generation_id=None,
                    source_artifact_id=None,
                    source_import_id=upscale_snapshot.source_import_id,
                    retry_of_generation_id=item.generation.id,
                    retry_attempt=item.generation.retry_attempt + 1,
                    pending_limit=self._settings.queue_max_pending_jobs,
                )
            if (
                item.generation.parent_generation_id is None
                or upscale_snapshot.source_artifact_id is None
            ):
                raise GenerationQueueServiceError("アップスケール設定を復元できません。")
            return self._repository.enqueue_upscale(
                item.generation.settings_snapshot,
                upscale_snapshot,
                parent_generation_id=item.generation.parent_generation_id,
                source_artifact_id=upscale_snapshot.source_artifact_id,
                retry_of_generation_id=item.generation.id,
                retry_attempt=item.generation.retry_attempt + 1,
                pending_limit=self._settings.queue_max_pending_jobs,
            )
        return self._repository.enqueue_single(
            item.generation.settings_snapshot,
            kind=item.generation.kind,
            parent_generation_id=item.generation.parent_generation_id,
            retry_of_generation_id=item.generation.id,
            retry_attempt=item.generation.retry_attempt + 1,
            pending_limit=self._settings.queue_max_pending_jobs,
        )

    def _notify_generation_enqueued(self) -> None:
        if self._generation_enqueued_callback is None:
            return
        try:
            self._generation_enqueued_callback()
        except Exception as exc:  # noqa: BLE001 - queue persistence already succeeded
            logger.warning(
                "generation enqueue lifecycle arm failed error=%s",
                type(exc).__name__,
            )

    def _wake(self) -> None:
        if self._wake_callback is not None:
            self._wake_callback()

    def _notify_state_changed(self) -> None:
        if self._state_changed_callback is None:
            return
        try:
            self._state_changed_callback()
        except Exception:  # noqa: BLE001 - notification is best effort after commit
            logger.warning("queue state change notification failed", exc_info=True)

    def _validate_batch(
        self,
        count: int,
        strategy: BatchSeedStrategy,
        start_seed: int | None,
        seed_step: int,
        name: str,
    ) -> None:
        if not 1 <= count <= self._settings.batch_max_items:
            raise GenerationQueueServiceError("バッチ枚数が範囲外です。")
        if not name.strip() or len(name.strip()) > 200:
            raise GenerationQueueServiceError("バッチ名を確認してください。")
        if seed_step <= 0:
            raise GenerationQueueServiceError("seed増分は1以上にしてください。")
        if strategy is BatchSeedStrategy.SEQUENTIAL and start_seed is None:
            raise GenerationQueueServiceError("連番seedには開始seedが必要です。")
        if start_seed is not None and not 0 <= start_seed <= MAX_SEED:
            raise GenerationQueueServiceError("開始seedが範囲外です。")
        if strategy is BatchSeedStrategy.SEQUENTIAL and start_seed is not None:
            final_seed = start_seed + (count - 1) * seed_step
            if final_seed > MAX_SEED:
                raise GenerationQueueServiceError("バッチseedが上限を超えます。")


def _resolve_seed(seed: int) -> int:
    return secrets.randbelow(MAX_SEED + 1) if seed == RANDOM_SEED else seed


def _enqueue_error_message(error: Exception, subject: str) -> str:
    if "capacity" in str(error):
        return "キュー待ち件数の上限に達しています。"
    return f"{subject}をキューへ追加できませんでした。"


def _batch_seeds(
    count: int,
    strategy: BatchSeedStrategy,
    start_seed: int | None,
    seed_step: int,
) -> tuple[int, ...]:
    if strategy is BatchSeedStrategy.RANDOM:
        return tuple(_resolve_seed(RANDOM_SEED) for _ in range(count))
    if start_seed is None:
        raise GenerationQueueServiceError("連番seedには開始seedが必要です。")
    return tuple(start_seed + index * seed_step for index in range(count))


def _seed_strategy(value: BatchSeedStrategy | str) -> BatchSeedStrategy:
    try:
        return value if isinstance(value, BatchSeedStrategy) else BatchSeedStrategy(value)
    except ValueError as exc:
        raise GenerationQueueServiceError("seed方式が不正です。") from exc


__all__ = [
    "BatchEnqueueResult",
    "CancellationResult",
    "GenerationCancellationAdapterProtocol",
    "GenerationQueueService",
    "GenerationQueueServiceError",
    "MAX_SEED",
    "QueueEnqueueResult",
]


def _shared_prompt_id(item: GenerationQueueItem) -> str | None:
    """Return the only safe prompt ID, rejecting a Generation/Job mismatch."""

    generation_prompt = item.generation.comfy_prompt_id
    job_prompt = item.job.prompt_id
    if generation_prompt is not None and job_prompt is not None and generation_prompt != job_prompt:
        return None
    return job_prompt or generation_prompt
