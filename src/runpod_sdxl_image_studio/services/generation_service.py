"""Application service for one fixed SDXL generation job."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIError,
    ComfyUIPromptError,
    ComfyUIWebSocketError,
    WorkflowError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import PromptHistory
from runpod_sdxl_image_studio.adapters.comfyui.websocket_client import ComfyUIWebSocketClient
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationCompletionRepositoryProtocol,
    GenerationFailureRepositoryProtocol,
    GenerationJobRepositoryProtocol,
    GenerationQueueRepositoryProtocol,
    GenerationRepositoryError,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.generation_metadata_storage import (
    GenerationMetadataStorage,
)
from runpod_sdxl_image_studio.adapters.storage.history_thumbnail_storage import (
    HistoryThumbnailStorage,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import (
    GenerationErrorCode,
    GenerationKind,
    GenerationProgress,
    GenerationResult,
    GenerationStatus,
    StoredImage,
)
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.job import GenerationJob
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.generation_errors import (
    ArtifactPersistenceError,
    CompletionPersistenceError,
    FailurePersistenceError,
    GenerationPersistenceError,
    PromptPersistenceError,
    RecoveryPersistenceError,
    persistence_error_code,
    persistence_error_message,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[GenerationProgress], None]


class HistoryTimeoutError(ComfyUIPromptError):
    """History polling reached its configured attempt limit."""


class CapabilitiesProvider(Protocol):
    def __call__(self) -> Awaitable[CapabilityRefreshResult]: ...


class LoraUsageRecorder(Protocol):
    def record_usage(self, file_names: tuple[str, ...], completed_at: datetime) -> None: ...


class GenerationService:
    """Coordinate validation, queueing, monitoring, recovery, and local storage."""

    def __init__(
        self,
        client: ComfyUIClient,
        workflow_adapter: WorkflowAdapter,
        websocket_client: ComfyUIWebSocketClient,
        storage: LocalStorageAdapter,
        capabilities_provider: CapabilitiesProvider,
        settings: Settings | None = None,
        lora_catalog_service: LoraUsageRecorder | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        id_factory: Callable[[], UUID] = uuid4,
        generation_repository: GenerationRepositoryProtocol | None = None,
        artifact_repository: GenerationArtifactRepositoryProtocol | None = None,
        completion_repository: GenerationCompletionRepositoryProtocol | None = None,
        job_repository: GenerationJobRepositoryProtocol | None = None,
        queue_repository: GenerationQueueRepositoryProtocol | None = None,
        failure_repository: GenerationFailureRepositoryProtocol | None = None,
        thumbnail_storage: HistoryThumbnailStorage | None = None,
        metadata_storage: GenerationMetadataStorage | None = None,
    ) -> None:
        self._client = client
        self._workflow_adapter = workflow_adapter
        self._websocket_client = websocket_client
        self._storage = storage
        self._capabilities_provider = capabilities_provider
        self._settings = settings or get_settings()
        self._lora_catalog_service = lora_catalog_service
        self._sleep = sleep
        self._id_factory = id_factory
        self._generation_repository = generation_repository
        self._artifact_repository = artifact_repository
        self._completion_repository = completion_repository
        self._job_repository = job_repository
        self._queue_repository = queue_repository
        self._failure_repository = failure_repository
        self._thumbnail_storage = thumbnail_storage
        self._metadata_storage = metadata_storage
        repositories = (
            generation_repository,
            job_repository,
            artifact_repository,
            queue_repository,
            completion_repository,
            failure_repository,
        )
        if any(repository is not None for repository in repositories) and any(
            repository is None for repository in repositories
        ):
            raise ValueError("generation persistence repositories must be configured together")
        self._jobs: dict[UUID, GenerationJob] = {}
        self._results: dict[UUID, GenerationResult] = {}
        self._lock = asyncio.Lock()

    @property
    def jobs(self) -> Mapping[UUID, GenerationJob]:
        return self._jobs

    def get_result(self, generation_id: UUID) -> GenerationResult | None:
        """Return a result while the process is alive."""

        return self._results.get(generation_id)

    async def recover_prompt(self, generation_id: UUID, prompt_id: str) -> bool:
        """Complete one existing prompt from history without submitting it again."""

        if (
            self._generation_repository is None
            or self._job_repository is None
            or self._artifact_repository is None
        ):
            return False
        try:
            generation = self._generation_repository.get_by_id(generation_id)
            job = self._job_repository.get_by_generation(generation_id)
            if generation is None or job is None:
                return False
            if generation.status is GenerationStatus.COMPLETED:
                return True
            if generation.status in {GenerationStatus.FAILED, GenerationStatus.CANCELLED}:
                return False
            existing_image = self._artifact_repository.get_primary_image(generation_id)
            if existing_image is not None:
                self._complete_existing_generation(generation_id, job.id)
                return True
            history = await self._client.get_prompt_history(prompt_id)
            if history.is_failed:
                self._mark_failed_pair(
                    job,
                    GenerationErrorCode.COMFYUI_EXECUTION.value,
                    "ComfyUIで生成が失敗しました。",
                )
                return False
            if not history.is_completed or not history.outputs:
                return False
            image_bytes = await self._client.get_output_image(history.outputs[0])
            stored_image = self._storage.store_image(
                image_bytes, generation_id, generation.created_at
            )
            recovery_job = GenerationJob(
                generation_id=generation_id,
                status=GenerationStatus.RUNNING,
                id=job.id,
                prompt_id=prompt_id,
                created_at=generation.created_at,
                stored_image=stored_image,
            )
            self._complete_job(
                recovery_job,
                generation.settings_snapshot.to_generation_settings(),
                generation.created_at,
                generation.kind,
                generation.parent_generation_id,
                is_recovery=True,
            )
            return True
        except (ComfyUIError, GenerationRepositoryError, StorageError, ValueError) as exc:
            logger.error(
                "Generation recovery failed generation=%s prompt_id=%s error=%s",
                generation_id,
                prompt_id,
                type(exc).__name__,
                exc_info=True,
            )
            return False

    async def generate(
        self,
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None = None,
        *,
        parent_generation_id: UUID | None = None,
        kind: GenerationKind = GenerationKind.STANDARD,
    ) -> GenerationResult:
        """Run one generation and retain its result in memory."""

        async with self._lock:
            return await self._generate_locked(
                settings, progress_callback, parent_generation_id, kind
            )

    async def _generate_locked(
        self,
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None,
        parent_generation_id: UUID | None,
        kind: GenerationKind,
    ) -> GenerationResult:
        generation_id = self._id_factory()
        created_at = datetime.now(UTC)
        seed = _resolve_seed(settings.seed)
        resolved_settings = settings.model_copy(update={"seed": seed})
        job = GenerationJob(generation_id=generation_id, status=GenerationStatus.PENDING)
        self._jobs[generation_id] = job
        if not self._persist_pending(
            resolved_settings, generation_id, kind, parent_generation_id, created_at, job
        ):
            job.status = GenerationStatus.FAILED
            job.error_message = "生成履歴を保存できませんでした。"
            failed_at = datetime.now(UTC)
            try:
                self._persist_failure(
                    job,
                    error_code=GenerationErrorCode.DATABASE.value,
                    error_summary=job.error_message,
                    failed_at=failed_at,
                )
            except FailurePersistenceError as exc:
                logger.error(
                    "Failure persistence also failed generation=%s job=%s "
                    "original_error=pending_record_failure",
                    job.generation_id,
                    job.id,
                    exc_info=True,
                )
                job.error_message = persistence_error_message(exc)
            result = GenerationResult(
                generation_id=generation_id,
                prompt_id="",
                status=GenerationStatus.FAILED,
                seed=seed,
                stored_image=None,
                error_message=job.error_message,
                created_at=created_at,
            )
            self._results[generation_id] = result
            return result
        self._emit(
            progress_callback,
            GenerationProgress(
                prompt_id="",
                state=GenerationStatus.PENDING,
                current_node=None,
                value=None,
                maximum=None,
                percentage=0.0,
                message="設定を確認中です",
            ),
        )

        try:
            await asyncio.wait_for(
                self._run_job(
                    job,
                    resolved_settings,
                    created_at,
                    progress_callback,
                    kind,
                    parent_generation_id,
                ),
                timeout=self._settings.generation_timeout_seconds,
            )
            result = self._result_for_job(job, seed, created_at)
            if (
                result.status is GenerationStatus.COMPLETED
                and self._lora_catalog_service is not None
                and resolved_settings.loras
            ):
                completed_at = datetime.now(UTC)
                try:
                    self._lora_catalog_service.record_usage(
                        tuple(lora.name for lora in resolved_settings.loras),
                        completed_at,
                    )
                except Exception:  # noqa: BLE001 - usage statistics are best effort
                    logger.warning("LoRA usage statistics update failed", exc_info=True)
        except Exception as exc:  # noqa: BLE001 - boundary converts failures to safe UI text
            logger.error(
                "Generation job failed generation=%s job=%s prompt_id=%s error=%s",
                generation_id,
                job.id,
                job.prompt_id or "",
                type(exc).__name__,
                exc_info=True,
            )
            job.status = GenerationStatus.FAILED
            error_code = _generation_error_code(exc)
            error_summary = _safe_generation_error(exc)
            job.error_code = error_code
            job.error_summary = error_summary
            job.error_message = error_summary
            try:
                self._persist_failure(
                    job,
                    error_code=error_code,
                    error_summary=error_summary,
                    failed_at=datetime.now(UTC),
                )
            except FailurePersistenceError as failure_error:
                logger.error(
                    "Failure persistence failed generation=%s job=%s prompt_id=%s "
                    "original_error=%s error_code=%s",
                    generation_id,
                    job.id,
                    job.prompt_id or "",
                    type(exc).__name__,
                    error_code,
                    exc_info=True,
                )
                job.error_message = _safe_generation_error(failure_error, original_error=exc)
            result = GenerationResult(
                generation_id=generation_id,
                prompt_id=job.prompt_id or "",
                status=GenerationStatus.FAILED,
                seed=seed,
                stored_image=None,
                error_message=job.error_message,
                created_at=created_at,
            )
        self._results[generation_id] = result
        return result

    def _persist_pending(
        self,
        settings: GenerationSettings,
        generation_id: UUID,
        kind: GenerationKind,
        parent_generation_id: UUID | None,
        created_at: datetime,
        job: GenerationJob,
    ) -> bool:
        if self._generation_repository is None:
            return True
        try:
            snapshot = GenerationSettingsSnapshot.from_settings(settings)
            self._generation_repository.create_pending(
                snapshot,
                kind=kind,
                parent_generation_id=parent_generation_id,
                generation_id=generation_id,
                created_at=created_at,
            )
            if self._job_repository is not None:
                persisted_job = self._job_repository.create(
                    job.__class__(
                        generation_id=job.generation_id,
                        status=GenerationStatus.PENDING,
                        id=job.id,
                        created_at=created_at,
                    )
                )
                job.id = persisted_job.id
            return True
        except (GenerationRepositoryError, ValueError) as exc:
            logger.error("Generation persistence failed before prompt: %s", type(exc).__name__)
            return False

    def _persist_failure(
        self,
        job: GenerationJob,
        *,
        error_code: str,
        error_summary: str,
        failed_at: datetime,
    ) -> None:
        if self._failure_repository is None:
            return
        try:
            self._failure_repository.fail_generation(
                generation_id=job.generation_id,
                job_id=job.id,
                error_code=error_code,
                error_summary=error_summary,
                failed_at=failed_at,
            )
        except GenerationRepositoryError as exc:
            logger.error(
                "Failed to persist generation failure generation=%s job=%s prompt_id=%s "
                "error_code=%s",
                job.generation_id,
                job.id,
                job.prompt_id or "",
                error_code,
                exc_info=True,
            )
            raise FailurePersistenceError(
                "generation failure state could not be persisted"
            ) from exc

    async def _run_job(
        self,
        job: GenerationJob,
        settings: GenerationSettings,
        created_at: datetime,
        progress_callback: ProgressCallback | None,
        kind: GenerationKind,
        parent_generation_id: UUID | None,
    ) -> None:
        capabilities_result = await self._capabilities_provider()
        capabilities = capabilities_result.capabilities
        if not capabilities_result.is_success or capabilities is None:
            raise ComfyUIPromptError("ComfyUI capabilities are unavailable")
        _validate_generation(settings, capabilities, self._settings)

        workflow = self._workflow_adapter.build_txt2img_workflow(settings)
        client_id = str(self._id_factory())
        queued = await self._client.queue_prompt(workflow, client_id)
        if queued.node_errors:
            raise ComfyUIPromptError("ComfyUI rejected workflow nodes")
        job.prompt_id = queued.prompt_id
        job.status = GenerationStatus.QUEUED
        try:
            if self._queue_repository is not None:
                self._queue_repository.mark_queued(
                    generation_id=job.generation_id,
                    job_id=job.id,
                    prompt_id=queued.prompt_id,
                )
        except GenerationRepositoryError as exc:
            logger.error(
                "Prompt ID persistence failed generation=%s job=%s prompt_id=%s",
                job.generation_id,
                job.id,
                queued.prompt_id,
            )
            raise PromptPersistenceError("prompt ID could not be persisted") from exc
        self._emit(
            progress_callback,
            GenerationProgress(
                prompt_id=queued.prompt_id,
                state=GenerationStatus.QUEUED,
                current_node=None,
                value=None,
                maximum=None,
                percentage=0.0,
                message="ComfyUIキューへ投入しました",
            ),
        )

        websocket_failed = False
        try:
            async for progress in self._websocket_client.watch_prompt(queued.prompt_id, client_id):
                job.status = progress.state
                self._persist_progress(job, progress)
                self._emit(progress_callback, progress)
                if progress.state is GenerationStatus.FAILED:
                    websocket_failed = True
                    break
        except ComfyUIWebSocketError:
            websocket_failed = True
            logger.info("Falling back to ComfyUI history after WebSocket loss")

        history = await self._poll_history(queued.prompt_id)
        if history.is_failed or not history.is_completed:
            raise ComfyUIPromptError("ComfyUI did not complete the prompt")
        if websocket_failed:
            self._emit(
                progress_callback,
                GenerationProgress(
                    prompt_id=queued.prompt_id,
                    state=GenerationStatus.RUNNING,
                    current_node=None,
                    value=None,
                    maximum=None,
                    percentage=None,
                    message="履歴から生成結果を復旧しました",
                ),
            )
        if not history.outputs:
            raise ComfyUIPromptError("ComfyUI completed without an output image")
        image_bytes = await self._client.get_output_image(history.outputs[0])
        job.stored_image = self._storage.store_image(
            image_bytes,
            job.generation_id,
            created_at,
        )
        self._complete_job(job, settings, created_at, kind, parent_generation_id)

    def _persist_progress(self, job: GenerationJob, progress: GenerationProgress) -> None:
        try:
            if (
                progress.state is GenerationStatus.RUNNING
                and self._generation_repository is not None
            ):
                self._generation_repository.mark_running(job.generation_id)
            if self._job_repository is not None:
                self._job_repository.update_progress(
                    job.id, progress.value, progress.maximum, progress.current_node
                )
        except GenerationRepositoryError:
            logger.warning("Failed to persist progress generation=%s", job.generation_id)

    def _complete_job(
        self,
        job: GenerationJob,
        settings: GenerationSettings,
        created_at: datetime,
        kind: GenerationKind,
        parent_generation_id: UUID | None,
        *,
        is_recovery: bool = False,
    ) -> None:
        if job.stored_image is None:
            raise GenerationPersistenceError("stored image is missing")
        image = job.stored_image
        completed_at = datetime.now(UTC)
        primary_artifact = self._primary_image_artifact(job, image, completed_at)
        if self._generation_repository is not None:
            if self._completion_repository is not None:
                try:
                    self._completion_repository.complete_generation(
                        job.generation_id,
                        job.id,
                        primary_artifact,
                        completed_at,
                    )
                except GenerationRepositoryError as exc:
                    logger.error(
                        "Generation completion persistence failed generation=%s job=%s "
                        "prompt_id=%s image_path=%s error=%s",
                        job.generation_id,
                        job.id,
                        job.prompt_id or "",
                        primary_artifact.local_path,
                        type(exc).__name__,
                    )
                    error_type = (
                        RecoveryPersistenceError if is_recovery else CompletionPersistenceError
                    )
                    raise error_type("generation completion could not be persisted") from exc
            else:
                if self._job_repository is None:
                    raise CompletionPersistenceError("job persistence is unavailable")
                self._persist_primary_image_artifact(job, primary_artifact)
                try:
                    self._generation_repository.mark_completed(job.generation_id, completed_at)
                    self._job_repository.mark_completed(job.id, completed_at)
                except GenerationRepositoryError as exc:
                    logger.error(
                        "Generation completion persistence failed generation=%s job=%s "
                        "prompt_id=%s image_path=%s",
                        job.generation_id,
                        job.id,
                        job.prompt_id or "",
                        primary_artifact.local_path,
                    )
                    error_type = (
                        RecoveryPersistenceError if is_recovery else CompletionPersistenceError
                    )
                    raise error_type("generation completion could not be persisted") from exc
        job.status = GenerationStatus.COMPLETED
        job.completed_at = completed_at
        self._persist_optional_artifacts(
            job,
            settings,
            image,
            completed_at,
            kind,
            parent_generation_id,
        )

    def _complete_existing_generation(self, generation_id: UUID, job_id: UUID) -> None:
        completed_at = datetime.now(UTC)
        if self._completion_repository is not None:
            try:
                self._completion_repository.complete_existing_artifact(
                    generation_id,
                    job_id,
                    completed_at,
                )
            except GenerationRepositoryError as exc:
                logger.error(
                    "Existing artifact recovery persistence failed generation=%s job=%s",
                    generation_id,
                    job_id,
                    exc_info=True,
                )
                raise RecoveryPersistenceError(
                    "existing generation completion could not be persisted"
                ) from exc
            return
        if self._generation_repository is None or self._job_repository is None:
            raise RecoveryPersistenceError("generation persistence is unavailable")
        try:
            self._generation_repository.mark_completed(generation_id, completed_at)
            self._job_repository.mark_completed(job_id, completed_at)
        except GenerationRepositoryError as exc:
            logger.error(
                "Existing artifact completion persistence failed generation=%s job=%s",
                generation_id,
                job_id,
            )
            raise RecoveryPersistenceError(
                "existing generation completion could not be persisted"
            ) from exc

    def _persist_primary_image_artifact(
        self, job: GenerationJob, artifact: GenerationArtifact
    ) -> None:
        if self._artifact_repository is None:
            raise ArtifactPersistenceError("primary image artifact repository is unavailable")
        try:
            self._artifact_repository.add(artifact)
        except (GenerationRepositoryError, StorageError, OSError) as exc:
            logger.error(
                "Required image artifact persistence failed generation=%s "
                "prompt_id=%s image_path=%s error=%s",
                artifact.generation_id,
                job.prompt_id or "",
                artifact.local_path,
                type(exc).__name__,
            )
            raise ArtifactPersistenceError("primary image artifact could not be persisted") from exc

    def _persist_optional_artifacts(
        self,
        job: GenerationJob,
        settings: GenerationSettings,
        image: StoredImage,
        created_at: datetime,
        kind: GenerationKind,
        parent_generation_id: UUID | None,
    ) -> None:
        if self._artifact_repository is None:
            return
        try:
            existing_types = {
                artifact.artifact_type
                for artifact in self._artifact_repository.list_by_generation(job.generation_id)
            }
        except GenerationRepositoryError as exc:
            logger.warning(
                "Optional artifact lookup warning generation=%s error=%s",
                job.generation_id,
                type(exc).__name__,
            )
            existing_types = set()
        image_path = image.path
        if self._metadata_storage is not None:
            if ArtifactType.METADATA in existing_types:
                pass
            else:
                try:
                    payload = _sidecar_payload(job, settings, image, kind, parent_generation_id)
                    sidecar_path = self._metadata_storage.save_for_image(image_path, payload)
                    self._artifact_repository.add(
                        GenerationArtifact(
                            id=self._id_factory(),
                            generation_id=job.generation_id,
                            artifact_type=ArtifactType.METADATA,
                            local_path=self._metadata_storage.relative_path(sidecar_path),
                            sha256=self._metadata_storage.sha256(sidecar_path),
                            size_bytes=sidecar_path.stat().st_size,
                            width=None,
                            height=None,
                            mime_type="application/json",
                            created_at=created_at,
                        )
                    )
                except (GenerationRepositoryError, StorageError, OSError) as exc:
                    logger.warning(
                        "Generation sidecar warning generation=%s error=%s",
                        job.generation_id,
                        type(exc).__name__,
                    )
        if self._thumbnail_storage is not None and ArtifactType.THUMBNAIL not in existing_types:
            try:
                thumbnail_path = self._thumbnail_storage.save(
                    image_path, job.generation_id, created_at
                )
                self._artifact_repository.add(
                    GenerationArtifact(
                        id=self._id_factory(),
                        generation_id=job.generation_id,
                        artifact_type=ArtifactType.THUMBNAIL,
                        local_path=self._thumbnail_storage.relative_path(thumbnail_path),
                        sha256=self._thumbnail_storage.sha256(thumbnail_path),
                        size_bytes=thumbnail_path.stat().st_size,
                        width=None,
                        height=None,
                        mime_type="image/webp",
                        created_at=created_at,
                    )
                )
            except (GenerationRepositoryError, StorageError, OSError) as exc:
                logger.warning(
                    "Generation thumbnail warning generation=%s error=%s",
                    job.generation_id,
                    type(exc).__name__,
                )

    def _primary_image_artifact(
        self,
        job: GenerationJob,
        image: StoredImage,
        created_at: datetime,
    ) -> GenerationArtifact:
        return GenerationArtifact(
            id=self._id_factory(),
            generation_id=job.generation_id,
            artifact_type=ArtifactType.IMAGE,
            local_path=self._storage.relative_path(image.path),
            sha256=image.sha256,
            size_bytes=image.size_bytes,
            width=image.width,
            height=image.height,
            mime_type=image.mime_type,
            created_at=created_at,
        )

    def _mark_failed_pair(self, job: GenerationJob, code: str, summary: str) -> None:
        job.status = GenerationStatus.FAILED
        job.error_code = code
        job.error_summary = summary
        self._persist_failure(
            job,
            error_code=code,
            error_summary=summary,
            failed_at=datetime.now(UTC),
        )

    async def _poll_history(self, prompt_id: str) -> PromptHistory:
        last_error: ComfyUIError | None = None
        for attempt in range(self._settings.history_max_attempts):
            try:
                history = await self._client.get_prompt_history(prompt_id)
            except ComfyUIError as exc:
                last_error = exc
            else:
                if history.is_completed or history.is_failed:
                    return history
            if attempt + 1 < self._settings.history_max_attempts:
                await self._sleep(self._settings.history_poll_interval_seconds)
        if last_error is not None:
            raise last_error
        raise HistoryTimeoutError("ComfyUI history did not become available")

    @staticmethod
    def _result_for_job(
        job: GenerationJob,
        seed: int,
        created_at: datetime,
    ) -> GenerationResult:
        if job.status is not GenerationStatus.COMPLETED or job.stored_image is None:
            raise ComfyUIPromptError("Generation did not produce a stored image")
        return GenerationResult(
            generation_id=job.generation_id,
            prompt_id=job.prompt_id or "",
            status=GenerationStatus.COMPLETED,
            seed=seed,
            stored_image=job.stored_image,
            error_message=None,
            created_at=created_at,
        )

    @staticmethod
    def _emit(callback: ProgressCallback | None, progress: GenerationProgress) -> None:
        if callback is not None:
            callback(progress)


def _resolve_seed(seed: int) -> int:
    return secrets.randbelow(2**64) if seed == -1 else seed


def _validate_generation(
    settings: GenerationSettings, capabilities: object, limits: Settings
) -> None:
    checkpoints = getattr(capabilities, "checkpoints", ())
    samplers = getattr(capabilities, "samplers", ())
    schedulers = getattr(capabilities, "schedulers", ())
    vaes = getattr(capabilities, "vaes", ())
    loras = getattr(capabilities, "loras", ())
    node_classes: frozenset[str] = getattr(capabilities, "available_node_classes", frozenset())
    if settings.checkpoint_name not in checkpoints:
        raise WorkflowError("Selected checkpoint is unavailable")
    if settings.sampler_name not in samplers:
        raise WorkflowError("Selected sampler is unavailable")
    if settings.scheduler_name not in schedulers:
        raise WorkflowError("Selected scheduler is unavailable")
    if settings.vae_name is not None and settings.vae_name not in vaes:
        logger.warning("Unavailable VAE selected: %s", settings.vae_name)
        raise WorkflowError("Selected VAE is unavailable")
    if len(settings.loras) > limits.max_loras:
        raise WorkflowError("The configured maximum number of LoRAs was exceeded")
    if settings.loras and "LoraLoader" not in node_classes:
        raise WorkflowError("LoRA loading is unavailable in ComfyUI")
    if settings.vae_name is not None and "VAELoader" not in node_classes:
        raise WorkflowError("External VAE loading is unavailable in ComfyUI")
    missing_loras = [lora.name for lora in settings.loras if lora.name not in loras]
    if missing_loras:
        logger.warning("Unavailable LoRAs selected: %s", ", ".join(missing_loras))
        raise WorkflowError("One or more selected LoRAs are unavailable")
    if settings.width > limits.max_width or settings.height > limits.max_height:
        raise WorkflowError("Requested image dimensions exceed the configured limit")
    if settings.width * settings.height > limits.max_pixels:
        raise WorkflowError("Requested image area exceeds the configured limit")


def _safe_generation_error(
    error: Exception,
    *,
    original_error: Exception | None = None,
) -> str:
    if isinstance(error, GenerationPersistenceError):
        if isinstance(error, FailurePersistenceError) and original_error is not None:
            return "生成に失敗しました。加えて、履歴の失敗状態を完全に保存できませんでした。"
        return persistence_error_message(error)
    if isinstance(error, GenerationRepositoryError):
        return "生成結果の保存状態を確定できませんでした。"
    if isinstance(error, WorkflowError):
        return "生成設定を確認できませんでした。"
    if isinstance(error, StorageError):
        return "生成画像を保存できませんでした。"
    if isinstance(error, ComfyUIError):
        return "ComfyUIで画像生成を完了できませんでした"
    if isinstance(error, TimeoutError):
        return "画像生成が制限時間を超えました。"
    return "画像生成に失敗しました。"


def _generation_error_code(error: Exception) -> str:
    if isinstance(error, GenerationPersistenceError):
        return persistence_error_code(error)
    if isinstance(error, WorkflowError):
        return GenerationErrorCode.WORKFLOW.value
    if isinstance(error, StorageError):
        return GenerationErrorCode.STORAGE.value
    if isinstance(error, ComfyUIWebSocketError):
        return GenerationErrorCode.COMFYUI_EXECUTION.value
    if isinstance(error, HistoryTimeoutError):
        return GenerationErrorCode.HISTORY_TIMEOUT.value
    if isinstance(error, ComfyUIPromptError):
        return GenerationErrorCode.COMFYUI_PROMPT.value
    if isinstance(error, ComfyUIError):
        return GenerationErrorCode.COMFYUI_CONNECTION.value
    if isinstance(error, TimeoutError):
        return GenerationErrorCode.HISTORY_TIMEOUT.value
    return (
        GenerationErrorCode.DATABASE.value
        if isinstance(error, GenerationRepositoryError)
        else "generation_error"
    )


def _sidecar_payload(
    job: GenerationJob,
    settings: GenerationSettings,
    image: object,
    kind: GenerationKind,
    parent_generation_id: UUID | None,
) -> dict[str, object]:
    stored = image
    return {
        "schema_version": 1,
        "generation_id": str(job.generation_id),
        "kind": kind.value,
        "parent_generation_id": str(parent_generation_id) if parent_generation_id else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": datetime.now(UTC).isoformat(),
        "settings": GenerationSettingsSnapshot.from_settings(settings).model_dump(mode="json"),
        "workflow_template_id": settings.workflow_template_id,
        "workflow_template_version": settings.workflow_template_version,
        "comfy_prompt_id": job.prompt_id,
        "image": {
            "sha256": stored.sha256,  # type: ignore[attr-defined]
            "width": stored.width,  # type: ignore[attr-defined]
            "height": stored.height,  # type: ignore[attr-defined]
            "mime_type": stored.mime_type,  # type: ignore[attr-defined]
        },
    }
