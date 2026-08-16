"""Application service for one fixed SDXL generation job."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIError,
    ComfyUIPromptError,
    ComfyUIWebSocketError,
    WorkflowError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    PromptHistory,
    PromptHistoryStatus,
    RemotePromptStatus,
)
from runpod_sdxl_image_studio.adapters.comfyui.upscale_workflow_adapter import (
    UpscaleWorkflowAdapter,
)
from runpod_sdxl_image_studio.adapters.comfyui.websocket_client import ComfyUIWebSocketClient
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationCompletionRepositoryProtocol,
    GenerationFailureRepositoryProtocol,
    GenerationJobRepositoryProtocol,
    GenerationQueueRepositoryProtocol,
    GenerationRepositoryError,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.metadata_import_repository import (
    MetadataImportRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepositoryError,
    UpscaleSettingsRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.generation_metadata_storage import (
    GenerationMetadataStorage,
)
from runpod_sdxl_image_studio.adapters.storage.history_thumbnail_storage import (
    HistoryThumbnailStorage,
)
from runpod_sdxl_image_studio.adapters.storage.imported_image_storage import (
    ImportedImageStorage,
    ImportedImageStorageError,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.detailer import DEFAULT_DETAILER_REGISTRY, DetailerKind
from runpod_sdxl_image_studio.domain.generation import (
    GenerationErrorCode,
    GenerationKind,
    GenerationProgress,
    GenerationResult,
    GenerationStatus,
    StoredImage,
)
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_queue import (
    OptionalArtifactRepairOutcome,
    ReconciliationOutcome,
)
from runpod_sdxl_image_studio.domain.generation_settings import (
    LEGACY_WORKFLOW_TEMPLATE_VERSION,
    MAX_SEED,
    PIXEL_HIRES_WORKFLOW_VERSIONS,
    RANDOM_SEED,
    GenerationSettings,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.job import GenerationJob
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.domain.upscale_snapshot import (
    UpscaleSettingsSnapshot,
    UpscaleSourceKind,
)
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
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
)
from runpod_sdxl_image_studio.services.upscale_enqueue_service import (
    UpscaleEnqueueError,
    verify_source_artifact,
)
from runpod_sdxl_image_studio.services.upscale_output_validation import (
    validate_upscale_output,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[GenerationProgress], None]
CancelCheck = Callable[[], bool]

_RECONCILIATION_UPSCALE_FAILURE_CODES = frozenset(
    {
        "upscale_output_dimension_mismatch",
        "upscale_output_invalid",
        "upscale_settings_missing",
    }
)


class HistoryTimeoutError(ComfyUIPromptError):
    """履歴のポーリングが設定された試行回数に達した。"""


class GenerationCancelledError(RuntimeError):
    """The queue requested cancellation before a prompt was submitted."""


class CapabilitiesProvider(Protocol):
    def __call__(self) -> Awaitable[CapabilityRefreshResult]: ...


class LoraUsageRecorder(Protocol):
    def record_usage(self, file_names: tuple[str, ...], completed_at: datetime) -> None: ...


class PromptSubmissionCoordinator(Protocol):
    """Coordinates durable state around the non-idempotent prompt request."""

    started: bool
    submitted: bool

    def begin(self) -> str: ...

    def mark_submitted(self, prompt_id: str, submission_token: str) -> None: ...


class GenerationService:
    """検証、生成監視、復旧、ローカル保存を調整するApplication Service。"""

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
        start_repository: GenerationStartRepositoryProtocol | None = None,
        progress_repository: GenerationProgressRepositoryProtocol | None = None,
        persistence: GenerationPersistenceRepositories | None = None,
        thumbnail_storage: HistoryThumbnailStorage | None = None,
        metadata_storage: GenerationMetadataStorage | None = None,
        upscale_settings_repository: UpscaleSettingsRepositoryProtocol | None = None,
        upscale_workflow_adapter: UpscaleWorkflowAdapter | None = None,
        upscaler_catalog: UpscalerCatalog | None = None,
        upscaler_catalog_provider: Callable[[], UpscalerCatalog] | None = None,
        metadata_import_repository: MetadataImportRepositoryProtocol | None = None,
        imported_image_storage: ImportedImageStorage | None = None,
        drive_sync_enqueue_handler: Callable[[UUID], object] | None = None,
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
        individual_repositories = (
            generation_repository,
            job_repository,
            artifact_repository,
            start_repository,
            queue_repository,
            progress_repository,
            completion_repository,
            failure_repository,
        )
        if persistence is not None and any(
            repository is not None for repository in individual_repositories
        ):
            raise ValueError("persistence cannot be combined with individual repositories")
        if persistence is None and any(
            repository is not None for repository in individual_repositories
        ):
            if not all(repository is not None for repository in individual_repositories):
                raise ValueError("generation persistence repositories must be configured together")
            persistence = GenerationPersistenceRepositories(
                generation=cast(GenerationRepositoryProtocol, generation_repository),
                job=cast(GenerationJobRepositoryProtocol, job_repository),
                artifact=cast(GenerationArtifactRepositoryProtocol, artifact_repository),
                start=cast(GenerationStartRepositoryProtocol, start_repository),
                queue=cast(GenerationQueueRepositoryProtocol, queue_repository),
                progress=cast(GenerationProgressRepositoryProtocol, progress_repository),
                completion=cast(GenerationCompletionRepositoryProtocol, completion_repository),
                failure=cast(GenerationFailureRepositoryProtocol, failure_repository),
            )
        self._generation_repository = persistence.generation if persistence else None
        self._job_repository = persistence.job if persistence else None
        self._artifact_repository = persistence.artifact if persistence else None
        self._queue_repository = persistence.queue if persistence else None
        self._progress_repository = persistence.progress if persistence else None
        self._completion_repository = persistence.completion if persistence else None
        self._failure_repository = persistence.failure if persistence else None
        all_repositories = individual_repositories
        if any(repository is not None for repository in all_repositories) and any(
            repository is None for repository in all_repositories
        ):
            raise ValueError("generation persistence repositories must be configured together")
        self._start_repository = persistence.start if persistence else None
        self._thumbnail_storage = thumbnail_storage
        self._metadata_storage = metadata_storage
        self._upscale_settings_repository = upscale_settings_repository
        self._upscale_workflow_adapter = upscale_workflow_adapter
        self._upscaler_catalog = upscaler_catalog or UpscalerCatalog.scan(
            self._settings.upscaler_dir
        )
        self._upscaler_catalog_provider = upscaler_catalog_provider
        self._metadata_import_repository = metadata_import_repository
        self._imported_image_storage = imported_image_storage
        self._drive_sync_enqueue_handler = drive_sync_enqueue_handler
        self._jobs: dict[UUID, GenerationJob] = {}
        self._results: dict[UUID, GenerationResult] = {}
        self._lock = asyncio.Lock()

    @property
    def jobs(self) -> Mapping[UUID, GenerationJob]:
        return self._jobs

    def get_result(self, generation_id: UUID) -> GenerationResult | None:
        """Return a result while the process is alive."""

        return self._results.get(generation_id)

    def set_drive_sync_enqueue_handler(self, handler: Callable[[UUID], object] | None) -> None:
        """Attach the independent Drive queue after the app has built its services."""

        self._drive_sync_enqueue_handler = handler

    def repair_optional_artifacts(self, generation_id: UUID) -> OptionalArtifactRepairOutcome:
        """Repair missing metadata or thumbnails for one already completed generation."""

        if (
            self._generation_repository is None
            or self._job_repository is None
            or self._artifact_repository is None
        ):
            return OptionalArtifactRepairOutcome.UNAVAILABLE
        try:
            generation = self._generation_repository.get_by_id(generation_id)
            job = self._job_repository.get_by_generation(generation_id)
            if generation is None or job is None:
                return OptionalArtifactRepairOutcome.UNAVAILABLE
            artifacts = self._artifact_repository.list_by_generation(generation_id)
        except Exception as exc:  # noqa: BLE001 - typed outcome must contain read failures
            logger.warning(
                "Optional artifact repair lookup failed generation=%s error=%s",
                generation_id,
                type(exc).__name__,
                exc_info=True,
            )
            return OptionalArtifactRepairOutcome.UNAVAILABLE

        existing_types = {artifact.artifact_type for artifact in artifacts}
        missing_types = {ArtifactType.METADATA, ArtifactType.THUMBNAIL} - existing_types
        if not missing_types:
            return OptionalArtifactRepairOutcome.ALREADY_COMPLETE
        if (
            generation.status is not GenerationStatus.COMPLETED
            or job.status is not GenerationStatus.COMPLETED
            or generation.completed_at is None
            or job.completed_at is None
            or _utc(generation.completed_at) != _utc(job.completed_at)
        ):
            return OptionalArtifactRepairOutcome.DEFERRED

        primary_artifacts = tuple(
            sorted(
                (
                    artifact
                    for artifact in artifacts
                    if artifact.artifact_type is ArtifactType.IMAGE
                ),
                key=lambda artifact: artifact.display_order,
            )
        )
        if not primary_artifacts:
            return OptionalArtifactRepairOutcome.DEFERRED
        try:
            stored_images = tuple(
                self._stored_image_from_artifact(artifact) for artifact in primary_artifacts
            )
            settings = generation.settings_snapshot.to_generation_settings()
            upscale_snapshot = None
            if generation.kind is GenerationKind.UPSCALE:
                if self._upscale_settings_repository is None:
                    return OptionalArtifactRepairOutcome.DEFERRED
                upscale_snapshot = self._upscale_settings_repository.get_by_generation(
                    generation_id
                )
                if upscale_snapshot is None:
                    return OptionalArtifactRepairOutcome.DEFERRED
            self._persist_optional_artifacts(
                job,
                settings,
                stored_images[0],
                generation.completed_at,
                generation.kind,
                generation.parent_generation_id,
                existing_artifacts=artifacts,
                persisted_upscale_snapshot=upscale_snapshot,
                load_upscale_settings=False,
                images=stored_images,
            )
            repaired_artifacts = self._artifact_repository.list_by_generation(generation_id)
        except Exception as exc:  # noqa: BLE001 - one repair must not alter terminal state
            logger.warning(
                "Optional artifact repair deferred generation=%s error=%s",
                generation_id,
                type(exc).__name__,
                exc_info=True,
            )
            return OptionalArtifactRepairOutcome.DEFERRED

        repaired_types = {artifact.artifact_type for artifact in repaired_artifacts}
        return (
            OptionalArtifactRepairOutcome.REPAIRED
            if missing_types <= repaired_types
            else OptionalArtifactRepairOutcome.DEFERRED
        )

    def _stored_image_from_artifact(self, artifact: GenerationArtifact) -> StoredImage:
        if (
            artifact.width is None
            or artifact.height is None
            or artifact.size_bytes <= 0
            or not artifact.mime_type
        ):
            raise UpscaleEnqueueError(
                "optional_artifact_primary_invalid",
                "primary image artifact metadata is incomplete",
            )
        verified = verify_source_artifact(artifact, self._settings)
        return StoredImage(
            path=verified.path,
            sha256=verified.sha256,
            size_bytes=artifact.size_bytes,
            width=verified.width,
            height=verified.height,
            mime_type=artifact.mime_type,
        )

    async def reconcile_prompt(self, generation_id: UUID, prompt_id: str) -> ReconciliationOutcome:
        """Reconcile one existing prompt without ever submitting it again."""

        if (
            self._generation_repository is None
            or self._job_repository is None
            or self._artifact_repository is None
        ):
            return ReconciliationOutcome.UNAVAILABLE
        try:
            generation = self._generation_repository.get_by_id(generation_id)
            job = self._job_repository.get_by_generation(generation_id)
            if generation is None or job is None:
                return ReconciliationOutcome.UNAVAILABLE
            if generation.status is GenerationStatus.COMPLETED:
                return ReconciliationOutcome.COMPLETED
            if generation.status in {GenerationStatus.FAILED, GenerationStatus.CANCELLED}:
                return ReconciliationOutcome.FAILED
            existing_image = self._artifact_repository.get_primary_image(generation_id)
            if existing_image is not None:
                self._complete_existing_generation(generation_id, job.id)
                return ReconciliationOutcome.COMPLETED
            status_reader = getattr(self._client, "get_remote_prompt_status", None)
            if callable(status_reader):
                remote_state = await status_reader(prompt_id)
                if remote_state.status is RemotePromptStatus.NOT_FOUND:
                    return ReconciliationOutcome.NOT_FOUND
                if remote_state.status in {
                    RemotePromptStatus.PENDING,
                    RemotePromptStatus.IN_PROGRESS,
                }:
                    return ReconciliationOutcome.IN_PROGRESS
                if remote_state.status is RemotePromptStatus.CANCELLED:
                    return ReconciliationOutcome.CANCELLED
                if remote_state.status is RemotePromptStatus.FAILED:
                    self._mark_failed_pair(
                        job,
                        GenerationErrorCode.COMFYUI_EXECUTION.value,
                        "ComfyUI reported a failed remote prompt",
                    )
                    return ReconciliationOutcome.FAILED
                if remote_state.status is RemotePromptStatus.UNAVAILABLE:
                    return ReconciliationOutcome.UNAVAILABLE
            history = await self._client.get_prompt_history(prompt_id)
            if history.status is PromptHistoryStatus.NOT_FOUND:
                return ReconciliationOutcome.NOT_FOUND
            if history.status is PromptHistoryStatus.INTERRUPTED:
                return ReconciliationOutcome.CANCELLED
            if history.status is PromptHistoryStatus.FAILED:
                self._mark_failed_pair(
                    job,
                    GenerationErrorCode.COMFYUI_EXECUTION.value,
                    "ComfyUIで生成が失敗しました。",
                )
                return ReconciliationOutcome.FAILED
            if history.status is PromptHistoryStatus.UNKNOWN:
                return ReconciliationOutcome.UNAVAILABLE
            if not history.is_completed or not history.outputs:
                return ReconciliationOutcome.IN_PROGRESS
            persisted_upscale = None
            stored_images: list[StoredImage] = []
            for output in history.outputs:
                image_bytes = await self._client.get_output_image(output)
                if generation.kind is GenerationKind.UPSCALE:
                    try:
                        persisted_upscale = self._validate_upscale_output(
                            generation_id, image_bytes
                        )
                    except UpscaleEnqueueError as exc:
                        if exc.code in _RECONCILIATION_UPSCALE_FAILURE_CODES:
                            return self._fail_reconciled_upscale(job, exc)
                        raise
                stored_image = self._storage.store_image(
                    image_bytes,
                    generation_id,
                    generation.created_at,
                    kind=generation.kind,
                    client_local_date=generation.settings_snapshot.client_local_date,
                )
                if persisted_upscale is not None:
                    try:
                        self._ensure_stored_upscale_dimensions(stored_image, persisted_upscale)
                    except UpscaleEnqueueError as exc:
                        if exc.code in _RECONCILIATION_UPSCALE_FAILURE_CODES:
                            return self._fail_reconciled_upscale(job, exc)
                        raise
                stored_images.append(stored_image)
            recovery_job = GenerationJob(
                generation_id=generation_id,
                status=GenerationStatus.RUNNING,
                id=job.id,
                prompt_id=prompt_id,
                created_at=generation.created_at,
                stored_image=stored_images[0],
                stored_images=tuple(stored_images),
            )
            self._complete_job(
                recovery_job,
                generation.settings_snapshot.to_generation_settings(),
                generation.created_at,
                generation.kind,
                generation.parent_generation_id,
                is_recovery=True,
            )
            return ReconciliationOutcome.COMPLETED
        except (
            ComfyUIError,
            GenerationRepositoryError,
            UpscaleSettingsRepositoryError,
            UpscaleEnqueueError,
            StorageError,
            ValueError,
        ) as exc:
            logger.error(
                "Generation recovery failed generation=%s prompt_id=%s error=%s",
                generation_id,
                prompt_id,
                type(exc).__name__,
                exc_info=True,
            )
            return ReconciliationOutcome.UNAVAILABLE

    def _validate_upscale_output(
        self, generation_id: UUID, image_bytes: bytes
    ) -> UpscaleSettingsSnapshot:
        if self._upscale_settings_repository is None:
            raise UpscaleEnqueueError(
                "upscale_settings_unavailable", "upscale settings repository is unavailable"
            )
        return validate_upscale_output(
            generation_id, image_bytes, self._upscale_settings_repository
        )

    def _fail_reconciled_upscale(
        self, job: GenerationJob, error: UpscaleEnqueueError
    ) -> ReconciliationOutcome:
        self._mark_failed_pair(job, error.code, _safe_upscale_error(error))
        return ReconciliationOutcome.FAILED

    @staticmethod
    def _ensure_stored_upscale_dimensions(
        stored_image: StoredImage, snapshot: UpscaleSettingsSnapshot
    ) -> None:
        if (stored_image.width, stored_image.height) != (
            snapshot.target_width,
            snapshot.target_height,
        ):
            raise UpscaleEnqueueError(
                "upscale_output_dimension_mismatch",
                "stored upscale output dimensions do not match the persisted target",
            )

    async def recover_prompt(self, generation_id: UUID, prompt_id: str) -> bool:
        """Compatibility wrapper for the pre-Phase 4 recovery service."""

        return (
            await self.reconcile_prompt(generation_id, prompt_id)
        ) is ReconciliationOutcome.COMPLETED

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

    async def execute_persisted(
        self,
        generation_id: UUID,
        job_id: UUID,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
        submission_coordinator: PromptSubmissionCoordinator | None = None,
    ) -> GenerationResult:
        """Execute an already persisted pending queue item without creating records."""

        if (
            self._generation_repository is None
            or self._job_repository is None
            or self._start_repository is None
        ):
            raise GenerationPersistenceError("generation persistence is unavailable")
        async with self._lock:
            generation = self._generation_repository.get_by_id(generation_id)
            persisted_job = self._job_repository.get_by_generation(generation_id)
            if generation is None or persisted_job is None:
                raise GenerationPersistenceError("persisted generation or job was not found")
            if persisted_job.id != job_id or persisted_job.generation_id != generation_id:
                raise GenerationPersistenceError("persisted generation and job do not match")
            if generation.status in {
                GenerationStatus.COMPLETED,
                GenerationStatus.FAILED,
                GenerationStatus.CANCELLED,
            }:
                return GenerationResult(
                    generation_id=generation_id,
                    prompt_id=generation.comfy_prompt_id or "",
                    status=generation.status,
                    seed=generation.settings_snapshot.seed,
                    stored_image=None,
                    error_message=generation.error_summary,
                    created_at=generation.created_at,
                )
            if generation.comfy_prompt_id or persisted_job.prompt_id:
                raise GenerationPersistenceError(
                    "prompt ID already exists; reconciliation is required"
                )
            if (
                generation.status is not GenerationStatus.PENDING
                or persisted_job.status is not GenerationStatus.PENDING
            ):
                raise GenerationPersistenceError("persisted queue item is not pending")

            settings = generation.settings_snapshot.to_generation_settings()
            job = GenerationJob(
                generation_id=generation_id,
                status=GenerationStatus.PENDING,
                id=job_id,
                prompt_id=None,
                created_at=generation.created_at,
                updated_at=persisted_job.updated_at,
            )
            self._jobs[generation_id] = job
            self._emit(
                progress_callback,
                GenerationProgress(
                    prompt_id="",
                    state=GenerationStatus.PENDING,
                    percentage=0.0,
                    message="キューから生成を開始します",
                ),
            )
            try:
                await asyncio.wait_for(
                    self._run_job(
                        job,
                        settings,
                        generation.created_at,
                        progress_callback,
                        generation.kind,
                        generation.parent_generation_id,
                        cancel_check=cancel_check,
                        submission_coordinator=submission_coordinator,
                    ),
                    timeout=self._settings.generation_timeout_seconds,
                )
                result = self._result_for_job(job, settings.seed, generation.created_at)
                if (
                    self._lora_catalog_service is not None
                    and settings.loras
                    and result.status is GenerationStatus.COMPLETED
                ):
                    try:
                        self._lora_catalog_service.record_usage(
                            tuple(lora.name for lora in settings.loras), datetime.now(UTC)
                        )
                    except Exception:  # noqa: BLE001 - usage statistics are best effort
                        logger.warning("LoRA usage statistics update failed", exc_info=True)
            except GenerationCancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - execution boundary persists safe failures
                if (
                    submission_coordinator is not None
                    and submission_coordinator.started
                    and not submission_coordinator.submitted
                ):
                    raise
                error_code = _generation_error_code(exc)
                error_summary = _safe_generation_error(exc)
                job.status = GenerationStatus.FAILED
                job.error_code = error_code
                job.error_summary = error_summary
                job.error_message = error_summary
                self._persist_failure(
                    job,
                    error_code=error_code,
                    error_summary=error_summary,
                    failed_at=datetime.now(UTC),
                )
                result = GenerationResult(
                    generation_id=generation_id,
                    prompt_id=job.prompt_id or "",
                    status=GenerationStatus.FAILED,
                    seed=settings.seed,
                    stored_image=None,
                    error_message=error_summary,
                    created_at=generation.created_at,
                )
            self._results[generation_id] = result
            return result

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
        if self._start_repository is None:
            return True
        try:
            snapshot = GenerationSettingsSnapshot.from_settings(settings)
            persisted_generation, persisted_job = self._start_repository.create_pending(
                snapshot,
                generation_id=generation_id,
                job_id=job.id,
                kind=kind,
                parent_generation_id=parent_generation_id,
                created_at=created_at,
            )
            if (
                persisted_generation.id != generation_id
                or persisted_job.id != job.id
                or persisted_job.generation_id != generation_id
                or persisted_generation.created_at != persisted_job.created_at
            ):
                raise GenerationRepositoryError("pending generation pair is inconsistent")
            job.created_at = persisted_job.created_at
            job.updated_at = persisted_job.updated_at
            return True
        except (GenerationRepositoryError, ValueError) as exc:
            logger.error(
                "Pending generation pair could not be persisted generation=%s job=%s error=%s",
                generation_id,
                job.id,
                type(exc).__name__,
                exc_info=True,
            )
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
        cancel_check: CancelCheck | None = None,
        submission_coordinator: PromptSubmissionCoordinator | None = None,
    ) -> None:
        if kind is GenerationKind.UPSCALE:
            if (
                self._upscale_settings_repository is None
                or self._upscale_workflow_adapter is None
                or self._artifact_repository is None
                or self._generation_repository is None
            ):
                raise GenerationPersistenceError("upscale persistence is unavailable")
            upscale_snapshot = self._upscale_settings_repository.get_by_generation(
                job.generation_id
            )
            if upscale_snapshot is None:
                raise GenerationPersistenceError("upscale settings were not found")
            if upscale_snapshot.source_kind is UpscaleSourceKind.GENERATION_ARTIFACT:
                if (
                    upscale_snapshot.source_generation_id is None
                    or upscale_snapshot.source_artifact_id is None
                ):
                    raise ComfyUIPromptError("upscale source artifact is incomplete")
                source_artifact = self._artifact_repository.get_primary_image(
                    upscale_snapshot.source_generation_id
                )
                if (
                    source_artifact is None
                    or source_artifact.id != upscale_snapshot.source_artifact_id
                ):
                    raise ComfyUIPromptError("upscale source artifact was not found")
                source = verify_source_artifact(source_artifact, self._settings)
                source_bytes = source.path.read_bytes()
                source_generation = self._generation_repository.get_by_id(
                    upscale_snapshot.source_generation_id
                )
                if source_generation is None:
                    raise ComfyUIPromptError("upscale parent generation was not found")
                source_settings = source_generation.settings_snapshot
            else:
                if (
                    upscale_snapshot.source_import_id is None
                    or self._metadata_import_repository is None
                    or self._imported_image_storage is None
                ):
                    raise ComfyUIPromptError("upscale import source is unavailable")
                imported = self._metadata_import_repository.get_by_id(
                    upscale_snapshot.source_import_id
                )
                if imported is None:
                    raise ComfyUIPromptError("upscale import source was not found")
                try:
                    source_bytes = self._imported_image_storage.read_verified(
                        imported.imported_image
                    )
                except ImportedImageStorageError as exc:
                    raise UpscaleEnqueueError(
                        exc.code, "metadata import source could not be verified"
                    ) from exc
                source_settings = GenerationSettingsSnapshot.from_settings(settings)
            await self._preflight_upscale(upscale_snapshot, source_settings)
            upload = getattr(self._client, "upload_input_image", None)
            if not callable(upload):
                raise ComfyUIPromptError("ComfyUI input image upload is unavailable")
            source_token = await upload(
                source_bytes, job.generation_id, upscale_snapshot.source_sha256
            )
            source_token_name = (
                source_token.filename if hasattr(source_token, "filename") else str(source_token)
            )
            if not source_token_name.strip():
                raise ComfyUIPromptError("ComfyUI input image upload returned no token")
            if upscale_snapshot.method.value == "image":
                workflow = self._upscale_workflow_adapter.build_image_upscale_workflow(
                    source_token_name, upscale_snapshot
                )
            else:
                workflow = self._upscale_workflow_adapter.build_latent_upscale_workflow(
                    source_token_name,
                    source_settings,
                    upscale_snapshot,
                )
        else:
            capabilities_result = await self._capabilities_provider()
            capabilities = capabilities_result.capabilities
            if not capabilities_result.is_success or capabilities is None:
                raise ComfyUIPromptError("ComfyUI capabilities are unavailable")
            _validate_generation(settings, capabilities, self._settings)

            if cancel_check is not None and cancel_check():
                raise GenerationCancelledError("generation was cancelled before prompt submission")

            workflow = self._workflow_adapter.build_txt2img_workflow(settings)

        if cancel_check is not None and cancel_check():
            raise GenerationCancelledError("generation was cancelled before prompt submission")
        submission_token = (
            submission_coordinator.begin()
            if submission_coordinator is not None
            else str(self._id_factory())
        )
        client_id = submission_token
        queued = await self._client.queue_prompt(workflow, client_id)
        if queued.node_errors:
            raise ComfyUIPromptError("ComfyUI rejected workflow nodes")
        job.prompt_id = queued.prompt_id
        job.status = GenerationStatus.QUEUED
        try:
            if submission_coordinator is not None:
                submission_coordinator.mark_submitted(queued.prompt_id, submission_token)
            elif self._queue_repository is not None:
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
                if cancel_check is not None and cancel_check():
                    raise GenerationCancelledError("generation was cancelled during execution")
                job.status = progress.state
                self._persist_progress(job, progress)
                self._emit(progress_callback, progress)
                if progress.state is GenerationStatus.FAILED:
                    websocket_failed = True
                    break
        except ComfyUIWebSocketError:
            websocket_failed = True
            logger.info("Falling back to ComfyUI history after WebSocket loss")

        if cancel_check is not None and cancel_check():
            raise GenerationCancelledError("generation was cancelled during execution")
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
        if cancel_check is not None and cancel_check():
            raise GenerationCancelledError("generation was cancelled during execution")
        persisted_upscale = None
        stored_images: list[StoredImage] = []
        for output in history.outputs:
            image_bytes = await self._client.get_output_image(output)
            if kind is GenerationKind.UPSCALE:
                persisted_upscale = self._validate_upscale_output(job.generation_id, image_bytes)
            stored_images.append(
                self._storage.store_image(
                    image_bytes,
                    job.generation_id,
                    created_at,
                    kind=kind,
                    client_local_date=settings.client_local_date,
                )
            )
        job.stored_images = tuple(stored_images)
        job.stored_image = stored_images[0]
        if (
            kind is GenerationKind.UPSCALE
            and persisted_upscale is None
            and self._upscale_settings_repository is not None
        ):
            persisted_upscale = self._upscale_settings_repository.get_by_generation(
                job.generation_id
            )
            if persisted_upscale is not None and (
                job.stored_image.width != persisted_upscale.target_width
                or job.stored_image.height != persisted_upscale.target_height
            ):
                raise UpscaleEnqueueError(
                    "upscale_output_dimension_mismatch",
                    "ComfyUIのアップスケール出力寸法が要求値と一致しません。",
                )
        if persisted_upscale is not None and job.stored_image is not None:
            self._ensure_stored_upscale_dimensions(job.stored_image, persisted_upscale)
        self._complete_job(job, settings, created_at, kind, parent_generation_id)

    async def _preflight_upscale(
        self,
        upscale_snapshot: UpscaleSettingsSnapshot,
        source_settings: GenerationSettingsSnapshot,
    ) -> None:
        """Check all deterministic ComfyUI requirements before source upload."""

        try:
            capabilities_result = await self._capabilities_provider()
        except Exception as exc:  # noqa: BLE001 - capability failure is retryable and safe
            raise UpscaleEnqueueError(
                "upscale_capabilities_unavailable",
                "ComfyUI capabilities could not be retrieved",
            ) from exc
        capabilities = capabilities_result.capabilities
        if not capabilities_result.is_success or capabilities is None:
            raise UpscaleEnqueueError(
                "upscale_capabilities_unavailable",
                "ComfyUI capabilities are unavailable",
            )

        required_nodes = (
            (
                "LoadImage",
                "UpscaleModelLoader",
                "ImageUpscaleWithModel",
                "ImageScale",
                "SaveImage",
            )
            if upscale_snapshot.method.value == "image"
            else (
                "LoadImage",
                "CheckpointLoaderSimple",
                "CLIPTextEncode",
                "VAEEncode",
                "LatentUpscale",
                "KSampler",
                "VAEDecode",
                "SaveImage",
            )
        )
        available_nodes: frozenset[str] = getattr(
            capabilities, "available_node_classes", frozenset()
        )
        if any(node not in available_nodes for node in required_nodes):
            raise UpscaleEnqueueError(
                "upscale_required_node_missing",
                "one or more required upscale nodes are unavailable",
            )

        if upscale_snapshot.method.value == "image":
            model_name = upscale_snapshot.upscaler_name or ""
            current_catalog = (
                self._upscaler_catalog_provider()
                if self._upscaler_catalog_provider is not None
                else self._upscaler_catalog
            )
            if model_name not in getattr(
                capabilities, "upscale_models", ()
            ) or not current_catalog.contains(model_name):
                raise UpscaleEnqueueError(
                    "upscale_model_missing",
                    "the selected upscaler is unavailable",
                )
            return

        if source_settings.checkpoint_name not in getattr(capabilities, "checkpoints", ()):
            raise UpscaleEnqueueError(
                "upscale_checkpoint_missing", "the source checkpoint is unavailable"
            )
        if source_settings.sampler_name not in getattr(capabilities, "samplers", ()):
            raise UpscaleEnqueueError("upscale_sampler_missing", "the sampler is unavailable")
        if source_settings.scheduler_name not in getattr(capabilities, "schedulers", ()):
            raise UpscaleEnqueueError("upscale_scheduler_missing", "the scheduler is unavailable")
        if len(source_settings.loras) > self._settings.max_loras:
            raise UpscaleEnqueueError(
                "upscale_lora_limit_exceeded", "the configured LoRA limit was exceeded"
            )
        missing_loras = tuple(
            lora.name
            for lora in source_settings.loras
            if lora.name not in getattr(capabilities, "loras", ())
        )
        if missing_loras or (source_settings.loras and "LoraLoader" not in available_nodes):
            raise UpscaleEnqueueError(
                "upscale_lora_missing", "one or more source LoRAs are unavailable"
            )
        if source_settings.vae_name is not None:
            if source_settings.vae_name not in getattr(capabilities, "vaes", ()):
                raise UpscaleEnqueueError("upscale_vae_missing", "the source VAE is unavailable")
            if "VAELoader" not in available_nodes:
                raise UpscaleEnqueueError(
                    "upscale_required_node_missing", "the external VAE node is unavailable"
                )

    def _persist_progress(self, job: GenerationJob, progress: GenerationProgress) -> None:
        try:
            if self._progress_repository is not None:
                self._progress_repository.update_progress(
                    generation_id=job.generation_id,
                    job_id=job.id,
                    state=progress.state,
                    value=progress.value,
                    maximum=progress.maximum,
                    current_node=progress.current_node,
                    updated_at=datetime.now(UTC),
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
        images = job.stored_images or (job.stored_image,)
        if any(image is None for image in images):
            raise GenerationPersistenceError("stored image is missing")
        stored_images = tuple(image for image in images if image is not None)
        image = stored_images[0]
        completed_at = datetime.now(UTC)
        primary_artifacts = tuple(
            self._primary_image_artifact(job, item, completed_at, display_order=index)
            for index, item in enumerate(stored_images)
        )
        if self._generation_repository is not None:
            if self._completion_repository is not None:
                try:
                    if len(primary_artifacts) == 1:
                        self._completion_repository.complete_generation(
                            job.generation_id,
                            job.id,
                            primary_artifacts[0],
                            completed_at,
                        )
                    else:
                        complete_batch = getattr(
                            self._completion_repository, "complete_generation_batch", None
                        )
                        if not callable(complete_batch):
                            raise GenerationRepositoryError(
                                "batch completion repository is unavailable"
                            )
                        complete_batch(job.generation_id, job.id, primary_artifacts, completed_at)
                except GenerationRepositoryError as exc:
                    logger.error(
                        "Generation completion persistence failed generation=%s job=%s "
                        "prompt_id=%s image_path=%s error=%s",
                        job.generation_id,
                        job.id,
                        job.prompt_id or "",
                        primary_artifacts[0].local_path,
                        type(exc).__name__,
                    )
                    error_type = (
                        RecoveryPersistenceError if is_recovery else CompletionPersistenceError
                    )
                    raise error_type("generation completion could not be persisted") from exc
            else:
                if self._job_repository is None:
                    raise CompletionPersistenceError("job persistence is unavailable")
                for primary_artifact in primary_artifacts:
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
            images=stored_images,
        )
        self._notify_drive_sync(job.generation_id)

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
        else:
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
        self._notify_drive_sync(generation_id)

    def _notify_drive_sync(self, generation_id: UUID) -> None:
        if self._drive_sync_enqueue_handler is None:
            return
        try:
            self._drive_sync_enqueue_handler(generation_id)
        except Exception as exc:  # noqa: BLE001 - Drive failure must not reopen generation
            logger.warning(
                "Drive sync enqueue warning generation=%s error=%s",
                generation_id,
                type(exc).__name__,
                exc_info=True,
            )

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
        *,
        existing_artifacts: tuple[GenerationArtifact, ...] | None = None,
        persisted_upscale_snapshot: UpscaleSettingsSnapshot | None = None,
        load_upscale_settings: bool = True,
        images: tuple[StoredImage, ...] | None = None,
    ) -> None:
        if self._artifact_repository is None:
            return
        if existing_artifacts is None:
            try:
                current_artifacts = self._artifact_repository.list_by_generation(job.generation_id)
            except Exception as exc:  # noqa: BLE001 - optional metadata must not change completion
                logger.warning(
                    "Optional artifact lookup warning generation=%s error=%s",
                    job.generation_id,
                    type(exc).__name__,
                )
                return
        else:
            current_artifacts = existing_artifacts
        existing_types = {artifact.artifact_type for artifact in current_artifacts}
        existing_thumbnail_orders = {
            artifact.display_order
            for artifact in current_artifacts
            if artifact.artifact_type is ArtifactType.THUMBNAIL
        }
        all_images = images or (image,)
        image_path = image.path
        if self._metadata_storage is not None:
            if ArtifactType.METADATA in existing_types:
                pass
            else:
                try:
                    payload = _sidecar_payload(
                        job,
                        settings,
                        image,
                        kind,
                        parent_generation_id,
                        created_at,
                        images=all_images,
                    )
                    if kind is GenerationKind.UPSCALE and load_upscale_settings:
                        if self._upscale_settings_repository is None:
                            upscale_snapshot = None
                        else:
                            upscale_snapshot = self._upscale_settings_repository.get_by_generation(
                                job.generation_id
                            )
                    else:
                        upscale_snapshot = persisted_upscale_snapshot
                    if upscale_snapshot is not None:
                        payload["upscale"] = upscale_snapshot.model_dump(mode="json")
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
                            display_order=0,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - optional metadata is best effort
                    logger.warning(
                        "Generation sidecar warning generation=%s error=%s",
                        job.generation_id,
                        type(exc).__name__,
                    )
        if self._thumbnail_storage is not None:
            for display_order, stored in enumerate(all_images):
                if display_order in existing_thumbnail_orders:
                    continue
                try:
                    if display_order == 0:
                        thumbnail_path = self._thumbnail_storage.save(
                            stored.path,
                            job.generation_id,
                            created_at,
                        )
                    else:
                        thumbnail_path = self._thumbnail_storage.save(
                            stored.path,
                            job.generation_id,
                            created_at,
                            display_order=display_order,
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
                            display_order=display_order,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - optional thumbnail is best effort
                    logger.warning(
                        "Generation thumbnail warning generation=%s order=%s error=%s",
                        job.generation_id,
                        display_order,
                        type(exc).__name__,
                    )

    def _primary_image_artifact(
        self,
        job: GenerationJob,
        image: StoredImage,
        created_at: datetime,
        *,
        display_order: int = 0,
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
            display_order=display_order,
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
            stored_images=job.stored_images or (job.stored_image,),
        )

    @staticmethod
    def _emit(callback: ProgressCallback | None, progress: GenerationProgress) -> None:
        if callback is not None:
            callback(progress)


def _resolve_seed(seed: int) -> int:
    return secrets.randbelow(MAX_SEED + 1) if seed == RANDOM_SEED else seed


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


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
    if settings.clip_skip > 1 and "CLIPSetLastLayer" not in node_classes:
        raise WorkflowError("CLIP skip is unavailable in ComfyUI")
    if settings.hires_fix:
        if settings.workflow_template_version == LEGACY_WORKFLOW_TEMPLATE_VERSION:
            required_hires_nodes = {"LatentUpscale", "VAEDecode"}
        elif settings.workflow_template_version in PIXEL_HIRES_WORKFLOW_VERSIONS:
            required_hires_nodes = {"ImageScaleBy", "VAEEncode", "VAEDecode"}
        else:
            raise WorkflowError("Hires.fix workflow version is unsupported")
        if not required_hires_nodes.issubset(node_classes):
            raise WorkflowError("Hires.fix is unavailable in ComfyUI")
    if settings.hires_fix and settings.width * settings.hires_scale > limits.max_width:
        raise WorkflowError("Hires.fix width exceeds the configured limit")
    if settings.hires_fix and settings.height * settings.hires_scale > limits.max_height:
        raise WorkflowError("Hires.fix height exceeds the configured limit")
    hires_pixels = settings.width * settings.hires_scale * settings.height * settings.hires_scale
    if settings.hires_fix and hires_pixels > limits.max_pixels:
        raise WorkflowError("Hires.fix image area exceeds the configured limit")
    if settings.hires_fix and settings.hires_sampler_name not in samplers:
        raise WorkflowError("Selected Hires.fix sampler is unavailable")
    if settings.hires_fix and settings.hires_scheduler_name not in schedulers:
        raise WorkflowError("Selected Hires.fix scheduler is unavailable")
    if settings.final_upscale:
        if "UpscaleModelLoader" not in node_classes or "ImageUpscaleWithModel" not in node_classes:
            raise WorkflowError("Final upscaling is unavailable in ComfyUI")
        final_model = settings.final_upscale_model
        if final_model is None:
            raise WorkflowError("A final upscale model must be selected")
        if final_model not in getattr(capabilities, "upscale_models", ()):
            raise WorkflowError("Selected final upscale model is unavailable")
    if settings.vae_name is not None and "VAELoader" not in node_classes:
        raise WorkflowError("External VAE loading is unavailable in ComfyUI")
    missing_loras = [lora.name for lora in settings.loras if lora.name not in loras]
    if missing_loras:
        logger.warning("Unavailable LoRAs selected: %s", ", ".join(missing_loras))
        raise WorkflowError("One or more selected LoRAs are unavailable")
    detector_models = getattr(capabilities, "detector_models", ())
    for detailer in settings.detailers:
        if not detailer.enabled:
            continue
        definition = DEFAULT_DETAILER_REGISTRY.get(detailer.kind)
        if definition is None:
            raise WorkflowError("Selected Detailer is unsupported")
        required_detailer_nodes = {
            definition.node_class,
            definition.detector_provider_class,
        }
        if not required_detailer_nodes.issubset(node_classes):
            if detailer.kind is DetailerKind.FACE:
                raise WorkflowError("FaceDetailer is unavailable in ComfyUI")
            raise WorkflowError("Selected Detailer is unavailable in ComfyUI")
        if detailer.kind is DetailerKind.FACE and detailer.detector_model not in detector_models:
            raise WorkflowError("Selected face detector is unavailable")
    if settings.width > limits.max_width or settings.height > limits.max_height:
        raise WorkflowError("Requested image dimensions exceed the configured limit")
    if settings.width * settings.height > limits.max_pixels:
        raise WorkflowError("Requested image area exceeds the configured limit")


def _safe_generation_error(
    error: Exception,
    *,
    original_error: Exception | None = None,
) -> str:
    if isinstance(error, UpscaleEnqueueError):
        return _safe_upscale_error(error)
    if isinstance(error, ImportedImageStorageError):
        return "metadata import source could not be verified"
    if isinstance(error, GenerationPersistenceError):
        if isinstance(error, FailurePersistenceError) and original_error is not None:
            return "生成に失敗しました。加えて、履歴の失敗状態を完全に保存できませんでした。"
        return persistence_error_message(error)
    if isinstance(error, UpscaleSettingsRepositoryError):
        return "upscale settings could not be read"
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


def _safe_upscale_error(error: UpscaleEnqueueError) -> str:
    messages = {
        "upscale_output_dimension_mismatch": "アップスケール出力の寸法が要求値と一致しません。",
        "upscale_output_invalid": "アップスケール出力画像を検証できません。",
        "upscale_settings_missing": "アップスケール設定を確認できません。",
        "upscale_capabilities_unavailable": "ComfyUIの機能情報を取得できません。",
        "upscale_required_node_missing": "アップスケールに必要なComfyUIノードがありません。",
        "upscale_checkpoint_missing": "アップスケールに必要なcheckpointがありません。",
        "upscale_sampler_missing": "アップスケールに必要なsamplerがありません。",
        "upscale_scheduler_missing": "アップスケールに必要なschedulerがありません。",
        "upscale_lora_missing": "アップスケールに必要なLoRAがありません。",
        "upscale_vae_missing": "アップスケールに必要なVAEがありません。",
        "upscale_model_missing": "アップスケールモデルがありません。",
    }
    return messages.get(error.code, "アップスケール処理を完了できません。")


def _generation_error_code(error: Exception) -> str:
    if isinstance(error, UpscaleEnqueueError):
        return error.code
    if isinstance(error, ImportedImageStorageError):
        return error.code
    if isinstance(error, UpscaleSettingsRepositoryError):
        return "upscale_settings_unavailable"
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
    image: StoredImage,
    kind: GenerationKind,
    parent_generation_id: UUID | None,
    completed_at: datetime,
    *,
    images: tuple[StoredImage, ...] = (),
) -> dict[str, object]:
    stored_images = images or (image,)
    return {
        "schema_version": 1,
        "generation_id": str(job.generation_id),
        "kind": kind.value,
        "parent_generation_id": str(parent_generation_id) if parent_generation_id else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": completed_at.isoformat(),
        "settings": GenerationSettingsSnapshot.from_settings(settings).model_dump(mode="json"),
        "workflow_template_id": settings.workflow_template_id,
        "workflow_template_version": settings.workflow_template_version,
        "comfy_prompt_id": job.prompt_id,
        "image": {
            "sha256": image.sha256,
            "width": image.width,
            "height": image.height,
            "mime_type": image.mime_type,
        },
        "images": [
            {
                "display_order": index,
                "sha256": item.sha256,
                "width": item.width,
                "height": item.height,
                "mime_type": item.mime_type,
            }
            for index, item in enumerate(stored_images)
        ],
    }
