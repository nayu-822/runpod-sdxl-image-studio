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
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import (
    GenerationProgress,
    GenerationResult,
    GenerationStatus,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.job import GenerationJob
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[GenerationProgress], None]


class CapabilitiesProvider(Protocol):
    def __call__(self) -> Awaitable[CapabilityRefreshResult]: ...


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
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._client = client
        self._workflow_adapter = workflow_adapter
        self._websocket_client = websocket_client
        self._storage = storage
        self._capabilities_provider = capabilities_provider
        self._settings = settings or get_settings()
        self._sleep = sleep
        self._id_factory = id_factory
        self._jobs: dict[UUID, GenerationJob] = {}
        self._results: dict[UUID, GenerationResult] = {}
        self._lock = asyncio.Lock()

    @property
    def jobs(self) -> Mapping[UUID, GenerationJob]:
        return self._jobs

    def get_result(self, generation_id: UUID) -> GenerationResult | None:
        """Return a result while the process is alive."""

        return self._results.get(generation_id)

    async def generate(
        self,
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Run one generation and retain its result in memory."""

        async with self._lock:
            return await self._generate_locked(settings, progress_callback)

    async def _generate_locked(
        self,
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None,
    ) -> GenerationResult:
        generation_id = self._id_factory()
        created_at = datetime.now(UTC)
        seed = _resolve_seed(settings.seed)
        resolved_settings = settings.model_copy(update={"seed": seed})
        job = GenerationJob(generation_id=generation_id, status=GenerationStatus.PENDING)
        self._jobs[generation_id] = job
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
                self._run_job(job, resolved_settings, progress_callback),
                timeout=self._settings.generation_timeout_seconds,
            )
            result = self._result_for_job(job, seed, created_at)
        except Exception as exc:  # noqa: BLE001 - boundary converts failures to safe UI text
            logger.error("Generation job failed: %s", type(exc).__name__)
            job.status = GenerationStatus.FAILED
            job.error_message = _safe_generation_error(exc)
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

    async def _run_job(
        self,
        job: GenerationJob,
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None,
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
            datetime.now(UTC),
        )
        job.status = GenerationStatus.COMPLETED

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
        raise ComfyUIPromptError("ComfyUI history did not become available")

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
    if settings.checkpoint_name not in checkpoints:
        raise WorkflowError("Selected checkpoint is unavailable")
    if settings.sampler_name not in samplers:
        raise WorkflowError("Selected sampler is unavailable")
    if settings.scheduler_name not in schedulers:
        raise WorkflowError("Selected scheduler is unavailable")
    if settings.width > limits.max_width or settings.height > limits.max_height:
        raise WorkflowError("Requested image dimensions exceed the configured limit")
    if settings.width * settings.height > limits.max_pixels:
        raise WorkflowError("Requested image area exceeds the configured limit")


def _safe_generation_error(error: Exception) -> str:
    if isinstance(error, WorkflowError):
        return "生成設定またはworkflowを検証できませんでした"
    if isinstance(error, StorageError):
        return "生成画像をローカルへ保存できませんでした"
    if isinstance(error, ComfyUIError):
        return "ComfyUIで画像生成を完了できませんでした"
    if isinstance(error, TimeoutError):
        return "画像生成が制限時間を超えました"
    return "画像生成に失敗しました"
