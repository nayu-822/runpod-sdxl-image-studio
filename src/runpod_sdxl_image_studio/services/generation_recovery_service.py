"""Bounded, non-resubmitting recovery for unfinished generations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import ComfyUIError
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    PromptHistoryStatus,
    RemotePromptStatus,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationCancellationRepositoryProtocol,
    GenerationFailureRepositoryProtocol,
    GenerationJobRepositoryProtocol,
    GenerationRepositoryError,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import Generation, GenerationErrorCode
from runpod_sdxl_image_studio.domain.job import GenerationJob

logger = logging.getLogger(__name__)

RECONCILIATION_PROMPT_MISSING_CODE = "reconciliation_prompt_missing"
RECONCILIATION_PROMPT_MISSING_SUMMARY = (
    "ComfyUI prompt was not found in either the remote queue or history "
    "after the reconciliation grace period"
)


class CompletedPromptHandler(Protocol):
    def __call__(self, generation_id: UUID, prompt_id: str) -> Awaitable[bool]: ...


class GenerationRecoveryService:
    def __init__(
        self,
        client: ComfyUIClient,
        generation_repository: GenerationRepositoryProtocol,
        job_repository: GenerationJobRepositoryProtocol,
        artifact_repository: GenerationArtifactRepositoryProtocol,
        settings: Settings | None = None,
        completed_prompt_handler: CompletedPromptHandler | None = None,
        failure_repository: GenerationFailureRepositoryProtocol | None = None,
        cancellation_repository: GenerationCancellationRepositoryProtocol | None = None,
    ) -> None:
        app_settings = settings or get_settings()
        self._client = client
        self._generation_repository = generation_repository
        self._job_repository = job_repository
        self._artifact_repository = artifact_repository
        self._stale_seconds = app_settings.stale_pending_seconds
        self._reconciliation_grace_seconds = app_settings.reconciliation_grace_seconds
        self._max_items = app_settings.recovery_max_items
        self._completed_prompt_handler = completed_prompt_handler
        self._failure_repository = failure_repository
        self._cancellation_repository = cancellation_repository

    async def recover(self, now: datetime | None = None) -> tuple[str, ...]:
        timestamp = _utc(now or datetime.now(UTC))
        messages: list[str] = []
        try:
            jobs = self._job_repository.list_recoverable(self._max_items)
        except GenerationRepositoryError:
            return ("未完了Generationの一覧を取得できませんでした。",)
        for job in jobs:
            try:
                generation = self._generation_repository.get_by_id(job.generation_id)
                if generation is None or generation.status.value in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    continue
                if job.prompt_id is None:
                    created = _utc(job.created_at or timestamp)
                    if (
                        generation.status.value == "pending"
                        and (timestamp - created).total_seconds() >= self._stale_seconds
                    ):
                        self._mark_failed(
                            job.generation_id,
                            job.id,
                            GenerationErrorCode.RECOVERY.value,
                            "送信前の処理が長時間停止したため終了しました。",
                            timestamp,
                        )
                        messages.append(f"{job.generation_id}: stale pending")
                    continue
                status_reader = getattr(self._client, "get_remote_prompt_status", None)
                if callable(status_reader):
                    remote_state = await status_reader(job.prompt_id)
                    if remote_state.status in {
                        RemotePromptStatus.PENDING,
                        RemotePromptStatus.IN_PROGRESS,
                        RemotePromptStatus.UNAVAILABLE,
                    }:
                        continue
                    if remote_state.status is RemotePromptStatus.NOT_FOUND:
                        if self._is_prompt_missing_stale(job, generation, timestamp):
                            self._mark_failed(
                                job.generation_id,
                                job.id,
                                RECONCILIATION_PROMPT_MISSING_CODE,
                                RECONCILIATION_PROMPT_MISSING_SUMMARY,
                                timestamp,
                            )
                            messages.append(f"{job.generation_id}: stale prompt not found")
                        continue
                    if remote_state.status is RemotePromptStatus.CANCELLED:
                        if self._cancellation_repository is None:
                            messages.append(
                                f"{job.generation_id}: cancelled persistence unavailable"
                            )
                        else:
                            self._cancellation_repository.cancel_generation(
                                job.generation_id,
                                job.id,
                                cancelled_at=timestamp,
                                error_code="comfyui_execution_interrupted",
                                error_summary="ComfyUI reported execution_interrupted",
                            )
                            messages.append(f"{job.generation_id}: cancelled")
                        continue
                    if remote_state.status is RemotePromptStatus.FAILED:
                        self._mark_failed(
                            job.generation_id,
                            job.id,
                            GenerationErrorCode.COMFYUI_EXECUTION.value,
                            "ComfyUI reported a failed remote prompt",
                            timestamp,
                        )
                        messages.append(f"{job.generation_id}: failed")
                        continue
                history = await self._client.get_prompt_history(job.prompt_id)
                if history.status is PromptHistoryStatus.NOT_FOUND:
                    if self._is_prompt_missing_stale(job, generation, timestamp):
                        self._mark_failed(
                            job.generation_id,
                            job.id,
                            RECONCILIATION_PROMPT_MISSING_CODE,
                            RECONCILIATION_PROMPT_MISSING_SUMMARY,
                            timestamp,
                        )
                        messages.append(f"{job.generation_id}: stale prompt not found")
                elif history.status is PromptHistoryStatus.INTERRUPTED:
                    if self._cancellation_repository is None:
                        messages.append(f"{job.generation_id}: cancelled persistence unavailable")
                    else:
                        self._cancellation_repository.cancel_generation(
                            job.generation_id,
                            job.id,
                            cancelled_at=timestamp,
                            error_code="comfyui_execution_interrupted",
                            error_summary="ComfyUI reported execution_interrupted",
                        )
                        messages.append(f"{job.generation_id}: cancelled")
                elif history.status is PromptHistoryStatus.FAILED:
                    self._mark_failed(
                        job.generation_id,
                        job.id,
                        GenerationErrorCode.COMFYUI_EXECUTION.value,
                        "ComfyUIで生成が失敗しました。",
                        timestamp,
                    )
                    messages.append(f"{job.generation_id}: failed")
                elif history.is_completed and self._completed_prompt_handler is not None:
                    if await self._completed_prompt_handler(job.generation_id, job.prompt_id):
                        messages.append(f"{job.generation_id}: completed")
                elif history.is_completed:
                    messages.append(f"{job.generation_id}: completed output requires processing")
            except (ComfyUIError, GenerationRepositoryError, OSError) as exc:
                logger.warning(
                    "Recovery warning generation=%s error=%s", job.generation_id, type(exc).__name__
                )
                messages.append(f"{job.generation_id}: 状態を維持しました")
        return tuple(messages)

    def _is_prompt_missing_stale(
        self,
        job: GenerationJob,
        generation: Generation,
        now: datetime,
    ) -> bool:
        reference = next(
            (
                value
                for value in (
                    job.updated_at,
                    job.created_at,
                    generation.updated_at,
                    generation.created_at,
                )
                if value is not None
            ),
            now,
        )
        reference_utc = _utc(reference)
        return (now - reference_utc).total_seconds() >= self._reconciliation_grace_seconds

    def _mark_failed(
        self,
        generation_id: UUID,
        job_id: UUID,
        error_code: str,
        error_summary: str,
        failed_at: datetime,
    ) -> None:
        if self._failure_repository is None:
            raise GenerationRepositoryError("atomic failure persistence is unavailable")
        self._failure_repository.fail_generation(
            generation_id,
            job_id,
            error_code=error_code,
            error_summary=error_summary,
            failed_at=failed_at,
        )


def _utc(value: datetime) -> datetime:
    """Normalize recovery timestamps before comparing or persisting them."""

    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
