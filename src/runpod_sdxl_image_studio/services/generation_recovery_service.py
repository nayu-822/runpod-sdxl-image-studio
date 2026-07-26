"""Bounded, non-resubmitting recovery for unfinished generations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import ComfyUIError
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationJobRepositoryProtocol,
    GenerationRepositoryError,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import GenerationErrorCode

logger = logging.getLogger(__name__)


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
    ) -> None:
        app_settings = settings or get_settings()
        self._client = client
        self._generation_repository = generation_repository
        self._job_repository = job_repository
        self._artifact_repository = artifact_repository
        self._stale_seconds = app_settings.stale_pending_seconds
        self._max_items = app_settings.recovery_max_items
        self._completed_prompt_handler = completed_prompt_handler

    async def recover(self, now: datetime | None = None) -> tuple[str, ...]:
        timestamp = now or datetime.now(UTC)
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
                    created = job.created_at or timestamp
                    if (
                        generation.status.value == "pending"
                        and (timestamp - created).total_seconds() >= self._stale_seconds
                    ):
                        self._generation_repository.mark_failed(
                            job.generation_id,
                            GenerationErrorCode.RECOVERY.value,
                            "送信前の処理が長時間停止したため終了しました。",
                        )
                        self._job_repository.mark_failed(
                            job.id,
                            GenerationErrorCode.RECOVERY.value,
                            "送信前の処理が長時間停止したため終了しました。",
                        )
                        messages.append(f"{job.generation_id}: stale pending")
                    continue
                history = await self._client.get_prompt_history(job.prompt_id)
                if history.is_failed:
                    self._generation_repository.mark_failed(
                        job.generation_id,
                        GenerationErrorCode.COMFYUI_EXECUTION.value,
                        "ComfyUIで生成が失敗しました。",
                    )
                    self._job_repository.mark_failed(
                        job.id,
                        GenerationErrorCode.COMFYUI_EXECUTION.value,
                        "ComfyUIで生成が失敗しました。",
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
