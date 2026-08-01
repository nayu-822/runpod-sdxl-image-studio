"""Execution boundary for jobs already persisted by the dispatch queue."""

from __future__ import annotations

from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.domain.generation import GenerationResult
from runpod_sdxl_image_studio.services.generation_errors import GenerationPersistenceError
from runpod_sdxl_image_studio.services.generation_service import (
    CancelCheck,
    GenerationService,
    ProgressCallback,
)


class GenerationExecutionService:
    """Keep queue execution separate from queue creation and UI handlers."""

    def __init__(
        self,
        generation_service: GenerationService,
        queue_repository: GenerationDispatchQueueRepositoryProtocol | None = None,
    ) -> None:
        self._generation_service = generation_service
        self._queue_repository = queue_repository

    async def execute_persisted(
        self,
        generation_id: UUID,
        job_id: UUID,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> GenerationResult:
        if self._queue_repository is not None:
            item = self._queue_repository.get_queue_item(generation_id)
            if item is None or item.entry.job_id != job_id:
                raise GenerationPersistenceError("queue generation and job do not match")
        return await self._generation_service.execute_persisted(
            generation_id,
            job_id,
            progress_callback,
            cancel_check,
        )


__all__ = ["GenerationExecutionService"]
