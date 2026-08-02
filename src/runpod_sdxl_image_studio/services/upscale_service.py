"""Execution facade for persisted upscale jobs.

The specialized workflow and source checks live in ``GenerationService`` so the
same completion, failure, and reconciliation rules are shared with Phase 4.
This facade lets the queue worker route by GenerationKind without exposing that
implementation detail to the worker.
"""

from __future__ import annotations

from uuid import UUID

from runpod_sdxl_image_studio.domain.generation import GenerationResult
from runpod_sdxl_image_studio.domain.generation_queue import ReconciliationOutcome
from runpod_sdxl_image_studio.services.generation_service import (
    CancelCheck,
    GenerationService,
    ProgressCallback,
    PromptSubmissionCoordinator,
)


class UpscaleService:
    def __init__(self, generation_service: GenerationService) -> None:
        self._generation_service = generation_service

    async def execute_persisted(
        self,
        generation_id: UUID,
        job_id: UUID,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
        submission_coordinator: PromptSubmissionCoordinator | None = None,
    ) -> GenerationResult:
        return await self._generation_service.execute_persisted(
            generation_id,
            job_id,
            progress_callback,
            cancel_check,
            submission_coordinator,
        )

    async def reconcile_prompt(self, generation_id: UUID, prompt_id: str) -> ReconciliationOutcome:
        return await self._generation_service.reconcile_prompt(generation_id, prompt_id)


__all__ = ["UpscaleService"]
