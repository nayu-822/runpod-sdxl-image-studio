"""Minimal in-memory job state used by GenerationService."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from runpod_sdxl_image_studio.domain.generation import GenerationStatus, StoredImage


@dataclass
class GenerationJob:
    generation_id: UUID
    status: GenerationStatus
    prompt_id: str | None = None
    error_message: str | None = None
    stored_image: StoredImage | None = None
