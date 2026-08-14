"""Minimal in-memory job state used by GenerationService."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.domain.generation import GenerationStatus, StoredImage


@dataclass
class GenerationJob:
    generation_id: UUID
    status: GenerationStatus
    id: UUID = field(default_factory=uuid4)
    prompt_id: str | None = None
    progress_value: int | None = None
    progress_maximum: int | None = None
    current_node: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    error_code: str | None = None
    error_summary: str | None = None
    error_message: str | None = None
    stored_image: StoredImage | None = None
    stored_images: tuple[StoredImage, ...] = ()
    worker_id: str | None = None
    claimed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    cancelled_at: datetime | None = None
