"""Durable state models for the Phase A interactive generation session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot


class InteractiveRunStatus(StrEnum):
    ACTIVE = "active"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class InteractiveGenerationRun:
    id: UUID
    status: InteractiveRunStatus
    batch_count: int
    batch_size: int
    settings_snapshot: GenerationSettingsSnapshot
    client_local_date: str
    generation_ids: tuple[UUID, ...]
    completed_generation_ids: tuple[UUID, ...]
    current_generation_id: UUID | None
    last_completed_generation_id: UUID | None
    cancel_requested_at: datetime | None
    error_code: str | None
    error_summary: str | None
    created_at: datetime
    completed_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True)
class InteractiveRunView:
    """A safe UI-facing projection; raw ComfyUI payloads never enter it."""

    run: InteractiveGenerationRun
    completed_count: int
    current_generation_status: str | None
    result_image_paths: tuple[Path, ...]


__all__ = ["InteractiveGenerationRun", "InteractiveRunStatus", "InteractiveRunView"]
