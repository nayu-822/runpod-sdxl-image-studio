"""In-memory generation and progress models for Phase 1B."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class GenerationStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GenerationProgress:
    prompt_id: str = ""
    state: GenerationStatus = GenerationStatus.RUNNING
    current_node: str | None = None
    value: int | None = None
    maximum: int | None = None
    percentage: float | None = None
    message: str = ""


@dataclass(frozen=True)
class StoredImage:
    path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    mime_type: str


@dataclass(frozen=True)
class GenerationResult:
    generation_id: UUID
    prompt_id: str
    status: GenerationStatus
    seed: int
    stored_image: StoredImage | None
    error_message: str | None
    created_at: datetime
