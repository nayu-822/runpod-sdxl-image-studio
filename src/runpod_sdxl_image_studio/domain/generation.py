"""Typed generation, progress, and result models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot


class GenerationStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationKind(StrEnum):
    STANDARD = "standard"
    DERIVED = "derived"
    UPSCALE = "upscale"


class GenerationErrorCode(StrEnum):
    VALIDATION = "validation_error"
    WORKFLOW = "workflow_error"
    COMFYUI_CONNECTION = "comfyui_connection_error"
    COMFYUI_PROMPT = "comfyui_prompt_error"
    COMFYUI_EXECUTION = "comfyui_execution_error"
    HISTORY_TIMEOUT = "history_timeout"
    OUTPUT_NOT_FOUND = "output_not_found"
    IMAGE_DOWNLOAD = "image_download_error"
    IMAGE_VALIDATION = "image_validation_error"
    STORAGE = "storage_error"
    DATABASE = "database_error"
    RECOVERY = "recovery_error"


def is_valid_status_transition(current: GenerationStatus, target: GenerationStatus) -> bool:
    """Return whether a persisted generation may move to ``target``."""

    if current is target:
        return True
    allowed = {
        GenerationStatus.PENDING: {
            GenerationStatus.QUEUED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        },
        GenerationStatus.QUEUED: {
            GenerationStatus.RUNNING,
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        },
        GenerationStatus.RUNNING: {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        },
    }
    return target in allowed.get(current, set())


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
class Generation:
    """Persisted generation record represented independently of SQLAlchemy."""

    id: UUID
    kind: GenerationKind
    status: GenerationStatus
    parent_generation_id: UUID | None
    settings_snapshot: GenerationSettingsSnapshot
    workflow_template_id: str
    workflow_template_version: str
    comfy_prompt_id: str | None
    favorite: bool
    user_note: str | None
    error_code: str | None
    error_summary: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime
    retry_of_generation_id: UUID | None = None
    retry_attempt: int = 0


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
