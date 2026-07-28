"""History query and paging models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from runpod_sdxl_image_studio.domain.generation import Generation, GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot


@dataclass(frozen=True)
class GenerationHistoryFilter:
    date: date | None = None
    status: GenerationStatus | None = None
    favorite: bool | None = None
    kind: GenerationKind | None = None
    offset: int = 0
    limit: int = 20
    start_utc: datetime | None = None
    end_utc: datetime | None = None


@dataclass(frozen=True)
class GenerationHistoryPage:
    generations: tuple[Generation, ...]
    page: int
    page_size: int
    total_count: int
    has_next: bool


@dataclass(frozen=True)
class GenerationHistoryItem:
    generation_id: str
    created_at_text: str
    status_text: str
    checkpoint_label: str
    lora_labels: tuple[str, ...]
    seed_text: str
    resolution_text: str
    thumbnail_path: str | None
    favorite: bool
    kind_text: str
    error_summary: str | None


@dataclass(frozen=True)
class GenerationDetailView:
    generation_id: str
    image_path: str | None
    thumbnail_path: str | None
    kind_text: str
    status_text: str
    parent_generation_id: str | None
    created_at_text: str
    started_at_text: str | None
    completed_at_text: str | None
    snapshot: GenerationSettingsSnapshot
    comfy_prompt_id: str | None
    image_sha256: str | None
    image_size_bytes: int | None
    favorite: bool
    user_note: str | None
    error_summary: str | None
    restore_warnings: tuple[str, ...]


@dataclass(frozen=True)
class RestoreSettingsResult:
    settings: GenerationSettings
    warnings: tuple[str, ...]
    parent_generation_id: UUID
    capability_unverified: bool = False


@dataclass(frozen=True)
class RegenerationPlan:
    settings: GenerationSettings
    parent_generation_id: UUID
    kind: GenerationKind = GenerationKind.DERIVED
