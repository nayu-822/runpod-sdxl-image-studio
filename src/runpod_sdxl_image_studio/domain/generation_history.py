"""History query and paging models."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from uuid import UUID

from runpod_sdxl_image_studio.domain.generation import Generation, GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot

MAX_HISTORY_SEARCH_TEXT_LENGTH = 500
MAX_HISTORY_PAGE_SIZE = 100


class GenerationHistorySort(StrEnum):
    """履歴一覧の安定した並び順。"""

    NEWEST = "newest"
    OLDEST = "oldest"
    SEED_ASC = "seed_asc"
    SEED_DESC = "seed_desc"
    RESOLUTION_DESC = "resolution_desc"
    RECENTLY_COMPLETED = "recently_completed"


class LoraSearchMode(StrEnum):
    """複数LoRA条件の結合方法。"""

    ANY = "any"
    ALL = "all"


@dataclass(frozen=True)
class GenerationHistoryQuery:
    """SQL検索へ渡す正規化済みの履歴検索条件。"""

    text: str | None = None
    checkpoint_names: tuple[str, ...] = ()
    vae_names: tuple[str, ...] = ()
    lora_names: tuple[str, ...] = ()
    lora_search_mode: LoraSearchMode = LoraSearchMode.ANY
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    statuses: tuple[GenerationStatus, ...] = ()
    kinds: tuple[GenerationKind, ...] = ()
    favorite_only: bool = False
    error_codes: tuple[str, ...] = ()
    parent_generation_id: UUID | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    sort: GenerationHistorySort = GenerationHistorySort.NEWEST
    page_size: int = 20
    cursor: str | None = None
    # Phase 3A compatibility fields. New callers should use the fields above.
    date: date | None = None
    status: GenerationStatus | None = None
    favorite: bool | None = None
    kind: GenerationKind | None = None
    offset: int = 0
    limit: int = 20
    start_utc: datetime | None = None
    end_utc: datetime | None = None

    def __post_init__(self) -> None:
        text = self.text.strip() if self.text is not None else None
        if text == "":
            text = None
        if text is not None and len(text) > MAX_HISTORY_SEARCH_TEXT_LENGTH:
            raise ValueError("history search text is too long")
        object.__setattr__(self, "text", text)
        for field_name in ("checkpoint_names", "vae_names", "lora_names", "error_codes"):
            values = tuple(
                value.strip()
                for value in getattr(self, field_name)
                if isinstance(value, str) and value.strip()
            )
            object.__setattr__(self, field_name, values)
        object.__setattr__(
            self, "statuses", tuple(GenerationStatus(value) for value in self.statuses)
        )
        object.__setattr__(self, "kinds", tuple(GenerationKind(value) for value in self.kinds))
        object.__setattr__(self, "lora_search_mode", LoraSearchMode(self.lora_search_mode))
        object.__setattr__(self, "sort", GenerationHistorySort(self.sort))
        requested_size = self.page_size if self.page_size != 20 else self.limit
        if requested_size < 1 or requested_size > MAX_HISTORY_PAGE_SIZE:
            raise ValueError("history page size is outside the allowed range")
        object.__setattr__(self, "page_size", requested_size)
        object.__setattr__(self, "limit", requested_size)
        if self.offset < 0:
            raise ValueError("history offset must not be negative")


GenerationHistoryFilter = GenerationHistoryQuery


@dataclass(frozen=True)
class GenerationHistoryPage:
    generations: tuple[Generation, ...]
    page: int
    page_size: int
    total_count: int
    has_next: bool
    next_cursor: str | None = None
    previous_cursor: str | None = None


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


def encode_history_cursor(sort: GenerationHistorySort, sort_value: str, generation_id: str) -> str:
    """Encode cursor data without exposing it as SQL text."""

    payload = json.dumps({"sort": sort.value, "value": sort_value, "id": generation_id})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def decode_history_cursor(cursor: str | None) -> tuple[str, str] | None:
    """Safely parse a cursor; malformed cursors start from the first page."""

    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("value"), str)
            or not isinstance(payload.get("id"), str)
        ):
            return None
        if not UUID(payload["id"]):
            return None
        return payload["value"], payload["id"]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
