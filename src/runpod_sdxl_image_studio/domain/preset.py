"""Preset domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.domain.preset_payload import (
    CURRENT_PRESET_SCHEMA_VERSION,
    PresetKind,
    PresetPayloadError,
    PresetPayloadValue,
    payload_for_kind,
)

MAX_PRESET_NAME_LENGTH = 100
MAX_PRESET_DESCRIPTION_LENGTH = 1000


@dataclass(frozen=True)
class Preset:
    """ユーザーが再利用する型付き設定。"""

    id: UUID
    kind: PresetKind
    name: str
    description: str | None
    payload: PresetPayloadValue
    schema_version: int
    favorite: bool
    usage_count: int
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise PresetPayloadError("preset name is required")
        if len(name) > MAX_PRESET_NAME_LENGTH:
            raise PresetPayloadError("preset name is too long")
        description = self.description.strip() if self.description is not None else None
        if description == "":
            description = None
        if description is not None and len(description) > MAX_PRESET_DESCRIPTION_LENGTH:
            raise PresetPayloadError("preset description is too long")
        if self.schema_version != CURRENT_PRESET_SCHEMA_VERSION:
            raise PresetPayloadError("unsupported preset schema version")
        payload_for_kind(self.kind, self.payload)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "created_at", _utc(self.created_at))
        object.__setattr__(self, "updated_at", _utc(self.updated_at))
        object.__setattr__(self, "last_used_at", _utc(self.last_used_at))

    @classmethod
    def create(
        cls,
        kind: PresetKind,
        name: str,
        payload: PresetPayloadValue,
        *,
        description: str | None = None,
        favorite: bool = False,
        now: datetime | None = None,
    ) -> Preset:
        timestamp = _utc(now or datetime.now(UTC))
        assert timestamp is not None
        return cls(
            uuid4(),
            kind,
            name,
            description,
            payload,
            payload.schema_version,
            favorite,
            0,
            None,
            timestamp,
            timestamp,
        )


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
