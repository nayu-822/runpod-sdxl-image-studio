"""Generation output artifact domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class ArtifactType(StrEnum):
    IMAGE = "image"
    THUMBNAIL = "thumbnail"
    METADATA = "metadata"


@dataclass(frozen=True)
class GenerationArtifact:
    id: UUID
    generation_id: UUID
    artifact_type: ArtifactType
    local_path: str
    sha256: str
    size_bytes: int
    width: int | None
    height: int | None
    mime_type: str
    created_at: datetime
