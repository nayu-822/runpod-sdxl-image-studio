"""Typed domain models for safe external image metadata import."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.lora import LoraSetting


class MetadataSourceKind(StrEnum):
    COMFYUI_PROMPT = "comfyui_prompt"
    APP_SIDECAR = "app_sidecar"
    WORKFLOW = "workflow"
    NONE = "none"


class MetadataImportStatus(StrEnum):
    READY = "ready"
    NEEDS_MAPPING = "needs_mapping"
    METADATA_MISSING = "metadata_missing"
    INVALID_METADATA = "invalid_metadata"


class MetadataFieldStatus(StrEnum):
    RESOLVED = "resolved"
    MISSING = "missing"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


class MetadataImportError(ValueError):
    """Safe error raised when imported metadata cannot be applied."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# This is a character-independent contract: all raw metadata is limited by
# its UTF-8 encoded byte length at the adapter boundary and again by the
# domain model.  Settings may choose a smaller operational limit, never a
# larger one.
MAX_METADATA_RAW_BYTES = 4_000_000


class MetadataRawSource(BaseModel):
    """Original metadata text retained outside the executable image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: MetadataSourceKind
    raw_text: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("raw_text")
    @classmethod
    def validate_raw_text_size(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value.encode("utf-8")) > MAX_METADATA_RAW_BYTES:
            raise ValueError("raw metadata exceeds the UTF-8 byte limit")
        return value


class MetadataFieldResolution(BaseModel):
    """Resolution state for one candidate field shown by the preview UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(min_length=1)
    status: MetadataFieldStatus
    value: Any = None
    message: str = ""


class MetadataModelMapping(BaseModel):
    """An explicit user-selected replacement for one model reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_kind: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)


class ImportedImage(BaseModel):
    """Canonical local representation of an externally supplied image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    original_filename: str = Field(min_length=1, max_length=500)
    stored_image_path: str = Field(min_length=1)
    source_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stored_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_mime_type: str = Field(min_length=1)
    created_at: datetime

    @property
    def path(self) -> Path:
        """Return the stored relative path as a Path for adapter consumers."""

        return Path(self.stored_image_path)


class MetadataImportCandidate(BaseModel):
    """Normalized, non-executable candidate extracted from trusted fields only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    source_kind: MetadataSourceKind = MetadataSourceKind.NONE
    positive_prompt: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    sampler_name: str | None = None
    scheduler_name: str | None = None
    checkpoint_name: str | None = None
    vae_name: str | None = None
    loras: tuple[LoraSetting, ...] = ()
    workflow_template_id: str = "sdxl_txt2img"
    workflow_template_version: str = "1.0"
    unresolved_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    resolutions: tuple[MetadataFieldResolution, ...] = ()
    model_mappings: tuple[MetadataModelMapping, ...] = ()

    @property
    def is_generation_ready(self) -> bool:
        required = (
            self.positive_prompt,
            self.negative_prompt,
            self.seed,
            self.width,
            self.height,
            self.steps,
            self.cfg_scale,
            self.sampler_name,
            self.scheduler_name,
            self.checkpoint_name,
        )
        return not self.unresolved_fields and all(value is not None for value in required)

    def with_mappings(self, mappings: tuple[MetadataModelMapping, ...]) -> MetadataImportCandidate:
        """Apply only explicit mappings; model strengths and LoRA order are retained."""

        replacements = {
            (mapping.model_kind, mapping.source_name): mapping.target_name for mapping in mappings
        }
        checkpoint = (
            replacements.get(("checkpoint", self.checkpoint_name), self.checkpoint_name)
            if self.checkpoint_name is not None
            else None
        )
        vae = (
            replacements.get(("vae", self.vae_name), self.vae_name)
            if self.vae_name is not None
            else None
        )
        loras = tuple(
            lora.model_copy(update={"name": replacements.get(("lora", lora.name), lora.name)})
            for lora in self.loras
        )
        unresolved = tuple(
            field
            for field in self.unresolved_fields
            if not (
                (field in {"checkpoint", "checkpoint_name"} and checkpoint != self.checkpoint_name)
                or (field in {"vae", "vae_name"} and vae != self.vae_name)
                or (field == "loras" and loras != self.loras)
            )
        )
        return self.model_copy(
            update={
                "checkpoint_name": checkpoint,
                "vae_name": vae,
                "loras": loras,
                "unresolved_fields": unresolved,
                "model_mappings": mappings,
            }
        )

    def to_generation_settings(self) -> GenerationSettings:
        """Convert only a fully resolved candidate into executable settings."""

        if not self.is_generation_ready:
            raise MetadataImportError(
                "metadata_import_unresolved",
                "metadata has unresolved generation fields",
            )
        try:
            return GenerationSettings(
                positive_prompt=self.positive_prompt or "",
                negative_prompt=self.negative_prompt or "",
                seed=self.seed if self.seed is not None else -1,
                width=self.width or 0,
                height=self.height or 0,
                steps=self.steps or 0,
                cfg_scale=self.cfg_scale if self.cfg_scale is not None else 0.0,
                sampler_name=self.sampler_name or "",
                scheduler_name=self.scheduler_name or "",
                checkpoint_name=self.checkpoint_name or "",
                vae_name=self.vae_name,
                loras=self.loras,
                workflow_template_id=self.workflow_template_id,
                workflow_template_version=self.workflow_template_version,
            )
        except ValueError as exc:
            raise MetadataImportError(
                "metadata_import_unresolved", "normalized metadata is not valid generation input"
            ) from exc


class MetadataImportPreview(BaseModel):
    """Safe view model returned before any generation or upscale is started."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    imported_image: ImportedImage
    status: MetadataImportStatus
    metadata_source: MetadataSourceKind
    candidate: MetadataImportCandidate | None = None
    candidates: tuple[MetadataImportCandidate, ...] = ()
    selected_metadata_source: MetadataSourceKind | None = None
    sidecar_hash_confirmed: bool = False
    raw_sources: tuple[MetadataRawSource, ...] = ()
    warnings: tuple[str, ...] = ()
    unresolved_fields: tuple[str, ...] = ()
    model_mappings: tuple[MetadataModelMapping, ...] = ()
    created_at: datetime


class MetadataImportRecord(BaseModel):
    """Persistence-shaped metadata import record with no absolute paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID = Field(default_factory=uuid4)
    imported_image: ImportedImage
    metadata_source: MetadataSourceKind
    metadata_status: MetadataImportStatus
    raw_sources: tuple[MetadataRawSource, ...] = ()
    candidate: MetadataImportCandidate | None = None
    candidates: tuple[MetadataImportCandidate, ...] = ()
    selected_metadata_source: MetadataSourceKind | None = None
    sidecar_hash_confirmed: bool = False
    normalized_snapshot_json: str | None = None
    normalized_snapshot_schema_version: int | None = None
    manual_mappings: tuple[MetadataModelMapping, ...] = ()
    warnings: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "ImportedImage",
    "MetadataFieldResolution",
    "MetadataFieldStatus",
    "MetadataImportCandidate",
    "MetadataImportError",
    "MetadataImportPreview",
    "MetadataImportRecord",
    "MetadataImportStatus",
    "MetadataModelMapping",
    "MetadataRawSource",
    "MetadataSourceKind",
    "MAX_METADATA_RAW_BYTES",
]
