"""Versioned, immutable persistence snapshot for an upscale request."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
    validate_upscaler_name,
)

CURRENT_UPSCALE_SNAPSHOT_SCHEMA_VERSION = 2


class UpscaleSourceKind(StrEnum):
    GENERATION_ARTIFACT = "generation_artifact"
    METADATA_IMPORT = "metadata_import"


class UpscaleSnapshotError(ValueError):
    """Raised when a persisted upscale snapshot is not safe to execute."""


class UpscaleSettingsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=CURRENT_UPSCALE_SNAPSHOT_SCHEMA_VERSION, ge=1)
    source_kind: UpscaleSourceKind = UpscaleSourceKind.GENERATION_ARTIFACT
    method: UpscaleMethod
    sizing_mode: UpscaleSizingMode
    requested_scale_factor: float | None = Field(default=None, gt=1.0)
    target_width: int = Field(gt=0)
    target_height: int = Field(gt=0)
    upscaler_name: str | None = None
    denoise: float | None = Field(default=None, ge=0.0, le=1.0)
    source_generation_id: UUID | None = None
    source_artifact_id: UUID | None = None
    source_import_id: UUID | None = None
    source_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    workflow_template_id: str = Field(min_length=1)
    workflow_template_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_combination(self) -> UpscaleSettingsSnapshot:
        if self.schema_version not in {1, CURRENT_UPSCALE_SNAPSHOT_SCHEMA_VERSION}:
            raise ValueError("unsupported upscale snapshot schema version")
        if self.sizing_mode is UpscaleSizingMode.FACTOR and self.requested_scale_factor is None:
            raise ValueError("factor upscale snapshot requires requested_scale_factor")
        if (
            self.sizing_mode is UpscaleSizingMode.DIMENSIONS
            and self.requested_scale_factor is not None
        ):
            raise ValueError("dimensions upscale snapshot cannot contain requested_scale_factor")
        if self.method is UpscaleMethod.IMAGE and (
            not self.upscaler_name or self.denoise is not None
        ):
            raise ValueError("image upscale snapshot has invalid model or denoise")
        if self.method is UpscaleMethod.IMAGE:
            validate_upscaler_name(self.upscaler_name)
        if self.method is UpscaleMethod.LATENT and (
            self.upscaler_name is not None or self.denoise is None
        ):
            raise ValueError("latent upscale snapshot has invalid model or denoise")
        if self.target_width % 64 or self.target_height % 64:
            raise ValueError("upscale snapshot dimensions must be multiples of 64")
        if self.target_width < self.source_width or self.target_height < self.source_height:
            raise ValueError("upscale snapshot output cannot be smaller than its source")
        if self.source_kind is UpscaleSourceKind.GENERATION_ARTIFACT:
            if self.source_generation_id is None or self.source_artifact_id is None:
                raise ValueError("generation source requires generation and artifact IDs")
            if self.source_import_id is not None:
                raise ValueError("generation source cannot have an import ID")
        elif (
            self.source_generation_id is not None
            or self.source_artifact_id is not None
            or self.source_import_id is None
        ):
            raise ValueError("import source requires only an import ID")
        return self

    @classmethod
    def from_settings(
        cls,
        settings: UpscaleSettings,
        *,
        source_generation_id: UUID | None = None,
        source_artifact_id: UUID | None = None,
        source_import_id: UUID | None = None,
        source_kind: UpscaleSourceKind | None = None,
        source_sha256: str,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> UpscaleSettingsSnapshot:
        resolved_kind = source_kind or (
            UpscaleSourceKind.METADATA_IMPORT
            if source_import_id is not None
            else UpscaleSourceKind.GENERATION_ARTIFACT
        )
        return cls(
            source_kind=resolved_kind,
            method=settings.method,
            sizing_mode=settings.sizing_mode,
            requested_scale_factor=settings.scale_factor,
            target_width=target_width,
            target_height=target_height,
            upscaler_name=settings.upscaler_name,
            denoise=settings.denoise,
            source_generation_id=source_generation_id,
            source_artifact_id=source_artifact_id,
            source_import_id=source_import_id,
            source_sha256=source_sha256,
            source_width=source_width,
            source_height=source_height,
            workflow_template_id=settings.workflow_template_id,
            workflow_template_version=settings.workflow_template_version,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> UpscaleSettingsSnapshot:
        try:
            parsed: Any = json.loads(payload)
            if not isinstance(parsed, dict):
                raise UpscaleSnapshotError("upscale snapshot JSON must be an object")
            version = parsed.get("schema_version")
            if version == 1:
                parsed = {
                    **parsed,
                    "schema_version": CURRENT_UPSCALE_SNAPSHOT_SCHEMA_VERSION,
                    "source_kind": UpscaleSourceKind.GENERATION_ARTIFACT.value,
                    "source_import_id": None,
                }
            elif version != CURRENT_UPSCALE_SNAPSHOT_SCHEMA_VERSION:
                raise UpscaleSnapshotError("unsupported upscale snapshot schema version")
            return cls.model_validate(parsed)
        except UpscaleSnapshotError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise UpscaleSnapshotError("upscale snapshot JSON is invalid") from exc

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_settings(self) -> UpscaleSettings:
        try:
            return UpscaleSettings(
                method=self.method,
                sizing_mode=self.sizing_mode,
                scale_factor=self.requested_scale_factor,
                target_width=self.target_width,
                target_height=self.target_height,
                upscaler_name=self.upscaler_name,
                denoise=self.denoise,
                workflow_template_id=self.workflow_template_id,
                workflow_template_version=self.workflow_template_version,
            )
        except ValidationError as exc:
            raise UpscaleSnapshotError("upscale snapshot settings are invalid") from exc


__all__ = [
    "CURRENT_UPSCALE_SNAPSHOT_SCHEMA_VERSION",
    "UpscaleSettingsSnapshot",
    "UpscaleSnapshotError",
    "UpscaleSourceKind",
]
