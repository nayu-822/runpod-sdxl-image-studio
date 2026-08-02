"""Typed validation and sizing rules for Phase 5 image upscaling."""

from __future__ import annotations

import math
import ntpath
import posixpath
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UpscaleMethod(StrEnum):
    IMAGE = "image"
    LATENT = "latent"


class UpscaleSizingMode(StrEnum):
    FACTOR = "factor"
    DIMENSIONS = "dimensions"


class UpscaleLoadLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UpscaleSize(NamedTuple):
    width: int
    height: int
    factor: float


class UpscaleSettings(BaseModel):
    """User-selected settings, deliberately independent from generation snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: UpscaleMethod
    sizing_mode: UpscaleSizingMode
    scale_factor: float | None = Field(default=None, gt=1.0)
    target_width: int | None = Field(default=None, gt=0)
    target_height: int | None = Field(default=None, gt=0)
    upscaler_name: str | None = None
    denoise: float | None = Field(default=None, ge=0.0, le=1.0)
    workflow_template_id: str = Field(default="", min_length=0)
    workflow_template_version: str = Field(default="1.0", min_length=1)

    @model_validator(mode="after")
    def validate_combination(self) -> UpscaleSettings:
        if self.sizing_mode is UpscaleSizingMode.FACTOR and self.scale_factor is None:
            raise ValueError("scale_factor is required for factor sizing")
        if self.sizing_mode is UpscaleSizingMode.DIMENSIONS and (
            self.target_width is None or self.target_height is None
        ):
            raise ValueError("target dimensions are required for dimensions sizing")
        if self.sizing_mode is UpscaleSizingMode.DIMENSIONS and self.scale_factor is not None:
            raise ValueError("scale_factor is not accepted for dimensions sizing")
        if self.method is UpscaleMethod.IMAGE:
            if not _safe_model_name(self.upscaler_name):
                raise ValueError("image upscale requires a safe upscaler_name")
            if self.denoise is not None:
                raise ValueError("image upscale does not accept denoise")
        else:
            if self.upscaler_name is not None:
                raise ValueError("latent upscale does not accept upscaler_name")
            if self.denoise is None:
                raise ValueError("latent upscale requires denoise")
        return self


def validate_upscaler_name(name: str | None) -> str:
    if not _safe_model_name(name):
        raise ValueError("upscaler name must be a non-empty relative path")
    assert name is not None
    return name.replace("\\", "/")


def resolve_output_size(
    settings: UpscaleSettings,
    source_width: int,
    source_height: int,
    *,
    max_width: int = 2048,
    max_height: int = 2048,
    max_pixels: int = 4_194_304,
    max_upscale_factor: float = 4.0,
) -> UpscaleSize:
    """Resolve a 64-aligned size using the verified source artifact dimensions."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if settings.sizing_mode is UpscaleSizingMode.FACTOR:
        assert settings.scale_factor is not None
        if settings.scale_factor > max_upscale_factor:
            raise ValueError("upscale factor exceeds the configured limit")
        width = math.ceil(source_width * settings.scale_factor / 64) * 64
        height = math.ceil(source_height * settings.scale_factor / 64) * 64
    else:
        assert settings.target_width is not None and settings.target_height is not None
        width, height = settings.target_width, settings.target_height
        if width % 64 or height % 64:
            raise ValueError("target dimensions must be multiples of 64")
        if max(width / source_width, height / source_height) > max_upscale_factor:
            raise ValueError("upscale dimensions exceed the configured factor limit")
    if width < source_width or height < source_height:
        raise ValueError("upscale output cannot be smaller than the source")
    if width > max_width or height > max_height or width * height > max_pixels:
        raise ValueError("upscale output exceeds configured limits")
    return UpscaleSize(width, height, width / source_width)


def estimate_load_level(
    method: UpscaleMethod,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> UpscaleLoadLevel:
    if source_width <= 0 or source_height <= 0 or output_width <= 0 or output_height <= 0:
        raise ValueError("source and output dimensions must be positive")
    pixel_ratio = (output_width * output_height) / (source_width * source_height)
    if method is UpscaleMethod.IMAGE:
        return (
            UpscaleLoadLevel.LOW
            if pixel_ratio <= 2
            else UpscaleLoadLevel.MEDIUM
            if pixel_ratio <= 4
            else UpscaleLoadLevel.HIGH
        )
    return (
        UpscaleLoadLevel.LOW
        if pixel_ratio <= 1.5
        else UpscaleLoadLevel.MEDIUM
        if pixel_ratio <= 3
        else UpscaleLoadLevel.HIGH
    )


def estimate_load(*args: object, **kwargs: object) -> UpscaleLoadLevel:
    """Compatibility alias used by UI and callers that call this a load estimate."""

    return estimate_load_level(*args, **kwargs)  # type: ignore[arg-type]


def _safe_model_name(name: str | None) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    normalized = name.replace("\\", "/")
    return not (
        posixpath.isabs(normalized)
        or ntpath.isabs(name)
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    )


__all__ = [
    "UpscaleLoadLevel",
    "UpscaleMethod",
    "UpscaleSettings",
    "UpscaleSize",
    "UpscaleSizingMode",
    "estimate_load",
    "estimate_load_level",
    "resolve_output_size",
    "validate_upscaler_name",
]
