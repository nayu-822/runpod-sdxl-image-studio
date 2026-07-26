"""Validated settings for one ordered LoRA application."""

from __future__ import annotations

import ntpath
import posixpath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoraSetting(BaseModel):
    """One LoRA selection and its model/CLIP strengths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    model_strength: float = Field(default=1.0, ge=-2.0, le=2.0)
    clip_strength: float = Field(default=1.0, ge=-2.0, le=2.0)
    order: int = Field(default=0, ge=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if not normalized or posixpath.isabs(normalized) or ntpath.isabs(normalized):
            raise ValueError("LoRA name must be a relative path")
        parts = normalized.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("LoRA name contains an unsafe path segment")
        return normalized
