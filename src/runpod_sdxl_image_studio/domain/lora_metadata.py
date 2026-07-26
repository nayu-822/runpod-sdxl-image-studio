"""Validated application-owned metadata for one local LoRA file."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from runpod_sdxl_image_studio.domain.lora import normalize_relative_lora_name

MAX_TRIGGER_WORDS = 50
MAX_TRIGGER_WORD_LENGTH = 100
MAX_COMPATIBLE_MODELS = 20
MAX_COMPATIBLE_MODEL_LENGTH = 100


def normalize_optional_text(value: object, max_length: int) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if "\n" in normalized or "\r" in normalized:
        raise ValueError("line breaks are not allowed")
    if len(normalized) > max_length:
        raise ValueError("text exceeds the maximum length")
    return normalized


def normalize_tokens(
    value: object,
    *,
    max_items: int,
    max_item_length: int,
) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value.split(",") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("tokens must be a sequence or comma-separated string")
    normalized: list[str] = []
    for item in values:
        token = str(item).strip()
        if not token:
            continue
        if len(token) > max_item_length:
            raise ValueError("token exceeds the maximum length")
        if token not in normalized:
            normalized.append(token)
    if len(normalized) > max_items:
        raise ValueError("too many tokens")
    return tuple(normalized)


class LoraMetadata(BaseModel):
    """Domain representation independent of SQLAlchemy and Gradio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID = Field(default_factory=uuid4)
    file_name: str = Field(min_length=1, max_length=500)
    display_name: str | None = None
    category: str | None = None
    is_favorite: bool = False
    trigger_words: tuple[str, ...] = ()
    recommended_model_strength: float | None = Field(default=None, ge=-2.0, le=2.0)
    recommended_clip_strength: float | None = Field(default=None, ge=-2.0, le=2.0)
    notes: str | None = Field(default=None, max_length=2000)
    compatible_models: tuple[str, ...] = ()
    thumbnail_path: str | None = None
    is_missing: bool = False
    usage_count: int = Field(default=0, ge=0)
    last_used_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        return normalize_relative_lora_name(value)

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: object) -> str | None:
        return normalize_optional_text(value, 100)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: object) -> str | None:
        return normalize_optional_text(value, 50)

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("trigger_words", mode="before")
    @classmethod
    def validate_trigger_words(cls, value: object) -> tuple[str, ...]:
        return normalize_tokens(
            value,
            max_items=MAX_TRIGGER_WORDS,
            max_item_length=MAX_TRIGGER_WORD_LENGTH,
        )

    @field_validator("compatible_models", mode="before")
    @classmethod
    def validate_compatible_models(cls, value: object) -> tuple[str, ...]:
        return normalize_tokens(
            value,
            max_items=MAX_COMPATIBLE_MODELS,
            max_item_length=MAX_COMPATIBLE_MODEL_LENGTH,
        )


class LoraMetadataUpdate(BaseModel):
    """Validated mutable metadata submitted by the management UI."""

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = None
    category: str | None = None
    is_favorite: bool = False
    trigger_words: tuple[str, ...] = ()
    recommended_model_strength: float | None = Field(default=None, ge=-2.0, le=2.0)
    recommended_clip_strength: float | None = Field(default=None, ge=-2.0, le=2.0)
    notes: str | None = Field(default=None, max_length=2000)
    compatible_models: tuple[str, ...] = ()

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: object) -> str | None:
        return normalize_optional_text(value, 100)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: object) -> str | None:
        return normalize_optional_text(value, 50)

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("trigger_words", mode="before")
    @classmethod
    def validate_trigger_words(cls, value: object) -> tuple[str, ...]:
        return normalize_tokens(
            value,
            max_items=MAX_TRIGGER_WORDS,
            max_item_length=MAX_TRIGGER_WORD_LENGTH,
        )

    @field_validator("compatible_models", mode="before")
    @classmethod
    def validate_compatible_models(cls, value: object) -> tuple[str, ...]:
        return normalize_tokens(
            value,
            max_items=MAX_COMPATIBLE_MODELS,
            max_item_length=MAX_COMPATIBLE_MODEL_LENGTH,
        )
