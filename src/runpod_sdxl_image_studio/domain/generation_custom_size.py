"""Persisted custom generation-size preferences."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerationCustomSize(BaseModel):
    """A validated size preset independent from a generation snapshot."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("width", "height")
    @classmethod
    def validate_multiple_of_64(cls, value: int) -> int:
        if value % 64 != 0:
            raise ValueError("custom generation sizes must be multiples of 64")
        return value

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @property
    def label(self) -> str:
        return f"Custom {self.width} × {self.height}"

    @property
    def selector_value(self) -> str:
        return f"custom:{self.id}"


__all__ = ["GenerationCustomSize"]
