"""Search and prompt helpers for the LoRA catalog."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoraSort(StrEnum):
    FAVORITES_RECENT = "favorites_recent"
    RECENT = "recent"
    USAGE = "usage"
    NAME = "name"


class LoraSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(default="", max_length=200)
    category: str | None = Field(default=None, max_length=50)
    favorites_only: bool = False
    include_missing: bool = False
    sort: LoraSort = LoraSort.FAVORITES_RECENT

    @field_validator("text", "category")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


def append_trigger_words(
    prompt: str,
    trigger_words: tuple[str, ...],
    *,
    max_length: int = 10_000,
) -> str:
    """Append unique trigger tokens without changing the negative prompt."""

    existing = prompt.strip()
    additions: list[str] = []
    for word in trigger_words:
        token = word.strip()
        if token and token not in additions and not _contains_prompt_token(existing, token):
            additions.append(token)
    if not additions:
        return prompt
    result = ", ".join(part for part in (existing, *additions) if part)
    if len(result) > max_length:
        raise ValueError("prompt exceeds the maximum length")
    return result


def _contains_prompt_token(prompt: str, token: str) -> bool:
    return token in {part.strip() for part in prompt.split(",")}
