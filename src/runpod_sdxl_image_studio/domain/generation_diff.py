"""Typed, safe differences between two generation snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class ChangeType(StrEnum):
    """差分の種類。"""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    REORDERED = "reordered"


@dataclass(frozen=True)
class ValueChange:
    field_name: str
    before: object
    after: object
    change_type: ChangeType


@dataclass(frozen=True)
class PromptTokenChange:
    value: str
    change_type: ChangeType
    before_index: int | None
    after_index: int | None


@dataclass(frozen=True)
class LoraChange:
    name: str
    change_type: ChangeType
    before_order: int | None = None
    after_order: int | None = None
    before_model_strength: float | None = None
    after_model_strength: float | None = None
    before_clip_strength: float | None = None
    after_clip_strength: float | None = None


@dataclass(frozen=True)
class GenerationDiff:
    source_generation_id: UUID
    target_generation_id: UUID
    positive_prompt_changes: tuple[PromptTokenChange, ...]
    negative_prompt_changes: tuple[PromptTokenChange, ...]
    setting_changes: tuple[ValueChange, ...]
    lora_changes: tuple[LoraChange, ...]
