"""Resolve opt-in LoRA trigger words at the generation boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.lora_search import append_trigger_words


class LoraTriggerResolutionError(ValueError):
    """Raised when enabled trigger metadata cannot be resolved safely."""


class LoraTriggerMetadata(Protocol):
    file_name: str
    trigger_words: tuple[str, ...]
    is_missing: bool


class LoraTriggerCatalog(Protocol):
    def get_by_file_name(self, file_name: str) -> Any: ...


def resolve_effective_positive_prompt(
    prompt: str,
    loras: Sequence[LoraSetting],
    catalog: LoraTriggerCatalog | None,
) -> str:
    """Append enabled trigger words in LoRA order using exact filename lookup.

    The UI state is deliberately not mutated.  The returned prompt is the
    value passed into GenerationSettings and therefore the value persisted in
    the immutable execution snapshot.
    """

    effective = prompt or ""
    for lora in sorted(loras, key=lambda item: item.order):
        if not lora.auto_add_trigger_words:
            continue
        if catalog is None:
            raise LoraTriggerResolutionError("LoRAのトリガーワード情報を取得できませんでした。")
        try:
            metadata = catalog.get_by_file_name(lora.name)
        except Exception as exc:  # noqa: BLE001 - safe generation boundary
            raise LoraTriggerResolutionError(
                "LoRAのトリガーワード情報を取得できませんでした。"
            ) from exc
        if metadata is None or metadata.is_missing or metadata.file_name != lora.name:
            raise LoraTriggerResolutionError("LoRAのトリガーワード情報を取得できませんでした。")
        try:
            effective = append_trigger_words(effective, metadata.trigger_words)
        except ValueError as exc:
            raise LoraTriggerResolutionError(
                "LoRAのトリガーワードを追加できませんでした。"
            ) from exc
    return effective


__all__ = ["LoraTriggerResolutionError", "resolve_effective_positive_prompt"]
