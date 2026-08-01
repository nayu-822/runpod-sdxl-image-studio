"""Gradio boundary for safe preset operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.preset_payload import PresetKind
from runpod_sdxl_image_studio.services.preset_service import PresetService, PresetServiceError


@dataclass(frozen=True)
class PresetTabComponents:
    search: gr.Textbox
    kind: gr.Dropdown
    favorite_only: gr.Checkbox
    results: gr.Dropdown
    message: gr.Markdown
    refresh: gr.Button


def build_preset_tab() -> PresetTabComponents:
    """Build a compact, mobile-friendly preset selector."""

    gr.Markdown("## プリセット")
    search = gr.Textbox(label="Preset検索")
    with gr.Row():
        kind = gr.Dropdown(
            [("すべて", ""), *((item.value, item.value) for item in PresetKind)],
            value="",
            label="種類",
        )
        favorite_only = gr.Checkbox(label="お気に入りのみ")
    results = gr.Dropdown([], label="Preset", allow_custom_value=False)
    refresh = gr.Button("Presetを検索")
    message = gr.Markdown("Presetを検索してください。")
    return PresetTabComponents(search, kind, favorite_only, results, message, refresh)


def make_preset_search_handler(
    service: PresetService,
) -> Callable[[str | None, str | None, bool], tuple[object, str]]:
    """Return a UI handler that exposes view values, not ORM rows."""

    def handler(text: str | None, kind: str | None, favorite_only: bool) -> tuple[object, str]:
        try:
            selected_kind = PresetKind(kind) if kind else None
            presets = service.search(text, kind=selected_kind, favorite_only=favorite_only)
            choices = [(preset.name, str(preset.id)) for preset in presets]
            return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), (
                f"{len(choices)}件"
            )
        except (PresetServiceError, ValueError) as exc:
            return gr.skip(), str(exc)

    return handler


def seed_copy_value(seed: int) -> int:
    """Return only the resolved integer seed for an explicit copy action."""

    return int(seed)


def preset_id(value: str | None) -> UUID | None:
    """Parse a selected preset id without making it executable input."""

    if not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None
