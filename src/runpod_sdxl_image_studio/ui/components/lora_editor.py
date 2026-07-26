"""A bounded, mobile-friendly editor for ordered LoRA settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias
from uuid import uuid4

import gradio as gr

from runpod_sdxl_image_studio.domain.lora import LoraSetting

LoraRowState: TypeAlias = dict[str, object]
LoraEditorState: TypeAlias = list[LoraRowState]


@dataclass(frozen=True)
class LoraRowComponents:
    """Controls for one preallocated LoRA row."""

    container: gr.Row
    name: gr.Dropdown
    model_strength: gr.Number
    clip_strength: gr.Number
    up_button: gr.Button
    down_button: gr.Button
    remove_button: gr.Button


@dataclass(frozen=True)
class LoraEditorComponents:
    """The bounded editor and its serializable state."""

    rows: tuple[LoraRowComponents, ...]
    add_button: gr.Button
    state: gr.State
    choices: gr.State


def empty_lora_state() -> LoraEditorState:
    """Return the initial state with one empty editable row."""

    return [_new_row()]


def normalize_lora_state(state: object, max_loras: int) -> LoraEditorState:
    """Normalize untrusted UI state to at most the configured row count."""

    if max_loras <= 0:
        return []
    if not isinstance(state, Sequence) or isinstance(state, (str, bytes, bytearray)):
        return empty_lora_state()[:max_loras]
    rows: LoraEditorState = []
    for item in state[:max_loras]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "row_id": str(item.get("row_id") or uuid4().hex),
                "lora_name": _optional_string(item.get("lora_name")),
                "model_strength": _strength(item.get("model_strength"), 1.0),
                "clip_strength": _strength(item.get("clip_strength"), 1.0),
            }
        )
    return rows or empty_lora_state()[:max_loras]


def lora_settings_from_state(state: object, max_loras: int = 8) -> tuple[LoraSetting, ...]:
    """Convert serializable UI rows into the typed domain tuple."""

    rows = normalize_lora_state(state, max_loras)
    return tuple(
        LoraSetting(
            name=str(row["lora_name"]),
            model_strength=_row_strength(row, "model_strength"),
            clip_strength=_row_strength(row, "clip_strength"),
            order=order,
        )
        for order, row in enumerate(rows)
        if row["lora_name"]
    )


def add_lora_row(state: object, max_loras: int) -> LoraEditorState:
    rows = normalize_lora_state(state, max_loras)
    return rows + [_new_row()] if len(rows) < max_loras else rows


def remove_lora_row(state: object, index: int, max_loras: int) -> LoraEditorState:
    rows = normalize_lora_state(state, max_loras)
    if 0 <= index < len(rows):
        rows.pop(index)
    return rows or empty_lora_state()[:max_loras]


def move_lora_row(state: object, index: int, delta: int, max_loras: int) -> LoraEditorState:
    rows = normalize_lora_state(state, max_loras)
    target = index + delta
    if 0 <= index < len(rows) and 0 <= target < len(rows):
        rows[index], rows[target] = rows[target], rows[index]
    return rows


def update_lora_row(
    state: object,
    index: int,
    lora_name: object,
    model_strength: object,
    clip_strength: object,
    max_loras: int,
) -> LoraEditorState:
    rows = normalize_lora_state(state, max_loras)
    if 0 <= index < len(rows):
        rows[index] = {
            **rows[index],
            "lora_name": _optional_string(lora_name),
            "model_strength": _strength(model_strength, 1.0),
            "clip_strength": _strength(clip_strength, 1.0),
        }
    return rows


def build_lora_editor(max_loras: int) -> LoraEditorComponents:
    """Build fixed rows so the editor remains usable on mobile layouts."""

    rows: list[LoraRowComponents] = []
    state = gr.State(empty_lora_state()[: max(1, max_loras)])
    choices = gr.State([])
    gr.Markdown("### LoRA")
    for index in range(max(1, max_loras)):
        with gr.Row(visible=index == 0) as container:
            name = gr.Dropdown([], label=f"LoRA {index + 1}", interactive=False)
            model_strength = gr.Number(
                value=1.0,
                minimum=-2.0,
                maximum=2.0,
                step=0.1,
                label="Model strength",
            )
            clip_strength = gr.Number(
                value=1.0,
                minimum=-2.0,
                maximum=2.0,
                step=0.1,
                label="CLIP strength",
            )
            up_button = gr.Button("↑", min_width=44)
            down_button = gr.Button("↓", min_width=44)
            remove_button = gr.Button("×", min_width=44)
        rows.append(
            LoraRowComponents(
                container,
                name,
                model_strength,
                clip_strength,
                up_button,
                down_button,
                remove_button,
            )
        )
    add_button = gr.Button("Add LoRA", interactive=max_loras > 1)
    return LoraEditorComponents(tuple(rows), add_button, state, choices)


def component_outputs(editor: LoraEditorComponents) -> list[Any]:
    """Return row outputs used by state mutation handlers."""

    outputs: list[Any] = []
    for row in editor.rows:
        outputs.extend(
            [
                row.container,
                row.name,
                row.model_strength,
                row.clip_strength,
                row.up_button,
                row.down_button,
                row.remove_button,
            ]
        )
    return outputs


def render_state_updates(
    state: object,
    choices: object,
    max_loras: int,
) -> tuple[object, ...]:
    """Render normalized state into preallocated component updates."""

    rows = normalize_lora_state(state, max_loras)
    available = tuple(str(value) for value in choices) if isinstance(choices, Sequence) else ()
    updates: list[object] = [rows]
    for index in range(max(1, max_loras)):
        row = rows[index] if index < len(rows) else _new_row()
        visible = index < len(rows)
        updates.extend(
            [
                gr.Row(visible=visible),
                gr.Dropdown(
                    choices=list(available),
                    value=preserve_lora_selection(row["lora_name"], available),
                    interactive=bool(available),
                ),
                gr.Number(value=_row_strength(row, "model_strength"), visible=visible),
                gr.Number(value=_row_strength(row, "clip_strength"), visible=visible),
                gr.Button(interactive=visible and index > 0),
                gr.Button(interactive=visible and index + 1 < len(rows)),
                gr.Button(interactive=visible),
            ]
        )
    updates.append(gr.Button(interactive=len(rows) < max_loras))
    return tuple(updates)


def preserve_lora_selection(current: object, choices: Sequence[str]) -> str | None:
    value = current if isinstance(current, str) else None
    return value if value in choices else None


def _new_row() -> LoraRowState:
    return {
        "row_id": uuid4().hex,
        "lora_name": None,
        "model_strength": 1.0,
        "clip_strength": 1.0,
    }


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _strength(value: object, default: float) -> float:
    try:
        return max(-2.0, min(2.0, float(str(value))))
    except (TypeError, ValueError):
        return default


def _row_strength(row: LoraRowState, key: str) -> float:
    return _strength(row.get(key), 1.0)
