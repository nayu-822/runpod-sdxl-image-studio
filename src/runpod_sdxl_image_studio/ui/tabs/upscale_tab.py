"""Gradio selection-only UI for Phase 5 upscale enqueueing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.services.upscale_enqueue_service import (
    UpscaleEnqueueError,
    UpscaleEnqueueService,
)


@dataclass(frozen=True)
class UpscaleTabComponents:
    parent_generation_id: Any
    latest_button: Any
    source_preview: Any
    method: Any
    sizing_mode: Any
    scale_factor: Any
    target_width: Any
    target_height: Any
    upscaler_name: Any
    denoise: Any
    plan: Any
    enqueue_button: Any
    status: Any
    result: Any
    comparison: Any


def build_upscale_tab(upscaler_choices: tuple[str, ...] = ()) -> UpscaleTabComponents:
    with gr.Row():
        parent_id = gr.Textbox(label="親Generation ID", placeholder="completed generation UUID")
        source_preview = gr.Image(label="親画像", interactive=False, type="filepath")
    method = gr.Radio(
        [
            ("画像アップスケール", UpscaleMethod.IMAGE.value),
            ("Latentアップスケール", UpscaleMethod.LATENT.value),
        ],
        value=UpscaleMethod.IMAGE.value,
        label="方式",
    )
    sizing = gr.Radio(
        [("倍率", UpscaleSizingMode.FACTOR.value), ("寸法", UpscaleSizingMode.DIMENSIONS.value)],
        value=UpscaleSizingMode.FACTOR.value,
        label="出力サイズ",
    )
    with gr.Row():
        factor = gr.Number(label="倍率", value=2.0, minimum=1.01, maximum=16.0)
        width = gr.Number(label="幅", value=1024, minimum=64, precision=0)
        height = gr.Number(label="高さ", value=1024, minimum=64, precision=0)
    upscaler = gr.Dropdown(list(upscaler_choices), label="Upscaler", allow_custom_value=False)
    denoise = gr.Slider(0, 1, value=0.35, step=0.01, label="Denoise（Latentのみ）")
    plan = gr.Markdown("出力サイズと負荷見積もりは親画像確認後に表示されます。")
    enqueue_button = gr.Button("アップスケールをキューへ追加", variant="primary")
    status = gr.Markdown()
    result = gr.Image(label="結果", interactive=False)
    comparison = gr.Gallery(label="親画像と結果の比較", columns=2, rows=1)
    return UpscaleTabComponents(
        parent_id,
        gr.Button("最新の完了画像を選択"),
        source_preview,
        method,
        sizing,
        factor,
        width,
        height,
        upscaler,
        denoise,
        plan,
        enqueue_button,
        status,
        result,
        comparison,
    )


def make_latest_parent_handler(
    service: UpscaleEnqueueService,
) -> Callable[[], tuple[str, str]]:
    def handler() -> tuple[str, str]:
        generation_id = service.latest_completed_generation_id()
        if generation_id is None:
            return "", "完了済みの一次画像がありません。"
        return str(generation_id), f"親画像を選択しました: `{generation_id}`"

    return handler


def make_upscale_enqueue_handler(
    service: UpscaleEnqueueService,
) -> Callable[..., tuple[Any, str]]:
    def handler(
        parent_generation_id: str,
        method: str,
        sizing_mode: str,
        scale_factor: float | None,
        target_width: float | None,
        target_height: float | None,
        upscaler_name: str | None,
        denoise: float | None,
    ) -> tuple[Any, str]:
        try:
            settings = _settings_from_inputs(
                method,
                sizing_mode,
                scale_factor,
                target_width,
                target_height,
                upscaler_name,
                denoise,
            )
            item = service.enqueue(UUID(parent_generation_id.strip()), settings)
            return gr.Button(
                interactive=True
            ), f"キューへ追加しました（順序 {item.entry.sequence}）。"
        except (ValueError, UpscaleEnqueueError) as exc:
            return gr.Button(interactive=True), f"アップスケールを追加できませんでした: {exc}"

    return handler


def make_upscale_plan_handler(
    service: UpscaleEnqueueService,
) -> Callable[..., str]:
    def handler(
        parent_generation_id: str,
        method: str,
        sizing_mode: str,
        scale_factor: float | None,
        target_width: float | None,
        target_height: float | None,
        upscaler_name: str | None,
        denoise: float | None,
    ) -> str:
        try:
            settings = _settings_from_inputs(
                method,
                sizing_mode,
                scale_factor,
                target_width,
                target_height,
                upscaler_name,
                denoise,
            )
            plan = service.plan(UUID(parent_generation_id.strip()), settings)
            return (
                f"予定サイズ: **{plan.target_width} × {plan.target_height}**  "
                f"（親 {plan.source_width} × {plan.source_height}、"
                f"負荷: `{plan.load_level.value}`）"
            )
        except (ValueError, UpscaleEnqueueError) as exc:
            return f"サイズ計画を確認できません: {exc}"

    return handler


def _settings_from_inputs(
    method: str,
    sizing_mode: str,
    scale_factor: float | None,
    target_width: float | None,
    target_height: float | None,
    upscaler_name: str | None,
    denoise: float | None,
) -> UpscaleSettings:
    return UpscaleSettings(
        method=UpscaleMethod(method),
        sizing_mode=UpscaleSizingMode(sizing_mode),
        scale_factor=scale_factor if sizing_mode == UpscaleSizingMode.FACTOR.value else None,
        target_width=int(target_width) if target_width is not None else None,
        target_height=int(target_height) if target_height is not None else None,
        upscaler_name=upscaler_name or None,
        denoise=denoise if method == UpscaleMethod.LATENT.value else None,
    )


__all__ = [
    "UpscaleTabComponents",
    "build_upscale_tab",
    "make_latest_parent_handler",
    "make_upscale_enqueue_handler",
    "make_upscale_plan_handler",
]
