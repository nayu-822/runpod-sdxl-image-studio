"""System tab components and service-backed event handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import gradio as gr

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import ComfyUIError
from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.ui.view_models import (
    capability_choices,
    lora_markdown,
    preserve_selection,
    status_markdown,
)


@dataclass(frozen=True)
class SystemTabComponents:
    """Handles required by the app builder for event wiring."""

    status_markdown: gr.Markdown
    connection_button: gr.Button
    refresh_button: gr.Button
    capability_message: gr.Markdown


@dataclass(frozen=True)
class GenerationTabComponents:
    """Read-only controls showing capabilities available for later generation."""

    checkpoint: gr.Dropdown
    vae: gr.Dropdown
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    upscaler: gr.Dropdown
    lora_list: gr.Markdown


def build_system_tab(comfyui_url: str, initial_markdown: str) -> SystemTabComponents:
    """Build the system tab without making a network request."""

    gr.Markdown("## システム")
    gr.Markdown(f"**接続先URL:** `{comfyui_url}`")
    status = gr.Markdown(initial_markdown, elem_id="comfyui-status")
    with gr.Row():
        connection_button = gr.Button("接続確認", variant="primary", min_width=140)
        refresh_button = gr.Button("モデル一覧を再読込", min_width=180)
    capability_message = gr.Markdown("")
    return SystemTabComponents(
        status_markdown=status,
        connection_button=connection_button,
        refresh_button=refresh_button,
        capability_message=capability_message,
    )


def build_generation_tab() -> GenerationTabComponents:
    """Build the Phase 1A generation tab with empty, non-editable choices."""

    gr.Markdown("## 生成")
    gr.Markdown("画像生成は Phase 1B で実装予定です。現在は ComfyUI の能力情報のみ確認できます。")
    with gr.Row():
        checkpoint = gr.Dropdown([], label="checkpoint", interactive=False)
        vae = gr.Dropdown([], label="VAE", interactive=False)
    with gr.Row():
        sampler = gr.Dropdown([], label="sampler", interactive=False)
        scheduler = gr.Dropdown([], label="scheduler", interactive=False)
    upscaler = gr.Dropdown([], label="upscaler", interactive=False)
    lora_list = gr.Markdown("**LoRA一覧:** 未取得")
    return GenerationTabComponents(
        checkpoint=checkpoint,
        vae=vae,
        sampler=sampler,
        scheduler=scheduler,
        upscaler=upscaler,
        lora_list=lora_list,
    )


def make_check_connection_handler(
    service: ComfyUIService,
    comfyui_url: str,
    timezone_name: str,
    generation: GenerationTabComponents,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create an async handler that obtains status through the service."""

    async def handler(
        checkpoint: str | None,
        vae: str | None,
        sampler: str | None,
        scheduler: str | None,
        upscaler: str | None,
    ) -> tuple[object, ...]:
        status = await service.get_status()
        updates = _capability_updates(
            status.capabilities,
            (checkpoint, vae, sampler, scheduler, upscaler),
            generation,
        )
        return (
            status_markdown(status, comfyui_url, timezone_name),
            "能力情報を更新しました" if status.capabilities is not None else status.message,
            *updates,
        )

    return handler


def make_refresh_handler(
    service: ComfyUIService,
    generation: GenerationTabComponents,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create an async handler for capability-only refreshes."""

    async def handler(
        checkpoint: str | None,
        vae: str | None,
        sampler: str | None,
        scheduler: str | None,
        upscaler: str | None,
    ) -> tuple[object, ...]:
        try:
            capabilities = await service.refresh_capabilities()
        except ComfyUIError:
            return ("能力情報を取得できませんでした",) + _empty_updates(generation)

        updates = _capability_updates(
            capabilities,
            (checkpoint, vae, sampler, scheduler, upscaler),
            generation,
        )
        return ("モデル一覧を更新しました", *updates)

    return handler


def _capability_updates(
    capabilities: ComfyUICapabilities | None,
    current_values: tuple[str | None, ...],
    generation: GenerationTabComponents,
) -> tuple[object, ...]:
    if capabilities is None:
        return _empty_updates(generation)
    choices = capability_choices(capabilities)
    dropdowns = (
        (generation.checkpoint, choices["checkpoint"], current_values[0]),
        (generation.vae, choices["vae"], current_values[1]),
        (generation.sampler, choices["sampler"], current_values[2]),
        (generation.scheduler, choices["scheduler"], current_values[3]),
        (generation.upscaler, choices["upscaler"], current_values[4]),
    )
    updates = tuple(
        gr.Dropdown(
            choices=list(available_choices),
            value=preserve_selection(current_value, available_choices),
            label=component.label,
            interactive=False,
        )
        for component, available_choices, current_value in dropdowns
    )
    return updates + (lora_markdown(capabilities),)


def _empty_updates(generation: GenerationTabComponents) -> tuple[object, ...]:
    return (
        gr.Dropdown([], label=generation.checkpoint.label, interactive=False),
        gr.Dropdown([], label=generation.vae.label, interactive=False),
        gr.Dropdown([], label=generation.sampler.label, interactive=False),
        gr.Dropdown([], label=generation.scheduler.label, interactive=False),
        gr.Dropdown([], label=generation.upscaler.label, interactive=False),
        "**LoRA一覧:** 未取得",
    )
