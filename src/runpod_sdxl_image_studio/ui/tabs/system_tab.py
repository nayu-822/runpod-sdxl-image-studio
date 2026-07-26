"""Generation and system tab components with service-backed handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import gradio as gr
from pydantic import ValidationError

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.domain.generation import GenerationProgress, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.ui.view_models import (
    capability_choices,
    lora_markdown,
    preserve_selection,
    status_markdown,
)

_DEFAULT_PROGRESS = gr.Progress()


@dataclass(frozen=True)
class SystemTabComponents:
    """Handles required by the app builder for event wiring."""

    status_markdown: gr.Markdown
    connection_button: gr.Button
    refresh_button: gr.Button
    capability_message: gr.Markdown


@dataclass(frozen=True)
class GenerationTabComponents:
    """Controls for the fixed SDXL txt2img workflow."""

    checkpoint: gr.Dropdown
    vae: gr.Dropdown
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    upscaler: gr.Dropdown
    lora_list: gr.Markdown
    positive_prompt: gr.Textbox
    negative_prompt: gr.Textbox
    size_preset: gr.Dropdown
    width: gr.Number
    height: gr.Number
    seed_mode: gr.Radio
    seed: gr.Number
    steps: gr.Number
    cfg_scale: gr.Number
    generate_button: gr.Button
    progress: gr.Markdown
    result_image: gr.Image
    result_details: gr.Markdown


def build_system_tab(comfyui_url: str, initial_markdown: str) -> SystemTabComponents:
    """Build the system tab without making a network request."""

    gr.Markdown("## システム")
    gr.Markdown(f"**接続先 URL:** `{comfyui_url}`")
    status = gr.Markdown(initial_markdown, elem_id="comfyui-status")
    with gr.Row():
        connection_button = gr.Button("接続確認", variant="primary", min_width=140)
        refresh_button = gr.Button("モデル一覧を更新", min_width=180)
    capability_message = gr.Markdown("")
    return SystemTabComponents(status, connection_button, refresh_button, capability_message)


def build_generation_tab() -> GenerationTabComponents:
    """Build a mobile-friendly fixed-workflow SDXL generation form."""

    gr.Markdown("## 画像生成")
    checkpoint = gr.Dropdown([], label="checkpoint", interactive=False)
    positive_prompt = gr.Textbox(label="Positive prompt", lines=5, max_lines=12)
    negative_prompt = gr.Textbox(label="Negative prompt", lines=3, max_lines=8)
    with gr.Row():
        size_preset = gr.Dropdown(
            ["1024 × 1024", "832 × 1216", "1216 × 832", "896 × 1152", "1152 × 896", "Custom"],
            value="1024 × 1024",
            label="Size",
        )
        width = gr.Number(value=1024, precision=0, label="Width")
        height = gr.Number(value=1024, precision=0, label="Height")
    with gr.Row():
        seed_mode = gr.Radio(["Random", "Fixed"], value="Random", label="Seed mode")
        seed = gr.Number(value=-1, precision=0, label="Seed")
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            steps = gr.Number(value=28, precision=0, label="Steps")
            cfg_scale = gr.Number(value=5.5, label="CFG")
        with gr.Row():
            sampler = gr.Dropdown([], label="sampler", interactive=False)
            scheduler = gr.Dropdown([], label="scheduler", interactive=False)
        with gr.Row():
            vae = gr.Dropdown([], label="VAE", interactive=False)
            upscaler = gr.Dropdown([], label="upscaler", interactive=False)
    lora_list = gr.Markdown("**LoRA list:** unavailable")
    generate_button = gr.Button("Generate", variant="primary", interactive=False, size="lg")
    progress = gr.Markdown("")
    result_image = gr.Image(label="Generated image", type="filepath")
    result_details = gr.Markdown("")
    return GenerationTabComponents(
        checkpoint=checkpoint,
        vae=vae,
        sampler=sampler,
        scheduler=scheduler,
        upscaler=upscaler,
        lora_list=lora_list,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        size_preset=size_preset,
        width=width,
        height=height,
        seed_mode=seed_mode,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        generate_button=generate_button,
        progress=progress,
        result_image=result_image,
        result_details=result_details,
    )


def make_check_connection_handler(
    service: ComfyUIService,
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
            status_markdown(status, timezone_name),
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
        refresh_result = await service.refresh_capabilities()
        if not refresh_result.is_success or refresh_result.capabilities is None:
            return (refresh_result.message,) + _empty_updates(generation)
        updates = _capability_updates(
            refresh_result.capabilities,
            (checkpoint, vae, sampler, scheduler, upscaler),
            generation,
        )
        return (refresh_result.message, *updates)

    return handler


def make_generate_handler(
    service: GenerationService,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create the UI boundary that constructs typed GenerationSettings."""

    async def handler(
        checkpoint: str | None,
        positive_prompt: str,
        negative_prompt: str,
        size_preset: str,
        width: float | int,
        height: float | int,
        seed_mode: str,
        seed: float | int,
        steps: float | int,
        cfg_scale: float | int,
        sampler: str | None,
        scheduler: str | None,
        progress: gr.Progress = _DEFAULT_PROGRESS,
    ) -> tuple[object, ...]:
        del size_preset
        try:
            generation_settings = GenerationSettings(
                positive_prompt=positive_prompt or "",
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
            )
        except (TypeError, ValueError, ValidationError):
            return gr.Button("Generate", interactive=True), "", None, "入力値を確認してください。"

        def on_progress(update: GenerationProgress) -> None:
            progress(update.percentage or 0.0, desc=update.message)

        result = await service.generate(generation_settings, on_progress)
        if result.status is GenerationStatus.COMPLETED and result.stored_image is not None:
            details = (
                f"Generation ID: `{result.generation_id}`\n"
                f"Prompt ID: `{result.prompt_id}`\n"
                f"Seed: `{result.seed}`\n"
                f"File: `{result.stored_image.path.name}`"
            )
            return (
                gr.Button("Generate", interactive=True),
                "Completed",
                str(result.stored_image.path),
                details,
            )
        return (
            gr.Button("Generate", interactive=True),
            "Failed",
            None,
            result.error_message or "画像生成に失敗しました",
        )

    return handler


def size_preset_values(preset: str) -> tuple[int, int]:
    """Return dimensions for one of the fixed mobile presets."""

    return {
        "1024 × 1024": (1024, 1024),
        "832 × 1216": (832, 1216),
        "1216 × 832": (1216, 832),
        "896 × 1152": (896, 1152),
        "1152 × 896": (1152, 896),
    }.get(preset, (1024, 1024))


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
            interactive=bool(available_choices),
        )
        for component, available_choices, current_value in dropdowns
    )
    can_generate = bool(
        capabilities.checkpoints and capabilities.samplers and capabilities.schedulers
    )
    return updates + (lora_markdown(capabilities), gr.Button(interactive=can_generate))


def _empty_updates(generation: GenerationTabComponents) -> tuple[object, ...]:
    return (
        gr.Dropdown([], label=generation.checkpoint.label, interactive=False),
        gr.Dropdown([], label=generation.vae.label, interactive=False),
        gr.Dropdown([], label=generation.sampler.label, interactive=False),
        gr.Dropdown([], label=generation.scheduler.label, interactive=False),
        gr.Dropdown([], label=generation.upscaler.label, interactive=False),
        "**LoRA list:** unavailable",
        gr.Button(interactive=False),
    )
