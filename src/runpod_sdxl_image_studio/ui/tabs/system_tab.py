"""Generation and system tab components with service-backed handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import gradio as gr
from pydantic import ValidationError

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.domain.generation import (
    GenerationKind,
    GenerationProgress,
    GenerationStatus,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.lora_catalog_service import (
    LoraCatalogError,
    LoraCatalogService,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    LoraEditorComponents,
    build_lora_editor,
    component_outputs,
    lora_settings_from_state,
    render_state_updates,
)
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
    checkpoint_choices: gr.State
    vae_choices: gr.State
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    upscaler: gr.Dropdown
    lora_list: gr.Markdown
    lora_editor: LoraEditorComponents
    lora_category_filter: gr.Dropdown
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
    batch_count: gr.Number
    batch_seed_strategy: gr.Radio
    batch_start_seed: gr.Number
    batch_seed_step: gr.Number
    batch_name: gr.Textbox
    batch_enqueue_button: gr.Button
    batch_message: gr.Markdown
    progress: gr.Markdown
    result_image: gr.Image
    result_details: gr.Markdown
    restored_from_generation: gr.State
    regeneration_valid: gr.State
    regeneration_requested: gr.State


def capability_refresh_outputs(generation: GenerationTabComponents) -> tuple[Any, ...]:
    """Single source of truth for capability refresh event outputs."""

    return (
        generation.checkpoint,
        generation.vae,
        generation.sampler,
        generation.scheduler,
        generation.upscaler,
        generation.checkpoint_choices,
        generation.vae_choices,
        generation.lora_list,
        generation.generate_button,
        generation.lora_editor.choices,
        generation.lora_editor.state,
        *component_outputs(generation.lora_editor),
        generation.lora_editor.add_button,
        generation.lora_category_filter,
    )


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


def build_generation_tab(max_loras: int = 8) -> GenerationTabComponents:
    """Build a mobile-friendly fixed-workflow SDXL generation form."""

    gr.Markdown("## 画像生成")
    checkpoint = gr.Dropdown([], label="checkpoint", interactive=False)
    checkpoint_choices = gr.State(None)
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
        seed_mode = gr.Radio(
            ["Random", "Fixed", "Previous seed"], value="Random", label="Seed mode"
        )
        seed = gr.Number(value=-1, precision=0, label="Seed")
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            steps = gr.Number(value=28, precision=0, label="Steps")
            cfg_scale = gr.Number(value=5.5, label="CFG")
        with gr.Row():
            sampler = gr.Dropdown([], label="sampler", interactive=False)
            scheduler = gr.Dropdown([], label="scheduler", interactive=False)
        with gr.Row():
            vae = gr.Dropdown(
                [("Checkpoint内蔵VAE", None)],
                value=None,
                label="VAE",
                interactive=True,
            )
            vae_choices = gr.State(None)
            upscaler = gr.Dropdown([], label="upscaler", interactive=False)
    lora_list = gr.Markdown("**LoRA list:** unavailable")
    lora_editor = build_lora_editor(max_loras)
    lora_category_filter = gr.Dropdown([], label="LoRAカテゴリ", interactive=False)
    generate_button = gr.Button(
        "生成をキューへ追加", variant="primary", interactive=False, size="lg"
    )
    with gr.Accordion("バッチ生成", open=False):
        batch_count = gr.Number(value=2, precision=0, label="生成枚数")
        batch_seed_strategy = gr.Radio(
            [("ランダム", "random"), ("連番", "sequential")],
            value="random",
            label="seed方式",
        )
        batch_start_seed = gr.Number(value=0, precision=0, label="開始seed")
        batch_seed_step = gr.Number(value=1, precision=0, label="seed増分")
        batch_name = gr.Textbox(value="Batch", label="バッチ名", max_length=200)
        batch_enqueue_button = gr.Button("バッチをキューへ追加", variant="primary")
        batch_message = gr.Markdown("")
    progress = gr.Markdown("")
    result_image = gr.Image(label="Generated image", type="filepath")
    result_details = gr.Markdown("")
    return GenerationTabComponents(
        checkpoint=checkpoint,
        vae=vae,
        checkpoint_choices=checkpoint_choices,
        vae_choices=vae_choices,
        sampler=sampler,
        scheduler=scheduler,
        upscaler=upscaler,
        lora_list=lora_list,
        lora_editor=lora_editor,
        lora_category_filter=lora_category_filter,
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
        batch_count=batch_count,
        batch_seed_strategy=batch_seed_strategy,
        batch_start_seed=batch_start_seed,
        batch_seed_step=batch_seed_step,
        batch_name=batch_name,
        batch_enqueue_button=batch_enqueue_button,
        batch_message=batch_message,
        progress=progress,
        result_image=result_image,
        result_details=result_details,
        restored_from_generation=gr.State(None),
        regeneration_valid=gr.State(False),
        regeneration_requested=gr.State(False),
    )


def make_check_connection_handler(
    service: ComfyUIService,
    timezone_name: str,
    generation: GenerationTabComponents,
    catalog_service: LoraCatalogService | None = None,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create an async handler that obtains status through the service."""

    async def handler(
        checkpoint: str | None,
        vae: str | None,
        sampler: str | None,
        scheduler: str | None,
        upscaler: str | None,
        lora_state: object = None,
        lora_choices: object = None,
        lora_category: str | None = None,
    ) -> tuple[object, ...]:
        status = await service.get_status()
        catalog_choices = _catalog_choices(catalog_service, status.capabilities)
        catalog_categories = _catalog_categories(catalog_service)
        updates = _capability_updates(
            status.capabilities,
            (checkpoint, vae, sampler, scheduler, upscaler),
            generation,
            lora_state,
            catalog_choices,
            lora_category,
            catalog_categories,
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
    catalog_service: LoraCatalogService | None = None,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create an async handler for capability-only refreshes."""

    async def handler(
        checkpoint: str | None,
        vae: str | None,
        sampler: str | None,
        scheduler: str | None,
        upscaler: str | None,
        lora_state: object = None,
        lora_choices: object = None,
        lora_category: str | None = None,
    ) -> tuple[object, ...]:
        refresh_result = await service.refresh_capabilities()
        if not refresh_result.is_success or refresh_result.capabilities is None:
            return (refresh_result.message,) + _preserve_updates(generation)
        updates = _capability_updates(
            refresh_result.capabilities,
            (checkpoint, vae, sampler, scheduler, upscaler),
            generation,
            lora_state,
            _catalog_choices(catalog_service, refresh_result.capabilities),
            lora_category,
            _catalog_categories(catalog_service),
        )
        return (refresh_result.message, *updates)

    return handler


def make_generate_handler(
    service: GenerationService,
    max_loras: int,
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
        vae: str | None = None,
        lora_state: object = None,
        restored_from_generation_id: str | None = None,
        regeneration_valid: bool = False,
        regeneration_requested: bool = False,
        progress: gr.Progress = _DEFAULT_PROGRESS,
    ) -> tuple[object, ...]:
        del size_preset
        try:
            loras = lora_settings_from_state(lora_state, max_loras=max_loras)
        except (TypeError, ValueError, ValidationError):
            return (
                gr.Button("Generate", interactive=True),
                "",
                None,
                "LoRAの強度または件数を確認してください。",
                False,
            )
        try:
            generation_settings = GenerationSettings(
                positive_prompt=positive_prompt or "",
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                vae_name=vae,
                loras=loras,
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
            )
        except (TypeError, ValueError, ValidationError):
            return (
                gr.Button("Generate", interactive=True),
                "",
                None,
                "入力値を確認してください。",
                False,
            )

        def on_progress(update: GenerationProgress) -> None:
            report_gradio_progress(progress, update)

        if regeneration_requested and not restored_from_generation_id:
            return (
                gr.Button("Generate", interactive=True),
                "",
                None,
                "履歴設定の復元に失敗したため、再生成を開始しませんでした。",
                False,
            )
        if regeneration_requested and not regeneration_valid:
            return (
                gr.Button("Generate", interactive=True),
                "",
                None,
                "利用できない設定があるため、再生成を開始しませんでした。",
                False,
            )
        if seed_mode == "Previous seed" and not restored_from_generation_id:
            return (
                gr.Button("Generate", interactive=True),
                "",
                None,
                "復元元Generationがありません。",
                False,
            )
        parent_id: UUID | None = None
        if restored_from_generation_id:
            try:
                parent_id = UUID(restored_from_generation_id)
            except ValueError:
                return (
                    gr.Button("Generate", interactive=True),
                    "",
                    None,
                    "復元元Generationが不正です。",
                    False,
                )
        if parent_id is None:
            result = await service.generate(generation_settings, on_progress)
        else:
            result = await service.generate(
                generation_settings,
                on_progress,
                parent_generation_id=parent_id,
                kind=GenerationKind.DERIVED,
            )
        if result.status is GenerationStatus.COMPLETED and result.stored_image is not None:
            details = (
                f"Generation ID: `{result.generation_id}`\n"
                f"Prompt ID: `{result.prompt_id}`\n"
                f"Seed: `{result.seed}`\n"
                f"VAE: `{generation_settings.vae_name or 'Checkpoint内蔵VAE'}`\n"
                + "".join(
                    f"LoRA {index + 1}: `{lora.name}` "
                    f"(model={lora.model_strength:g}, clip={lora.clip_strength:g})\n"
                    for index, lora in enumerate(generation_settings.loras)
                )
                + f"File: `{result.stored_image.path.name}`"
            )
            return (
                gr.Button("Generate", interactive=True),
                "Completed",
                str(result.stored_image.path),
                details,
                False,
            )
        return (
            gr.Button("Generate", interactive=True),
            "Failed",
            None,
            result.error_message or "画像生成に失敗しました",
            False,
        )

    return handler


def make_enqueue_handler(
    service: GenerationQueueService,
    max_loras: int,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create the non-blocking UI boundary that only persists queue work."""

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
        vae: str | None = None,
        lora_state: object = None,
        restored_from_generation_id: str | None = None,
        regeneration_valid: bool = False,
        regeneration_requested: bool = False,
    ) -> tuple[object, ...]:
        del size_preset
        try:
            loras = lora_settings_from_state(lora_state, max_loras=max_loras)
            generation_settings = GenerationSettings(
                positive_prompt=positive_prompt or "",
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                vae_name=vae,
                loras=loras,
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
            )
        except (TypeError, ValueError, ValidationError):
            return (
                gr.Button("生成をキューへ追加", interactive=True),
                "",
                None,
                "入力値を確認してください。",
                False,
            )
        if regeneration_requested and not restored_from_generation_id:
            return (
                gr.Button("生成をキューへ追加", interactive=True),
                "",
                None,
                "履歴設定の復元に失敗したため、キューへ追加しませんでした。",
                False,
            )
        if regeneration_requested and not regeneration_valid:
            return (
                gr.Button("生成をキューへ追加", interactive=True),
                "",
                None,
                "利用できない設定があるため、キューへ追加しませんでした。",
                False,
            )
        if seed_mode == "Previous seed" and not restored_from_generation_id:
            return (
                gr.Button("生成をキューへ追加", interactive=True),
                "",
                None,
                "復元元Generationがありません。",
                False,
            )
        parent_id: UUID | None = None
        if restored_from_generation_id:
            try:
                parent_id = UUID(restored_from_generation_id)
            except ValueError:
                return (
                    gr.Button("生成をキューへ追加", interactive=True),
                    "",
                    None,
                    "復元元Generationが不正です。",
                    False,
                )
        try:
            queued = service.enqueue(
                generation_settings,
                parent_generation_id=parent_id,
            )
        except GenerationQueueServiceError as exc:
            return (
                gr.Button("生成をキューへ追加", interactive=True),
                "",
                None,
                str(exc),
                False,
            )
        details = (
            f"Queued Generation ID: `{queued.item.generation.id}`\n"
            f"Queue position: `{queued.queue_position}`\n"
            f"Seed: `{queued.item.generation.settings_snapshot.seed}`"
        )
        return (
            gr.Button("生成をキューへ追加", interactive=True),
            "Queued",
            None,
            details,
            False,
        )

    return handler


def make_batch_enqueue_handler(
    service: GenerationQueueService,
    max_loras: int,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create the UI boundary for one atomic batch enqueue."""

    async def handler(
        checkpoint: str | None,
        positive_prompt: str,
        negative_prompt: str,
        width: float | int,
        height: float | int,
        seed_mode: str,
        seed: float | int,
        steps: float | int,
        cfg_scale: float | int,
        sampler: str | None,
        scheduler: str | None,
        vae: str | None,
        lora_state: object,
        count: float | int,
        strategy: str,
        start_seed: float | int,
        seed_step: float | int,
        name: str,
    ) -> tuple[object, ...]:
        try:
            settings = GenerationSettings(
                positive_prompt=positive_prompt or "",
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                vae_name=vae,
                loras=lora_settings_from_state(lora_state, max_loras=max_loras),
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
            )
            result = service.enqueue_batch(
                settings,
                count=int(count),
                seed_strategy=strategy,
                start_seed=int(start_seed) if start_seed is not None else None,
                seed_step=int(seed_step),
                name=name or "Batch",
            )
        except (TypeError, ValueError, ValidationError, GenerationQueueServiceError) as exc:
            return (
                gr.Button("バッチをキューへ追加", interactive=True),
                "",
                str(exc)
                if isinstance(exc, GenerationQueueServiceError)
                else "入力値を確認してください。",
            )
        return (
            gr.Button("バッチをキューへ追加", interactive=True),
            "Queued",
            f"Batch `{result.batch.id}` を{len(result.items)}件キューへ追加しました。",
        )

    return handler


def _legacy_disable_generate_button() -> gr.Button:
    """Disable the action before the queued generation handler starts."""

    return gr.Button(
        value="生成中...",
        interactive=False,
    )


def _legacy_disable_generate_button_with_mojibake() -> gr.Button:
    """Disable the action before the queued generation handler starts."""

    return gr.Button(
        value="生成中...",
        interactive=False,
    )


def disable_generate_button() -> gr.Button:
    """Disable the action before the queued generation handler starts."""

    return gr.Button(
        value="\u751f\u6210\u4e2d...",
        interactive=False,
    )


def disable_enqueue_button() -> gr.Button:
    """Disable the queue button while the atomic enqueue request is running."""

    return gr.Button(value="キューへ追加中...", interactive=False)


def disable_batch_enqueue_button() -> gr.Button:
    """Disable the batch button before its transaction starts."""

    return gr.Button(value="Batch投入中...", interactive=False)


def report_gradio_progress(progress: gr.Progress, update: GenerationProgress) -> None:
    """Convert domain 0-100 progress into Gradio's 0-1 or tuple format."""

    if update.value is not None and update.maximum is not None and update.maximum > 0:
        value = min(update.maximum, max(0, update.value))
        progress((value, update.maximum), desc=update.message)
    elif update.percentage is not None:
        percentage = min(100.0, max(0.0, update.percentage))
        progress(percentage / 100.0, desc=update.message)
    else:
        progress(0.0, desc=update.message)


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
    current_values: tuple[object, ...],
    generation: GenerationTabComponents,
    lora_state: object = None,
    lora_choice_options: object = None,
    lora_category: str | None = None,
    category_options: Sequence[str] = (),
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
    updates: list[object] = []
    for component, available_choices, current_value in dropdowns:
        if component is generation.vae:
            vae_choices: list[tuple[str, str | None]] = [("Checkpoint内蔵VAE", None)] + [
                (value, value) for value in available_choices
            ]
            updates.append(
                gr.Dropdown(
                    choices=vae_choices,
                    value=current_value if current_value in available_choices else None,
                    label=component.label,
                    interactive=True,
                )
            )
        else:
            updates.append(
                gr.Dropdown(
                    choices=list(available_choices),
                    value=preserve_selection(
                        current_value if isinstance(current_value, str) else None,
                        available_choices,
                    ),
                    label=component.label,
                    interactive=bool(available_choices),
                )
            )
    can_generate = bool(
        capabilities.checkpoints and capabilities.samplers and capabilities.schedulers
    )
    rendered = render_state_updates(
        lora_state,
        lora_choice_options if lora_choice_options is not None else capabilities.loras,
        len(generation.lora_editor.rows),
        clear_unavailable=True,
    )
    selected_options = (
        list(lora_choice_options)
        if isinstance(lora_choice_options, Sequence)
        and not isinstance(lora_choice_options, (str, bytes, bytearray))
        else list(capabilities.loras)
    )
    return tuple(updates) + (
        list(choices["checkpoint"]),
        list(choices["vae"]),
        lora_markdown(capabilities),
        gr.Button(interactive=can_generate),
        selected_options,
        *rendered,
        gr.Dropdown(
            choices=list(category_options),
            value=preserve_selection(lora_category, tuple(category_options)),
            label=generation.lora_category_filter.label,
            interactive=bool(category_options),
        ),
    )


def _empty_updates(generation: GenerationTabComponents) -> tuple[object, ...]:
    rendered = render_state_updates(None, [], len(generation.lora_editor.rows))
    return (
        gr.Dropdown([], label=generation.checkpoint.label, interactive=False),
        gr.Dropdown([], label=generation.vae.label, interactive=False),
        gr.Dropdown([], label=generation.sampler.label, interactive=False),
        gr.Dropdown([], label=generation.scheduler.label, interactive=False),
        gr.Dropdown([], label=generation.upscaler.label, interactive=False),
        None,
        None,
        "**LoRA list:** unavailable",
        gr.Button(interactive=False),
        None,
        *rendered,
        gr.Dropdown([], label=generation.lora_category_filter.label, interactive=False),
    )


def _preserve_updates(generation: GenerationTabComponents) -> tuple[object, ...]:
    """Leave all editable controls untouched after a refresh failure."""

    return tuple(gr.skip() for _ in capability_refresh_outputs(generation))


def _catalog_choices(
    catalog_service: LoraCatalogService | None,
    capabilities: ComfyUICapabilities | None,
) -> object:
    if capabilities is None:
        return None
    if catalog_service is None:
        return capabilities.loras
    try:
        catalog_service.sync_with_capabilities(capabilities.loras)
        return catalog_service.selector_options()
    except LoraCatalogError:
        return capabilities.loras


def _catalog_categories(catalog_service: LoraCatalogService | None) -> tuple[str, ...]:
    if catalog_service is None:
        return ()
    try:
        return catalog_service.categories()
    except Exception:  # noqa: BLE001 - capability refresh must remain usable
        return ()
