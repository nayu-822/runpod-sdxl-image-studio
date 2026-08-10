"""Generation and system tab components with service-backed handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import wraps
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
from runpod_sdxl_image_studio.domain.generation_form_state import GenerationFormStateSnapshot
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.generation_preflight_service import (
    GenerationPreflightService,
)
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.lora_catalog_service import (
    LoraCatalogError,
    LoraCatalogService,
)
from runpod_sdxl_image_studio.services.pod_lifecycle_service import PodLifecycleService
from runpod_sdxl_image_studio.services.state_sync_service import StateSyncService
from runpod_sdxl_image_studio.services.system_health_service import SystemHealthService
from runpod_sdxl_image_studio.ui.components.generation_status_card import (
    build_generation_status_card,
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
    preflight_markdown,
    preserve_selection,
    state_sync_markdown,
    status_markdown,
    system_error_history_markdown,
    system_health_markdown,
)

_DEFAULT_PROGRESS = gr.Progress()


@dataclass(frozen=True)
class SystemTabComponents:
    """Handles required by the app builder for event wiring."""

    status_markdown: gr.Markdown
    connection_button: gr.Button
    refresh_button: gr.Button
    capability_message: gr.Markdown
    health_markdown: gr.Markdown
    health_refresh_button: gr.Button
    error_history_markdown: gr.Markdown
    state_sync_markdown: gr.Markdown
    state_backup_button: gr.Button
    state_sync_message: gr.Markdown
    lifecycle_markdown: gr.Markdown
    lifecycle_refresh_button: gr.Button
    lifecycle_terminate_button: gr.Button
    lifecycle_toggle_button: gr.Button
    lifecycle_message: gr.Markdown


@dataclass(frozen=True)
class GenerationTabComponents:
    """Controls for the fixed SDXL txt2img workflow."""

    checkpoint: gr.Dropdown
    vae: gr.Dropdown
    checkpoint_choices: gr.State
    vae_choices: gr.State
    sampler_choices: gr.State
    scheduler_choices: gr.State
    upscaler_choices: gr.State
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
    restored_form_state: gr.State
    regeneration_valid: gr.State
    regeneration_requested: gr.State
    positive_clear_button: gr.Button
    negative_clear_button: gr.Button
    recent_refresh: gr.Button
    recent_checkpoints: gr.Dropdown
    recent_checkpoint_apply: gr.Button
    recent_vaes: gr.Dropdown
    recent_vae_apply: gr.Button
    recent_loras: gr.Dropdown
    recent_lora_add: gr.Button
    recent_generation_presets: gr.Dropdown
    recent_prompt_presets: gr.Dropdown
    recent_lora_presets: gr.Dropdown
    recent_preset_apply: gr.Button
    recent_message: gr.Markdown
    status_card: gr.Markdown
    active_generation_id: gr.State
    status_poll_timer: gr.Timer
    result_seed: gr.Textbox
    result_favorite: gr.Checkbox
    result_regenerate_button: gr.Button
    result_edit_button: gr.Button
    result_upscale_button: gr.Button
    result_message: gr.Markdown


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
        generation.sampler_choices,
        generation.scheduler_choices,
        generation.upscaler_choices,
        generation.lora_list,
        generation.generate_button,
        generation.lora_editor.choices,
        generation.lora_editor.state,
        *component_outputs(generation.lora_editor),
        generation.lora_editor.add_button,
        generation.lora_category_filter,
    )


def build_system_tab(
    comfyui_url: str,
    initial_markdown: str,
    initial_health_markdown: str = "### System Health\nNot checked",
    initial_error_history_markdown: str = "### Recent errors\nNo recent operational errors.",
    initial_state_sync_markdown: str = "### State backup status\nNot checked",
    initial_lifecycle_markdown: str = "## Pod Lifecycle\nAuto-Terminate unavailable",
) -> SystemTabComponents:
    """Build the system tab without making a network request."""

    gr.Markdown("## システム")
    gr.Markdown(f"**接続先 URL:** `{comfyui_url}`")
    status = gr.Markdown(initial_markdown, elem_id="comfyui-status")
    with gr.Row(elem_classes=["system-actions"]):
        connection_button = gr.Button("接続確認", variant="primary", min_width=140)
        refresh_button = gr.Button("モデル一覧を更新", min_width=180)
    capability_message = gr.Markdown("")
    with gr.Row(elem_classes=["system-health-layout"]):
        with (
            gr.Column(elem_classes=["system-health-column"]),
            gr.Accordion("System Health", open=True, elem_classes=["system-health-section"]),
        ):
            health_markdown = gr.Markdown(initial_health_markdown, elem_id="system-health")
            health_refresh_button = gr.Button(
                "Refresh system status",
                elem_classes=["mobile-tap-button"],
            )
        with (
            gr.Column(elem_classes=["system-error-column"]),
            gr.Accordion("Recent errors", open=False, elem_classes=["system-error-section"]),
        ):
            gr.Markdown(
                "Detailed investigation uses server-side Generation ID, Job ID, or error_code."
            )
            error_history_markdown = gr.Markdown(initial_error_history_markdown)
    with gr.Accordion("State backup", open=True, elem_classes=["system-state-sync-section"]):
        state_sync_markdown = gr.Markdown(initial_state_sync_markdown)
        state_backup_button = gr.Button(
            "今すぐ状態をバックアップ",
            elem_classes=["mobile-tap-button"],
        )
        state_sync_message = gr.Markdown("")
    with gr.Accordion("Pod Lifecycle", open=True, elem_classes=["system-lifecycle-section"]):
        lifecycle_markdown = gr.Markdown(initial_lifecycle_markdown)
        with gr.Row(elem_classes=["system-actions"]):
            lifecycle_refresh_button = gr.Button(
                "Terminate安全判定を更新", elem_classes=["mobile-tap-button"]
            )
            lifecycle_terminate_button = gr.Button(
                "安全ならPodをTerminate", variant="primary", elem_classes=["mobile-tap-button"]
            )
            lifecycle_toggle_button = gr.Button(
                "Auto-Terminate状態", elem_classes=["mobile-tap-button"]
            )
        lifecycle_message = gr.Markdown("")
    return SystemTabComponents(
        status,
        connection_button,
        refresh_button,
        capability_message,
        health_markdown,
        health_refresh_button,
        error_history_markdown,
        state_sync_markdown,
        state_backup_button,
        state_sync_message,
        lifecycle_markdown,
        lifecycle_refresh_button,
        lifecycle_terminate_button,
        lifecycle_toggle_button,
        lifecycle_message,
    )


def build_generation_tab(max_loras: int = 8) -> GenerationTabComponents:
    """Build a mobile-friendly fixed-workflow SDXL generation form."""

    gr.Markdown("## 画像生成")
    checkpoint_choices = gr.State(None)
    vae_choices = gr.State(None)
    sampler_choices = gr.State(None)
    scheduler_choices = gr.State(None)
    upscaler_choices = gr.State(None)
    with gr.Row(elem_classes=["generation-layout"]):
        with gr.Column(elem_classes=["generation-primary"]):
            with gr.Accordion("最近使った項目", open=True, elem_classes=["recent-settings"]):
                recent_refresh = gr.Button(
                    "生成の最近項目を更新", elem_classes=["mobile-tap-button"]
                )
                recent_checkpoints = gr.Dropdown([], label="最近のcheckpoint")
                recent_checkpoint_apply = gr.Button(
                    "checkpointを反映", elem_classes=["mobile-tap-button"]
                )
                recent_vaes = gr.Dropdown([], label="最近のVAE")
                recent_vae_apply = gr.Button("VAEを反映", elem_classes=["mobile-tap-button"])
                recent_loras = gr.Dropdown([], label="最近のLoRA")
                recent_lora_add = gr.Button("最近のLoRAを追加", elem_classes=["mobile-tap-button"])
                recent_generation_presets = gr.Dropdown([], label="最近のPreset")
                recent_preset_apply = gr.Button(
                    "最近のPresetを適用", elem_classes=["mobile-tap-button"]
                )
                recent_prompt_presets = gr.Dropdown([], visible=False)
                recent_lora_presets = gr.Dropdown([], visible=False)
                recent_message = gr.Markdown("")

            checkpoint = gr.Dropdown([], label="checkpoint", interactive=False)
            with gr.Column(elem_classes=["prompt-editor"]):
                positive_prompt = gr.Textbox(
                    label="Positive prompt",
                    lines=6,
                    max_lines=16,
                    elem_classes=["prompt-editor"],
                )
                positive_clear_button = gr.Button(
                    "Positive promptをクリア", elem_classes=["mobile-tap-button"]
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    lines=4,
                    max_lines=12,
                    elem_classes=["prompt-editor", "negative"],
                )
                negative_clear_button = gr.Button(
                    "Negative promptをクリア", elem_classes=["mobile-tap-button"]
                )
            lora_list = gr.Markdown("**LoRA一覧:** 未取得")
            lora_editor = build_lora_editor(max_loras)
            lora_category_filter = gr.Dropdown([], label="LoRAカテゴリ", interactive=False)
            size_preset = gr.Dropdown(
                ["1024 × 1024", "832 × 1216", "1216 × 832", "896 × 1152", "1152 × 896", "Custom"],
                value="1024 × 1024",
                label="サイズプリセット",
            )
            with gr.Row(elem_classes=["size-dimensions"]):
                width = gr.Number(value=1024, precision=0, label="幅")
                height = gr.Number(value=1024, precision=0, label="高さ")
            with gr.Row(elem_classes=["seed-controls"]):
                seed_mode = gr.Radio(
                    ["Random", "Fixed", "Previous seed"], value="Random", label="Seed方式"
                )
                seed = gr.Number(value=-1, precision=0, label="Seed")
            with gr.Column(elem_classes=["generation-sticky-action"]):
                generate_button = gr.Button(
                    "生成をキューへ追加",
                    variant="primary",
                    interactive=False,
                    size="lg",
                    elem_classes=["mobile-tap-button"],
                )

        with gr.Column(elem_classes=["generation-preview"]):
            status_components = build_generation_status_card()
            progress = gr.Markdown("")
            result_image = gr.Image(label="生成画像", type="filepath")
            result_details = gr.Markdown("")
            with gr.Row(elem_classes=["result-actions"]):
                result_regenerate_button = gr.Button(
                    "同条件で再生成", elem_classes=["mobile-tap-button"]
                )
                result_edit_button = gr.Button("設定を編集", elem_classes=["mobile-tap-button"])
                result_upscale_button = gr.Button(
                    "アップスケール", elem_classes=["mobile-tap-button"]
                )
                result_favorite = gr.Checkbox(label="お気に入り", value=False)
            result_seed = gr.Textbox(
                label="実使用Seed（コピー）",
                interactive=False,
                show_copy_button=True,
            )
            result_message = gr.Markdown("")

    with gr.Accordion("高度な設定", open=False, elem_classes=["generation-advanced"]):
        with gr.Row(elem_classes=["size-dimensions"]):
            steps = gr.Number(value=28, precision=0, label="Steps")
            cfg_scale = gr.Number(value=5.5, label="CFG")
        with gr.Row(elem_classes=["size-dimensions"]):
            sampler = gr.Dropdown([], label="Sampler", interactive=False)
            scheduler = gr.Dropdown([], label="Scheduler", interactive=False)
        with gr.Row(elem_classes=["size-dimensions"]):
            vae = gr.Dropdown(
                [("Checkpoint内蔵VAE", None)],
                value=None,
                label="VAE",
                interactive=True,
            )
            upscaler = gr.Dropdown([], label="アップスケーラー", interactive=False)

    with gr.Accordion("バッチ生成", open=False, elem_classes=["generation-batch"]):
        batch_count = gr.Number(value=2, precision=0, label="生成枚数")
        batch_seed_strategy = gr.Radio(
            [("ランダム", "random"), ("連番", "sequential")],
            value="random",
            label="Seed方式",
        )
        batch_start_seed = gr.Number(value=0, precision=0, label="開始Seed")
        batch_seed_step = gr.Number(value=1, precision=0, label="Seed増分")
        batch_name = gr.Textbox(value="Batch", label="バッチ名", max_length=200)
        batch_enqueue_button = gr.Button(
            "バッチをキューへ追加", variant="primary", elem_classes=["mobile-tap-button"]
        )
        batch_message = gr.Markdown("")
    return GenerationTabComponents(
        checkpoint=checkpoint,
        vae=vae,
        checkpoint_choices=checkpoint_choices,
        vae_choices=vae_choices,
        sampler_choices=sampler_choices,
        scheduler_choices=scheduler_choices,
        upscaler_choices=upscaler_choices,
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
        restored_form_state=gr.State(None),
        regeneration_valid=gr.State(False),
        regeneration_requested=gr.State(False),
        positive_clear_button=positive_clear_button,
        negative_clear_button=negative_clear_button,
        recent_refresh=recent_refresh,
        recent_checkpoints=recent_checkpoints,
        recent_checkpoint_apply=recent_checkpoint_apply,
        recent_vaes=recent_vaes,
        recent_vae_apply=recent_vae_apply,
        recent_loras=recent_loras,
        recent_lora_add=recent_lora_add,
        recent_generation_presets=recent_generation_presets,
        recent_preset_apply=recent_preset_apply,
        recent_prompt_presets=recent_prompt_presets,
        recent_lora_presets=recent_lora_presets,
        recent_message=recent_message,
        status_card=status_components.card,
        active_generation_id=status_components.active_generation_id,
        status_poll_timer=status_components.poll_timer,
        result_seed=result_seed,
        result_favorite=result_favorite,
        result_regenerate_button=result_regenerate_button,
        result_edit_button=result_edit_button,
        result_upscale_button=result_upscale_button,
        result_message=result_message,
    )


def make_system_health_handler(
    service: SystemHealthService,
    timezone_name: str,
) -> Callable[..., Awaitable[tuple[str, str]]]:
    """Create the manual/demo.load health snapshot handler."""

    async def handler() -> tuple[str, str]:
        view = await service.get_health()
        return (
            system_health_markdown(view, timezone_name),
            system_error_history_markdown(view.recent_errors, timezone_name),
        )

    return handler


def make_state_backup_handler(
    service: StateSyncService,
    timezone_name: str,
) -> Callable[[], Awaitable[tuple[str, str]]]:
    """Create the explicit state backup action for the System tab."""

    async def handler() -> tuple[str, str]:
        view = await service.backup()
        return state_sync_markdown(view, timezone_name), view.last_message

    return handler


def make_pod_lifecycle_readiness_handler(
    service: PodLifecycleService,
) -> Callable[[], Awaitable[tuple[str, str]]]:
    """Create the check-only System tab action."""

    async def handler() -> tuple[str, str]:
        readiness = await service.check_readiness()
        return _pod_lifecycle_markdown(service, readiness), (
            "SAFE TO TERMINATE" if readiness.is_safe else "NOT READY"
        )

    return handler


def make_pod_lifecycle_terminate_handler(
    service: PodLifecycleService,
) -> Callable[[], Awaitable[tuple[str, str]]]:
    """Create the guarded manual self-termination action."""

    async def handler() -> tuple[str, str]:
        readiness = await service.check_readiness()
        if not readiness.is_safe:
            return _pod_lifecycle_markdown(service, readiness), "NOT READY; Pod was not terminated"
        try:
            await service.request_terminate(readiness=readiness)
        except Exception as exc:  # noqa: BLE001 - no adapter details at the UI boundary
            return _pod_lifecycle_markdown(service, readiness), str(exc)
        return _pod_lifecycle_markdown(service, readiness), "Termination request accepted"

    return handler


def make_pod_lifecycle_toggle_handler(
    service: PodLifecycleService,
) -> Callable[[], tuple[str, str]]:
    """Toggle the current session flag without changing RunPod Template config."""

    def handler() -> tuple[str, str]:
        session = service.session or service.initialize_session()
        if session is None:
            return "## Pod Lifecycle\nAuto Terminate unavailable", "RunPod Identity unavailable"
        updated = service.set_auto_terminate_enabled(not session.auto_terminate_enabled)
        assert updated is not None
        enabled_label = "ENABLED" if updated.auto_terminate_enabled else "PAUSED"
        return (
            f"## Pod Lifecycle\nAuto Terminate: `{enabled_label}`\n"
            f"Session: `{updated.status.value}`",
            "Auto-Terminate state updated for this session",
        )

    return handler


def _pod_lifecycle_markdown(service: PodLifecycleService, readiness: object) -> str:
    session = service.session
    is_safe = bool(getattr(readiness, "is_safe", False))
    reasons = tuple(getattr(readiness, "block_reasons", ()))
    state = session.status.value if session is not None else "unavailable"
    enabled = "ENABLED" if session is not None and session.auto_terminate_enabled else "DISABLED"
    lines = [
        "## Pod Lifecycle",
        f"Auto Terminate: `{enabled}`",
        f"Session: `{state}`",
        f"Terminate readiness: `{'SAFE TO TERMINATE' if is_safe else 'NOT READY'}`",
    ]
    readiness_labels = (
        ("Generation", "generation_ready"),
        ("ComfyUI Queue", "comfyui_ready"),
        ("Model Transfer", "model_transfer_ready"),
        ("Drive Image/Metadata", "drive_sync_ready"),
        ("Drive Manifest", "manifest_ready"),
        ("State Backup", "state_backup_ready"),
        ("RunPod Identity", "runpod_identity_ready"),
    )
    for label, attribute in readiness_labels:
        value = bool(getattr(readiness, attribute, False))
        lines.append(f"- {label}: `{'ready' if value else 'blocked'}`")
    if reasons:
        lines.append("**Block reasons:**")
        lines.extend(f"- `{reason}`" for reason in reasons)
    return "\n".join(lines)


def make_check_connection_handler(
    service: ComfyUIService,
    timezone_name: str,
    generation: GenerationTabComponents,
    catalog_service: LoraCatalogService | None = None,
    capabilities_callback: Callable[[ComfyUICapabilities | None], None] | None = None,
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
        if capabilities_callback is not None:
            capabilities_callback(status.capabilities)
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
    capabilities_callback: Callable[[ComfyUICapabilities | None], None] | None = None,
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
            if capabilities_callback is not None:
                capabilities_callback(None)
            return (refresh_result.message,) + _preserve_updates(generation)
        if capabilities_callback is not None:
            capabilities_callback(refresh_result.capabilities)
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
    preflight_service: GenerationPreflightService | None = None,
    form_state_saver: Callable[[GenerationFormStateSnapshot], object] | None = None,
) -> Callable[..., Awaitable[tuple[object, object, object, object, object, object]]]:
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
        upscaler: str | None = None,
    ) -> tuple[object, object, object, object, object, object]:
        del size_preset

        def failure(message: str) -> tuple[object, object, object, object, object, object]:
            return (
                gr.Button("生成をキューへ追加", interactive=True),
                "",
                None,
                message,
                False,
                gr.skip(),
            )

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
            return failure("入力値を確認してください。")
        if regeneration_requested and not restored_from_generation_id:
            return failure("履歴設定の復元に失敗したため、キューへ追加しませんでした。")
        if regeneration_requested and not regeneration_valid:
            return failure("利用できない設定があるため、キューへ追加しませんでした。")
        if seed_mode == "Previous seed" and not restored_from_generation_id:
            return failure("復元元Generationがありません。")
        parent_id: UUID | None = None
        if restored_from_generation_id:
            try:
                parent_id = UUID(restored_from_generation_id)
            except ValueError:
                return failure("復元元Generationが不正です。")
        if regeneration_requested and parent_id is not None:
            try:
                parent_item = service.get_job_detail(parent_id)
            except GenerationQueueServiceError:
                return failure("再生成元Generationを確認できないため、キューへ追加しませんでした。")
            if parent_item is None:
                return failure("再生成元Generationが見つからないため、キューへ追加しませんでした。")
            if parent_item.generation.status is not GenerationStatus.COMPLETED:
                return failure("完了済みGenerationだけを再生成できます。")
        preflight_warning = ""
        if preflight_service is not None:
            try:
                preflight = await preflight_service.check(generation_settings)
            except Exception:  # noqa: BLE001 - hide adapter details at the UI boundary
                return failure("Generation preflight could not be completed; nothing was queued")
            if not preflight.is_ready:
                return failure(preflight_markdown(preflight))
            if preflight.warnings:
                preflight_warning = preflight_markdown(preflight)
        try:
            queued = service.enqueue(
                generation_settings,
                parent_generation_id=parent_id,
            )
        except GenerationQueueServiceError as exc:
            return failure(str(exc))
        details = (
            f"Queued Generation ID: `{queued.item.generation.id}`\n"
            f"Queue position: `{queued.queue_position}`\n"
            f"Seed: `{queued.item.generation.settings_snapshot.seed}`"
            + (f"\n\n{preflight_warning}" if preflight_warning else "")
        )
        if form_state_saver is not None:
            try:
                form_state_saver(
                    GenerationFormStateSnapshot.from_ui(
                        positive_prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        seed_mode=seed_mode,
                        seed=-1 if seed_mode == "Random" else int(seed),
                        width=int(width),
                        height=int(height),
                        steps=int(steps),
                        cfg_scale=float(cfg_scale),
                        sampler_name=sampler or "",
                        scheduler_name=scheduler or "",
                        checkpoint_name=checkpoint or "",
                        vae_name=vae,
                        upscaler_name=upscaler,
                        loras=loras,
                    )
                )
            except Exception:  # noqa: BLE001 - enqueue success must remain durable
                details += "\n\nLast-used form state could not be saved"
        return (
            gr.Button("生成をキューへ追加", interactive=True),
            "Queued",
            None,
            details,
            False,
            str(queued.item.generation.id),
        )

    return handler


def make_batch_enqueue_handler(
    service: GenerationQueueService,
    max_loras: int,
    preflight_service: GenerationPreflightService | None = None,
    form_state_saver: Callable[[GenerationFormStateSnapshot], object] | None = None,
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
        upscaler: str | None = None,
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
            preflight_message = ""
            if preflight_service is not None:
                try:
                    preflight = await preflight_service.check(settings)
                except Exception:  # noqa: BLE001 - restore the action on preflight failure
                    return (
                        gr.Button("バッチをキューへ追加", interactive=True),
                        "",
                        "Generation preflight could not be completed; nothing was queued",
                    )
                if not preflight.is_ready:
                    return (
                        gr.Button("バッチをキューへ追加", interactive=True),
                        "",
                        preflight_markdown(preflight),
                    )
                if preflight.warnings:
                    preflight_message = preflight_markdown(preflight)
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
        if form_state_saver is not None:
            try:
                form_state_saver(
                    GenerationFormStateSnapshot.from_ui(
                        positive_prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        seed_mode=seed_mode,
                        seed=-1 if seed_mode == "Random" else int(seed),
                        width=int(width),
                        height=int(height),
                        steps=int(steps),
                        cfg_scale=float(cfg_scale),
                        sampler_name=sampler or "",
                        scheduler_name=scheduler or "",
                        checkpoint_name=checkpoint or "",
                        vae_name=vae,
                        upscaler_name=upscaler,
                        loras=lora_settings_from_state(lora_state, max_loras=max_loras),
                    )
                )
            except Exception:  # noqa: BLE001 - queue success must remain durable
                preflight_message += "\n\nLast-used form state could not be saved"
        return (
            gr.Button("バッチをキューへ追加", interactive=True),
            "Queued",
            (
                f"Batch `{result.batch.id}` を{len(result.items)}件キューへ追加しました。"
                + (f"\n\n{preflight_message}" if preflight_message else "")
            ),
        )

    return _wrap_batch_handler(handler)


def _wrap_batch_handler(
    handler: Callable[..., Awaitable[tuple[object, ...]]],
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Normalize every Batch outcome to the restored Japanese action label."""

    @wraps(handler)
    async def wrapped(*args: object, **kwargs: object) -> tuple[object, ...]:
        try:
            result = await handler(*args, **kwargs)
        except Exception:  # noqa: BLE001 - UI boundary must restore the action
            return (
                gr.Button(
                    value="\u30d0\u30c3\u30c1\u3092\u30ad\u30e5\u30fc\u3078\u8ffd\u52a0",
                    interactive=True,
                ),
                "",
                "Batch enqueue failed; nothing was queued",
            )
        if not result:
            return result
        return (
            gr.Button(
                value="\u30d0\u30c3\u30c1\u3092\u30ad\u30e5\u30fc\u3078\u8ffd\u52a0",
                interactive=True,
            ),
            *result[1:],
        )

    return wrapped


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
        list(choices["sampler"]),
        list(choices["scheduler"]),
        list(choices["upscaler"]),
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
        None,
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
