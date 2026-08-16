"""Generation and system tab components with service-backed handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from typing import Any
from uuid import UUID

import gradio as gr
from pydantic import ValidationError

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.domain.detailer import (
    DEFAULT_DETAILER_REGISTRY,
    DetailerKind,
    DetailerSettings,
)
from runpod_sdxl_image_studio.domain.generation import (
    GenerationKind,
    GenerationProgress,
    GenerationStatus,
)
from runpod_sdxl_image_studio.domain.generation_form_state import GenerationFormStateSnapshot
from runpod_sdxl_image_studio.domain.generation_settings import (
    CURRENT_WORKFLOW_TEMPLATE_VERSION,
    GenerationSettings,
)
from runpod_sdxl_image_studio.domain.interactive_run import InteractiveRunView
from runpod_sdxl_image_studio.jobs.startup_model_restore import (
    StartupModelRestoreRuntime,
    StartupRestoreState,
)
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.generation_custom_size_service import (
    GenerationCustomSizeError,
    GenerationCustomSizeService,
)
from runpod_sdxl_image_studio.services.generation_preflight_service import (
    GenerationPreflightService,
)
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.interactive_generation_service import (
    InteractiveGenerationError,
    InteractiveGenerationService,
)
from runpod_sdxl_image_studio.services.lora_catalog_service import (
    LoraCatalogError,
    LoraCatalogService,
)
from runpod_sdxl_image_studio.services.lora_trigger_service import (
    LoraTriggerResolutionError,
    resolve_effective_positive_prompt,
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
    normalize_lora_state,
    render_state_updates,
)
from runpod_sdxl_image_studio.ui.view_models import (
    capability_choices,
    lora_markdown,
    preflight_markdown,
    preserve_selection,
    selected_lora_summary_markdown,
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
    face_detector_model: gr.Dropdown
    lora_list: gr.Markdown
    lora_summary: gr.Markdown
    lora_editor: LoraEditorComponents
    lora_category_filter: gr.Dropdown
    positive_prompt: gr.Textbox
    negative_prompt: gr.Textbox
    positive_paste_button: gr.Button
    positive_copy_button: gr.Button
    positive_clipboard_message: gr.Markdown
    negative_paste_button: gr.Button
    negative_copy_button: gr.Button
    negative_clipboard_message: gr.Markdown
    size_preset: gr.Dropdown
    width: gr.Number
    height: gr.Number
    custom_size_delete_button: gr.Button
    custom_size_message: gr.Markdown
    seed_mode: gr.Radio
    seed: gr.Number
    steps: gr.Number
    cfg_scale: gr.Number
    clip_skip: gr.Number
    hires_fix: gr.Checkbox
    hires_scale: gr.Number
    hires_resize_method: gr.Dropdown
    hires_steps: gr.Number
    hires_cfg_scale: gr.Number
    hires_sampler: gr.Dropdown
    hires_scheduler: gr.Dropdown
    hires_denoise: gr.Number
    final_upscale: gr.Checkbox
    final_upscale_message: gr.Markdown
    face_detailer_enabled: gr.Checkbox
    face_positive_prompt: gr.Textbox
    face_negative_prompt: gr.Textbox
    face_denoise: gr.Number
    face_steps: gr.Number
    face_cfg_scale: gr.Number
    face_sampler: gr.Dropdown
    face_scheduler: gr.Dropdown
    face_guide_size: gr.Number
    face_max_size: gr.Number
    face_bbox_threshold: gr.Number
    face_bbox_dilation: gr.Number
    face_bbox_crop_factor: gr.Number
    face_feather: gr.Number
    face_detailer_details: gr.Accordion
    workflow_template_id: gr.State
    workflow_template_version: gr.State
    generate_button: gr.Button
    interactive_gallery_generation_id: gr.State
    interactive_selected_generation_id: gr.State
    interactive_selected_image_index: gr.State
    interactive_restore_button: gr.Button
    batch_count: gr.Number
    batch_size: gr.Number
    batch_seed_strategy: gr.Radio
    batch_start_seed: gr.Number
    batch_seed_step: gr.Number
    batch_name: gr.Textbox
    batch_enqueue_button: gr.Button
    batch_message: gr.Markdown
    interactive_start_button: gr.Button
    interactive_cancel_button: gr.Button
    interactive_status: gr.Markdown
    interactive_run_id: gr.State
    interactive_result_gallery: gr.Gallery
    interactive_client_local_date: gr.Textbox
    interactive_poll_timer: gr.Timer
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
    status_surface: gr.Group
    status_card: gr.Markdown
    active_generation_id: gr.State
    status_poll_timer: gr.Timer
    startup_restore_timer: gr.Timer
    startup_restore_applied: gr.State
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
        generation.hires_sampler,
        generation.hires_scheduler,
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
        generation.face_detector_model,
    )


def startup_restore_form_outputs(generation: GenerationTabComponents) -> tuple[Any, ...]:
    """Single source of truth for the form fields restored at startup."""

    return (
        generation.positive_prompt,
        generation.negative_prompt,
        generation.seed_mode,
        generation.seed,
        generation.width,
        generation.height,
        generation.steps,
        generation.cfg_scale,
        generation.clip_skip,
        generation.hires_fix,
        generation.hires_scale,
        generation.hires_resize_method,
        generation.hires_steps,
        generation.hires_cfg_scale,
        generation.hires_sampler,
        generation.hires_scheduler,
        generation.hires_denoise,
        generation.final_upscale,
        generation.face_detailer_enabled,
        generation.face_positive_prompt,
        generation.face_negative_prompt,
        generation.face_denoise,
        generation.face_steps,
        generation.face_cfg_scale,
        generation.face_sampler,
        generation.face_scheduler,
        generation.face_guide_size,
        generation.face_max_size,
        generation.face_bbox_threshold,
        generation.face_bbox_dilation,
        generation.face_bbox_crop_factor,
        generation.face_feather,
        generation.face_detailer_details,
    )


def startup_restore_form_values(snapshot: GenerationFormStateSnapshot) -> tuple[object, ...]:
    """Return startup-restored form values in the same order as their components."""

    return (
        snapshot.positive_prompt,
        snapshot.negative_prompt,
        snapshot.ui_seed_mode,
        snapshot.seed,
        snapshot.width,
        snapshot.height,
        snapshot.steps,
        snapshot.cfg_scale,
        snapshot.clip_skip,
        snapshot.hires_fix,
        snapshot.hires_scale,
        snapshot.hires_resize_method,
        snapshot.hires_steps,
        snapshot.hires_cfg_scale,
        snapshot.hires_sampler_name,
        snapshot.hires_scheduler_name,
        snapshot.hires_denoise,
        snapshot.final_upscale,
        snapshot.face_detailer_enabled,
        snapshot.face_detailer.positive_prompt if snapshot.face_detailer else "",
        snapshot.face_detailer.negative_prompt if snapshot.face_detailer else "",
        snapshot.face_detailer.denoise if snapshot.face_detailer else 0.22,
        snapshot.face_detailer.steps if snapshot.face_detailer else 20,
        snapshot.face_detailer.cfg_scale if snapshot.face_detailer else 5.0,
        snapshot.face_detailer.sampler_name if snapshot.face_detailer else "euler_ancestral",
        snapshot.face_detailer.scheduler_name if snapshot.face_detailer else "normal",
        snapshot.face_detailer.guide_size if snapshot.face_detailer else 768,
        snapshot.face_detailer.max_size if snapshot.face_detailer else 1024,
        snapshot.face_detailer.bbox_threshold if snapshot.face_detailer else 0.5,
        snapshot.face_detailer.bbox_dilation if snapshot.face_detailer else 10,
        snapshot.face_detailer.bbox_crop_factor if snapshot.face_detailer else 2.0,
        snapshot.face_detailer.feather if snapshot.face_detailer else 5,
        gr.Accordion(visible=bool(snapshot.face_detailer_enabled)),
    )


def startup_restore_outputs(
    generation: GenerationTabComponents, capability_message: gr.Markdown
) -> tuple[Any, ...]:
    """Single source of truth for the startup restore event outputs."""

    return (
        generation.startup_restore_timer,
        capability_message,
        *capability_refresh_outputs(generation),
        *startup_restore_form_outputs(generation),
        generation.restored_form_state,
        generation.startup_restore_applied,
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


def build_generation_tab(
    max_loras: int = 8,
    custom_size_choices: Sequence[tuple[str, str]] = (),
) -> GenerationTabComponents:
    """Build the image-first, progressive-disclosure SDXL generation form."""

    gr.Markdown("## 画像を作る")
    checkpoint_choices = gr.State(None)
    vae_choices = gr.State(None)
    sampler_choices = gr.State(None)
    scheduler_choices = gr.State(None)
    upscaler_choices = gr.State(None)
    with gr.Row(elem_classes=["generation-layout"]):
        with gr.Column(elem_classes=["generation-primary"]):
            checkpoint = gr.Dropdown([], label="モデル", interactive=False)
            lora_summary = gr.Markdown(
                selected_lora_summary_markdown([], max_loras),
                elem_classes=["lora-summary"],
            )
            with gr.Column(elem_classes=["prompt-editor"]):
                gr.Markdown("### Prompt")
                positive_prompt = gr.Textbox(
                    label="Positive",
                    lines=6,
                    max_lines=16,
                    elem_classes=["prompt-editor"],
                )
                with gr.Row(elem_classes=["prompt-actions"]):
                    positive_paste_button = gr.Button(
                        "貼り付け", elem_classes=["mobile-tap-button"]
                    )
                    positive_copy_button = gr.Button("コピー", elem_classes=["mobile-tap-button"])
                    positive_clear_button = gr.Button("クリア", elem_classes=["mobile-tap-button"])
                positive_clipboard_message = gr.Markdown("")
                negative_prompt = gr.Textbox(
                    label="Negative",
                    lines=4,
                    max_lines=12,
                    elem_classes=["prompt-editor", "negative"],
                )
                with gr.Row(elem_classes=["prompt-actions"]):
                    negative_paste_button = gr.Button(
                        "貼り付け", elem_classes=["mobile-tap-button"]
                    )
                    negative_copy_button = gr.Button("コピー", elem_classes=["mobile-tap-button"])
                    negative_clear_button = gr.Button("クリア", elem_classes=["mobile-tap-button"])
                negative_clipboard_message = gr.Markdown("")
            lora_list = gr.Markdown("**LoRA一覧:** 未取得", visible=False)
            with gr.Accordion("LoRAを編集", open=False, elem_classes=["lora-editor-section"]):
                lora_editor = build_lora_editor(max_loras)
                lora_category_filter = gr.Dropdown([], label="LoRAカテゴリ", interactive=False)
            size_preset = gr.Dropdown(
                [
                    "1024 × 1024",
                    "832 × 1216",
                    "1216 × 832",
                    "896 × 1152",
                    "1152 × 896",
                    *custom_size_choices,
                    "Custom",
                ],
                value="1024 × 1024",
                label="サイズ",
            )
            with gr.Row(elem_classes=["size-dimensions"]):
                width = gr.Number(value=1024, precision=0, label="幅", visible=False)
                height = gr.Number(value=1024, precision=0, label="高さ", visible=False)
            custom_size_delete_button = gr.Button(
                "削除",
                interactive=False,
                visible=False,
                elem_classes=["mobile-tap-button"],
            )
            custom_size_message = gr.Markdown("")
            with gr.Row(elem_classes=["compact-controls"]):
                batch_size = gr.Number(
                    value=1,
                    precision=0,
                    minimum=1,
                    maximum=4,
                    label="1回の枚数",
                )
                batch_count = gr.Number(value=2, precision=0, label="回数")
            with gr.Row(elem_classes=["compact-controls"]):
                hires_fix = gr.Checkbox(value=False, label="Hires.fix")
                face_detailer_enabled = gr.Checkbox(value=False, label="顔を補正")
                final_upscale = gr.Checkbox(value=False, label="4x upscale")
            upscaler = gr.Dropdown(
                [],
                label="アップスケーラー",
                interactive=False,
                visible=False,
            )
            final_upscale_message = gr.Markdown("", elem_classes=["validation-message"])
            with gr.Column(elem_classes=["generation-sticky-action"]):
                generate_button = gr.Button(
                    "生成",
                    variant="primary",
                    interactive=False,
                    size="lg",
                    elem_classes=["mobile-tap-button"],
                )
                interactive_cancel_button = gr.Button(
                    "キャンセル",
                    interactive=False,
                    visible=False,
                    elem_classes=["mobile-tap-button"],
                )

        with gr.Column(elem_classes=["generation-preview"]):
            status_components = build_generation_status_card()
            startup_restore_timer = gr.Timer(value=1.0, active=True)
            startup_restore_applied = gr.State(False)
            interactive_status = gr.Markdown("", visible=False)
            progress = gr.Markdown("")
            batch_message = gr.Markdown("")
            interactive_result_gallery = gr.Gallery(
                label="生成結果",
                columns=2,
                rows=2,
                object_fit="contain",
                visible=False,
                elem_classes=["interactive-result-gallery"],
            )
            interactive_restore_button = gr.Button(
                "設定を読み込む",
                visible=False,
                elem_classes=["mobile-tap-button"],
            )
            # Keep the legacy result components as hidden state targets for old handlers.
            result_image = gr.Image(label="生成画像", type="filepath", visible=False)
            result_details = gr.Markdown("", visible=False)
            with gr.Row(elem_classes=["result-actions"], visible=False):
                result_regenerate_button = gr.Button(
                    "同条件で再生成", visible=False, elem_classes=["mobile-tap-button"]
                )
                result_edit_button = gr.Button(
                    "設定を編集", visible=False, elem_classes=["mobile-tap-button"]
                )
                result_upscale_button = gr.Button(
                    "アップスケール", visible=False, elem_classes=["mobile-tap-button"]
                )
                result_favorite = gr.Checkbox(label="お気に入り", value=False, visible=False)
            result_seed = gr.Textbox(
                label="実使用Seed（コピー）",
                interactive=False,
                show_copy_button=True,
                visible=False,
            )
            result_message = gr.Markdown("", visible=False)

    with gr.Accordion("高度な設定", open=False, elem_classes=["generation-advanced"]):
        with gr.Row(elem_classes=["seed-controls"]):
            seed_mode = gr.Radio(
                ["Random", "Fixed", "Previous seed"], value="Random", label="Seed方式"
            )
            seed = gr.Number(value=-1, precision=0, label="Seed")
        with gr.Row(elem_classes=["size-dimensions"]):
            steps = gr.Number(value=28, precision=0, label="Steps")
            cfg_scale = gr.Number(value=5.5, label="CFG")
            clip_skip = gr.Number(value=1, precision=0, minimum=1, maximum=12, label="CLIP skip")
        with gr.Row(elem_classes=["size-dimensions"]):
            hires_scale = gr.Number(value=1.5, minimum=1.0, maximum=4.0, label="Hires倍率")
            hires_resize_method = gr.Dropdown(
                ["lanczos", "nearest-exact", "bilinear", "bicubic"],
                value="lanczos",
                label="Hires resize",
            )
            hires_denoise = gr.Number(value=0.4, minimum=0.0, maximum=1.0, label="Hires denoise")
        with gr.Row(elem_classes=["size-dimensions"]):
            hires_steps = gr.Number(
                value=20, precision=0, minimum=1, maximum=150, label="Hires Steps"
            )
            hires_cfg_scale = gr.Number(value=5.5, label="Hires CFG")
            hires_sampler = gr.Dropdown(
                ["euler"], value="euler", label="Hires sampler", interactive=False
            )
            hires_scheduler = gr.Dropdown(
                ["normal"], value="normal", label="Hires scheduler", interactive=False
            )
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
        with gr.Accordion(
            "顔補正の詳細",
            open=False,
            visible=False,
            elem_classes=["face-detailer-section"],
        ) as face_detailer_details:
            face_detector_model = gr.Dropdown([], label="検出モデル", interactive=False)
            with gr.Row(elem_classes=["size-dimensions"]):
                face_positive_prompt = gr.Textbox(
                    value=DEFAULT_DETAILER_REGISTRY.default_settings(
                        DetailerKind.FACE
                    ).positive_prompt,
                    label="Face Positive",
                    lines=3,
                )
                face_negative_prompt = gr.Textbox(
                    value=DEFAULT_DETAILER_REGISTRY.default_settings(
                        DetailerKind.FACE
                    ).negative_prompt,
                    label="Face Negative",
                    lines=3,
                )
            with gr.Row(elem_classes=["size-dimensions"]):
                face_denoise = gr.Number(value=0.22, minimum=0.0, maximum=1.0, label="denoise")
                face_steps = gr.Number(value=20, precision=0, minimum=1, maximum=150, label="steps")
                face_cfg_scale = gr.Number(value=5.0, label="Face CFG")
                face_sampler = gr.Dropdown(
                    ["euler_ancestral"], value="euler_ancestral", label="sampler"
                )
                face_scheduler = gr.Dropdown(["normal"], value="normal", label="scheduler")
            with gr.Accordion("検出設定", open=False):
                with gr.Row(elem_classes=["size-dimensions"]):
                    face_guide_size = gr.Number(value=768, precision=0, label="guide size")
                    face_max_size = gr.Number(value=1024, precision=0, label="max size")
                    face_bbox_threshold = gr.Number(
                        value=0.5, minimum=0.0, maximum=1.0, label="bbox threshold"
                    )
                    face_bbox_dilation = gr.Number(
                        value=10, precision=0, minimum=0, label="bbox dilation"
                    )
                with gr.Row(elem_classes=["size-dimensions"]):
                    face_bbox_crop_factor = gr.Number(
                        value=2.0, minimum=0.1, label="bbox crop factor"
                    )
                    face_feather = gr.Number(value=5, precision=0, minimum=0, label="feather")

        with gr.Accordion("最近使った設定", open=False, elem_classes=["recent-settings"]):
            recent_refresh = gr.Button("最近の設定を更新", elem_classes=["mobile-tap-button"])
            recent_checkpoints = gr.Dropdown([], label="最近のモデル")
            recent_checkpoint_apply = gr.Button("モデルを反映", elem_classes=["mobile-tap-button"])
            recent_vaes = gr.Dropdown([], label="最近のVAE")
            recent_vae_apply = gr.Button("VAEを反映", elem_classes=["mobile-tap-button"])
            recent_loras = gr.Dropdown([], label="最近のLoRA")
            recent_lora_add = gr.Button("LoRAを追加", elem_classes=["mobile-tap-button"])
            recent_generation_presets = gr.Dropdown([], label="最近のPreset")
            recent_preset_apply = gr.Button("Presetを適用", elem_classes=["mobile-tap-button"])
            recent_prompt_presets = gr.Dropdown([], visible=False)
            recent_lora_presets = gr.Dropdown([], visible=False)
            recent_message = gr.Markdown("")

    with gr.Accordion("バッチ生成", open=False, visible=False, elem_classes=["generation-batch"]):
        batch_seed_strategy = gr.Radio(
            [("ランダム", "random"), ("連番", "sequential")],
            value="random",
            label="Seed方式",
        )
        batch_start_seed = gr.Number(value=0, precision=0, label="開始Seed")
        batch_seed_step = gr.Number(value=1, precision=0, label="Seed増分")
        batch_name = gr.Textbox(value="Batch", label="バッチ名", max_length=200)
        batch_enqueue_button = gr.Button(
            "バッチをキューへ追加",
            variant="primary",
            visible=False,
            elem_classes=["mobile-tap-button"],
        )
        interactive_client_local_date = gr.Textbox(value="", visible=False)
        interactive_start_button = gr.Button(
            "対話的生成を開始", variant="primary", visible=False, elem_classes=["mobile-tap-button"]
        )
        interactive_run_id = gr.State(None)
        interactive_gallery_generation_id = gr.State(None)
        interactive_selected_generation_id = gr.State(None)
        interactive_selected_image_index = gr.State(None)
        interactive_poll_timer = gr.Timer(value=3.0, active=True)
    workflow_template_id = gr.State("sdxl_txt2img")
    workflow_template_version = gr.State(CURRENT_WORKFLOW_TEMPLATE_VERSION)
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
        face_detector_model=face_detector_model,
        lora_list=lora_list,
        lora_summary=lora_summary,
        lora_editor=lora_editor,
        lora_category_filter=lora_category_filter,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        positive_paste_button=positive_paste_button,
        positive_copy_button=positive_copy_button,
        positive_clipboard_message=positive_clipboard_message,
        negative_paste_button=negative_paste_button,
        negative_copy_button=negative_copy_button,
        negative_clipboard_message=negative_clipboard_message,
        size_preset=size_preset,
        width=width,
        height=height,
        custom_size_delete_button=custom_size_delete_button,
        custom_size_message=custom_size_message,
        seed_mode=seed_mode,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        clip_skip=clip_skip,
        hires_fix=hires_fix,
        hires_scale=hires_scale,
        hires_resize_method=hires_resize_method,
        hires_steps=hires_steps,
        hires_cfg_scale=hires_cfg_scale,
        hires_sampler=hires_sampler,
        hires_scheduler=hires_scheduler,
        hires_denoise=hires_denoise,
        final_upscale=final_upscale,
        final_upscale_message=final_upscale_message,
        face_detailer_enabled=face_detailer_enabled,
        face_positive_prompt=face_positive_prompt,
        face_negative_prompt=face_negative_prompt,
        face_denoise=face_denoise,
        face_steps=face_steps,
        face_cfg_scale=face_cfg_scale,
        face_sampler=face_sampler,
        face_scheduler=face_scheduler,
        face_guide_size=face_guide_size,
        face_max_size=face_max_size,
        face_bbox_threshold=face_bbox_threshold,
        face_bbox_dilation=face_bbox_dilation,
        face_bbox_crop_factor=face_bbox_crop_factor,
        face_feather=face_feather,
        face_detailer_details=face_detailer_details,
        workflow_template_id=workflow_template_id,
        workflow_template_version=workflow_template_version,
        generate_button=generate_button,
        interactive_gallery_generation_id=interactive_gallery_generation_id,
        interactive_selected_generation_id=interactive_selected_generation_id,
        interactive_selected_image_index=interactive_selected_image_index,
        interactive_restore_button=interactive_restore_button,
        batch_count=batch_count,
        batch_size=batch_size,
        batch_seed_strategy=batch_seed_strategy,
        batch_start_seed=batch_start_seed,
        batch_seed_step=batch_seed_step,
        batch_name=batch_name,
        batch_enqueue_button=batch_enqueue_button,
        batch_message=batch_message,
        interactive_start_button=interactive_start_button,
        interactive_cancel_button=interactive_cancel_button,
        interactive_status=interactive_status,
        interactive_run_id=interactive_run_id,
        interactive_result_gallery=interactive_result_gallery,
        interactive_client_local_date=interactive_client_local_date,
        interactive_poll_timer=interactive_poll_timer,
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
        status_surface=status_components.surface,
        status_card=status_components.card,
        active_generation_id=status_components.active_generation_id,
        status_poll_timer=status_components.poll_timer,
        startup_restore_timer=startup_restore_timer,
        startup_restore_applied=startup_restore_applied,
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
        try:
            manual_handler = getattr(service, "manual_drain_backup_and_terminate", None)
            if callable(manual_handler):
                readiness = await manual_handler()
            else:
                # Compatibility for small test doubles and older integrations.
                readiness = await service.drain_backup_and_terminate(require_armed=False)
        except Exception as exc:  # noqa: BLE001 - no adapter details at the UI boundary
            try:
                readiness = await service.check_readiness()
            except Exception:  # noqa: BLE001 - keep the UI boundary safe
                readiness = None
            if readiness is None:
                return _pod_lifecycle_markdown(service, object()), str(exc)
            return _pod_lifecycle_markdown(service, readiness), str(exc)
        if not readiness.is_safe:
            return _pod_lifecycle_markdown(
                service, readiness
            ), "NOT READY; termination was not requested"
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
        final_upscale: bool = False,
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
            final_upscale=bool(final_upscale),
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
        final_upscale: bool = False,
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
            final_upscale=bool(final_upscale),
        )
        return (refresh_result.message, *updates)

    return handler


def _choice_values(choices: Sequence[object]) -> tuple[object, ...]:
    """Return dropdown values from string or ``(label, value)`` choices."""

    return tuple(
        choice[1] if isinstance(choice, tuple) and len(choice) == 2 else choice
        for choice in choices
    )


def make_startup_restore_handler(
    runtime: StartupModelRestoreRuntime,
    service: ComfyUIService,
    generation: GenerationTabComponents,
    catalog_service: LoraCatalogService | None = None,
    capabilities_callback: Callable[[ComfyUICapabilities | None], None] | None = None,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Apply the desired form only after background model preparation is terminal."""

    capability_count = len(capability_refresh_outputs(generation))
    form_tail_count = len(startup_restore_form_outputs(generation)) + 1

    async def handler(
        checkpoint: str | None,
        vae: str | None,
        sampler: str | None,
        scheduler: str | None,
        upscaler: str | None,
        lora_state: object = None,
        lora_choices: object = None,
        lora_category: str | None = None,
        final_upscale: bool = False,
        startup_restore_applied: bool = False,
    ) -> tuple[object, ...]:
        del checkpoint, vae, sampler, scheduler, upscaler
        status = runtime.status()
        if startup_restore_applied or not status.is_terminal or status.snapshot is None:
            return (
                gr.Timer(active=not startup_restore_applied and not status.is_terminal),
                status.message,
                *(gr.skip() for _ in range(capability_count)),
                *(gr.skip() for _ in range(form_tail_count)),
                True if status.is_terminal and status.snapshot is None else startup_restore_applied,
            )
        if status.state is StartupRestoreState.FAILED:
            return (
                gr.Timer(active=False),
                status.message,
                *(gr.skip() for _ in range(capability_count)),
                *(gr.skip() for _ in range(form_tail_count)),
                startup_restore_applied,
            )

        capabilities = (
            status.capabilities if isinstance(status.capabilities, ComfyUICapabilities) else None
        )
        if capabilities is None:
            refresh_result = await service.refresh_capabilities()
            if not refresh_result.is_success or refresh_result.capabilities is None:
                return (
                    gr.Timer(active=True),
                    "前回設定の反映待ち: ComfyUI能力情報を取得できません。",
                    *(gr.skip() for _ in range(capability_count)),
                    *(gr.skip() for _ in range(form_tail_count)),
                    startup_restore_applied,
                )
            capabilities = refresh_result.capabilities
        if capabilities_callback is not None:
            capabilities_callback(capabilities)
        snapshot = status.snapshot
        desired_lora_state = [
            {
                "row_id": f"restored-{index}",
                "lora_name": item.name,
                "model_strength": item.model_strength,
                "clip_strength": item.clip_strength,
                "auto_add_trigger_words": item.name in snapshot.auto_trigger_lora_names,
            }
            for index, item in enumerate(snapshot.loras)
        ]
        updates = _capability_updates(
            capabilities,
            (
                snapshot.checkpoint_name,
                snapshot.vae_name,
                snapshot.sampler_name,
                snapshot.scheduler_name,
                snapshot.upscaler_name,
                snapshot.face_detailer.detector_model if snapshot.face_detailer else None,
            ),
            generation,
            desired_lora_state,
            _catalog_choices(catalog_service, capabilities),
            lora_category,
            _catalog_categories(catalog_service),
            preserve_unavailable=True,
            final_upscale=bool(snapshot.final_upscale),
        )
        message = status.message
        if status.missing:
            message += "\n不足model: " + ", ".join(status.missing)
        return (
            gr.Timer(active=False),
            message,
            *updates,
            *startup_restore_form_values(snapshot),
            snapshot.model_dump(mode="json"),
            True,
        )

    return handler


def _face_detailers_from_ui(
    enabled: bool,
    detector_model: str | None,
    positive_prompt: str | None,
    negative_prompt: str | None,
    denoise: float | int,
    steps: float | int,
    cfg_scale: float | int,
    sampler: str | None,
    scheduler: str | None,
    guide_size: float | int,
    max_size: float | int,
    bbox_threshold: float | int,
    bbox_dilation: float | int,
    bbox_crop_factor: float | int,
    feather: float | int,
) -> tuple[DetailerSettings, ...]:
    if not enabled:
        return ()
    defaults = DEFAULT_DETAILER_REGISTRY.default_settings(DetailerKind.FACE)
    detailer_values = defaults.model_dump()
    detailer_values.update(
        enabled=True,
        detector_model=detector_model or defaults.detector_model,
        positive_prompt=defaults.positive_prompt if positive_prompt is None else positive_prompt,
        negative_prompt=defaults.negative_prompt if negative_prompt is None else negative_prompt,
        denoise=float(denoise),
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        sampler_name=sampler or defaults.sampler_name,
        scheduler_name=scheduler or defaults.scheduler_name,
        guide_size=int(guide_size),
        max_size=int(max_size),
        bbox_threshold=float(bbox_threshold),
        bbox_dilation=int(bbox_dilation),
        bbox_crop_factor=float(bbox_crop_factor),
        feather=int(feather),
    )
    detailer = DetailerSettings.model_validate(detailer_values)
    return (detailer,)


def make_generate_handler(
    service: GenerationService,
    max_loras: int,
    lora_catalog_service: LoraCatalogService | None = None,
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
        clip_skip: float | int = 1,
        hires_fix: bool = False,
        hires_scale: float | int = 1.5,
        hires_resize_method: str = "lanczos",
        hires_steps: float | int = 20,
        hires_cfg_scale: float | int = 5.5,
        hires_sampler: str | None = "euler",
        hires_scheduler: str | None = "normal",
        hires_denoise: float | int = 0.4,
        final_upscale: bool = False,
        upscaler: str | None = None,
        progress: gr.Progress = _DEFAULT_PROGRESS,
        face_detailer_enabled: bool = False,
        face_detector_model: str | None = None,
        face_positive_prompt: str | None = None,
        face_negative_prompt: str | None = None,
        face_denoise: float | int = 0.22,
        face_steps: float | int = 20,
        face_cfg_scale: float | int = 5.0,
        face_sampler: str | None = "euler_ancestral",
        face_scheduler: str | None = "normal",
        face_guide_size: float | int = 768,
        face_max_size: float | int = 1024,
        face_bbox_threshold: float | int = 0.5,
        face_bbox_dilation: float | int = 10,
        face_bbox_crop_factor: float | int = 2.0,
        face_feather: float | int = 5,
    ) -> tuple[object, ...]:
        del size_preset
        try:
            loras = lora_settings_from_state(lora_state, max_loras=max_loras)
            detailers = _face_detailers_from_ui(
                face_detailer_enabled,
                face_detector_model,
                face_positive_prompt,
                face_negative_prompt,
                face_denoise,
                face_steps,
                face_cfg_scale,
                face_sampler,
                face_scheduler,
                face_guide_size,
                face_max_size,
                face_bbox_threshold,
                face_bbox_dilation,
                face_bbox_crop_factor,
                face_feather,
            )
            effective_positive_prompt = resolve_effective_positive_prompt(
                positive_prompt or "", loras, lora_catalog_service
            )
        except LoraTriggerResolutionError as exc:
            return (
                gr.Button("Generate", interactive=True),
                "",
                None,
                str(exc),
                False,
            )
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
                positive_prompt=effective_positive_prompt,
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                vae_name=vae,
                loras=loras,
                detailers=detailers,
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
                clip_skip=int(clip_skip),
                hires_fix=bool(hires_fix),
                hires_scale=float(hires_scale),
                hires_resize_method=hires_resize_method,
                hires_steps=int(hires_steps),
                hires_cfg_scale=float(hires_cfg_scale),
                hires_sampler_name=hires_sampler or "euler",
                hires_scheduler_name=hires_scheduler or "normal",
                hires_denoise=float(hires_denoise),
                final_upscale=bool(final_upscale),
                final_upscale_model=upscaler if final_upscale else None,
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
    interactive_service: InteractiveGenerationService | None = None,
    lora_catalog_service: LoraCatalogService | None = None,
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
        clip_skip: float | int = 1,
        hires_fix: bool = False,
        hires_scale: float | int = 1.5,
        hires_resize_method: str = "lanczos",
        hires_steps: float | int = 20,
        hires_cfg_scale: float | int = 5.5,
        hires_sampler: str | None = "euler",
        hires_scheduler: str | None = "normal",
        hires_denoise: float | int = 0.4,
        final_upscale: bool = False,
        face_detailer_enabled: bool = False,
        face_detector_model: str | None = None,
        face_positive_prompt: str | None = None,
        face_negative_prompt: str | None = None,
        face_denoise: float | int = 0.22,
        face_steps: float | int = 20,
        face_cfg_scale: float | int = 5.0,
        face_sampler: str | None = "euler_ancestral",
        face_scheduler: str | None = "normal",
        face_guide_size: float | int = 768,
        face_max_size: float | int = 1024,
        face_bbox_threshold: float | int = 0.5,
        face_bbox_dilation: float | int = 10,
        face_bbox_crop_factor: float | int = 2.0,
        face_feather: float | int = 5,
    ) -> tuple[object, object, object, object, object, object]:
        del size_preset

        def failure(message: str) -> tuple[object, object, object, object, object, object]:
            return (
                gr.Button("生成", interactive=True),
                "",
                None,
                message,
                False,
                gr.skip(),
            )

        try:
            loras = lora_settings_from_state(lora_state, max_loras=max_loras)
            detailers = _face_detailers_from_ui(
                face_detailer_enabled,
                face_detector_model,
                face_positive_prompt,
                face_negative_prompt,
                face_denoise,
                face_steps,
                face_cfg_scale,
                face_sampler,
                face_scheduler,
                face_guide_size,
                face_max_size,
                face_bbox_threshold,
                face_bbox_dilation,
                face_bbox_crop_factor,
                face_feather,
            )
            effective_positive_prompt = resolve_effective_positive_prompt(
                positive_prompt or "", loras, lora_catalog_service
            )
            generation_settings = GenerationSettings(
                positive_prompt=effective_positive_prompt,
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                vae_name=vae,
                loras=loras,
                detailers=detailers,
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
                clip_skip=int(clip_skip),
                hires_fix=bool(hires_fix),
                hires_scale=float(hires_scale),
                hires_resize_method=hires_resize_method,
                hires_steps=int(hires_steps),
                hires_cfg_scale=float(hires_cfg_scale),
                hires_sampler_name=hires_sampler or "euler",
                hires_scheduler_name=hires_scheduler or "normal",
                hires_denoise=float(hires_denoise),
                final_upscale=bool(final_upscale),
                final_upscale_model=upscaler if final_upscale else None,
            )
        except LoraTriggerResolutionError as exc:
            return failure(str(exc))
        except (TypeError, ValueError, ValidationError):
            return failure(
                final_upscale_validation_message(bool(final_upscale), upscaler)
                or "入力値を確認してください。"
            )
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
            if interactive_service is None:
                queued = service.enqueue(
                    generation_settings,
                    parent_generation_id=parent_id,
                )
            else:

                def check_interactive_admission() -> None:
                    try:
                        interactive_service.ensure_no_active_run()
                    except InteractiveGenerationError as exc:
                        raise GenerationQueueServiceError(str(exc)) from exc

                queued = service.enqueue(
                    generation_settings,
                    parent_generation_id=parent_id,
                    admission_check=check_interactive_admission,
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
                        positive_prompt=effective_positive_prompt,
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
                        auto_trigger_lora_names=tuple(
                            lora.name for lora in loras if lora.auto_add_trigger_words
                        ),
                        face_detailer_enabled=bool(face_detailer_enabled),
                        face_detailer=detailers[0] if detailers else None,
                        clip_skip=int(clip_skip),
                        hires_fix=bool(hires_fix),
                        hires_scale=float(hires_scale),
                        hires_resize_method=hires_resize_method,
                        hires_steps=int(hires_steps),
                        hires_cfg_scale=float(hires_cfg_scale),
                        hires_sampler_name=hires_sampler or "euler",
                        hires_scheduler_name=hires_scheduler or "normal",
                        hires_denoise=float(hires_denoise),
                        final_upscale=bool(final_upscale),
                    )
                )
            except Exception:  # noqa: BLE001 - enqueue success must remain durable
                details += "\n\nLast-used form state could not be saved"
        return (
            gr.Button("生成", interactive=True),
            "Queued",
            None,
            details,
            False,
            str(queued.item.generation.id),
        )

    return handler


def final_upscale_validation_message(final_upscale: bool, upscaler: str | None) -> str:
    """Return a visible, safe hint before a final-upscale request is submitted."""

    if final_upscale and not upscaler:
        return "Final 4x upscaleを使う場合はアップスケーラーを選択してください。"
    return ""


def final_upscaler_visibility(final_upscale: bool) -> gr.Dropdown:
    """Show the upscaler selector only while final upscale is enabled."""

    return gr.Dropdown(visible=bool(final_upscale))


def make_batch_enqueue_handler(
    service: GenerationQueueService,
    max_loras: int,
    preflight_service: GenerationPreflightService | None = None,
    form_state_saver: Callable[[GenerationFormStateSnapshot], object] | None = None,
    lora_catalog_service: LoraCatalogService | None = None,
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
        clip_skip: float | int = 1,
        hires_fix: bool = False,
        hires_scale: float | int = 1.5,
        hires_resize_method: str = "lanczos",
        hires_steps: float | int = 20,
        hires_cfg_scale: float | int = 5.5,
        hires_sampler: str | None = "euler",
        hires_scheduler: str | None = "normal",
        hires_denoise: float | int = 0.4,
        final_upscale: bool = False,
        face_detailer_enabled: bool = False,
        face_detector_model: str | None = None,
        face_positive_prompt: str | None = None,
        face_negative_prompt: str | None = None,
        face_denoise: float | int = 0.22,
        face_steps: float | int = 20,
        face_cfg_scale: float | int = 5.0,
        face_sampler: str | None = "euler_ancestral",
        face_scheduler: str | None = "normal",
        face_guide_size: float | int = 768,
        face_max_size: float | int = 1024,
        face_bbox_threshold: float | int = 0.5,
        face_bbox_dilation: float | int = 10,
        face_bbox_crop_factor: float | int = 2.0,
        face_feather: float | int = 5,
    ) -> tuple[object, ...]:
        try:
            loras = lora_settings_from_state(lora_state, max_loras=max_loras)
            detailers = _face_detailers_from_ui(
                face_detailer_enabled,
                face_detector_model,
                face_positive_prompt,
                face_negative_prompt,
                face_denoise,
                face_steps,
                face_cfg_scale,
                face_sampler,
                face_scheduler,
                face_guide_size,
                face_max_size,
                face_bbox_threshold,
                face_bbox_dilation,
                face_bbox_crop_factor,
                face_feather,
            )
            effective_positive_prompt = resolve_effective_positive_prompt(
                positive_prompt or "", loras, lora_catalog_service
            )
            settings = GenerationSettings(
                positive_prompt=effective_positive_prompt,
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                vae_name=vae,
                loras=loras,
                detailers=detailers,
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
                clip_skip=int(clip_skip),
                hires_fix=bool(hires_fix),
                hires_scale=float(hires_scale),
                hires_resize_method=hires_resize_method,
                hires_steps=int(hires_steps),
                hires_cfg_scale=float(hires_cfg_scale),
                hires_sampler_name=hires_sampler or "euler",
                hires_scheduler_name=hires_scheduler or "normal",
                hires_denoise=float(hires_denoise),
                final_upscale=bool(final_upscale),
                final_upscale_model=upscaler if final_upscale else None,
            )
            preflight_message = ""
            if preflight_service is not None:
                try:
                    preflight = (
                        await preflight_service.check(
                            settings,
                            uses_upscaler=True,
                            upscaler_name=settings.final_upscale_model,
                        )
                        if settings.final_upscale
                        else await preflight_service.check(settings)
                    )
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
        except LoraTriggerResolutionError as exc:
            return (
                gr.Button("バッチをキューへ追加", interactive=True),
                "",
                str(exc),
            )
        except (TypeError, ValueError, ValidationError, GenerationQueueServiceError) as exc:
            return (
                gr.Button("バッチをキューへ追加", interactive=True),
                "",
                str(exc)
                if isinstance(exc, GenerationQueueServiceError)
                else final_upscale_validation_message(bool(final_upscale), upscaler)
                or "入力値を確認してください。",
            )
        if form_state_saver is not None:
            try:
                form_state_saver(
                    GenerationFormStateSnapshot.from_ui(
                        positive_prompt=effective_positive_prompt,
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
                        auto_trigger_lora_names=tuple(
                            lora.name for lora in loras if lora.auto_add_trigger_words
                        ),
                        face_detailer_enabled=bool(face_detailer_enabled),
                        face_detailer=detailers[0] if detailers else None,
                        clip_skip=int(clip_skip),
                        hires_fix=bool(hires_fix),
                        hires_scale=float(hires_scale),
                        hires_resize_method=hires_resize_method,
                        hires_steps=int(hires_steps),
                        hires_cfg_scale=float(hires_cfg_scale),
                        hires_sampler_name=hires_sampler or "euler",
                        hires_scheduler_name=hires_scheduler or "normal",
                        hires_denoise=float(hires_denoise),
                        final_upscale=bool(final_upscale),
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


def make_interactive_start_handler(
    service: InteractiveGenerationService,
    max_loras: int,
    preflight_service: GenerationPreflightService | None = None,
    form_state_saver: Callable[[GenerationFormStateSnapshot], object] | None = None,
    lora_catalog_service: LoraCatalogService | None = None,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    """Create the small UI boundary for one durable interactive run."""

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
        vae: str | None,
        lora_state: object,
        batch_count: float | int,
        batch_size: float | int,
        upscaler: str | None,
        clip_skip: float | int,
        hires_fix: bool,
        hires_scale: float | int,
        hires_resize_method: str,
        hires_steps: float | int,
        hires_cfg_scale: float | int,
        hires_sampler: str | None,
        hires_scheduler: str | None,
        hires_denoise: float | int,
        final_upscale: bool,
        client_local_date: str | None,
        workflow_template_id: str = "sdxl_txt2img",
        workflow_template_version: str = CURRENT_WORKFLOW_TEMPLATE_VERSION,
        face_detailer_enabled: bool = False,
        face_detector_model: str | None = None,
        face_positive_prompt: str | None = None,
        face_negative_prompt: str | None = None,
        face_denoise: float | int = 0.22,
        face_steps: float | int = 20,
        face_cfg_scale: float | int = 5.0,
        face_sampler: str | None = "euler_ancestral",
        face_scheduler: str | None = "normal",
        face_guide_size: float | int = 768,
        face_max_size: float | int = 1024,
        face_bbox_threshold: float | int = 0.5,
        face_bbox_dilation: float | int = 10,
        face_bbox_crop_factor: float | int = 2.0,
        face_feather: float | int = 5,
    ) -> tuple[object, ...]:
        del size_preset
        try:
            loras = lora_settings_from_state(lora_state, max_loras=max_loras)
            detailers = _face_detailers_from_ui(
                face_detailer_enabled,
                face_detector_model,
                face_positive_prompt,
                face_negative_prompt,
                face_denoise,
                face_steps,
                face_cfg_scale,
                face_sampler,
                face_scheduler,
                face_guide_size,
                face_max_size,
                face_bbox_threshold,
                face_bbox_dilation,
                face_bbox_crop_factor,
                face_feather,
            )
            effective_positive_prompt = resolve_effective_positive_prompt(
                positive_prompt or "", loras, lora_catalog_service
            )
            settings = GenerationSettings(
                positive_prompt=effective_positive_prompt,
                negative_prompt=negative_prompt or "",
                checkpoint_name=checkpoint or "",
                sampler_name=sampler or "",
                scheduler_name=scheduler or "",
                vae_name=vae,
                loras=loras,
                detailers=detailers,
                width=int(width),
                height=int(height),
                seed=-1 if seed_mode == "Random" else int(seed),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
                clip_skip=int(clip_skip),
                hires_fix=bool(hires_fix),
                hires_scale=float(hires_scale),
                hires_resize_method=hires_resize_method,
                hires_steps=int(hires_steps),
                hires_cfg_scale=float(hires_cfg_scale),
                hires_sampler_name=hires_sampler or "euler",
                hires_scheduler_name=hires_scheduler or "normal",
                hires_denoise=float(hires_denoise),
                final_upscale=bool(final_upscale),
                final_upscale_model=upscaler if final_upscale else None,
                workflow_template_id=workflow_template_id or "sdxl_txt2img",
                workflow_template_version=workflow_template_version
                or CURRENT_WORKFLOW_TEMPLATE_VERSION,
            )
            if preflight_service is not None:
                preflight = (
                    await preflight_service.check(
                        settings,
                        uses_upscaler=True,
                        upscaler_name=settings.final_upscale_model,
                    )
                    if settings.final_upscale
                    else await preflight_service.check(settings)
                )
                if not preflight.is_ready:
                    return _interactive_idle_outputs(preflight_markdown(preflight))
            view = service.start(
                settings,
                batch_count=int(batch_count),
                batch_size=int(batch_size),
                client_local_date=client_local_date or "",
            )
        except LoraTriggerResolutionError as exc:
            return _interactive_idle_outputs(str(exc))
        except (TypeError, ValueError, ValidationError, InteractiveGenerationError) as exc:
            if bool(final_upscale) and not upscaler:
                message = final_upscale_validation_message(True, upscaler)
            else:
                message = (
                    "入力値または実行条件を確認してください。"
                    if isinstance(exc, ValidationError)
                    else str(exc)
                )
            return _interactive_idle_outputs(message)
        except Exception:  # noqa: BLE001 - hide adapter details at the UI boundary
            return _interactive_idle_outputs("生成を開始できませんでした。")
        if form_state_saver is not None:
            with suppress(Exception):
                form_state_saver(
                    GenerationFormStateSnapshot.from_ui(
                        positive_prompt=effective_positive_prompt,
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
                        upscaler_name=upscaler if final_upscale else None,
                        loras=loras,
                        auto_trigger_lora_names=tuple(
                            lora.name for lora in loras if lora.auto_add_trigger_words
                        ),
                        face_detailer_enabled=bool(face_detailer_enabled),
                        face_detailer=detailers[0] if detailers else None,
                        clip_skip=int(clip_skip),
                        hires_fix=bool(hires_fix),
                        hires_scale=float(hires_scale),
                        hires_resize_method=hires_resize_method,
                        hires_steps=int(hires_steps),
                        hires_cfg_scale=float(hires_cfg_scale),
                        hires_sampler_name=hires_sampler or "euler",
                        hires_scheduler_name=hires_scheduler or "normal",
                        hires_denoise=float(hires_denoise),
                        final_upscale=bool(final_upscale),
                    )
                )
        return _interactive_view_outputs(view)

    return handler


def make_interactive_refresh_handler(
    service: InteractiveGenerationService,
) -> Callable[[str | None, str | None, object], tuple[object, ...]]:
    """Create a reload/timer handler that restores the active run from SQLite."""

    def handler(
        run_id: str | None = None,
        selected_generation_id: str | None = None,
        selected_index: object = None,
    ) -> tuple[object, ...]:
        try:
            identifier = UUID(run_id) if run_id else None
            if identifier is not None:
                if not service.is_current_run(identifier):
                    # A poll queued before a newer start must not overwrite the new run.
                    return (gr.skip(),) * 8
                view = service.refresh(identifier)
            else:
                view = service.restore()
        except InteractiveGenerationError as exc:
            del exc
            return _interactive_idle_outputs("生成状態を取得できませんでした。")
        except ValueError:
            return _interactive_idle_outputs("生成状態を取得できませんでした。")
        if view is None:
            return _interactive_idle_outputs()
        return _interactive_view_outputs(view, selected_generation_id, selected_index)

    return handler


def make_interactive_cancel_handler(
    service: InteractiveGenerationService,
) -> Callable[[str | None], Awaitable[tuple[object, ...]]]:
    """Create the cancel boundary for the active run."""

    async def handler(run_id: str | None) -> tuple[object, ...]:
        try:
            identifier = UUID(run_id) if run_id else None
            view = await service.cancel(identifier)
        except (ValueError, InteractiveGenerationError) as exc:
            del exc
            return _interactive_cancel_outputs(
                run_id,
                "キャンセル処理を継続しています。",
            )
        if view is None:
            return _interactive_idle_outputs()
        return _interactive_view_outputs(view)

    return handler


def _interactive_view_outputs(
    view: InteractiveRunView,
    selected_generation_id: object = None,
    selected_index: object = None,
) -> tuple[object, ...]:
    run = view.run
    status = run.status.value
    completed_count = view.completed_count
    batch_count = run.batch_count
    current_status = view.current_generation_status
    current_batch = (
        min(batch_count, completed_count + 1)
        if status in {"active", "cancelling"}
        else completed_count
    )
    del current_status
    if status == "active":
        detail = f"生成中 · {current_batch} / {batch_count}"
    elif status == "cancelling":
        detail = "キャンセル中…"
    elif status == "completed":
        detail = "完了"
    elif status == "failed":
        detail = "生成に失敗しました"
    else:
        detail = "キャンセルしました"
    paths = list(view.result_image_paths)
    gallery_generation_id = run.last_completed_generation_id
    if gallery_generation_id not in view.run.completed_generation_ids and completed_count:
        gallery_generation_id = view.run.completed_generation_ids[-1]
    gallery_generation_value = (
        str(gallery_generation_id) if gallery_generation_id is not None and paths else None
    )
    selected = (
        selected_index
        if isinstance(selected_index, int)
        and not isinstance(selected_index, bool)
        and 0 <= selected_index < len(paths)
        and selected_generation_id == gallery_generation_value
        else None
    )
    selected_generation_value = gallery_generation_value if selected is not None else None
    return (
        str(run.id),
        gr.Markdown(value=detail, visible=True),
        gr.Gallery(value=paths, visible=bool(paths)),
        gr.Button(value="設定を読み込む", visible=selected is not None),
        "",
        gallery_generation_value,
        selected_generation_value,
        selected,
    )


def _interactive_idle_outputs(message: str = "") -> tuple[object, ...]:
    """Return the hidden initial state for the user-facing status surface."""

    return (
        None,
        gr.Markdown(value="", visible=False),
        gr.Gallery(value=[], visible=False),
        gr.Button(value="設定を読み込む", visible=False),
        message,
        None,
        None,
        None,
    )


def _interactive_cancel_outputs(run_id: str | None, message: str) -> tuple[object, ...]:
    """Keep cancellation visible while invalidating any Gallery selection."""

    return (
        run_id,
        gr.Markdown(value="キャンセル中…", visible=True),
        gr.Gallery(value=[], visible=False),
        gr.Button(value="設定を読み込む", visible=False),
        message,
        None,
        None,
        None,
    )


def interactive_action_updates(status: str | None) -> tuple[object, object]:
    """Derive button state from the durable run projection, not browser state."""

    value = status or ""
    if ": cancelling" in value or "キャンセル中" in value:
        return gr.Button(value="生成", interactive=False), gr.Button(
            value="キャンセル中…", interactive=False, visible=True
        )
    if ": active" in value or value.startswith("生成中"):
        return gr.Button(value="生成", interactive=False), gr.Button(
            value="キャンセル", interactive=True, visible=True
        )
    return gr.Button(value="生成", interactive=True), gr.Button(
        value="キャンセル", interactive=False, visible=False
    )


def make_interactive_gallery_select_handler(
    service: InteractiveGenerationService,
) -> Callable[..., tuple[object, object, gr.Button]]:
    """Validate and store the displayed Generation identity with the image index."""

    def handler(
        selection: gr.SelectData,
        run_id: str | None = None,
        displayed_generation_id: str | None = None,
    ) -> tuple[str | None, int | None, gr.Button]:
        index = getattr(selection, "index", selection)
        if isinstance(index, tuple):
            index = index[0] if index else None
        selected = (
            index if isinstance(index, int) and not isinstance(index, bool) and index >= 0 else None
        )
        if selected is None or not run_id or not displayed_generation_id:
            return None, None, gr.Button(value="設定を読み込む", visible=False)
        try:
            generation_id = service.resolve_gallery_generation(
                UUID(run_id), UUID(displayed_generation_id), selected
            )
        except (InteractiveGenerationError, ValueError):
            return None, None, gr.Button(value="設定を読み込む", visible=False)
        return str(generation_id), selected, gr.Button(value="設定を読み込む", visible=True)

    return handler


def make_interactive_gallery_restore_handler(
    service: InteractiveGenerationService,
    restore_handler: Callable[..., tuple[object, ...]],
) -> Callable[..., tuple[object, ...]]:
    """Validate a Gallery index against the server-owned completed Generation."""

    def handler(
        run_id: str | None,
        selected_generation_id: object,
        selected_index: object,
        checkpoint_choices: object = None,
        vae_choices: object = None,
        lora_choices: object = None,
    ) -> tuple[object, ...]:
        try:
            if (
                run_id is None
                or not isinstance(selected_generation_id, str)
                or not isinstance(selected_index, int)
                or isinstance(selected_index, bool)
            ):
                raise InteractiveGenerationError("今回の生成結果から画像を選択してください。")
            generation_id = service.resolve_gallery_generation(
                UUID(run_id), UUID(selected_generation_id), selected_index
            )
            return restore_handler(
                str(generation_id), checkpoint_choices, vae_choices, lora_choices
            )
        except (InteractiveGenerationError, ValueError):
            # The existing restore handler has a stable safe failure shape.
            return restore_handler(None, checkpoint_choices, vae_choices, lora_choices)

    return handler


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


def disable_generate_button_and_clear_gallery_selection() -> tuple[gr.Button, None, None, None]:
    """Disable Generate and discard all state from the previous result batch."""

    return disable_generate_button(), None, None, None


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


BUILTIN_SIZE_DIMENSIONS = frozenset(
    {
        (1024, 1024),
        (832, 1216),
        (1216, 832),
        (896, 1152),
        (1152, 896),
    }
)


def make_size_preset_handler(
    custom_size_service: GenerationCustomSizeService,
) -> Callable[[str | None], tuple[object, ...]]:
    """Apply built-in or server-owned custom dimensions to the form."""

    def handler(preset: str | None) -> tuple[object, ...]:
        if preset and preset.startswith("custom:"):
            try:
                saved = custom_size_service.resolve(preset)
            except GenerationCustomSizeError as exc:
                return gr.skip(), gr.skip(), gr.Button(interactive=False, visible=False), str(exc)
            if saved is not None:
                return (
                    gr.Number(value=saved.width, visible=True),
                    gr.Number(value=saved.height, visible=True),
                    gr.Button(interactive=True, visible=True),
                    f"保存済みサイズ: {saved.width} × {saved.height}",
                )
        if preset == "Custom":
            return (
                gr.Number(value=1024, visible=True),
                gr.Number(value=1024, visible=True),
                gr.Button(interactive=False, visible=False),
                "幅と高さを入力してください。",
            )
        width, height = size_preset_values(preset or "")
        return (
            gr.Number(value=width, visible=False),
            gr.Number(value=height, visible=False),
            gr.Button(interactive=False, visible=False),
            "",
        )

    return handler


def make_custom_size_delete_handler(
    custom_size_service: GenerationCustomSizeService,
) -> Callable[[str | None], tuple[object, ...]]:
    """Delete only a server-issued custom selector value."""

    def handler(preset: str | None) -> tuple[object, ...]:
        if not preset or not preset.startswith("custom:"):
            return (
                gr.skip(),
                gr.Button(interactive=False, visible=False),
                "組み込みサイズは削除できません。",
            )
        try:
            custom_size_service.delete(preset.removeprefix("custom:"))
        except GenerationCustomSizeError as exc:
            return gr.skip(), gr.Button(interactive=True, visible=True), str(exc)
        choices = [
            "1024 × 1024",
            "832 × 1216",
            "1216 × 832",
            "896 × 1152",
            "1152 × 896",
            *custom_size_service.selector_options(),
            "Custom",
        ]
        return (
            gr.Dropdown(choices=choices, value="Custom"),
            gr.Button(interactive=False, visible=False),
            "保存済みサイズを削除しました。",
        )

    return handler


def make_custom_size_refresh_handler(
    custom_size_service: GenerationCustomSizeService,
    interactive_service: InteractiveGenerationService,
) -> Callable[[str | None, str | None], tuple[object, ...]]:
    """Register a completed run's dimensions and refresh selector choices."""

    def handler(run_id: str | None, current_preset: str | None) -> tuple[object, ...]:
        try:
            view = None
            if run_id:
                view = interactive_service.refresh(UUID(run_id))
            if view is not None and view.completed_count > 0:
                settings = view.run.settings_snapshot
                if (settings.width, settings.height) not in BUILTIN_SIZE_DIMENSIONS:
                    custom_size_service.add(settings.width, settings.height)
            choices: list[object] = [
                "1024 × 1024",
                "832 × 1216",
                "1216 × 832",
                "896 × 1152",
                "1152 × 896",
                *custom_size_service.selector_options(),
                "Custom",
            ]
            return gr.Dropdown(choices=choices, value=current_preset or "1024 × 1024"), ""
        except (GenerationCustomSizeError, InteractiveGenerationError, ValueError):
            try:
                choices = [
                    "1024 × 1024",
                    "832 × 1216",
                    "1216 × 832",
                    "896 × 1152",
                    "1152 × 896",
                    *custom_size_service.selector_options(),
                    "Custom",
                ]
                return gr.Dropdown(choices=choices, value=current_preset), ""
            except GenerationCustomSizeError:
                return gr.skip(), "保存済みサイズを更新できませんでした。"

    return handler


def _capability_updates(
    capabilities: ComfyUICapabilities | None,
    current_values: tuple[object, ...],
    generation: GenerationTabComponents,
    lora_state: object = None,
    lora_choice_options: object = None,
    lora_category: str | None = None,
    category_options: Sequence[str] = (),
    *,
    preserve_unavailable: bool = False,
    final_upscale: bool = False,
) -> tuple[object, ...]:
    if capabilities is None:
        return _empty_updates(generation)
    choices = capability_choices(capabilities)
    dropdowns = (
        (generation.checkpoint, choices["checkpoint"], current_values[0]),
        (generation.vae, choices["vae"], current_values[1]),
        (generation.sampler, choices["sampler"], current_values[2]),
        (generation.scheduler, choices["scheduler"], current_values[3]),
        (generation.hires_sampler, choices["sampler"], current_values[2]),
        (generation.hires_scheduler, choices["scheduler"], current_values[3]),
        (generation.upscaler, choices["upscaler"], current_values[4]),
    )
    updates: list[object] = []
    for component, available_choices, current_value in dropdowns:
        current_string = current_value if isinstance(current_value, str) else None
        display_choices: list[str | tuple[str, str]] = list(available_choices)
        if (
            preserve_unavailable
            and current_string is not None
            and current_string not in available_choices
        ):
            display_choices.append((f"{current_string}（現在利用不可）", current_string))
        if component is generation.vae:
            vae_choices: list[tuple[str, str | None]] = [("Checkpoint内蔵VAE", None)] + [
                (value, value) if isinstance(value, str) else value for value in display_choices
            ]
            updates.append(
                gr.Dropdown(
                    choices=vae_choices,
                    value=(
                        current_string
                        if current_string in available_choices
                        or (
                            preserve_unavailable
                            and current_string in _choice_values(display_choices)
                        )
                        else None
                    ),
                    label=component.label,
                    interactive=True,
                )
            )
        else:
            updates.append(
                gr.Dropdown(
                    choices=display_choices,
                    value=(
                        preserve_selection(current_string, available_choices)
                        if not preserve_unavailable
                        else current_string
                        if current_string in _choice_values(display_choices)
                        else None
                    ),
                    label=component.label,
                    interactive=bool(display_choices),
                    visible=(final_upscale if component is generation.upscaler else True),
                )
            )
    detector_choices = list(capabilities.detector_models)
    current_detector = (
        current_values[5]
        if len(current_values) > 5 and isinstance(current_values[5], str)
        else generation.face_detector_model.value
        if isinstance(generation.face_detector_model.value, str)
        else None
    )
    can_generate = bool(
        capabilities.checkpoints and capabilities.samplers and capabilities.schedulers
    )
    if preserve_unavailable:
        can_generate = can_generate and not any(
            isinstance(value, str) and value and value not in choices[key]
            for value, key in (
                (current_values[0], "checkpoint"),
                (current_values[1], "vae"),
                (current_values[2], "sampler"),
                (current_values[3], "scheduler"),
                (current_values[4], "upscaler"),
            )
        )
        available_lora_values = set(
            _choice_values(
                lora_choice_options
                if isinstance(lora_choice_options, Sequence)
                and not isinstance(lora_choice_options, (str, bytes, bytearray))
                else capabilities.loras
            )
        )
        can_generate = can_generate and all(
            not isinstance(row.get("lora_name"), str) or row["lora_name"] in available_lora_values
            for row in normalize_lora_state(lora_state, len(generation.lora_editor.rows))
        )
    rendered = render_state_updates(
        lora_state,
        lora_choice_options if lora_choice_options is not None else capabilities.loras,
        len(generation.lora_editor.rows),
        clear_unavailable=not preserve_unavailable,
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
        gr.Button(value="生成", interactive=can_generate),
        selected_options,
        *rendered,
        gr.Dropdown(
            choices=list(category_options),
            value=preserve_selection(lora_category, tuple(category_options)),
            label=generation.lora_category_filter.label,
            interactive=bool(category_options),
        ),
        gr.Dropdown(
            choices=detector_choices,
            value=preserve_selection(current_detector, tuple(detector_choices)),
            label=generation.face_detector_model.label,
            interactive=bool(detector_choices),
        ),
    )


def _empty_updates(generation: GenerationTabComponents) -> tuple[object, ...]:
    rendered = render_state_updates(None, [], len(generation.lora_editor.rows))
    return (
        gr.Dropdown([], label=generation.checkpoint.label, interactive=False),
        gr.Dropdown([], label=generation.vae.label, interactive=False),
        gr.Dropdown([], label=generation.sampler.label, interactive=False),
        gr.Dropdown([], label=generation.scheduler.label, interactive=False),
        gr.Dropdown([], label=generation.hires_sampler.label, interactive=False),
        gr.Dropdown([], label=generation.hires_scheduler.label, interactive=False),
        gr.Dropdown([], label=generation.upscaler.label, interactive=False, visible=False),
        None,
        None,
        None,
        None,
        None,
        "**LoRA list:** unavailable",
        gr.Button(value="生成", interactive=False),
        None,
        *rendered,
        gr.Dropdown([], label=generation.lora_category_filter.label, interactive=False),
        gr.Dropdown([], label=generation.face_detector_model.label, interactive=False),
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
