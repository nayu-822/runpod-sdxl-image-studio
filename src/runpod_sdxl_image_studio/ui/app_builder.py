"""Composition root for the Phase 1A Gradio application."""

from __future__ import annotations

import gradio as gr

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.websocket_client import ComfyUIWebSocketClient
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    build_generation_tab,
    build_system_tab,
    disable_generate_button,
    make_check_connection_handler,
    make_generate_handler,
    make_refresh_handler,
    size_preset_values,
)
from runpod_sdxl_image_studio.ui.view_models import initial_status_markdown
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template

APP_TITLE = "RunPod SDXL Image Studio"
APP_CSS = """
.gradio-container { max-width: 960px !important; width: 100% !important; }
@media (max-width: 640px) {
  .gradio-container { padding: 0.75rem !important; }
  button { min-height: 2.75rem; }
}
"""


def build_app(
    settings: Settings | None = None,
    service: ComfyUIService | None = None,
) -> gr.Blocks:
    """Build the UI without starting a server or contacting ComfyUI."""

    app_settings = settings or get_settings()
    client = ComfyUIClient(app_settings)
    comfyui_service = service or ComfyUIService(client)
    loaded_workflow = load_txt2img_template(
        app_settings.workflow_dir.parent if app_settings.workflow_dir.exists() else None
    )
    generation_service = GenerationService(
        client,
        WorkflowAdapter(loaded_workflow.as_mapping()),
        ComfyUIWebSocketClient(app_settings),
        LocalStorageAdapter(app_settings),
        comfyui_service.refresh_capabilities,
        app_settings,
    )
    with gr.Blocks(title=APP_TITLE, css=APP_CSS) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        with gr.Tab("生成"):
            generation = build_generation_tab()
        with gr.Tab("システム"):
            system = build_system_tab(
                app_settings.comfyui_base_url,
                initial_status_markdown(),
            )

        capability_inputs = [
            generation.checkpoint,
            generation.vae,
            generation.sampler,
            generation.scheduler,
            generation.upscaler,
        ]
        capability_outputs = [
            generation.checkpoint,
            generation.vae,
            generation.sampler,
            generation.scheduler,
            generation.upscaler,
            generation.lora_list,
            generation.generate_button,
        ]
        system.connection_button.click(
            fn=make_check_connection_handler(
                comfyui_service,
                app_settings.timezone,
                generation,
            ),
            inputs=capability_inputs,
            outputs=[system.status_markdown, system.capability_message, *capability_outputs],
        )
        system.refresh_button.click(
            fn=make_refresh_handler(comfyui_service, generation),
            inputs=capability_inputs,
            outputs=[system.capability_message, *capability_outputs],
        )
        generation.size_preset.change(
            fn=lambda preset: size_preset_values(preset),
            inputs=[generation.size_preset],
            outputs=[generation.width, generation.height],
        )
        generate_event = generation.generate_button.click(
            fn=disable_generate_button,
            outputs=[generation.generate_button],
            queue=False,
        )
        generate_event.then(
            fn=make_generate_handler(generation_service),
            inputs=[
                generation.checkpoint,
                generation.positive_prompt,
                generation.negative_prompt,
                generation.size_preset,
                generation.width,
                generation.height,
                generation.seed_mode,
                generation.seed,
                generation.steps,
                generation.cfg_scale,
                generation.sampler,
                generation.scheduler,
            ],
            outputs=[
                generation.generate_button,
                generation.progress,
                generation.result_image,
                generation.result_details,
            ],
            concurrency_limit=1,
        )
    return demo
