"""Composition root for the Phase 1A Gradio application."""

from __future__ import annotations

import gradio as gr

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    build_generation_tab,
    build_system_tab,
    make_check_connection_handler,
    make_refresh_handler,
)
from runpod_sdxl_image_studio.ui.view_models import initial_status_markdown

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
    comfyui_service = service or ComfyUIService(ComfyUIClient(app_settings))
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
    return demo
