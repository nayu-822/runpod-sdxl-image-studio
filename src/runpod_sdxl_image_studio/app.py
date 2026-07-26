"""Gradio entry point for the Phase 0 application shell."""

from __future__ import annotations

import gradio as gr

from runpod_sdxl_image_studio.config import Settings, get_settings

APP_TITLE = "RunPod SDXL Image Studio"
APP_CSS = """
/* Keep the Phase 0 shell usable on narrow mobile screens. */
.gradio-container { max-width: 960px !important; width: 100% !important; }
@media (max-width: 640px) {
  .gradio-container { padding: 0.75rem !important; }
  button { min-height: 2.75rem; }
}
"""


def build_app(settings: Settings | None = None) -> gr.Blocks:
    """Build the UI without starting a server or contacting external services."""

    app_settings = settings or get_settings()
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown("Phase 0 setup complete")

        with gr.Group():
            gr.Markdown("### Runtime configuration")
            gr.Markdown(f"**Environment:** `{app_settings.environment}`")
            gr.Markdown(f"**ComfyUI URL:** `{app_settings.comfyui_base_url}`")
            gr.Markdown(f"**Data directory:** `{app_settings.data_dir}`")

        gr.Markdown(
            "ComfyUI connectivity, image generation, database persistence, and Google Drive "
            "sync will be added in later phases."
        )

    return demo


def main() -> None:
    """Launch the Gradio server using the configured host and port."""

    settings = get_settings()
    build_app(settings).launch(
        server_name=settings.host,
        server_port=settings.port,
        css=APP_CSS,
        share=False,
    )


if __name__ == "__main__":
    main()
