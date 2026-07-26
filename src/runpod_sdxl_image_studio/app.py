"""Gradio entry point for the application."""

from __future__ import annotations

from runpod_sdxl_image_studio.config import get_settings
from runpod_sdxl_image_studio.ui.app_builder import APP_CSS, build_app


def main() -> None:
    """Launch the Gradio server using configured host and port."""

    settings = get_settings()
    build_app(settings).launch(
        server_name=settings.host,
        server_port=settings.port,
        css=APP_CSS,
        share=False,
    )


if __name__ == "__main__":
    main()
