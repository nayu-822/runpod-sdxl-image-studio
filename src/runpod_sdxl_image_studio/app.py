"""Gradio entry point for the application."""

from __future__ import annotations

from runpod_sdxl_image_studio.config import get_settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database
from runpod_sdxl_image_studio.ui.app_builder import build_app, build_application_runtime

__all__ = ["build_app", "main"]


def main() -> None:
    """Launch the Gradio server using configured host and port."""

    settings = get_settings()
    upgrade_database(settings)
    runtime = build_application_runtime(settings)
    runtime.start()
    try:
        runtime.demo.launch(
            server_name=settings.host,
            server_port=settings.port,
            share=False,
        )
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
