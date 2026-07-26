from __future__ import annotations

import gradio as gr

from runpod_sdxl_image_studio.app import build_app
from runpod_sdxl_image_studio.config import Settings


def test_build_app_returns_blocks_without_starting_server() -> None:
    settings = Settings(_env_file=None, environment="test", comfyui_base_url="http://example.test")

    demo = build_app(settings)

    assert isinstance(demo, gr.Blocks)
    assert demo.config["title"] == "RunPod SDXL Image Studio"
