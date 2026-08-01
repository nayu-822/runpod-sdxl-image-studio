from __future__ import annotations

import gradio as gr

from runpod_sdxl_image_studio.app import build_app
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.ui.view_models import initial_status_markdown


def test_build_app_returns_blocks_without_starting_server() -> None:
    settings = Settings(_env_file=None, environment="test", comfyui_base_url="http://example.test")

    demo = build_app(settings)

    assert isinstance(demo, gr.Blocks)
    assert demo.config["title"] == "RunPod SDXL Image Studio"
    assert ".gradio-container" in demo.config["css"]


def test_status_markdown_does_not_duplicate_the_dedicated_url_field() -> None:
    initial_markdown = initial_status_markdown()

    assert "接続先URL" not in initial_markdown


def test_recent_settings_refresh_event_updates_six_dropdowns_and_message() -> None:
    demo = build_app(Settings(_env_file=None, environment="ui-test"))
    refresh_id = next(
        component["id"]
        for component in demo.config["components"]
        if component["type"] == "button"
        and component["props"].get("value") == "最近使った設定を更新"
    )
    message_id = next(
        component["id"]
        for component in demo.config["components"]
        if component["type"] == "markdown"
        and component["props"].get("value") == "Presetを検索または保存してください。"
    )
    dependency = next(
        item for item in demo.config["dependencies"] if item["targets"] == [(refresh_id, "click")]
    )

    assert len(dependency["outputs"]) == 7
    assert dependency["outputs"][-1] == message_id
