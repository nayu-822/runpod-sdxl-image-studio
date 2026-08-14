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


def test_startup_restore_event_wires_all_hires_form_outputs() -> None:
    demo = build_app(Settings(_env_file=None, environment="ui-test"))
    timer_id = next(
        component["id"]
        for component in demo.config["components"]
        if component["type"] == "timer" and component["props"].get("value") == 1.0
    )
    dependency = next(
        item for item in demo.config["dependencies"] if item["targets"] == [(timer_id, "tick")]
    )
    components = {component["id"]: component for component in demo.config["components"]}

    def component_id(label: str) -> int:
        matches = [
            component["id"]
            for component in demo.config["components"]
            if component["props"].get("label") == label
        ]
        assert len(matches) == 1
        return matches[0]

    hires_output_ids = [
        component_id("CLIP skip"),
        component_id("Hires.fix"),
        component_id("Hires倍率"),
        component_id("Hires resize"),
        component_id("Hires Steps"),
        component_id("Hires CFG"),
        component_id("Hires sampler"),
        component_id("Hires scheduler"),
        component_id("Hires denoise"),
        component_id("Final 4x upscale"),
    ]

    assert dependency["outputs"][-(len(hires_output_ids) + 2) : -2] == hires_output_ids
    assert all(
        components[component_id]["type"] == "state" for component_id in dependency["outputs"][-2:]
    )
