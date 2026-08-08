from __future__ import annotations

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.ui.app_builder import build_app


def test_phase8_generation_surface_is_mobile_ready_and_status_poll_is_wired() -> None:
    demo = build_app(Settings(_env_file=None, environment="phase8-ui-test"))
    components = demo.config["components"]

    generate_buttons = [
        component
        for component in components
        if component["type"] == "button" and component["props"].get("value") == "生成をキューへ追加"
    ]
    timers = [component for component in components if component["type"] == "timer"]
    action_labels = {
        component["props"].get("value") for component in components if component["type"] == "button"
    }

    assert len(generate_buttons) == 1
    assert timers and timers[0]["props"].get("value") == 5
    assert {"同条件で再生成", "設定を編集", "アップスケール", "Seedをコピー"}.issubset(
        action_labels
    )
    assert ".generation-sticky-action" in demo.config["css"]
    assert "safe-area-inset-bottom" in demo.config["css"]

    timer_id = timers[0]["id"]
    timer_dependencies = [
        dependency
        for dependency in demo.config["dependencies"]
        if (timer_id, "tick") in dependency["targets"]
    ]
    assert len(timer_dependencies) == 1
    assert len(timer_dependencies[0]["outputs"]) == 7
