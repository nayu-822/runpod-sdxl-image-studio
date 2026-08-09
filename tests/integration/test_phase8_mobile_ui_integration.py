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
    copyable_seed_fields = [
        component
        for component in components
        if component["type"] == "textbox" and component["props"].get("show_copy_button") is True
    ]

    assert len(generate_buttons) == 1
    assert timers and timers[0]["props"].get("value") == 5
    assert {"同条件で再生成", "設定を編集", "アップスケール"}.issubset(action_labels)
    assert any(
        component["props"].get("label") == "実使用Seed（コピー）"
        for component in copyable_seed_fields
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
    assert len(timer_dependencies[0]["outputs"]) == 6

    result_regenerate_id = next(
        component["id"]
        for component in components
        if component["type"] == "button"
        and component["props"].get("value") == "同条件で再生成"
        and component["props"].get("elem_classes") == ["mobile-tap-button"]
    )
    result_regenerate_dependency = next(
        dependency
        for dependency in demo.config["dependencies"]
        if dependency["targets"] == [(result_regenerate_id, "click")]
    )
    assert result_regenerate_dependency["queue"] is False
    assert len(result_regenerate_dependency["outputs"]) == 2

    restore_dependency = next(
        dependency
        for dependency in demo.config["dependencies"]
        if dependency.get("trigger_after") == result_regenerate_dependency["id"]
    )
    enqueue_dependency = next(
        dependency
        for dependency in demo.config["dependencies"]
        if dependency.get("trigger_after") == restore_dependency["id"]
    )
    assert len(enqueue_dependency["outputs"]) == 6

    action_input_sets = []
    for action_label in ("設定を編集", "アップスケール"):
        action_id = next(
            component["id"]
            for component in components
            if component["type"] == "button" and component["props"].get("value") == action_label
        )
        action_dependency = next(
            dependency
            for dependency in demo.config["dependencies"]
            if dependency["targets"] == [(action_id, "click")]
        )
        action_input_sets.append(set(action_dependency["inputs"]))
    assert action_input_sets[0] & action_input_sets[1]


def test_phase8_batch_enqueue_does_not_mutate_active_generation_state() -> None:
    demo = build_app(Settings(_env_file=None, environment="phase8-ui-test"))
    components = demo.config["components"]
    dependencies = demo.config["dependencies"]

    def button_id(label: str) -> int:
        return next(
            component["id"]
            for component in components
            if component["type"] == "button" and component["props"].get("value") == label
        )

    def click_dependency(label: str) -> dict[str, object]:
        component_id = button_id(label)
        return next(
            dependency
            for dependency in dependencies
            if dependency["targets"] == [(component_id, "click")]
        )

    edit_dependency = click_dependency("設定を編集")
    upscale_dependency = click_dependency("アップスケール")
    active_state_ids = set(edit_dependency["inputs"]) & set(upscale_dependency["inputs"])
    assert len(active_state_ids) == 1
    active_state_id = next(iter(active_state_ids))

    disable_dependency = click_dependency("バッチをキューへ追加")
    batch_dependency = next(
        dependency
        for dependency in dependencies
        if dependency.get("trigger_after") == disable_dependency["id"]
    )
    poll_dependency = next(
        dependency
        for dependency in dependencies
        if dependency.get("trigger_after") == batch_dependency["id"]
    )

    assert len(batch_dependency["outputs"]) == 3
    assert active_state_id not in batch_dependency["outputs"]
    assert len(poll_dependency["outputs"]) == 6
    assert active_state_id not in poll_dependency["outputs"]
