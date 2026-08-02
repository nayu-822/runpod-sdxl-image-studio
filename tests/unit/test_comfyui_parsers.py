from __future__ import annotations

import json
from pathlib import Path

import pytest

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUIObjectInfo,
    PromptHistoryStatus,
    RemotePromptStatus,
)
from runpod_sdxl_image_studio.adapters.comfyui.parsers import (
    parse_capabilities,
    parse_prompt_history,
    parse_remote_prompt_status,
    parse_system_stats,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "comfyui"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_parse_capabilities_extracts_and_sorts_all_supported_choices() -> None:
    object_info = ComfyUIObjectInfo(nodes=_fixture("object_info.json"))

    capabilities = parse_capabilities(object_info)

    assert capabilities.checkpoints == (
        "test-model-a.safetensors",
        "test-model-b.safetensors",
    )
    assert capabilities.vaes == ("test-vae.safetensors",)
    assert capabilities.samplers == ("ddim", "euler")
    assert capabilities.schedulers == ("karras", "normal")
    assert capabilities.loras == ("test-character.safetensors",)
    assert capabilities.upscale_models == ("test-upscaler.pth",)
    assert capabilities.available_node_classes == frozenset(
        {"CheckpointLoaderSimple", "VAELoader", "KSampler", "LoraLoader", "UpscaleModelLoader"}
    )
    assert capabilities.warnings == ()


def test_parse_capabilities_warns_for_missing_nodes_without_failing() -> None:
    object_info = ComfyUIObjectInfo(nodes={"KSampler": _fixture("object_info.json")["KSampler"]})

    capabilities = parse_capabilities(object_info)

    assert capabilities.checkpoints == ()
    assert capabilities.samplers == ("ddim", "euler")
    assert any("CheckpointLoaderSimple" in warning for warning in capabilities.warnings)
    assert any("UpscaleModelLoader" in warning for warning in capabilities.warnings)


def test_parse_capabilities_rejects_absolute_paths_and_unknown_shapes() -> None:
    object_info = ComfyUIObjectInfo(
        nodes={
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {"ckpt_name": [["C:\\secret\\model.safetensors", "ok.safetensors"]]}
                }
            }
        }
    )

    capabilities = parse_capabilities(object_info)

    assert capabilities.checkpoints == ("ok.safetensors",)
    assert capabilities.warnings


def test_parse_system_stats_accepts_optional_fields_and_unknown_fields() -> None:
    payload = _fixture("system_stats.json")
    payload["unknown_field"] = {"ignored": True}
    payload["system"]["optional_field"] = "ignored"  # type: ignore[index]

    stats = parse_system_stats(payload)

    assert stats.system_os == "linux"
    assert stats.python_version == "3.12.8"
    assert stats.comfyui_version == "0.3.30"
    assert stats.devices[0].vram_total == 17179869184


def test_parse_system_stats_tolerates_missing_devices() -> None:
    stats = parse_system_stats({"system": {"os": "linux"}})

    assert stats.system_os == "linux"
    assert stats.devices == ()


def test_parse_prompt_history_classifies_execution_interrupted() -> None:
    history = parse_prompt_history(
        {
            "prompt-1": {
                "status": {
                    "status_str": "error",
                    "messages": [["execution_interrupted", {"node_id": "4"}]],
                }
            }
        },
        "prompt-1",
    )

    assert history.status is PromptHistoryStatus.INTERRUPTED
    assert history.is_interrupted
    assert not history.is_failed


def test_parse_prompt_history_keeps_unknown_status_safe() -> None:
    history = parse_prompt_history({"prompt-1": {"status": {"status_str": "future"}}}, "prompt-1")

    assert history.status is PromptHistoryStatus.UNKNOWN
    assert not history.is_completed
    assert not history.is_failed


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "pending"}, RemotePromptStatus.PENDING),
        ({"state": "running"}, RemotePromptStatus.IN_PROGRESS),
        ({"status": "success"}, RemotePromptStatus.COMPLETED),
        ({"status": "error"}, RemotePromptStatus.FAILED),
        ({"cancelled": True}, RemotePromptStatus.CANCELLED),
        ({"not_found": True}, RemotePromptStatus.NOT_FOUND),
        ({"future": "shape"}, RemotePromptStatus.UNAVAILABLE),
    ],
)
def test_parse_remote_prompt_status_is_typed_and_safe(
    payload: dict[str, object], expected: RemotePromptStatus
) -> None:
    state = parse_remote_prompt_status(payload, "prompt-1")

    assert state.prompt_id == "prompt-1"
    assert state.status is expected
