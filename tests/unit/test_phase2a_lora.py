from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import WorkflowError
from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    add_lora_row,
    lora_settings_from_state,
    move_lora_row,
    remove_lora_row,
    update_lora_row,
)
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template


def _settings(**overrides: object) -> GenerationSettings:
    values: dict[str, object] = {
        "checkpoint_name": "test-model-a.safetensors",
        "sampler_name": "euler",
        "scheduler_name": "normal",
        "seed": 42,
    }
    values.update(overrides)
    return GenerationSettings(**values)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "  ",
        "/absolute.safetensors",
        "C:/absolute.safetensors",
        " C:/absolute.safetensors",
        "../escape.safetensors",
        "a/../b",
    ],
)
def test_lora_setting_rejects_empty_absolute_and_traversal_names(name: str) -> None:
    with pytest.raises(ValueError):
        LoraSetting(name=name)


@pytest.mark.parametrize("strength", [-2.0, 0.0, 2.0])
def test_lora_strength_boundaries_are_inclusive(strength: float) -> None:
    setting = LoraSetting(name="style.safetensors", model_strength=strength, clip_strength=strength)
    assert setting.model_strength == strength


@pytest.mark.parametrize("strength", [-2.01, 2.01])
def test_lora_strength_outside_range_is_rejected(strength: float) -> None:
    with pytest.raises(ValueError):
        LoraSetting(name="style.safetensors", model_strength=strength)


def test_generation_settings_reject_duplicate_lora_names_and_orders() -> None:
    first = LoraSetting(name="style.safetensors", order=0)
    with pytest.raises(ValueError):
        _settings(loras=(first, LoraSetting(name="style.safetensors", order=1)))
    with pytest.raises(ValueError):
        _settings(loras=(first, LoraSetting(name="character.safetensors", order=0)))


def test_editor_state_supports_add_remove_reorder_and_domain_conversion() -> None:
    state = add_lora_row([], 3)
    state = update_lora_row(state, 0, "style.safetensors", 0.5, 0.25, 3)
    state = update_lora_row(state, 1, "character.safetensors", 1.2, 0.8, 3)
    assert [row["lora_name"] for row in state] == ["style.safetensors", "character.safetensors"]

    reordered = move_lora_row(state, 1, -1, 3)
    settings = lora_settings_from_state(reordered, 3)
    assert [setting.name for setting in settings] == ["character.safetensors", "style.safetensors"]
    assert [setting.order for setting in settings] == [0, 1]
    assert remove_lora_row(reordered, 0, 3)[0]["lora_name"] == "style.safetensors"


def test_workflow_adds_ordered_lora_chain_and_external_vae_without_mutating_template() -> None:
    source = load_txt2img_template().as_mapping()
    original = deepcopy(source)
    settings = _settings(
        vae_name="test-vae.safetensors",
        loras=(
            LoraSetting(name="character.safetensors", model_strength=0.7, order=1),
            LoraSetting(name="style.safetensors", clip_strength=0.4, order=0),
        ),
    )

    workflow = WorkflowAdapter(source).build_txt2img_workflow(settings)

    assert workflow["vae_external"] == {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "test-vae.safetensors"},
    }
    assert workflow["8"]["inputs"]["vae"] == ["vae_external", 0]  # type: ignore[index]
    assert workflow["lora_000"]["inputs"]["lora_name"] == "style.safetensors"  # type: ignore[index]
    assert workflow["lora_001"]["inputs"]["model"] == ["lora_000", 0]  # type: ignore[index]
    assert workflow["3"]["inputs"]["model"] == ["lora_001", 0]  # type: ignore[index]
    assert workflow["6"]["inputs"]["clip"] == ["lora_001", 1]  # type: ignore[index]
    assert workflow["7"]["inputs"]["clip"] == ["lora_001", 1]  # type: ignore[index]
    assert "vae_external" not in original
    assert "lora_000" not in original


def test_workflow_does_not_add_optional_nodes_when_not_selected() -> None:
    workflow = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(
        _settings()
    )

    assert "vae_external" not in workflow
    assert not any(
        isinstance(node, dict) and node.get("class_type") == "LoraLoader"
        for node in workflow.values()
    )
    assert workflow["8"]["inputs"]["vae"] == ["4", 2]  # type: ignore[index]


def test_service_limits_loras_and_requires_optional_node_capabilities(tmp_path: Path) -> None:
    from runpod_sdxl_image_studio.services.generation_service import _validate_generation

    settings = Settings(_env_file=None, data_dir=tmp_path, max_loras=1)
    capabilities = ComfyUICapabilities(
        checkpoints=("test-model-a.safetensors",),
        vaes=("test-vae.safetensors",),
        samplers=("euler",),
        schedulers=("normal",),
        loras=("style.safetensors",),
        upscale_models=(),
        available_node_classes=frozenset(),
        warnings=(),
    )
    with pytest.raises(WorkflowError, match="maximum"):
        _validate_generation(
            _settings(
                loras=(
                    LoraSetting(name="style.safetensors"),
                    LoraSetting(name="character.safetensors", order=1),
                )
            ),
            capabilities,
            settings,
        )
    with pytest.raises(WorkflowError, match="LoRA loading"):
        _validate_generation(
            _settings(loras=(LoraSetting(name="style.safetensors"),)), capabilities, settings
        )
    with pytest.raises(WorkflowError, match="VAE loading"):
        _validate_generation(_settings(vae_name="test-vae.safetensors"), capabilities, settings)
