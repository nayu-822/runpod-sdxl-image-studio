from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    WorkflowError,
    WorkflowTemplateError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIOutputImage,
    PromptHistory,
    QueuedPrompt,
)
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import (
    GenerationProgress,
    GenerationResult,
    GenerationStatus,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    add_lora_row,
    lora_settings_from_state,
    move_lora_row,
    remove_lora_row,
    render_state_updates,
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


def test_lora_settings_from_state_uses_configured_limit_above_eight() -> None:
    state: object = []
    for index in range(10):
        state = add_lora_row(state, 12)
        state = update_lora_row(
            state,
            index,
            f"lora-{index}.safetensors",
            1.0,
            1.0,
            12,
        )

    settings = lora_settings_from_state(state, max_loras=12)

    assert len(settings) == 10
    assert [setting.order for setting in settings] == list(range(10))


def test_lora_editor_respects_configured_limit_below_eight(tmp_path: Path) -> None:
    state: object = []
    for index in range(3):
        state = add_lora_row(state, 3)
        state = update_lora_row(state, index, f"lora-{index}.safetensors", 1.0, 1.0, 3)
    fourth = add_lora_row(state, 3)

    assert len(fourth) == 3
    assert len(lora_settings_from_state(fourth, max_loras=3)) == 3

    from runpod_sdxl_image_studio.services.generation_service import _validate_generation

    settings = Settings(_env_file=None, data_dir=tmp_path, max_loras=3)
    capabilities = ComfyUICapabilities(
        checkpoints=("test-model-a.safetensors",),
        vaes=(),
        samplers=("euler",),
        schedulers=("normal",),
        loras=tuple(f"lora-{index}.safetensors" for index in range(4)),
        upscale_models=(),
        available_node_classes=frozenset({"LoraLoader"}),
        warnings=(),
    )
    _validate_generation(
        _settings(
            loras=tuple(
                LoraSetting(name=f"lora-{index}.safetensors", order=index) for index in range(3)
            )
        ),
        capabilities,
        settings,
    )
    with pytest.raises(WorkflowError, match="maximum"):
        _validate_generation(
            _settings(
                loras=tuple(
                    LoraSetting(name=f"lora-{index}.safetensors", order=index) for index in range(4)
                )
            ),
            capabilities,
            settings,
        )


def test_ui_state_strength_is_not_silently_clamped() -> None:
    state = [
        {
            "row_id": "one",
            "lora_name": "style.safetensors",
            "model_strength": 3.0,
            "clip_strength": -3.0,
        }
    ]

    with pytest.raises(ValueError):
        lora_settings_from_state(state, max_loras=3)

    invalid_state = [
        {
            "row_id": "one",
            "lora_name": "style.safetensors",
            "model_strength": "not-a-number",
            "clip_strength": 1.0,
        }
    ]
    with pytest.raises(ValueError):
        lora_settings_from_state(invalid_state, max_loras=3)

    default_state = [
        {
            "row_id": "one",
            "lora_name": "style.safetensors",
            "model_strength": "",
            "clip_strength": None,
        }
    ]
    settings = lora_settings_from_state(default_state, max_loras=3)
    assert settings[0].model_strength == 1.0
    assert settings[0].clip_strength == 1.0


def test_restore_render_keeps_unavailable_lora_and_all_row_values() -> None:
    state = [
        {
            "row_id": "one",
            "lora_name": "removed.safetensors",
            "model_strength": 0.4,
            "clip_strength": 0.8,
        },
        {
            "row_id": "two",
            "lora_name": "available.safetensors",
            "model_strength": 1.2,
            "clip_strength": 0.6,
        },
    ]

    updates = render_state_updates(state, ["available.safetensors"], 3)

    assert updates[0][0]["auto_add_trigger_words"] is False
    assert updates[0][1]["auto_add_trigger_words"] is False
    assert updates[2].value == "removed.safetensors"
    assert ("removed.safetensors（現在利用不可）", "removed.safetensors") in updates[2].choices
    assert updates[3].value == 0.4
    assert updates[4].value == 0.8
    assert updates[10].value == "available.safetensors"


@pytest.mark.asyncio
async def test_generate_handler_passes_configured_lora_limit() -> None:
    class CaptureService:
        def __init__(self) -> None:
            self.settings: GenerationSettings | None = None

        async def generate(
            self,
            settings: GenerationSettings,
            progress_callback: object,
        ) -> GenerationResult:
            del progress_callback
            self.settings = settings
            return GenerationResult(
                generation_id=UUID(int=1),
                prompt_id="prompt-1",
                status=GenerationStatus.FAILED,
                seed=42,
                stored_image=None,
                error_message="failed",
                created_at=datetime.now(UTC),
            )

    state: object = []
    for index in range(10):
        state = add_lora_row(state, 12)
        state = update_lora_row(state, index, f"lora-{index}.safetensors", 1.0, 1.0, 12)

    service = CaptureService()
    from runpod_sdxl_image_studio.ui.tabs.system_tab import make_generate_handler

    handler = make_generate_handler(service, 12)  # type: ignore[arg-type]
    await handler(
        "test-model-a.safetensors",
        "positive",
        "negative",
        "Custom",
        1024,
        1024,
        "Fixed",
        42,
        28,
        5.5,
        "euler",
        "normal",
        None,
        state,
    )

    assert service.settings is not None
    assert len(service.settings.loras) == 10

    _, _, image, details, _ = await handler(
        "test-model-a.safetensors",
        "positive",
        "negative",
        "Custom",
        1024,
        1024,
        "Fixed",
        42,
        28,
        5.5,
        "euler",
        "normal",
        None,
        [
            {
                "row_id": "invalid",
                "lora_name": "lora-0.safetensors",
                "model_strength": 3.0,
                "clip_strength": 1.0,
            }
        ],
    )
    assert image is None
    assert "LoRA" in details
    assert "pydantic" not in details.lower()


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


def test_workflow_three_loras_preserves_model_clip_strength_and_chain() -> None:
    settings = _settings(
        loras=tuple(
            LoraSetting(
                name=f"lora-{index}.safetensors",
                model_strength=index / 10,
                clip_strength=-(index / 10),
                order=index,
            )
            for index in range(3)
        )
    )

    workflow = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(
        settings
    )

    assert [workflow[f"lora_{index:03d}"]["inputs"]["model"] for index in range(3)] == [  # type: ignore[index]
        ["4", 0],
        ["lora_000", 0],
        ["lora_001", 0],
    ]
    assert [workflow[f"lora_{index:03d}"]["inputs"]["clip"] for index in range(3)] == [  # type: ignore[index]
        ["4", 1],
        ["lora_000", 1],
        ["lora_001", 1],
    ]
    for index in range(3):
        node = workflow[f"lora_{index:03d}"]["inputs"]  # type: ignore[index]
        assert node["strength_model"] == index / 10
        assert node["strength_clip"] == -(index / 10)
    assert workflow["3"]["inputs"]["model"] == ["lora_002", 0]  # type: ignore[index]
    assert workflow["6"]["inputs"]["clip"] == ["lora_002", 1]  # type: ignore[index]
    assert workflow["7"]["inputs"]["clip"] == ["lora_002", 1]  # type: ignore[index]


@pytest.mark.parametrize("reserved_id", ["lora_000", "vae_external"])
def test_workflow_rejects_reserved_optional_node_id_collision(reserved_id: str) -> None:
    template = deepcopy(load_txt2img_template().as_mapping())
    template["workflow"][reserved_id] = {  # type: ignore[index]
        "class_type": "Placeholder",
        "inputs": {},
    }
    settings = _settings(
        vae_name="test-vae.safetensors" if reserved_id == "vae_external" else None,
        loras=(LoraSetting(name="style.safetensors"),) if reserved_id == "lora_000" else (),
    )

    with pytest.raises(WorkflowTemplateError, match="reserved"):
        WorkflowAdapter(template).build_txt2img_workflow(settings)


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


@pytest.mark.asyncio
async def test_generation_service_posts_three_loras_and_external_vae(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.queue_calls = 0
            self.workflows: list[dict[str, object]] = []

        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del client_id
            self.queue_calls += 1
            self.workflows.append(workflow)  # type: ignore[arg-type]
            return QueuedPrompt("prompt-2a", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                True,
                False,
                (ComfyUIOutputImage("output.png", "", "output"),),
                None,
            )

        async def get_output_image(self, image: ComfyUIOutputImage) -> bytes:
            del image
            output = BytesIO()
            Image.new("RGB", (4, 4), "blue").save(output, format="PNG")
            return output.getvalue()

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            del client_id
            yield GenerationProgress(
                prompt_id=prompt_id,
                state=GenerationStatus.COMPLETED,
                message="completed",
            )

    capabilities = ComfyUICapabilities(
        checkpoints=("test-model-a.safetensors",),
        vaes=("test-vae.safetensors",),
        samplers=("euler",),
        schedulers=("normal",),
        loras=("lora-0.safetensors", "lora-1.safetensors", "lora-2.safetensors"),
        upscale_models=(),
        available_node_classes=frozenset({"LoraLoader", "VAELoader"}),
        warnings=(),
    )

    async def capabilities_provider() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        generation_timeout_seconds=5,
        max_loras=3,
    )
    usage_calls: list[tuple[tuple[str, ...], datetime]] = []

    class FailingUsageRecorder:
        def record_usage(self, file_names: tuple[str, ...], completed_at: datetime) -> None:
            usage_calls.append((file_names, completed_at))
            raise RuntimeError("database is temporarily unavailable")

    client = FakeClient()
    service = GenerationService(
        client,  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        FakeWebSocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capabilities_provider,
        settings,
        lora_catalog_service=FailingUsageRecorder(),
        id_factory=lambda: UUID(int=22),
    )
    generation_settings = _settings(
        seed=123,
        vae_name="test-vae.safetensors",
        loras=tuple(
            LoraSetting(
                name=f"lora-{index}.safetensors",
                model_strength=0.5 + index / 10,
                clip_strength=0.2 + index / 10,
                order=index,
            )
            for index in range(3)
        ),
    )

    result = await service.generate(generation_settings)

    assert result.status is GenerationStatus.COMPLETED
    assert result.seed == 123
    assert client.queue_calls == 1
    workflow = client.workflows[0]
    assert [workflow[f"lora_{index:03d}"]["inputs"]["lora_name"] for index in range(3)] == [  # type: ignore[index]
        f"lora-{index}.safetensors" for index in range(3)
    ]
    assert workflow["vae_external"]["inputs"]["vae_name"] == "test-vae.safetensors"  # type: ignore[index]
    assert workflow["3"]["inputs"]["seed"] == 123  # type: ignore[index]
    assert result.stored_image is not None
    assert result.status is GenerationStatus.COMPLETED
    assert len(usage_calls) == 1
    assert usage_calls[0][0] == (
        "lora-0.safetensors",
        "lora-1.safetensors",
        "lora-2.safetensors",
    )
    assert usage_calls[0][1] > result.created_at


@pytest.mark.asyncio
async def test_generation_service_rejects_missing_lora_before_prompt(tmp_path: Path) -> None:
    class NoPromptClient:
        queue_calls = 0

        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del workflow, client_id
            self.queue_calls += 1
            return QueuedPrompt("unexpected", 1, {})

    capabilities = ComfyUICapabilities(
        checkpoints=("test-model-a.safetensors",),
        vaes=(),
        samplers=("euler",),
        schedulers=("normal",),
        loras=(),
        upscale_models=(),
        available_node_classes=frozenset({"LoraLoader"}),
        warnings=(),
    )

    async def capabilities_provider() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    settings = Settings(_env_file=None, data_dir=tmp_path, generation_timeout_seconds=5)
    client = NoPromptClient()
    service = GenerationService(
        client,  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        object(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capabilities_provider,
        settings,
        id_factory=lambda: UUID(int=23),
    )

    result = await service.generate(_settings(loras=(LoraSetting(name="missing.safetensors"),)))

    assert result.status is GenerationStatus.FAILED
    assert client.queue_calls == 0
    assert result.error_message == "生成設定を確認できませんでした。"
