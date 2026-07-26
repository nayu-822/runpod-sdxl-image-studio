from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
import pytest
import respx
from PIL import Image

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIOutputImage,
    PromptHistory,
    QueuedPrompt,
)
from runpod_sdxl_image_studio.adapters.comfyui.parsers import parse_prompt_history
from runpod_sdxl_image_studio.adapters.comfyui.websocket_client import parse_websocket_message
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "comfyui"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _settings() -> GenerationSettings:
    return GenerationSettings(
        positive_prompt="a test image",
        negative_prompt="blurry",
        checkpoint_name="test-model-a.safetensors",
        sampler_name="euler",
        scheduler_name="normal",
        seed=123,
    )


def test_fixed_workflow_binds_only_allowed_generation_values() -> None:
    template = load_txt2img_template()
    workflow = WorkflowAdapter(template.as_mapping()).build_txt2img_workflow(_settings())

    assert workflow["4"]["inputs"]["ckpt_name"] == "test-model-a.safetensors"  # type: ignore[index]
    assert workflow["3"]["inputs"]["seed"] == 123  # type: ignore[index]
    assert workflow["9"]["inputs"]["filename_prefix"] == "runpod_sdxl_image_studio"  # type: ignore[index]


def test_history_and_websocket_fixtures_are_normalized() -> None:
    history = parse_prompt_history(_fixture("history_completed.json"), "prompt-123")
    progress = parse_websocket_message(_fixture("websocket_progress.json"), "prompt-123")
    completed = parse_websocket_message(_fixture("websocket_completed.json"), "prompt-123")

    assert history.is_completed is True
    assert history.outputs == (
        ComfyUIOutputImage("runpod_sdxl_image_studio_00001.png", "", "output"),
    )
    assert progress is not None and progress.percentage == 50.0
    assert completed is not None and completed.state is GenerationStatus.COMPLETED


@pytest.mark.asyncio
@respx.mock
async def test_prompt_and_view_endpoints_validate_and_fetch_image() -> None:
    respx.post("http://comfy.test:8188/prompt").mock(
        return_value=httpx.Response(200, json=_fixture("prompt_response.json"))
    )
    respx.get("http://comfy.test:8188/view").mock(
        return_value=httpx.Response(200, headers={"content-type": "image/png"}, content=b"png")
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    queued = await client.queue_prompt({"3": {"class_type": "KSampler"}}, str(UUID(int=1)))
    image = await client.get_output_image(ComfyUIOutputImage("image.png", "nested", "output"))

    assert queued.prompt_id == "prompt-123"
    assert image == b"png"
    await client.close()


def test_local_storage_validates_and_atomically_saves_png(tmp_path: Path) -> None:
    output = BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="PNG")
    settings = Settings(_env_file=None, data_dir=tmp_path)
    stored = LocalStorageAdapter(settings).store_image(
        output.getvalue(), UUID(int=2), datetime(2026, 7, 26, tzinfo=UTC)
    )

    assert stored.path.exists()
    assert stored.width == 8 and stored.height == 8
    assert stored.sha256
    with pytest.raises(StorageError):
        LocalStorageAdapter(settings).store_image(b"not an image", UUID(int=3), datetime.now(UTC))


@pytest.mark.asyncio
async def test_generation_service_resolves_seed_and_recovers_result(tmp_path: Path) -> None:
    class FakeClient:
        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            assert workflow["3"]["inputs"]["seed"] == 123  # type: ignore[index]
            return QueuedPrompt("prompt-123", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return parse_prompt_history(_fixture("history_completed.json"), prompt_id)

        async def get_output_image(self, image: ComfyUIOutputImage) -> bytes:
            output = BytesIO()
            Image.new("RGB", (4, 4), "blue").save(output, format="PNG")
            return output.getvalue()

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            yield parse_websocket_message(_fixture("websocket_completed.json"), prompt_id)

    capabilities = ComfyUICapabilities(
        checkpoints=("test-model-a.safetensors",),
        vaes=(),
        samplers=("euler",),
        schedulers=("normal",),
        loras=(),
        upscale_models=(),
        available_node_classes=frozenset(),
        warnings=(),
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        generation_timeout_seconds=5,
        history_poll_interval_seconds=0.001,
    )
    service = GenerationService(
        FakeClient(),  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        FakeWebSocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        lambda: _async_capability_result(capabilities),
        settings,
        id_factory=lambda: UUID(int=10),
    )

    result = await service.generate(_settings())

    assert result.status is GenerationStatus.COMPLETED
    assert result.seed == 123
    assert result.stored_image is not None
    assert result.stored_image.path.exists()


async def _async_capability_result(capabilities: ComfyUICapabilities) -> CapabilityRefreshResult:
    return CapabilityRefreshResult(True, "ok", capabilities)
