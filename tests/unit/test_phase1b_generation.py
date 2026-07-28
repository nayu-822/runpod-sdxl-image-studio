from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
import pytest
import respx
from PIL import Image

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIResponseError,
    ComfyUIWebSocketDisconnectedError,
    WorkflowBindingError,
    WorkflowTemplateError,
    WorkflowValidationError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIOutputImage,
    PromptHistory,
    QueuedPrompt,
)
from runpod_sdxl_image_studio.adapters.comfyui.parsers import parse_prompt_history
from runpod_sdxl_image_studio.adapters.comfyui.websocket_client import (
    ComfyUIWebSocketClient,
    parse_websocket_message,
)
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import (
    GenerationProgress,
    GenerationResult,
    GenerationStatus,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.ui.app_builder import build_app
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    disable_generate_button,
    report_gradio_progress,
)
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
    image_output = BytesIO()
    Image.new("RGB", (2, 2), "red").save(image_output, format="PNG")
    image_bytes = image_output.getvalue()
    respx.post("http://comfy.test:8188/prompt").mock(
        return_value=httpx.Response(200, json=_fixture("prompt_response.json"))
    )
    respx.get("http://comfy.test:8188/view").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=image_bytes,
        )
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    queued = await client.queue_prompt({"3": {"class_type": "KSampler"}}, str(UUID(int=1)))
    image = await client.get_output_image(ComfyUIOutputImage("image.png", "nested", "output"))

    assert queued.prompt_id == "prompt-123"
    assert image == image_bytes
    await client.close()


def test_local_storage_validates_and_atomically_saves_png(tmp_path: Path) -> None:
    output = BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="PNG")
    settings = Settings(_env_file=None, data_dir=tmp_path)
    stored = LocalStorageAdapter(settings).store_image(
        output.getvalue(), UUID(int=2), datetime(2026, 7, 26, tzinfo=UTC)
    )

    assert stored.path.exists()
    assert stored.path.parent.name == "generated"
    assert stored.path.name.startswith("20260726_")
    assert stored.width == 8 and stored.height == 8
    assert stored.sha256
    with pytest.raises(StorageError):
        LocalStorageAdapter(settings).store_image(b"not an image", UUID(int=3), datetime.now(UTC))
    assert not list(stored.path.parent.glob("*.tmp"))


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


def test_workflow_adapter_rejects_invalid_templates_without_mutating_source() -> None:
    source = load_txt2img_template().as_mapping()
    original = deepcopy(source)

    missing_binding = deepcopy(source)
    missing_binding["bindings"].pop("seed")  # type: ignore[union-attr]
    with pytest.raises(WorkflowTemplateError):
        WorkflowAdapter(missing_binding).build_txt2img_workflow(_settings())

    missing_input = deepcopy(source)
    missing_input["workflow"]["4"]["inputs"].pop("ckpt_name")  # type: ignore[index]
    with pytest.raises(WorkflowBindingError):
        WorkflowAdapter(missing_input).build_txt2img_workflow(_settings())

    missing_class = deepcopy(source)
    missing_class["workflow"]["4"]["class_type"] = "UnknownNode"  # type: ignore[index]
    with pytest.raises(WorkflowTemplateError):
        WorkflowAdapter(missing_class).build_txt2img_workflow(_settings())

    unserializable = deepcopy(source)
    unserializable["workflow"]["3"]["inputs"]["bad"] = object()  # type: ignore[index]
    with pytest.raises(WorkflowValidationError):
        WorkflowAdapter(unserializable).build_txt2img_workflow(_settings())

    assert source == original


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("width", 65),
        ("height", 65),
        ("steps", 0),
        ("steps", 151),
        ("cfg_scale", -0.1),
        ("cfg_scale", 30.1),
        ("seed", -2),
        ("seed", 2**64),
        ("checkpoint_name", "  "),
        ("sampler_name", ""),
        ("scheduler_name", "\t"),
    ],
)
def test_generation_settings_rejects_boundary_values(field_name: str, value: object) -> None:
    with pytest.raises(ValueError):
        GenerationSettings(**{**_settings().model_dump(), field_name: value})


def test_generation_settings_accepts_inclusive_boundaries() -> None:
    settings = GenerationSettings(
        **{
            **_settings().model_dump(),
            "width": 2048,
            "height": 2048,
            "steps": 150,
            "cfg_scale": 30,
            "seed": 2**64 - 1,
        }
    )

    assert settings.width * settings.height == 4_194_304


@pytest.mark.asyncio
async def test_generation_service_recovers_after_websocket_disconnect_without_real_sleep(
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.history_calls = 0

        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            return QueuedPrompt("prompt-123", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            self.history_calls += 1
            if self.history_calls < 3:
                return PromptHistory(prompt_id, False, False, (), None)
            return parse_prompt_history(_fixture("history_completed.json"), prompt_id)

        async def get_output_image(self, image: ComfyUIOutputImage) -> bytes:
            output = BytesIO()
            Image.new("RGB", (4, 4), "blue").save(output, format="PNG")
            return output.getvalue()

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            if False:
                yield None
            raise ComfyUIWebSocketDisconnectedError("disconnect")

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
    sleeps: list[float] = []

    async def fake_sleep(value: float) -> None:
        sleeps.append(value)

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        generation_timeout_seconds=5,
        history_poll_interval_seconds=2,
        history_max_attempts=3,
    )
    client = FakeClient()
    service = GenerationService(
        client,  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        FakeWebSocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        lambda: _async_capability_result(capabilities),
        settings,
        sleep=fake_sleep,
        id_factory=lambda: UUID(int=11),
    )

    result = await service.generate(_settings())

    assert result.status is GenerationStatus.COMPLETED
    assert client.history_calls == 3
    assert sleeps == [2, 2]


@pytest.mark.asyncio
async def test_generation_service_fails_after_history_max_attempts_without_unlimited_polling(
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.history_calls = 0

        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            return QueuedPrompt("prompt-123", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            self.history_calls += 1
            return PromptHistory(prompt_id, False, False, (), None)

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            if False:
                yield None
            raise ComfyUIWebSocketDisconnectedError("private disconnect detail")

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
    sleeps: list[float] = []

    async def fake_sleep(value: float) -> None:
        sleeps.append(value)

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        generation_timeout_seconds=5,
        history_poll_interval_seconds=2,
        history_max_attempts=3,
    )
    client = FakeClient()
    service = GenerationService(
        client,  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        FakeWebSocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        lambda: _async_capability_result(capabilities),
        settings,
        sleep=fake_sleep,
        id_factory=lambda: UUID(int=12),
    )

    result = await service.generate(_settings())

    assert result.status is GenerationStatus.FAILED
    assert client.history_calls == 3
    assert sleeps == [2, 2]
    assert result.error_message == "ComfyUIで画像生成を完了できませんでした"
    assert "private disconnect detail" not in (result.error_message or "")


@pytest.mark.asyncio
async def test_failed_generation_handler_reenables_button() -> None:
    class FailedService:
        async def generate(
            self, settings: GenerationSettings, progress_callback: object
        ) -> GenerationResult:
            return GenerationResult(
                generation_id=UUID(int=13),
                prompt_id="prompt-123",
                status=GenerationStatus.FAILED,
                seed=123,
                stored_image=None,
                error_message="安全な生成エラー",
                created_at=datetime.now(UTC),
            )

    from runpod_sdxl_image_studio.ui.tabs.system_tab import make_generate_handler

    handler = make_generate_handler(FailedService(), 8)  # type: ignore[arg-type]
    button, status, image, details, _ = await handler(
        "test-model-a.safetensors",
        "positive",
        "negative",
        "Custom",
        1024,
        1024,
        "Fixed",
        123,
        28,
        5.5,
        "euler",
        "normal",
    )

    assert button.interactive is True
    assert status == "Failed"
    assert image is None
    assert details == "安全な生成エラー"


@pytest.mark.asyncio
async def test_regeneration_validation_failure_stops_before_service_call() -> None:
    class UnexpectedService:
        async def generate(
            self, settings: GenerationSettings, progress_callback: object
        ) -> GenerationResult:
            del settings, progress_callback
            raise AssertionError("regeneration must not start")

    from runpod_sdxl_image_studio.ui.tabs.system_tab import make_generate_handler

    handler = make_generate_handler(UnexpectedService(), 8)  # type: ignore[arg-type]
    button, status, image, details, requested = await handler(
        "test-model-a.safetensors",
        "positive",
        "negative",
        "Custom",
        1024,
        1024,
        "Fixed",
        123,
        28,
        5.5,
        "euler",
        "normal",
        None,
        None,
        str(UUID(int=1)),
        False,
        True,
    )

    assert button.interactive is True
    assert status == ""
    assert image is None
    assert "再生成" in details
    assert requested is False


def test_websocket_abnormal_events_are_ignored_or_normalized() -> None:
    wrong_prompt = parse_websocket_message(
        {"type": "progress", "data": {"prompt_id": "prompt-b", "value": 1, "max": 2}},
        "prompt-a",
    )
    binary_like = parse_websocket_message({"type": "binary", "data": {}}, "prompt-a")
    unknown = parse_websocket_message({"type": "future_event", "data": {}}, "prompt-a")
    error = parse_websocket_message(_fixture("websocket_error.json"), "prompt-123")

    assert wrong_prompt is None
    assert binary_like is None
    assert unknown is None
    assert error is not None and error.state is GenerationStatus.FAILED


@pytest.mark.asyncio
async def test_websocket_client_ignores_binary_and_unrelated_messages() -> None:
    messages = [
        b"binary frame",
        json.dumps({"type": "progress", "data": {"prompt_id": "other", "value": 1, "max": 2}}),
        json.dumps(_fixture("websocket_progress.json")),
        json.dumps(_fixture("websocket_completed.json")),
    ]

    class FakeSocket:
        async def recv(self) -> str | bytes:
            return messages.pop(0)

    class FakeConnection:
        async def __aenter__(self) -> FakeSocket:
            return FakeSocket()

        async def __aexit__(self, *args: object) -> None:
            return None

    settings = Settings(_env_file=None, generation_timeout_seconds=1)
    client = ComfyUIWebSocketClient(settings, connect=lambda *args, **kwargs: FakeConnection())

    updates = [update async for update in client.watch_prompt("prompt-123", "client-1")]

    assert [update.state for update in updates] == [
        GenerationStatus.RUNNING,
        GenerationStatus.COMPLETED,
    ]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    ("content_type", "content"),
    [("text/html", b"<html>"), ("application/json", b"{}"), ("image/png", b"not-png")],
)
async def test_view_endpoint_rejects_non_image_payloads(content_type: str, content: bytes) -> None:
    respx.get("http://comfy.test:8188/view").mock(
        return_value=httpx.Response(200, headers={"content-type": content_type}, content=content)
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUIResponseError):
        await client.get_output_image(ComfyUIOutputImage("image.png", "", "output"))

    await client.close()


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("headers", [{"content-length": "99"}, {}])
async def test_view_endpoint_rejects_oversized_declared_or_actual_bytes(
    headers: dict[str, str],
) -> None:
    content = b"x" * 8
    headers = {"content-type": "image/png", **headers}
    respx.get("http://comfy.test:8188/view").mock(
        return_value=httpx.Response(200, headers=headers, content=content)
    )
    settings = Settings(_env_file=None, max_output_image_bytes=4)
    client = ComfyUIClient(settings)

    with pytest.raises(ComfyUIResponseError):
        await client.get_output_image(ComfyUIOutputImage("image.png", "", "output"))

    await client.close()


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "image",
    [
        ComfyUIOutputImage("../image.png", "", "output"),
        ComfyUIOutputImage("image.png", "../outside", "output"),
        ComfyUIOutputImage("C:\\image.png", "", "output"),
        ComfyUIOutputImage("image.png", "C:\\outside", "output"),
    ],
)
async def test_view_endpoint_rejects_posix_and_windows_traversal(image: ComfyUIOutputImage) -> None:
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUIResponseError):
        await client.get_output_image(image)

    await client.close()


def _legacy_test_disable_generate_button_is_immediate_and_noninteractive() -> None:
    update = disable_generate_button()

    assert update.interactive is False
    assert update.value == "生成中..."
    assert update.value
    assert "逕" not in update.value


def test_disable_generate_button_has_the_expected_label() -> None:
    update = disable_generate_button()

    assert update.value == "生成中..."
    assert update.value
    assert "逕滓" not in update.value
    assert update.interactive is False


def test_gradio_progress_conversion_is_bounded_and_safe() -> None:
    calls: list[tuple[object, str]] = []

    class FakeProgress:
        def __call__(self, value: object, desc: str) -> None:
            calls.append((value, desc))

    report_gradio_progress(
        FakeProgress(),  # type: ignore[arg-type]
        parse_websocket_message({"type": "progress", "data": {"value": -3, "max": 0}}, "prompt-a")
        or GenerationProgress(message="fallback"),
    )
    report_gradio_progress(
        FakeProgress(),
        GenerationProgress(value=99, maximum=28, percentage=101, message="progress"),  # type: ignore[arg-type]
    )

    assert calls[0][0] == 0.0
    assert calls[1][0] == (28, 28)


def test_generation_event_disables_button_before_queued_handler() -> None:
    demo = build_app(Settings(_env_file=None, environment="ui-test"))
    dependencies = demo.config["dependencies"]
    disable_event = next(
        dependency
        for dependency in dependencies
        if dependency["api_name"] == "disable_generate_button"
    )
    generation_event = next(
        dependency
        for dependency in dependencies
        if dependency.get("trigger_after") == disable_event["id"]
    )

    assert disable_event["queue"] is False
    assert generation_event["queue"] is True
    assert generation_event["trigger_after"] == disable_event["id"]
