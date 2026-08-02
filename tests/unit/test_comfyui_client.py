from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIConnectionError,
    ComfyUIResponseError,
    ComfyUITimeoutError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import PromptHistoryStatus

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "comfyui"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.asyncio
@respx.mock
async def test_get_system_stats_uses_normalized_url_and_parses_response() -> None:
    route = respx.get("http://comfy.test:8188/system_stats").mock(
        return_value=httpx.Response(200, json=_fixture("system_stats.json"))
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188/", timeout=7)

    stats = await client.get_system_stats()

    assert route.called
    assert stats.comfyui_version == "0.3.30"
    assert client.base_url == "http://comfy.test:8188"
    assert client.timeout == 7
    await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_object_info_is_available_without_exposing_raw_http_response() -> None:
    respx.get("http://comfy.test:8188/object_info").mock(
        return_value=httpx.Response(200, json=_fixture("object_info.json"))
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    object_info = await client.get_object_info()

    assert "CheckpointLoaderSimple" in object_info.nodes
    assert isinstance(object_info.nodes["KSampler"], dict)
    await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_client_construction_does_not_make_a_request() -> None:
    route = respx.get("http://comfy.test:8188/system_stats").mock(
        return_value=httpx.Response(200, json=_fixture("system_stats.json"))
    )

    client = ComfyUIClient(base_url="http://comfy.test:8188")

    assert not route.called
    await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_connection_errors_are_translated() -> None:
    respx.get("http://comfy.test:8188/system_stats").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUIConnectionError):
        await client.get_system_stats()


@pytest.mark.asyncio
@respx.mock
async def test_timeout_errors_are_translated() -> None:
    respx.get("http://comfy.test:8188/system_stats").mock(
        side_effect=httpx.ReadTimeout("request timed out")
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUITimeoutError):
        await client.get_system_stats()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 500])
@respx.mock
async def test_http_errors_are_translated(status_code: int) -> None:
    respx.get("http://comfy.test:8188/system_stats").mock(
        return_value=httpx.Response(status_code, text="error details are not exposed")
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUIResponseError):
        await client.get_system_stats()


@pytest.mark.asyncio
@respx.mock
async def test_invalid_json_is_translated() -> None:
    respx.get("http://comfy.test:8188/system_stats").mock(
        return_value=httpx.Response(200, text="not-json")
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUIResponseError):
        await client.get_system_stats()


@pytest.mark.asyncio
@respx.mock
async def test_check_connection_propagates_connection_error() -> None:
    respx.get("http://comfy.test:8188/system_stats").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUIConnectionError):
        await client.check_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [True, False])
@respx.mock
async def test_cancel_job_requires_boolean_cancelled_response(cancelled: bool) -> None:
    route = respx.post("http://comfy.test:8188/api/jobs/prompt-1/cancel").mock(
        return_value=httpx.Response(200, json={"cancelled": cancelled})
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    assert await client.cancel_job("prompt-1") is cancelled
    assert route.called
    await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_cancel_job_rejects_missing_boolean_response() -> None:
    respx.post("http://comfy.test:8188/api/jobs/prompt-1/cancel").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    with pytest.raises(ComfyUIResponseError):
        await client.cancel_job("prompt-1")
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/queue", {"delete": ["prompt-1"]}),
        ("/interrupt", {"prompt_id": "prompt-1"}),
    ],
)
@respx.mock
async def test_legacy_cancel_operations_accept_empty_success_body(
    path: str, payload: dict[str, object]
) -> None:
    route = respx.post(f"http://comfy.test:8188{path}").mock(return_value=httpx.Response(204))
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    if path == "/queue":
        await client.delete_queued_prompt("prompt-1")
    else:
        await client.interrupt_prompt("prompt-1")
    assert route.called
    assert json.loads(route.calls[0].request.content) == payload
    await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_prompt_history_404_is_typed_as_not_found() -> None:
    respx.get("http://comfy.test:8188/history/prompt-1").mock(
        return_value=httpx.Response(404, text="missing")
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")

    history = await client.get_prompt_history("prompt-1")

    assert history.status is PromptHistoryStatus.NOT_FOUND
    assert history.exists is False
    await client.close()
