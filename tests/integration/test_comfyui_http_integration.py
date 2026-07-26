from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "comfyui"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.asyncio
@respx.mock
async def test_service_client_parser_flow_uses_fixture_http_responses() -> None:
    stats_route = respx.get("http://comfy.test:8188/system_stats").mock(
        return_value=httpx.Response(200, json=_fixture("system_stats.json"))
    )
    object_info_route = respx.get("http://comfy.test:8188/object_info").mock(
        return_value=httpx.Response(200, json=_fixture("object_info.json"))
    )
    settings = Settings(_env_file=None, comfyui_base_url="http://comfy.test:8188")

    async with ComfyUIClient(settings) as client:
        status = await ComfyUIService(client).get_status()

    assert stats_route.called
    assert object_info_route.called
    assert status.is_connected is True
    assert status.system_stats is not None
    assert status.system_stats.devices[0].name == "NVIDIA Test GPU"
    assert status.capabilities is not None
    assert status.capabilities.loras == ("test-character.safetensors",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_kind", "expected_message"),
    [
        ("timeout", "ComfyUIへの接続がタイムアウトしました"),
        ("connection", "ComfyUIへ接続できませんでした"),
        ("http_error", "ComfyUIから正常な応答を取得できませんでした"),
        ("invalid_json", "ComfyUIから正常な応答を取得できませんでした"),
    ],
)
@respx.mock
async def test_real_client_errors_are_converted_by_service(
    response_kind: str,
    expected_message: str,
) -> None:
    route = respx.get("http://comfy.test:8188/system_stats")
    if response_kind == "timeout":
        route.mock(side_effect=httpx.ReadTimeout("internal timeout details"))
    elif response_kind == "connection":
        route.mock(side_effect=httpx.ConnectError("connection refused"))
    elif response_kind == "http_error":
        route.mock(return_value=httpx.Response(500, text="private response details"))
    else:
        route.mock(return_value=httpx.Response(200, text="not-json"))

    settings = Settings(_env_file=None, comfyui_base_url="http://comfy.test:8188")
    async with ComfyUIClient(settings) as client:
        status = await ComfyUIService(client).get_status()

    assert route.called
    assert status.is_connected is False
    assert status.error_summary == expected_message
    assert "internal" not in status.message
    assert "private" not in status.message
