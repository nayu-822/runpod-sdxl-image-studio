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
