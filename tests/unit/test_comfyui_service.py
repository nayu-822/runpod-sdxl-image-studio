from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import ComfyUITimeoutError
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUIConnectionResult,
    ComfyUIObjectInfo,
    ComfyUISystemStats,
)
from runpod_sdxl_image_studio.adapters.comfyui.parsers import parse_object_info
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "comfyui"


class FakeClient:
    def __init__(self) -> None:
        self.connection_result = ComfyUIConnectionResult(
            is_connected=True,
            message="接続できます",
            checked_at=datetime.now(UTC),
            system_stats=ComfyUISystemStats("linux", "3.12", False, "0.3.30", ()),
        )
        self.object_info = parse_object_info(
            json.loads((FIXTURE_DIR / "object_info.json").read_text(encoding="utf-8"))
        )

    async def check_connection(self) -> ComfyUIConnectionResult:
        return self.connection_result

    async def get_object_info(self) -> ComfyUIObjectInfo:
        return self.object_info


@pytest.mark.asyncio
async def test_service_aggregates_status_and_capabilities() -> None:
    client = FakeClient()

    status = await ComfyUIService(client).get_status()

    assert status.is_connected is True
    assert status.capabilities is not None
    assert status.capabilities.checkpoints == (
        "test-model-a.safetensors",
        "test-model-b.safetensors",
    )
    assert status.error_summary is None


@pytest.mark.asyncio
async def test_service_preserves_parser_warnings() -> None:
    client = FakeClient()
    client.object_info = ComfyUIObjectInfo(nodes={"KSampler": client.object_info.nodes["KSampler"]})

    status = await ComfyUIService(client).get_status()

    assert status.capabilities is not None
    assert status.warnings


@pytest.mark.asyncio
async def test_service_converts_low_level_timeout_to_safe_status() -> None:
    client = FakeClient()

    async def timeout() -> ComfyUIConnectionResult:
        raise ComfyUITimeoutError("internal timeout details")

    client.check_connection = timeout  # type: ignore[method-assign]

    status = await ComfyUIService(client).get_status()

    assert status.is_connected is False
    assert status.error_summary == "ComfyUIへの接続がタイムアウトしました"
    assert "internal timeout" not in status.message
