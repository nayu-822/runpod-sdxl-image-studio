from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import gradio as gr
import pytest

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUISystemStats,
)
from runpod_sdxl_image_studio.domain.pod_lifecycle import AutoTerminateState, TerminateReadiness
from runpod_sdxl_image_studio.domain.system_status import (
    CapabilityRefreshResult,
    ComfyUIStatus,
)
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    build_generation_tab,
    capability_refresh_outputs,
    make_check_connection_handler,
    make_pod_lifecycle_terminate_handler,
    make_refresh_handler,
)


class FakeService:
    def __init__(self) -> None:
        self.called = False

    async def get_status(self) -> ComfyUIStatus:
        self.called = True
        capabilities = ComfyUICapabilities(
            checkpoints=("model.safetensors",),
            vaes=("vae.safetensors",),
            samplers=("euler",),
            schedulers=("normal",),
            loras=("style.safetensors",),
            upscale_models=("upscaler.pth",),
            available_node_classes=frozenset(),
            warnings=(),
        )
        return ComfyUIStatus(
            is_connected=True,
            message="接続できます",
            checked_at=datetime.now(UTC),
            system_stats=ComfyUISystemStats("linux", "3.12", False, "0.3.30", ()),
            capabilities=capabilities,
            warnings=(),
            error_summary=None,
        )

    async def refresh_capabilities(self) -> CapabilityRefreshResult:
        return CapabilityRefreshResult(False, "更新に失敗しました", None)


class FakeLifecycleService:
    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.session = SimpleNamespace(
            status=AutoTerminateState.DRAINING,
            auto_terminate_enabled=True,
        )

    async def drain_backup_and_terminate(self, *, require_armed: bool) -> TerminateReadiness:
        self.calls.append(require_armed)
        return TerminateReadiness(
            is_safe=True,
            checked_at=datetime.now(UTC),
            generation_ready=True,
            comfyui_ready=True,
            model_transfer_ready=True,
            drive_sync_ready=True,
            manifest_ready=True,
            state_backup_ready=True,
            runpod_identity_ready=True,
        )


@pytest.mark.asyncio
async def test_system_handler_uses_service_and_returns_dropdown_updates() -> None:
    service = FakeService()
    with gr.Blocks():
        generation = build_generation_tab()
        handler = make_check_connection_handler(service, "Asia/Tokyo", generation)

        result = await handler(None, None, None, None, None)

    assert service.called is True
    assert "接続" in result[0]
    assert result[2].choices
    assert result[2].choices[0][0] == "model.safetensors"
    assert result[6].choices
    assert result[6].choices[0][0] == "upscaler.pth"
    assert "style.safetensors" in result[14]


@pytest.mark.asyncio
async def test_refresh_handler_keeps_app_alive_on_service_failure() -> None:
    service = FakeService()
    with gr.Blocks():
        generation = build_generation_tab()
        handler = make_refresh_handler(service, generation)

        result = await handler(None, None, None, None, None)

    assert result[0] == "更新に失敗しました"
    assert all(
        isinstance(update, dict) and update.get("__type__") == "update" for update in result[1:]
    )
    assert len(result[1:]) == len(capability_refresh_outputs(generation))


@pytest.mark.asyncio
async def test_preserve_output_shape_follows_configured_editor_rows() -> None:
    service = FakeService()
    with gr.Blocks():
        generation = build_generation_tab(max_loras=3)
        handler = make_refresh_handler(service, generation)
        result = await handler(None, None, None, None, None)

    assert len(result[1:]) == len(capability_refresh_outputs(generation))


@pytest.mark.asyncio
async def test_manual_terminate_handler_uses_common_drain_backup_path() -> None:
    service = FakeLifecycleService()
    result = await make_pod_lifecycle_terminate_handler(service)()

    assert service.calls == [False]
    assert "SAFE TO TERMINATE" in result[0]
    assert result[1] == "Termination request accepted"
