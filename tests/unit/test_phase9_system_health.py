from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIDeviceInfo,
    ComfyUISystemStats,
)
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.system_error_repository import (
    SystemErrorEventRepository,
    sanitize_error_text,
)
from runpod_sdxl_image_studio.adapters.storage.disk_usage import DiskUsage
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import DriveSyncStatus
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.preflight import (
    PreflightIssue,
    PreflightResult,
    PreflightSeverity,
)
from runpod_sdxl_image_studio.domain.system_status import (
    ComfyUIStatus,
    DriveHealthView,
    SystemHealthStatus,
)
from runpod_sdxl_image_studio.services.generation_preflight_service import (
    GenerationPreflightService,
)
from runpod_sdxl_image_studio.services.system_health_service import SystemHealthService
from runpod_sdxl_image_studio.ui.tabs.system_tab import make_enqueue_handler

NOW = datetime(2026, 8, 9, 12, tzinfo=UTC)
REQUIRED_NODES = frozenset(
    {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
        "LoraLoader",
        "VAELoader",
    }
)


def _capabilities(*, missing: set[str] | None = None) -> ComfyUICapabilities:
    available = REQUIRED_NODES - (missing or set())
    return ComfyUICapabilities(
        checkpoints=("checkpoint.safetensors",),
        vaes=("vae.safetensors",),
        samplers=("euler",),
        schedulers=("normal",),
        loras=("style.safetensors",),
        upscale_models=("4x.pth",),
        available_node_classes=frozenset(available),
        warnings=(),
    )


class _FakeComfyUI:
    def __init__(self, *, connected: bool = True, missing_nodes: set[str] | None = None) -> None:
        self.status = ComfyUIStatus(
            is_connected=connected,
            message="connected" if connected else "ComfyUI unavailable",
            checked_at=NOW,
            system_stats=ComfyUISystemStats(
                "linux",
                "3.11",
                False,
                "0.3.30",
                (ComfyUIDeviceInfo("RTX", "cuda", 0, 16_000, 8_000, None, None),),
            ),
            capabilities=_capabilities(missing=missing_nodes) if connected else None,
            warnings=(),
            error_summary=None if connected else "connection failed",
        )

    async def get_status(self) -> ComfyUIStatus:
        return self.status


class _FakeDisk:
    def __init__(self, usage: DiskUsage) -> None:
        self.usage_value = usage
        self.calls = 0

    def usage(self, _path: Path) -> DiskUsage:
        self.calls += 1
        return self.usage_value


class _FakeDrive:
    is_configured = True

    async def check_connection(self) -> object:
        return SimpleNamespace(status=SimpleNamespace(value="connected"))

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        return {
            DriveSyncStatus.PENDING: 2,
            DriveSyncStatus.SYNCED: 1,
            DriveSyncStatus.FAILED: 1,
        }

    def list_jobs(self, _limit: int = 50) -> tuple[object, ...]:
        return (
            SimpleNamespace(
                status=DriveSyncStatus.SYNCED,
                completed_at=NOW,
                generation_id=uuid4(),
            ),
        )

    def capacity(self) -> object:
        return SimpleNamespace(unsynced_bytes=1234)


class _FakeQueue:
    def list_jobs(self, *, limit: int = 200) -> tuple[object, ...]:
        del limit
        return tuple(
            SimpleNamespace(generation=SimpleNamespace(status=status))
            for status in (
                GenerationStatus.PENDING,
                GenerationStatus.RUNNING,
                GenerationStatus.FAILED,
            )
        )


def _settings(tmp_path: Path, **updates: object) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        min_free_disk_bytes=100,
        warning_free_disk_bytes=500,
        **updates,
    )


def _generation_settings(**updates: object) -> GenerationSettings:
    values: dict[str, object] = {
        "positive_prompt": "portrait",
        "negative_prompt": "",
        "checkpoint_name": "checkpoint.safetensors",
        "sampler_name": "euler",
        "scheduler_name": "normal",
        "vae_name": None,
        "loras": (),
        "width": 1024,
        "height": 1024,
        "seed": 1,
        "steps": 28,
        "cfg_scale": 5.5,
    }
    values.update(updates)
    return GenerationSettings(**values)


@pytest.mark.asyncio
async def test_system_health_aggregates_comfy_queue_storage_drive_and_models(
    tmp_path: Path,
) -> None:
    service = SystemHealthService(
        _FakeComfyUI(),
        _FakeQueue(),
        _FakeDrive(),
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        now_factory=lambda: NOW,
    )

    view = await service.get_health()

    assert view.overall_status is SystemHealthStatus.WARNING
    assert view.comfyui_connected is True
    assert view.comfyui_version == "0.3.30"
    assert view.gpu_name == "RTX"
    assert view.vram_total == 16_000
    assert view.vram_free == 8_000
    assert (view.pending_count, view.running_count, view.failed_count) == (1, 1, 1)
    assert view.local_used_bytes == 700
    assert view.unsynced_bytes == 1234
    assert view.drive == DriveHealthView(True, True, NOW, 2, 1)
    assert (view.checkpoint_count, view.lora_count, view.vae_count, view.upscaler_count) == (
        1,
        1,
        1,
        1,
    )


@pytest.mark.asyncio
async def test_disconnected_comfyui_is_an_error(tmp_path: Path) -> None:
    service = SystemHealthService(
        _FakeComfyUI(connected=False),
        _FakeQueue(),
        None,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
    )

    view = await service.get_health()

    assert view.overall_status is SystemHealthStatus.ERROR
    assert view.comfyui.connected is False
    assert view.models.checkpoint_count == 0


@pytest.mark.asyncio
async def test_capability_snapshot_failure_is_an_error(tmp_path: Path) -> None:
    comfyui = _FakeComfyUI()
    comfyui.status = replace(
        comfyui.status,
        capabilities=None,
        error_summary="capability refresh failed",
    )
    service = SystemHealthService(
        comfyui,
        _FakeQueue(),
        None,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
    )

    view = await service.get_health()

    assert view.overall_status is SystemHealthStatus.ERROR
    assert (view.checkpoint_count, view.lora_count, view.vae_count, view.upscaler_count) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "ready", "code"),
    [
        (DiskUsage(1000, 200, 800), True, None),
        (DiskUsage(1000, 700, 300), True, None),
        (DiskUsage(1000, 950, 50), False, "disk_space_critical"),
    ],
)
async def test_preflight_disk_thresholds(
    tmp_path: Path,
    usage: DiskUsage,
    ready: bool,
    code: str | None,
) -> None:
    disk = _FakeDisk(usage)
    service = GenerationPreflightService(
        _FakeComfyUI(),
        _settings(tmp_path),
        disk_usage_adapter=disk,
        workflow_template={"required_node_classes": sorted(REQUIRED_NODES)},
    )

    result = await service.check(_generation_settings())

    assert result.is_ready is ready
    assert disk.calls == 1
    if code is None:
        assert any(issue.code == "disk_space_low" for issue in result.warnings) is (
            usage.free_bytes < 500
        )
    else:
        assert code in {issue.code for issue in result.errors}


@pytest.mark.asyncio
async def test_preflight_detects_missing_models_and_required_nodes(tmp_path: Path) -> None:
    service = GenerationPreflightService(
        _FakeComfyUI(missing_nodes={"LoraLoader"}),
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 800)),
        workflow_template={"required_node_classes": sorted(REQUIRED_NODES)},
    )

    result = await service.check(
        _generation_settings(
            checkpoint_name="missing.safetensors",
            vae_name="missing.vae",
            loras=(LoraSetting(name="missing.safetensors"),),
        ),
        uses_upscaler=True,
        upscaler_name="missing.pth",
    )

    assert result.is_ready is False
    assert {
        "checkpoint_missing",
        "vae_missing",
        "lora_missing",
        "upscaler_missing",
        "required_node_missing",
    }.issubset({issue.code for issue in result.errors})


@pytest.mark.asyncio
async def test_preflight_warning_is_ready_and_drive_is_not_a_hard_stop(tmp_path: Path) -> None:
    class _DisconnectedDrive:
        async def check_connection(self) -> object:
            return SimpleNamespace(status=SimpleNamespace(value="failed"))

    service = GenerationPreflightService(
        _FakeComfyUI(),
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        workflow_template={"required_node_classes": sorted(REQUIRED_NODES)},
        drive_status_provider=_DisconnectedDrive().check_connection,
    )

    result = await service.check(_generation_settings())

    assert result.is_ready is True
    assert any(issue.code == "disk_space_low" for issue in result.warnings)
    assert any(issue.code == "drive_not_connected" for issue in result.warnings)


class _FakeEnqueue:
    def __init__(self) -> None:
        self.calls = 0

    def enqueue(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        return SimpleNamespace(
            item=SimpleNamespace(
                generation=SimpleNamespace(
                    id=uuid4(),
                    settings_snapshot=SimpleNamespace(seed=1),
                )
            ),
            queue_position=1,
        )

    def get_job_detail(self, _generation_id: object) -> None:
        return None


class _FakePreflight:
    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        self.calls = 0

    async def check(self, _settings: GenerationSettings) -> PreflightResult:
        self.calls += 1
        return self.result


def _preflight_inputs() -> tuple[object, ...]:
    return (
        "checkpoint.safetensors",
        "positive",
        "negative",
        "1024x1024",
        1024,
        1024,
        "Fixed",
        1,
        28,
        5.5,
        "euler",
        "normal",
        None,
        None,
        None,
        False,
        False,
    )


@pytest.mark.asyncio
async def test_preflight_error_does_not_enqueue_and_warning_does_enqueue() -> None:
    blocked = PreflightResult(
        False,
        (PreflightIssue("checkpoint_missing", "checkpoint missing", PreflightSeverity.ERROR),),
        (),
        NOW,
    )
    queue = _FakeEnqueue()
    preflight = _FakePreflight(blocked)
    handler = make_enqueue_handler(queue, 2, preflight)  # type: ignore[arg-type]

    result = await handler(*_preflight_inputs())

    assert preflight.calls == 1
    assert queue.calls == 0
    assert "checkpoint_missing" in result[3]

    warning = PreflightResult(
        True,
        (),
        (PreflightIssue("disk_space_low", "disk is getting low", PreflightSeverity.WARNING),),
        NOW,
    )
    warning_preflight = _FakePreflight(warning)
    warning_handler = make_enqueue_handler(queue, 2, warning_preflight)  # type: ignore[arg-type]
    warning_result = await warning_handler(*_preflight_inputs())

    assert warning_preflight.calls == 1
    assert queue.calls == 1
    assert warning_result[1] == "Queued"
    assert "disk_space_low" in warning_result[3]


def test_system_error_repository_sanitizes_and_limits_history(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'errors.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    repository = SystemErrorEventRepository(create_session_factory(engine))

    event = repository.record(
        category="preflight",
        severity="error",
        error_code="disk_space_critical",
        summary="token=secret-value C:\\private\\secret.log\x1b[31m",
        details="A" * 20_000,
        created_at=NOW,
    )
    listed = repository.list_recent()

    assert len(listed) == 1
    assert listed[0].id == event.id
    assert "secret-value" not in listed[0].summary
    assert "C:\\private" not in listed[0].summary
    assert len(listed[0].details or "") <= 2_000
    assert sanitize_error_text("\x1b[31mwarning\x1b[0m", max_length=100) == "warning"
    engine.dispose()
