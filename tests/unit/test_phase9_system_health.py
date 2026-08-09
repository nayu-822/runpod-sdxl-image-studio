from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, update

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIDeviceInfo,
    ComfyUISystemStats,
)
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import (
    Base,
    DriveSyncJobModel,
    DriveSyncRecordModel,
    GenerationArtifactModel,
    GenerationJobModel,
    GenerationModel,
)
from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.system_error_repository import (
    SystemErrorEventRepository,
    sanitize_error_text,
)
from runpod_sdxl_image_studio.adapters.storage.disk_usage import DiskUsage
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import DriveSyncStatus
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import BatchSeedStrategy, QueueHealthCounts
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.preflight import (
    PreflightIssue,
    PreflightResult,
    PreflightSeverity,
)
from runpod_sdxl_image_studio.domain.system_status import (
    ComfyUIStatus,
    DriveHealthAvailability,
    DriveHealthView,
    QueueHealthAvailability,
    SystemHealthStatus,
)
from runpod_sdxl_image_studio.services.generation_preflight_service import (
    GenerationPreflightService,
)
from runpod_sdxl_image_studio.services.system_health_service import SystemHealthService
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    make_batch_enqueue_handler,
    make_enqueue_handler,
)
from runpod_sdxl_image_studio.ui.tabs.upscale_tab import make_upscale_enqueue_details_handler
from runpod_sdxl_image_studio.ui.view_models import system_health_markdown

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

    def __init__(self) -> None:
        self.latest_synced_job = SimpleNamespace(
            status=DriveSyncStatus.SYNCED,
            completed_at=NOW,
            updated_at=NOW,
            generation_id=uuid4(),
        )

    async def check_connection(self) -> object:
        return SimpleNamespace(status=SimpleNamespace(value="connected"))

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        return {
            DriveSyncStatus.PENDING: 2,
            DriveSyncStatus.SYNCED: 1,
            DriveSyncStatus.FAILED: 1,
        }

    def list_jobs(self, _limit: int = 50) -> tuple[object, ...]:
        raise AssertionError("System Health must use get_latest_synced_job")

    def get_latest_synced_job(self) -> object:
        return self.latest_synced_job

    def get_latest_unresolved_failure(self) -> object | None:
        return None

    def capacity(self) -> object:
        return SimpleNamespace(unsynced_bytes=1234)


class _FakeQueue:
    def __init__(self) -> None:
        self.failed_generation_id = uuid4()
        self.failed_job_id = uuid4()

    def get_health_counts(self) -> QueueHealthCounts:
        return QueueHealthCounts(pending_count=1, running_count=1, failed_count=1)

    def list_recent_failed(self, limit: int = 100) -> tuple[object, ...]:
        assert limit == 100
        return (
            SimpleNamespace(
                generation=SimpleNamespace(
                    id=self.failed_generation_id,
                    status=GenerationStatus.FAILED,
                    updated_at=NOW,
                    error_code="generation_failed",
                    error_summary="generation failed",
                ),
                job=SimpleNamespace(id=self.failed_job_id),
            ),
        )


class _UnsafeFailedQueue(_FakeQueue):
    def list_recent_failed(self, limit: int = 100) -> tuple[object, ...]:
        assert limit == 100
        return (
            SimpleNamespace(
                generation=SimpleNamespace(
                    id=self.failed_generation_id,
                    status=GenerationStatus.FAILED,
                    updated_at=NOW,
                    error_code="generation_failed",
                    error_summary="Cookie: session=secret-from-generation",
                ),
                job=SimpleNamespace(id=self.failed_job_id),
            ),
        )


class _FailingQueue:
    def get_health_counts(self) -> QueueHealthCounts:
        raise RuntimeError("token=queue-secret")

    def list_recent_failed(self, _limit: int = 100) -> tuple[object, ...]:
        raise AssertionError("recent failures must not be read after count failure")


class _FailingComfyUI:
    async def get_status(self) -> ComfyUIStatus:
        raise RuntimeError("Authorization: Bearer comfy-secret")


class _FailingDisk:
    def usage(self, _path: Path) -> DiskUsage:
        raise RuntimeError("/workspace/private.log")


class _FailingDrive:
    is_configured = True

    async def check_connection(self) -> object:
        raise RuntimeError("Cookie: session=drive-secret")

    def status_counts(self) -> dict[str, int]:
        raise RuntimeError("RCLONE_CONFIG=/secret/path")

    def list_jobs(self, _limit: int = 50) -> tuple[object, ...]:
        raise RuntimeError("C:\\private\\secret.log")

    def get_latest_synced_job(self) -> object:
        raise RuntimeError("C:\\private\\secret.log")

    def capacity(self) -> object:
        raise RuntimeError("/mnt/data/private.log")


class _FailedDrive:
    is_configured = True

    def __init__(self) -> None:
        self.failure_at = NOW - timedelta(minutes=5)

    async def check_connection(self) -> object:
        return SimpleNamespace(status=SimpleNamespace(value="connected"))

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        return {DriveSyncStatus.FAILED: 1}

    def list_jobs(self, _limit: int = 50) -> tuple[object, ...]:
        return (
            SimpleNamespace(
                status=DriveSyncStatus.FAILED,
                updated_at=self.failure_at,
                completed_at=self.failure_at,
            ),
        )

    def get_latest_synced_job(self) -> object | None:
        return None

    def get_latest_unresolved_failure(self) -> object:
        return SimpleNamespace(updated_at=self.failure_at, completed_at=self.failure_at)

    def capacity(self) -> object:
        return SimpleNamespace(unsynced_bytes=100)


class _PartialDrive:
    is_configured = True

    def __init__(self, failed_read: str) -> None:
        self.failed_read = failed_read

    async def check_connection(self) -> object:
        if self.failed_read == "connection":
            raise RuntimeError("connection unavailable")
        return SimpleNamespace(status=SimpleNamespace(value="connected"))

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        if self.failed_read == "status":
            raise RuntimeError("status unavailable")
        return {DriveSyncStatus.PENDING: 2, DriveSyncStatus.FAILED: 1}

    def list_jobs(self, _limit: int = 50) -> tuple[object, ...]:
        raise AssertionError("System Health must not read a limited job history")

    def get_latest_synced_job(self) -> object:
        if self.failed_read == "history":
            raise RuntimeError("history unavailable")
        return SimpleNamespace(status=DriveSyncStatus.SYNCED, completed_at=NOW, updated_at=NOW)

    def get_latest_unresolved_failure(self) -> object | None:
        if self.failed_read == "history":
            raise RuntimeError("failure history unavailable")
        return None

    def capacity(self) -> object:
        if self.failed_read == "capacity":
            raise RuntimeError("capacity unavailable")
        return SimpleNamespace(unsynced_bytes=1234)


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
async def test_system_health_read_does_not_mark_state_dirty_without_persisted_error(
    tmp_path: Path,
) -> None:
    state_changes: list[str] = []
    service = SystemHealthService(
        _FakeComfyUI(),
        _FakeQueue(),
        _FakeDrive(),
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        now_factory=lambda: NOW,
        state_changed_callback=lambda: state_changes.append("changed"),
    )

    await service.get_health()

    assert state_changes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_read", "availability_field"),
    [
        ("connection", "connection_available"),
        ("status", "sync_status_available"),
        ("capacity", "capacity_available"),
        ("history", "job_history_available"),
    ],
)
async def test_drive_partial_health_read_is_warning_and_value_is_unavailable(
    tmp_path: Path,
    failed_read: str,
    availability_field: str,
) -> None:
    service = SystemHealthService(
        _FakeComfyUI(),
        _FakeQueue(),
        _PartialDrive(failed_read),
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        now_factory=lambda: NOW,
    )

    view = await service.get_health()
    markdown = system_health_markdown(view, "Asia/Tokyo")

    assert view.overall_status is SystemHealthStatus.WARNING
    assert getattr(view.drive, availability_field) is DriveHealthAvailability.UNAVAILABLE
    assert "unavailable" in markdown
    if failed_read == "status":
        assert view.pending_sync_count is None
        assert view.failed_sync_count is None
    elif failed_read == "connection":
        assert "connected `unavailable`" in markdown
    elif failed_read == "capacity":
        assert view.unsynced_bytes is None
        assert "unsynced `unavailable`" in markdown
    else:
        assert "last sync `unavailable`" in markdown


@pytest.mark.asyncio
async def test_system_health_queue_counts_cover_all_rows_and_recent_errors_are_failed_only(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'queue-health.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    repository = GenerationDispatchQueueRepository(session_factory)
    snapshot = GenerationSettingsSnapshot.from_settings(_generation_settings())
    items = tuple(
        repository.enqueue_single(
            snapshot.model_copy(update={"seed": index}),
            enqueued_at=NOW + timedelta(seconds=index),
        )
        for index in range(250)
    )
    statuses = [GenerationStatus.COMPLETED] * 247
    statuses.extend((GenerationStatus.FAILED, GenerationStatus.RUNNING, GenerationStatus.PENDING))
    with session_factory() as session:
        for index, item in enumerate(items):
            updated_at = NOW + timedelta(seconds=index)
            values = {
                "status": statuses[index].value,
                "updated_at": updated_at,
                "completed_at": updated_at
                if statuses[index] is GenerationStatus.COMPLETED
                else None,
            }
            session.execute(
                update(GenerationModel)
                .where(GenerationModel.id == str(item.generation.id))
                .values(**values)
            )
            session.execute(
                update(GenerationJobModel)
                .where(GenerationJobModel.id == str(item.job.id))
                .values(status=statuses[index].value, updated_at=updated_at)
            )
        session.commit()

    service = SystemHealthService(
        _FakeComfyUI(),
        repository,
        None,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
    )
    view = await service.get_health()

    assert view.queue_available is QueueHealthAvailability.AVAILABLE
    assert (view.pending_count, view.running_count, view.failed_count) == (1, 1, 1)
    assert any(
        event.generation_id == items[247].generation.id and event.error_code == "generation_failed"
        for event in view.recent_errors
    )
    assert all(event.generation_id != items[0].generation.id for event in view.recent_errors)
    engine.dispose()


def test_queue_health_separates_historical_failures_from_unresolved_retry_chains(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'queue-retries.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    repository = GenerationDispatchQueueRepository(create_session_factory(engine))
    snapshot = GenerationSettingsSnapshot.from_settings(_generation_settings(seed=10))
    original = repository.enqueue_single(snapshot, enqueued_at=NOW)

    with create_session_factory(engine)() as session:
        session.execute(
            update(GenerationModel)
            .where(GenerationModel.id == str(original.generation.id))
            .values(status=GenerationStatus.FAILED.value, updated_at=NOW)
        )
        session.execute(
            update(GenerationJobModel)
            .where(GenerationJobModel.id == str(original.job.id))
            .values(status=GenerationStatus.FAILED.value, updated_at=NOW)
        )
        session.commit()

    counts = repository.get_health_counts()
    assert (counts.historical_failed_count, counts.unresolved_failed_count) == (1, 1)

    retry = repository.enqueue_single(
        snapshot,
        retry_of_generation_id=original.generation.id,
        retry_attempt=1,
        enqueued_at=NOW + timedelta(seconds=1),
    )
    counts = repository.get_health_counts()
    assert (counts.historical_failed_count, counts.unresolved_failed_count) == (1, 0)

    with create_session_factory(engine)() as session:
        session.execute(
            update(GenerationModel)
            .where(GenerationModel.id == str(retry.generation.id))
            .values(status=GenerationStatus.FAILED.value, updated_at=NOW + timedelta(seconds=2))
        )
        session.execute(
            update(GenerationJobModel)
            .where(GenerationJobModel.id == str(retry.job.id))
            .values(status=GenerationStatus.FAILED.value, updated_at=NOW + timedelta(seconds=2))
        )
        session.commit()

    retry_again = repository.enqueue_single(
        snapshot,
        retry_of_generation_id=retry.generation.id,
        retry_attempt=2,
        enqueued_at=NOW + timedelta(seconds=3),
    )
    with create_session_factory(engine)() as session:
        session.execute(
            update(GenerationModel)
            .where(GenerationModel.id == str(retry_again.generation.id))
            .values(status=GenerationStatus.COMPLETED.value, updated_at=NOW + timedelta(seconds=4))
        )
        session.execute(
            update(GenerationJobModel)
            .where(GenerationJobModel.id == str(retry_again.job.id))
            .values(status=GenerationStatus.COMPLETED.value, updated_at=NOW + timedelta(seconds=4))
        )
        session.commit()

    counts = repository.get_health_counts()
    assert (counts.historical_failed_count, counts.unresolved_failed_count) == (2, 0)
    recent = repository.list_recent_failed(100)
    assert {item.generation.id for item in recent} == {
        original.generation.id,
        retry.generation.id,
    }

    batch, batch_items = repository.enqueue_batch(
        (snapshot, snapshot.model_copy(update={"seed": 11})),
        name="Phase 9 batch",
        seed_strategy=BatchSeedStrategy.SEQUENTIAL,
        start_seed=10,
        seed_step=1,
        enqueued_at=NOW,
    )
    with create_session_factory(engine)() as session:
        for item in batch_items:
            session.execute(
                update(GenerationModel)
                .where(GenerationModel.id == str(item.generation.id))
                .values(status=GenerationStatus.FAILED.value, updated_at=NOW)
            )
            session.execute(
                update(GenerationJobModel)
                .where(GenerationJobModel.id == str(item.job.id))
                .values(status=GenerationStatus.FAILED.value, updated_at=NOW)
            )
        session.commit()
    batch_retry, batch_retry_items = repository.enqueue_batch(
        tuple(item.generation.settings_snapshot for item in batch_items),
        name="Phase 9 batch retry",
        seed_strategy=BatchSeedStrategy.SEQUENTIAL,
        start_seed=10,
        seed_step=1,
        retry_of_batch_id=batch.id,
        retry_of_generations=tuple(item.generation.id for item in batch_items),
        retry_attempts=(1, 1),
        enqueued_at=NOW + timedelta(seconds=1),
    )
    assert batch_retry.id != batch.id
    with create_session_factory(engine)() as session:
        for item in batch_retry_items:
            session.execute(
                update(GenerationModel)
                .where(GenerationModel.id == str(item.generation.id))
                .values(status=GenerationStatus.COMPLETED.value, updated_at=NOW)
            )
            session.execute(
                update(GenerationJobModel)
                .where(GenerationJobModel.id == str(item.job.id))
                .values(status=GenerationStatus.COMPLETED.value, updated_at=NOW)
            )
        session.commit()
    counts = repository.get_health_counts()
    assert counts.unresolved_failed_count == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_queue_read_failure_is_unavailable_and_recorded_safely(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'health-errors.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    error_repository = SystemErrorEventRepository(create_session_factory(engine))
    state_changes: list[str] = []
    service = SystemHealthService(
        _FakeComfyUI(),
        _FailingQueue(),
        None,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        error_history_repository=error_repository,
        now_factory=lambda: NOW,
        state_changed_callback=lambda: state_changes.append("changed"),
    )

    view = await service.get_health()

    assert view.queue_available is QueueHealthAvailability.UNAVAILABLE
    assert view.overall_status is SystemHealthStatus.ERROR
    assert "**Queue:** unavailable" in system_health_markdown(view, "Asia/Tokyo")
    events = error_repository.list_recent()
    assert [event.error_code for event in events] == ["system_queue_status_failed"]
    assert "queue-secret" not in (events[0].summary + (events[0].details or ""))
    assert state_changes == ["changed"]
    engine.dispose()


@pytest.mark.asyncio
async def test_system_health_retrieval_failures_use_fixed_codes_and_dedupe(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'health-failures.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    error_repository = SystemErrorEventRepository(create_session_factory(engine))
    service = SystemHealthService(
        _FailingComfyUI(),
        _FailingQueue(),
        _FailingDrive(),
        _settings(tmp_path),
        disk_usage_adapter=_FailingDisk(),
        error_history_repository=error_repository,
        now_factory=lambda: NOW,
    )

    await service.get_health()
    await service.get_health()

    events = error_repository.list_recent()
    codes = {event.error_code for event in events}
    assert codes == {
        "system_comfyui_status_failed",
        "system_queue_status_failed",
        "system_disk_status_failed",
        "system_drive_connection_failed",
        "system_drive_status_failed",
        "system_drive_capacity_failed",
    }
    assert len(events) == len(codes)
    assert all("secret" not in (event.summary + (event.details or "")) for event in events)
    engine.dispose()


@pytest.mark.asyncio
async def test_synthetic_generation_error_history_is_sanitized_for_ui(tmp_path: Path) -> None:
    service = SystemHealthService(
        _FakeComfyUI(),
        _UnsafeFailedQueue(),
        None,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        now_factory=lambda: NOW,
    )

    view = await service.get_health()

    generation_event = next(event for event in view.recent_errors if event.category == "generation")
    assert "secret-from-generation" not in generation_event.summary


@pytest.mark.asyncio
async def test_drive_failed_event_uses_persisted_failure_time(tmp_path: Path) -> None:
    drive = _FailedDrive()
    service = SystemHealthService(
        _FakeComfyUI(),
        _FakeQueue(),
        drive,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        now_factory=lambda: NOW,
    )

    first = await service.get_health()
    second = await service.get_health()
    first_event = next(event for event in first.recent_errors if event.category == "drive_sync")
    second_event = next(event for event in second.recent_errors if event.category == "drive_sync")

    assert first_event.created_at == drive.failure_at
    assert second_event.created_at == drive.failure_at
    assert first.drive.last_failure_at == drive.failure_at
    assert second.drive.last_failure_at == drive.failure_at


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completed_at", "updated_at", "expected"),
    [
        (NOW - timedelta(days=2), NOW, NOW - timedelta(days=2)),
        (None, NOW - timedelta(days=3), NOW - timedelta(days=3)),
    ],
)
async def test_drive_last_sync_uses_latest_synced_job_time(
    tmp_path: Path,
    completed_at: datetime | None,
    updated_at: datetime,
    expected: datetime,
) -> None:
    drive = _FakeDrive()
    drive.latest_synced_job = SimpleNamespace(
        status=DriveSyncStatus.SYNCED,
        completed_at=completed_at,
        updated_at=updated_at,
    )
    service = SystemHealthService(
        _FakeComfyUI(),
        _FakeQueue(),
        drive,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
        now_factory=lambda: NOW,
    )

    view = await service.get_health()

    assert view.drive.last_sync_at == expected


def _drive_test_rows(
    *, index: int, status: str, updated_at: datetime, completed_at: datetime | None
) -> tuple[object, ...]:
    generation_id = uuid4()
    artifact_id = uuid4()
    record_id = uuid4()
    job_id = uuid4()
    is_failure = status == DriveSyncStatus.FAILED.value
    return (
        GenerationModel(
            id=str(generation_id),
            kind="standard",
            status=GenerationStatus.COMPLETED.value,
            settings_snapshot_json="{}",
            snapshot_schema_version=1,
            workflow_template_id="phase9-test",
            workflow_template_version="1",
            created_at=updated_at,
            updated_at=updated_at,
        ),
        GenerationArtifactModel(
            id=str(artifact_id),
            generation_id=str(generation_id),
            artifact_type="image",
            local_path=f"images/{index}.png",
            sha256=f"{index:064x}",
            size_bytes=1,
            mime_type="image/png",
            created_at=updated_at,
        ),
        DriveSyncRecordModel(
            id=str(record_id),
            generation_id=str(generation_id),
            status=status,
            remote_name="gdrive",
            remote_base_path="RunPod/Images",
            remote_image_path=f"RunPod/Images/{index}.png",
            remote_metadata_path=f"RunPod/Images/{index}.json",
            image_artifact_id=str(artifact_id),
            image_sha256=f"{index:064x}",
            image_size_bytes=1,
            synced_at=updated_at if status == DriveSyncStatus.SYNCED.value else None,
            error_code="drive_copy_failed" if is_failure else None,
            error_summary="failure" if is_failure else None,
            created_at=updated_at,
            updated_at=updated_at,
        ),
        DriveSyncJobModel(
            id=str(job_id),
            sync_record_id=str(record_id),
            generation_id=str(generation_id),
            queue_sequence=index + 1,
            status=status,
            completed_at=completed_at,
            error_code="drive_copy_failed" if is_failure else None,
            error_summary="failure" if is_failure else None,
            retryable=True,
            image_artifact_id=str(artifact_id),
            image_sha256=f"{index:064x}",
            image_size_bytes=1,
            created_at=updated_at,
            updated_at=updated_at,
        ),
    )


def test_drive_latest_synced_job_is_not_limited_by_recent_job_history(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'drive-latest-sync.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    sync_at = NOW - timedelta(days=1)
    rows: list[object] = []
    for index in range(102):
        status = (
            DriveSyncStatus.SYNCED.value
            if index == 0
            else (DriveSyncStatus.FAILED.value if index % 2 else DriveSyncStatus.PENDING.value)
        )
        updated_at = sync_at if index == 0 else NOW + timedelta(minutes=index)
        completed_at = sync_at if index == 0 else (updated_at if status == "failed" else None)
        rows.extend(
            _drive_test_rows(
                index=index,
                status=status,
                updated_at=updated_at,
                completed_at=completed_at,
            )
        )
    with factory() as session:
        session.add_all(rows)
        session.commit()

    latest = DriveSyncRepository(factory).get_latest_synced_job()

    assert latest is not None
    assert latest.completed_at == sync_at
    assert latest.updated_at == sync_at
    engine.dispose()


def test_drive_latest_synced_job_falls_back_to_updated_at(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'drive-latest-sync-fallback.sqlite3').as_posix()}"
    )
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    updated_at = NOW - timedelta(hours=4)
    with factory() as session:
        session.add_all(
            _drive_test_rows(
                index=0,
                status=DriveSyncStatus.SYNCED.value,
                updated_at=updated_at,
                completed_at=None,
            )
        )
        session.commit()

    latest = DriveSyncRepository(factory).get_latest_synced_job()

    assert latest is not None
    assert latest.completed_at is None
    assert latest.updated_at == updated_at
    engine.dispose()


def test_drive_latest_synced_job_is_kept_after_current_record_failure(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'drive-previous-sync.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    sync_at = NOW - timedelta(days=2)
    generation, artifact, record, job = _drive_test_rows(
        index=0,
        status=DriveSyncStatus.SYNCED.value,
        updated_at=sync_at,
        completed_at=sync_at,
    )
    record.status = DriveSyncStatus.FAILED.value  # type: ignore[attr-defined]
    record.error_code = "drive_copy_failed"  # type: ignore[attr-defined]
    with factory() as session:
        session.add_all((generation, artifact, record, job))
        session.commit()

    latest = DriveSyncRepository(factory).get_latest_synced_job()

    assert latest is not None
    assert latest.completed_at == sync_at
    with factory() as session:
        current_record = session.get(DriveSyncRecordModel, record.id)  # type: ignore[attr-defined]
        current_job = session.get(DriveSyncJobModel, job.id)  # type: ignore[attr-defined]
        assert current_record is not None
        assert current_record.status == DriveSyncStatus.FAILED.value
        assert current_job is not None
        assert current_job.status == DriveSyncStatus.SYNCED.value
    engine.dispose()


def test_drive_latest_unresolved_failure_is_not_limited_by_recent_job_history(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'drive-health.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    failure_at = NOW - timedelta(days=1)
    rows: list[object] = []
    for index in range(102):
        generation_id = uuid4()
        artifact_id = uuid4()
        record_id = uuid4()
        job_id = uuid4()
        is_failure = index == 0
        updated_at = failure_at if is_failure else NOW + timedelta(minutes=index)
        rows.extend(
            (
                GenerationModel(
                    id=str(generation_id),
                    kind="standard",
                    status=GenerationStatus.COMPLETED.value,
                    settings_snapshot_json="{}",
                    snapshot_schema_version=1,
                    workflow_template_id="phase9-test",
                    workflow_template_version="1",
                    created_at=updated_at,
                    updated_at=updated_at,
                ),
                GenerationArtifactModel(
                    id=str(artifact_id),
                    generation_id=str(generation_id),
                    artifact_type="image",
                    local_path=f"images/{index}.png",
                    sha256=f"{index:064x}",
                    size_bytes=1,
                    mime_type="image/png",
                    created_at=updated_at,
                ),
                DriveSyncRecordModel(
                    id=str(record_id),
                    generation_id=str(generation_id),
                    status="failed" if is_failure else "synced",
                    remote_name="gdrive",
                    remote_base_path="RunPod/Images",
                    remote_image_path=f"RunPod/Images/{index}.png",
                    remote_metadata_path=f"RunPod/Images/{index}.json",
                    image_artifact_id=str(artifact_id),
                    image_sha256=f"{index:064x}",
                    image_size_bytes=1,
                    synced_at=None if is_failure else updated_at,
                    error_code="drive_copy_failed" if is_failure else None,
                    error_summary="failure" if is_failure else None,
                    created_at=updated_at,
                    updated_at=updated_at,
                ),
                DriveSyncJobModel(
                    id=str(job_id),
                    sync_record_id=str(record_id),
                    generation_id=str(generation_id),
                    queue_sequence=index + 1,
                    status="failed" if is_failure else "synced",
                    completed_at=updated_at,
                    error_code="drive_copy_failed" if is_failure else None,
                    error_summary="failure" if is_failure else None,
                    retryable=True,
                    image_artifact_id=str(artifact_id),
                    image_sha256=f"{index:064x}",
                    image_size_bytes=1,
                    created_at=updated_at,
                    updated_at=updated_at,
                ),
            )
        )
    with factory() as session:
        session.add_all(rows)
        session.commit()

    repository = DriveSyncRepository(factory)
    jobs = repository.list_jobs(100)
    latest = repository.get_latest_unresolved_failure()

    assert all(job.status is not DriveSyncStatus.FAILED for job in jobs)
    assert repository.status_counts()[DriveSyncStatus.FAILED] == 1
    assert latest is not None
    assert latest.updated_at == failure_at
    engine.dispose()


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


class _FakeUpscaleEnqueue:
    def __init__(self) -> None:
        self.calls = 0

    def enqueue(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        return SimpleNamespace(
            generation=SimpleNamespace(
                id=uuid4(),
                settings_snapshot=SimpleNamespace(width=1024, height=1024),
            ),
            entry=SimpleNamespace(sequence=1),
        )

    def enqueue_import(self, *_args: object, **_kwargs: object) -> object:
        return self.enqueue()


class _FakeUpscalePreflight:
    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        self.calls = 0

    async def check_upscale(self, *_args: object, **_kwargs: object) -> PreflightResult:
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


@pytest.mark.asyncio
async def test_upscale_preflight_error_does_not_persist_and_warning_continues() -> None:
    blocked = PreflightResult(
        False,
        (PreflightIssue("upscaler_missing", "upscaler missing", PreflightSeverity.ERROR),),
        (),
        NOW,
    )
    queue = _FakeUpscaleEnqueue()
    preflight = _FakeUpscalePreflight(blocked)
    handler = make_upscale_enqueue_details_handler(queue, preflight)  # type: ignore[arg-type]
    inputs = ("", "image", "factor", 2.0, 1024, 1024, "4x.pth", None)

    blocked_result = await handler(*inputs)

    assert preflight.calls == 1
    assert queue.calls == 0
    assert "upscaler_missing" in blocked_result[1]
    assert blocked_result[0].interactive is True

    warning = PreflightResult(
        True,
        (),
        (PreflightIssue("disk_space_low", "disk is getting low", PreflightSeverity.WARNING),),
        NOW,
    )
    warning_preflight = _FakeUpscalePreflight(warning)
    warning_handler = make_upscale_enqueue_details_handler(
        queue,
        warning_preflight,  # type: ignore[arg-type]
    )
    warning_result = await warning_handler(*inputs)

    assert warning_preflight.calls == 1
    assert queue.calls == 1
    assert "disk_space_low" in warning_result[1]


@pytest.mark.asyncio
async def test_batch_preflight_error_restores_batch_action_label() -> None:
    blocked = PreflightResult(
        False,
        (PreflightIssue("checkpoint_missing", "checkpoint missing", PreflightSeverity.ERROR),),
        (),
        NOW,
    )
    handler = make_batch_enqueue_handler(  # type: ignore[arg-type]
        object(),
        2,
        _FakePreflight(blocked),
    )

    result = await handler(
        "checkpoint.safetensors",
        "positive",
        "negative",
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
        2,
        "random",
        1,
        1,
        "Batch",
    )

    assert result[0].value == "バッチをキューへ追加"
    assert "checkpoint_missing" in result[2]


@pytest.mark.asyncio
async def test_upscale_preflight_uses_workflow_specific_requirements(tmp_path: Path) -> None:
    image_nodes = frozenset(
        {"LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "SaveImage"}
    )
    image_capabilities = ComfyUICapabilities(
        checkpoints=(),
        vaes=(),
        samplers=(),
        schedulers=(),
        loras=(),
        upscale_models=("4x.pth",),
        available_node_classes=image_nodes,
        warnings=(),
    )
    comfyui = _FakeComfyUI()
    comfyui.status = replace(comfyui.status, capabilities=image_capabilities)
    service = GenerationPreflightService(
        comfyui,
        _settings(tmp_path),
        disk_usage_adapter=_FakeDisk(DiskUsage(1000, 700, 300)),
    )

    result = await service.check_upscale("image", upscaler_name="4x.pth")

    assert result.is_ready is True
    assert not any(issue.code == "checkpoint_missing" for issue in result.errors)

    latent_nodes = frozenset(
        {
            "LoadImage",
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "VAEEncode",
            "LatentUpscale",
            "KSampler",
            "VAEDecode",
            "SaveImage",
        }
    )
    latent_capabilities = replace(
        image_capabilities,
        checkpoints=("checkpoint.safetensors",),
        samplers=("euler",),
        schedulers=("normal",),
        available_node_classes=latent_nodes,
    )
    comfyui.status = replace(comfyui.status, capabilities=latent_capabilities)
    latent_result = await service.check_upscale(
        "latent",
        upscaler_name=None,
        source_settings=_generation_settings(),
    )

    assert latent_result.is_ready is True


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


@pytest.mark.parametrize(
    "value",
    [
        "Authorization: Bearer secret-value",
        "Bearer secret-value",
        "token=secret-value",
        "access_token=secret-value",
        "client_secret=secret-value",
        "Cookie: session=secret-value",
        "Set-Cookie: session=secret-value",
        "password=secret-value",
        "RCLONE_CONFIG=/secret/path",
        "rclone credential: secret-value",
        "/workspace/private.log",
        "/home/user/private.log",
        "/mnt/data/private.log",
        "/opt/app/private.log",
        "C:\\private\\secret.log",
    ],
)
def test_system_error_sanitizer_redacts_supported_secret_and_path_forms(value: str) -> None:
    sanitized = sanitize_error_text(value, max_length=500)

    assert sanitized is not None
    assert "secret-value" not in sanitized
    assert all(
        path not in sanitized
        for path in (
            "/workspace/private.log",
            "/home/user/private.log",
            "/mnt/data/private.log",
            "/opt/app/private.log",
            "C:\\private\\secret.log",
        )
    )


@pytest.mark.parametrize(
    "value",
    [
        "Authorization: Bearer secret-a",
        "Cookie: session=secret-a",
        "Cookie: session=secret-a; refresh=secret-b",
        "Cookie: a=one; b=two; c=three",
        "Set-Cookie: session=secret-a; HttpOnly; Secure",
        "Set-Cookie: a=one, b=two",
    ],
)
def test_system_error_sanitizer_redacts_entire_cookie_and_header_values(value: str) -> None:
    sanitized = sanitize_error_text(value, max_length=500)

    assert sanitized == "<redacted-secret>"
    assert all(secret not in (sanitized or "") for secret in ("secret-a", "secret-b"))


def test_system_error_repository_deduplicates_same_category_and_code_without_update(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'dedupe.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    repository = SystemErrorEventRepository(create_session_factory(engine))

    first = repository.record(
        category="system_health",
        severity="error",
        error_code="system_queue_status_failed",
        summary="fixed summary",
        created_at=NOW,
    )
    second = repository.record(
        category="system_health",
        severity="error",
        error_code="system_queue_status_failed",
        summary="a different safe summary",
        created_at=NOW + timedelta(seconds=1),
    )

    assert second.id == first.id
    listed = repository.list_recent()
    assert len(listed) == 1
    assert listed[0].summary == "fixed summary"
    engine.dispose()


def test_system_error_search_uses_only_generation_job_and_error_code_filters(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'error-search.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    repository = SystemErrorEventRepository(create_session_factory(engine))
    generation_id = uuid4()
    job_id = uuid4()
    event = repository.record(
        category="generation",
        severity="error",
        error_code="generation_failed",
        summary="safe summary",
        generation_id=generation_id,
        job_id=job_id,
        created_at=NOW,
    )

    assert repository.search(generation_id=generation_id) == (event,)
    assert repository.search(job_id=job_id) == (event,)
    assert repository.search(error_code="generation_failed") == (event,)
    assert repository.search(error_code="C:\\private\\secret.log") == ()
    engine.dispose()
