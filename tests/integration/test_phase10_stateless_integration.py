"""Phase 10 startup ordering tests with SQLite, Alembic, and local fake storage."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
    sqlite_database_path,
)
from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database
from runpod_sdxl_image_studio.domain.generation import GenerationKind
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.state_sync import StateRestoreStatus, StateSyncStatus
from runpod_sdxl_image_studio.services.state_restore_service import StateRestoreService
from runpod_sdxl_image_studio.services.state_snapshot_service import StateSnapshotService
from runpod_sdxl_image_studio.services.state_sync_service import StateSyncService
from runpod_sdxl_image_studio.services.stateless_reconciliation_service import (
    StatelessReconciliationService,
)
from runpod_sdxl_image_studio.ui.app_builder import ApplicationRuntime

ROOT = Path(__file__).parents[2]
NOW = datetime(2026, 8, 9, 3, 0, tzinfo=UTC)


class _LocalStateStorage:
    is_configured = True

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def upload(self, local_path: Path, relative_path: str) -> None:
        self.objects[relative_path] = local_path.read_bytes()

    async def download(self, relative_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[relative_path])


class _WorkerRuntime:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_url=f"sqlite:///{(tmp_path / 'database' / 'image_studio.sqlite3').as_posix()}",
        rclone_remote="drive",
        rclone_base_path="studio",
        state_sync_enabled=True,
        state_sync_debounce_seconds=60,
    )


def _snapshot() -> GenerationSettingsSnapshot:
    return GenerationSettingsSnapshot.from_settings(
        GenerationSettings(
            positive_prompt="a cat",
            negative_prompt="",
            seed=42,
            width=64,
            height=64,
            steps=20,
            cfg_scale=7,
            sampler_name="euler",
            scheduler_name="normal",
            checkpoint_name="sdxl.safetensors",
        )
    )


def test_startup_restore_migration_reconciliation_then_worker_start(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    upgrade_database(settings, ROOT)
    engine = create_image_studio_engine(settings)
    factory = create_session_factory(engine)
    GenerationStartRepository(factory).create_pending(
        _snapshot(),
        generation_id=uuid4(),
        job_id=uuid4(),
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=NOW,
    )
    storage = _LocalStateStorage()
    backup = StateSyncService(
        settings,
        storage=storage,  # type: ignore[arg-type]
        snapshot_service=StateSnapshotService(settings, now_factory=lambda: NOW),
        now_factory=lambda: NOW,
    )
    assert asyncio.run(backup.backup()).remote_sha256
    backup.close()
    engine.dispose()

    database_path = sqlite_database_path(settings)
    assert database_path is not None
    database_path.unlink()
    restored = asyncio.run(
        StateRestoreService(settings, storage=storage).restore_if_missing_async()  # type: ignore[arg-type]
    )
    assert restored.status is StateRestoreStatus.RESTORED
    assert restored.metadata is not None

    upgrade_database(settings, ROOT)
    restored_engine = create_image_studio_engine(settings)
    restored_factory = create_session_factory(restored_engine)
    restored_state_sync = StateSyncService(
        settings,
        storage=storage,  # type: ignore[arg-type]
        snapshot_service=StateSnapshotService(settings, now_factory=lambda: NOW),
        now_factory=lambda: NOW,
        initial_remote_sha256=restored.metadata.sha256,
    )
    reconciliation = StatelessReconciliationService(
        GenerationDispatchQueueRepository(restored_factory),
        DriveSyncRepository(restored_factory),
        now_factory=lambda: NOW,
        state_changed_callback=restored_state_sync.mark_dirty,
    )
    queue_runtime = _WorkerRuntime()
    drive_runtime = _WorkerRuntime()
    runtime = ApplicationRuntime(
        demo=object(),  # type: ignore[arg-type]
        queue_runtime=queue_runtime,  # type: ignore[arg-type]
        drive_sync_runtime=drive_runtime,  # type: ignore[arg-type]
        state_sync_service=restored_state_sync,
        stateless_reconciliation_service=reconciliation,
        run_stateless_reconciliation=True,
    )

    pre_start_result = reconciliation.reconcile()
    assert pre_start_result.is_success is True
    assert pre_start_result.generation_reconciled_count == 1
    reconciled_backup = asyncio.run(restored_state_sync.backup())
    assert reconciled_backup.status is StateSyncStatus.SYNCED
    assert reconciled_backup.remote_sha256 != restored.metadata.sha256
    runtime.start()

    assert queue_runtime.start_calls == 1
    assert drive_runtime.start_calls == 1
    result = reconciliation.reconcile()
    assert result.is_success is True
    assert result.generation_reconciled_count == 0
    runtime.stop()
    restored_engine.dispose()

    database_path.unlink()
    restored_again = asyncio.run(
        StateRestoreService(settings, storage=storage).restore_if_missing_async()  # type: ignore[arg-type]
    )
    assert restored_again.status is StateRestoreStatus.RESTORED
    restored_again_engine = create_image_studio_engine(settings)
    restored_again_factory = create_session_factory(restored_again_engine)
    second_reconciliation = StatelessReconciliationService(
        GenerationDispatchQueueRepository(restored_again_factory),
        DriveSyncRepository(restored_again_factory),
        now_factory=lambda: NOW,
    )
    second_result = second_reconciliation.reconcile()
    assert second_result.is_success is True
    assert second_result.generation_reconciled_count == 0
    assert second_result.drive_reconciled_count == 0
    restored_again_engine.dispose()


def test_restore_failure_leaves_remote_latest_unchanged(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    storage = _LocalStateStorage()
    storage.objects["latest.json"] = json.dumps(
        {
            "schema_version": 1,
            "filename": "backups/old.sqlite3",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "created_at": "2026-08-09T03:00:00Z",
        }
    ).encode()
    original_pointer = storage.objects["latest.json"]

    class _UnavailableStorage(_LocalStateStorage):
        async def download(self, relative_path: str, local_path: Path) -> None:
            del relative_path, local_path
            raise TimeoutError("simulated remote timeout")

    unavailable = _UnavailableStorage()
    unavailable.objects.update(storage.objects)
    result = asyncio.run(
        StateRestoreService(settings, storage=unavailable).restore_if_missing_async()  # type: ignore[arg-type]
    )

    assert result.status is StateRestoreStatus.UNAVAILABLE
    assert unavailable.objects["latest.json"] == original_pointer
    assert sqlite_database_path(settings) is not None
    assert not sqlite_database_path(settings).exists()  # type: ignore[union-attr]
