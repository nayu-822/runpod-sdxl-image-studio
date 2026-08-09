"""Phase 10 stateless state backup, restore, and reconciliation tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
    ensure_database_directory,
    sqlite_database_path,
)
from runpod_sdxl_image_studio.adapters.database.models import (
    Base,
    DriveManifestJobModel,
    DriveSyncJobModel,
    DriveSyncRecordModel,
    GenerationArtifactModel,
    GenerationJobModel,
    GenerationModel,
    GenerationQueueEntryModel,
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
from runpod_sdxl_image_studio.adapters.drive.google_drive_adapter import GoogleDriveAdapter
from runpod_sdxl_image_studio.adapters.rclone.state_backup_storage import StateBackupStorage
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import DriveDestination, DriveSyncStatus
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.state_sync import StateRestoreStatus, StateSyncStatus
from runpod_sdxl_image_studio.services.state_restore_service import StateRestoreService
from runpod_sdxl_image_studio.services.state_snapshot_service import (
    StateSnapshotError,
    StateSnapshotService,
)
from runpod_sdxl_image_studio.services.state_sync_service import StateSyncService
from runpod_sdxl_image_studio.ui.tabs.system_tab import make_state_backup_handler

NOW = datetime(2026, 8, 9, 3, 0, tzinfo=UTC)


class _FakeStateStorage:
    is_configured = True

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_calls: list[str] = []
        self.download_calls: list[str] = []
        self.fail_upload = False

    async def upload(self, local_path: Path, relative_path: str) -> None:
        self.upload_calls.append(relative_path)
        if self.fail_upload:
            raise OSError("simulated state upload failure")
        self.objects[relative_path] = local_path.read_bytes()

    async def download(self, relative_path: str, local_path: Path) -> None:
        self.download_calls.append(relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[relative_path])


class _TransferAdapter:
    def __init__(self) -> None:
        self.uploads: list[tuple[DriveDestination, str, float | None]] = []

    async def copy_file(
        self,
        local_path: Path,
        destination: DriveDestination,
        relative_remote_path: str,
        *,
        total_bytes: int = 0,
        current_artifact: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        del local_path, total_bytes, current_artifact
        self.uploads.append((destination, relative_remote_path, timeout_seconds))

    async def copy_from_remote(
        self,
        destination: DriveDestination,
        relative_remote_path: str,
        local_path: Path,
    ) -> None:
        del destination, relative_remote_path, local_path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    database_path = tmp_path / "database" / "image_studio.sqlite3"
    values: dict[str, object] = {
        "_env_file": None,
        "data_dir": tmp_path,
        "database_url": f"sqlite:///{database_path.as_posix()}",
        "rclone_remote": "drive",
        "rclone_base_path": "studio",
        "state_sync_enabled": True,
        "state_sync_debounce_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


def _database(settings: Settings):
    ensure_database_directory(settings)
    engine = create_image_studio_engine(settings)
    Base.metadata.create_all(engine)
    return engine, create_session_factory(engine)


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


def _create_state_database(settings: Settings):
    engine, factory = _database(settings)
    connection = sqlite3.connect(str(sqlite_database_path(settings)))
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS state_probe (value TEXT NOT NULL)")
        connection.execute("INSERT INTO state_probe(value) VALUES ('ok')")
        connection.commit()
    finally:
        connection.close()
    return engine, factory


def test_sqlite_snapshot_uses_backup_api_and_cleans_temp_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine, _ = _create_state_database(settings)
    service = StateSnapshotService(settings, now_factory=lambda: NOW)

    snapshot = service.create_snapshot()
    try:
        assert snapshot.path != sqlite_database_path(settings)
        assert snapshot.path.exists()
        assert snapshot.metadata.size_bytes == snapshot.path.stat().st_size
        assert snapshot.metadata.sha256 == hashlib.sha256(snapshot.path.read_bytes()).hexdigest()
        StateSnapshotService.verify_snapshot(snapshot.path)
    finally:
        snapshot.path.unlink(missing_ok=True)
        engine.dispose()

    assert not list((tmp_path / ".state-sync").glob("state-*.sqlite3"))


def test_invalid_sqlite_snapshot_is_removed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ensure_database_directory(settings)
    database_path = sqlite_database_path(settings)
    assert database_path is not None
    database_path.write_bytes(b"not a sqlite database")

    with pytest.raises(StateSnapshotError):
        StateSnapshotService(settings).create_snapshot()

    assert not list((tmp_path / ".state-sync").glob("state-*.sqlite3"))


def test_state_backup_uploads_snapshot_metadata_and_latest_pointer_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine, _ = _create_state_database(settings)
    storage = _FakeStateStorage()
    service = StateSyncService(
        settings,
        storage=storage,  # type: ignore[arg-type]
        snapshot_service=StateSnapshotService(settings, now_factory=lambda: NOW),
        now_factory=lambda: NOW,
    )

    first = asyncio.run(service.backup())
    assert first.status is StateSyncStatus.SYNCED
    assert storage.upload_calls == [
        "backups/20260809T030000Z.sqlite3",
        "backups/20260809T030000Z.sqlite3.metadata.json",
        "latest.json",
    ]
    pointer = json.loads(storage.objects["latest.json"])
    assert pointer["filename"] == "backups/20260809T030000Z.sqlite3"
    assert pointer["sha256"] == first.remote_sha256
    assert pointer["size_bytes"] == first.remote_size_bytes

    second = asyncio.run(service.backup())
    assert second.status is StateSyncStatus.SYNCED
    assert len(storage.upload_calls) == 3
    assert "省略" in second.last_message
    assert not list((tmp_path / ".state-sync").glob("state-*.json"))
    engine.dispose()


def test_state_backup_failure_does_not_leave_local_snapshot(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine, _ = _create_state_database(settings)
    storage = _FakeStateStorage()
    storage.fail_upload = True
    service = StateSyncService(
        settings,
        storage=storage,  # type: ignore[arg-type]
        snapshot_service=StateSnapshotService(settings, now_factory=lambda: NOW),
        now_factory=lambda: NOW,
    )

    result = asyncio.run(service.backup())
    assert result.status is StateSyncStatus.FAILED
    assert storage.upload_calls == ["backups/20260809T030000Z.sqlite3"]
    assert not list((tmp_path / ".state-sync").glob("*"))
    engine.dispose()


def test_state_backup_storage_uses_state_subdir_and_upload_timeout(tmp_path: Path) -> None:
    settings = _settings(tmp_path, state_sync_subdir="state-backups")
    adapter = _TransferAdapter()
    local_path = tmp_path / "snapshot.sqlite3"
    local_path.write_bytes(b"snapshot")

    asyncio.run(StateBackupStorage(settings, adapter).upload(local_path, "backups/a.sqlite3"))

    assert adapter.uploads == [
        (
            DriveDestination("drive", "studio"),
            "state-backups/backups/a.sqlite3",
            settings.state_sync_upload_timeout_seconds,
        )
    ]


def test_restore_only_when_local_database_is_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine, _ = _create_state_database(settings)
    storage = _FakeStateStorage()
    backup_service = StateSyncService(
        settings,
        storage=storage,  # type: ignore[arg-type]
        snapshot_service=StateSnapshotService(settings, now_factory=lambda: NOW),
        now_factory=lambda: NOW,
    )
    asyncio.run(backup_service.backup())
    engine.dispose()

    database_path = sqlite_database_path(settings)
    assert database_path is not None
    database_path.unlink()
    restore = StateRestoreService(settings, storage=storage)  # type: ignore[arg-type]
    result = asyncio.run(restore.restore_if_missing_async())
    assert result.status is StateRestoreStatus.RESTORED
    assert database_path.exists()
    StateSnapshotService.verify_snapshot(database_path)

    calls_before = len(storage.download_calls)
    skipped = asyncio.run(restore.restore_if_missing_async())
    assert skipped.status is StateRestoreStatus.SKIPPED_LOCAL
    assert len(storage.download_calls) == calls_before


def test_restore_without_latest_pointer_creates_no_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    storage = _FakeStateStorage()
    restore = StateRestoreService(settings, storage=storage)  # type: ignore[arg-type]

    result = asyncio.run(restore.restore_if_missing_async())
    assert result.status is StateRestoreStatus.NO_BACKUP
    assert not sqlite_database_path(settings).exists()  # type: ignore[union-attr]


def test_restore_rejects_backup_hash_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine, _ = _create_state_database(settings)
    storage = _FakeStateStorage()
    backup_service = StateSyncService(
        settings,
        storage=storage,  # type: ignore[arg-type]
        snapshot_service=StateSnapshotService(settings, now_factory=lambda: NOW),
        now_factory=lambda: NOW,
    )
    asyncio.run(backup_service.backup())
    engine.dispose()
    database_path = sqlite_database_path(settings)
    assert database_path is not None
    database_path.unlink()
    storage.objects["backups/20260809T030000Z.sqlite3"] = b"tampered"

    result = asyncio.run(
        StateRestoreService(settings, storage=storage).restore_if_missing_async()  # type: ignore[arg-type]
    )
    assert result.status is StateRestoreStatus.FAILED
    assert not database_path.exists()


def test_restore_missing_backup_body_fails_without_creating_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    storage = _FakeStateStorage()
    storage.objects["latest.json"] = json.dumps(
        {
            "schema_version": 1,
            "filename": "backups/missing.sqlite3",
            "sha256": "0" * 64,
            "size_bytes": 1,
            "created_at": "2026-08-09T03:00:00Z",
        }
    ).encode()

    result = asyncio.run(
        StateRestoreService(settings, storage=storage).restore_if_missing_async()  # type: ignore[arg-type]
    )
    assert result.status is StateRestoreStatus.FAILED
    assert not sqlite_database_path(settings).exists()  # type: ignore[union-attr]


def test_manual_state_backup_handler_returns_markdown_and_message(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine, _ = _create_state_database(settings)
    storage = _FakeStateStorage()
    service = StateSyncService(
        settings,
        storage=storage,  # type: ignore[arg-type]
        snapshot_service=StateSnapshotService(settings, now_factory=lambda: NOW),
        now_factory=lambda: NOW,
    )

    markdown, message = asyncio.run(make_state_backup_handler(service, "Asia/Tokyo")())
    assert "State backup status" in markdown
    assert "`synced`" in markdown
    assert "状態バックアップが完了しました" in message
    engine.dispose()


def _create_generation_pair(factory, *, status: str = GenerationStatus.RUNNING.value):
    generation_id, job_id = uuid4(), uuid4()
    GenerationStartRepository(factory).create_pending(
        _snapshot(),
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=NOW,
    )
    with factory() as session:
        generation = session.get(GenerationModel, str(generation_id))
        job = session.get(GenerationJobModel, str(job_id))
        assert generation is not None and job is not None
        generation.status = status
        generation.comfy_prompt_id = "prompt-old"
        generation.updated_at = NOW
        job.status = status
        job.comfy_prompt_id = "prompt-old"
        job.worker_id = "worker-old"
        job.claimed_at = NOW
        job.lease_expires_at = NOW
        job.updated_at = NOW
        session.add(
            GenerationQueueEntryModel(
                generation_id=str(generation_id),
                job_id=str(job_id),
                worker_id="worker-old",
                claimed_at=NOW,
                lease_expires_at=NOW,
                cancel_requested_at=NOW,
                submission_state="submitted",
                submission_token="token-old",
                enqueued_at=NOW,
                updated_at=NOW,
            )
        )
        session.commit()
    return generation_id, job_id


@pytest.mark.parametrize(
    "status",
    [
        GenerationStatus.PENDING.value,
        GenerationStatus.QUEUED.value,
        GenerationStatus.RUNNING.value,
    ],
)
def test_stateless_generation_reconciliation_fails_active_work_without_resubmission(
    tmp_path: Path, status: str
) -> None:
    settings = _settings(tmp_path)
    engine, factory = _database(settings)
    generation_id, job_id = _create_generation_pair(factory, status=status)
    repository = GenerationDispatchQueueRepository(factory)

    assert repository.reconcile_stateless_restore(now=NOW) == 1
    with factory() as session:
        generation = session.get(GenerationModel, str(generation_id))
        job = session.get(GenerationJobModel, str(job_id))
        entry = session.scalar(
            select(GenerationQueueEntryModel).where(
                GenerationQueueEntryModel.generation_id == str(generation_id)
            )
        )
        assert generation is not None and job is not None and entry is not None
        assert generation.status == GenerationStatus.FAILED.value
        assert job.status == GenerationStatus.FAILED.value
        assert generation.error_code == "stateless_restore_interrupted"
        assert job.error_code == "stateless_restore_interrupted"
        assert generation.comfy_prompt_id == "prompt-old"
        assert job.comfy_prompt_id == "prompt-old"
        assert generation.completed_at is not None
        assert job.completed_at is not None
        assert generation.completed_at.replace(tzinfo=UTC) == NOW
        assert job.completed_at.replace(tzinfo=UTC) == NOW
        assert job.worker_id is None and job.claimed_at is None and job.lease_expires_at is None
        assert entry.worker_id is None and entry.claimed_at is None
        assert entry.lease_expires_at is None
        assert entry.cancel_requested_at is not None
        assert entry.cancel_requested_at.replace(tzinfo=UTC) == NOW
        assert entry.submission_state == "submitted"

    assert repository.reconcile_stateless_restore(now=NOW) == 0
    engine.dispose()


def test_stateless_drive_reconciliation_fails_jobs_and_manifest_jobs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine, factory = _database(settings)
    generation_id, job_id = _create_generation_pair(factory)
    record_id, artifact_id, drive_job_id = uuid4(), uuid4(), uuid4()
    with factory() as session:
        session.add(
            GenerationArtifactModel(
                id=str(artifact_id),
                generation_id=str(generation_id),
                artifact_type="image",
                local_path="generations/image.png",
                sha256="a" * 64,
                size_bytes=3,
                width=1,
                height=1,
                mime_type="image/png",
                created_at=NOW,
            )
        )
        session.flush()
        session.add(
            DriveSyncRecordModel(
                id=str(record_id),
                generation_id=str(generation_id),
                status=DriveSyncStatus.SYNCING.value,
                remote_name="drive",
                remote_base_path="studio",
                remote_image_path="generated/image.png",
                remote_metadata_path="generated/image.json",
                image_artifact_id=str(artifact_id),
                metadata_artifact_id=None,
                image_sha256="a" * 64,
                metadata_sha256=None,
                image_size_bytes=3,
                metadata_size_bytes=None,
                attempt_count=1,
                last_attempt_at=NOW,
                synced_at=None,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.flush()
        session.add(
            DriveSyncJobModel(
                id=str(drive_job_id),
                sync_record_id=str(record_id),
                generation_id=str(generation_id),
                queue_sequence=1,
                status=DriveSyncStatus.SYNCING.value,
                progress_bytes=0,
                total_bytes=3,
                progress_percentage=0,
                current_artifact="image",
                worker_id="drive-worker",
                pid=42,
                claimed_at=NOW,
                lease_expires_at=NOW,
                started_at=NOW,
                completed_at=None,
                retryable=True,
                image_artifact_id=str(artifact_id),
                metadata_artifact_id=None,
                image_sha256="a" * 64,
                metadata_sha256=None,
                image_size_bytes=3,
                metadata_size_bytes=None,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            DriveManifestJobModel(
                id=str(uuid4()),
                local_date="2026-08-09",
                remote_name="drive",
                remote_base_path="studio",
                remote_manifest_path="2026-08-09/manifests/manifest.jsonl",
                queue_sequence=2,
                status=DriveSyncStatus.PENDING.value,
                worker_id=None,
                pid=None,
                claimed_at=None,
                lease_expires_at=None,
                started_at=None,
                completed_at=None,
                retryable=True,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.commit()

    repository = DriveSyncRepository(factory)
    assert repository.reconcile_stateless_restore(NOW) == 2
    with factory() as session:
        job = session.get(DriveSyncJobModel, str(drive_job_id))
        record = session.get(DriveSyncRecordModel, str(record_id))
        manifest = session.scalar(select(DriveManifestJobModel))
        assert job is not None and record is not None and manifest is not None
        assert job.status == DriveSyncStatus.FAILED.value
        assert record.status == DriveSyncStatus.FAILED.value
        assert job.error_code == "stateless_restore_missing_local_artifact"
        assert record.error_code == "stateless_restore_missing_local_artifact"
        assert job.retryable is False
        assert job.worker_id is None and job.pid is None
        assert job.claimed_at is None and job.lease_expires_at is None
        assert manifest.status == DriveSyncStatus.FAILED.value
        assert manifest.error_code == "stateless_restore_missing_local_artifact"
        assert manifest.retryable is False

    assert repository.reconcile_stateless_restore(NOW) == 0
    engine.dispose()


def test_state_sync_config_rejects_unsafe_subdir_and_empty_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _settings(tmp_path, state_sync_subdir="../outside")
    with pytest.raises(ValueError):
        _settings(tmp_path, state_sync_max_backups=0)


def test_rclone_restore_command_is_copyto_and_validates_destination(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    adapter = GoogleDriveAdapter(settings)
    command = adapter.build_copy_from_remote_command(
        DriveDestination("drive", "studio"),
        "backups/latest.sqlite3",
        tmp_path / "restore.sqlite3",
    )
    assert "copyto" in command
    assert "sync" not in command
    assert command[-2:] == (
        "drive:studio/backups/latest.sqlite3",
        str(tmp_path / "restore.sqlite3"),
    )
