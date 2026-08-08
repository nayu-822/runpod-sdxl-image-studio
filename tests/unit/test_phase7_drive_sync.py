"""Phase 7 Drive synchronization tests using SQLite and a fake rclone adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.adapters.drive.google_drive_adapter import GoogleDriveAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveSyncErrorCode,
    DriveSyncProgress,
    DriveSyncStatus,
    validate_remote_base_path,
    validate_remote_name,
)
from runpod_sdxl_image_studio.domain.generation import GenerationKind
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.services.drive_sync_service import DriveSyncService


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 64), "white").save(output, format="PNG")
    return output.getvalue()


def _settings() -> GenerationSettings:
    return GenerationSettings(
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


class FakeDriveAdapter:
    def __init__(self, *, fail_relative_path: str | None = None) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.fail_relative_path = fail_relative_path

    async def check_connection(self):
        from runpod_sdxl_image_studio.domain.drive_sync import (
            DriveConnectionResult,
            DriveConnectionStatus,
        )

        return DriveConnectionResult(DriveConnectionStatus.CONNECTED, "connected")

    async def copy_file(
        self,
        local_path: Path,
        relative_remote_path: str,
        *,
        progress_callback=None,
        total_bytes: int = 0,
    ) -> None:
        self.calls.append((local_path, relative_remote_path))
        if relative_remote_path == self.fail_relative_path:
            error = RuntimeError("fake transfer failed")
            error.code = DriveSyncErrorCode.TRANSFER_FAILED.value  # type: ignore[attr-defined]
            raise error
        if progress_callback is not None:
            result = progress_callback(DriveSyncProgress(total_bytes, total_bytes, 100.0, None))
            if asyncio.iscoroutine(result):
                await result


def _fixture(tmp_path: Path, *, adapter: FakeDriveAdapter | None = None):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        rclone_remote="drive",
        rclone_base_path="studio",
        max_output_image_bytes=1_000_000,
        max_metadata_sidecar_bytes=1_000_000,
    )
    created_at = datetime(2026, 8, 7, 15, 30, tzinfo=UTC)
    generation_id, job_id = uuid4(), uuid4()
    GenerationStartRepository(factory).create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=created_at,
    )
    image_bytes = _png()
    image_path = tmp_path / "generations" / "2026-08-08" / "generated" / "image.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image_bytes)
    image = GenerationArtifact(
        id=uuid4(),
        generation_id=generation_id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/2026-08-08/generated/image.png",
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        size_bytes=len(image_bytes),
        width=64,
        height=64,
        mime_type="image/png",
        created_at=created_at,
    )
    artifacts = GenerationArtifactRepository(factory)
    artifacts.add(image)
    GenerationCompletionRepository(factory).complete_generation(
        generation_id, job_id, image, created_at
    )
    metadata_payload = {
        "schema_version": 1,
        "generation_id": str(generation_id),
        "kind": "standard",
        "settings": {"seed": 42},
    }
    metadata_bytes = json.dumps(metadata_payload, separators=(",", ":")).encode("utf-8")
    metadata_path = image_path.with_suffix(".json")
    metadata_path.write_bytes(metadata_bytes)
    metadata = GenerationArtifact(
        id=uuid4(),
        generation_id=generation_id,
        artifact_type=ArtifactType.METADATA,
        local_path="generations/2026-08-08/generated/image.json",
        sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        size_bytes=len(metadata_bytes),
        width=None,
        height=None,
        mime_type="application/json",
        created_at=created_at,
    )
    artifacts.add(metadata)
    fake = adapter or FakeDriveAdapter()
    service = DriveSyncService(
        DriveSyncRepository(factory),
        GenerationRepository(factory),
        artifacts,
        settings,
        fake,
    )
    return service, DriveSyncRepository(factory), generation_id, image_path, metadata_path, fake


def test_drive_sync_success_copies_image_metadata_and_manifest(tmp_path: Path) -> None:
    service, repository, generation_id, image_path, metadata_path, adapter = _fixture(tmp_path)

    record = service.enqueue_generation(generation_id)
    assert record is not None
    before_capacity = service.capacity()
    assert before_capacity.unsynced_bytes == record.image_size_bytes + record.metadata_size_bytes
    assert before_capacity.synced_cache_bytes == 0
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    result = asyncio.run(service.process_job(claimed, "worker-1"))

    assert result is not None
    assert result.status is DriveSyncStatus.SYNCED
    assert [path for _, path in adapter.calls] == [
        "2026-08-08/generated/20260808_003000_" + generation_id.hex[:8] + ".png",
        "2026-08-08/generated/20260808_003000_" + generation_id.hex[:8] + ".json",
        "2026-08-08/manifests/manifest.jsonl",
    ]
    assert image_path.exists()
    assert metadata_path.exists()
    manifest = tmp_path / ".drive-sync-manifests" / "2026-08-08" / "manifest.jsonl"
    assert manifest.exists()
    assert str(generation_id) in manifest.read_text(encoding="utf-8")
    assert "prompt" not in manifest.read_text(encoding="utf-8")
    after_capacity = service.capacity()
    assert after_capacity.unsynced_bytes == 0
    assert after_capacity.synced_cache_bytes == record.image_size_bytes + record.metadata_size_bytes
    assert len(service.cache_candidates()) == 1


def test_drive_sync_partial_failure_preserves_local_and_marks_retryable(tmp_path: Path) -> None:
    adapter = FakeDriveAdapter()
    service, repository, generation_id, image_path, metadata_path, _ = _fixture(
        tmp_path, adapter=adapter
    )
    record = service.enqueue_generation(generation_id)
    assert record is not None
    adapter.fail_relative_path = record.remote_metadata_path
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    result = asyncio.run(service.process_job(claimed, "worker-1"))

    assert result is not None
    assert result.status is DriveSyncStatus.FAILED
    assert result.error_code == DriveSyncErrorCode.TRANSFER_FAILED.value
    assert result.synced_at is None
    assert image_path.exists() and metadata_path.exists()
    assert len(adapter.calls) == 2
    retried, retry_job = service.retry_generation(generation_id)
    assert retried.status is DriveSyncStatus.PENDING
    assert retry_job is not None
    same_record, active_job = service.retry_generation(generation_id)
    assert same_record.id == retried.id
    assert active_job is not None
    assert active_job.id == retry_job.id


def test_drive_sync_source_mutation_fails_before_any_copy(tmp_path: Path) -> None:
    service, repository, generation_id, image_path, _, adapter = _fixture(tmp_path)
    record = service.enqueue_generation(generation_id)
    assert record is not None
    image_path.write_bytes(b"changed")
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    result = asyncio.run(service.process_job(claimed, "worker-1"))

    assert result is not None
    assert result.status is DriveSyncStatus.FAILED
    assert result.error_code == DriveSyncErrorCode.SOURCE_CHANGED.value
    assert adapter.calls == []


def test_drive_sync_stale_claim_becomes_retryable_failure_without_pid(tmp_path: Path) -> None:
    service, repository, generation_id, _, _, _ = _fixture(tmp_path)
    assert service.enqueue_generation(generation_id) is not None
    claimed = repository.claim_next("old-worker", 120)
    assert claimed is not None
    assert claimed.claimed_at is not None

    assert repository.reconcile_stale(claimed.claimed_at + timedelta(seconds=121)) == 1
    job = repository.get_job(claimed.id)
    record = repository.get_by_generation(generation_id)
    assert job is not None and record is not None
    assert job.status is DriveSyncStatus.FAILED
    assert job.error_code == DriveSyncErrorCode.STALE.value
    assert job.retryable is True
    assert job.pid is None
    assert record.status is DriveSyncStatus.FAILED


@pytest.mark.parametrize(
    ("value", "valid"),
    [("drive", True), ("", True), ("../drive", False), ("drive/name", False), ("-drive", False)],
)
def test_drive_remote_name_validation(value: str, valid: bool) -> None:
    if valid:
        validate_remote_name(value)
    else:
        with pytest.raises(ValueError):
            validate_remote_name(value)


def test_drive_remote_base_path_rejects_traversal() -> None:
    assert validate_remote_base_path("studio/base") == "studio/base"
    with pytest.raises(ValueError):
        validate_remote_base_path("studio/../outside")


def test_rclone_adapter_uses_argument_array_copyto_without_sync(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        rclone_remote="drive",
        rclone_base_path="studio",
        rclone_config=tmp_path / "rclone.conf",
    )
    command = GoogleDriveAdapter(settings).build_copy_command(
        tmp_path / "image.png", "2026-08-08/generated/image.png"
    )
    assert command[0] == "rclone"
    assert "--config" in command
    assert str(settings.rclone_config) in command
    assert "copyto" in command
    assert "sync" not in command
    assert "--stats-one-line-json" in command
