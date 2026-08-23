"""Phase 7 Drive synchronization tests using SQLite and a fake rclone adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory, session_scope
from runpod_sdxl_image_studio.adapters.database.models import Base, DriveManifestJobModel
from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepository,
    DriveSyncRepositoryError,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepository,
)
from runpod_sdxl_image_studio.adapters.drive.google_drive_adapter import (
    GoogleDriveAdapter,
    GoogleDriveAdapterError,
    _progress_from_line,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveConnectionStatus,
    DriveDestination,
    DriveManifestState,
    DriveSyncErrorCode,
    DriveSyncProgress,
    DriveSyncStatus,
    build_google_drive_output_folder,
    validate_remote_base_path,
    validate_remote_name,
)
from runpod_sdxl_image_studio.domain.generation import GenerationKind
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSettingsSnapshot
from runpod_sdxl_image_studio.jobs.drive_sync_worker import DriveSyncWorker
from runpod_sdxl_image_studio.services.drive_sync_service import (
    DriveSyncService,
    DriveSyncServiceError,
)
from runpod_sdxl_image_studio.services.upscale_enqueue_service import UpscaleEnqueueService
from runpod_sdxl_image_studio.ui.tabs.drive_sync_tab import (
    make_drive_manifest_handler,
    make_drive_resync_handler,
)


def test_active_manifest_check_is_unbounded_and_does_not_list_jobs() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    now = datetime(2026, 8, 10, tzinfo=UTC)
    with session_scope(factory) as session:
        session.add(
            DriveManifestJobModel(
                id=str(uuid4()),
                local_date="2026-08-10",
                remote_name="drive",
                remote_base_path="studio",
                remote_manifest_path="2026-08-10/manifests/manifest.jsonl",
                queue_sequence=1,
                status=DriveSyncStatus.SYNCING.value,
                progress_bytes=0,
                total_bytes=1,
                progress_percentage=0.0,
                current_artifact=None,
                worker_id="worker-1",
                pid=None,
                claimed_at=now,
                lease_expires_at=None,
                started_at=now,
                completed_at=None,
                error_code=None,
                error_summary=None,
                retryable=True,
                log_path=None,
                created_at=now,
                updated_at=now,
            )
        )
    repository = DriveSyncRepository(factory)
    repository.list_manifest_jobs = lambda limit=50: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("termination safety must not list a bounded manifest window")
    )

    assert repository.has_active_manifest_jobs()


def _png(size: tuple[int, int] = (64, 64)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def _settings(*, final_upscale: bool = False) -> GenerationSettings:
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
        final_upscale=final_upscale,
        final_upscale_model="4x-UltraSharp.pth" if final_upscale else None,
    )


class FakeDriveAdapter:
    def __init__(self, *, fail_relative_path: str | None = None) -> None:
        self.calls: list[tuple[Path, DriveDestination, str]] = []
        self.remote_files: dict[tuple[DriveDestination, str], bytes] = {}
        self.fail_relative_path = fail_relative_path
        self.progress_events: list[DriveSyncProgress] = []
        self.progress_observer = None
        self.progress_snapshots: list[tuple[int, str | None, int | None]] = []

    async def check_connection(self):
        from runpod_sdxl_image_studio.domain.drive_sync import (
            DriveConnectionResult,
            DriveConnectionStatus,
        )

        return DriveConnectionResult(DriveConnectionStatus.CONNECTED, "connected")

    async def copy_file(
        self,
        local_path: Path,
        destination: DriveDestination,
        relative_remote_path: str,
        *,
        progress_callback=None,
        total_bytes: int = 0,
        current_artifact: str | None = None,
        process_started_callback=None,
        process_finished_callback=None,
        log_path: str | None = None,
    ) -> None:
        del log_path
        self.calls.append((local_path, destination, relative_remote_path))
        if process_started_callback is not None:
            result = process_started_callback(4242)
            if asyncio.iscoroutine(result):
                await result
        if relative_remote_path == self.fail_relative_path:
            error = RuntimeError("fake transfer failed")
            error.code = DriveSyncErrorCode.TRANSFER_FAILED.value  # type: ignore[attr-defined]
            if process_finished_callback is not None:
                result = process_finished_callback()
                if asyncio.iscoroutine(result):
                    await result
            raise error
        if progress_callback is not None:
            for percentage in (0.0, 25.0, 50.0, 75.0, 100.0):
                progress = DriveSyncProgress(
                    int(total_bytes * percentage / 100),
                    total_bytes,
                    percentage,
                    current_artifact,
                )
                self.progress_events.append(progress)
                result = progress_callback(progress)
                if asyncio.iscoroutine(result):
                    await result
                if self.progress_observer is not None:
                    observed = self.progress_observer()
                    self.progress_snapshots.append(
                        (
                            observed.progress_bytes,
                            observed.current_artifact,
                            observed.pid,
                        )
                    )
        if process_finished_callback is not None:
            result = process_finished_callback()
            if asyncio.iscoroutine(result):
                await result
        self.remote_files[(destination, relative_remote_path)] = local_path.read_bytes()


class FakeUpscaleSettingsRepository:
    def __init__(self, snapshot: UpscaleSettingsSnapshot) -> None:
        self.snapshot = snapshot

    def get_by_generation(self, generation_id: UUID) -> UpscaleSettingsSnapshot | None:
        return self.snapshot if self.snapshot.source_generation_id == generation_id else None


def _fixture(
    tmp_path: Path,
    *,
    adapter: FakeDriveAdapter | None = None,
    kind: GenerationKind = GenerationKind.STANDARD,
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
    generation_settings: GenerationSettings | None = None,
    upscale_factor: float | None = None,
):
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
    created_at = created_at or datetime(2026, 8, 7, 15, 30, tzinfo=UTC)
    completed_at = completed_at or created_at
    generation_settings = generation_settings or _settings()
    generation_id, job_id = uuid4(), uuid4()
    GenerationStartRepository(factory).create_pending(
        GenerationSettingsSnapshot.from_settings(generation_settings),
        generation_id=generation_id,
        job_id=job_id,
        kind=kind,
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
        generation_id, job_id, image, completed_at
    )
    metadata_payload = {
        "schema_version": 1,
        "generation_id": str(generation_id),
        "kind": kind.value,
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
    upscale_settings_repository = None
    if kind is GenerationKind.UPSCALE:
        if upscale_factor is None:
            raise ValueError("upscale_factor is required for upscale fixture")
        target_size = int(64 * upscale_factor)
        upscale_settings_repository = FakeUpscaleSettingsRepository(
            UpscaleSettingsSnapshot(
                method=UpscaleMethod.IMAGE,
                sizing_mode=UpscaleSizingMode.FACTOR,
                requested_scale_factor=upscale_factor,
                target_width=target_size,
                target_height=target_size,
                upscaler_name="upscalers/test.pth",
                source_generation_id=generation_id,
                source_artifact_id=image.id,
                source_sha256=image.sha256,
                source_width=image.width,
                source_height=image.height,
                workflow_template_id="sdxl_image_upscale",
                workflow_template_version="1.0",
            )
        )
    service = DriveSyncService(
        DriveSyncRepository(factory),
        GenerationRepository(factory),
        artifacts,
        settings,
        fake,
        upscale_settings_repository=upscale_settings_repository,
    )
    return service, DriveSyncRepository(factory), generation_id, image_path, metadata_path, fake


def _actual_upscale_drive_record(tmp_path: Path, scale_factor: float):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        upscaler_dir=tmp_path / "upscalers",
        rclone_remote="drive",
        rclone_base_path="studio",
        max_width=256,
        max_height=256,
        max_output_image_bytes=1_000_000,
        max_metadata_sidecar_bytes=1_000_000,
    )
    generations = GenerationRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    now = datetime(2026, 8, 7, 15, 30, tzinfo=UTC)

    parent_id, parent_job_id = uuid4(), uuid4()
    GenerationStartRepository(factory).create_pending(
        GenerationSettingsSnapshot.from_settings(_settings(final_upscale=True)),
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=now,
    )
    parent_path = tmp_path / "generations" / "2026-08-08" / "generated" / "parent.png"
    parent_path.parent.mkdir(parents=True)
    parent_bytes = _png()
    parent_path.write_bytes(parent_bytes)
    parent_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/2026-08-08/generated/parent.png",
        sha256=hashlib.sha256(parent_bytes).hexdigest(),
        size_bytes=len(parent_bytes),
        width=64,
        height=64,
        mime_type="image/png",
        created_at=now,
    )
    artifacts.add(parent_artifact)
    GenerationCompletionRepository(factory).complete_generation(
        parent_id, parent_job_id, parent_artifact, now
    )

    enqueue = UpscaleEnqueueService(
        generations,
        artifacts,
        GenerationDispatchQueueRepository(factory),
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
    )
    item = enqueue.enqueue(
        parent_id,
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=scale_factor,
            upscaler_name="4x.pth",
        ),
    )

    child_size = int(64 * scale_factor)
    child_path = (
        tmp_path / "generations" / "2026-08-08" / "upscaled" / f"child-{int(scale_factor)}x.png"
    )
    child_path.parent.mkdir(parents=True)
    child_bytes = _png((child_size, child_size))
    child_path.write_bytes(child_bytes)
    child_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=item.generation.id,
        artifact_type=ArtifactType.IMAGE,
        local_path=child_path.relative_to(tmp_path).as_posix(),
        sha256=hashlib.sha256(child_bytes).hexdigest(),
        size_bytes=len(child_bytes),
        width=child_size,
        height=child_size,
        mime_type="image/png",
        created_at=now,
    )
    artifacts.add(child_artifact)
    completed_at = now + timedelta(minutes=1)
    GenerationCompletionRepository(factory).complete_generation(
        item.generation.id, item.job.id, child_artifact, completed_at
    )
    metadata_path = child_path.with_suffix(".json")
    metadata_bytes = b'{"schema_version":1}'
    metadata_path.write_bytes(metadata_bytes)
    artifacts.add(
        GenerationArtifact(
            id=uuid4(),
            generation_id=item.generation.id,
            artifact_type=ArtifactType.METADATA,
            local_path=metadata_path.relative_to(tmp_path).as_posix(),
            sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            size_bytes=len(metadata_bytes),
            width=None,
            height=None,
            mime_type="application/json",
            created_at=now,
        )
    )

    adapter = FakeDriveAdapter()
    drive = DriveSyncService(
        DriveSyncRepository(factory),
        generations,
        artifacts,
        settings,
        adapter,
        upscale_settings_repository=UpscaleSettingsRepository(factory),
    )
    record = drive.enqueue_generation(item.generation.id)
    persisted_child = generations.get_by_id(item.generation.id)
    engine.dispose()
    assert persisted_child is not None
    return record, item, persisted_child


def _add_second_image(service: DriveSyncService, generation_id: UUID, image_path: Path) -> Path:
    created_at = datetime(2026, 8, 7, 15, 30, tzinfo=UTC)
    second_bytes = _png()
    second_path = image_path.with_name("second.png")
    second_path.write_bytes(second_bytes)
    service._artifact_repository.add(  # type: ignore[attr-defined]
        GenerationArtifact(
            id=uuid4(),
            generation_id=generation_id,
            artifact_type=ArtifactType.IMAGE,
            local_path="generations/2026-08-08/generated/second.png",
            sha256=hashlib.sha256(second_bytes).hexdigest(),
            size_bytes=len(second_bytes),
            width=64,
            height=64,
            mime_type="image/png",
            created_at=created_at,
            display_order=1,
        )
    )
    return second_path


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
    manifest_job = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job is not None
    assert asyncio.run(service.process_manifest_job(manifest_job, "worker-1")) is not None

    assert [path for _, _, path in adapter.calls] == [
        "20260808_x1/20260808_003000_" + generation_id.hex[:8] + ".png",
        "20260808_x1/20260808_003000_" + generation_id.hex[:8] + ".json",
        "2026-08-08/manifests/manifest.jsonl",
    ]
    assert image_path.exists()
    assert metadata_path.exists()
    manifest = tmp_path / ".drive-sync-manifests" / "2026-08-08" / "manifest.jsonl"
    assert manifest.exists()
    assert str(generation_id) in manifest.read_text(encoding="utf-8")
    assert "prompt" not in manifest.read_text(encoding="utf-8")


def test_drive_sync_uses_final_upscale_snapshot_for_x4_folder(tmp_path: Path) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(
        tmp_path,
        generation_settings=_settings(final_upscale=True),
    )

    record = service.enqueue_generation(generation_id)

    assert record is not None
    expected_stem = "20260808_003000_" + generation_id.hex[:8]
    assert record.remote_image_path == f"20260808_x4/{expected_stem}.png"
    assert record.remote_metadata_path == f"20260808_x4/{expected_stem}.json"
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    assert asyncio.run(service.process_job(claimed, "worker-1")) is not None
    assert [path for _, _, path in adapter.calls[:2]] == [
        record.remote_image_path,
        record.remote_metadata_path,
    ]


@pytest.mark.parametrize(
    ("kind", "upscale_factor", "final_upscale", "expected_folder"),
    [
        (GenerationKind.STANDARD, None, False, "20260808_x1"),
        (GenerationKind.STANDARD, None, True, "20260808_x4"),
        (GenerationKind.UPSCALE, 2.0, True, "20260808_x1"),
        (GenerationKind.UPSCALE, 4.0, True, "20260808_x4"),
    ],
)
def test_drive_sync_classifies_persisted_generation_kind_and_upscale_factor(
    tmp_path: Path,
    kind: GenerationKind,
    upscale_factor: float | None,
    final_upscale: bool,
    expected_folder: str,
) -> None:
    service, _, generation_id, _, _, _ = _fixture(
        tmp_path,
        kind=kind,
        generation_settings=_settings(final_upscale=final_upscale),
        upscale_factor=upscale_factor,
    )

    record = service.enqueue_generation(generation_id)

    assert record is not None
    assert record.remote_image_path.startswith(f"{expected_folder}/")
    assert record.remote_metadata_path.startswith(f"{expected_folder}/")


@pytest.mark.parametrize(
    ("scale_factor", "expected_folder"),
    [(2.0, "20260808_x1"), (4.0, "20260808_x4")],
)
def test_upscale_enqueue_child_uses_its_persisted_factor_for_drive_folder(
    tmp_path: Path,
    scale_factor: float,
    expected_folder: str,
) -> None:
    record, item, persisted_child = _actual_upscale_drive_record(tmp_path, scale_factor)

    assert item.generation.settings_snapshot.final_upscale is True
    assert persisted_child.kind is GenerationKind.UPSCALE
    assert persisted_child.settings_snapshot.final_upscale is True
    assert record is not None
    assert record.remote_image_path.startswith(f"{expected_folder}/")
    assert record.remote_metadata_path.startswith(f"{expected_folder}/")


@pytest.mark.parametrize(
    ("target_width", "target_height", "expected_folder"),
    [
        (2048, 2048, "20260808_x1"),
        (4096, 4096, "20260808_x4"),
    ],
)
def test_drive_sync_classifies_upscale_dimensions_without_inherited_final_flag(
    target_width: int,
    target_height: int,
    expected_folder: str,
) -> None:
    folder = build_google_drive_output_folder(
        kind=GenerationKind.UPSCALE,
        created_at=datetime(2026, 8, 7, 15, 30, tzinfo=UTC),
        final_upscale=True,
        source_width=1024,
        source_height=1024,
        target_width=target_width,
        target_height=target_height,
    )

    assert folder == expected_folder


def test_drive_sync_uses_tokyo_completion_date_not_upload_date_or_utc_date(
    tmp_path: Path,
) -> None:
    service, _, generation_id, _, _, _ = _fixture(
        tmp_path,
        created_at=datetime(2026, 8, 21, 15, 30, tzinfo=UTC),
        completed_at=datetime(2026, 8, 21, 15, 45, tzinfo=UTC),
    )

    record = service.enqueue_generation(generation_id)

    assert record is not None
    assert record.remote_image_path.startswith("20260822_x1/")
    assert record.remote_metadata_path.startswith("20260822_x1/")
    assert "20260821_" not in record.remote_image_path


def test_drive_sync_retry_and_resync_keep_the_same_generation_folder(tmp_path: Path) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(
        tmp_path,
        created_at=datetime(2026, 8, 21, 15, 30, tzinfo=UTC),
        completed_at=datetime(2026, 8, 21, 15, 45, tzinfo=UTC),
    )
    initial = service.enqueue_generation(generation_id)
    assert initial is not None
    assert initial.remote_image_path.startswith("20260822_x1/")
    first_job = repository.claim_next("worker-1", 120)
    assert first_job is not None
    assert asyncio.run(service.process_job(first_job, "worker-1")) is not None
    manifest_job = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job is not None
    assert asyncio.run(service.process_manifest_job(manifest_job, "worker-1")) is not None

    retried, retry_job = service.retry_generation(generation_id, resync=True)

    assert retry_job is not None
    assert retried.remote_image_path == initial.remote_image_path
    assert retried.remote_metadata_path == initial.remote_metadata_path
    assert retried.remote_image_path.startswith("20260822_x1/")
    assert adapter.calls[-1][2] == manifest_job.remote_manifest_path


def test_drive_sync_transfers_all_ordered_images_and_manifest_entries(tmp_path: Path) -> None:
    service, repository, generation_id, image_path, _, adapter = _fixture(tmp_path)
    created_at = datetime(2026, 8, 7, 15, 30, tzinfo=UTC)
    second_bytes = _png()
    second_path = image_path.with_name("second.png")
    second_path.write_bytes(second_bytes)
    service._artifact_repository.add(  # type: ignore[attr-defined]
        GenerationArtifact(
            id=uuid4(),
            generation_id=generation_id,
            artifact_type=ArtifactType.IMAGE,
            local_path="generations/2026-08-08/generated/second.png",
            sha256=hashlib.sha256(second_bytes).hexdigest(),
            size_bytes=len(second_bytes),
            width=64,
            height=64,
            mime_type="image/png",
            created_at=created_at,
            display_order=1,
        )
    )

    record = service.enqueue_generation(generation_id)
    assert record is not None
    assert [item.display_order for item in record.artifacts] == [0, 1]
    assert len({item.remote_image_path for item in record.artifacts}) == 2
    job = repository.claim_next("worker-1", 120)
    assert job is not None
    synced = asyncio.run(service.process_job(job, "worker-1"))
    assert synced is not None and synced.status is DriveSyncStatus.SYNCED
    assert all(item.image_synced and item.metadata_synced for item in synced.artifacts)
    image_calls = [call for call in adapter.calls if call[2].endswith(".png")]
    assert len(image_calls) == 2
    assert {call[2] for call in image_calls} == {
        item.remote_image_path for item in synced.artifacts
    }

    manifest_job = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job is not None
    assert asyncio.run(service.process_manifest_job(manifest_job, "worker-1")) is not None
    manifest = json.loads(
        (tmp_path / ".drive-sync-manifests" / "2026-08-08" / "manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert [item["display_order"] for item in manifest["images"]] == [0, 1]


def test_drive_sync_retry_resumes_after_partial_multi_image_failure(tmp_path: Path) -> None:
    service, repository, generation_id, image_path, _, adapter = _fixture(tmp_path)
    created_at = datetime(2026, 8, 7, 15, 30, tzinfo=UTC)
    second_bytes = _png()
    second_path = image_path.with_name("second.png")
    second_path.write_bytes(second_bytes)
    service._artifact_repository.add(  # type: ignore[attr-defined]
        GenerationArtifact(
            id=uuid4(),
            generation_id=generation_id,
            artifact_type=ArtifactType.IMAGE,
            local_path="generations/2026-08-08/generated/second.png",
            sha256=hashlib.sha256(second_bytes).hexdigest(),
            size_bytes=len(second_bytes),
            width=64,
            height=64,
            mime_type="image/png",
            created_at=created_at,
            display_order=1,
        )
    )
    pending = service.enqueue_generation(generation_id)
    assert pending is not None
    adapter.fail_relative_path = pending.artifacts[1].remote_image_path
    first_job = repository.claim_next("worker-1", 120)
    assert first_job is not None
    failed = asyncio.run(service.process_job(first_job, "worker-1"))
    assert failed is not None and failed.status is DriveSyncStatus.FAILED
    assert failed.artifacts[0].image_synced is True
    assert failed.artifacts[1].image_synced is False
    calls_before_retry = len(adapter.calls)

    adapter.fail_relative_path = None
    retried, retry_job = service.retry_generation(generation_id)
    assert retried.status is DriveSyncStatus.PENDING
    assert retry_job is not None
    assert retried.remote_image_path == pending.remote_image_path
    assert retried.remote_metadata_path == pending.remote_metadata_path
    claimed_retry = repository.claim_next("worker-1", 120)
    assert claimed_retry is not None
    completed = asyncio.run(service.process_job(claimed_retry, "worker-1"))
    assert completed is not None and completed.status is DriveSyncStatus.SYNCED
    retry_calls = adapter.calls[calls_before_retry:]
    assert pending.artifacts[0].remote_image_path not in {call[2] for call in retry_calls}
    assert all(item.image_synced and item.metadata_synced for item in completed.artifacts)
    after_capacity = service.capacity()
    assert after_capacity.unsynced_bytes == 0
    assert after_capacity.synced_cache_bytes == sum(
        item.image_size_bytes for item in completed.artifacts
    ) + (completed.artifacts[0].metadata_size_bytes or 0)
    assert len(service.cache_candidates()) == 1


def test_drive_sync_retry_changed_destination_copies_all_artifacts(tmp_path: Path) -> None:
    adapter = FakeDriveAdapter()
    service, repository, generation_id, image_path, _, _ = _fixture(tmp_path, adapter=adapter)
    _add_second_image(service, generation_id, image_path)
    pending = service.enqueue_generation(generation_id)
    assert pending is not None
    adapter.fail_relative_path = pending.artifacts[1].remote_image_path
    first_job = repository.claim_next("worker-1", 120)
    assert first_job is not None
    failed = asyncio.run(service.process_job(first_job, "worker-1"))
    assert failed is not None and failed.status is DriveSyncStatus.FAILED
    assert failed.artifacts[0].image_synced is True
    assert failed.artifacts[1].image_synced is False
    calls_before_retry = len(adapter.calls)

    service._settings.rclone_remote = "drive-b"
    service._settings.rclone_base_path = "studio-b"
    retried, retry_job = service.retry_generation(generation_id)
    assert retry_job is not None
    assert all(not item.image_synced and not item.metadata_synced for item in retried.artifacts)

    adapter.fail_relative_path = None
    claimed_retry = repository.claim_next("worker-1", 120)
    assert claimed_retry is not None
    completed = asyncio.run(service.process_job(claimed_retry, "worker-1"))
    assert completed is not None and completed.status is DriveSyncStatus.SYNCED
    retry_calls = adapter.calls[calls_before_retry:]
    assert len(retry_calls) == 3
    assert all(call[1] == DriveDestination("drive-b", "studio-b") for call in retry_calls)
    assert {call[2] for call in retry_calls} == {
        *(item.remote_image_path for item in completed.artifacts),
        completed.remote_metadata_path,
    }


def test_drive_sync_resync_same_destination_copies_all_artifacts_again(tmp_path: Path) -> None:
    adapter = FakeDriveAdapter()
    service, repository, generation_id, image_path, _, _ = _fixture(tmp_path, adapter=adapter)
    _add_second_image(service, generation_id, image_path)
    assert service.enqueue_generation(generation_id) is not None
    initial_job = repository.claim_next("worker-1", 120)
    assert initial_job is not None
    assert asyncio.run(service.process_job(initial_job, "worker-1")) is not None
    initial_manifest = repository.claim_next_manifest("worker-1", 120)
    assert initial_manifest is not None
    assert asyncio.run(service.process_manifest_job(initial_manifest, "worker-1")) is not None
    adapter.calls.clear()

    resynced, resync_job = service.retry_generation(generation_id, resync=True)
    assert resync_job is not None
    assert all(not item.image_synced and not item.metadata_synced for item in resynced.artifacts)
    claimed_resync = repository.claim_next("worker-1", 120)
    assert claimed_resync is not None
    completed = asyncio.run(service.process_job(claimed_resync, "worker-1"))
    assert completed is not None and completed.status is DriveSyncStatus.SYNCED
    assert len(adapter.calls) == 3
    assert all(call[1] == DriveDestination("drive", "studio") for call in adapter.calls)
    assert {call[2] for call in adapter.calls} == {
        *(item.remote_image_path for item in completed.artifacts),
        completed.remote_metadata_path,
    }


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
        tmp_path / "image.png",
        DriveDestination("drive", "studio"),
        "20260808_x1/image.png",
    )
    assert command[0] == "rclone"
    assert "--config" in command
    assert str(settings.rclone_config) in command
    assert "copyto" in command
    assert "sync" not in command
    assert "--stats-one-line-json" not in command
    assert "--use-json-log" in command
    assert command[-5:] == (
        "--use-json-log",
        "--stats",
        "1s",
        "--stats-log-level",
        "NOTICE",
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            '{"level":"notice","msg":"stats","stats":{"bytes":100,"totalBytes":200}}',
            (100, 200, 50.0),
        ),
        ('{"bytes":100,"totalBytes":200,"percentage":25}', (100, 200, 25.0)),
        ('{"level":"info","msg":"Copied (new)"}', None),
        ("not json", None),
        ('{"stats":{"bytes":"100","totalBytes":200}}', None),
        ('{"stats":{"bytes":-1,"totalBytes":200}}', None),
        ('{"stats":{"bytes":100}}', (100, 1000, 10.0)),
        ('{"stats":{"bytes":300,"totalBytes":200}}', (300, 300, 100.0)),
        ('{"stats":{"bytes":100,"totalBytes":200,"percentage":150}}', (100, 200, 100.0)),
        ('{"stats":{"bytes":100,"totalBytes":200,"percentage":-10}}', (100, 200, 0.0)),
        ('{"stats":{"bytes":100,"totalBytes":200,"percentage":NaN}}', None),
        ('{"stats":{"bytes":100,"totalBytes":200,"percentage":Infinity}}', None),
    ],
)
def test_rclone_json_progress_parser_supports_nested_and_legacy_stats(
    line: str, expected: tuple[int, int, float] | None
) -> None:
    result = _progress_from_line(line, 1000, "image")
    if expected is None:
        assert result is None
        return

    assert result is not None
    assert (result.progress_bytes, result.total_bytes, result.progress_percentage) == expected
    assert result.current_artifact == "image"


def test_pending_job_uses_destination_snapshot_after_settings_change(tmp_path: Path) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(tmp_path)
    record = service.enqueue_generation(generation_id)
    assert record is not None
    service._settings.rclone_remote = "drive-b"
    service._settings.rclone_base_path = "studio-b"

    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    assert asyncio.run(service.process_job(claimed, "worker-1")) is not None

    assert adapter.calls[0][1] == DriveDestination("drive", "studio")
    persisted = repository.get_by_generation(generation_id)
    assert persisted is not None
    assert (persisted.remote_name, persisted.remote_base_path) == ("drive", "studio")
    manifest_job = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job is not None
    assert manifest_job.destination == DriveDestination("drive", "studio")


def test_explicit_retry_snapshots_current_destination_without_changing_deterministic_path(
    tmp_path: Path,
) -> None:
    adapter = FakeDriveAdapter()
    service, repository, generation_id, _, _, _ = _fixture(tmp_path, adapter=adapter)
    record = service.enqueue_generation(generation_id)
    assert record is not None
    adapter.fail_relative_path = record.remote_metadata_path
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    assert asyncio.run(service.process_job(claimed, "worker-1")) is not None

    service._settings.rclone_remote = "drive-b"
    service._settings.rclone_base_path = "studio-b"
    retried, retry_job = service.retry_generation(generation_id)

    assert retry_job is not None
    assert (retried.remote_name, retried.remote_base_path) == ("drive-b", "studio-b")
    assert retried.remote_image_path == record.remote_image_path
    assert retry_job.status is DriveSyncStatus.PENDING


def test_resync_rejects_pending_old_manifest_and_atomic_race_without_new_destination_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, repository, generation_id, _, _, _ = _fixture(tmp_path)
    assert service.enqueue_generation(generation_id) is not None
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    assert asyncio.run(service.process_job(claimed, "worker-1")) is not None
    assert (
        repository.manifest_state_for_destination("2026-08-08", DriveDestination("drive", "studio"))
        is DriveManifestState.PENDING
    )

    service._settings.rclone_remote = "drive-b"
    service._settings.rclone_base_path = "studio-b"
    with monkeypatch.context() as context:
        context.setattr(
            service._repository,
            "manifest_state_for_destination",
            lambda local_date, destination: DriveManifestState.SYNCED,
        )
        with pytest.raises(DriveSyncServiceError) as resync_error:
            service.retry_generation(generation_id, resync=True)

    assert resync_error.value.code == DriveSyncErrorCode.MANIFEST_REBUILD_REQUIRED.value
    record = repository.get_by_generation(generation_id)
    assert record is not None
    assert record.status is DriveSyncStatus.SYNCED
    assert (record.remote_name, record.remote_base_path) == ("drive", "studio")
    drive_jobs = repository.list_jobs()
    assert len(drive_jobs) == 1
    assert drive_jobs[0].status is DriveSyncStatus.SYNCED
    manifest_jobs = repository.list_manifest_jobs()
    assert len(manifest_jobs) == 1
    assert manifest_jobs[0].status is DriveSyncStatus.PENDING


def test_resync_with_missing_local_image_does_not_enqueue_or_copy(tmp_path: Path) -> None:
    adapter = FakeDriveAdapter()
    service, repository, generation_id, image_path, _, _ = _fixture(
        tmp_path,
        adapter=adapter,
    )
    assert service.enqueue_generation(generation_id) is not None
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    assert asyncio.run(service.process_job(claimed, "worker-1")) is not None
    manifest_job = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job is not None
    assert asyncio.run(service.process_manifest_job(manifest_job, "worker-1")) is not None
    existing_job_count = len(repository.list_jobs())
    adapter.calls.clear()
    image_path.unlink()

    with pytest.raises(DriveSyncServiceError) as error:
        service.retry_generation(generation_id, resync=True)

    assert error.value.code == DriveSyncErrorCode.SOURCE_MISSING.value
    assert len(repository.list_jobs()) == existing_job_count
    assert adapter.calls == []


def test_process_pid_and_live_progress_are_persisted_with_lease_owner(
    tmp_path: Path,
) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(tmp_path)
    assert service.enqueue_generation(generation_id) is not None
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    adapter.progress_observer = lambda: repository.get_job(claimed.id)

    result = asyncio.run(service.process_job(claimed, "worker-1"))

    assert result is not None
    assert [event.progress_percentage for event in adapter.progress_events[:5]] == [
        0.0,
        25.0,
        50.0,
        75.0,
        100.0,
    ]
    assert any(0 < progress < 100 for progress, _, _ in adapter.progress_snapshots)
    assert any(pid == 4242 for _, _, pid in adapter.progress_snapshots)
    persisted = repository.get_job(claimed.id)
    assert persisted is not None
    assert persisted.pid is None
    assert repository.mark_process_started(claimed.id, "other-worker", 9000) is False


def test_manifest_is_filtered_by_destination(tmp_path: Path) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(tmp_path)
    assert service.enqueue_generation(generation_id) is not None
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    assert asyncio.run(service.process_job(claimed, "worker-1")) is not None

    other = service._write_manifest("2026-08-08", DriveDestination("drive-b", "studio-b"))
    assert str(generation_id) not in other.read_text(encoding="utf-8")
    manifest = service._write_manifest("2026-08-08", DriveDestination("drive", "studio"))
    assert str(generation_id) in manifest.read_text(encoding="utf-8")
    assert len(adapter.calls) == 2


def test_resync_preserves_old_remote_manifest_and_builds_new_destination_manifest(
    tmp_path: Path,
) -> None:
    adapter = FakeDriveAdapter()
    service, repository, generation_id, _, _, _ = _fixture(tmp_path, adapter=adapter)
    assert service.enqueue_generation(generation_id) is not None
    drive_job_a = repository.claim_next("worker-1", 120)
    assert drive_job_a is not None
    assert asyncio.run(service.process_job(drive_job_a, "worker-1")) is not None
    manifest_job_a = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job_a is not None
    assert asyncio.run(service.process_manifest_job(manifest_job_a, "worker-1")) is not None

    old_manifest = adapter.remote_files[
        (DriveDestination("drive", "studio"), manifest_job_a.remote_manifest_path)
    ]
    assert str(generation_id).encode() in old_manifest
    calls_before_resync = len(adapter.calls)

    service._settings.rclone_remote = "drive-b"
    service._settings.rclone_base_path = "studio-b"
    resynced, resync_job = service.retry_generation(generation_id, resync=True)
    assert resync_job is not None and resync_job.status is DriveSyncStatus.PENDING
    assert (resynced.remote_name, resynced.remote_base_path) == ("drive-b", "studio-b")
    assert all(not item.image_synced and not item.metadata_synced for item in resynced.artifacts)

    drive_job_b = repository.claim_next("worker-1", 120)
    assert drive_job_b is not None
    synced_b = asyncio.run(service.process_job(drive_job_b, "worker-1"))
    assert synced_b is not None and synced_b.status is DriveSyncStatus.SYNCED
    assert len(adapter.calls) - calls_before_resync == 2
    assert all(call[1] == DriveDestination("drive-b", "studio-b") for call in adapter.calls[-2:])
    assert {call[2] for call in adapter.calls[-2:]} == {
        synced_b.artifacts[0].remote_image_path,
        synced_b.remote_metadata_path,
    }
    manifest_job_b = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job_b is not None
    assert manifest_job_b.destination == DriveDestination("drive-b", "studio-b")
    assert asyncio.run(service.process_manifest_job(manifest_job_b, "worker-1")) is not None

    new_manifest = adapter.remote_files[
        (DriveDestination("drive-b", "studio-b"), manifest_job_b.remote_manifest_path)
    ]
    assert str(generation_id).encode() in new_manifest
    assert (
        adapter.remote_files[
            (DriveDestination("drive", "studio"), manifest_job_a.remote_manifest_path)
        ]
        == old_manifest
    )


def test_manifest_failure_keeps_sync_synced_and_rebuilds_the_affected_date(
    tmp_path: Path,
) -> None:
    adapter = FakeDriveAdapter()
    service, repository, generation_id, _, _, _ = _fixture(tmp_path, adapter=adapter)
    assert service.enqueue_generation(generation_id) is not None
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None
    synced = asyncio.run(service.process_job(claimed, "worker-1"))
    assert synced is not None and synced.status is DriveSyncStatus.SYNCED

    manifest_job = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job is not None
    adapter.fail_relative_path = manifest_job.remote_manifest_path
    failed = asyncio.run(service.process_manifest_job(manifest_job, "worker-1"))
    assert failed is not None and failed.status is DriveSyncStatus.FAILED
    record = repository.get_by_generation(generation_id)
    assert record is not None
    assert record.status is DriveSyncStatus.SYNCED
    assert record.error_code == DriveSyncErrorCode.MANIFEST_FAILED.value
    assert (
        repository.manifest_state_for_destination("2026-08-08", DriveDestination("drive", "studio"))
        is DriveManifestState.FAILED
    )
    targets = service.list_manifest_failure_targets()
    assert len(targets) == 1
    assert targets[0].local_date == "2026-08-08"
    assert targets[0].remote_name == "drive"
    assert targets[0].remote_base_path == "studio"

    service._settings.rclone_remote = "drive-b"
    service._settings.rclone_base_path = "studio-b"
    with pytest.raises(DriveSyncServiceError) as resync_error:
        service.retry_generation(generation_id, resync=True)
    assert resync_error.value.code == DriveSyncErrorCode.MANIFEST_REBUILD_REQUIRED.value
    assert str(resync_error.value) == "再同期前に旧保存先のManifestを再構築してください"
    blocked_record = repository.get_by_generation(generation_id)
    assert blocked_record is not None
    assert (blocked_record.remote_name, blocked_record.remote_base_path) == ("drive", "studio")
    blocked_targets = service.list_manifest_failure_targets()
    assert len(blocked_targets) == 1
    assert blocked_targets[0].remote_name == "drive"
    assert blocked_targets[0].remote_base_path == "studio"

    adapter.fail_relative_path = None
    assert service.retry_failed_manifests() == ("2026-08-08",)
    rebuilt_job = repository.claim_next_manifest("worker-1", 120)
    assert rebuilt_job is not None
    rebuilt = asyncio.run(service.process_manifest_job(rebuilt_job, "worker-1"))
    assert rebuilt is not None and rebuilt.status is DriveSyncStatus.SYNCED
    record = repository.get_by_generation(generation_id)
    assert record is not None
    assert record.status is DriveSyncStatus.SYNCED
    assert record.error_code is None
    assert service.list_manifest_failure_targets() == ()

    resynced, resync_job = service.retry_generation(generation_id, resync=True)
    assert resync_job is not None and resync_job.status is DriveSyncStatus.PENDING
    assert (resynced.remote_name, resynced.remote_base_path) == ("drive-b", "studio-b")


def test_manifest_enqueue_failure_recovers_using_stored_destination_after_config_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(tmp_path)
    assert service.enqueue_generation(generation_id) is not None
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    def fail_manifest_enqueue(job: object) -> None:
        del job
        raise DriveSyncRepositoryError("simulated manifest enqueue failure")

    with monkeypatch.context() as context:
        context.setattr(service._repository, "enqueue_manifest", fail_manifest_enqueue)
        synced = asyncio.run(service.process_job(claimed, "worker-1"))

    assert synced is not None and synced.status is DriveSyncStatus.SYNCED
    record = repository.get_by_generation(generation_id)
    assert record is not None
    assert record.error_code == DriveSyncErrorCode.MANIFEST_FAILED.value
    targets = service.list_manifest_failure_targets()
    assert len(targets) == 1
    assert targets[0].local_date == "2026-08-08"
    assert targets[0].remote_name == "drive"
    assert targets[0].remote_base_path == "studio"

    service._settings.rclone_remote = "drive-b"
    service._settings.rclone_base_path = "studio-b"
    with pytest.raises(DriveSyncServiceError) as resync_error:
        service.retry_generation(generation_id, resync=True)
    assert resync_error.value.code == DriveSyncErrorCode.MANIFEST_REBUILD_REQUIRED.value
    blocked_record = repository.get_by_generation(generation_id)
    assert blocked_record is not None
    assert (blocked_record.remote_name, blocked_record.remote_base_path) == ("drive", "studio")
    blocked_targets = service.list_manifest_failure_targets()
    assert len(blocked_targets) == 1
    assert blocked_targets[0].remote_name == "drive"
    assert blocked_targets[0].remote_base_path == "studio"
    assert service.retry_failed_manifests() == ("2026-08-08",)
    manifest_job = repository.claim_next_manifest("worker-1", 120)
    assert manifest_job is not None
    rebuilt = asyncio.run(service.process_manifest_job(manifest_job, "worker-1"))

    assert rebuilt is not None and rebuilt.status is DriveSyncStatus.SYNCED
    assert adapter.calls[-1][1] == DriveDestination("drive", "studio")
    record = repository.get_by_generation(generation_id)
    assert record is not None and record.error_code is None
    assert service.list_manifest_failure_targets() == ()

    resynced, resync_job = service.retry_generation(generation_id, resync=True)
    assert resync_job is not None and resync_job.status is DriveSyncStatus.PENDING
    assert (resynced.remote_name, resynced.remote_base_path) == ("drive-b", "studio-b")


def test_worker_processes_sync_and_manifest_without_gradio(tmp_path: Path) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(tmp_path)
    assert service.enqueue_generation(generation_id) is not None
    restarted_service = DriveSyncService(
        repository,
        service._generation_repository,
        service._artifact_repository,
        service._settings,
        adapter,
    )
    worker = DriveSyncWorker(repository, restarted_service, service._settings, worker_id="worker-1")

    assert asyncio.run(worker.run_once()) is True
    assert asyncio.run(worker.run_once()) is True
    assert [path for _, _, path in adapter.calls][-1].endswith("manifest.jsonl")
    record = repository.get_by_generation(generation_id)
    assert record is not None and record.status is DriveSyncStatus.SYNCED


def test_expired_drive_sync_lease_becomes_retryable_stale_without_copy(
    tmp_path: Path,
) -> None:
    service, repository, generation_id, _, _, adapter = _fixture(tmp_path)
    assert service.enqueue_generation(generation_id) is not None
    old_job = repository.claim_next("old-worker", 1)
    assert old_job is not None and old_job.claimed_at is not None
    assert repository.reconcile_stale(old_job.claimed_at + timedelta(seconds=2)) == 1

    restarted_service = DriveSyncService(
        repository,
        service._generation_repository,
        service._artifact_repository,
        service._settings,
        adapter,
    )
    worker = DriveSyncWorker(
        repository, restarted_service, service._settings, worker_id="new-worker"
    )

    assert asyncio.run(worker.run_once()) is False
    failed = repository.get_job(old_job.id)
    assert failed is not None
    assert failed.status is DriveSyncStatus.FAILED
    assert failed.error_code == DriveSyncErrorCode.STALE.value
    assert failed.retryable is True
    assert adapter.calls == []


class _MetadataRepairView:
    def __init__(self, repository) -> None:
        self._repository = repository
        self._hide_metadata_once = True

    def list_by_generation(self, generation_id):
        artifacts = self._repository.list_by_generation(generation_id)
        if self._hide_metadata_once:
            self._hide_metadata_once = False
            return tuple(
                artifact for artifact in artifacts if artifact.artifact_type is ArtifactType.IMAGE
            )
        return artifacts


def test_metadata_repair_is_used_before_drive_enqueue(tmp_path: Path) -> None:
    service, _, generation_id, _, _, _ = _fixture(tmp_path)
    service._artifact_repository = _MetadataRepairView(service._artifact_repository)
    repaired: list[UUID] = []
    service._metadata_repair_handler = lambda value: repaired.append(value)

    record = service.enqueue_generation(generation_id)

    assert record is not None
    assert record.status is DriveSyncStatus.PENDING
    assert repaired == [generation_id]


def test_manifest_ui_handler_registers_request_and_returns_stable_outputs(tmp_path: Path) -> None:
    service, repository, _, _, _, _ = _fixture(tmp_path)
    handler = make_drive_manifest_handler(service)

    outputs = handler("2026-08-08")

    assert len(outputs) == 2
    queued = repository.claim_next_manifest("worker-1", 120)
    assert queued is not None
    assert queued.local_date == "2026-08-08"


def test_manifest_ui_handler_hides_internal_errors() -> None:
    class BrokenService:
        def enqueue_manifest_rebuild(self, local_date=None):
            del local_date
            raise RuntimeError("raw stderr / absolute path / RCLONE_CONFIG")

    outputs = make_drive_manifest_handler(BrokenService())("2026-08-08")

    assert len(outputs) == 2
    assert "RCLONE_CONFIG" not in str(outputs)
    assert "absolute path" not in str(outputs)


def test_resync_ui_handler_shows_safe_manifest_rebuild_message() -> None:
    class BlockedService:
        def resync_synced(self):
            raise DriveSyncServiceError(
                DriveSyncErrorCode.MANIFEST_REBUILD_REQUIRED.value,
                "再同期前に旧保存先のManifestを再構築してください",
                retryable=False,
            )

    outputs = make_drive_resync_handler(BlockedService())()

    assert len(outputs) == 2
    assert outputs[1] == "再同期前に旧保存先のManifestを再構築してください"
    assert "/" not in outputs[1]


@pytest.mark.parametrize("date_symlink", [False, True])
def test_manifest_symlink_escape_is_rejected_without_writing_outside(
    tmp_path: Path, date_symlink: bool
) -> None:
    service, _, _, _, _, _ = _fixture(tmp_path)
    outside = tmp_path.parent / f"outside-manifest-{uuid4().hex}"
    outside.mkdir()
    manifest_root = tmp_path / ".drive-sync-manifests"
    try:
        if date_symlink:
            manifest_root.mkdir()
            (manifest_root / "2026-08-08").symlink_to(outside, target_is_directory=True)
        else:
            manifest_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(DriveSyncServiceError):
        service._write_manifest("2026-08-08", DriveDestination("drive", "studio"))
    assert not (outside / "manifest.jsonl").exists()


class _MarkSyncedFailureRepository(DriveSyncRepository):
    def mark_synced(self, job_id, worker_id, synced_at):
        del job_id, worker_id, synced_at
        raise DriveSyncRepositoryError("simulated mark_synced failure")


def test_database_failure_after_remote_success_does_not_delete_sources_or_remote(
    tmp_path: Path,
) -> None:
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
    image_path = tmp_path / "generations" / "image.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image_bytes)
    image = GenerationArtifact(
        id=uuid4(),
        generation_id=generation_id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/image.png",
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
    metadata_bytes = json.dumps(
        {"schema_version": 1, "generation_id": str(generation_id)}, separators=(",", ":")
    ).encode("utf-8")
    metadata_path = tmp_path / "generations" / "image.json"
    metadata_path.write_bytes(metadata_bytes)
    artifacts.add(
        GenerationArtifact(
            id=uuid4(),
            generation_id=generation_id,
            artifact_type=ArtifactType.METADATA,
            local_path="generations/image.json",
            sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            size_bytes=len(metadata_bytes),
            width=None,
            height=None,
            mime_type="application/json",
            created_at=created_at,
        )
    )
    repository = _MarkSyncedFailureRepository(factory)
    adapter = FakeDriveAdapter()
    service = DriveSyncService(
        repository,
        GenerationRepository(factory),
        artifacts,
        settings,
        adapter,
    )
    record = service.enqueue_generation(generation_id)
    assert record is not None
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    result = asyncio.run(service.process_job(claimed, "worker-1"))

    assert result is not None and result.status is DriveSyncStatus.FAILED
    assert image_path.exists()
    assert metadata_path.exists()
    assert len(adapter.calls) == 2
    retried, retry_job = service.retry_generation(generation_id)
    assert retry_job is not None
    assert (retried.remote_name, retried.remote_base_path) == ("drive", "studio")


class _FakeStream:
    def __init__(self, lines: tuple[bytes, ...]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        return self._lines.pop(0) if self._lines else b""


class _SlowProcess:
    def __init__(self) -> None:
        self.pid = 9898
        self.returncode = 0
        self.stdout = _FakeStream((b'{"bytes":1,"totalBytes":4,"percentage":25}\n',))
        self.stderr = _FakeStream(
            (
                b"token=secret\n",
                b'{"access_token":"secret"}\n',
                b'{"client_secret":"secret"}\n',
                b"Authorization: Bearer secret\n",
                b"password: secret\n",
                b"RCLONE_CONFIG=/secret/path\n",
            )
        )
        self.killed = False

    async def wait(self) -> int:
        if not self.killed:
            await asyncio.sleep(0.03)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _CompletedProcess:
    def __init__(
        self,
        *,
        returncode: int,
        stdout_lines: tuple[bytes, ...] = (),
        stderr_lines: tuple[bytes, ...] = (),
    ) -> None:
        self.pid = 9899
        self.returncode = returncode
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines)
        self.killed = False

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _drive_adapter_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        rclone_remote="drive",
        rclone_base_path="studio",
        rclone_config=tmp_path / "rclone.conf",
    )


def test_copy_file_nonzero_rclone_exit_raises_safe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _CompletedProcess(
        returncode=1,
        stderr_lines=(b"Error: unknown flag: --stats-one-line-json\n",),
    )

    async def create_process(*args, **kwargs):
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    adapter = GoogleDriveAdapter(_drive_adapter_settings(tmp_path))

    with pytest.raises(GoogleDriveAdapterError) as caught:
        asyncio.run(
            adapter.copy_file(
                tmp_path / "image.png",
                DriveDestination("drive", "studio"),
                "20260808_x1/image.png",
            )
        )

    assert caught.value.code == DriveSyncErrorCode.TRANSFER_FAILED.value


def test_copy_file_executable_not_found_raises_stable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def create_process(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("rclone")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    adapter = GoogleDriveAdapter(_drive_adapter_settings(tmp_path))

    with pytest.raises(GoogleDriveAdapterError) as caught:
        asyncio.run(
            adapter.copy_file(
                tmp_path / "image.png",
                DriveDestination("drive", "studio"),
                "20260808_x1/image.png",
            )
        )

    assert caught.value.code == DriveSyncErrorCode.RCLONE_NOT_FOUND.value


def test_progress_callback_failure_does_not_fail_successful_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _CompletedProcess(
        returncode=0,
        stdout_lines=(b'{"level":"notice","stats":{"bytes":1,"totalBytes":4}}\n',),
    )

    async def create_process(*args, **kwargs):
        del args, kwargs
        return process

    callback_calls: list[DriveSyncProgress] = []

    def progress_callback(progress: DriveSyncProgress) -> None:
        callback_calls.append(progress)
        raise RuntimeError("progress persistence failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    adapter = GoogleDriveAdapter(_drive_adapter_settings(tmp_path))

    asyncio.run(
        adapter.copy_file(
            tmp_path / "image.png",
            DriveDestination("drive", "studio"),
            "20260808_x1/image.png",
            progress_callback=progress_callback,
            total_bytes=4,
        )
    )

    assert len(callback_calls) == 1
    assert process.returncode == 0


def test_connection_timeout_is_separate_from_transfer_and_progress_is_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processes: list[_SlowProcess] = []

    async def create_process(*args, **kwargs):
        del args, kwargs
        process = _SlowProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        rclone_remote="drive",
        rclone_base_path="studio",
        rclone_connection_timeout_seconds=0.01,
        rclone_transfer_timeout_seconds=None,
    )
    adapter = GoogleDriveAdapter(settings)
    connection = asyncio.run(adapter.check_connection())
    assert connection.status is DriveConnectionStatus.FAILED

    progress: list[DriveSyncProgress] = []
    started: list[int] = []
    finished: list[bool] = []
    asyncio.run(
        adapter.copy_file(
            tmp_path / "image.png",
            DriveDestination("drive", "studio"),
            "20260808_x1/image.png",
            progress_callback=progress.append,
            total_bytes=4,
            process_started_callback=started.append,
            process_finished_callback=lambda: finished.append(True),
            log_path="logs/drive_sync/timeout-test.log",
        )
    )

    assert len(processes) == 2
    assert started == [9898]
    assert finished == [True]
    assert [event.progress_percentage for event in progress] == [25.0]
    log = tmp_path / "logs" / "drive_sync" / "timeout-test.log"
    assert log.exists()
    contents = log.read_text(encoding="utf-8")
    assert "secret" not in contents
    assert "token" not in contents
    assert "access_token" not in contents
    assert "client_secret" not in contents
    assert "Authorization" not in contents
    assert "password" not in contents
    assert "Bearer" not in contents
    assert "/secret/path" not in contents
    assert "RCLONE_CONFIG" not in contents
    assert "operation_started=connection" not in contents
    assert "operation_started=copyto" in contents
    assert "pid_started=true" in contents
    assert "progress_bytes=1 total_bytes=4 percentage=25.0 current_artifact=unknown" in contents
    assert "operation_finished=copyto" in contents
