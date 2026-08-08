"""Phase 7 Alembic boundary checks."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from PIL import Image
from sqlalchemy import create_engine, inspect

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIOutputImage,
    PromptHistory,
    QueuedPrompt,
)
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepository,
    DriveSyncRepositoryError,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
    GenerationFailureRepository,
    GenerationJobRepository,
    GenerationQueueRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.adapters.storage.generation_metadata_storage import (
    GenerationMetadataStorage,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveConnectionResult,
    DriveConnectionStatus,
    DriveSyncStatus,
)
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.drive_sync_service import DriveSyncService
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template

ROOT = Path(__file__).parents[2]


class _NoopDriveAdapter:
    async def check_connection(self) -> DriveConnectionResult:
        return DriveConnectionResult(DriveConnectionStatus.CONNECTED, "connected")

    async def copy_file(self, *args, **kwargs) -> None:
        raise AssertionError("generation completion must not copy to Drive")


class _FailingEnqueueRepository(DriveSyncRepository):
    def enqueue(self, record, job):
        del record, job
        raise DriveSyncRepositoryError("simulated enqueue failure")


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 64), "white").save(output, format="PNG")
    return output.getvalue()


def _generation_service_fixture(tmp_path: Path, repository_type=DriveSyncRepository):
    database_url = f"sqlite:///{(tmp_path / 'generation-service.sqlite3').as_posix()}"
    settings = Settings(
        _env_file=None,
        database_url=database_url,
        data_dir=tmp_path,
        rclone_remote="drive-a",
        rclone_base_path="studio-a",
        history_poll_interval_seconds=0.001,
        generation_timeout_seconds=5,
        max_output_image_bytes=1_000_000,
        max_metadata_sidecar_bytes=1_000_000,
    )
    command.upgrade(_config(database_url), "head")
    engine = create_engine(database_url)
    factory = create_session_factory(engine)
    generations = GenerationRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    jobs = GenerationJobRepository(factory)
    queue = GenerationQueueRepository(factory)
    start = GenerationStartRepository(factory)
    progress = GenerationProgressRepository(factory)
    completion = GenerationCompletionRepository(factory)
    failure = GenerationFailureRepository(factory)
    drive_repository = repository_type(factory)
    drive_service = DriveSyncService(
        drive_repository,
        generations,
        artifacts,
        settings,
        _NoopDriveAdapter(),
    )

    class FakeClient:
        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del workflow, client_id
            return QueuedPrompt("generation-prompt", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                True,
                False,
                (ComfyUIOutputImage("generated.png", "", "output"),),
                None,
            )

        async def get_output_image(self, output: ComfyUIOutputImage) -> bytes:
            del output
            return _png()

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            del prompt_id, client_id
            if False:
                yield None

    capabilities = ComfyUICapabilities(
        checkpoints=("sdxl.safetensors",),
        vaes=(),
        samplers=("euler",),
        schedulers=("normal",),
        loras=(),
        upscale_models=(),
        available_node_classes=frozenset(
            {
                "CheckpointLoaderSimple",
                "CLIPTextEncode",
                "EmptyLatentImage",
                "KSampler",
                "VAEDecode",
                "SaveImage",
            }
        ),
        warnings=(),
    )

    async def capability_provider() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    service = GenerationService(
        FakeClient(),  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        FakeWebSocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capability_provider,
        settings,
        persistence=GenerationPersistenceRepositories(
            generation=generations,
            job=jobs,
            artifact=artifacts,
            start=start,
            queue=queue,
            progress=progress,
            completion=completion,
            failure=failure,
        ),
        metadata_storage=GenerationMetadataStorage(tmp_path),
        drive_sync_enqueue_handler=drive_service.enqueue_generation,
    )
    return engine, service, generations, jobs, artifacts, drive_repository


def _generation_settings() -> GenerationSettings:
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


def _completed_generation_for_drive(tmp_path: Path):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        rclone_remote="drive-a",
        rclone_base_path="studio-a",
        max_output_image_bytes=1_000_000,
        max_metadata_sidecar_bytes=1_000_000,
    )
    created_at = datetime(2026, 8, 7, 15, 30, tzinfo=UTC)
    generation_id, job_id = uuid4(), uuid4()
    generation_settings = GenerationSettings(
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
    GenerationStartRepository(factory).create_pending(
        GenerationSettingsSnapshot.from_settings(generation_settings),
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=created_at,
    )
    image_buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
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
    return engine, factory, settings, generation_id


def _config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_phase7_migration_adds_only_sync_tables_and_downgrades_cleanly(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'phase7.sqlite3').as_posix()}"
    command.upgrade(_config(database_url), "0012_phase6_legacy_metadata_candidates")
    engine = create_engine(database_url)
    before = inspect(engine)
    assert before.has_table("generations")
    assert before.has_table("metadata_imports")
    assert not before.has_table("drive_sync_records")
    assert not before.has_table("drive_sync_jobs")
    assert not before.has_table("drive_manifest_jobs")

    command.upgrade(_config(database_url), "0013_phase7_drive_sync")
    upgraded = inspect(engine)
    assert upgraded.has_table("drive_sync_records")
    assert upgraded.has_table("drive_sync_jobs")
    assert {column["name"] for column in upgraded.get_columns("drive_sync_records")} >= {
        "generation_id",
        "status",
        "remote_image_path",
        "remote_metadata_path",
        "image_sha256",
        "metadata_sha256",
    }
    assert {column["name"] for column in upgraded.get_columns("drive_sync_jobs")} >= {
        "sync_record_id",
        "queue_sequence",
        "progress_bytes",
        "total_bytes",
        "worker_id",
        "lease_expires_at",
    }

    command.upgrade(_config(database_url), "0014_phase7_drive_sync_hardening")
    hardened = inspect(engine)
    assert hardened.has_table("drive_manifest_jobs")
    assert {column["name"] for column in hardened.get_columns("drive_manifest_jobs")} >= {
        "local_date",
        "remote_name",
        "remote_base_path",
        "remote_manifest_path",
        "pid",
        "lease_expires_at",
    }

    command.downgrade(_config(database_url), "0013_phase7_drive_sync")
    after_hardening_downgrade = inspect(engine)
    assert not after_hardening_downgrade.has_table("drive_manifest_jobs")
    assert after_hardening_downgrade.has_table("drive_sync_records")
    assert after_hardening_downgrade.has_table("drive_sync_jobs")

    command.downgrade(_config(database_url), "0012_phase6_legacy_metadata_candidates")
    downgraded = inspect(engine)
    assert not downgraded.has_table("drive_sync_records")
    assert not downgraded.has_table("drive_sync_jobs")
    assert downgraded.has_table("generations")
    assert downgraded.has_table("metadata_imports")
    engine.dispose()


def test_generation_completion_and_drive_enqueue_are_separate_persistence_boundaries(
    tmp_path: Path,
) -> None:
    engine, factory, settings, generation_id = _completed_generation_for_drive(tmp_path)
    drive_repository = DriveSyncRepository(factory)
    service = DriveSyncService(
        drive_repository,
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        settings,
        _NoopDriveAdapter(),
    )

    record = service.enqueue_generation(generation_id)

    generation = GenerationRepository(factory).get_by_id(generation_id)
    job = GenerationJobRepository(factory).get_by_generation(generation_id)
    queued_job = drive_repository.list_jobs()[0]
    assert record is not None and record.status is DriveSyncStatus.PENDING
    assert generation is not None and generation.status.value == "completed"
    assert generation.completed_at is not None
    assert job is not None and job.status.value == "completed"
    assert job.completed_at is not None
    assert queued_job.status is DriveSyncStatus.PENDING
    assert (record.remote_name, record.remote_base_path) == ("drive-a", "studio-a")
    engine.dispose()


def test_drive_enqueue_failure_does_not_reopen_completed_generation(
    tmp_path: Path,
) -> None:
    engine, factory, settings, generation_id = _completed_generation_for_drive(tmp_path)
    failing_repository = _FailingEnqueueRepository(factory)
    service = DriveSyncService(
        failing_repository,
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        settings,
        _NoopDriveAdapter(),
    )

    try:
        service.enqueue_generation(generation_id)
    except DriveSyncRepositoryError:
        pass
    else:
        raise AssertionError("enqueue failure should be surfaced to the caller")

    generation = GenerationRepository(factory).get_by_id(generation_id)
    job = GenerationJobRepository(factory).get_by_generation(generation_id)
    assert generation is not None and generation.status.value == "completed"
    assert job is not None and job.status.value == "completed"
    engine.dispose()


def test_generation_service_completion_enqueues_drive_after_local_commit(tmp_path: Path) -> None:
    engine, service, generations, jobs, artifacts, drive_repository = _generation_service_fixture(
        tmp_path
    )
    try:
        result = asyncio.run(service.generate(_generation_settings()))

        assert result.status is GenerationStatus.COMPLETED
        assert result.stored_image is not None
        generation = generations.get_by_id(result.generation_id)
        job = jobs.get_by_generation(result.generation_id)
        persisted_artifacts = artifacts.list_by_generation(result.generation_id)
        drive_record = drive_repository.get_by_generation(result.generation_id)
        drive_jobs = drive_repository.list_jobs()

        assert generation is not None
        assert generation.status is GenerationStatus.COMPLETED
        assert generation.comfy_prompt_id == "generation-prompt"
        assert generation.completed_at is not None
        assert job is not None
        assert job.status is GenerationStatus.COMPLETED
        assert job.completed_at is not None
        assert (
            sum(artifact.artifact_type is ArtifactType.IMAGE for artifact in persisted_artifacts)
            == 1
        )
        assert (
            sum(artifact.artifact_type is ArtifactType.METADATA for artifact in persisted_artifacts)
            == 1
        )
        assert result.stored_image.path.exists()
        assert drive_record is not None and drive_record.status is DriveSyncStatus.PENDING
        assert len(drive_jobs) == 1
        assert drive_jobs[0].status is DriveSyncStatus.PENDING
    finally:
        engine.dispose()


def test_generation_service_drive_enqueue_failure_keeps_completed_pair_and_image(
    tmp_path: Path,
) -> None:
    engine, service, generations, jobs, artifacts, drive_repository = _generation_service_fixture(
        tmp_path, _FailingEnqueueRepository
    )
    try:
        result = asyncio.run(service.generate(_generation_settings()))

        assert result.status is GenerationStatus.COMPLETED
        assert result.stored_image is not None
        generation = generations.get_by_id(result.generation_id)
        job = jobs.get_by_generation(result.generation_id)
        image_artifacts = tuple(
            artifact
            for artifact in artifacts.list_by_generation(result.generation_id)
            if artifact.artifact_type is ArtifactType.IMAGE
        )

        assert generation is not None and generation.status is GenerationStatus.COMPLETED
        assert generation.completed_at is not None
        assert job is not None and job.status is GenerationStatus.COMPLETED
        assert job.completed_at is not None
        assert len(image_artifacts) == 1
        assert result.stored_image.path.exists()
        assert drive_repository.get_by_generation(result.generation_id) is None
    finally:
        engine.dispose()
