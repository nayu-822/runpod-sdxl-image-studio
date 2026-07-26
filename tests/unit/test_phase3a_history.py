from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from PIL import Image
from sqlalchemy import inspect

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIOutputImage,
    PromptHistory,
    QueuedPrompt,
)
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationJobRepository,
    GenerationRepository,
    GenerationRepositoryError,
)
from runpod_sdxl_image_studio.adapters.storage.generation_metadata_storage import (
    GenerationMetadataStorage,
)
from runpod_sdxl_image_studio.adapters.storage.history_thumbnail_storage import (
    HistoryThumbnailStorage,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database
from runpod_sdxl_image_studio.domain.generation import (
    GenerationKind,
    GenerationProgress,
    GenerationStatus,
)
from runpod_sdxl_image_studio.domain.generation_artifact import (
    ArtifactType,
    GenerationArtifact,
)
from runpod_sdxl_image_studio.domain.generation_history import GenerationHistoryFilter
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import (
    GenerationSettingsSnapshot,
    SnapshotError,
)
from runpod_sdxl_image_studio.domain.job import GenerationJob
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.generation_history_service import (
    GenerationHistoryService,
)
from runpod_sdxl_image_studio.services.generation_recovery_service import (
    GenerationRecoveryService,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template


def _settings() -> GenerationSettings:
    return GenerationSettings(
        positive_prompt="猫、青い目",
        negative_prompt="low quality",
        seed=123,
        width=1024,
        height=832,
        steps=28,
        cfg_scale=5.5,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="sdxl.safetensors",
        vae_name="vae.safetensors",
    )


def _repositories(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_url=f"sqlite:///{(tmp_path / 'history.sqlite3').as_posix()}",
    )
    upgrade_database(settings, Path(__file__).parents[2])
    engine = create_image_studio_engine(settings)
    factory = create_session_factory(engine)
    return (
        settings,
        engine,
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        GenerationJobRepository(factory),
    )


def test_snapshot_round_trip_rejects_unknown_version_and_corrupt_json() -> None:
    snapshot = GenerationSettingsSnapshot.from_settings(_settings())
    restored = GenerationSettingsSnapshot.from_json(snapshot.to_json())

    assert restored == snapshot
    assert restored.to_generation_settings() == _settings()
    with pytest.raises(SnapshotError):
        GenerationSettingsSnapshot.from_json('{"schema_version": 99}')
    with pytest.raises(SnapshotError):
        GenerationSettingsSnapshot.from_json("not-json")


def test_generation_repository_persists_transitions_parent_and_filters(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, jobs = _repositories(tmp_path)
    generation = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        created_at=datetime(2026, 7, 26, 1, 0, tzinfo=UTC),
    )
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))
    queued = repository.mark_queued(generation.id, "prompt-1")
    jobs.update_prompt_id(job.id, "prompt-1")
    running = repository.mark_running(generation.id)
    jobs.update_progress(job.id, 3, 28, "KSampler")
    completed = repository.mark_completed(generation.id)
    jobs.mark_completed(job.id)

    artifact = artifacts.add(
        GenerationArtifact(
            id=UUID(int=101),
            generation_id=generation.id,
            artifact_type=ArtifactType.IMAGE,
            local_path="generations/2026-07-26/generated/image.png",
            sha256="a" * 64,
            size_bytes=10,
            width=1024,
            height=832,
            mime_type="image/png",
            created_at=datetime.now(UTC),
        )
    )
    assert queued.status is GenerationStatus.QUEUED
    assert running.status is GenerationStatus.RUNNING
    assert completed.status is GenerationStatus.COMPLETED
    assert artifacts.get_primary_image(generation.id) == artifact
    with pytest.raises(GenerationRepositoryError):
        repository.mark_running(generation.id)
    assert artifacts.add(artifact) == artifact

    parent = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    derived = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        kind=GenerationKind.DERIVED,
        parent_generation_id=parent.id,
    )
    assert derived.parent_generation_id == parent.id
    page = repository.list_history(GenerationHistoryFilter(limit=1, favorite=False))
    assert page.page_size == 1
    assert page.has_next is True
    assert page.generations
    engine.dispose()
    del settings


def test_generation_repository_rejects_invalid_parent_and_migration_round_trip(
    tmp_path: Path,
) -> None:
    settings, engine, repository, _, _ = _repositories(tmp_path)
    with pytest.raises(GenerationRepositoryError):
        repository.create_pending(
            GenerationSettingsSnapshot.from_settings(_settings()),
            parent_generation_id=UUID(int=999),
        )
    inspector = inspect(engine)
    assert {"generations", "generation_artifacts", "generation_jobs"} <= {
        table for table in inspector.get_table_names() if table != "alembic_version"
    }
    engine.dispose()
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option("script_location", str(Path(__file__).parents[2] / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url or "")
    command.downgrade(config, "0001_lora_metadata")
    downgraded = create_image_studio_engine(settings)
    assert not inspect(downgraded).has_table("generations")
    assert inspect(downgraded).has_table("lora_metadata")
    downgraded.dispose()
    command.upgrade(config, "head")


def test_history_service_converts_details_and_updates_user_fields(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, jobs = _repositories(tmp_path)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    repository.mark_queued(generation.id, "prompt-history")
    repository.mark_completed(generation.id)
    service = GenerationHistoryService(repository, artifacts, settings)

    detail = service.get_detail(generation.id)
    assert detail.generation_id == str(generation.id)
    assert detail.snapshot.seed == 123
    assert service.restore_settings(
        generation.id,
        checkpoints=("other.safetensors",),
        vaes=("other.vae",),
        loras=(),
        max_loras=8,
    ).warnings
    updated = service.set_favorite(generation.id, True)
    assert updated.favorite is True
    updated = service.update_note(generation.id, "メモ\n2行目")
    assert updated.user_note == "メモ\n2行目"
    assert jobs.list_recoverable() == ()
    engine.dispose()


@pytest.mark.asyncio
async def test_generation_service_persists_pending_failure_before_prompt(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, jobs = _repositories(tmp_path)
    called = False

    class Client:
        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del workflow, client_id
            nonlocal called
            called = True
            raise AssertionError("prompt must not be sent")

    async def capabilities() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(False, "offline", None)

    service = GenerationService(
        Client(),  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        object(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capabilities,
        settings,
        generation_repository=repository,
        artifact_repository=artifacts,
        job_repository=jobs,
    )
    result = await service.generate(_settings())

    assert result.status is GenerationStatus.FAILED
    assert called is False
    persisted = repository.get_by_id(result.generation_id)
    assert persisted is not None and persisted.status is GenerationStatus.FAILED
    assert jobs.list_recoverable() == ()
    engine.dispose()


@pytest.mark.asyncio
async def test_generation_service_persists_artifacts_and_snapshot(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, jobs = _repositories(tmp_path)
    image = BytesIO()
    Image.new("RGB", (8, 4), "red").save(image, format="PNG")
    image_bytes = image.getvalue()
    capabilities_value = ComfyUICapabilities(
        checkpoints=("sdxl.safetensors",),
        vaes=("vae.safetensors",),
        samplers=("euler",),
        schedulers=("normal",),
        loras=(),
        upscale_models=(),
        available_node_classes=frozenset({"VAELoader"}),
        warnings=(),
    )

    class Client:
        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del workflow, client_id
            return QueuedPrompt("prompt-success", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                True,
                False,
                (ComfyUIOutputImage("image.png", "", "output"),),
                None,
            )

        async def get_output_image(self, output: ComfyUIOutputImage) -> bytes:
            del output
            return image_bytes

    class Websocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            del prompt_id, client_id
            yield GenerationProgress(state=GenerationStatus.RUNNING, value=1, maximum=2)

    async def capabilities() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities_value)

    service = GenerationService(
        Client(),  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        Websocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capabilities,
        settings,
        generation_repository=repository,
        artifact_repository=artifacts,
        job_repository=jobs,
        thumbnail_storage=HistoryThumbnailStorage(settings),
        metadata_storage=GenerationMetadataStorage(settings.data_dir),
    )
    result = await service.generate(_settings())

    assert result.status is GenerationStatus.COMPLETED
    persisted = repository.get_by_id(result.generation_id)
    assert persisted is not None and persisted.settings_snapshot.seed == 123
    assert {
        artifact.artifact_type for artifact in artifacts.list_by_generation(result.generation_id)
    } == {ArtifactType.IMAGE, ArtifactType.THUMBNAIL, ArtifactType.METADATA}
    assert jobs.list_recoverable() == ()
    engine.dispose()


@pytest.mark.asyncio
async def test_recovery_marks_stale_pending_and_comfyui_failure_without_resubmitting(
    tmp_path: Path,
) -> None:
    settings, engine, repository, artifacts, jobs = _repositories(tmp_path)
    stale_time = datetime(2026, 7, 26, 0, 0, tzinfo=UTC)
    stale = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()), created_at=stale_time
    )
    stale_job = jobs.create(
        GenerationJob(stale.id, GenerationStatus.PENDING, created_at=stale_time)
    )
    failed = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    failed_job = jobs.create(GenerationJob(failed.id, GenerationStatus.PENDING))
    repository.mark_queued(failed.id, "prompt-failed")
    jobs.update_prompt_id(failed_job.id, "prompt-failed")

    class Client:
        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            assert prompt_id == "prompt-failed"
            return PromptHistory("prompt-failed", False, True, (), "execution failed")

    recovery = GenerationRecoveryService(
        Client(),  # type: ignore[arg-type]
        repository,
        jobs,
        artifacts,
        settings,
    )
    messages = await recovery.recover(datetime(2026, 7, 26, 1, 0, tzinfo=UTC))

    assert str(stale.id) in " ".join(messages)
    assert repository.get_by_id(stale.id).status is GenerationStatus.FAILED  # type: ignore[union-attr]
    assert repository.get_by_id(failed.id).status is GenerationStatus.FAILED  # type: ignore[union-attr]
    assert jobs.list_recoverable() == ()
    assert stale_job.id != failed_job.id
    engine.dispose()
