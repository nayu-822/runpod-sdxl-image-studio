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
from runpod_sdxl_image_studio.adapters.database.repositories import (
    generation_repository as repository_module,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
    GenerationJobRepository,
    GenerationRepository,
    GenerationRepositoryError,
)
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
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
from runpod_sdxl_image_studio.domain.lora import LoraSetting
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
        GenerationCompletionRepository(factory),
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
    settings, engine, repository, artifacts, _, jobs = _repositories(tmp_path)
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


def test_completion_repository_commits_required_rows_atomically(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))
    repository.mark_queued(generation.id, "prompt-atomic")
    jobs.update_prompt_id(job.id, "prompt-atomic")
    image_artifact = GenerationArtifact(
        id=UUID(int=202),
        generation_id=generation.id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/2026-07-26/generated/atomic.png",
        sha256="b" * 64,
        size_bytes=20,
        width=1024,
        height=832,
        mime_type="image/png",
        created_at=datetime.now(UTC),
    )

    completion.complete_generation(generation.id, job.id, image_artifact)

    assert repository.get_by_id(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert jobs.get_by_generation(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert artifacts.get_primary_image(generation.id) == image_artifact

    rollback_generation = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings())
    )
    rollback_job = jobs.create(GenerationJob(rollback_generation.id, GenerationStatus.PENDING))
    repository.mark_queued(rollback_generation.id, "prompt-rollback")
    jobs.update_prompt_id(rollback_job.id, "prompt-rollback")

    def fail_job_completion(row: object, completed_at: datetime) -> None:
        del row, completed_at
        raise GenerationRepositoryError("forced transaction failure")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(repository_module, "_mark_job_completed", fail_job_completion)
    try:
        with pytest.raises(GenerationRepositoryError):
            completion.complete_generation(
                rollback_generation.id,
                rollback_job.id,
                GenerationArtifact(
                    id=UUID(int=203),
                    generation_id=rollback_generation.id,
                    artifact_type=ArtifactType.IMAGE,
                    local_path=image_artifact.local_path,
                    sha256=image_artifact.sha256,
                    size_bytes=image_artifact.size_bytes,
                    width=image_artifact.width,
                    height=image_artifact.height,
                    mime_type=image_artifact.mime_type,
                    created_at=image_artifact.created_at,
                ),
            )
    finally:
        monkeypatch.undo()
    assert artifacts.get_primary_image(rollback_generation.id) is None
    assert repository.get_by_id(rollback_generation.id).status is GenerationStatus.QUEUED  # type: ignore[union-attr]
    assert jobs.get_by_generation(rollback_generation.id).status is GenerationStatus.QUEUED  # type: ignore[union-attr]
    engine.dispose()


def test_generation_repository_rejects_invalid_parent_and_migration_round_trip(
    tmp_path: Path,
) -> None:
    settings, engine, repository, _, _, _ = _repositories(tmp_path)
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
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
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


def test_restore_distinguishes_unverified_capabilities_from_empty_lists(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, _, _ = _repositories(tmp_path)
    snapshot_settings = _settings().model_copy(
        update={"loras": (LoraSetting(name="style.safetensors", model_strength=0.7, order=0),)}
    )
    generation = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(snapshot_settings)
    )
    service = GenerationHistoryService(repository, artifacts, settings)

    unverified = service.restore_settings(generation.id, max_loras=8)
    assert unverified.capability_unverified is True
    assert unverified.warnings == (
        "現在のComfyUI一覧を取得していないため、モデルの存在確認は行っていません。",
    )

    verified_empty = service.restore_settings(
        generation.id,
        checkpoints=(),
        vaes=(),
        loras=(),
        max_loras=8,
    )
    assert verified_empty.capability_unverified is False
    assert any("checkpoint" in warning for warning in verified_empty.warnings)
    assert any("VAE" in warning for warning in verified_empty.warnings)
    assert any("LoRA" in warning for warning in verified_empty.warnings)
    engine.dispose()


@pytest.mark.asyncio
async def test_generation_service_persists_pending_failure_before_prompt(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
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
        completion_repository=completion,
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
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
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
        completion_repository=completion,
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
async def test_required_completion_failure_keeps_image_but_never_returns_success(
    tmp_path: Path,
) -> None:
    settings, engine, repository, artifacts, _, jobs = _repositories(tmp_path)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    persisted_job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))
    repository.mark_queued(generation.id, "prompt-required-failure")
    jobs.update_prompt_id(persisted_job.id, "prompt-required-failure")
    image_bytes = BytesIO()
    Image.new("RGB", (8, 4), "green").save(image_bytes, format="PNG")
    storage = LocalStorageAdapter(settings)
    stored = storage.store_image(image_bytes.getvalue(), generation.id, generation.created_at)

    class FailingCompletion:
        def complete_generation(
            self,
            generation_id: UUID,
            job_id: UUID,
            image_artifact: GenerationArtifact,
            completed_at: datetime,
        ) -> None:
            del generation_id, job_id, image_artifact, completed_at
            raise GenerationRepositoryError("database is unavailable")

    service = GenerationService(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        storage,
        lambda: None,  # type: ignore[arg-type]
        settings,
        generation_repository=repository,
        artifact_repository=artifacts,
        completion_repository=FailingCompletion(),  # type: ignore[arg-type]
        job_repository=jobs,
    )
    job = GenerationJob(
        generation_id=generation.id,
        status=GenerationStatus.QUEUED,
        id=persisted_job.id,
        prompt_id="prompt-required-failure",
        stored_image=stored,
    )

    with pytest.raises(GenerationRepositoryError):
        service._complete_job(
            job, _settings(), generation.created_at, GenerationKind.STANDARD, None
        )
    assert stored.path.exists()
    assert artifacts.get_primary_image(generation.id) is None
    assert repository.get_by_id(generation.id).status is GenerationStatus.QUEUED  # type: ignore[union-attr]
    assert jobs.get_by_generation(generation.id).status is GenerationStatus.QUEUED  # type: ignore[union-attr]
    engine.dispose()


@pytest.mark.asyncio
async def test_optional_thumbnail_failure_does_not_fail_completed_generation(
    tmp_path: Path,
) -> None:
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    persisted_job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))
    repository.mark_queued(generation.id, "prompt-optional-failure")
    jobs.update_prompt_id(persisted_job.id, "prompt-optional-failure")
    image_bytes = BytesIO()
    Image.new("RGB", (8, 4), "yellow").save(image_bytes, format="PNG")
    storage = LocalStorageAdapter(settings)
    stored = storage.store_image(image_bytes.getvalue(), generation.id, generation.created_at)

    class FailingThumbnail:
        def save(self, image_path: Path, generation_id: UUID, created_at: datetime) -> Path:
            del image_path, generation_id, created_at
            raise StorageError("thumbnail is unavailable")

    service = GenerationService(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        storage,
        lambda: None,  # type: ignore[arg-type]
        settings,
        generation_repository=repository,
        artifact_repository=artifacts,
        completion_repository=completion,
        job_repository=jobs,
        thumbnail_storage=FailingThumbnail(),  # type: ignore[arg-type]
    )
    job = GenerationJob(
        generation_id=generation.id,
        status=GenerationStatus.QUEUED,
        id=persisted_job.id,
        prompt_id="prompt-optional-failure",
        stored_image=stored,
    )

    service._complete_job(job, _settings(), generation.created_at, GenerationKind.STANDARD, None)
    assert job.status is GenerationStatus.COMPLETED
    assert repository.get_by_id(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert jobs.get_by_generation(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert {item.artifact_type for item in artifacts.list_by_generation(generation.id)} == {
        ArtifactType.IMAGE
    }
    engine.dispose()


@pytest.mark.asyncio
async def test_recovery_marks_stale_pending_and_comfyui_failure_without_resubmitting(
    tmp_path: Path,
) -> None:
    settings, engine, repository, artifacts, _, jobs = _repositories(tmp_path)
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


@pytest.mark.asyncio
async def test_recovery_reconciles_existing_primary_artifact_idempotently(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
    created_at = datetime(2026, 7, 26, 1, 0, tzinfo=UTC)
    generation = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()), created_at=created_at
    )
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING, created_at=created_at))
    repository.mark_queued(generation.id, "prompt-existing")
    jobs.update_prompt_id(job.id, "prompt-existing")
    image_bytes = BytesIO()
    Image.new("RGB", (8, 4), "blue").save(image_bytes, format="PNG")
    stored = LocalStorageAdapter(settings).store_image(
        image_bytes.getvalue(), generation.id, created_at
    )
    artifacts.add(
        GenerationArtifact(
            id=UUID(int=303),
            generation_id=generation.id,
            artifact_type=ArtifactType.IMAGE,
            local_path=stored.path.relative_to(settings.data_dir).as_posix(),
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            width=stored.width,
            height=stored.height,
            mime_type=stored.mime_type,
            created_at=created_at,
        )
    )

    class Client:
        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            raise AssertionError(f"recovery must not redownload {prompt_id}")

    async def capabilities() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(False, "unused", None)

    service = GenerationService(
        Client(),  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        object(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capabilities,
        settings,
        generation_repository=repository,
        artifact_repository=artifacts,
        completion_repository=completion,
        job_repository=jobs,
    )

    assert await service.recover_prompt(generation.id, "prompt-existing") is True
    assert await service.recover_prompt(generation.id, "prompt-existing") is True
    assert repository.get_by_id(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert jobs.get_by_generation(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert len(artifacts.list_by_generation(generation.id)) == 1
    engine.dispose()
