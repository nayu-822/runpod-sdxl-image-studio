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
    generation_progress_repository as progress_module,
)
from runpod_sdxl_image_studio.adapters.database.repositories import (
    generation_repository as repository_module,
)
from runpod_sdxl_image_studio.adapters.database.repositories import (
    generation_start_repository as start_module,
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
from runpod_sdxl_image_studio.services.generation_errors import (
    ArtifactPersistenceError,
    CompletionPersistenceError,
    FailurePersistenceError,
    GenerationPersistenceError,
    PromptPersistenceError,
    RecoveryPersistenceError,
    persistence_error_code,
    persistence_error_message,
)
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


def test_queue_repository_persists_prompt_and_status_for_both_rows_atomically(
    tmp_path: Path,
) -> None:
    settings, engine, repository, _, _, jobs = _repositories(tmp_path)
    queue = repository_module.GenerationQueueRepository(repository._session_factory)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))

    queue.mark_queued(generation.id, job.id, "prompt-queue")
    queue.mark_queued(generation.id, job.id, "prompt-queue")

    persisted_generation = repository.get_by_id(generation.id)
    persisted_job = jobs.get_by_generation(generation.id)
    assert persisted_generation is not None
    assert persisted_job is not None
    assert persisted_generation.status is GenerationStatus.QUEUED
    assert persisted_job.status is GenerationStatus.QUEUED
    assert persisted_generation.comfy_prompt_id == "prompt-queue"
    assert persisted_job.prompt_id == "prompt-queue"
    engine.dispose()


def test_queue_repository_rolls_back_both_rows_when_job_update_fails(tmp_path: Path) -> None:
    settings, engine, repository, _, _, jobs = _repositories(tmp_path)
    queue = repository_module.GenerationQueueRepository(repository._session_factory)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))

    def fail_job_queue(row: object, prompt_id: str) -> None:
        del row, prompt_id
        raise GenerationRepositoryError("forced job queue failure")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(repository_module, "_mark_job_queued", fail_job_queue)
    try:
        with pytest.raises(GenerationRepositoryError):
            queue.mark_queued(generation.id, job.id, "prompt-rollback")
    finally:
        monkeypatch.undo()

    persisted_generation = repository.get_by_id(generation.id)
    persisted_job = jobs.get_by_generation(generation.id)
    assert persisted_generation is not None
    assert persisted_job is not None
    assert persisted_generation.status is GenerationStatus.PENDING
    assert persisted_job.status is GenerationStatus.PENDING
    assert persisted_generation.comfy_prompt_id is None
    assert persisted_job.prompt_id is None
    engine.dispose()


def test_queue_repository_rejects_prompt_id_reuse_without_changing_original_pair(
    tmp_path: Path,
) -> None:
    settings, engine, repository, _, _, jobs = _repositories(tmp_path)
    queue = repository_module.GenerationQueueRepository(repository._session_factory)
    first = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    first_job = jobs.create(GenerationJob(first.id, GenerationStatus.PENDING))
    second = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    second_job = jobs.create(GenerationJob(second.id, GenerationStatus.PENDING))
    queue.mark_queued(first.id, first_job.id, "prompt-unique")

    with pytest.raises(GenerationRepositoryError):
        queue.mark_queued(second.id, second_job.id, "prompt-unique")

    persisted_first = repository.get_by_id(first.id)
    persisted_second = repository.get_by_id(second.id)
    assert persisted_first is not None
    assert persisted_second is not None
    assert persisted_first.comfy_prompt_id == "prompt-unique"
    assert persisted_first.status is GenerationStatus.QUEUED
    assert persisted_second.comfy_prompt_id is None
    assert persisted_second.status is GenerationStatus.PENDING
    assert jobs.get_by_generation(second.id).prompt_id is None  # type: ignore[union-attr]
    engine.dispose()


def test_persistence_error_types_codes_and_messages_are_stable() -> None:
    errors = (
        (
            PromptPersistenceError("secret database detail"),
            "prompt_persistence_error",
            "生成要求はComfyUIへ送信されましたが、履歴へ関連付けできませんでした。"
            "同じ生成要求の再送信は行っていません。",
        ),
        (
            ArtifactPersistenceError("C:/private/database.sqlite3"),
            "artifact_persistence_error",
            "画像は保存されましたが、履歴へ画像情報を登録できませんでした。",
        ),
        (
            CompletionPersistenceError("sql traceback"),
            "completion_persistence_error",
            "画像は保存されましたが、履歴の完了状態を確定できませんでした。",
        ),
        (
            RecoveryPersistenceError("absolute path"),
            "recovery_persistence_error",
            "未完了生成の結果は確認できましたが、履歴の復旧状態を保存できませんでした。",
        ),
        (
            FailurePersistenceError("database URL"),
            "failure_persistence_error",
            "生成は失敗しましたが、履歴の失敗状態を完全に保存できませんでした。",
        ),
    )
    for error, expected_code, expected_message in errors:
        assert isinstance(error, GenerationPersistenceError)
        assert persistence_error_code(error) == expected_code
        assert persistence_error_message(error) == expected_message
        assert "secret" not in persistence_error_message(error)
        assert "database" not in persistence_error_message(error)


def test_failure_repository_updates_pending_queued_and_running_pairs_atomically(
    tmp_path: Path,
) -> None:
    settings, engine, repository, _, _, jobs = _repositories(tmp_path)
    failure = repository_module.GenerationFailureRepository(repository._session_factory)
    failed_at = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)

    for index, status in enumerate(
        (GenerationStatus.PENDING, GenerationStatus.QUEUED, GenerationStatus.RUNNING)
    ):
        generation = repository.create_pending(
            GenerationSettingsSnapshot.from_settings(_settings())
        )
        job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))
        if status is GenerationStatus.QUEUED:
            repository.mark_queued(generation.id, f"prompt-failure-{index}")
            jobs.update_prompt_id(job.id, f"prompt-failure-{index}")
        elif status is GenerationStatus.RUNNING:
            repository.mark_queued(generation.id, f"prompt-failure-{index}")
            jobs.update_prompt_id(job.id, f"prompt-failure-{index}")
            repository.mark_running(generation.id)
            jobs.update_progress(job.id, 1, 2, "KSampler")

        failure.fail_generation(
            generation.id,
            job.id,
            error_code="comfyui_execution_error",
            error_summary="ComfyUIで生成が失敗しました。",
            failed_at=failed_at,
        )
        persisted_generation = repository.get_by_id(generation.id)
        persisted_job = jobs.get_by_generation(generation.id)
        assert persisted_generation is not None
        assert persisted_job is not None
        assert persisted_generation.status is GenerationStatus.FAILED
        assert persisted_job.status is GenerationStatus.FAILED
        assert persisted_generation.error_code == persisted_job.error_code
        assert persisted_generation.error_summary == persisted_job.error_summary
        assert persisted_generation.completed_at == failed_at
        assert persisted_job.completed_at == failed_at

    engine.dispose()


def test_failure_repository_is_idempotent_and_preserves_first_failure(tmp_path: Path) -> None:
    settings, engine, repository, _, _, jobs = _repositories(tmp_path)
    failure = repository_module.GenerationFailureRepository(repository._session_factory)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))
    first_failed_at = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
    second_failed_at = datetime(2026, 7, 29, 13, 0, tzinfo=UTC)

    failure.fail_generation(
        generation.id,
        job.id,
        error_code="workflow_error",
        error_summary="生成設定を確認できませんでした。",
        failed_at=first_failed_at,
    )
    failure.fail_generation(
        generation.id,
        job.id,
        error_code="workflow_error",
        error_summary="生成設定を確認できませんでした。",
        failed_at=second_failed_at,
    )

    persisted_generation = repository.get_by_id(generation.id)
    persisted_job = jobs.get_by_generation(generation.id)
    assert persisted_generation is not None
    assert persisted_job is not None
    assert persisted_generation.completed_at == first_failed_at
    assert persisted_job.completed_at == first_failed_at
    with pytest.raises(GenerationRepositoryError):
        failure.fail_generation(
            generation.id,
            job.id,
            error_code="database_error",
            error_summary="別の失敗情報",
            failed_at=second_failed_at,
        )
    engine.dispose()


def test_failure_repository_rolls_back_when_job_update_fails(tmp_path: Path) -> None:
    settings, engine, repository, _, _, jobs = _repositories(tmp_path)
    failure = repository_module.GenerationFailureRepository(repository._session_factory)
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))

    def fail_job(row: object, **values: object) -> None:
        del row, values
        raise GenerationRepositoryError("forced job failure")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(repository_module, "_mark_job_failed", fail_job)
    try:
        with pytest.raises(GenerationRepositoryError):
            failure.fail_generation(
                generation.id,
                job.id,
                error_code="storage_error",
                error_summary="生成画像を保存できませんでした。",
                failed_at=datetime.now(UTC),
            )
    finally:
        monkeypatch.undo()

    persisted_generation = repository.get_by_id(generation.id)
    persisted_job = jobs.get_by_generation(generation.id)
    assert persisted_generation is not None
    assert persisted_job is not None
    assert persisted_generation.status is GenerationStatus.PENDING
    assert persisted_job.status is GenerationStatus.PENDING
    assert persisted_generation.error_code is None
    assert persisted_job.error_code is None
    engine.dispose()


def test_failure_repository_rejects_completed_and_inconsistent_pairs(tmp_path: Path) -> None:
    settings, engine, repository, _, _, jobs = _repositories(tmp_path)
    failure = repository_module.GenerationFailureRepository(repository._session_factory)
    completed = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    completed_job = jobs.create(GenerationJob(completed.id, GenerationStatus.PENDING))
    repository.mark_queued(completed.id, "prompt-completed")
    jobs.update_prompt_id(completed_job.id, "prompt-completed")
    repository.mark_completed(completed.id)
    jobs.mark_completed(completed_job.id)
    with pytest.raises(GenerationRepositoryError):
        failure.fail_generation(
            completed.id,
            completed_job.id,
            error_code="database_error",
            error_summary="不正な失敗更新",
            failed_at=datetime.now(UTC),
        )

    inconsistent = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    inconsistent_job = jobs.create(GenerationJob(inconsistent.id, GenerationStatus.PENDING))
    repository.mark_failed(inconsistent.id, "database_error", "Generationだけ失敗")
    with pytest.raises(GenerationRepositoryError):
        failure.fail_generation(
            inconsistent.id,
            inconsistent_job.id,
            error_code="database_error",
            error_summary="同じ失敗情報",
            failed_at=datetime.now(UTC),
        )
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
        queue_repository=repository_module.GenerationQueueRepository(repository._session_factory),
        failure_repository=repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
        start_repository=start_module.GenerationStartRepository(repository._session_factory),
        progress_repository=progress_module.GenerationProgressRepository(
            repository._session_factory
        ),
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
    events: list[str] = []
    prompt_calls = 0
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
            nonlocal prompt_calls
            prompt_calls += 1
            events.append("prompt")
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
            events.append("websocket")
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
        queue_repository=repository_module.GenerationQueueRepository(repository._session_factory),
        failure_repository=repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
        start_repository=start_module.GenerationStartRepository(repository._session_factory),
        progress_repository=progress_module.GenerationProgressRepository(
            repository._session_factory
        ),
        thumbnail_storage=HistoryThumbnailStorage(settings),
        metadata_storage=GenerationMetadataStorage(settings.data_dir),
    )
    result = await service.generate(_settings())

    assert result.status is GenerationStatus.COMPLETED
    persisted = repository.get_by_id(result.generation_id)
    persisted_job = jobs.get_by_generation(result.generation_id)
    assert persisted is not None and persisted.settings_snapshot.seed == 123
    assert persisted.comfy_prompt_id == "prompt-success"
    assert persisted_job is not None and persisted_job.prompt_id == "prompt-success"
    assert prompt_calls == 1
    assert events == ["prompt", "websocket"]
    assert {
        artifact.artifact_type for artifact in artifacts.list_by_generation(result.generation_id)
    } == {ArtifactType.IMAGE, ArtifactType.THUMBNAIL, ArtifactType.METADATA}
    assert jobs.list_recoverable() == ()
    engine.dispose()


@pytest.mark.asyncio
async def test_prompt_persistence_failure_stops_monitoring_without_resubmitting(
    tmp_path: Path,
) -> None:
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
    image_bytes = b"must not be downloaded"
    websocket_calls = 0
    history_calls = 0
    image_calls = 0

    class Client:
        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del workflow, client_id
            return QueuedPrompt("prompt-database-failure", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            del prompt_id
            nonlocal history_calls
            history_calls += 1
            raise AssertionError("history must not be requested")

        async def get_output_image(self, output: ComfyUIOutputImage) -> bytes:
            del output
            nonlocal image_calls
            image_calls += 1
            return image_bytes

    class Websocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            del prompt_id, client_id
            nonlocal websocket_calls
            websocket_calls += 1
            raise AssertionError("WebSocket must not be started")
            yield  # pragma: no cover

    class FailingQueue:
        def mark_queued(self, generation_id: UUID, job_id: UUID, prompt_id: str) -> None:
            del generation_id, job_id
            assert prompt_id == "prompt-database-failure"
            raise GenerationRepositoryError("database is unavailable")

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
        queue_repository=FailingQueue(),  # type: ignore[arg-type]
        failure_repository=repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
        start_repository=start_module.GenerationStartRepository(repository._session_factory),
        progress_repository=progress_module.GenerationProgressRepository(
            repository._session_factory
        ),
    )
    result = await service.generate(_settings())

    persisted = repository.get_by_id(result.generation_id)
    persisted_job = jobs.get_by_generation(result.generation_id)
    assert result.status is GenerationStatus.FAILED
    assert result.prompt_id == "prompt-database-failure"
    assert result.error_message == (
        "生成要求はComfyUIへ送信されましたが、履歴へ関連付けできませんでした。"
        "同じ生成要求の再送信は行っていません。"
    )
    assert persisted is not None and persisted.status is GenerationStatus.FAILED
    assert persisted.error_code == "prompt_persistence_error"
    assert persisted.comfy_prompt_id is None
    assert persisted_job is not None and persisted_job.status is GenerationStatus.FAILED
    assert persisted_job.error_code == "prompt_persistence_error"
    assert persisted_job.prompt_id is None
    assert websocket_calls == 0
    assert history_calls == 0
    assert image_calls == 0
    engine.dispose()


def test_generation_service_rejects_partial_persistence_configuration(tmp_path: Path) -> None:
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)
    configured = {
        "generation_repository": repository,
        "artifact_repository": artifacts,
        "completion_repository": completion,
        "job_repository": jobs,
        "queue_repository": repository_module.GenerationQueueRepository(
            repository._session_factory
        ),
        "failure_repository": repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
    }
    for missing_name in configured:
        partial = {name: value for name, value in configured.items() if name != missing_name}
        with pytest.raises(ValueError, match="configured together"):
            GenerationService(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                lambda: None,  # type: ignore[arg-type]
                settings,
                **partial,
            )
    engine.dispose()


def test_failure_repository_error_is_wrapped_without_exposing_low_level_details(
    tmp_path: Path,
) -> None:
    settings, engine, repository, artifacts, completion, jobs = _repositories(tmp_path)

    class FailingFailureRepository:
        def fail_generation(self, **values: object) -> None:
            del values
            raise GenerationRepositoryError("sqlite://secret database detail")

    service = GenerationService(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        lambda: None,  # type: ignore[arg-type]
        settings,
        generation_repository=repository,
        artifact_repository=artifacts,
        completion_repository=completion,
        job_repository=jobs,
        queue_repository=repository_module.GenerationQueueRepository(repository._session_factory),
        failure_repository=FailingFailureRepository(),  # type: ignore[arg-type]
        start_repository=start_module.GenerationStartRepository(repository._session_factory),
        progress_repository=progress_module.GenerationProgressRepository(
            repository._session_factory
        ),
    )
    generation = repository.create_pending(GenerationSettingsSnapshot.from_settings(_settings()))
    job = jobs.create(GenerationJob(generation.id, GenerationStatus.PENDING))
    with pytest.raises(FailurePersistenceError) as error_info:
        service._persist_failure(
            job,
            error_code="workflow_error",
            error_summary="生成設定を確認できませんでした。",
            failed_at=datetime.now(UTC),
        )
    assert isinstance(error_info.value.__cause__, GenerationRepositoryError)
    assert "sqlite://" not in str(error_info.value)
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
        queue_repository=repository_module.GenerationQueueRepository(repository._session_factory),
        failure_repository=repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
        start_repository=start_module.GenerationStartRepository(repository._session_factory),
        progress_repository=progress_module.GenerationProgressRepository(
            repository._session_factory
        ),
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
        queue_repository=repository_module.GenerationQueueRepository(repository._session_factory),
        failure_repository=repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
        start_repository=start_module.GenerationStartRepository(repository._session_factory),
        progress_repository=progress_module.GenerationProgressRepository(
            repository._session_factory
        ),
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
        failure_repository=repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
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
        queue_repository=repository_module.GenerationQueueRepository(repository._session_factory),
        failure_repository=repository_module.GenerationFailureRepository(
            repository._session_factory
        ),
        start_repository=start_module.GenerationStartRepository(repository._session_factory),
        progress_repository=progress_module.GenerationProgressRepository(
            repository._session_factory
        ),
    )

    assert await service.recover_prompt(generation.id, "prompt-existing") is True
    assert await service.recover_prompt(generation.id, "prompt-existing") is True
    assert repository.get_by_id(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert jobs.get_by_generation(generation.id).status is GenerationStatus.COMPLETED  # type: ignore[union-attr]
    assert len(artifacts.list_by_generation(generation.id)) == 1
    engine.dispose()
