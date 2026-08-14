"""Phase A integration coverage using SQLite and fake application boundaries."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories import (
    generation_dispatch_queue_repository as dispatch_module,
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
from runpod_sdxl_image_studio.adapters.database.repositories.interactive_run_repository import (
    InteractiveRunRepository,
    InteractiveRunRepositoryError,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.interactive_run import InteractiveRunStatus
from runpod_sdxl_image_studio.domain.job import GenerationJob
from runpod_sdxl_image_studio.jobs.generation_queue_worker import GenerationQueueWorker
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
)
from runpod_sdxl_image_studio.services.generation_queue_service import GenerationQueueService
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.interactive_generation_service import (
    InteractiveGenerationError,
    InteractiveGenerationService,
)
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template


def _png(color: str = "blue") -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _settings(**updates: object) -> GenerationSettings:
    values: dict[str, object] = {
        "positive_prompt": "a test image",
        "negative_prompt": "blurry",
        "checkpoint_name": "sdxl.safetensors",
        "sampler_name": "euler",
        "scheduler_name": "normal",
        "seed": 123,
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg_scale": 5.5,
    }
    values.update(updates)
    return GenerationSettings(**values)


def _database() -> tuple[object, object]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine, create_session_factory(engine)


def test_phase_a_completion_persists_multiple_ordered_images_atomically(tmp_path: Path) -> None:
    engine, factory = _database()
    settings = Settings(_env_file=None, data_dir=tmp_path)
    generation_settings = _settings(batch_size=2, client_local_date="2026-08-13")
    snapshot = GenerationSettingsSnapshot.from_settings(generation_settings)
    generation_id = uuid4()
    job_id = uuid4()
    start = GenerationStartRepository(factory)  # type: ignore[arg-type]
    start.create_pending(
        snapshot,
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
    )
    storage = LocalStorageAdapter(settings)
    first = storage.store_image(
        _png("blue"),
        generation_id,
        datetime(2026, 8, 13, tzinfo=UTC),
        client_local_date="2026-08-13",
    )
    second = storage.store_image(
        _png("green"),
        generation_id,
        datetime(2026, 8, 13, tzinfo=UTC),
        client_local_date="2026-08-13",
    )
    repositories = GenerationPersistenceRepositories(
        generation=GenerationRepository(factory),  # type: ignore[arg-type]
        job=GenerationJobRepository(factory),  # type: ignore[arg-type]
        artifact=GenerationArtifactRepository(factory),  # type: ignore[arg-type]
        start=start,
        queue=GenerationQueueRepository(factory),  # type: ignore[arg-type]
        progress=GenerationProgressRepository(factory),  # type: ignore[arg-type]
        completion=GenerationCompletionRepository(factory),  # type: ignore[arg-type]
        failure=GenerationFailureRepository(factory),  # type: ignore[arg-type]
    )
    service = GenerationService(
        object(),  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        object(),  # type: ignore[arg-type]
        storage,
        lambda: None,  # type: ignore[arg-type]
        settings,
        persistence=repositories,
    )
    job = GenerationJobRepository(factory).get_by_generation(generation_id)  # type: ignore[arg-type]
    assert job is not None
    job.status = GenerationStatus.RUNNING
    job.prompt_id = "prompt-batch"
    job.stored_image = first
    job.stored_images = (first, second)

    service._complete_job(  # type: ignore[attr-defined]
        job,
        generation_settings,
        datetime(2026, 8, 13, tzinfo=UTC),
        GenerationKind.STANDARD,
        None,
    )
    result = service._result_for_job(  # type: ignore[attr-defined]
        job,
        generation_settings.seed,
        datetime(2026, 8, 13, tzinfo=UTC),
    )

    generation = GenerationRepository(factory).get_by_id(generation_id)  # type: ignore[arg-type]
    persisted_job = GenerationJobRepository(factory).get_by_generation(generation_id)  # type: ignore[arg-type]
    artifacts = GenerationArtifactRepository(factory).list_by_generation(generation_id)  # type: ignore[arg-type]
    images = tuple(artifact for artifact in artifacts if artifact.artifact_type.value == "image")
    assert generation is not None and generation.status is GenerationStatus.COMPLETED
    assert persisted_job is not None and persisted_job.status is GenerationStatus.COMPLETED
    assert generation.completed_at is not None and persisted_job.completed_at is not None
    assert [artifact.display_order for artifact in images] == [0, 1]
    assert len(images) == 2
    assert result.status is GenerationStatus.COMPLETED
    assert result.stored_images == (first, second)
    assert all(image.path.exists() for image in result.stored_images)

    # The completion boundary is idempotent and must not duplicate ordered artifacts.
    service._complete_job(  # type: ignore[attr-defined]
        job,
        generation_settings,
        datetime(2026, 8, 13, tzinfo=UTC),
        GenerationKind.STANDARD,
        None,
    )
    assert (
        len(
            tuple(
                artifact
                for artifact in GenerationArtifactRepository(factory).list_by_generation(
                    generation_id
                )  # type: ignore[arg-type]
                if artifact.artifact_type.value == "image"
            )
        )
        == 2
    )
    engine.dispose()  # type: ignore[union-attr]


def test_phase_a_interactive_batch_runs_fifo_and_persists_two_images_per_generation(
    tmp_path: Path,
) -> None:
    engine, factory = _database()
    settings = Settings(_env_file=None, data_dir=tmp_path, queue_max_pending_jobs=20)
    dispatch = dispatch_module.GenerationDispatchQueueRepository(factory)  # type: ignore[arg-type]
    queue = GenerationQueueService(dispatch, settings)
    runs = InteractiveRunRepository(factory)  # type: ignore[arg-type]
    artifact_repository = GenerationArtifactRepository(factory)  # type: ignore[arg-type]
    interactive = InteractiveGenerationService(
        runs,
        queue,
        settings,
        artifact_repository=artifact_repository,
    )
    generation_settings = _settings(batch_size=2, client_local_date="2026-08-13")
    storage = LocalStorageAdapter(settings)
    start = GenerationStartRepository(factory)  # type: ignore[arg-type]
    generation_repository = GenerationRepository(factory)  # type: ignore[arg-type]
    job_repository = GenerationJobRepository(factory)  # type: ignore[arg-type]
    completion = GenerationCompletionRepository(factory)  # type: ignore[arg-type]
    persistence = GenerationPersistenceRepositories(
        generation=generation_repository,
        job=job_repository,
        artifact=artifact_repository,
        start=start,
        queue=GenerationQueueRepository(factory),  # type: ignore[arg-type]
        progress=GenerationProgressRepository(factory),  # type: ignore[arg-type]
        completion=completion,
        failure=GenerationFailureRepository(factory),  # type: ignore[arg-type]
    )
    generation_service = GenerationService(
        object(),  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        object(),  # type: ignore[arg-type]
        storage,
        lambda: None,  # type: ignore[arg-type]
        settings,
        persistence=persistence,
    )
    run = interactive.start(
        generation_settings,
        batch_count=3,
        batch_size=2,
        client_local_date="2026-08-13",
    )
    execution_order: list[UUID] = []

    class FakeExecution:
        async def execute_persisted(
            self, generation_id: UUID, job_id: UUID, *args: object, **kwargs: object
        ) -> None:
            del args, kwargs
            execution_order.append(generation_id)
            persisted_generation = generation_repository.get_by_id(generation_id)
            persisted_job = job_repository.get_by_generation(generation_id)
            assert persisted_generation is not None
            assert persisted_job is not None and persisted_job.id == job_id
            job = GenerationJob(
                generation_id=generation_id,
                status=GenerationStatus.RUNNING,
                id=job_id,
                prompt_id=f"prompt-{len(execution_order)}",
                created_at=persisted_generation.created_at,
            )
            first = storage.store_image(
                _png("blue"),
                generation_id,
                persisted_generation.created_at,
                client_local_date="2026-08-13",
            )
            second = storage.store_image(
                _png("green"),
                generation_id,
                persisted_generation.created_at,
                client_local_date="2026-08-13",
            )
            job.stored_image = first
            job.stored_images = (first, second)
            generation_service._complete_job(  # type: ignore[attr-defined]
                job,
                persisted_generation.settings_snapshot.to_generation_settings(),
                persisted_generation.created_at,
                persisted_generation.kind,
                persisted_generation.parent_generation_id,
            )

    worker = GenerationQueueWorker(
        dispatch,
        FakeExecution(),
        settings,
        worker_id="phase-a-worker",
    )
    assert asyncio.run(worker.run_once()) is True
    assert asyncio.run(worker.run_once()) is True
    assert asyncio.run(worker.run_once()) is True
    assert asyncio.run(worker.run_once()) is False
    assert execution_order == list(run.run.generation_ids)

    restored = interactive.refresh(run.run.id)
    assert restored is not None
    assert restored.run.status is InteractiveRunStatus.COMPLETED
    assert restored.completed_count == 3
    for generation_id in run.run.generation_ids:
        artifacts = artifact_repository.list_by_generation(generation_id)
        images = tuple(
            artifact for artifact in artifacts if artifact.artifact_type.value == "image"
        )
        assert [artifact.display_order for artifact in images] == [0, 1]
        assert len(images) == 2
    assert len(restored.result_image_paths) == 2
    engine.dispose()  # type: ignore[union-attr]


def test_phase_a_interactive_run_has_one_active_owner_and_cancel_reconciles_queue() -> None:
    engine, factory = _database()
    settings = Settings(_env_file=None, max_batch_count=4, queue_max_pending_jobs=20)
    dispatch = dispatch_module.GenerationDispatchQueueRepository(factory)  # type: ignore[arg-type]
    queue = GenerationQueueService(dispatch, settings)
    runs = InteractiveRunRepository(factory)  # type: ignore[arg-type]
    service = InteractiveGenerationService(runs, queue, settings)

    first = service.start(
        _settings(seed=321),
        batch_count=3,
        batch_size=2,
        client_local_date="2026-08-13",
    )
    assert first.run.status is InteractiveRunStatus.ACTIVE
    assert first.run.batch_count == 3
    assert len(first.run.generation_ids) == 3
    for generation_id in first.run.generation_ids:
        item = dispatch.get_queue_item(generation_id)
        assert item is not None
        assert item.generation.settings_snapshot.batch_size == 2

    with pytest.raises(InteractiveGenerationError):
        service.start(
            _settings(seed=999),
            batch_count=1,
            batch_size=1,
            client_local_date="2026-08-13",
        )

    runs.request_cancel(first.run.id)
    with pytest.raises(InteractiveGenerationError):
        service.start(
            _settings(seed=1000),
            batch_count=1,
            batch_size=1,
            client_local_date="2026-08-13",
        )

    cancelled = asyncio.run(service.cancel(first.run.id))
    assert cancelled is not None
    assert cancelled.run.status is InteractiveRunStatus.CANCELLED
    assert cancelled.completed_count == 0
    assert all(
        dispatch.get_queue_item(generation_id).generation.status is GenerationStatus.CANCELLED
        for generation_id in first.run.generation_ids
    )

    failed_run = service.start(
        _settings(seed=654),
        batch_count=3,
        batch_size=2,
        client_local_date="2026-08-13",
    )
    dispatch.mark_reconciliation_failed(failed_run.run.generation_ids[0], "simulated failure")
    reconciled = service.refresh(failed_run.run.id)
    assert reconciled is not None
    assert reconciled.run.status is InteractiveRunStatus.FAILED
    assert reconciled.completed_count == 0
    assert dispatch.get_queue_item(failed_run.run.generation_ids[1]).generation.status is (
        GenerationStatus.CANCELLED
    )
    assert dispatch.get_queue_item(failed_run.run.generation_ids[2]).generation.status is (
        GenerationStatus.CANCELLED
    )
    engine.dispose()  # type: ignore[union-attr]


def test_phase_a_interactive_repository_restores_latest_completed_run() -> None:
    engine, factory = _database()
    settings = _settings(seed=456)
    snapshot = GenerationSettingsSnapshot.from_settings(settings)
    repository = InteractiveRunRepository(factory)  # type: ignore[arg-type]
    run = repository.create_active(
        snapshot,
        batch_count=1,
        batch_size=1,
        client_local_date="2026-08-13",
    )
    generation_id = UUID("00000000-0000-0000-0000-000000000123")
    repository.attach_generations(run.id, (generation_id,))
    completed = repository.update_progress(
        run.id,
        completed_generation_ids=(generation_id,),
        current_generation_id=None,
        status=InteractiveRunStatus.COMPLETED,
        completed_at=datetime(2026, 8, 13, 1, tzinfo=UTC),
    )
    assert completed.status is InteractiveRunStatus.COMPLETED
    assert repository.get_active() is None
    restored = repository.get_latest_completed()
    assert restored is not None
    assert restored.id == run.id
    assert restored.last_completed_generation_id == generation_id

    with pytest.raises(InteractiveRunRepositoryError):
        repository.update_progress(
            run.id,
            completed_generation_ids=(uuid4(),),
            current_generation_id=None,
        )
    engine.dispose()  # type: ignore[union-attr]
