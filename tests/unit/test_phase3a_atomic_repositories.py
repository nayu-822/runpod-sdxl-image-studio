from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from runpod_sdxl_image_studio.adapters.database.repositories import generation_progress_repository
from runpod_sdxl_image_studio.adapters.database.repositories import (
    generation_repository as repository_module,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_job_repository import (
    GenerationJobRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from tests.unit.test_phase3a_history import _repositories, _settings


def test_start_repository_creates_pending_pair_with_shared_values(tmp_path: Path) -> None:
    _, engine, generation_repository, _, _, _ = _repositories(tmp_path)
    start = GenerationStartRepository(generation_repository._session_factory)
    generation_id = UUID(int=101)
    job_id = UUID(int=102)
    created_at = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

    generation, job = start.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=created_at,
    )

    assert generation.id == generation_id
    assert generation.status is GenerationStatus.PENDING
    assert job.id == job_id
    assert job.generation_id == generation_id
    assert job.status is GenerationStatus.PENDING
    assert generation.created_at == job.created_at == created_at
    assert generation.settings_snapshot == GenerationSettingsSnapshot.from_settings(_settings())
    engine.dispose()


def test_start_repository_rolls_back_generation_when_job_insert_fails(tmp_path: Path) -> None:
    _, engine, generation_repository, _, _, _ = _repositories(tmp_path)
    start = GenerationStartRepository(generation_repository._session_factory)
    duplicate_job_id = UUID(int=201)
    first_generation_id = UUID(int=202)
    second_generation_id = UUID(int=203)
    snapshot = GenerationSettingsSnapshot.from_settings(_settings())

    start.create_pending(
        snapshot,
        generation_id=first_generation_id,
        job_id=duplicate_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(repository_module.GenerationRepositoryError):
        start.create_pending(
            snapshot,
            generation_id=second_generation_id,
            job_id=duplicate_job_id,
            kind=GenerationKind.STANDARD,
            parent_generation_id=None,
            created_at=datetime.now(UTC),
        )

    assert generation_repository.get_by_id(second_generation_id) is None
    engine.dispose()


def test_start_repository_rejects_missing_parent_without_inserting_pair(tmp_path: Path) -> None:
    _, engine, generation_repository, _, _, _ = _repositories(tmp_path)
    start = GenerationStartRepository(generation_repository._session_factory)
    generation_id = UUID(int=251)

    with pytest.raises(repository_module.GenerationRepositoryError):
        start.create_pending(
            GenerationSettingsSnapshot.from_settings(_settings()),
            generation_id=generation_id,
            job_id=UUID(int=252),
            kind=GenerationKind.DERIVED,
            parent_generation_id=UUID(int=253),
            created_at=datetime.now(UTC),
        )

    assert generation_repository.get_by_id(generation_id) is None
    engine.dispose()


def test_progress_repository_updates_pair_atomically_and_is_idempotent(tmp_path: Path) -> None:
    _, engine, generation_repository, _, _, _ = _repositories(tmp_path)
    session_factory = generation_repository._session_factory
    start = GenerationStartRepository(session_factory)
    queue = repository_module.GenerationQueueRepository(session_factory)
    progress = GenerationProgressRepository(session_factory)
    generation_id = UUID(int=301)
    job_id = UUID(int=302)
    snapshot = GenerationSettingsSnapshot.from_settings(_settings())
    start.create_pending(
        snapshot,
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
    )
    queue.mark_queued(generation_id, job_id, "progress-prompt")
    updated_at = datetime(2026, 7, 30, 12, 1, tzinfo=UTC)

    for value in (3, 4):
        progress.update_progress(
            generation_id,
            job_id,
            state=GenerationStatus.RUNNING,
            value=value,
            maximum=10,
            current_node="KSampler",
            updated_at=updated_at,
        )

    persisted_generation = generation_repository.get_by_id(generation_id)
    persisted_job = GenerationJobRepository(session_factory).get_by_generation(generation_id)
    assert persisted_generation is not None
    assert persisted_job is not None
    assert persisted_generation.status is GenerationStatus.RUNNING
    assert persisted_job.status is GenerationStatus.RUNNING
    assert persisted_job.progress_value == 4
    assert persisted_job.progress_maximum == 10
    assert persisted_job.current_node == "KSampler"
    assert persisted_generation.updated_at == persisted_job.updated_at == updated_at
    engine.dispose()


def test_progress_repository_rolls_back_running_transition(tmp_path: Path) -> None:
    _, engine, generation_repository, _, _, _ = _repositories(tmp_path)
    session_factory = generation_repository._session_factory
    start = GenerationStartRepository(session_factory)
    queue = repository_module.GenerationQueueRepository(session_factory)
    progress = GenerationProgressRepository(session_factory)
    generation_id = UUID(int=401)
    job_id = UUID(int=402)
    start.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
    )
    queue.mark_queued(generation_id, job_id, "rollback-progress")

    def fail_job(*args: object) -> None:
        del args
        raise repository_module.GenerationRepositoryError("forced progress failure")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(generation_progress_repository, "_mark_job_running", fail_job)
    try:
        with pytest.raises(repository_module.GenerationRepositoryError):
            progress.update_progress(
                generation_id,
                job_id,
                state=GenerationStatus.RUNNING,
                value=1,
                maximum=2,
                current_node="KSampler",
                updated_at=datetime.now(UTC),
            )
    finally:
        monkeypatch.undo()

    persisted_generation = generation_repository.get_by_id(generation_id)
    assert persisted_generation is not None
    assert persisted_generation.status is GenerationStatus.QUEUED
    assert (
        GenerationJobRepository(session_factory).get_by_generation(generation_id).status
        is GenerationStatus.QUEUED
    )  # type: ignore[union-attr]
    engine.dispose()


def test_progress_repository_rejects_invalid_progress_values(tmp_path: Path) -> None:
    _, engine, generation_repository, _, _, _ = _repositories(tmp_path)
    session_factory = generation_repository._session_factory
    start = GenerationStartRepository(session_factory)
    progress = GenerationProgressRepository(session_factory)
    generation_id = UUID(int=501)
    job_id = UUID(int=502)
    start.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
    )

    with pytest.raises(repository_module.GenerationRepositoryError):
        progress.update_progress(
            generation_id,
            job_id,
            state=GenerationStatus.RUNNING,
            value=3,
            maximum=2,
            current_node=None,
            updated_at=datetime.now(UTC),
        )
    engine.dispose()


@pytest.mark.asyncio
async def test_service_does_not_send_prompt_or_failure_when_start_pair_fails(
    tmp_path: Path,
) -> None:
    settings, engine, generation_repository, artifacts, completion, jobs = _repositories(tmp_path)
    session_factory = generation_repository._session_factory
    prompt_calls = 0
    failure = repository_module.GenerationFailureRepository(session_factory)

    class Client:
        async def queue_prompt(self, workflow: object, client_id: str) -> object:
            del workflow, client_id
            nonlocal prompt_calls
            prompt_calls += 1
            raise AssertionError("prompt must not be sent")

    class FailingStart:
        def create_pending(self, snapshot: object, **values: object) -> object:
            del snapshot, values
            raise repository_module.GenerationRepositoryError("forced start failure")

    service = GenerationService(
        Client(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        lambda: None,  # type: ignore[arg-type]
        settings,
        generation_repository=generation_repository,
        artifact_repository=artifacts,
        completion_repository=completion,
        job_repository=jobs,
        queue_repository=repository_module.GenerationQueueRepository(session_factory),
        failure_repository=failure,
        start_repository=FailingStart(),  # type: ignore[arg-type]
        progress_repository=GenerationProgressRepository(session_factory),
    )

    result = await service.generate(_settings())

    assert result.status is GenerationStatus.FAILED
    assert prompt_calls == 0
    assert generation_repository.get_by_id(result.generation_id) is None
    engine.dispose()
