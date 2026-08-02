"""Phase 4 persistence, batch, cancellation, retry, and worker tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from runpod_sdxl_image_studio.adapters.comfyui.cancellation import ComfyUICancellationAdapter
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIConnectionError,
    ComfyUIResponseError,
    ComfyUITimeoutError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUIOutputImage,
    ComfyUIQueueStatus,
    PromptHistory,
    PromptHistoryStatus,
)
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory, session_scope
from runpod_sdxl_image_studio.adapters.database.models import (
    Base,
    GenerationJobModel,
    GenerationModel,
    GenerationQueueEntryModel,
)
from runpod_sdxl_image_studio.adapters.database.repositories import (
    generation_dispatch_queue_repository as dispatch_module,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
    GenerationDispatchQueueRepositoryError,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationQueueRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import (
    BatchSeedStrategy,
    CancellationOutcome,
    ReconciliationOutcome,
    SubmissionState,
)
from runpod_sdxl_image_studio.domain.generation_settings import MAX_SEED, GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.jobs.generation_queue_worker import GenerationQueueWorker
from runpod_sdxl_image_studio.services.generation_errors import PromptPersistenceError
from runpod_sdxl_image_studio.services.generation_queue_service import (
    CancellationResult,
    GenerationQueueService,
    GenerationQueueServiceError,
)


def _database() -> tuple[Any, Any]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine, create_session_factory(engine)


def _settings(seed: int = 123) -> GenerationSettings:
    return GenerationSettings(
        positive_prompt="a cat",
        negative_prompt="low quality",
        seed=seed,
        width=1024,
        height=1024,
        steps=28,
        cfg_scale=5.5,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="sdxl.safetensors",
        vae_name="vae.safetensors",
    )


def test_queue_repository_is_fifo_and_leases_are_recoverable() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    snapshot = GenerationSettingsSnapshot.from_settings(_settings())
    first = repository.enqueue_single(snapshot)
    second = repository.enqueue_single(snapshot)
    claimed = repository.claim_next(
        "worker-a",
        lease_seconds=30,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert claimed is not None
    assert claimed.entry.sequence == first.entry.sequence
    assert claimed.entry.worker_id == "worker-a"
    assert claimed.job.worker_id == "worker-a"

    repository.reconcile_expired_claims(now=datetime(2026, 1, 1, 0, 0, 31, tzinfo=UTC))
    reclaimed = repository.claim_next(
        "worker-b",
        lease_seconds=30,
        now=datetime(2026, 1, 1, 0, 0, 32, tzinfo=UTC),
    )
    assert reclaimed is not None
    assert reclaimed.entry.sequence == first.entry.sequence
    repository.release_claim(reclaimed.entry.sequence, "worker-b")
    repository.mark_cancelled(first.generation.id)

    remaining = repository.claim_next(
        "worker-c",
        lease_seconds=30,
        now=datetime(2026, 1, 1, 0, 0, 33, tzinfo=UTC),
    )
    assert remaining is not None
    assert remaining.entry.sequence == second.entry.sequence
    engine.dispose()


def test_submission_state_is_persisted_and_claim_is_not_reusable() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    claimed = repository.claim_next("worker-a", lease_seconds=30)
    assert claimed is not None

    submitting = repository.begin_submission(claimed.entry.sequence, "worker-a")
    assert submitting.entry.submission_state is SubmissionState.SUBMITTING
    assert submitting.entry.submission_token
    assert repository.claim_next("worker-b", lease_seconds=30) is None

    submitted = repository.mark_submitted(
        item.entry.sequence,
        "worker-a",
        submitting.entry.submission_token or "",
        "prompt-1",
    )
    assert submitted.entry.submission_state is SubmissionState.SUBMITTED
    assert submitted.generation.status is GenerationStatus.QUEUED
    assert submitted.job.prompt_id == "prompt-1"
    assert repository.claim_next("worker-b", lease_seconds=30) is None
    engine.dispose()


def test_seed_limits_are_sqlite_safe_and_random_resolution_is_repeatedly_persistable() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    service = GenerationQueueService(
        repository,
        Settings(_env_file=None, queue_max_pending_jobs=32),
    )
    assert _settings(0).seed == 0
    assert _settings(MAX_SEED).seed == MAX_SEED
    with pytest.raises(ValueError):
        _settings(MAX_SEED + 1)
    for _ in range(20):
        result = service.enqueue(_settings(-1))
        assert 0 <= result.item.generation.settings_snapshot.seed <= MAX_SEED
    engine.dispose()


def test_retry_requests_are_idempotent_for_generation_and_batch() -> None:
    engine, factory = _database()
    dispatch_repository = GenerationDispatchQueueRepository(factory)
    generation_repository = GenerationRepository(factory)
    service = GenerationQueueService(dispatch_repository, Settings(_env_file=None))

    source = service.enqueue(_settings()).item
    generation_repository.mark_failed(source.generation.id, "test", "failed")
    first = service.retry(source.generation.id)
    second = service.retry(source.generation.id)
    assert first.item.generation.id == second.item.generation.id
    assert len(dispatch_repository.list_queue()) == 2

    batch = service.enqueue_batch(
        _settings(),
        count=2,
        seed_strategy=BatchSeedStrategy.SEQUENTIAL,
        start_seed=1,
        seed_step=1,
        name="retry batch",
    ).batch
    batch_items = dispatch_repository.list_batch_items(batch.id)
    generation_repository.mark_failed(batch_items[0].generation.id, "test", "failed")
    first_batch_retry = service.retry_failed_batch(batch.id)
    second_batch_retry = service.retry_failed_batch(batch.id)
    assert first_batch_retry is not None and second_batch_retry is not None
    assert first_batch_retry.batch.id == second_batch_retry.batch.id
    assert len(dispatch_repository.list_queue()) == 5
    engine.dispose()


def test_reconciliation_outcome_in_progress_does_not_fail_prompt_job() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    queue_repository = GenerationQueueRepository(factory)
    queue_repository.mark_queued(item.generation.id, item.job.id, "prompt-1")

    async def reconcile(_item: object) -> ReconciliationOutcome:
        return ReconciliationOutcome.IN_PROGRESS

    worker = GenerationQueueWorker(
        repository,
        object(),
        Settings(_env_file=None, reconciliation_grace_seconds=60),
        reconcile_handler=reconcile,
    )
    asyncio.run(worker.reconcile())
    reconciled = repository.get_queue_item(item.generation.id)
    assert reconciled is not None
    assert reconciled.generation.status is GenerationStatus.QUEUED
    engine.dispose()


@pytest.mark.parametrize("stale", [False, True])
def test_worker_only_fails_true_not_found_after_grace(stale: bool) -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    GenerationQueueRepository(factory).mark_queued(item.generation.id, item.job.id, "prompt-1")
    if stale:
        with session_scope(factory) as session:
            entry = session.get(GenerationQueueEntryModel, item.entry.sequence)
            assert entry is not None
            entry.updated_at = datetime(2020, 1, 1, tzinfo=UTC)

    async def reconcile(_item: object) -> ReconciliationOutcome:
        return ReconciliationOutcome.NOT_FOUND

    worker = GenerationQueueWorker(
        repository,
        object(),
        Settings(
            _env_file=None,
            reconciliation_grace_seconds=0 if stale else 60,
        ),
        reconcile_handler=reconcile,
    )
    asyncio.run(worker.reconcile())
    reconciled = repository.get_queue_item(item.generation.id)
    assert reconciled is not None
    expected = GenerationStatus.FAILED if stale else GenerationStatus.QUEUED
    assert reconciled.generation.status is expected
    engine.dispose()


def test_prompt_persistence_failure_is_quarantined_without_resubmission() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    calls = 0

    class FakeExecution:
        async def execute_persisted(self, *args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            coordinator = kwargs["submission_coordinator"]
            assert coordinator is not None
            coordinator.begin()
            raise PromptPersistenceError("test persistence failure")

    worker = GenerationQueueWorker(
        repository,
        FakeExecution(),
        Settings(_env_file=None),
        worker_id="worker-a",
    )
    assert asyncio.run(worker.run_once()) is True
    assert asyncio.run(worker.run_once()) is False
    quarantined = repository.get_queue_item(item.generation.id)
    assert quarantined is not None
    assert quarantined.entry.submission_state is SubmissionState.AMBIGUOUS
    assert quarantined.generation.error_code == "prompt_submission_ambiguous"
    assert quarantined.job.error_code == "prompt_submission_ambiguous"
    assert calls == 1
    engine.dispose()


def test_prompt_failure_after_submission_does_not_make_item_claimable_again() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    calls = 0

    class FakeExecution:
        async def execute_persisted(self, *args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            coordinator = kwargs["submission_coordinator"]
            assert coordinator is not None
            token = coordinator.begin()
            coordinator.mark_submitted("prompt-once", token)
            raise RuntimeError("failure persistence path")

    worker = GenerationQueueWorker(
        repository,
        FakeExecution(),
        Settings(_env_file=None),
        worker_id="worker-a",
    )
    assert asyncio.run(worker.run_once()) is True
    assert asyncio.run(worker.run_once()) is False
    submitted = repository.get_queue_item(item.generation.id)
    assert submitted is not None
    assert submitted.entry.submission_state is SubmissionState.SUBMITTED
    assert submitted.job.prompt_id == "prompt-once"
    assert calls == 1
    engine.dispose()


def test_batch_seed_strategies_and_capacity_are_persisted() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    settings = Settings(_env_file=None, queue_max_pending_jobs=4, batch_max_items=4)
    service = GenerationQueueService(repository, settings)

    result = service.enqueue_batch(
        _settings(),
        count=3,
        seed_strategy=BatchSeedStrategy.SEQUENTIAL,
        start_seed=100,
        seed_step=7,
        name="test batch",
    )

    assert result.batch.item_count == 3
    assert [item.generation.settings_snapshot.seed for item in result.items] == [100, 107, 114]
    assert [item.entry.batch_index for item in result.items] == [0, 1, 2]
    service.enqueue(_settings(999))
    with pytest.raises(GenerationQueueServiceError, match="上限"):
        service.enqueue(_settings(1000))
    engine.dispose()


def test_cancel_pending_and_retry_keep_snapshots_and_link_generations() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    settings = Settings(_env_file=None)
    service = GenerationQueueService(repository, settings)
    item = service.enqueue(_settings(77)).item

    cancelled = asyncio.run(service.cancel(item.generation.id))
    assert cancelled.generation.status is GenerationStatus.CANCELLED
    assert cancelled.job.status is GenerationStatus.CANCELLED

    retry = service.retry(item.generation.id).item
    assert retry.generation.status is GenerationStatus.PENDING
    assert retry.generation.retry_of_generation_id == item.generation.id
    assert retry.generation.retry_attempt == 1
    assert retry.generation.settings_snapshot == item.generation.settings_snapshot
    engine.dispose()


def test_cancel_does_not_mark_mismatched_pending_pair_cancelled() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    with session_scope(factory) as session:
        generation = session.get(GenerationModel, str(item.generation.id))
        job = session.get(GenerationJobModel, str(item.job.id))
        assert generation is not None and job is not None
        generation.comfy_prompt_id = "generation-prompt"
        job.comfy_prompt_id = "job-prompt"

    service = GenerationQueueService(repository, Settings(_env_file=None))
    with pytest.raises(GenerationQueueServiceError, match="prompt ID"):
        asyncio.run(service.cancel(item.generation.id))

    current = repository.get_queue_item(item.generation.id)
    assert current is not None
    assert current.generation.status is GenerationStatus.PENDING
    assert current.job.status is GenerationStatus.PENDING
    assert current.generation.comfy_prompt_id == "generation-prompt"
    assert current.job.prompt_id == "job-prompt"
    engine.dispose()


def test_running_cancel_requires_adapter_confirmation() -> None:
    engine, factory = _database()
    dispatch_repository = GenerationDispatchQueueRepository(factory)
    comfy_queue = GenerationQueueRepository(factory)
    item = dispatch_repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    comfy_queue.mark_queued(item.generation.id, item.job.id, "prompt-1")

    class Adapter:
        def __init__(self, confirmed: bool) -> None:
            self.confirmed = confirmed

        async def cancel_prompt(self, prompt_id: str) -> CancellationResult:
            assert prompt_id == "prompt-1"
            return CancellationResult(requested=True, confirmed=self.confirmed)

    with pytest.raises(GenerationQueueServiceError):
        asyncio.run(
            GenerationQueueService(
                dispatch_repository, Settings(_env_file=None), Adapter(False)
            ).cancel(item.generation.id)
        )
    assert (
        dispatch_repository.get_queue_item(item.generation.id).generation.status
        is GenerationStatus.QUEUED
    )  # type: ignore[union-attr]

    cancelled = asyncio.run(
        GenerationQueueService(dispatch_repository, Settings(_env_file=None), Adapter(True)).cancel(
            item.generation.id
        )
    )
    assert cancelled.generation.status is GenerationStatus.CANCELLED
    engine.dispose()


def test_worker_retries_persistent_cancel_until_comfyui_confirms() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    GenerationQueueRepository(factory).mark_queued(item.generation.id, item.job.id, "prompt-1")
    repository.request_cancel(item.generation.id)

    class Adapter:
        calls = 0

        async def cancel_prompt(self, prompt_id: str) -> CancellationResult:
            self.calls += 1
            return CancellationResult(requested=True, confirmed=self.calls > 1)

    adapter = Adapter()
    worker = GenerationQueueWorker(
        repository,
        object(),
        Settings(_env_file=None),
        cancellation_adapter=adapter,
    )

    asyncio.run(worker.reconcile())
    first = repository.get_queue_item(item.generation.id)
    assert first is not None and first.generation.status is GenerationStatus.QUEUED
    asyncio.run(worker.reconcile())
    second = repository.get_queue_item(item.generation.id)
    assert second is not None and second.generation.status is GenerationStatus.CANCELLED
    assert adapter.calls == 2
    engine.dispose()


def test_worker_does_not_overwrite_completion_when_cancel_races() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    GenerationQueueRepository(factory).mark_queued(item.generation.id, item.job.id, "prompt-race")
    repository.request_cancel(item.generation.id)

    class Adapter:
        async def cancel_prompt(self, prompt_id: str) -> CancellationResult:
            assert prompt_id == "prompt-race"
            return CancellationResult(
                requested=True,
                outcome=CancellationOutcome.COMPLETED,
            )

    async def reconcile(_: object) -> ReconciliationOutcome:
        return ReconciliationOutcome.COMPLETED

    worker = GenerationQueueWorker(
        repository,
        object(),
        Settings(_env_file=None),
        cancellation_adapter=Adapter(),
        reconcile_handler=reconcile,
    )
    asyncio.run(worker.reconcile())

    current = repository.get_queue_item(item.generation.id)
    assert current is not None
    assert current.generation.status is GenerationStatus.QUEUED
    assert current.job.status is GenerationStatus.QUEUED
    engine.dispose()


def test_ambiguous_prompt_can_be_linked_or_explicitly_failed() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    claimed = repository.claim_next("worker-a", lease_seconds=30)
    assert claimed is not None
    repository.begin_submission(claimed.entry.sequence, "worker-a")
    repository.mark_submission_ambiguous(
        item.entry.sequence, "worker-a", "network outcome was unknown"
    )

    linked = repository.link_ambiguous_prompt(item.generation.id, "prompt-manual")
    assert linked.entry.submission_state is SubmissionState.SUBMITTED
    assert linked.generation.status is GenerationStatus.QUEUED
    assert linked.job.prompt_id == "prompt-manual"
    with pytest.raises(GenerationDispatchQueueRepositoryError):
        repository.link_ambiguous_prompt(item.generation.id, "prompt-again")

    second = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings(456)))
    claimed_second = repository.claim_next("worker-a", lease_seconds=30)
    assert claimed_second is not None
    repository.begin_submission(claimed_second.entry.sequence, "worker-a")
    repository.mark_submission_ambiguous(second.entry.sequence, "worker-a", "unknown")
    failed = repository.fail_ambiguous_prompt(second.generation.id)
    assert failed.generation.status is GenerationStatus.FAILED
    assert failed.job.status is GenerationStatus.FAILED
    assert failed.generation.error_code == "prompt_submission_ambiguous_resolved"
    engine.dispose()


def test_mismatched_prompt_ids_can_be_manually_resolved_without_resubmission() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    GenerationQueueRepository(factory).mark_queued(
        item.generation.id, item.job.id, "generation-prompt"
    )
    with session_scope(factory) as session:
        job = session.get(GenerationJobModel, str(item.job.id))
        assert job is not None
        job.comfy_prompt_id = "job-prompt"

    ambiguous = repository.mark_prompt_id_mismatch(item.generation.id)
    assert ambiguous.generation.comfy_prompt_id == "generation-prompt"
    assert ambiguous.job.prompt_id == "job-prompt"
    resolved = repository.link_ambiguous_prompt(item.generation.id, "generation-prompt")
    assert resolved.entry.submission_state is SubmissionState.SUBMITTED
    assert resolved.generation.comfy_prompt_id == "generation-prompt"
    assert resolved.job.prompt_id == "generation-prompt"
    assert resolved.generation.status is GenerationStatus.QUEUED
    engine.dispose()


def test_batch_enqueue_rolls_back_all_rows_on_mid_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    original = dispatch_module._insert_generation_and_job
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise GenerationDispatchQueueRepositoryError("test batch failure")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dispatch_module, "_insert_generation_and_job", fail_second)
    snapshots = tuple(
        GenerationSettingsSnapshot.from_settings(_settings(seed)) for seed in (1, 2, 3)
    )
    with pytest.raises(GenerationDispatchQueueRepositoryError):
        repository.enqueue_batch(
            snapshots,
            name="atomic",
            seed_strategy=BatchSeedStrategy.SEQUENTIAL,
            start_seed=1,
            seed_step=1,
        )
    assert repository.list_queue() == ()
    engine.dispose()


def test_reconciliation_does_not_fail_ambiguous_item_without_prompt() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    old = datetime(2020, 1, 1, tzinfo=UTC)
    with session_scope(factory) as session:
        generation = session.get(GenerationModel, str(item.generation.id))
        job = session.get(GenerationJobModel, str(item.job.id))
        entry = session.get(GenerationQueueEntryModel, item.entry.sequence)
        assert generation is not None and job is not None and entry is not None
        generation.status = GenerationStatus.QUEUED.value
        job.status = GenerationStatus.QUEUED.value
        generation.updated_at = old
        job.updated_at = old
        entry.updated_at = old
        entry.submission_state = "ambiguous"

    worker = GenerationQueueWorker(
        repository,
        object(),
        Settings(_env_file=None, reconciliation_grace_seconds=0),
        reconcile_handler=None,
    )
    asyncio.run(worker.reconcile())
    reconciled = repository.get_queue_item(item.generation.id)
    assert reconciled is not None
    assert reconciled.generation.status is GenerationStatus.QUEUED
    assert reconciled.entry.submission_state.value == "ambiguous"
    engine.dispose()


def test_expired_cancel_request_finishes_unsubmitted_pending_item() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    claimed_at = datetime(2026, 1, 1, tzinfo=UTC)
    claimed = repository.claim_next("worker-a", lease_seconds=10, now=claimed_at)
    assert claimed is not None
    requested = repository.request_cancel(item.generation.id, now=claimed_at)
    assert requested.entry.cancel_requested_at == claimed_at

    reconciled_count = repository.reconcile_expired_claims(
        now=datetime(2026, 1, 1, 0, 0, 11, tzinfo=UTC)
    )

    assert reconciled_count == 1
    cancelled = repository.get_queue_item(item.generation.id)
    assert cancelled is not None
    assert cancelled.generation.status is GenerationStatus.CANCELLED
    assert cancelled.job.status is GenerationStatus.CANCELLED
    assert cancelled.entry.worker_id is None
    engine.dispose()


@pytest.mark.parametrize("queue_state", ["pending", "running"])
def test_comfyui_cancellation_confirms_queue_removal(
    queue_state: str,
) -> None:
    class FakeClient:
        active = True
        deleted = False
        interrupted = False

        async def get_queue_status(self) -> ComfyUIQueueStatus:
            prompt_ids = ("prompt-1",) if self.active else ()
            return ComfyUIQueueStatus(
                pending_prompt_ids=prompt_ids if queue_state == "pending" else (),
                running_prompt_ids=prompt_ids if queue_state == "running" else (),
            )

        async def delete_queued_prompt(self, prompt_id: str) -> None:
            assert prompt_id == "prompt-1"
            self.deleted = True
            self.active = False

        async def interrupt_prompt(self, prompt_id: str) -> None:
            assert prompt_id == "prompt-1"
            self.interrupted = True
            self.active = False

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(prompt_id, False, False, (), None, False)

    client = FakeClient()
    adapter = ComfyUICancellationAdapter(
        client, Settings(_env_file=None, history_max_attempts=2), sleep=lambda _: asyncio.sleep(0)
    )

    result = asyncio.run(adapter.cancel_prompt("prompt-1"))

    assert result.requested is True
    assert result.confirmed is True
    assert client.deleted is (queue_state == "pending")
    assert client.interrupted is (queue_state == "running")


def test_comfyui_cancellation_does_not_confirm_completed_prompt() -> None:
    class FakeClient:
        async def get_queue_status(self) -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus((), ())

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                True,
                False,
                (ComfyUIOutputImage("image.png", "", "output"),),
                None,
            )

    adapter = ComfyUICancellationAdapter(FakeClient())
    result = asyncio.run(adapter.cancel_prompt("prompt-1"))

    assert result.requested is True
    assert result.confirmed is False


def test_comfyui_cancellation_does_not_confirm_prompt_still_in_history() -> None:
    class FakeClient:
        async def get_queue_status(self) -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus((), ())

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(prompt_id, False, False, (), None, False)

    adapter = ComfyUICancellationAdapter(FakeClient())
    result = asyncio.run(adapter.cancel_prompt("prompt-1"))

    assert result.requested is True
    assert result.confirmed is False


def test_comfyui_cancellation_keeps_request_only_when_confirmation_unavailable() -> None:
    class FakeClient:
        async def get_queue_status(self) -> ComfyUIQueueStatus:
            raise ComfyUIConnectionError("offline")

    adapter = ComfyUICancellationAdapter(FakeClient())
    result = asyncio.run(adapter.cancel_prompt("prompt-1"))

    assert result.requested is True
    assert result.confirmed is False


def test_comfyui_modern_cancellation_is_confirmed_without_legacy_fallback() -> None:
    class FakeClient:
        modern_calls = 0
        legacy_calls = 0

        async def cancel_job(self, prompt_id: str) -> bool:
            assert prompt_id == "prompt-1"
            self.modern_calls += 1
            return True

        async def get_queue_status(self) -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus((), ())

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(prompt_id, False, False, (), None, False)

        async def delete_queued_prompt(self, prompt_id: str) -> None:
            self.legacy_calls += 1

        async def interrupt_prompt(self, prompt_id: str) -> None:
            self.legacy_calls += 1

    client = FakeClient()
    result = asyncio.run(ComfyUICancellationAdapter(client).cancel_prompt("prompt-1"))

    assert result.confirmed is True
    assert client.modern_calls == 1
    assert client.legacy_calls == 0


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (PromptHistory("prompt-1", True, False, (), None), CancellationOutcome.COMPLETED),
        (PromptHistory("prompt-1", False, True, (), "failed"), CancellationOutcome.FAILED),
        (PromptHistory("prompt-1", False, False, (), None), CancellationOutcome.UNAVAILABLE),
        (PromptHistory("prompt-1", False, False, (), None, False), CancellationOutcome.NOT_FOUND),
    ],
)
def test_comfyui_modern_false_observes_state_without_legacy_fallback(
    history: PromptHistory,
    expected: CancellationOutcome,
) -> None:
    class FakeClient:
        legacy_calls = 0

        async def cancel_job(self, prompt_id: str) -> bool:
            return False

        async def get_queue_status(self) -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus((), ())

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return history

        async def delete_queued_prompt(self, prompt_id: str) -> None:
            self.legacy_calls += 1

        async def interrupt_prompt(self, prompt_id: str) -> None:
            self.legacy_calls += 1

    client = FakeClient()
    result = asyncio.run(
        ComfyUICancellationAdapter(
            client,
            Settings(_env_file=None, history_max_attempts=1),
        ).cancel_prompt("prompt-1")
    )

    assert result.requested is True
    assert result.confirmed is False
    assert result.outcome is expected
    assert client.legacy_calls == 0


def test_comfyui_history_running_is_in_progress_even_when_queue_is_empty() -> None:
    class FakeClient:
        async def cancel_job(self, prompt_id: str) -> bool:
            return False

        async def get_queue_status(self) -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus((), ())

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                False,
                False,
                (),
                None,
                True,
                status=PromptHistoryStatus.IN_PROGRESS,
            )

    result = asyncio.run(
        ComfyUICancellationAdapter(
            FakeClient(),
            Settings(_env_file=None, history_max_attempts=1),
        ).cancel_prompt("prompt-1")
    )

    assert result.outcome is CancellationOutcome.IN_PROGRESS


def test_comfyui_interrupted_history_confirms_cancellation() -> None:
    class FakeClient:
        async def cancel_job(self, prompt_id: str) -> bool:
            return True

        async def get_queue_status(self) -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus((), ())

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                False,
                False,
                (),
                None,
                True,
                status=PromptHistoryStatus.INTERRUPTED,
            )

    result = asyncio.run(ComfyUICancellationAdapter(FakeClient()).cancel_prompt("prompt-1"))

    assert result.outcome is CancellationOutcome.CANCELLED
    assert result.confirmed is True


@pytest.mark.parametrize("status_code", [404, 405])
def test_comfyui_modern_404_or_405_uses_legacy_fallback(status_code: int) -> None:
    class FakeClient:
        deleted = False

        async def cancel_job(self, prompt_id: str) -> bool:
            raise ComfyUIResponseError("not supported", status_code=status_code)

        async def get_queue_status(self) -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus(("prompt-1",) if not self.deleted else (), ())

        async def delete_queued_prompt(self, prompt_id: str) -> None:
            self.deleted = True

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(prompt_id, False, False, (), None, False)

    client = FakeClient()
    result = asyncio.run(ComfyUICancellationAdapter(client).cancel_prompt("prompt-1"))

    assert result.confirmed is True
    assert client.deleted is True


def test_comfyui_modern_timeout_does_not_try_legacy() -> None:
    class FakeClient:
        async def cancel_job(self, prompt_id: str) -> bool:
            raise ComfyUITimeoutError("timed out")

        async def get_queue_status(self) -> ComfyUIQueueStatus:
            raise AssertionError("timeout must not fall back")

    result = asyncio.run(ComfyUICancellationAdapter(FakeClient()).cancel_prompt("prompt-1"))

    assert result.confirmed is False


def test_failed_batch_retry_only_enqueues_failed_items() -> None:
    engine, factory = _database()
    dispatch_repository = GenerationDispatchQueueRepository(factory)
    generation_repository = GenerationRepository(factory)
    service = GenerationQueueService(dispatch_repository, Settings(_env_file=None))
    batch = service.enqueue_batch(
        _settings(),
        count=3,
        seed_strategy="sequential",
        start_seed=1,
        seed_step=1,
        name="retryable",
    ).batch
    source = dispatch_repository.list_batch_items(batch.id)
    generation_repository.mark_failed(source[1].generation.id, "test", "failed")

    retry = service.retry_failed_batch(batch.id)

    assert retry is not None
    assert len(retry.items) == 1
    assert retry.items[0].generation.retry_of_generation_id == source[1].generation.id
    assert retry.items[0].generation.settings_snapshot.seed == 2
    assert retry.batch.retry_of_batch_id == batch.id
    engine.dispose()


def test_worker_executes_claimed_persisted_item_without_creating_prompt() -> None:
    engine, factory = _database()
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    settings = Settings(_env_file=None, queue_poll_interval_seconds=1)
    calls: list[tuple[Any, ...]] = []

    class FakeExecution:
        async def execute_persisted(self, *args: object, **kwargs: object) -> None:
            calls.append(args)

    worker = GenerationQueueWorker(repository, FakeExecution(), settings, worker_id="worker")
    assert asyncio.run(worker.run_once()) is True
    assert len(calls) == 1
    assert calls[0][:2] == (item.generation.id, item.job.id)
    assert repository.get_queue_item(item.generation.id) is not None
    assert repository.get_queue_item(item.generation.id).entry.worker_id is None  # type: ignore[union-attr]
    engine.dispose()


def test_phase4_migration_creates_and_round_trips_queue_tables(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_url=f"sqlite:///{(tmp_path / 'phase4.sqlite3').as_posix()}",
    )
    root = Path(__file__).parents[2]
    upgrade_database(settings, root)
    engine = create_engine(settings.database_url)
    names = set(inspect(engine).get_table_names())
    assert {"generation_batches", "generation_queue_entries"} <= names
    engine.dispose()

    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.downgrade(config, "-1")
    command.upgrade(config, "head")
    engine = create_engine(settings.database_url)
    assert inspect(engine).has_table("generation_queue_entries")
    engine.dispose()


def test_phase4_migration_backfills_prompt_and_marks_status_mismatch(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'phase4-backfill.sqlite3').as_posix()}"
    root = Path(__file__).parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0004_phase4_persistent_generation_queue")
    engine = create_engine(database_url)
    snapshot = GenerationSettingsSnapshot.from_settings(_settings()).to_json()
    timestamp = "2026-01-01 00:00:00"
    mismatch_generation = "00000000-0000-0000-0000-000000000001"
    mismatch_job = "00000000-0000-0000-0000-000000000011"
    prompt_generation = "00000000-0000-0000-0000-000000000002"
    prompt_job = "00000000-0000-0000-0000-000000000012"
    normal_generation = "00000000-0000-0000-0000-000000000003"
    normal_job = "00000000-0000-0000-0000-000000000013"
    with engine.begin() as connection:
        for generation_id, status, prompt_id in (
            (mismatch_generation, "queued", None),
            (prompt_generation, "queued", "prompt-existing"),
        ):
            connection.execute(
                text(
                    """INSERT INTO generations
                    (id, kind, status, parent_generation_id, retry_of_generation_id,
                     retry_attempt, settings_snapshot_json, snapshot_schema_version,
                     checkpoint_name, vae_name, seed, width, height,
                     positive_prompt_search, negative_prompt_search,
                     workflow_template_id, workflow_template_version, comfy_prompt_id,
                     favorite, user_note, error_code, error_summary, created_at,
                     started_at, completed_at, updated_at)
                    VALUES (:id, 'standard', :status, NULL, NULL, 0, :snapshot, 1,
                     'sdxl.safetensors', NULL, 123, 1024, 1024, 'a cat', '',
                     'sdxl_txt2img', '1.0', :prompt, 0, NULL, NULL, NULL,
                     :timestamp, NULL, NULL, :timestamp)"""
                ),
                {
                    "id": generation_id,
                    "status": status,
                    "snapshot": snapshot,
                    "prompt": prompt_id,
                    "timestamp": timestamp,
                },
            )
        for job_id, generation_id, prompt_id in (
            (mismatch_job, mismatch_generation, None),
            (prompt_job, prompt_generation, "prompt-existing"),
        ):
            connection.execute(
                text(
                    """INSERT INTO generation_jobs
                    (id, generation_id, status, comfy_prompt_id, progress_value,
                     progress_maximum, current_node, worker_id, claimed_at,
                     lease_expires_at, cancel_requested_at, cancelled_at, error_code,
                     error_summary, created_at, started_at, completed_at, updated_at)
                    VALUES (:id, :generation_id, 'queued', :prompt, NULL, NULL, NULL,
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL, :timestamp, NULL, NULL,
                     :timestamp)"""
                ),
                {
                    "id": job_id,
                    "generation_id": generation_id,
                    "prompt": prompt_id,
                    "timestamp": timestamp,
                },
            )
        connection.execute(
            text(
                """INSERT INTO generation_queue_entries
                (generation_id, job_id, batch_id, batch_index, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, enqueued_at, updated_at)
                VALUES (:generation_id, :job_id, NULL, 0, NULL, NULL, NULL, NULL,
                 :timestamp, :timestamp)"""
            ),
            {
                "generation_id": mismatch_generation,
                "job_id": mismatch_job,
                "timestamp": timestamp,
            },
        )
        connection.execute(
            text(
                """INSERT INTO generation_queue_entries
                (generation_id, job_id, batch_id, batch_index, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, enqueued_at, updated_at)
                VALUES (:generation_id, :job_id, NULL, 0, NULL, NULL, NULL, NULL,
                 :timestamp, :timestamp)"""
            ),
            {
                "generation_id": prompt_generation,
                "job_id": prompt_job,
                "timestamp": timestamp,
            },
        )
        connection.execute(
            text(
                """INSERT INTO generations
                (id, kind, status, parent_generation_id, retry_of_generation_id,
                 retry_attempt, settings_snapshot_json, snapshot_schema_version,
                 checkpoint_name, vae_name, seed, width, height,
                 positive_prompt_search, negative_prompt_search,
                 workflow_template_id, workflow_template_version, comfy_prompt_id,
                 favorite, user_note, error_code, error_summary, created_at,
                 started_at, completed_at, updated_at)
                VALUES (:id, 'standard', 'pending', NULL, NULL, 0, :snapshot, 1,
                 'sdxl.safetensors', NULL, 123, 1024, 1024, 'a cat', '',
                 'sdxl_txt2img', '1.0', NULL, 0, NULL, NULL, NULL,
                 :timestamp, NULL, NULL, :timestamp)"""
            ),
            {"id": normal_generation, "snapshot": snapshot, "timestamp": timestamp},
        )
        connection.execute(
            text(
                """INSERT INTO generation_jobs
                (id, generation_id, status, comfy_prompt_id, progress_value,
                 progress_maximum, current_node, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, cancelled_at, error_code,
                 error_summary, created_at, started_at, completed_at, updated_at)
                VALUES (:id, :generation_id, 'pending', NULL, NULL, NULL, NULL,
                 NULL, NULL, NULL, NULL, NULL, NULL, NULL, :timestamp, NULL,
                 NULL, :timestamp)"""
            ),
            {"id": normal_job, "generation_id": normal_generation, "timestamp": timestamp},
        )
        connection.execute(
            text(
                """INSERT INTO generation_queue_entries
                (generation_id, job_id, batch_id, batch_index, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, enqueued_at, updated_at)
                VALUES (:generation_id, :job_id, NULL, 0, NULL, NULL, NULL, NULL,
                 :timestamp, :timestamp)"""
            ),
            {"generation_id": normal_generation, "job_id": normal_job, "timestamp": timestamp},
        )
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    with engine.connect() as connection:
        mismatch = connection.execute(
            text("SELECT status, error_code FROM generations WHERE id=:id"),
            {"id": mismatch_generation},
        ).one()
        prompt_entry = connection.execute(
            text("SELECT submission_state FROM generation_queue_entries WHERE generation_id=:id"),
            {"id": prompt_generation},
        ).one()
        normal = connection.execute(
            text(
                """SELECT g.status, j.status AS job_status, q.submission_state,
                          g.error_code
                   FROM generations AS g
                   JOIN generation_jobs AS j ON j.generation_id=g.id
                   JOIN generation_queue_entries AS q ON q.generation_id=g.id
                  WHERE g.id=:id"""
            ),
            {"id": normal_generation},
        ).one()
    assert mismatch.status == "failed"
    assert mismatch.error_code == "migration_status_ambiguous"
    assert prompt_entry.submission_state == "submitted"
    assert normal.status == "pending"
    assert normal.job_status == "pending"
    assert normal.submission_state == "ready"
    engine.dispose()


def test_phase4_0008_quarantines_old_0007_pending_to_failed_correction(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'phase4-0008.sqlite3').as_posix()}"
    root = Path(__file__).parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0007_phase4_terminal_state_repair")
    engine = create_engine(database_url)
    snapshot = GenerationSettingsSnapshot.from_settings(_settings()).to_json()
    timestamp = "2026-01-01 00:00:00"
    generation_id = "00000000-0000-0000-0000-000000000401"
    job_id = "00000000-0000-0000-0000-000000000411"
    with engine.begin() as connection:
        connection.execute(
            text(
                """INSERT INTO generations
                (id, kind, status, parent_generation_id, retry_of_generation_id,
                 retry_attempt, settings_snapshot_json, snapshot_schema_version,
                 checkpoint_name, vae_name, seed, width, height,
                 positive_prompt_search, negative_prompt_search,
                 workflow_template_id, workflow_template_version, comfy_prompt_id,
                 favorite, user_note, error_code, error_summary, created_at,
                 started_at, completed_at, updated_at)
                VALUES (:id, 'standard', 'failed', NULL, NULL, 0, :snapshot, 1,
                 'sdxl.safetensors', NULL, 123, 1024, 1024, 'a cat', '',
                 'sdxl_txt2img', '1.0', NULL, 0, NULL,
                 'migration_status_mismatch', 'old', :timestamp, NULL,
                 :timestamp, :timestamp)"""
            ),
            {"id": generation_id, "snapshot": snapshot, "timestamp": timestamp},
        )
        connection.execute(
            text(
                """INSERT INTO generation_jobs
                (id, generation_id, status, comfy_prompt_id, progress_value,
                 progress_maximum, current_node, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, cancelled_at, error_code,
                 error_summary, created_at, started_at, completed_at, updated_at)
                VALUES (:id, :generation_id, 'failed', NULL, NULL, NULL, NULL,
                 NULL, NULL, NULL, NULL, NULL, 'migration_status_mismatch',
                 'old', :timestamp, NULL, :timestamp, :timestamp)"""
            ),
            {"id": job_id, "generation_id": generation_id, "timestamp": timestamp},
        )
        connection.execute(
            text(
                """INSERT INTO generation_queue_entries
                (generation_id, job_id, batch_id, batch_index, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, submission_state,
                 submission_token, submission_started_at, enqueued_at, updated_at)
                VALUES (:generation_id, :job_id, NULL, 0, NULL, NULL, NULL, NULL,
                        'ambiguous', NULL, NULL, :timestamp, :timestamp)"""
            ),
            {"generation_id": generation_id, "job_id": job_id, "timestamp": timestamp},
        )
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """SELECT g.status, g.error_code, q.submission_state
                   FROM generations AS g
                   JOIN generation_queue_entries AS q ON q.generation_id=g.id
                  WHERE g.id=:id"""
            ),
            {"id": generation_id},
        ).one()
    assert row.status == "failed"
    assert row.error_code == "migration_status_ambiguous"
    assert row.submission_state == "ambiguous"
    engine.dispose()


def test_phase4_migration_deduplicates_retry_links_before_unique_indexes(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'phase4-duplicates.sqlite3').as_posix()}"
    root = Path(__file__).parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0004_phase4_persistent_generation_queue")
    engine = create_engine(database_url)
    snapshot = GenerationSettingsSnapshot.from_settings(_settings()).to_json()
    timestamp = "2026-01-01 00:00:00"
    source_generation = "00000000-0000-0000-0000-000000000101"
    retry_generation_a = "00000000-0000-0000-0000-000000000102"
    retry_generation_b = "00000000-0000-0000-0000-000000000103"
    source_batch = "00000000-0000-0000-0000-000000000201"
    retry_batch_a = "00000000-0000-0000-0000-000000000202"
    retry_batch_b = "00000000-0000-0000-0000-000000000203"
    with engine.begin() as connection:
        for generation_id, retry_of in (
            (source_generation, None),
            (retry_generation_a, source_generation),
            (retry_generation_b, source_generation),
        ):
            connection.execute(
                text(
                    """INSERT INTO generations
                    (id, kind, status, parent_generation_id, retry_of_generation_id,
                     retry_attempt, settings_snapshot_json, snapshot_schema_version,
                     checkpoint_name, vae_name, seed, width, height,
                     positive_prompt_search, negative_prompt_search,
                     workflow_template_id, workflow_template_version, comfy_prompt_id,
                     favorite, user_note, error_code, error_summary, created_at,
                     started_at, completed_at, updated_at)
                    VALUES (:id, 'standard', 'failed', NULL, :retry_of, 0, :snapshot, 1,
                     'sdxl.safetensors', NULL, 123, 1024, 1024, 'a cat', '',
                     'sdxl_txt2img', '1.0', NULL, 0, NULL, NULL, NULL,
                     :timestamp, NULL, :timestamp, :timestamp)"""
                ),
                {
                    "id": generation_id,
                    "retry_of": retry_of,
                    "snapshot": snapshot,
                    "timestamp": timestamp,
                },
            )
        for batch_id, retry_of in (
            (source_batch, None),
            (retry_batch_a, source_batch),
            (retry_batch_b, source_batch),
        ):
            connection.execute(
                text(
                    """INSERT INTO generation_batches
                    (id, name, item_count, seed_strategy, start_seed, seed_step,
                     retry_of_batch_id, created_at, updated_at)
                    VALUES (:id, 'batch', 1, 'random', NULL, 1, :retry_of,
                            :timestamp, :timestamp)"""
                ),
                {"id": batch_id, "retry_of": retry_of, "timestamp": timestamp},
            )
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    with engine.connect() as connection:
        generation_links = connection.execute(
            text(
                "SELECT id, retry_of_generation_id FROM generations "
                "WHERE id IN (:a, :b) ORDER BY id"
            ),
            {"a": retry_generation_a, "b": retry_generation_b},
        ).all()
        batch_links = connection.execute(
            text(
                "SELECT id, retry_of_batch_id FROM generation_batches "
                "WHERE id IN (:a, :b) ORDER BY id"
            ),
            {"a": retry_batch_a, "b": retry_batch_b},
        ).all()
    assert generation_links[0].retry_of_generation_id == source_generation
    assert generation_links[1].retry_of_generation_id is None
    assert batch_links[0].retry_of_batch_id == source_batch
    assert batch_links[1].retry_of_batch_id is None
    engine.dispose()


def test_phase4_followup_migration_repairs_terminal_and_mismatched_rows(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'phase4-followup.sqlite3').as_posix()}"
    root = Path(__file__).parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0006_phase4_reconcile_existing_state")
    engine = create_engine(database_url)
    snapshot = GenerationSettingsSnapshot.from_settings(_settings()).to_json()
    timestamp = "2026-01-01 00:00:00"

    def insert_pair(
        connection: Any,
        generation_id: str,
        job_id: str,
        generation_status: str,
        job_status: str,
        generation_prompt: str | None,
        job_prompt: str | None,
        queue_state: str,
        cancel_requested: bool = False,
    ) -> None:
        connection.execute(
            text(
                """INSERT INTO generations
                (id, kind, status, parent_generation_id, retry_of_generation_id,
                 retry_attempt, settings_snapshot_json, snapshot_schema_version,
                 checkpoint_name, vae_name, seed, width, height,
                 positive_prompt_search, negative_prompt_search,
                 workflow_template_id, workflow_template_version, comfy_prompt_id,
                 favorite, user_note, error_code, error_summary, created_at,
                 started_at, completed_at, updated_at)
                VALUES (:id, 'standard', :status, NULL, NULL, 0, :snapshot, 1,
                 'sdxl.safetensors', NULL, 123, 1024, 1024, 'a cat', '',
                 'sdxl_txt2img', '1.0', :prompt, 0, NULL, NULL, NULL,
                 :timestamp, NULL, NULL, :timestamp)"""
            ),
            {
                "id": generation_id,
                "status": generation_status,
                "snapshot": snapshot,
                "prompt": generation_prompt,
                "timestamp": timestamp,
            },
        )
        connection.execute(
            text(
                """INSERT INTO generation_jobs
                (id, generation_id, status, comfy_prompt_id, progress_value,
                 progress_maximum, current_node, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, cancelled_at, error_code,
                 error_summary, created_at, started_at, completed_at, updated_at)
                VALUES (:id, :generation_id, :status, :prompt, NULL, NULL, NULL,
                 NULL, NULL, NULL, :cancel_requested, NULL, NULL, NULL,
                 :timestamp, NULL, NULL, :timestamp)"""
            ),
            {
                "id": job_id,
                "generation_id": generation_id,
                "status": job_status,
                "prompt": job_prompt,
                "cancel_requested": timestamp if cancel_requested else None,
                "timestamp": timestamp,
            },
        )
        connection.execute(
            text(
                """INSERT INTO generation_queue_entries
                (generation_id, job_id, batch_id, batch_index, worker_id, claimed_at,
                 lease_expires_at, cancel_requested_at, submission_state,
                 submission_token, submission_started_at, enqueued_at, updated_at)
                VALUES (:generation_id, :job_id, NULL, 0, NULL, NULL, NULL, :cancel,
                        :state, NULL, NULL, :timestamp, :timestamp)"""
            ),
            {
                "generation_id": generation_id,
                "job_id": job_id,
                "cancel": timestamp if cancel_requested else None,
                "state": queue_state,
                "timestamp": timestamp,
            },
        )

    with engine.begin() as connection:
        insert_pair(
            connection,
            "00000000-0000-0000-0000-000000000301",
            "00000000-0000-0000-0000-000000000311",
            "failed",
            "failed",
            "prompt-failed",
            "prompt-failed",
            "submitted",
        )
        insert_pair(
            connection,
            "00000000-0000-0000-0000-000000000302",
            "00000000-0000-0000-0000-000000000312",
            "pending",
            "pending",
            None,
            None,
            "ready",
            cancel_requested=True,
        )
        insert_pair(
            connection,
            "00000000-0000-0000-0000-000000000303",
            "00000000-0000-0000-0000-000000000313",
            "queued",
            "queued",
            "prompt-a",
            "prompt-b",
            "submitted",
        )
        insert_pair(
            connection,
            "00000000-0000-0000-0000-000000000304",
            "00000000-0000-0000-0000-000000000314",
            "completed",
            "queued",
            "prompt-complete",
            "prompt-complete",
            "submitted",
        )
        connection.execute(
            text(
                """INSERT INTO generation_artifacts
                (id, generation_id, artifact_type, local_path, sha256, size_bytes,
                 width, height, mime_type, created_at)
                VALUES ('00000000-0000-0000-0000-000000000321',
                        '00000000-0000-0000-0000-000000000304', 'image',
                        'generations/image.png', 'a', 1, 1, 1, 'image/png', :timestamp)"""
            ),
            {"timestamp": timestamp},
        )
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """SELECT g.id, g.status AS generation_status, j.status AS job_status,
                          g.comfy_prompt_id AS generation_prompt,
                          j.comfy_prompt_id AS job_prompt,
                          q.submission_state, g.error_code
                   FROM generations AS g
                   JOIN generation_jobs AS j ON j.generation_id=g.id
                   JOIN generation_queue_entries AS q ON q.generation_id=g.id
                  WHERE g.id IN (:failed, :cancelled, :mismatch, :completed)
                  ORDER BY g.id"""
            ),
            {
                "failed": "00000000-0000-0000-0000-000000000301",
                "cancelled": "00000000-0000-0000-0000-000000000302",
                "mismatch": "00000000-0000-0000-0000-000000000303",
                "completed": "00000000-0000-0000-0000-000000000304",
            },
        ).all()
    by_id = {row.id: row for row in rows}
    assert by_id["00000000-0000-0000-0000-000000000301"].generation_status == "failed"
    assert by_id["00000000-0000-0000-0000-000000000301"].job_status == "failed"
    assert by_id["00000000-0000-0000-0000-000000000302"].generation_status == "pending"
    assert by_id["00000000-0000-0000-0000-000000000302"].job_status == "pending"
    assert by_id["00000000-0000-0000-0000-000000000302"].submission_state == "ambiguous"
    mismatch = by_id["00000000-0000-0000-0000-000000000303"]
    assert mismatch.generation_prompt == "prompt-a"
    assert mismatch.job_prompt == "prompt-b"
    assert mismatch.submission_state == "ambiguous"
    assert mismatch.error_code == "migration_prompt_id_mismatch"
    assert by_id["00000000-0000-0000-0000-000000000304"].job_status == "completed"
    engine.dispose()
