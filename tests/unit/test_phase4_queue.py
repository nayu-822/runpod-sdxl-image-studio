"""Phase 4 persistence, batch, cancellation, retry, and worker tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

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
from runpod_sdxl_image_studio.domain.generation_queue import BatchSeedStrategy
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.jobs.generation_queue_worker import GenerationQueueWorker
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


def test_reconciliation_fails_stale_queued_item_without_prompt() -> None:
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

    worker = GenerationQueueWorker(
        repository,
        object(),
        Settings(_env_file=None, reconciliation_grace_seconds=0),
        reconcile_handler=None,
    )
    asyncio.run(worker.reconcile())
    reconciled = repository.get_queue_item(item.generation.id)
    assert reconciled is not None
    assert reconciled.generation.status is GenerationStatus.FAILED
    assert reconciled.generation.error_code == "reconciliation_prompt_missing"
    engine.dispose()


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
