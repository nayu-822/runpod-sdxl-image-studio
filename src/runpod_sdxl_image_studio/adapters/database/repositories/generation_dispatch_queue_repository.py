"""SQLite-backed FIFO dispatch queue repository for Phase 4."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import (
    GenerationBatchModel,
    GenerationJobModel,
    GenerationLoraModel,
    GenerationModel,
    GenerationQueueEntryModel,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    _generation_domain,
    _job_domain,
)
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import (
    BatchSeedStrategy,
    GenerationBatch,
    GenerationQueueEntry,
    GenerationQueueItem,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot


class GenerationDispatchQueueRepositoryError(RuntimeError):
    """Safe application-facing persistence error for the dispatch queue."""


class GenerationDispatchQueueRepositoryProtocol(Protocol):
    def enqueue_single(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        kind: GenerationKind = GenerationKind.STANDARD,
        parent_generation_id: UUID | None = None,
        generation_id: UUID | None = None,
        job_id: UUID | None = None,
        batch_id: UUID | None = None,
        batch_index: int = 0,
        retry_of_generation_id: UUID | None = None,
        retry_attempt: int = 0,
        enqueued_at: datetime | None = None,
    ) -> GenerationQueueItem: ...

    def enqueue_batch(
        self,
        snapshots: Sequence[GenerationSettingsSnapshot],
        *,
        name: str,
        seed_strategy: BatchSeedStrategy,
        start_seed: int | None,
        seed_step: int,
        retry_of_batch_id: UUID | None = None,
        enqueued_at: datetime | None = None,
        retry_of_generations: Sequence[UUID | None] | None = None,
        retry_attempts: Sequence[int] | None = None,
    ) -> tuple[GenerationBatch, tuple[GenerationQueueItem, ...]]: ...

    def claim_next(
        self, worker_id: str, *, lease_seconds: float, now: datetime | None = None
    ) -> GenerationQueueItem | None: ...

    def renew_lease(
        self,
        sequence: int,
        worker_id: str,
        *,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> GenerationQueueItem: ...

    def release_claim(self, sequence: int, worker_id: str) -> None: ...

    def request_cancel(
        self, generation_id: UUID, *, now: datetime | None = None
    ) -> GenerationQueueItem: ...

    def mark_cancelled(
        self, generation_id: UUID, *, now: datetime | None = None
    ) -> GenerationQueueItem: ...

    def list_queue(
        self,
        *,
        statuses: Sequence[GenerationStatus] | None = None,
        batch_id: UUID | None = None,
        limit: int = 200,
    ) -> tuple[GenerationQueueItem, ...]: ...

    def get_queue_item(self, generation_id: UUID) -> GenerationQueueItem | None: ...

    def list_batch_items(self, batch_id: UUID) -> tuple[GenerationQueueItem, ...]: ...

    def reconcile_expired_claims(self, *, now: datetime | None = None) -> int: ...

    def mark_reconciliation_failed(
        self,
        generation_id: UUID,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem: ...


class GenerationDispatchQueueRepository(GenerationDispatchQueueRepositoryProtocol):
    """Atomic queue persistence that does not depend on ``FOR UPDATE``."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def enqueue_single(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        kind: GenerationKind = GenerationKind.STANDARD,
        parent_generation_id: UUID | None = None,
        generation_id: UUID | None = None,
        job_id: UUID | None = None,
        batch_id: UUID | None = None,
        batch_index: int = 0,
        retry_of_generation_id: UUID | None = None,
        retry_attempt: int = 0,
        enqueued_at: datetime | None = None,
    ) -> GenerationQueueItem:
        if batch_index < 0 or retry_attempt < 0:
            raise GenerationDispatchQueueRepositoryError("queue indexes must not be negative")
        timestamp = _utc(enqueued_at or datetime.now(UTC))
        generation_id = generation_id or uuid4()
        job_id = job_id or uuid4()
        try:
            with session_scope(self._session_factory) as session:
                generation_row, job_row = _insert_generation_and_job(
                    session,
                    snapshot,
                    generation_id=generation_id,
                    job_id=job_id,
                    kind=kind,
                    parent_generation_id=parent_generation_id,
                    retry_of_generation_id=retry_of_generation_id,
                    retry_attempt=retry_attempt,
                    timestamp=timestamp,
                )
                if (
                    batch_id is not None
                    and session.get(GenerationBatchModel, str(batch_id)) is None
                ):
                    raise GenerationDispatchQueueRepositoryError("generation batch was not found")
                entry = GenerationQueueEntryModel(
                    generation_id=str(generation_id),
                    job_id=str(job_id),
                    batch_id=str(batch_id) if batch_id is not None else None,
                    batch_index=batch_index,
                    enqueued_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(entry)
                session.flush()
                return _queue_item(session, entry, generation_row, job_row)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "generation could not be enqueued"
            ) from exc

    def enqueue_batch(
        self,
        snapshots: Sequence[GenerationSettingsSnapshot],
        *,
        name: str,
        seed_strategy: BatchSeedStrategy,
        start_seed: int | None,
        seed_step: int,
        retry_of_batch_id: UUID | None = None,
        enqueued_at: datetime | None = None,
        retry_of_generations: Sequence[UUID | None] | None = None,
        retry_attempts: Sequence[int] | None = None,
    ) -> tuple[GenerationBatch, tuple[GenerationQueueItem, ...]]:
        if not snapshots:
            raise GenerationDispatchQueueRepositoryError("batch must contain at least one item")
        if retry_of_generations is not None and len(retry_of_generations) != len(snapshots):
            raise GenerationDispatchQueueRepositoryError(
                "retry generation count does not match batch"
            )
        if retry_attempts is not None and len(retry_attempts) != len(snapshots):
            raise GenerationDispatchQueueRepositoryError("retry attempt count does not match batch")
        if not name.strip() or len(name.strip()) > 200:
            raise GenerationDispatchQueueRepositoryError("batch name is invalid")
        if seed_step <= 0 or (start_seed is not None and start_seed < 0):
            raise GenerationDispatchQueueRepositoryError("batch seed settings are invalid")
        timestamp = _utc(enqueued_at or datetime.now(UTC))
        batch_id = uuid4()
        try:
            with session_scope(self._session_factory) as session:
                if (
                    retry_of_batch_id is not None
                    and session.get(GenerationBatchModel, str(retry_of_batch_id)) is None
                ):
                    raise GenerationDispatchQueueRepositoryError("retry batch was not found")
                batch_row = GenerationBatchModel(
                    id=str(batch_id),
                    name=name.strip(),
                    item_count=len(snapshots),
                    seed_strategy=seed_strategy.value,
                    start_seed=start_seed,
                    seed_step=seed_step,
                    retry_of_batch_id=(
                        str(retry_of_batch_id) if retry_of_batch_id is not None else None
                    ),
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(batch_row)
                session.flush()
                items: list[GenerationQueueItem] = []
                for index, snapshot in enumerate(snapshots):
                    generation_row, job_row = _insert_generation_and_job(
                        session,
                        snapshot,
                        generation_id=uuid4(),
                        job_id=uuid4(),
                        kind=GenerationKind.STANDARD,
                        parent_generation_id=None,
                        retry_of_generation_id=(
                            retry_of_generations[index]
                            if retry_of_generations is not None
                            else None
                        ),
                        retry_attempt=(retry_attempts[index] if retry_attempts is not None else 0),
                        timestamp=timestamp,
                    )
                    entry = GenerationQueueEntryModel(
                        generation_id=generation_row.id,
                        job_id=job_row.id,
                        batch_id=str(batch_id),
                        batch_index=index,
                        enqueued_at=timestamp,
                        updated_at=timestamp,
                    )
                    session.add(entry)
                    session.flush()
                    items.append(_queue_item(session, entry, generation_row, job_row))
                return _batch_domain(batch_row), tuple(items)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError("batch could not be enqueued") from exc

    def claim_next(
        self, worker_id: str, *, lease_seconds: float, now: datetime | None = None
    ) -> GenerationQueueItem | None:
        normalized_worker = _worker_id(worker_id)
        timestamp = _utc(now or datetime.now(UTC))
        lease_until = timestamp + _lease_delta(lease_seconds)
        try:
            with session_scope(self._session_factory) as session:
                candidates = session.scalars(
                    select(GenerationQueueEntryModel)
                    .join(
                        GenerationModel,
                        GenerationModel.id == GenerationQueueEntryModel.generation_id,
                    )
                    .join(
                        GenerationJobModel,
                        GenerationJobModel.id == GenerationQueueEntryModel.job_id,
                    )
                    .where(
                        GenerationModel.status == GenerationStatus.PENDING.value,
                        GenerationJobModel.status == GenerationStatus.PENDING.value,
                        GenerationQueueEntryModel.cancel_requested_at.is_(None),
                        GenerationJobModel.cancel_requested_at.is_(None),
                        or_(
                            GenerationQueueEntryModel.lease_expires_at.is_(None),
                            GenerationQueueEntryModel.lease_expires_at <= timestamp,
                        ),
                    )
                    .order_by(GenerationQueueEntryModel.sequence.asc())
                    .limit(20)
                ).all()
                for candidate in candidates:
                    claimed = session.execute(
                        update(GenerationQueueEntryModel)
                        .where(
                            GenerationQueueEntryModel.sequence == candidate.sequence,
                            GenerationQueueEntryModel.cancel_requested_at.is_(None),
                            or_(
                                GenerationQueueEntryModel.lease_expires_at.is_(None),
                                GenerationQueueEntryModel.lease_expires_at <= timestamp,
                            ),
                        )
                        .values(
                            worker_id=normalized_worker,
                            claimed_at=timestamp,
                            lease_expires_at=lease_until,
                            updated_at=timestamp,
                        )
                    ).rowcount  # type: ignore[attr-defined]
                    if claimed != 1:
                        continue
                    job = session.get(GenerationJobModel, candidate.job_id)
                    generation = session.get(GenerationModel, candidate.generation_id)
                    if job is None or generation is None:
                        raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                    job.worker_id = normalized_worker
                    job.claimed_at = timestamp
                    job.lease_expires_at = lease_until
                    job.updated_at = timestamp
                    session.flush()
                    return _queue_item(session, candidate, generation, job)
                return None
        except GenerationDispatchQueueRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError("queue claim failed") from exc

    def renew_lease(
        self,
        sequence: int,
        worker_id: str,
        *,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        lease_until = timestamp + _lease_delta(lease_seconds)
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.sequence == sequence,
                        GenerationQueueEntryModel.worker_id == _worker_id(worker_id),
                    )
                )
                if entry is None:
                    raise GenerationDispatchQueueRepositoryError("queue lease was not found")
                entry.lease_expires_at = lease_until
                entry.updated_at = timestamp
                job = session.get(GenerationJobModel, entry.job_id)
                generation = session.get(GenerationModel, entry.generation_id)
                if job is None or generation is None:
                    raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                job.lease_expires_at = lease_until
                job.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise GenerationDispatchQueueRepositoryError("queue lease renewal failed") from exc

    def release_claim(self, sequence: int, worker_id: str) -> None:
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.sequence == sequence,
                        GenerationQueueEntryModel.worker_id == _worker_id(worker_id),
                    )
                )
                if entry is None:
                    return
                entry.worker_id = None
                entry.claimed_at = None
                entry.lease_expires_at = None
                entry.updated_at = datetime.now(UTC)
                job = session.get(GenerationJobModel, entry.job_id)
                if job is not None:
                    job.worker_id = None
                    job.claimed_at = None
                    job.lease_expires_at = None
                    job.updated_at = datetime.now(UTC)
                session.flush()
        except SQLAlchemyError as exc:
            raise GenerationDispatchQueueRepositoryError("queue claim release failed") from exc

    def request_cancel(
        self, generation_id: UUID, *, now: datetime | None = None
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.generation_id == str(generation_id)
                    )
                )
                if entry is None:
                    raise GenerationDispatchQueueRepositoryError("queue item was not found")
                generation = session.get(GenerationModel, str(generation_id))
                job = session.get(GenerationJobModel, entry.job_id)
                if generation is None or job is None:
                    raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                status = GenerationStatus(generation.status)
                if status in {GenerationStatus.COMPLETED, GenerationStatus.FAILED}:
                    raise GenerationDispatchQueueRepositoryError("terminal job cannot be cancelled")
                entry.cancel_requested_at = entry.cancel_requested_at or timestamp
                entry.updated_at = timestamp
                job.cancel_requested_at = job.cancel_requested_at or timestamp
                job.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "cancel request could not be saved"
            ) from exc

    def mark_cancelled(
        self, generation_id: UUID, *, now: datetime | None = None
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.generation_id == str(generation_id)
                    )
                )
                if entry is None:
                    raise GenerationDispatchQueueRepositoryError("queue item was not found")
                generation = session.get(GenerationModel, str(generation_id))
                job = session.get(GenerationJobModel, entry.job_id)
                if generation is None or job is None:
                    raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                status = GenerationStatus(generation.status)
                if status in {GenerationStatus.COMPLETED, GenerationStatus.FAILED}:
                    raise GenerationDispatchQueueRepositoryError("terminal job cannot be cancelled")
                if status is not GenerationStatus.CANCELLED:
                    generation.status = GenerationStatus.CANCELLED.value
                    generation.completed_at = timestamp
                    generation.updated_at = timestamp
                if GenerationStatus(job.status) is not GenerationStatus.CANCELLED:
                    job.status = GenerationStatus.CANCELLED.value
                    job.cancelled_at = timestamp
                    job.completed_at = timestamp
                    job.updated_at = timestamp
                entry.worker_id = None
                entry.claimed_at = None
                entry.lease_expires_at = None
                entry.cancel_requested_at = entry.cancel_requested_at or timestamp
                entry.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError("job could not be cancelled") from exc

    def list_queue(
        self,
        *,
        statuses: Sequence[GenerationStatus] | None = None,
        batch_id: UUID | None = None,
        limit: int = 200,
    ) -> tuple[GenerationQueueItem, ...]:
        try:
            with session_scope(self._session_factory) as session:
                statement = (
                    select(GenerationQueueEntryModel)
                    .join(
                        GenerationModel,
                        GenerationModel.id == GenerationQueueEntryModel.generation_id,
                    )
                    .order_by(GenerationQueueEntryModel.sequence.asc())
                    .limit(min(max(1, limit), 500))
                )
                if statuses:
                    statement = statement.where(
                        GenerationModel.status.in_([status.value for status in statuses])
                    )
                if batch_id is not None:
                    statement = statement.where(GenerationQueueEntryModel.batch_id == str(batch_id))
                entries = session.scalars(statement).all()
                return tuple(_queue_item_from_entry(session, entry) for entry in entries)
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError("queue could not be listed") from exc

    def get_queue_item(self, generation_id: UUID) -> GenerationQueueItem | None:
        try:
            with session_scope(self._session_factory) as session:
                return _load_item_by_generation(session, generation_id)
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError("queue item could not be read") from exc

    def list_batch_items(self, batch_id: UUID) -> tuple[GenerationQueueItem, ...]:
        return self.list_queue(batch_id=batch_id, limit=500)

    def reconcile_expired_claims(self, *, now: datetime | None = None) -> int:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                entries = session.scalars(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.lease_expires_at.is_not(None),
                        GenerationQueueEntryModel.lease_expires_at <= timestamp,
                        GenerationQueueEntryModel.cancel_requested_at.is_(None),
                    )
                ).all()
                for entry in entries:
                    entry.worker_id = None
                    entry.claimed_at = None
                    entry.lease_expires_at = None
                    entry.updated_at = timestamp
                    job = session.get(GenerationJobModel, entry.job_id)
                    if job is not None:
                        job.worker_id = None
                        job.claimed_at = None
                        job.lease_expires_at = None
                        job.updated_at = timestamp
                session.flush()
                return len(entries)
        except SQLAlchemyError as exc:
            raise GenerationDispatchQueueRepositoryError("queue reconciliation failed") from exc

    def mark_reconciliation_failed(
        self,
        generation_id: UUID,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        summary = reason.strip()[:1000] or "queue item could not be reconciled"
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.generation_id == str(generation_id)
                    )
                )
                generation = session.get(GenerationModel, str(generation_id))
                if entry is None or generation is None:
                    raise GenerationDispatchQueueRepositoryError("queue item was not found")
                job = session.get(GenerationJobModel, entry.job_id)
                if job is None:
                    raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                if GenerationStatus(generation.status) not in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.CANCELLED,
                    GenerationStatus.FAILED,
                }:
                    generation.status = GenerationStatus.FAILED.value
                    generation.error_code = "reconciliation_prompt_missing"
                    generation.error_summary = summary
                    generation.completed_at = timestamp
                    generation.updated_at = timestamp
                if GenerationStatus(job.status) not in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.CANCELLED,
                    GenerationStatus.FAILED,
                }:
                    job.status = GenerationStatus.FAILED.value
                    job.error_code = "reconciliation_prompt_missing"
                    job.error_summary = summary
                    job.completed_at = timestamp
                    job.updated_at = timestamp
                entry.worker_id = None
                entry.claimed_at = None
                entry.lease_expires_at = None
                entry.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "queue reconciliation failure could not be persisted"
            ) from exc


def _insert_generation_and_job(
    session: Session,
    snapshot: GenerationSettingsSnapshot,
    *,
    generation_id: UUID,
    job_id: UUID,
    kind: GenerationKind,
    parent_generation_id: UUID | None,
    retry_of_generation_id: UUID | None,
    retry_attempt: int,
    timestamp: datetime,
) -> tuple[GenerationModel, GenerationJobModel]:
    if (
        parent_generation_id is not None
        and session.get(GenerationModel, str(parent_generation_id)) is None
    ):
        raise GenerationDispatchQueueRepositoryError("parent generation was not found")
    if (
        retry_of_generation_id is not None
        and session.get(GenerationModel, str(retry_of_generation_id)) is None
    ):
        raise GenerationDispatchQueueRepositoryError("retry generation was not found")
    generation = GenerationModel(
        id=str(generation_id),
        kind=kind.value,
        status=GenerationStatus.PENDING.value,
        parent_generation_id=str(parent_generation_id) if parent_generation_id else None,
        retry_of_generation_id=str(retry_of_generation_id) if retry_of_generation_id else None,
        retry_attempt=retry_attempt,
        settings_snapshot_json=snapshot.to_json(),
        snapshot_schema_version=snapshot.schema_version,
        checkpoint_name=snapshot.checkpoint_name,
        vae_name=snapshot.vae_name,
        seed=snapshot.seed,
        width=snapshot.width,
        height=snapshot.height,
        positive_prompt_search=snapshot.positive_prompt,
        negative_prompt_search=snapshot.negative_prompt,
        workflow_template_id=snapshot.workflow_template_id,
        workflow_template_version=snapshot.workflow_template_version,
        favorite=False,
        created_at=timestamp,
        updated_at=timestamp,
    )
    job = GenerationJobModel(
        id=str(job_id),
        generation_id=str(generation_id),
        status=GenerationStatus.PENDING.value,
        comfy_prompt_id=None,
        created_at=timestamp,
        updated_at=timestamp,
    )
    session.add(generation)
    session.flush()
    session.add(job)
    session.add_all(
        GenerationLoraModel(
            generation_id=str(generation_id),
            lora_name=lora.name,
            order_index=lora.order,
            model_strength=lora.model_strength,
            clip_strength=lora.clip_strength,
        )
        for lora in snapshot.loras
    )
    session.flush()
    return generation, job


def _load_item_by_generation(session: Session, generation_id: UUID) -> GenerationQueueItem | None:
    entry = session.scalar(
        select(GenerationQueueEntryModel).where(
            GenerationQueueEntryModel.generation_id == str(generation_id)
        )
    )
    return _queue_item_from_entry(session, entry) if entry is not None else None


def _queue_item_from_entry(
    session: Session, entry: GenerationQueueEntryModel
) -> GenerationQueueItem:
    generation = session.get(GenerationModel, entry.generation_id)
    job = session.get(GenerationJobModel, entry.job_id)
    if generation is None or job is None or job.generation_id != entry.generation_id:
        raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
    return _queue_item(session, entry, generation, job)


def _queue_item(
    session: Session,
    entry: GenerationQueueEntryModel,
    generation: GenerationModel,
    job: GenerationJobModel,
) -> GenerationQueueItem:
    batch = session.get(GenerationBatchModel, entry.batch_id) if entry.batch_id else None
    return GenerationQueueItem(
        entry=GenerationQueueEntry(
            sequence=entry.sequence,
            generation_id=UUID(entry.generation_id),
            job_id=UUID(entry.job_id),
            batch_id=UUID(entry.batch_id) if entry.batch_id else None,
            batch_index=entry.batch_index,
            worker_id=entry.worker_id,
            claimed_at=_utc_optional(entry.claimed_at),
            lease_expires_at=_utc_optional(entry.lease_expires_at),
            cancel_requested_at=_utc_optional(entry.cancel_requested_at),
            enqueued_at=_utc(entry.enqueued_at),
            updated_at=_utc(entry.updated_at),
        ),
        generation=_generation_domain(generation),
        job=_job_domain(job),
        batch=_batch_domain(batch) if batch is not None else None,
    )


def _batch_domain(row: GenerationBatchModel) -> GenerationBatch:
    return GenerationBatch(
        id=UUID(row.id),
        name=row.name,
        item_count=row.item_count,
        seed_strategy=BatchSeedStrategy(row.seed_strategy),
        start_seed=row.start_seed,
        seed_step=row.seed_step,
        retry_of_batch_id=UUID(row.retry_of_batch_id) if row.retry_of_batch_id else None,
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
    )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _utc_optional(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None


def _lease_delta(seconds: float) -> timedelta:
    if seconds <= 0:
        raise GenerationDispatchQueueRepositoryError("lease duration must be positive")
    return timedelta(seconds=seconds)


def _worker_id(worker_id: str) -> str:
    normalized = worker_id.strip()
    if not normalized or len(normalized) > 200:
        raise GenerationDispatchQueueRepositoryError("worker id is invalid")
    return normalized


__all__ = [
    "GenerationDispatchQueueRepository",
    "GenerationDispatchQueueRepositoryError",
    "GenerationDispatchQueueRepositoryProtocol",
]
