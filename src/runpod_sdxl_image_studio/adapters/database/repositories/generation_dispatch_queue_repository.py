"""SQLite-backed FIFO dispatch queue repository for Phase 4."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, aliased, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import (
    GenerationArtifactModel,
    GenerationBatchModel,
    GenerationJobModel,
    GenerationLoraModel,
    GenerationModel,
    GenerationQueueEntryModel,
    GenerationUpscaleSettingsModel,
    MetadataImportModel,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    _generation_domain,
    _job_domain,
)
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType
from runpod_sdxl_image_studio.domain.generation_queue import (
    BatchSeedStrategy,
    GenerationBatch,
    GenerationQueueEntry,
    GenerationQueueItem,
    QueueHealthCounts,
    SubmissionState,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSettingsSnapshot


class GenerationDispatchQueueRepositoryError(RuntimeError):
    """Safe application-facing persistence error for the dispatch queue."""


_AMBIGUOUS_PROMPT_RESOLUTION_CODES = frozenset(
    {
        "prompt_submission_ambiguous",
        "migration_prompt_id_mismatch",
        "migration_status_ambiguous",
        "migration_status_mismatch",
    }
)
_MIGRATION_AMBIGUOUS_PROMPT_RECOVERY_CODES = frozenset(
    {
        "migration_prompt_id_mismatch",
        "migration_status_ambiguous",
        "migration_status_mismatch",
    }
)


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
        pending_limit: int | None = None,
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
        pending_limit: int | None = None,
        enqueued_at: datetime | None = None,
        retry_of_generations: Sequence[UUID | None] | None = None,
        retry_attempts: Sequence[int] | None = None,
    ) -> tuple[GenerationBatch, tuple[GenerationQueueItem, ...]]: ...

    def enqueue_upscale(
        self,
        snapshot: GenerationSettingsSnapshot,
        upscale_snapshot: UpscaleSettingsSnapshot,
        *,
        parent_generation_id: UUID | None,
        source_artifact_id: UUID | None = None,
        source_import_id: UUID | None = None,
        generation_id: UUID | None = None,
        job_id: UUID | None = None,
        retry_of_generation_id: UUID | None = None,
        retry_attempt: int = 0,
        pending_limit: int | None = None,
        enqueued_at: datetime | None = None,
    ) -> GenerationQueueItem: ...

    def claim_next(
        self, worker_id: str, *, lease_seconds: float, now: datetime | None = None
    ) -> GenerationQueueItem | None: ...

    def begin_submission(
        self, sequence: int, worker_id: str, *, now: datetime | None = None
    ) -> GenerationQueueItem: ...

    def mark_submitted(
        self,
        sequence: int,
        worker_id: str,
        submission_token: str,
        prompt_id: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem: ...

    def mark_submission_ambiguous(
        self,
        sequence: int,
        worker_id: str,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem: ...

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

    def link_ambiguous_prompt(
        self,
        generation_id: UUID,
        prompt_id: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem: ...

    def fail_ambiguous_prompt(
        self,
        generation_id: UUID,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem: ...

    def list_queue(
        self,
        *,
        statuses: Sequence[GenerationStatus] | None = None,
        batch_id: UUID | None = None,
        limit: int = 200,
    ) -> tuple[GenerationQueueItem, ...]: ...

    def get_health_counts(self) -> QueueHealthCounts: ...

    def list_recent_failed(self, limit: int = 100) -> tuple[GenerationQueueItem, ...]: ...

    def get_queue_item(self, generation_id: UUID) -> GenerationQueueItem | None: ...

    def get_latest_status_candidate(self) -> GenerationQueueItem | None: ...

    def list_batch_items(self, batch_id: UUID) -> tuple[GenerationQueueItem, ...]: ...

    def reconcile_expired_claims(self, *, now: datetime | None = None) -> int: ...

    def reconcile_stateless_restore(self, *, now: datetime | None = None) -> int: ...

    def mark_reconciliation_failed(
        self,
        generation_id: UUID,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem: ...

    def mark_prompt_id_mismatch(
        self,
        generation_id: UUID,
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
        pending_limit: int | None = None,
        enqueued_at: datetime | None = None,
    ) -> GenerationQueueItem:
        if batch_index < 0 or retry_attempt < 0:
            raise GenerationDispatchQueueRepositoryError("queue indexes must not be negative")
        timestamp = _utc(enqueued_at or datetime.now(UTC))
        generation_id = generation_id or uuid4()
        job_id = job_id or uuid4()
        try:
            with session_scope(self._session_factory) as session:
                _begin_immediate_if_sqlite(session)
                if retry_of_generation_id is not None:
                    existing = session.scalar(
                        select(GenerationQueueEntryModel)
                        .join(
                            GenerationModel,
                            GenerationModel.id == GenerationQueueEntryModel.generation_id,
                        )
                        .where(
                            GenerationModel.retry_of_generation_id == str(retry_of_generation_id)
                        )
                    )
                    if existing is not None:
                        return _queue_item_from_entry(session, existing)
                _check_pending_capacity(session, pending_limit, 1)
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
                    submission_state=SubmissionState.READY.value,
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

    def enqueue_upscale(
        self,
        snapshot: GenerationSettingsSnapshot,
        upscale_snapshot: UpscaleSettingsSnapshot,
        *,
        parent_generation_id: UUID | None,
        source_artifact_id: UUID | None = None,
        source_import_id: UUID | None = None,
        generation_id: UUID | None = None,
        job_id: UUID | None = None,
        retry_of_generation_id: UUID | None = None,
        retry_attempt: int = 0,
        pending_limit: int | None = None,
        enqueued_at: datetime | None = None,
    ) -> GenerationQueueItem:
        """Create the upscale Generation, Job, settings and queue row atomically."""

        if retry_attempt < 0:
            raise GenerationDispatchQueueRepositoryError("retry attempt must not be negative")
        if upscale_snapshot.source_generation_id != parent_generation_id:
            raise GenerationDispatchQueueRepositoryError("upscale parent does not match snapshot")
        if upscale_snapshot.source_artifact_id != source_artifact_id:
            raise GenerationDispatchQueueRepositoryError("upscale artifact does not match snapshot")
        if upscale_snapshot.source_import_id != source_import_id:
            raise GenerationDispatchQueueRepositoryError("upscale import does not match snapshot")
        if (source_artifact_id is None) == (source_import_id is None):
            raise GenerationDispatchQueueRepositoryError("exactly one upscale source is required")
        timestamp = _utc(enqueued_at or datetime.now(UTC))
        generation_id = generation_id or uuid4()
        job_id = job_id or uuid4()
        try:
            with session_scope(self._session_factory) as session:
                _begin_immediate_if_sqlite(session)
                if retry_of_generation_id is not None:
                    existing = session.scalar(
                        select(GenerationQueueEntryModel)
                        .join(
                            GenerationModel,
                            GenerationModel.id == GenerationQueueEntryModel.generation_id,
                        )
                        .where(
                            GenerationModel.retry_of_generation_id == str(retry_of_generation_id)
                        )
                    )
                    if existing is not None:
                        return _queue_item_from_entry(session, existing)
                _check_pending_capacity(session, pending_limit, 1)
                if source_artifact_id is not None:
                    if parent_generation_id is None:
                        raise GenerationDispatchQueueRepositoryError(
                            "generation source requires a parent"
                        )
                    parent = session.get(GenerationModel, str(parent_generation_id))
                    if parent is None or parent.status != GenerationStatus.COMPLETED.value:
                        raise GenerationDispatchQueueRepositoryError(
                            "upscale parent is not completed"
                        )
                    artifact = session.get(GenerationArtifactModel, str(source_artifact_id))
                    if artifact is None or artifact.generation_id != str(parent_generation_id):
                        raise GenerationDispatchQueueRepositoryError(
                            "upscale source artifact was not found"
                        )
                    if artifact.artifact_type != ArtifactType.IMAGE.value:
                        raise GenerationDispatchQueueRepositoryError(
                            "upscale source artifact is not an image"
                        )
                    if (
                        artifact.width != upscale_snapshot.source_width
                        or artifact.height != upscale_snapshot.source_height
                    ):
                        raise GenerationDispatchQueueRepositoryError(
                            "upscale source dimensions do not match"
                        )
                else:
                    imported = session.get(MetadataImportModel, str(source_import_id))
                    if imported is None or imported.image_mime_type != "image/png":
                        raise GenerationDispatchQueueRepositoryError(
                            "upscale import source was not found"
                        )
                    if (
                        imported.image_width != upscale_snapshot.source_width
                        or imported.image_height != upscale_snapshot.source_height
                    ):
                        raise GenerationDispatchQueueRepositoryError(
                            "upscale import source dimensions do not match"
                        )
                generation, job = _insert_generation_and_job(
                    session,
                    snapshot,
                    generation_id=generation_id,
                    job_id=job_id,
                    kind=GenerationKind.UPSCALE,
                    parent_generation_id=parent_generation_id,
                    retry_of_generation_id=retry_of_generation_id,
                    retry_attempt=retry_attempt,
                    timestamp=timestamp,
                )
                session.add(
                    GenerationUpscaleSettingsModel(
                        generation_id=str(generation_id),
                        source_kind=upscale_snapshot.source_kind.value,
                        source_artifact_id=(
                            str(source_artifact_id) if source_artifact_id is not None else None
                        ),
                        source_import_id=(
                            str(source_import_id) if source_import_id is not None else None
                        ),
                        method=upscale_snapshot.method.value,
                        sizing_mode=upscale_snapshot.sizing_mode.value,
                        scale_factor=upscale_snapshot.requested_scale_factor,
                        target_width=upscale_snapshot.target_width,
                        target_height=upscale_snapshot.target_height,
                        upscaler_name=upscale_snapshot.upscaler_name,
                        denoise=upscale_snapshot.denoise,
                        settings_snapshot_json=upscale_snapshot.to_json(),
                        snapshot_schema_version=upscale_snapshot.schema_version,
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                )
                entry = GenerationQueueEntryModel(
                    generation_id=str(generation_id),
                    job_id=str(job_id),
                    batch_index=0,
                    submission_state=SubmissionState.READY.value,
                    enqueued_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(entry)
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError("upscale could not be enqueued") from exc

    def enqueue_batch(
        self,
        snapshots: Sequence[GenerationSettingsSnapshot],
        *,
        name: str,
        seed_strategy: BatchSeedStrategy,
        start_seed: int | None,
        seed_step: int,
        retry_of_batch_id: UUID | None = None,
        pending_limit: int | None = None,
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
                _begin_immediate_if_sqlite(session)
                if (
                    retry_of_batch_id is not None
                    and session.get(GenerationBatchModel, str(retry_of_batch_id)) is None
                ):
                    raise GenerationDispatchQueueRepositoryError("retry batch was not found")
                if retry_of_batch_id is not None:
                    existing_batch = session.scalar(
                        select(GenerationBatchModel).where(
                            GenerationBatchModel.retry_of_batch_id == str(retry_of_batch_id)
                        )
                    )
                    if existing_batch is not None:
                        existing_items = tuple(
                            _queue_item_from_entry(session, entry)
                            for entry in session.scalars(
                                select(GenerationQueueEntryModel)
                                .where(GenerationQueueEntryModel.batch_id == existing_batch.id)
                                .order_by(GenerationQueueEntryModel.batch_index.asc())
                            ).all()
                        )
                        return _batch_domain(existing_batch), existing_items
                _check_pending_capacity(session, pending_limit, len(snapshots))
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
                        submission_state=SubmissionState.READY.value,
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
                        GenerationQueueEntryModel.submission_state == SubmissionState.READY.value,
                        GenerationJobModel.cancel_requested_at.is_(None),
                        GenerationModel.comfy_prompt_id.is_(None),
                        GenerationJobModel.comfy_prompt_id.is_(None),
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
                            GenerationQueueEntryModel.submission_state
                            == SubmissionState.READY.value,
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
                    ).rowcount
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

    def begin_submission(
        self, sequence: int, worker_id: str, *, now: datetime | None = None
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        normalized_worker = _worker_id(worker_id)
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.sequence == sequence,
                        GenerationQueueEntryModel.worker_id == normalized_worker,
                        GenerationQueueEntryModel.submission_state == SubmissionState.READY.value,
                        GenerationQueueEntryModel.cancel_requested_at.is_(None),
                    )
                )
                if entry is None:
                    raise GenerationDispatchQueueRepositoryError(
                        "queue entry is not ready for submission"
                    )
                generation = session.get(GenerationModel, entry.generation_id)
                job = session.get(GenerationJobModel, entry.job_id)
                if generation is None or job is None:
                    raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                if GenerationStatus(generation.status) is not GenerationStatus.PENDING or (
                    GenerationStatus(job.status) is not GenerationStatus.PENDING
                ):
                    raise GenerationDispatchQueueRepositoryError(
                        "queue entry is not pending for submission"
                    )
                token = str(uuid4())
                entry.submission_state = SubmissionState.SUBMITTING.value
                entry.submission_token = token
                entry.submission_started_at = timestamp
                entry.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "prompt submission state could not be started"
            ) from exc

    def mark_submitted(
        self,
        sequence: int,
        worker_id: str,
        submission_token: str,
        prompt_id: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        normalized_worker = _worker_id(worker_id)
        token = submission_token.strip()
        prompt = prompt_id.strip()
        if not token or not prompt or len(prompt) > 100:
            raise GenerationDispatchQueueRepositoryError(
                "prompt submission identifiers are invalid"
            )
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.sequence == sequence,
                        GenerationQueueEntryModel.worker_id == normalized_worker,
                    )
                )
                if entry is None:
                    raise GenerationDispatchQueueRepositoryError("queue lease was not found")
                generation = session.get(GenerationModel, entry.generation_id)
                job = session.get(GenerationJobModel, entry.job_id)
                if generation is None or job is None:
                    raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                if entry.submission_state == SubmissionState.SUBMITTED.value:
                    if job.comfy_prompt_id == prompt and generation.comfy_prompt_id == prompt:
                        return _queue_item(session, entry, generation, job)
                    raise GenerationDispatchQueueRepositoryError(
                        "queue entry has already been submitted"
                    )
                if (
                    entry.submission_state != SubmissionState.SUBMITTING.value
                    or entry.submission_token != token
                ):
                    raise GenerationDispatchQueueRepositoryError(
                        "prompt submission token does not match"
                    )
                generation.comfy_prompt_id = prompt
                generation.status = GenerationStatus.QUEUED.value
                generation.updated_at = timestamp
                job.comfy_prompt_id = prompt
                job.status = GenerationStatus.QUEUED.value
                job.updated_at = timestamp
                entry.submission_state = SubmissionState.SUBMITTED.value
                entry.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "prompt submission result could not be persisted"
            ) from exc

    def mark_submission_ambiguous(
        self,
        sequence: int,
        worker_id: str,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        normalized_worker = _worker_id(worker_id)
        summary = reason.strip()[:1000] or "prompt submission outcome could not be determined"
        try:
            with session_scope(self._session_factory) as session:
                entry = session.scalar(
                    select(GenerationQueueEntryModel).where(
                        GenerationQueueEntryModel.sequence == sequence,
                        GenerationQueueEntryModel.worker_id == normalized_worker,
                    )
                )
                if entry is None:
                    raise GenerationDispatchQueueRepositoryError("queue lease was not found")
                generation = session.get(GenerationModel, entry.generation_id)
                job = session.get(GenerationJobModel, entry.job_id)
                if generation is None or job is None:
                    raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
                if entry.submission_state == SubmissionState.SUBMITTED.value:
                    return _queue_item(session, entry, generation, job)
                entry.submission_state = SubmissionState.AMBIGUOUS.value
                generation.error_code = "prompt_submission_ambiguous"
                generation.error_summary = summary
                generation.updated_at = timestamp
                job.error_code = "prompt_submission_ambiguous"
                job.error_summary = summary
                entry.worker_id = None
                entry.claimed_at = None
                entry.lease_expires_at = None
                entry.updated_at = timestamp
                job.worker_id = None
                job.claimed_at = None
                job.lease_expires_at = None
                job.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "ambiguous prompt submission could not be persisted"
            ) from exc

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
                if status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED,
                }:
                    return _queue_item(session, entry, generation, job)
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
                job_status = GenerationStatus(job.status)
                if status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                } or job_status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                }:
                    return _queue_item(session, entry, generation, job)
                if status is GenerationStatus.CANCELLED:
                    return _queue_item(session, entry, generation, job)
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

    def link_ambiguous_prompt(
        self,
        generation_id: UUID,
        prompt_id: str,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        prompt = _prompt_identifier(prompt_id)
        try:
            with session_scope(self._session_factory) as session:
                _begin_immediate_if_sqlite(session)
                entry, generation, job = _load_queue_rows(session, generation_id)
                _require_ambiguous_prompt_resolution(entry, generation, job)
                migration_recovery = GenerationStatus.FAILED.value in {
                    generation.status,
                    job.status,
                } and bool(
                    {generation.error_code, job.error_code}.intersection(
                        _MIGRATION_AMBIGUOUS_PROMPT_RECOVERY_CODES
                    )
                )
                summary = "ambiguous prompt manually linked: prompt id was supplied by an operator"
                generation.comfy_prompt_id = prompt
                generation.status = GenerationStatus.QUEUED.value
                if migration_recovery:
                    generation.completed_at = None
                generation.error_code = "prompt_submission_ambiguous_linked"
                generation.error_summary = summary
                generation.updated_at = timestamp
                job.comfy_prompt_id = prompt
                job.status = GenerationStatus.QUEUED.value
                if migration_recovery:
                    job.completed_at = None
                    job.cancelled_at = None
                job.worker_id = None
                job.claimed_at = None
                job.lease_expires_at = None
                job.error_code = "prompt_submission_ambiguous_linked"
                job.error_summary = summary
                job.updated_at = timestamp
                entry.submission_state = SubmissionState.SUBMITTED.value
                entry.worker_id = None
                entry.claimed_at = None
                entry.lease_expires_at = None
                entry.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "ambiguous prompt could not be linked"
            ) from exc

    def fail_ambiguous_prompt(
        self,
        generation_id: UUID,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                _begin_immediate_if_sqlite(session)
                entry, generation, job = _load_queue_rows(session, generation_id)
                _require_ambiguous_prompt_resolution(entry, generation, job)
                summary = "prompt absence was manually confirmed by an operator"
                if GenerationStatus.COMPLETED.value in {generation.status, job.status} or (
                    GenerationStatus.CANCELLED.value in {generation.status, job.status}
                ):
                    raise GenerationDispatchQueueRepositoryError(
                        "completed or cancelled queue item cannot be marked failed"
                    )
                generation.status = GenerationStatus.FAILED.value
                generation.error_code = "prompt_submission_absence_confirmed"
                generation.error_summary = summary
                generation.completed_at = timestamp
                generation.updated_at = timestamp
                job.status = GenerationStatus.FAILED.value
                job.error_code = "prompt_submission_absence_confirmed"
                job.error_summary = summary
                job.completed_at = timestamp
                job.worker_id = None
                job.claimed_at = None
                job.lease_expires_at = None
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
                "ambiguous prompt could not be marked failed"
            ) from exc

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

    def get_health_counts(self) -> QueueHealthCounts:
        """Count every queued Generation without materializing queue items."""

        statuses = (
            GenerationStatus.PENDING.value,
            GenerationStatus.QUEUED.value,
            GenerationStatus.RUNNING.value,
            GenerationStatus.FAILED.value,
        )
        try:
            with session_scope(self._session_factory) as session:
                rows = session.execute(
                    select(
                        GenerationModel.status,
                        func.count(GenerationQueueEntryModel.generation_id),
                    )
                    .join(
                        GenerationQueueEntryModel,
                        GenerationQueueEntryModel.generation_id == GenerationModel.id,
                    )
                    .where(GenerationModel.status.in_(statuses))
                    .group_by(GenerationModel.status)
                ).all()
                counts = {status: 0 for status in statuses}
                for status, count in rows:
                    counts[str(status)] = int(count)
                return QueueHealthCounts(
                    pending_count=counts[GenerationStatus.PENDING.value]
                    + counts[GenerationStatus.QUEUED.value],
                    running_count=counts[GenerationStatus.RUNNING.value],
                    historical_failed_count=counts[GenerationStatus.FAILED.value],
                    unresolved_failed_count=self._count_unresolved_failed_generations(session),
                )
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "queue health counts could not be read"
            ) from exc

    @staticmethod
    def _count_unresolved_failed_generations(session: Session) -> int:
        """Count failed queue rows that have no retry child Generation."""

        retry_child = aliased(GenerationModel)
        retry_exists = select(retry_child.id).where(
            retry_child.retry_of_generation_id == GenerationModel.id
        )
        statement = (
            select(func.count(GenerationModel.id))
            .select_from(GenerationModel)
            .join(
                GenerationQueueEntryModel,
                GenerationQueueEntryModel.generation_id == GenerationModel.id,
            )
            .where(
                GenerationModel.status == GenerationStatus.FAILED.value,
                ~retry_exists.exists(),
            )
        )
        return int(session.scalar(statement) or 0)

    def list_recent_failed(self, limit: int = 100) -> tuple[GenerationQueueItem, ...]:
        """Return only recent failed items, ordered by Generation update time."""

        try:
            with session_scope(self._session_factory) as session:
                statement = (
                    select(GenerationQueueEntryModel)
                    .join(
                        GenerationModel,
                        GenerationModel.id == GenerationQueueEntryModel.generation_id,
                    )
                    .where(GenerationModel.status == GenerationStatus.FAILED.value)
                    .order_by(GenerationModel.updated_at.desc(), GenerationModel.id.desc())
                    .limit(min(max(1, limit), 100))
                )
                entries = session.scalars(statement).all()
                return tuple(_queue_item_from_entry(session, entry) for entry in entries)
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "recent failed queue items could not be read"
            ) from exc

    def get_queue_item(self, generation_id: UUID) -> GenerationQueueItem | None:
        try:
            with session_scope(self._session_factory) as session:
                return _load_item_by_generation(session, generation_id)
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError("queue item could not be read") from exc

    def get_latest_status_candidate(self) -> GenerationQueueItem | None:
        """Return the newest active queue item, or the newest item after that."""

        active_statuses = (
            GenerationStatus.PENDING.value,
            GenerationStatus.QUEUED.value,
            GenerationStatus.RUNNING.value,
        )
        try:
            with session_scope(self._session_factory) as session:
                active_statement = (
                    select(GenerationQueueEntryModel)
                    .join(
                        GenerationModel,
                        GenerationModel.id == GenerationQueueEntryModel.generation_id,
                    )
                    .where(GenerationModel.status.in_(active_statuses))
                    .order_by(GenerationQueueEntryModel.sequence.desc())
                    .limit(1)
                )
                entry = session.scalars(active_statement).first()
                if entry is None:
                    latest_statement = (
                        select(GenerationQueueEntryModel)
                        .order_by(GenerationQueueEntryModel.sequence.desc())
                        .limit(1)
                    )
                    entry = session.scalars(latest_statement).first()
                return _queue_item_from_entry(session, entry) if entry is not None else None
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "latest queue status candidate could not be read"
            ) from exc

    def list_batch_items(self, batch_id: UUID) -> tuple[GenerationQueueItem, ...]:
        return self.list_queue(batch_id=batch_id, limit=500)

    def reconcile_expired_claims(self, *, now: datetime | None = None) -> int:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                entries = session.scalars(
                    select(GenerationQueueEntryModel).where(
                        or_(
                            (
                                GenerationQueueEntryModel.lease_expires_at.is_not(None)
                                & (GenerationQueueEntryModel.lease_expires_at <= timestamp)
                            ),
                            (
                                GenerationQueueEntryModel.cancel_requested_at.is_not(None)
                                & (
                                    GenerationQueueEntryModel.submission_state
                                    == SubmissionState.READY.value
                                )
                            ),
                        )
                    )
                ).all()
                for entry in entries:
                    generation = session.get(GenerationModel, entry.generation_id)
                    job = session.get(GenerationJobModel, entry.job_id)
                    if (
                        entry.cancel_requested_at is not None
                        and entry.submission_state == SubmissionState.READY.value
                        and generation is not None
                        and job is not None
                        and GenerationStatus(generation.status) is GenerationStatus.PENDING
                        and GenerationStatus(job.status) is GenerationStatus.PENDING
                        and generation.comfy_prompt_id is None
                        and job.comfy_prompt_id is None
                    ):
                        generation.status = GenerationStatus.CANCELLED.value
                        generation.completed_at = timestamp
                        generation.updated_at = timestamp
                        job.status = GenerationStatus.CANCELLED.value
                        job.cancelled_at = timestamp
                        job.completed_at = timestamp
                        job.updated_at = timestamp
                    if entry.submission_state == SubmissionState.SUBMITTING.value:
                        entry.submission_state = SubmissionState.AMBIGUOUS.value
                    entry.worker_id = None
                    entry.claimed_at = None
                    entry.lease_expires_at = None
                    entry.updated_at = timestamp
                    if job is not None:
                        job.worker_id = None
                        job.claimed_at = None
                        job.lease_expires_at = None
                        job.updated_at = timestamp
                session.flush()
                return len(entries)
        except SQLAlchemyError as exc:
            raise GenerationDispatchQueueRepositoryError("queue reconciliation failed") from exc

    def reconcile_stateless_restore(self, *, now: datetime | None = None) -> int:
        """Fail unfinished restored work so a new Pod never resends an old prompt."""

        timestamp = _utc(now or datetime.now(UTC))
        active_statuses = (
            GenerationStatus.PENDING.value,
            GenerationStatus.QUEUED.value,
            GenerationStatus.RUNNING.value,
        )
        terminal_statuses = {
            GenerationStatus.COMPLETED.value,
            GenerationStatus.FAILED.value,
            GenerationStatus.CANCELLED.value,
        }
        summary = "Stateless復元後の未完了Jobを再開せず失敗として終了しました。"
        try:
            with session_scope(self._session_factory) as session:
                rows = session.execute(
                    select(GenerationModel, GenerationJobModel, GenerationQueueEntryModel)
                    .join(
                        GenerationJobModel,
                        GenerationJobModel.generation_id == GenerationModel.id,
                    )
                    .outerjoin(
                        GenerationQueueEntryModel,
                        GenerationQueueEntryModel.generation_id == GenerationModel.id,
                    )
                    .where(
                        or_(
                            GenerationModel.status.in_(active_statuses),
                            GenerationJobModel.status.in_(active_statuses),
                            GenerationQueueEntryModel.cancel_requested_at.is_not(None),
                        )
                    )
                ).all()
                reconciled = 0
                for generation, job, entry in rows:
                    changed = False
                    if generation.status not in terminal_statuses:
                        generation.status = GenerationStatus.FAILED.value
                        generation.error_code = "stateless_restore_interrupted"
                        generation.error_summary = summary
                        generation.completed_at = timestamp
                        generation.updated_at = timestamp
                        changed = True
                    if job.status not in terminal_statuses:
                        job.status = GenerationStatus.FAILED.value
                        job.error_code = "stateless_restore_interrupted"
                        job.error_summary = summary
                        job.completed_at = timestamp
                        job.updated_at = timestamp
                        changed = True
                    if (
                        job.worker_id is not None
                        or job.claimed_at is not None
                        or job.lease_expires_at is not None
                    ):
                        job.worker_id = None
                        job.claimed_at = None
                        job.lease_expires_at = None
                        job.updated_at = timestamp
                        changed = True
                    if entry is not None and (
                        entry.worker_id is not None
                        or entry.claimed_at is not None
                        or entry.lease_expires_at is not None
                    ):
                        entry.worker_id = None
                        entry.claimed_at = None
                        entry.lease_expires_at = None
                        entry.updated_at = timestamp
                        changed = True
                    if changed:
                        reconciled += 1
                session.flush()
                return reconciled
        except SQLAlchemyError as exc:
            raise GenerationDispatchQueueRepositoryError(
                "stateless generation reconciliation failed"
            ) from exc

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

    def mark_prompt_id_mismatch(
        self,
        generation_id: UUID,
        *,
        now: datetime | None = None,
    ) -> GenerationQueueItem:
        """Quarantine a pair whose persisted prompt IDs cannot be reconciled safely."""

        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                entry, generation, job = _load_queue_rows(session, generation_id)
                entry.submission_state = SubmissionState.AMBIGUOUS.value
                generation.error_code = "migration_prompt_id_mismatch"
                generation.error_summary = "GenerationとJobのprompt IDが一致しません。"
                generation.updated_at = timestamp
                job.error_code = "migration_prompt_id_mismatch"
                job.error_summary = "GenerationとJobのprompt IDが一致しません。"
                job.updated_at = timestamp
                entry.worker_id = None
                entry.claimed_at = None
                entry.lease_expires_at = None
                entry.updated_at = timestamp
                job.worker_id = None
                job.claimed_at = None
                job.lease_expires_at = None
                job.updated_at = timestamp
                session.flush()
                return _queue_item(session, entry, generation, job)
        except GenerationDispatchQueueRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationDispatchQueueRepositoryError(
                "prompt ID mismatch could not be quarantined"
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


def _load_queue_rows(
    session: Session, generation_id: UUID
) -> tuple[GenerationQueueEntryModel, GenerationModel, GenerationJobModel]:
    entry = session.scalar(
        select(GenerationQueueEntryModel).where(
            GenerationQueueEntryModel.generation_id == str(generation_id)
        )
    )
    generation = session.get(GenerationModel, str(generation_id))
    job = session.get(GenerationJobModel, entry.job_id) if entry is not None else None
    if entry is None or generation is None or job is None:
        raise GenerationDispatchQueueRepositoryError("queue entry was not found")
    if job.generation_id != entry.generation_id:
        raise GenerationDispatchQueueRepositoryError("queue entry is orphaned")
    return entry, generation, job


def _require_ambiguous_prompt_resolution(
    entry: GenerationQueueEntryModel,
    generation: GenerationModel,
    job: GenerationJobModel,
) -> None:
    if entry.submission_state != SubmissionState.AMBIGUOUS.value:
        raise GenerationDispatchQueueRepositoryError("queue entry is not ambiguous")
    generation_status = GenerationStatus(generation.status)
    job_status = GenerationStatus(job.status)
    error_codes = {generation.error_code, job.error_code}
    if not error_codes.intersection(_AMBIGUOUS_PROMPT_RESOLUTION_CODES):
        raise GenerationDispatchQueueRepositoryError(
            "ambiguous queue item lacks a supported resolution audit code"
        )
    if GenerationStatus.COMPLETED in {generation_status, job_status} or (
        GenerationStatus.CANCELLED in {generation_status, job_status}
    ):
        raise GenerationDispatchQueueRepositoryError("terminal queue item cannot be resolved")
    if GenerationStatus.FAILED in {generation_status, job_status} and not error_codes.intersection(
        _MIGRATION_AMBIGUOUS_PROMPT_RECOVERY_CODES
    ):
        raise GenerationDispatchQueueRepositoryError(
            "failed ambiguous queue item can only be resolved through migration recovery"
        )


def _prompt_identifier(value: str) -> str:
    prompt = value.strip()
    if (
        not prompt
        or len(prompt) > 100
        or "/" in prompt
        or "\\" in prompt
        or prompt in {".", ".."}
        or ".." in prompt
    ):
        raise GenerationDispatchQueueRepositoryError("prompt id is invalid")
    return prompt


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
            submission_state=SubmissionState(entry.submission_state),
            submission_token=entry.submission_token,
            submission_started_at=_utc_optional(entry.submission_started_at),
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


def _begin_immediate_if_sqlite(session: Session) -> None:
    bind = session.get_bind()
    if bind.dialect.name == "sqlite":
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")


def _check_pending_capacity(session: Session, limit: int | None, additional: int) -> None:
    if limit is None:
        return
    if limit <= 0 or additional < 1:
        raise GenerationDispatchQueueRepositoryError("queue capacity values are invalid")
    pending_count = int(
        session.scalar(
            select(func.count())
            .select_from(GenerationModel)
            .where(GenerationModel.status == GenerationStatus.PENDING.value)
        )
        or 0
    )
    if pending_count > limit - additional:
        raise GenerationDispatchQueueRepositoryError("queue pending capacity exceeded")


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
