"""SQLite repository for the single active interactive generation run."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import InteractiveGenerationRunModel
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
    GenerationDispatchQueueRepositoryError,
    _begin_immediate_if_sqlite,
)
from runpod_sdxl_image_studio.domain.generation_queue import (
    BatchSeedStrategy,
    GenerationBatch,
    GenerationQueueItem,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import (
    GenerationSettingsSnapshot,
    SnapshotError,
)
from runpod_sdxl_image_studio.domain.interactive_run import (
    InteractiveGenerationRun,
    InteractiveRunStatus,
)


class InteractiveRunRepositoryError(RuntimeError):
    """A safe persistence error for interactive run state."""


class InteractiveRunRepositoryProtocol(Protocol):
    def create_active(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        batch_count: int,
        batch_size: int,
        client_local_date: str,
        run_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> InteractiveGenerationRun: ...

    def create_active_with_batch(
        self,
        snapshots: Sequence[GenerationSettingsSnapshot],
        *,
        batch_count: int,
        batch_size: int,
        client_local_date: str,
        name: str,
        seed_strategy: BatchSeedStrategy,
        start_seed: int | None,
        seed_step: int,
        pending_limit: int | None,
        run_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> tuple[InteractiveGenerationRun, GenerationBatch, tuple[GenerationQueueItem, ...]]: ...

    def get_active(self) -> InteractiveGenerationRun | None: ...

    def get_latest_completed(self) -> InteractiveGenerationRun | None: ...

    def get_by_id(self, run_id: UUID) -> InteractiveGenerationRun | None: ...

    def attach_generations(
        self, run_id: UUID, generation_ids: tuple[UUID, ...]
    ) -> InteractiveGenerationRun: ...

    def update_progress(
        self,
        run_id: UUID,
        *,
        completed_generation_ids: tuple[UUID, ...],
        current_generation_id: UUID | None,
        status: InteractiveRunStatus | None = None,
        error_code: str | None = None,
        error_summary: str | None = None,
        completed_at: datetime | None = None,
    ) -> InteractiveGenerationRun: ...

    def request_cancel(
        self, run_id: UUID, requested_at: datetime | None = None
    ) -> InteractiveGenerationRun: ...


class InteractiveRunRepository(InteractiveRunRepositoryProtocol):
    """Persist run lifecycle transitions in short SQLite transactions."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_active(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        batch_count: int,
        batch_size: int,
        client_local_date: str,
        run_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> InteractiveGenerationRun:
        if not 1 <= batch_count <= 100:
            raise InteractiveRunRepositoryError(
                "interactive batch count is outside the allowed range"
            )
        if not 1 <= batch_size <= 4:
            raise InteractiveRunRepositoryError(
                "interactive batch size is outside the allowed range"
            )
        timestamp = _utc(created_at or datetime.now(UTC))
        identifier = run_id or uuid4()
        try:
            with session_scope(self._session_factory) as session:
                row = InteractiveGenerationRunModel(
                    id=str(identifier),
                    status=InteractiveRunStatus.ACTIVE.value,
                    batch_count=batch_count,
                    batch_size=batch_size,
                    settings_snapshot_json=snapshot.to_json(),
                    snapshot_schema_version=snapshot.schema_version,
                    client_local_date=client_local_date,
                    generation_ids_json="[]",
                    completed_generation_ids_json="[]",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(row)
                session.flush()
                return _to_domain(row)
        except IntegrityError as exc:
            raise InteractiveRunRepositoryError(
                "another interactive run is already active"
            ) from exc
        except (SQLAlchemyError, SnapshotError, ValueError) as exc:
            raise InteractiveRunRepositoryError("interactive run could not be created") from exc

    def create_active_with_batch(
        self,
        snapshots: Sequence[GenerationSettingsSnapshot],
        *,
        batch_count: int,
        batch_size: int,
        client_local_date: str,
        name: str,
        seed_strategy: BatchSeedStrategy,
        start_seed: int | None,
        seed_step: int,
        pending_limit: int | None,
        run_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> tuple[InteractiveGenerationRun, GenerationBatch, tuple[GenerationQueueItem, ...]]:
        """Create an interactive run and its complete queue projection atomically."""

        if len(snapshots) != batch_count:
            raise InteractiveRunRepositoryError(
                "interactive generation count does not match snapshots"
            )
        if not 1 <= batch_count <= 100 or not 1 <= batch_size <= 4:
            raise InteractiveRunRepositoryError("interactive batch values are invalid")
        timestamp = _utc(created_at or datetime.now(UTC))
        identifier = run_id or uuid4()
        queue_repository = GenerationDispatchQueueRepository(self._session_factory)
        try:
            with session_scope(self._session_factory) as session:
                _begin_immediate_if_sqlite(session)
                row = InteractiveGenerationRunModel(
                    id=str(identifier),
                    status=InteractiveRunStatus.ACTIVE.value,
                    batch_count=batch_count,
                    batch_size=batch_size,
                    settings_snapshot_json=snapshots[0].to_json(),
                    snapshot_schema_version=snapshots[0].schema_version,
                    client_local_date=client_local_date,
                    generation_ids_json="[]",
                    completed_generation_ids_json="[]",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(row)
                session.flush()
                batch, items = queue_repository.enqueue_batch_in_session(
                    session,
                    snapshots,
                    name=name,
                    seed_strategy=seed_strategy,
                    start_seed=start_seed,
                    seed_step=seed_step,
                    pending_limit=pending_limit,
                    enqueued_at=timestamp,
                )
                generation_ids = tuple(item.generation.id for item in items)
                row.generation_ids_json = json.dumps(
                    [str(value) for value in generation_ids], separators=(",", ":")
                )
                row.current_generation_id = str(generation_ids[0])
                row.updated_at = timestamp
                session.flush()
                return _to_domain(row), batch, items
        except InteractiveRunRepositoryError:
            raise
        except IntegrityError as exc:
            raise InteractiveRunRepositoryError(
                "another interactive run is already active"
            ) from exc
        except (
            GenerationDispatchQueueRepositoryError,
            SQLAlchemyError,
            SnapshotError,
            ValueError,
        ) as exc:
            raise InteractiveRunRepositoryError(
                "interactive run and batch could not be created"
            ) from exc

    def get_active(self) -> InteractiveGenerationRun | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(InteractiveGenerationRunModel)
                    .where(
                        InteractiveGenerationRunModel.status.in_(
                            [
                                InteractiveRunStatus.ACTIVE.value,
                                InteractiveRunStatus.CANCELLING.value,
                            ]
                        )
                    )
                    .order_by(InteractiveGenerationRunModel.created_at.desc())
                    .limit(1)
                )
                return _to_domain(row) if row is not None else None
        except (SQLAlchemyError, SnapshotError, ValueError) as exc:
            raise InteractiveRunRepositoryError("active interactive run could not be read") from exc

    def get_latest_completed(self) -> InteractiveGenerationRun | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(InteractiveGenerationRunModel)
                    .where(
                        InteractiveGenerationRunModel.status == InteractiveRunStatus.COMPLETED.value
                    )
                    .order_by(
                        InteractiveGenerationRunModel.completed_at.desc(),
                        InteractiveGenerationRunModel.updated_at.desc(),
                    )
                    .limit(1)
                )
                return _to_domain(row) if row is not None else None
        except (SQLAlchemyError, SnapshotError, ValueError) as exc:
            raise InteractiveRunRepositoryError(
                "latest completed interactive run could not be read"
            ) from exc

    def get_by_id(self, run_id: UUID) -> InteractiveGenerationRun | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(InteractiveGenerationRunModel, str(run_id))
                return _to_domain(row) if row is not None else None
        except (SQLAlchemyError, SnapshotError, ValueError) as exc:
            raise InteractiveRunRepositoryError("interactive run could not be read") from exc

    def attach_generations(
        self, run_id: UUID, generation_ids: tuple[UUID, ...]
    ) -> InteractiveGenerationRun:
        try:
            with session_scope(self._session_factory) as session:
                row = _require(session, run_id)
                if len(generation_ids) != row.batch_count:
                    raise InteractiveRunRepositoryError(
                        "interactive generation count does not match"
                    )
                row.generation_ids_json = json.dumps([str(value) for value in generation_ids])
                row.current_generation_id = str(generation_ids[0]) if generation_ids else None
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _to_domain(row)
        except InteractiveRunRepositoryError:
            raise
        except (SQLAlchemyError, SnapshotError, ValueError) as exc:
            raise InteractiveRunRepositoryError(
                "interactive run generations could not be attached"
            ) from exc

    def update_progress(
        self,
        run_id: UUID,
        *,
        completed_generation_ids: tuple[UUID, ...],
        current_generation_id: UUID | None,
        status: InteractiveRunStatus | None = None,
        error_code: str | None = None,
        error_summary: str | None = None,
        completed_at: datetime | None = None,
    ) -> InteractiveGenerationRun:
        try:
            with session_scope(self._session_factory) as session:
                row = _require(session, run_id)
                known_ids = set(_uuid_list(row.generation_ids_json))
                if not set(completed_generation_ids).issubset(known_ids):
                    raise InteractiveRunRepositoryError(
                        "completed generation is not part of the run"
                    )
                if status is not None:
                    row.status = status.value
                row.completed_generation_ids_json = json.dumps(
                    [str(value) for value in completed_generation_ids]
                )
                row.current_generation_id = (
                    str(current_generation_id) if current_generation_id is not None else None
                )
                row.last_completed_generation_id = (
                    str(completed_generation_ids[-1]) if completed_generation_ids else None
                )
                if error_code is not None:
                    row.error_code = error_code
                if error_summary is not None:
                    row.error_summary = error_summary[:1000]
                if completed_at is not None:
                    row.completed_at = _utc(completed_at)
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _to_domain(row)
        except InteractiveRunRepositoryError:
            raise
        except (SQLAlchemyError, SnapshotError, ValueError) as exc:
            raise InteractiveRunRepositoryError(
                "interactive run progress could not be saved"
            ) from exc

    def request_cancel(
        self, run_id: UUID, requested_at: datetime | None = None
    ) -> InteractiveGenerationRun:
        try:
            with session_scope(self._session_factory) as session:
                row = _require(session, run_id)
                if row.status != InteractiveRunStatus.ACTIVE.value:
                    return _to_domain(row)
                row.status = InteractiveRunStatus.CANCELLING.value
                row.cancel_requested_at = _utc(requested_at or datetime.now(UTC))
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _to_domain(row)
        except InteractiveRunRepositoryError:
            raise
        except (SQLAlchemyError, SnapshotError, ValueError) as exc:
            raise InteractiveRunRepositoryError(
                "interactive run cancellation could not be saved"
            ) from exc


def _require(session: Session, run_id: UUID) -> InteractiveGenerationRunModel:
    row = session.get(InteractiveGenerationRunModel, str(run_id))
    if row is None:
        raise InteractiveRunRepositoryError("interactive run was not found")
    return row


def _to_domain(row: InteractiveGenerationRunModel) -> InteractiveGenerationRun:
    return InteractiveGenerationRun(
        id=UUID(row.id),
        status=InteractiveRunStatus(row.status),
        batch_count=row.batch_count,
        batch_size=row.batch_size,
        settings_snapshot=GenerationSettingsSnapshot.from_json(row.settings_snapshot_json),
        client_local_date=row.client_local_date,
        generation_ids=_uuid_list(row.generation_ids_json),
        completed_generation_ids=_uuid_list(row.completed_generation_ids_json),
        current_generation_id=UUID(row.current_generation_id)
        if row.current_generation_id
        else None,
        last_completed_generation_id=(
            UUID(row.last_completed_generation_id) if row.last_completed_generation_id else None
        ),
        cancel_requested_at=_utc(row.cancel_requested_at),
        error_code=row.error_code,
        error_summary=row.error_summary,
        created_at=_utc(row.created_at) or datetime.now(UTC),
        completed_at=_utc(row.completed_at),
        updated_at=_utc(row.updated_at) or datetime.now(UTC),
    )


def _uuid_list(payload: str) -> tuple[UUID, ...]:
    values = json.loads(payload)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("interactive run generation list is invalid")
    return tuple(UUID(value) for value in values)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "InteractiveRunRepository",
    "InteractiveRunRepositoryError",
    "InteractiveRunRepositoryProtocol",
]
