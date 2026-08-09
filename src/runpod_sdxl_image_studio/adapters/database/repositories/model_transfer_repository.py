"""SQLite repository for durable remote model transfer jobs."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import ModelTransferJobModel
from runpod_sdxl_image_studio.domain.model_transfer import (
    ModelTransferErrorCode,
    ModelTransferJob,
    ModelTransferProgress,
    ModelTransferStatus,
    RemoteModelEntry,
    RemoteModelKind,
    normalize_model_relative_path,
)


class ModelTransferRepositoryError(RuntimeError):
    """Safe persistence boundary error."""


class ModelTransferRepositoryProtocol(Protocol):
    def enqueue(
        self,
        entry: RemoteModelEntry,
        local_relative_path: str,
        *,
        now: datetime | None = None,
    ) -> ModelTransferJob: ...

    def claim_next(self, worker_id: str, lease_seconds: float) -> ModelTransferJob | None: ...

    def update_progress(
        self, job_id: UUID, worker_id: str, progress: ModelTransferProgress
    ) -> bool: ...

    def renew_lease(self, job_id: UUID, worker_id: str, lease_seconds: float) -> bool: ...

    def update_process(self, job_id: UUID, worker_id: str, pid: int | None) -> bool: ...

    def mark_completed(
        self, job_id: UUID, worker_id: str, local_sha256: str, *, now: datetime | None = None
    ) -> ModelTransferJob: ...

    def mark_already_prepared(
        self, job_id: UUID, local_sha256: str, *, now: datetime | None = None
    ) -> ModelTransferJob: ...

    def mark_failed(
        self,
        job_id: UUID,
        worker_id: str,
        error_code: str,
        error_summary: str,
        *,
        retryable: bool,
        now: datetime | None = None,
    ) -> ModelTransferJob: ...

    def request_cancel(self, job_id: UUID, *, now: datetime | None = None) -> ModelTransferJob: ...

    def mark_cancelled(
        self, job_id: UUID, worker_id: str, *, now: datetime | None = None
    ) -> ModelTransferJob: ...

    def retry(
        self,
        job_id: UUID,
        entry: RemoteModelEntry,
        local_relative_path: str,
        *,
        now: datetime | None = None,
    ) -> ModelTransferJob: ...

    def get(self, job_id: UUID) -> ModelTransferJob | None: ...

    def list_jobs(self, limit: int = 100) -> tuple[ModelTransferJob, ...]: ...

    def reconcile_stale(self, *, now: datetime | None = None) -> int: ...

    def reconcile_interrupted(self, *, now: datetime | None = None) -> int: ...

    def reconcile_stateless_restore(self, now: datetime | None = None) -> int: ...

    def repair_completed(
        self, job_id: UUID, local_sha256: str, *, now: datetime | None = None
    ) -> ModelTransferJob: ...


class ModelTransferRepository(ModelTransferRepositoryProtocol):
    """Persist model transfer state with terminal-state and active-job guards."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def enqueue(
        self,
        entry: RemoteModelEntry,
        local_relative_path: str,
        *,
        now: datetime | None = None,
    ) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        local_path = normalize_model_relative_path(local_relative_path)
        identity = _remote_identity(entry)
        try:
            with session_scope(self._session_factory) as session:
                existing = _active_row(session, entry.kind, entry.relative_path, identity)
                if existing is not None:
                    return existing.to_domain()
                row = ModelTransferJobModel(
                    id=str(uuid4()),
                    kind=entry.kind.value,
                    remote_relative_path=entry.relative_path,
                    local_relative_path=local_path,
                    remote_size_bytes=entry.size_bytes,
                    remote_hash_algorithm=entry.remote_hash_algorithm,
                    remote_hash=entry.remote_hash,
                    remote_modified_at=entry.modified_at,
                    remote_identity=identity,
                    status=ModelTransferStatus.PENDING.value,
                    progress_bytes=0,
                    total_bytes=entry.size_bytes,
                    progress_percentage=0.0,
                    retryable=True,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(row)
                session.flush()
                return row.to_domain()
        except IntegrityError:
            # A second UI request may race the partial unique active index.
            try:
                with session_scope(self._session_factory) as session:
                    existing = _active_row(session, entry.kind, entry.relative_path, identity)
                    if existing is not None:
                        return existing.to_domain()
            except Exception as exc:  # noqa: BLE001 - hide persistence internals
                raise ModelTransferRepositoryError(
                    "model transfer enqueue could not be read"
                ) from exc
            raise ModelTransferRepositoryError("model transfer enqueue conflicted") from None
        except Exception as exc:  # noqa: BLE001 - safe repository boundary
            raise ModelTransferRepositoryError("model transfer enqueue could not be saved") from exc

    def claim_next(self, worker_id: str, lease_seconds: float) -> ModelTransferJob | None:
        timestamp = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(ModelTransferJobModel)
                    .where(ModelTransferJobModel.status == ModelTransferStatus.PENDING.value)
                    .order_by(
                        ModelTransferJobModel.created_at.asc(), ModelTransferJobModel.id.asc()
                    )
                    .limit(1)
                )
                if row is None:
                    return None
                row.status = ModelTransferStatus.DOWNLOADING.value
                row.worker_id = worker_id
                row.claimed_at = timestamp
                row.lease_expires_at = timestamp + timedelta(seconds=lease_seconds)
                row.started_at = row.started_at or timestamp
                row.updated_at = timestamp
                session.flush()
                return row.to_domain()
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError("model transfer job could not be claimed") from exc

    def update_progress(
        self, job_id: UUID, worker_id: str, progress: ModelTransferProgress
    ) -> bool:
        timestamp = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if not _owned_active(row, worker_id):
                    return False
                assert row is not None
                row.progress_bytes = progress.progress_bytes
                row.total_bytes = progress.total_bytes
                row.progress_percentage = progress.progress_percentage
                row.updated_at = timestamp
                return True
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "model transfer progress could not be saved"
            ) from exc

    def renew_lease(self, job_id: UUID, worker_id: str, lease_seconds: float) -> bool:
        timestamp = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if not _owned_active(row, worker_id):
                    return False
                assert row is not None
                row.lease_expires_at = timestamp + timedelta(seconds=lease_seconds)
                row.updated_at = timestamp
                return True
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError("model transfer lease could not be renewed") from exc

    def update_process(self, job_id: UUID, worker_id: str, pid: int | None) -> bool:
        timestamp = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if not _owned_active(row, worker_id):
                    return False
                assert row is not None
                row.pid = pid
                row.updated_at = timestamp
                return True
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "model transfer process state could not be saved"
            ) from exc

    def mark_completed(
        self, job_id: UUID, worker_id: str, local_sha256: str, *, now: datetime | None = None
    ) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if row is None:
                    raise ModelTransferRepositoryError("model transfer job was not found")
                if row.status == ModelTransferStatus.COMPLETED.value:
                    return row.to_domain()
                if not _owned_active(row, worker_id):
                    raise ModelTransferRepositoryError(
                        "model transfer job lease is no longer owned"
                    )
                row.status = ModelTransferStatus.COMPLETED.value
                row.local_sha256 = local_sha256
                row.progress_bytes = row.total_bytes
                row.progress_percentage = 100.0
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.completed_at = timestamp
                row.error_code = None
                row.error_summary = None
                row.updated_at = timestamp
                return row.to_domain()
        except ModelTransferRepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "model transfer completion could not be saved"
            ) from exc

    def mark_already_prepared(
        self, job_id: UUID, local_sha256: str, *, now: datetime | None = None
    ) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if row is None:
                    raise ModelTransferRepositoryError("model transfer job was not found")
                if row.status == ModelTransferStatus.COMPLETED.value:
                    return row.to_domain()
                if row.status != ModelTransferStatus.PENDING.value:
                    raise ModelTransferRepositoryError("model transfer job is no longer pending")
                row.status = ModelTransferStatus.COMPLETED.value
                row.local_sha256 = local_sha256
                row.progress_bytes = row.total_bytes
                row.progress_percentage = 100.0
                row.completed_at = timestamp
                row.error_code = "already_prepared"
                row.error_summary = "Matching local model was already prepared"
                row.retryable = False
                row.updated_at = timestamp
                return row.to_domain()
        except ModelTransferRepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError("existing model state could not be saved") from exc

    def mark_failed(
        self,
        job_id: UUID,
        worker_id: str,
        error_code: str,
        error_summary: str,
        *,
        retryable: bool,
        now: datetime | None = None,
    ) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if row is None:
                    raise ModelTransferRepositoryError("model transfer job was not found")
                if row.status in {
                    ModelTransferStatus.COMPLETED.value,
                    ModelTransferStatus.FAILED.value,
                    ModelTransferStatus.CANCELLED.value,
                }:
                    return row.to_domain()
                if row.worker_id != worker_id:
                    raise ModelTransferRepositoryError(
                        "model transfer job lease is no longer owned"
                    )
                row.status = ModelTransferStatus.FAILED.value
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.completed_at = timestamp
                row.error_code = error_code[:64]
                row.error_summary = _safe_summary(error_summary)
                row.retryable = retryable
                row.updated_at = timestamp
                return row.to_domain()
        except ModelTransferRepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError("model transfer failure could not be saved") from exc

    def request_cancel(self, job_id: UUID, *, now: datetime | None = None) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if row is None:
                    raise ModelTransferRepositoryError("model transfer job was not found")
                if row.status in {
                    ModelTransferStatus.COMPLETED.value,
                    ModelTransferStatus.FAILED.value,
                    ModelTransferStatus.CANCELLED.value,
                }:
                    return row.to_domain()
                row.updated_at = timestamp
                if row.status == ModelTransferStatus.PENDING.value:
                    row.status = ModelTransferStatus.CANCELLED.value
                    row.cancelled_at = timestamp
                    row.completed_at = timestamp
                    row.error_code = ModelTransferErrorCode.CANCELLED.value
                    row.error_summary = "Model transfer was cancelled before it started"
                    row.retryable = True
                else:
                    row.status = ModelTransferStatus.CANCEL_REQUESTED.value
                return row.to_domain()
        except ModelTransferRepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "model transfer cancellation could not be saved"
            ) from exc

    def mark_cancelled(
        self, job_id: UUID, worker_id: str, *, now: datetime | None = None
    ) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if row is None:
                    raise ModelTransferRepositoryError("model transfer job was not found")
                if row.status == ModelTransferStatus.CANCELLED.value:
                    return row.to_domain()
                if row.worker_id != worker_id or row.status not in {
                    ModelTransferStatus.DOWNLOADING.value,
                    ModelTransferStatus.CANCEL_REQUESTED.value,
                }:
                    raise ModelTransferRepositoryError(
                        "model transfer cancellation lease is invalid"
                    )
                row.status = ModelTransferStatus.CANCELLED.value
                row.cancelled_at = timestamp
                row.completed_at = timestamp
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.error_code = ModelTransferErrorCode.CANCELLED.value
                row.error_summary = "Model transfer was cancelled"
                row.updated_at = timestamp
                return row.to_domain()
        except ModelTransferRepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "model transfer cancellation could not be finalized"
            ) from exc

    def retry(
        self,
        job_id: UUID,
        entry: RemoteModelEntry,
        local_relative_path: str,
        *,
        now: datetime | None = None,
    ) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        local_path = normalize_model_relative_path(local_relative_path)
        identity = _remote_identity(entry)
        try:
            with session_scope(self._session_factory) as session:
                source = session.get(ModelTransferJobModel, str(job_id))
                if source is None:
                    raise ModelTransferRepositoryError("model transfer job was not found")
                if source.status not in {
                    ModelTransferStatus.FAILED.value,
                    ModelTransferStatus.CANCELLED.value,
                }:
                    raise ModelTransferRepositoryError(
                        "only failed or cancelled jobs can be retried"
                    )
                existing = _active_row(session, entry.kind, entry.relative_path, identity)
                if existing is not None:
                    return existing.to_domain()
                row = ModelTransferJobModel(
                    id=str(uuid4()),
                    kind=entry.kind.value,
                    remote_relative_path=entry.relative_path,
                    local_relative_path=local_path,
                    remote_size_bytes=entry.size_bytes,
                    remote_hash_algorithm=entry.remote_hash_algorithm,
                    remote_hash=entry.remote_hash,
                    remote_modified_at=entry.modified_at,
                    remote_identity=identity,
                    status=ModelTransferStatus.PENDING.value,
                    progress_bytes=0,
                    total_bytes=entry.size_bytes,
                    progress_percentage=0.0,
                    retryable=True,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(row)
                session.flush()
                return row.to_domain()
        except ModelTransferRepositoryError:
            raise
        except IntegrityError as exc:
            raise ModelTransferRepositoryError("model transfer retry conflicted") from exc
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError("model transfer retry could not be saved") from exc

    def get(self, job_id: UUID) -> ModelTransferJob | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                return row.to_domain() if row is not None else None
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError("model transfer job could not be read") from exc

    def list_jobs(self, limit: int = 100) -> tuple[ModelTransferJob, ...]:
        safe_limit = min(max(limit, 1), 500)
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(ModelTransferJobModel)
                    .order_by(
                        ModelTransferJobModel.created_at.desc(), ModelTransferJobModel.id.desc()
                    )
                    .limit(safe_limit)
                ).all()
                return tuple(row.to_domain() for row in rows)
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError("model transfer jobs could not be read") from exc

    def reconcile_stale(self, *, now: datetime | None = None) -> int:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(ModelTransferJobModel).where(
                        ModelTransferJobModel.status == ModelTransferStatus.DOWNLOADING.value,
                        ModelTransferJobModel.lease_expires_at.is_not(None),
                        ModelTransferJobModel.lease_expires_at <= timestamp,
                    )
                ).all()
                for row in rows:
                    _terminalize_failure(
                        row,
                        ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value,
                        "The previous model transfer lease expired",
                        timestamp,
                    )
                return len(rows)
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "stale model transfers could not be reconciled"
            ) from exc

    def reconcile_interrupted(self, *, now: datetime | None = None) -> int:
        """Fail jobs owned by a previous app process before this worker claims work."""

        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(ModelTransferJobModel).where(
                        ModelTransferJobModel.status.in_(
                            [
                                ModelTransferStatus.DOWNLOADING.value,
                                ModelTransferStatus.CANCEL_REQUESTED.value,
                            ]
                        )
                    )
                ).all()
                for row in rows:
                    _terminalize_failure(
                        row,
                        ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value,
                        "Model transfer was interrupted by an application restart",
                        timestamp,
                    )
                return len(rows)
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "interrupted model transfers could not be reconciled"
            ) from exc

    def reconcile_stateless_restore(self, now: datetime | None = None) -> int:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(ModelTransferJobModel).where(
                        ModelTransferJobModel.status.in_(
                            [
                                ModelTransferStatus.PENDING.value,
                                ModelTransferStatus.DOWNLOADING.value,
                                ModelTransferStatus.CANCEL_REQUESTED.value,
                            ]
                        )
                    )
                ).all()
                for row in rows:
                    _terminalize_failure(
                        row,
                        ModelTransferErrorCode.STATELESS_RESTORE_INTERRUPTED.value,
                        "Model transfer was interrupted during stateless restore",
                        timestamp,
                    )
                return len(rows)
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "stateless model transfer reconciliation failed"
            ) from exc

    def repair_completed(
        self, job_id: UUID, local_sha256: str, *, now: datetime | None = None
    ) -> ModelTransferJob:
        timestamp = _utc(now or datetime.now(UTC))
        recoverable_codes = {
            ModelTransferErrorCode.PERSISTENCE_FAILED.value,
            ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value,
        }
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ModelTransferJobModel, str(job_id))
                if row is None:
                    raise ModelTransferRepositoryError("model transfer job was not found")
                if row.status == ModelTransferStatus.COMPLETED.value:
                    return row.to_domain()
                recoverable = row.error_code in recoverable_codes or (
                    row.status == ModelTransferStatus.DOWNLOADING.value and row.error_code is None
                )
                if (
                    row.status
                    not in {
                        ModelTransferStatus.DOWNLOADING.value,
                        ModelTransferStatus.FAILED.value,
                    }
                    or not recoverable
                ):
                    raise ModelTransferRepositoryError("model transfer job is not recoverable")
                row.status = ModelTransferStatus.COMPLETED.value
                row.local_sha256 = local_sha256
                row.progress_bytes = row.total_bytes
                row.progress_percentage = 100.0
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.completed_at = timestamp
                row.error_code = None
                row.error_summary = None
                row.updated_at = timestamp
                return row.to_domain()
        except ModelTransferRepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTransferRepositoryError(
                "model transfer recovery could not be saved"
            ) from exc


def _active_row(
    session: Session,
    kind: RemoteModelKind,
    relative_path: str,
    identity: str,
) -> ModelTransferJobModel | None:
    return session.scalar(
        select(ModelTransferJobModel).where(
            ModelTransferJobModel.kind == kind.value,
            ModelTransferJobModel.remote_relative_path == relative_path,
            ModelTransferJobModel.remote_identity == identity,
            ModelTransferJobModel.status.in_(
                [
                    ModelTransferStatus.PENDING.value,
                    ModelTransferStatus.DOWNLOADING.value,
                    ModelTransferStatus.CANCEL_REQUESTED.value,
                ]
            ),
        )
    )


def _remote_identity(entry: RemoteModelEntry) -> str:
    return hashlib.sha256(entry.identity.encode("utf-8")).hexdigest()


def _owned_active(row: ModelTransferJobModel | None, worker_id: str) -> bool:
    return (
        row is not None
        and row.worker_id == worker_id
        and row.status
        in {
            ModelTransferStatus.DOWNLOADING.value,
            ModelTransferStatus.CANCEL_REQUESTED.value,
        }
    )


def _terminalize_failure(
    row: ModelTransferJobModel,
    error_code: str,
    error_summary: str,
    timestamp: datetime,
) -> None:
    row.status = ModelTransferStatus.FAILED.value
    row.worker_id = None
    row.pid = None
    row.claimed_at = None
    row.lease_expires_at = None
    row.completed_at = timestamp
    row.error_code = error_code[:64]
    row.error_summary = _safe_summary(error_summary)
    row.retryable = True
    row.updated_at = timestamp


def _safe_summary(value: str) -> str:
    return " ".join(value.split())[:1000]


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "ModelTransferRepository",
    "ModelTransferRepositoryError",
    "ModelTransferRepositoryProtocol",
]
