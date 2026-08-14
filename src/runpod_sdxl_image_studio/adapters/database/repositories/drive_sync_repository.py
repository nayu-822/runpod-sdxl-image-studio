"""SQLite repository for independent Google Drive synchronization state."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import (
    DriveManifestJobModel,
    DriveSyncJobModel,
    DriveSyncRecordModel,
    GenerationArtifactModel,
    GenerationModel,
)
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveCacheCandidate,
    DriveCapacity,
    DriveDestination,
    DriveManifestJob,
    DriveManifestState,
    DriveSyncArtifact,
    DriveSyncErrorCode,
    DriveSyncJob,
    DriveSyncProgress,
    DriveSyncRecord,
    DriveSyncStatus,
    utc,
)
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType


class DriveSyncRepositoryError(RuntimeError):
    """Safe persistence error for Drive synchronization records and jobs."""


class DriveManifestRebuildRequired(DriveSyncRepositoryError):
    """The old destination manifest is not settled enough for an explicit resync."""


@dataclass(frozen=True)
class DriveSyncDiscoveryCandidate:
    generation_id: UUID
    kind: str
    created_at: datetime


@dataclass(frozen=True)
class DriveManifestRecord:
    generation_id: UUID
    kind: str
    created_at: datetime
    remote_image_path: str
    remote_metadata_path: str
    image_sha256: str
    metadata_sha256: str
    image_size_bytes: int
    metadata_size_bytes: int
    synced_at: datetime
    remote_name: str
    remote_base_path: str
    artifacts: tuple[DriveSyncArtifact, ...] = ()


@dataclass(frozen=True)
class DriveManifestFailureTarget:
    local_date: str
    remote_name: str
    remote_base_path: str


class DriveSyncRepositoryProtocol(Protocol):
    def get_by_generation(self, generation_id: UUID) -> DriveSyncRecord | None: ...

    def get_job(self, job_id: UUID) -> DriveSyncJob | None: ...

    def get_manifest_job(self, job_id: UUID) -> DriveManifestJob | None: ...

    def enqueue(
        self, record: DriveSyncRecord, job: DriveSyncJob | None
    ) -> tuple[DriveSyncRecord, DriveSyncJob | None]: ...

    def retry(
        self,
        record: DriveSyncRecord,
        job: DriveSyncJob,
        *,
        require_manifest_ready: bool = False,
    ) -> tuple[DriveSyncRecord, DriveSyncJob]: ...

    def manifest_state_for_destination(
        self, local_date: str, destination: DriveDestination
    ) -> DriveManifestState: ...

    def claim_next(self, worker_id: str, lease_seconds: float) -> DriveSyncJob | None: ...

    def renew_lease(self, job_id: UUID, worker_id: str, lease_seconds: float) -> bool: ...

    def update_progress(
        self, job_id: UUID, worker_id: str, progress: DriveSyncProgress
    ) -> bool: ...

    def mark_process_started(self, job_id: UUID, worker_id: str, pid: int) -> bool: ...

    def mark_process_finished(self, job_id: UUID, worker_id: str) -> bool: ...

    def mark_synced(self, job_id: UUID, worker_id: str, synced_at: datetime) -> DriveSyncRecord: ...

    def mark_artifact_synced(
        self,
        job_id: UUID,
        worker_id: str,
        display_order: int,
        artifact_kind: str,
    ) -> bool: ...

    def mark_failed(
        self,
        job_id: UUID,
        worker_id: str | None,
        error_code: str,
        error_summary: str,
        retryable: bool = True,
    ) -> DriveSyncRecord: ...

    def mark_manifest_warning(self, record_id: UUID, summary: str) -> None: ...

    def enqueue_manifest(self, job: DriveManifestJob) -> DriveManifestJob: ...

    def claim_next_manifest(
        self, worker_id: str, lease_seconds: float
    ) -> DriveManifestJob | None: ...

    def renew_manifest_lease(self, job_id: UUID, worker_id: str, lease_seconds: float) -> bool: ...

    def update_manifest_progress(
        self, job_id: UUID, worker_id: str, progress: DriveSyncProgress
    ) -> bool: ...

    def mark_manifest_process_started(self, job_id: UUID, worker_id: str, pid: int) -> bool: ...

    def mark_manifest_process_finished(self, job_id: UUID, worker_id: str) -> bool: ...

    def mark_manifest_synced(
        self, job_id: UUID, worker_id: str, synced_at: datetime
    ) -> DriveManifestJob: ...

    def mark_manifest_failed(
        self,
        job_id: UUID,
        worker_id: str | None,
        error_code: str,
        error_summary: str,
        retryable: bool = True,
    ) -> DriveManifestJob: ...

    def clear_manifest_warning(self, local_date: str, destination: DriveDestination) -> None: ...

    def mark_manifest_warning_for_destination(
        self, local_date: str, destination: DriveDestination, summary: str
    ) -> None: ...

    def list_manifest_jobs(self, limit: int = 50) -> tuple[DriveManifestJob, ...]: ...

    def has_active_manifest_jobs(self) -> bool: ...

    def list_manifest_failure_targets(
        self, limit: int = 100
    ) -> tuple[DriveManifestFailureTarget, ...]: ...

    def reconcile_stale(self, now: datetime | None = None) -> int: ...

    def reconcile_stateless_restore(self, now: datetime | None = None) -> int: ...

    def list_jobs(self, limit: int = 50) -> tuple[DriveSyncJob, ...]: ...

    def get_latest_synced_job(self) -> DriveSyncJob | None: ...

    def get_latest_unresolved_failure(self) -> DriveSyncJob | None: ...

    def status_counts(self) -> dict[DriveSyncStatus, int]: ...

    def list_discovery_candidates(self, limit: int) -> tuple[DriveSyncDiscoveryCandidate, ...]: ...

    def list_manifest_records(
        self, local_date: str, remote_name: str, remote_base_path: str
    ) -> tuple[DriveManifestRecord, ...]: ...

    def capacity(self, *, total_bytes: int, used_bytes: int, free_bytes: int) -> DriveCapacity: ...

    def cache_candidates(self, limit: int = 100) -> tuple[DriveCacheCandidate, ...]: ...


class DriveSyncRepository(DriveSyncRepositoryProtocol):
    """Persist a SyncRecord and its active Job atomically where required."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_by_generation(self, generation_id: UUID) -> DriveSyncRecord | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(DriveSyncRecordModel).where(
                        DriveSyncRecordModel.generation_id == str(generation_id)
                    )
                )
                return _record_domain(row) if row is not None else None
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync record could not be read") from exc

    def get_job(self, job_id: UUID) -> DriveSyncJob | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncJobModel, str(job_id))
                return _job_domain(row) if row is not None else None
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync job could not be read") from exc

    def enqueue(
        self, record: DriveSyncRecord, job: DriveSyncJob | None
    ) -> tuple[DriveSyncRecord, DriveSyncJob | None]:
        """Insert a record and optional Job once, returning existing active state."""

        try:
            with session_scope(self._session_factory) as session:
                existing = session.scalar(
                    select(DriveSyncRecordModel).where(
                        DriveSyncRecordModel.generation_id == str(record.generation_id)
                    )
                )
                if existing is None:
                    existing = _record_model(record)
                    session.add(existing)
                    session.flush()
                current_record = _record_domain(existing)
                active = session.scalar(
                    select(DriveSyncJobModel)
                    .where(
                        DriveSyncJobModel.sync_record_id == existing.id,
                        DriveSyncJobModel.status.in_(
                            [DriveSyncStatus.PENDING.value, DriveSyncStatus.SYNCING.value]
                        ),
                    )
                    .order_by(DriveSyncJobModel.queue_sequence.asc())
                )
                if (
                    active is not None
                    or job is None
                    or current_record.status is DriveSyncStatus.SYNCED
                ):
                    return current_record, _job_domain(active) if active is not None else None
                new_job = _job_model(job, str(existing.id), session)
                session.add(new_job)
                session.flush()
                return current_record, _job_domain(new_job)
        except DriveSyncRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync enqueue could not be saved") from exc

    def retry(
        self,
        record: DriveSyncRecord,
        job: DriveSyncJob,
        *,
        require_manifest_ready: bool = False,
    ) -> tuple[DriveSyncRecord, DriveSyncJob]:
        """Reset one explicit retry/resync and create a new pending Job atomically."""

        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(DriveSyncRecordModel).where(
                        DriveSyncRecordModel.generation_id == str(record.generation_id)
                    )
                )
                if row is None:
                    raise DriveSyncRepositoryError("drive sync record was not found")
                if require_manifest_ready:
                    _require_manifest_ready(session, row)
                active = session.scalar(
                    select(DriveSyncJobModel).where(
                        DriveSyncJobModel.sync_record_id == row.id,
                        DriveSyncJobModel.status.in_(
                            [DriveSyncStatus.PENDING.value, DriveSyncStatus.SYNCING.value]
                        ),
                    )
                )
                if active is not None:
                    return _record_domain(row), _job_domain(active)
                _copy_record_values(row, record, status=DriveSyncStatus.PENDING)
                row.error_code = None
                row.error_summary = None
                row.synced_at = None
                new_job = _job_model(job, str(row.id), session)
                session.add(new_job)
                session.flush()
                return _record_domain(row), _job_domain(new_job)
        except DriveSyncRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync retry could not be saved") from exc

    def claim_next(self, worker_id: str, lease_seconds: float) -> DriveSyncJob | None:
        now = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(DriveSyncJobModel)
                    .where(DriveSyncJobModel.status == DriveSyncStatus.PENDING.value)
                    .order_by(DriveSyncJobModel.queue_sequence.asc())
                    .limit(1)
                )
                if row is None:
                    return None
                record = session.get(DriveSyncRecordModel, row.sync_record_id)
                if record is None:
                    row.status = DriveSyncStatus.FAILED.value
                    row.error_code = "drive_persistence_failed"
                    row.error_summary = "sync record was not found"
                    row.retryable = True
                    row.completed_at = now
                    row.updated_at = now
                    return None
                row.status = DriveSyncStatus.SYNCING.value
                row.worker_id = worker_id
                row.claimed_at = now
                row.lease_expires_at = now + timedelta(seconds=lease_seconds)
                row.started_at = row.started_at or now
                row.error_code = None
                row.error_summary = None
                row.updated_at = now
                record.status = DriveSyncStatus.SYNCING.value
                record.attempt_count += 1
                record.last_attempt_at = now
                record.error_code = None
                record.error_summary = None
                record.updated_at = now
                session.flush()
                return _job_domain(row)
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive sync job could not be claimed") from exc

    def renew_lease(self, job_id: UUID, worker_id: str, lease_seconds: float) -> bool:
        now = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncJobModel, str(job_id))
                if (
                    row is None
                    or row.status != DriveSyncStatus.SYNCING.value
                    or row.worker_id != worker_id
                ):
                    return False
                row.lease_expires_at = now + timedelta(seconds=lease_seconds)
                row.updated_at = now
                return True
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive sync lease could not be renewed") from exc

    def update_progress(self, job_id: UUID, worker_id: str, progress: DriveSyncProgress) -> bool:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncJobModel, str(job_id))
                if (
                    row is None
                    or row.status != DriveSyncStatus.SYNCING.value
                    or row.worker_id != worker_id
                ):
                    return False
                row.progress_bytes = progress.progress_bytes
                row.total_bytes = progress.total_bytes
                row.progress_percentage = progress.progress_percentage
                row.current_artifact = progress.current_artifact
                row.updated_at = datetime.now(UTC)
                return True
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync progress could not be saved") from exc

    def mark_artifact_synced(
        self,
        job_id: UUID,
        worker_id: str,
        display_order: int,
        artifact_kind: str,
    ) -> bool:
        """Persist one completed image/metadata transfer without closing the job."""

        if artifact_kind not in {"image", "metadata"} or display_order < 0:
            raise DriveSyncRepositoryError("drive sync artifact progress is invalid")
        try:
            with session_scope(self._session_factory) as session:
                job = session.get(DriveSyncJobModel, str(job_id))
                if (
                    job is None
                    or job.status != DriveSyncStatus.SYNCING.value
                    or job.worker_id != worker_id
                ):
                    return False
                record = session.get(DriveSyncRecordModel, job.sync_record_id)
                if record is None:
                    raise DriveSyncRepositoryError("drive sync record was not found")
                job_items = _sync_artifacts_from_row(job, status=job.status)
                record_items = _sync_artifacts_from_row(record, status=record.status)
                job_items = _set_artifact_progress(job_items, display_order, artifact_kind)
                record_items = _set_artifact_progress(record_items, display_order, artifact_kind)
                job.artifacts_json = _artifacts_json(job_items)
                record.artifacts_json = _artifacts_json(record_items)
                now = datetime.now(UTC)
                job.updated_at = now
                record.updated_at = now
                session.flush()
                return True
        except DriveSyncRepositoryError:
            raise
        except (SQLAlchemyError, ValueError, TypeError) as exc:
            raise DriveSyncRepositoryError(
                "drive sync artifact progress could not be saved"
            ) from exc

    def mark_process_started(self, job_id: UUID, worker_id: str, pid: int) -> bool:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncJobModel, str(job_id))
                if (
                    row is None
                    or row.status != DriveSyncStatus.SYNCING.value
                    or row.worker_id != worker_id
                ):
                    return False
                row.pid = pid
                row.updated_at = datetime.now(UTC)
                return True
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive sync process start could not be saved") from exc

    def mark_process_finished(self, job_id: UUID, worker_id: str) -> bool:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncJobModel, str(job_id))
                if row is None or row.worker_id != worker_id:
                    return False
                row.pid = None
                row.updated_at = datetime.now(UTC)
                return True
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive sync process finish could not be saved") from exc

    def mark_synced(self, job_id: UUID, worker_id: str, synced_at: datetime) -> DriveSyncRecord:
        timestamp = utc(synced_at)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncJobModel, str(job_id))
                if row is None or row.worker_id != worker_id:
                    raise DriveSyncRepositoryError("drive sync job lease is no longer owned")
                record = session.get(DriveSyncRecordModel, row.sync_record_id)
                if record is None:
                    raise DriveSyncRepositoryError("drive sync record was not found")
                if row.status is not None and row.status != DriveSyncStatus.SYNCING.value:
                    if record.status == DriveSyncStatus.SYNCED.value:
                        return _record_domain(record)
                    raise DriveSyncRepositoryError("drive sync job is no longer active")
                items = _sync_artifacts_from_row(row, status=row.status)
                if not items or not all(
                    item.image_synced and item.metadata_synced for item in items
                ):
                    raise DriveSyncRepositoryError(
                        "all Drive image and metadata artifacts must be synced first"
                    )
                row.status = DriveSyncStatus.SYNCED.value
                row.progress_bytes = row.total_bytes
                row.progress_percentage = 100.0
                row.current_artifact = None
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.completed_at = timestamp
                row.error_code = None
                row.error_summary = None
                row.retryable = False
                row.artifacts_json = _artifacts_json(items)
                row.updated_at = timestamp
                record.status = DriveSyncStatus.SYNCED.value
                record.synced_at = timestamp
                record.error_code = None
                record.error_summary = None
                record.artifacts_json = _artifacts_json(items)
                record.updated_at = timestamp
                session.flush()
                return _record_domain(record)
        except DriveSyncRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync success could not be saved") from exc

    def mark_failed(
        self,
        job_id: UUID,
        worker_id: str | None,
        error_code: str,
        error_summary: str,
        retryable: bool = True,
    ) -> DriveSyncRecord:
        now = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncJobModel, str(job_id))
                if row is None:
                    raise DriveSyncRepositoryError("drive sync job was not found")
                record = session.get(DriveSyncRecordModel, row.sync_record_id)
                if record is None:
                    raise DriveSyncRepositoryError("drive sync record was not found")
                if (
                    record.status == DriveSyncStatus.SYNCED.value
                    or row.status == DriveSyncStatus.SYNCED.value
                ):
                    return _record_domain(record)
                if worker_id is not None and row.worker_id != worker_id:
                    raise DriveSyncRepositoryError("drive sync job lease is no longer owned")
                row.status = DriveSyncStatus.FAILED.value
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.completed_at = now
                row.error_code = error_code
                row.error_summary = error_summary[:1000]
                row.retryable = retryable
                row.updated_at = now
                record.status = DriveSyncStatus.FAILED.value
                record.error_code = error_code
                record.error_summary = error_summary[:1000]
                record.updated_at = now
                session.flush()
                return _record_domain(record)
        except DriveSyncRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync failure could not be saved") from exc

    def mark_manifest_warning(self, record_id: UUID, summary: str) -> None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveSyncRecordModel, str(record_id))
                if row is None:
                    raise DriveSyncRepositoryError("drive sync record was not found")
                if row.status == DriveSyncStatus.SYNCED.value:
                    row.error_code = "drive_manifest_failed"
                    row.error_summary = summary[:1000]
                    row.updated_at = datetime.now(UTC)
        except DriveSyncRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive manifest warning could not be saved") from exc

    def enqueue_manifest(self, job: DriveManifestJob) -> DriveManifestJob:
        try:
            with session_scope(self._session_factory) as session:
                active = session.scalar(
                    select(DriveManifestJobModel)
                    .where(
                        DriveManifestJobModel.local_date == job.local_date,
                        DriveManifestJobModel.remote_name == job.remote_name,
                        DriveManifestJobModel.remote_base_path == job.remote_base_path,
                        DriveManifestJobModel.status.in_(
                            [DriveSyncStatus.PENDING.value, DriveSyncStatus.SYNCING.value]
                        ),
                    )
                    .order_by(DriveManifestJobModel.queue_sequence.asc())
                )
                if active is not None:
                    return _manifest_job_domain(active)
                row = _manifest_job_model(job, session)
                session.add(row)
                session.flush()
                return _manifest_job_domain(row)
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest job could not be queued") from exc

    def manifest_state_for_destination(
        self, local_date: str, destination: DriveDestination
    ) -> DriveManifestState:
        try:
            date.fromisoformat(local_date)
            with session_scope(self._session_factory) as session:
                return _manifest_state_in_session(session, local_date, destination)
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest state could not be read") from exc

    def get_manifest_job(self, job_id: UUID) -> DriveManifestJob | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveManifestJobModel, str(job_id))
                return _manifest_job_domain(row) if row is not None else None
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest job could not be read") from exc

    def claim_next_manifest(self, worker_id: str, lease_seconds: float) -> DriveManifestJob | None:
        now = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(DriveManifestJobModel)
                    .where(DriveManifestJobModel.status == DriveSyncStatus.PENDING.value)
                    .order_by(DriveManifestJobModel.queue_sequence.asc())
                    .limit(1)
                )
                if row is None:
                    return None
                row.status = DriveSyncStatus.SYNCING.value
                row.worker_id = worker_id
                row.claimed_at = now
                row.lease_expires_at = now + timedelta(seconds=lease_seconds)
                row.started_at = row.started_at or now
                row.error_code = None
                row.error_summary = None
                row.updated_at = now
                session.flush()
                return _manifest_job_domain(row)
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive manifest job could not be claimed") from exc

    def renew_manifest_lease(self, job_id: UUID, worker_id: str, lease_seconds: float) -> bool:
        now = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveManifestJobModel, str(job_id))
                if (
                    row is None
                    or row.status != DriveSyncStatus.SYNCING.value
                    or row.worker_id != worker_id
                ):
                    return False
                row.lease_expires_at = now + timedelta(seconds=lease_seconds)
                row.updated_at = now
                return True
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive manifest lease could not be renewed") from exc

    def update_manifest_progress(
        self, job_id: UUID, worker_id: str, progress: DriveSyncProgress
    ) -> bool:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveManifestJobModel, str(job_id))
                if (
                    row is None
                    or row.status != DriveSyncStatus.SYNCING.value
                    or row.worker_id != worker_id
                ):
                    return False
                row.progress_bytes = progress.progress_bytes
                row.total_bytes = progress.total_bytes
                row.progress_percentage = progress.progress_percentage
                row.current_artifact = progress.current_artifact
                row.updated_at = datetime.now(UTC)
                return True
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest progress could not be saved") from exc

    def mark_manifest_process_started(self, job_id: UUID, worker_id: str, pid: int) -> bool:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveManifestJobModel, str(job_id))
                if (
                    row is None
                    or row.status != DriveSyncStatus.SYNCING.value
                    or row.worker_id != worker_id
                ):
                    return False
                row.pid = pid
                row.updated_at = datetime.now(UTC)
                return True
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError(
                "drive manifest process start could not be saved"
            ) from exc

    def mark_manifest_process_finished(self, job_id: UUID, worker_id: str) -> bool:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveManifestJobModel, str(job_id))
                if row is None or row.worker_id != worker_id:
                    return False
                row.pid = None
                row.updated_at = datetime.now(UTC)
                return True
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError(
                "drive manifest process finish could not be saved"
            ) from exc

    def mark_manifest_synced(
        self, job_id: UUID, worker_id: str, synced_at: datetime
    ) -> DriveManifestJob:
        timestamp = utc(synced_at)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveManifestJobModel, str(job_id))
                if row is None or row.worker_id != worker_id:
                    raise DriveSyncRepositoryError("drive manifest job lease is no longer owned")
                if row.status != DriveSyncStatus.SYNCING.value:
                    if row.status == DriveSyncStatus.SYNCED.value:
                        return _manifest_job_domain(row)
                    raise DriveSyncRepositoryError("drive manifest job is no longer active")
                row.status = DriveSyncStatus.SYNCED.value
                row.progress_bytes = row.total_bytes
                row.progress_percentage = 100.0
                row.current_artifact = None
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.completed_at = timestamp
                row.error_code = None
                row.error_summary = None
                row.retryable = False
                row.updated_at = timestamp
                session.flush()
                return _manifest_job_domain(row)
        except DriveSyncRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest success could not be saved") from exc

    def mark_manifest_failed(
        self,
        job_id: UUID,
        worker_id: str | None,
        error_code: str,
        error_summary: str,
        retryable: bool = True,
    ) -> DriveManifestJob:
        now = datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(DriveManifestJobModel, str(job_id))
                if row is None:
                    raise DriveSyncRepositoryError("drive manifest job was not found")
                if row.status == DriveSyncStatus.SYNCED.value:
                    return _manifest_job_domain(row)
                if worker_id is not None and row.worker_id != worker_id:
                    raise DriveSyncRepositoryError("drive manifest job lease is no longer owned")
                row.status = DriveSyncStatus.FAILED.value
                row.worker_id = None
                row.pid = None
                row.claimed_at = None
                row.lease_expires_at = None
                row.completed_at = now
                row.error_code = error_code
                row.error_summary = error_summary[:1000]
                row.retryable = retryable
                row.updated_at = now
                session.flush()
                return _manifest_job_domain(row)
        except DriveSyncRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest failure could not be saved") from exc

    def clear_manifest_warning(self, local_date: str, destination: DriveDestination) -> None:
        self._update_manifest_warning(local_date, destination, None)

    def mark_manifest_warning_for_destination(
        self, local_date: str, destination: DriveDestination, summary: str
    ) -> None:
        self._update_manifest_warning(local_date, destination, summary)

    def _update_manifest_warning(
        self, local_date: str, destination: DriveDestination, summary: str | None
    ) -> None:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(DriveSyncRecordModel).where(
                        DriveSyncRecordModel.status == DriveSyncStatus.SYNCED.value,
                        DriveSyncRecordModel.remote_name == destination.remote_name,
                        DriveSyncRecordModel.remote_base_path == destination.base_path,
                    )
                ).all()
                for row in rows:
                    generation = session.get(GenerationModel, row.generation_id)
                    if generation is None or _tokyo_date(generation.created_at) != local_date:
                        continue
                    if summary is None:
                        if row.error_code == "drive_manifest_failed":
                            row.error_code = None
                            row.error_summary = None
                    else:
                        row.error_code = "drive_manifest_failed"
                        row.error_summary = summary[:1000]
                    row.updated_at = datetime.now(UTC)
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("drive manifest warning could not be saved") from exc

    def list_manifest_jobs(self, limit: int = 50) -> tuple[DriveManifestJob, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(DriveManifestJobModel)
                    .order_by(DriveManifestJobModel.queue_sequence.desc())
                    .limit(min(max(1, limit), 100))
                ).all()
                return tuple(_manifest_job_domain(row) for row in rows)
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest jobs could not be listed") from exc

    def has_active_manifest_jobs(self) -> bool:
        """Check all pending/syncing manifest jobs without a list limit."""

        try:
            with session_scope(self._session_factory) as session:
                statement = (
                    select(DriveManifestJobModel.id)
                    .where(
                        DriveManifestJobModel.status.in_(
                            [DriveSyncStatus.PENDING.value, DriveSyncStatus.SYNCING.value]
                        )
                    )
                    .limit(1)
                )
                return bool(session.scalar(select(statement.exists())))
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError(
                "active drive manifest jobs could not be checked"
            ) from exc

    def list_manifest_failure_targets(
        self, limit: int = 100
    ) -> tuple[DriveManifestFailureTarget, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(DriveManifestJobModel).order_by(
                        DriveManifestJobModel.queue_sequence.desc()
                    )
                ).all()
                latest: dict[tuple[str, str, str], DriveManifestJobModel] = {}
                for row in rows:
                    key = (row.local_date, row.remote_name, row.remote_base_path)
                    latest.setdefault(key, row)
                targets: dict[tuple[str, str, str], DriveManifestFailureTarget] = {
                    (local_date, remote_name, remote_base_path): DriveManifestFailureTarget(
                        local_date, remote_name, remote_base_path
                    )
                    for (local_date, remote_name, remote_base_path), row in latest.items()
                    if row.status == DriveSyncStatus.FAILED.value and row.retryable
                }

                warning_rows = session.execute(
                    select(
                        DriveSyncRecordModel.remote_name,
                        DriveSyncRecordModel.remote_base_path,
                        GenerationModel.created_at,
                    )
                    .join(
                        GenerationModel,
                        GenerationModel.id == DriveSyncRecordModel.generation_id,
                    )
                    .where(
                        DriveSyncRecordModel.status == DriveSyncStatus.SYNCED.value,
                        DriveSyncRecordModel.error_code == "drive_manifest_failed",
                    )
                ).all()
                for remote_name, remote_base_path, created_at in warning_rows:
                    local_date = _tokyo_date(created_at)
                    key = (local_date, remote_name, remote_base_path)
                    targets.setdefault(
                        key,
                        DriveManifestFailureTarget(local_date, remote_name, remote_base_path),
                    )
                return tuple(targets.values())[: min(max(1, limit), 100)]
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest failures could not be listed") from exc

    def reconcile_stale(self, now: datetime | None = None) -> int:
        timestamp = utc(now or datetime.now(UTC))
        count = 0
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(DriveSyncJobModel).where(
                        DriveSyncJobModel.status == DriveSyncStatus.SYNCING.value,
                        DriveSyncJobModel.lease_expires_at.is_not(None),
                        DriveSyncJobModel.lease_expires_at <= timestamp,
                    )
                ).all()
                for row in rows:
                    record = session.get(DriveSyncRecordModel, row.sync_record_id)
                    if record is None or record.status == DriveSyncStatus.SYNCED.value:
                        continue
                    row.status = DriveSyncStatus.FAILED.value
                    row.worker_id = None
                    row.pid = None
                    row.claimed_at = None
                    row.lease_expires_at = None
                    row.completed_at = timestamp
                    row.error_code = "drive_sync_stale"
                    row.error_summary = "同期Jobが期限切れになりました。"
                    row.retryable = True
                    row.updated_at = timestamp
                    record.status = DriveSyncStatus.FAILED.value
                    record.error_code = "drive_sync_stale"
                    record.error_summary = "同期Jobが期限切れになりました。"
                    record.updated_at = timestamp
                    count += 1
                manifest_rows = session.scalars(
                    select(DriveManifestJobModel).where(
                        DriveManifestJobModel.status == DriveSyncStatus.SYNCING.value,
                        DriveManifestJobModel.lease_expires_at.is_not(None),
                        DriveManifestJobModel.lease_expires_at <= timestamp,
                    )
                ).all()
                for row in manifest_rows:
                    row.status = DriveSyncStatus.FAILED.value
                    row.worker_id = None
                    row.pid = None
                    row.claimed_at = None
                    row.lease_expires_at = None
                    row.completed_at = timestamp
                    row.error_code = "drive_sync_stale"
                    row.error_summary = "同期Jobが期限切れになりました。"
                    row.retryable = True
                    row.updated_at = timestamp
                    count += 1
                return count
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("stale drive sync jobs could not be reconciled") from exc

    def reconcile_stateless_restore(self, now: datetime | None = None) -> int:
        """Fail restored Drive work because its local source may not exist on a new Pod."""

        timestamp = utc(now or datetime.now(UTC))
        unfinished = (DriveSyncStatus.PENDING.value, DriveSyncStatus.SYNCING.value)
        error_code = "stateless_restore_missing_local_artifact"
        error_summary = "Stateless復元後はローカルartifactを再開せず失敗として終了しました。"
        count = 0
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(DriveSyncJobModel).where(DriveSyncJobModel.status.in_(unfinished))
                ).all()
                for row in rows:
                    record = session.get(DriveSyncRecordModel, row.sync_record_id)
                    row.status = DriveSyncStatus.FAILED.value
                    row.worker_id = None
                    row.pid = None
                    row.claimed_at = None
                    row.lease_expires_at = None
                    row.completed_at = timestamp
                    row.error_code = error_code
                    row.error_summary = error_summary
                    row.retryable = False
                    row.updated_at = timestamp
                    if record is not None and record.status != DriveSyncStatus.SYNCED.value:
                        record.status = DriveSyncStatus.FAILED.value
                        record.error_code = error_code
                        record.error_summary = error_summary
                        record.updated_at = timestamp
                    count += 1
                manifest_rows = session.scalars(
                    select(DriveManifestJobModel).where(
                        DriveManifestJobModel.status.in_(unfinished)
                    )
                ).all()
                for row in manifest_rows:
                    row.status = DriveSyncStatus.FAILED.value
                    row.worker_id = None
                    row.pid = None
                    row.claimed_at = None
                    row.lease_expires_at = None
                    row.completed_at = timestamp
                    row.error_code = error_code
                    row.error_summary = error_summary
                    row.retryable = False
                    row.updated_at = timestamp
                    count += 1
                session.flush()
                return count
        except SQLAlchemyError as exc:
            raise DriveSyncRepositoryError("stateless Drive reconciliation failed") from exc

    def list_jobs(self, limit: int = 50) -> tuple[DriveSyncJob, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(DriveSyncJobModel)
                    .order_by(DriveSyncJobModel.queue_sequence.desc())
                    .limit(min(max(1, limit), 100))
                ).all()
                return tuple(_job_domain(row) for row in rows)
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync jobs could not be listed") from exc

    def get_latest_synced_job(self) -> DriveSyncJob | None:
        """Return the newest successful sync attempt without a history window."""

        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(DriveSyncJobModel)
                    .where(DriveSyncJobModel.status == DriveSyncStatus.SYNCED.value)
                    .order_by(
                        func.coalesce(
                            DriveSyncJobModel.completed_at,
                            DriveSyncJobModel.updated_at,
                        ).desc(),
                        DriveSyncJobModel.updated_at.desc(),
                        DriveSyncJobModel.id.desc(),
                    )
                    .limit(1)
                )
                return _job_domain(row) if row is not None else None
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("latest synced drive job could not be read") from exc

    def get_latest_unresolved_failure(self) -> DriveSyncJob | None:
        """Return the newest failed job whose sync record is still failed.

        The status count is record-based, while the timestamp must describe a
        failed attempt.  Joining both failed states prevents an old failed
        attempt from being reported after its record has subsequently synced.
        """

        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(DriveSyncJobModel)
                    .join(
                        DriveSyncRecordModel,
                        DriveSyncRecordModel.id == DriveSyncJobModel.sync_record_id,
                    )
                    .where(
                        DriveSyncRecordModel.status == DriveSyncStatus.FAILED.value,
                        DriveSyncJobModel.status == DriveSyncStatus.FAILED.value,
                    )
                    .order_by(DriveSyncJobModel.updated_at.desc(), DriveSyncJobModel.id.desc())
                    .limit(1)
                )
                return _job_domain(row) if row is not None else None
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError(
                "latest unresolved drive sync failure could not be read"
            ) from exc

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.execute(
                    select(
                        DriveSyncRecordModel.status, func.count(DriveSyncRecordModel.id)
                    ).group_by(DriveSyncRecordModel.status)
                ).all()
                return {DriveSyncStatus(status): int(count) for status, count in rows}
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync status could not be read") from exc

    def list_discovery_candidates(self, limit: int) -> tuple[DriveSyncDiscoveryCandidate, ...]:
        try:
            with session_scope(self._session_factory) as session:
                statement = (
                    select(GenerationModel.id, GenerationModel.kind, GenerationModel.created_at)
                    .join(
                        GenerationArtifactModel,
                        and_(
                            GenerationArtifactModel.generation_id == GenerationModel.id,
                            GenerationArtifactModel.artifact_type == ArtifactType.IMAGE.value,
                        ),
                    )
                    .outerjoin(
                        DriveSyncRecordModel,
                        DriveSyncRecordModel.generation_id == GenerationModel.id,
                    )
                    .where(
                        GenerationModel.status == GenerationStatus.COMPLETED.value,
                        DriveSyncRecordModel.id.is_(None),
                    )
                    .group_by(GenerationModel.id)
                    .order_by(GenerationModel.completed_at.asc(), GenerationModel.id.asc())
                    .limit(min(max(1, limit), 500))
                )
                rows = session.execute(statement).all()
                return tuple(
                    DriveSyncDiscoveryCandidate(UUID(row.id), row.kind, _utc(row.created_at))
                    for row in rows
                )
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive sync discovery could not be read") from exc

    def list_manifest_records(
        self, local_date: str, remote_name: str, remote_base_path: str
    ) -> tuple[DriveManifestRecord, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.execute(
                    select(
                        DriveSyncRecordModel,
                        GenerationModel.kind,
                        GenerationModel.created_at,
                    )
                    .join(
                        GenerationModel,
                        GenerationModel.id == DriveSyncRecordModel.generation_id,
                    )
                    .where(
                        DriveSyncRecordModel.status == DriveSyncStatus.SYNCED.value,
                        DriveSyncRecordModel.remote_name == remote_name,
                        DriveSyncRecordModel.remote_base_path == remote_base_path,
                    )
                    .order_by(GenerationModel.created_at.asc(), GenerationModel.id.asc())
                ).all()
                result: list[DriveManifestRecord] = []
                for record, kind, created_at in rows:
                    local = _utc(created_at).astimezone(_tokyo()).date().isoformat()
                    if (
                        local != local_date
                        or record.metadata_sha256 is None
                        or record.synced_at is None
                    ):
                        continue
                    artifacts = _sync_artifacts_from_row(record, status=record.status)
                    result.append(
                        DriveManifestRecord(
                            generation_id=UUID(record.generation_id),
                            kind=kind,
                            created_at=_utc(created_at),
                            remote_image_path=record.remote_image_path,
                            remote_metadata_path=record.remote_metadata_path,
                            image_sha256=record.image_sha256,
                            metadata_sha256=record.metadata_sha256,
                            image_size_bytes=record.image_size_bytes,
                            metadata_size_bytes=record.metadata_size_bytes or 0,
                            synced_at=_utc(record.synced_at),
                            remote_name=record.remote_name,
                            remote_base_path=record.remote_base_path,
                            artifacts=artifacts,
                        )
                    )
                return tuple(result)
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive manifest records could not be read") from exc

    def capacity(self, *, total_bytes: int, used_bytes: int, free_bytes: int) -> DriveCapacity:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.execute(
                    select(
                        GenerationModel.id,
                        DriveSyncRecordModel.status,
                        GenerationArtifactModel.artifact_type,
                        GenerationArtifactModel.size_bytes,
                        GenerationArtifactModel.created_at,
                    )
                    .join(
                        GenerationArtifactModel,
                        GenerationArtifactModel.generation_id == GenerationModel.id,
                    )
                    .outerjoin(
                        DriveSyncRecordModel,
                        DriveSyncRecordModel.generation_id == GenerationModel.id,
                    )
                    .where(GenerationModel.status == GenerationStatus.COMPLETED.value)
                    .order_by(GenerationArtifactModel.created_at.asc())
                ).all()
                grouped: dict[str, dict[str, object]] = defaultdict(dict)
                for generation_id, status, artifact_type, size, _created_at in rows:
                    entry = grouped[str(generation_id)]
                    entry.setdefault("status", status)
                    if artifact_type in {ArtifactType.IMAGE.value, ArtifactType.METADATA.value}:
                        previous = entry.get(artifact_type)
                        entry[artifact_type] = (previous if isinstance(previous, int) else 0) + int(
                            size
                        )
                unsynced = 0
                synced = 0
                for entry in grouped.values():
                    image_size = entry.get(ArtifactType.IMAGE.value)
                    metadata_size = entry.get(ArtifactType.METADATA.value)
                    size = (image_size if isinstance(image_size, int) else 0) + (
                        metadata_size if isinstance(metadata_size, int) else 0
                    )
                    if entry.get("status") == DriveSyncStatus.SYNCED.value:
                        synced += size
                    else:
                        unsynced += size
                return DriveCapacity(total_bytes, used_bytes, free_bytes, unsynced, synced)
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive capacity could not be calculated") from exc

    def cache_candidates(self, limit: int = 100) -> tuple[DriveCacheCandidate, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.execute(
                    select(
                        GenerationModel.id,
                        GenerationModel.kind,
                        GenerationModel.created_at,
                        GenerationArtifactModel.artifact_type,
                        GenerationArtifactModel.size_bytes,
                    )
                    .join(
                        GenerationArtifactModel,
                        GenerationArtifactModel.generation_id == GenerationModel.id,
                    )
                    .join(
                        DriveSyncRecordModel,
                        and_(
                            DriveSyncRecordModel.generation_id == GenerationModel.id,
                            DriveSyncRecordModel.status == DriveSyncStatus.SYNCED.value,
                        ),
                    )
                    .where(
                        GenerationModel.status == GenerationStatus.COMPLETED.value,
                        GenerationArtifactModel.artifact_type.in_(
                            [ArtifactType.IMAGE.value, ArtifactType.METADATA.value]
                        ),
                    )
                    .order_by(GenerationModel.created_at.desc(), GenerationModel.id.desc())
                ).all()
                grouped: dict[str, tuple[UUID, str, datetime]] = {}
                sizes: dict[str, dict[str, int]] = defaultdict(dict)
                for generation_id, kind, created_at, artifact_type, size in rows:
                    key = str(generation_id)
                    grouped.setdefault(key, (UUID(key), kind, _utc(created_at)))
                    if artifact_type in {ArtifactType.IMAGE.value, ArtifactType.METADATA.value}:
                        sizes[key][artifact_type] = sizes[key].get(artifact_type, 0) + int(size)
                result = [
                    DriveCacheCandidate(
                        generation_id,
                        kind,
                        created_at,
                        sum(sizes[key].values()),
                    )
                    for key, (generation_id, kind, created_at) in grouped.items()
                ]
                return tuple(result[: min(max(1, limit), 100)])
        except (SQLAlchemyError, ValueError) as exc:
            raise DriveSyncRepositoryError("drive cache candidates could not be read") from exc


def _require_manifest_ready(session: Session, row: DriveSyncRecordModel) -> None:
    if row.status != DriveSyncStatus.SYNCED.value:
        raise DriveManifestRebuildRequired("drive sync record is not synced")
    local_date = _manifest_local_date_from_record(row)
    destination = DriveDestination(row.remote_name, row.remote_base_path)
    if (
        row.error_code == DriveSyncErrorCode.MANIFEST_FAILED.value
        or _manifest_state_in_session(session, local_date, destination)
        is not DriveManifestState.SYNCED
    ):
        raise DriveManifestRebuildRequired("old destination manifest rebuild is required")


def _manifest_state_in_session(
    session: Session, local_date: str, destination: DriveDestination
) -> DriveManifestState:
    rows = session.scalars(
        select(DriveManifestJobModel)
        .where(
            DriveManifestJobModel.local_date == local_date,
            DriveManifestJobModel.remote_name == destination.remote_name,
            DriveManifestJobModel.remote_base_path == destination.base_path,
        )
        .order_by(DriveManifestJobModel.queue_sequence.desc())
    ).all()
    if not rows:
        return DriveManifestState.MISSING
    for row in rows:
        if row.status == DriveSyncStatus.SYNCING.value:
            return DriveManifestState.SYNCING
        if row.status == DriveSyncStatus.PENDING.value:
            return DriveManifestState.PENDING
    return DriveManifestState(rows[0].status)


def _manifest_local_date_from_record(row: DriveSyncRecordModel) -> str:
    image_date = _remote_path_date_prefix(row.remote_image_path)
    metadata_date = _remote_path_date_prefix(row.remote_metadata_path)
    if image_date != metadata_date:
        raise DriveManifestRebuildRequired("sync record remote dates are inconsistent")
    return image_date


def _remote_path_date_prefix(path: str) -> str:
    prefix = path.replace("\\", "/").split("/", 1)[0]
    try:
        return date.fromisoformat(prefix).isoformat()
    except ValueError as exc:
        raise DriveManifestRebuildRequired("sync record remote date is invalid") from exc


def _record_model(record: DriveSyncRecord) -> DriveSyncRecordModel:
    return DriveSyncRecordModel(
        id=str(record.id),
        generation_id=str(record.generation_id),
        status=record.status.value,
        remote_name=record.remote_name,
        remote_base_path=record.remote_base_path,
        remote_image_path=record.remote_image_path,
        remote_metadata_path=record.remote_metadata_path,
        image_artifact_id=str(record.image_artifact_id),
        metadata_artifact_id=(
            str(record.metadata_artifact_id) if record.metadata_artifact_id is not None else None
        ),
        image_sha256=record.image_sha256,
        metadata_sha256=record.metadata_sha256,
        image_size_bytes=record.image_size_bytes,
        metadata_size_bytes=record.metadata_size_bytes,
        artifacts_json=_artifacts_json(record.artifacts),
        attempt_count=record.attempt_count,
        last_attempt_at=record.last_attempt_at,
        synced_at=record.synced_at,
        error_code=record.error_code,
        error_summary=record.error_summary,
        created_at=utc(record.created_at),
        updated_at=utc(record.updated_at),
    )


def _copy_record_values(
    row: DriveSyncRecordModel, record: DriveSyncRecord, *, status: DriveSyncStatus
) -> None:
    row.status = status.value
    row.remote_name = record.remote_name
    row.remote_base_path = record.remote_base_path
    row.remote_image_path = record.remote_image_path
    row.remote_metadata_path = record.remote_metadata_path
    row.image_artifact_id = str(record.image_artifact_id)
    row.metadata_artifact_id = (
        str(record.metadata_artifact_id) if record.metadata_artifact_id is not None else None
    )
    row.image_sha256 = record.image_sha256
    row.metadata_sha256 = record.metadata_sha256
    row.image_size_bytes = record.image_size_bytes
    row.metadata_size_bytes = record.metadata_size_bytes
    row.artifacts_json = _artifacts_json(record.artifacts)
    row.updated_at = datetime.now(UTC)


def _job_model(job: DriveSyncJob, record_id: str, session: Session) -> DriveSyncJobModel:
    max_sequence = session.scalar(select(func.max(DriveSyncJobModel.queue_sequence))) or 0
    return DriveSyncJobModel(
        id=str(job.id or uuid4()),
        sync_record_id=record_id,
        generation_id=str(job.generation_id),
        queue_sequence=int(max_sequence) + 1,
        status=DriveSyncStatus.PENDING.value,
        progress_bytes=0,
        total_bytes=job.total_bytes,
        progress_percentage=0.0,
        current_artifact=None,
        worker_id=None,
        pid=None,
        claimed_at=None,
        lease_expires_at=None,
        started_at=None,
        completed_at=None,
        error_code=None,
        error_summary=None,
        retryable=True,
        log_path=job.log_path,
        image_artifact_id=str(job.image_artifact_id),
        metadata_artifact_id=(
            str(job.metadata_artifact_id) if job.metadata_artifact_id is not None else None
        ),
        image_sha256=job.image_sha256,
        metadata_sha256=job.metadata_sha256,
        image_size_bytes=job.image_size_bytes,
        metadata_size_bytes=job.metadata_size_bytes,
        artifacts_json=_artifacts_json(job.artifacts),
        created_at=utc(job.created_at),
        updated_at=utc(job.updated_at),
    )


def _manifest_job_model(job: DriveManifestJob, session: Session) -> DriveManifestJobModel:
    max_sequence = session.scalar(select(func.max(DriveManifestJobModel.queue_sequence))) or 0
    return DriveManifestJobModel(
        id=str(job.id),
        local_date=job.local_date,
        remote_name=job.remote_name,
        remote_base_path=job.remote_base_path,
        remote_manifest_path=job.remote_manifest_path,
        queue_sequence=int(max_sequence) + 1,
        status=DriveSyncStatus.PENDING.value,
        progress_bytes=0,
        total_bytes=job.total_bytes,
        progress_percentage=0.0,
        current_artifact=None,
        worker_id=None,
        pid=None,
        claimed_at=None,
        lease_expires_at=None,
        started_at=None,
        completed_at=None,
        error_code=None,
        error_summary=None,
        retryable=True,
        log_path=job.log_path,
        created_at=utc(job.created_at),
        updated_at=utc(job.updated_at),
    )


def _record_domain(row: DriveSyncRecordModel) -> DriveSyncRecord:
    return DriveSyncRecord(
        id=UUID(row.id),
        generation_id=UUID(row.generation_id),
        status=DriveSyncStatus(row.status),
        remote_name=row.remote_name,
        remote_base_path=row.remote_base_path,
        remote_image_path=row.remote_image_path,
        remote_metadata_path=row.remote_metadata_path,
        image_artifact_id=UUID(row.image_artifact_id),
        metadata_artifact_id=UUID(row.metadata_artifact_id) if row.metadata_artifact_id else None,
        image_sha256=row.image_sha256,
        metadata_sha256=row.metadata_sha256,
        image_size_bytes=row.image_size_bytes,
        metadata_size_bytes=row.metadata_size_bytes,
        attempt_count=row.attempt_count,
        last_attempt_at=_optional_utc(row.last_attempt_at),
        synced_at=_optional_utc(row.synced_at),
        error_code=row.error_code,
        error_summary=row.error_summary,
        created_at=utc(row.created_at),
        updated_at=utc(row.updated_at),
        artifacts=_sync_artifacts_from_row(row, status=row.status),
    )


def _job_domain(row: DriveSyncJobModel) -> DriveSyncJob:
    return DriveSyncJob(
        id=UUID(row.id),
        sync_record_id=UUID(row.sync_record_id),
        generation_id=UUID(row.generation_id),
        status=DriveSyncStatus(row.status),
        queue_sequence=row.queue_sequence,
        progress_bytes=row.progress_bytes,
        total_bytes=row.total_bytes,
        progress_percentage=row.progress_percentage,
        current_artifact=row.current_artifact,
        worker_id=row.worker_id,
        pid=row.pid,
        claimed_at=_optional_utc(row.claimed_at),
        lease_expires_at=_optional_utc(row.lease_expires_at),
        started_at=_optional_utc(row.started_at),
        completed_at=_optional_utc(row.completed_at),
        error_code=row.error_code,
        error_summary=row.error_summary,
        retryable=row.retryable,
        log_path=row.log_path,
        image_artifact_id=UUID(row.image_artifact_id),
        metadata_artifact_id=UUID(row.metadata_artifact_id) if row.metadata_artifact_id else None,
        image_sha256=row.image_sha256,
        metadata_sha256=row.metadata_sha256,
        image_size_bytes=row.image_size_bytes,
        metadata_size_bytes=row.metadata_size_bytes,
        created_at=utc(row.created_at),
        updated_at=utc(row.updated_at),
        artifacts=_sync_artifacts_from_row(row, status=row.status),
    )


def _artifacts_json(artifacts: tuple[DriveSyncArtifact, ...]) -> str | None:
    if not artifacts:
        return None
    return json.dumps(
        [
            {
                "display_order": item.display_order,
                "image_artifact_id": str(item.image_artifact_id),
                "remote_image_path": item.remote_image_path,
                "image_sha256": item.image_sha256,
                "image_size_bytes": item.image_size_bytes,
                "metadata_artifact_id": (
                    str(item.metadata_artifact_id)
                    if item.metadata_artifact_id is not None
                    else None
                ),
                "remote_metadata_path": item.remote_metadata_path,
                "metadata_sha256": item.metadata_sha256,
                "metadata_size_bytes": item.metadata_size_bytes,
                "image_synced": item.image_synced,
                "metadata_synced": item.metadata_synced,
            }
            for item in artifacts
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sync_artifacts_from_row(row: object, *, status: str) -> tuple[DriveSyncArtifact, ...]:
    payload = getattr(row, "artifacts_json", None)
    if payload:
        values = json.loads(payload)
        if not isinstance(values, list) or not values:
            raise ValueError("stored Drive artifact plan is invalid")
        result = tuple(_sync_artifact_from_mapping(value) for value in values)
        if tuple(item.display_order for item in result) != tuple(range(len(result))):
            raise ValueError("stored Drive artifact order is invalid")
        return result

    image_id = getattr(row, "image_artifact_id", None)
    if not isinstance(image_id, str):
        return ()
    metadata_id = getattr(row, "metadata_artifact_id", None)
    raw_metadata_size = getattr(row, "metadata_size_bytes", None)
    return (
        DriveSyncArtifact(
            display_order=0,
            image_artifact_id=UUID(image_id),
            remote_image_path=str(getattr(row, "remote_image_path", "")),
            image_sha256=str(getattr(row, "image_sha256", "")),
            image_size_bytes=int(getattr(row, "image_size_bytes", 0)),
            metadata_artifact_id=UUID(metadata_id) if metadata_id else None,
            remote_metadata_path=(
                str(getattr(row, "remote_metadata_path", ""))
                if getattr(row, "remote_metadata_path", None)
                else None
            ),
            metadata_sha256=(
                str(getattr(row, "metadata_sha256", ""))
                if getattr(row, "metadata_sha256", None)
                else None
            ),
            metadata_size_bytes=(raw_metadata_size if isinstance(raw_metadata_size, int) else None),
            image_synced=status == DriveSyncStatus.SYNCED.value,
            metadata_synced=(status == DriveSyncStatus.SYNCED.value and metadata_id is not None),
        ),
    )


def _sync_artifact_from_mapping(value: object) -> DriveSyncArtifact:
    if not isinstance(value, dict):
        raise ValueError("stored Drive artifact is invalid")
    try:
        return DriveSyncArtifact(
            display_order=int(value["display_order"]),
            image_artifact_id=UUID(str(value["image_artifact_id"])),
            remote_image_path=str(value["remote_image_path"]),
            image_sha256=str(value["image_sha256"]),
            image_size_bytes=int(value["image_size_bytes"]),
            metadata_artifact_id=(
                UUID(str(value["metadata_artifact_id"]))
                if value.get("metadata_artifact_id")
                else None
            ),
            remote_metadata_path=(
                str(value["remote_metadata_path"]) if value.get("remote_metadata_path") else None
            ),
            metadata_sha256=(
                str(value["metadata_sha256"]) if value.get("metadata_sha256") else None
            ),
            metadata_size_bytes=(
                int(value["metadata_size_bytes"])
                if value.get("metadata_size_bytes") is not None
                else None
            ),
            image_synced=bool(value.get("image_synced", False)),
            metadata_synced=bool(value.get("metadata_synced", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("stored Drive artifact is invalid") from exc


def _set_artifact_progress(
    artifacts: tuple[DriveSyncArtifact, ...], display_order: int, artifact_kind: str
) -> tuple[DriveSyncArtifact, ...]:
    for item in artifacts:
        if item.display_order == display_order:
            return tuple(
                (
                    replace(candidate, image_synced=True)
                    if artifact_kind == "image" and candidate.display_order == display_order
                    else (
                        replace(candidate, metadata_synced=True)
                        if artifact_kind == "metadata" and candidate.display_order == display_order
                        else candidate
                    )
                )
                for candidate in artifacts
            )
    raise DriveSyncRepositoryError("drive sync artifact was not found")


def _manifest_job_domain(row: DriveManifestJobModel) -> DriveManifestJob:
    return DriveManifestJob(
        id=UUID(row.id),
        local_date=row.local_date,
        status=DriveSyncStatus(row.status),
        remote_name=row.remote_name,
        remote_base_path=row.remote_base_path,
        remote_manifest_path=row.remote_manifest_path,
        queue_sequence=row.queue_sequence,
        progress_bytes=row.progress_bytes,
        total_bytes=row.total_bytes,
        progress_percentage=row.progress_percentage,
        current_artifact=row.current_artifact,
        worker_id=row.worker_id,
        pid=row.pid,
        claimed_at=_optional_utc(row.claimed_at),
        lease_expires_at=_optional_utc(row.lease_expires_at),
        started_at=_optional_utc(row.started_at),
        completed_at=_optional_utc(row.completed_at),
        error_code=row.error_code,
        error_summary=row.error_summary,
        retryable=row.retryable,
        log_path=row.log_path,
        created_at=utc(row.created_at),
        updated_at=utc(row.updated_at),
    )


def _optional_utc(value: datetime | None) -> datetime | None:
    return utc(value) if value is not None else None


def _utc(value: datetime | None) -> datetime:
    return utc(value or datetime.now(UTC))


def _tokyo() -> ZoneInfo:
    return ZoneInfo("Asia/Tokyo")


def _tokyo_date(value: datetime) -> str:
    return _utc(value).astimezone(_tokyo()).date().isoformat()


__all__ = [
    "DriveManifestRecord",
    "DriveManifestFailureTarget",
    "DriveManifestRebuildRequired",
    "DriveSyncDiscoveryCandidate",
    "DriveSyncRepository",
    "DriveSyncRepositoryError",
    "DriveSyncRepositoryProtocol",
]
