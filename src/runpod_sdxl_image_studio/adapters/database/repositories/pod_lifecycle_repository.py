"""SQLite repository for the current RunPod lifecycle session."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import PodLifecycleSessionModel
from runpod_sdxl_image_studio.domain.pod_lifecycle import AutoTerminateState, PodLifecycleSession


class PodLifecycleRepositoryError(RuntimeError):
    """Safe persistence error for lifecycle state."""


class PodLifecycleRepositoryProtocol(Protocol):
    def get_by_pod_id(self, pod_id: str) -> PodLifecycleSession | None: ...

    def get_or_create(
        self, pod_id: str, *, auto_terminate_enabled: bool, now: datetime | None = None
    ) -> PodLifecycleSession: ...

    def save(self, session: PodLifecycleSession) -> PodLifecycleSession: ...

    def set_status(
        self,
        session_id: UUID,
        status: AutoTerminateState,
        *,
        error_code: str | None = None,
        error_summary: str | None = None,
        now: datetime | None = None,
    ) -> PodLifecycleSession: ...


class PodLifecycleRepository(PodLifecycleRepositoryProtocol):
    """Persist lifecycle state without ever accepting an API key."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_by_pod_id(self, pod_id: str) -> PodLifecycleSession | None:
        normalized = _safe_pod_id(pod_id)
        if not normalized:
            return None
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(PodLifecycleSessionModel).where(
                        PodLifecycleSessionModel.pod_id == normalized
                    )
                )
                return _domain(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise PodLifecycleRepositoryError("lifecycle session could not be read") from exc

    def get_or_create(
        self, pod_id: str, *, auto_terminate_enabled: bool, now: datetime | None = None
    ) -> PodLifecycleSession:
        normalized = _safe_pod_id(pod_id)
        if not normalized:
            raise PodLifecycleRepositoryError("RunPod pod identity is missing")
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(PodLifecycleSessionModel).where(
                        PodLifecycleSessionModel.pod_id == normalized
                    )
                )
                if row is None:
                    row = PodLifecycleSessionModel(
                        id=str(uuid4()),
                        pod_id=normalized,
                        started_at=timestamp,
                        auto_terminate_enabled=auto_terminate_enabled,
                        auto_terminate_armed_at=None,
                        status=AutoTerminateState.IDLE.value,
                        last_activity_at=timestamp,
                        last_error_code=None,
                        last_error_summary=None,
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                    session.add(row)
                else:
                    row.updated_at = timestamp
                    row.last_activity_at = timestamp
                session.flush()
                return _domain(row)
        except SQLAlchemyError as exc:
            raise PodLifecycleRepositoryError("lifecycle session could not be created") from exc

    def save(self, lifecycle: PodLifecycleSession) -> PodLifecycleSession:
        timestamp = _utc(lifecycle.updated_at)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(PodLifecycleSessionModel, str(lifecycle.id))
                if row is None:
                    row = PodLifecycleSessionModel(
                        id=str(lifecycle.id),
                        pod_id=_safe_pod_id(lifecycle.pod_id),
                        started_at=_utc(lifecycle.started_at),
                        auto_terminate_enabled=lifecycle.auto_terminate_enabled,
                        auto_terminate_armed_at=_utc_or_none(lifecycle.auto_terminate_armed_at),
                        status=lifecycle.status.value,
                        last_activity_at=_utc_or_none(lifecycle.last_activity_at),
                        last_error_code=lifecycle.last_error_code,
                        last_error_summary=_bounded(lifecycle.last_error_summary),
                        created_at=_utc(lifecycle.created_at),
                        updated_at=timestamp,
                    )
                    session.add(row)
                else:
                    _apply(row, lifecycle, timestamp)
                session.flush()
                return _domain(row)
        except (SQLAlchemyError, ValueError) as exc:
            raise PodLifecycleRepositoryError("lifecycle session could not be saved") from exc

    def set_status(
        self,
        session_id: UUID,
        status: AutoTerminateState,
        *,
        error_code: str | None = None,
        error_summary: str | None = None,
        now: datetime | None = None,
    ) -> PodLifecycleSession:
        timestamp = _utc(now or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(PodLifecycleSessionModel, str(session_id))
                if row is None:
                    raise PodLifecycleRepositoryError("lifecycle session was not found")
                row.status = status.value
                row.last_activity_at = timestamp
                row.updated_at = timestamp
                row.last_error_code = error_code
                row.last_error_summary = _bounded(error_summary)
                session.flush()
                return _domain(row)
        except PodLifecycleRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise PodLifecycleRepositoryError("lifecycle status could not be saved") from exc


def _domain(row: PodLifecycleSessionModel) -> PodLifecycleSession:
    return PodLifecycleSession(
        id=UUID(row.id),
        pod_id=row.pod_id,
        started_at=_utc(row.started_at),
        auto_terminate_enabled=row.auto_terminate_enabled,
        auto_terminate_armed_at=_utc_or_none(row.auto_terminate_armed_at),
        status=AutoTerminateState(row.status),
        last_activity_at=_utc_or_none(row.last_activity_at),
        last_error_code=row.last_error_code,
        last_error_summary=row.last_error_summary,
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
    )


def _apply(
    row: PodLifecycleSessionModel, lifecycle: PodLifecycleSession, timestamp: datetime
) -> None:
    row.pod_id = _safe_pod_id(lifecycle.pod_id)
    row.started_at = _utc(lifecycle.started_at)
    row.auto_terminate_enabled = lifecycle.auto_terminate_enabled
    row.auto_terminate_armed_at = _utc_or_none(lifecycle.auto_terminate_armed_at)
    row.status = lifecycle.status.value
    row.last_activity_at = _utc_or_none(lifecycle.last_activity_at)
    row.last_error_code = lifecycle.last_error_code
    row.last_error_summary = _bounded(lifecycle.last_error_summary)
    row.updated_at = timestamp


def _safe_pod_id(value: str) -> str:
    normalized = value.strip()
    if len(normalized) > 200 or any(char in normalized for char in "\r\n\x00"):
        raise ValueError("pod id is invalid")
    return normalized


def _bounded(value: str | None) -> str | None:
    return value[:1000] if value is not None else None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _utc_or_none(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None


__all__ = [
    "PodLifecycleRepository",
    "PodLifecycleRepositoryError",
    "PodLifecycleRepositoryProtocol",
]
