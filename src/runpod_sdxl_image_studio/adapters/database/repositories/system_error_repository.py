"""Persistence for bounded, sanitized operational error history."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import SystemErrorEventModel
from runpod_sdxl_image_studio.domain.system_status import ErrorSeverity, SystemErrorEvent

MAX_ERROR_SUMMARY_LENGTH = 500
MAX_ERROR_DETAILS_LENGTH = 2_000

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|authorization|bearer|"
    r"rclone[_ -]?config|client[_ -]?secret)\b\s*[:=]\s*[^\s,;]+"
)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:(?:[A-Za-z]:[\\/]|/(?:workspace|home|root|tmp|var|etc)/)[^\s,;]+)"
)


class SystemErrorEventRepositoryError(RuntimeError):
    """Safe persistence boundary error."""


class SystemErrorEventRepositoryProtocol(Protocol):
    def append(self, event: SystemErrorEvent) -> SystemErrorEvent: ...

    def record(
        self,
        *,
        category: str,
        severity: ErrorSeverity | str,
        error_code: str,
        summary: str,
        generation_id: UUID | None = None,
        job_id: UUID | None = None,
        retryable: bool = False,
        details: str | None = None,
        created_at: datetime | None = None,
    ) -> SystemErrorEvent: ...

    def list_recent(self, limit: int = 100) -> tuple[SystemErrorEvent, ...]: ...


class SystemErrorEventRepository(SystemErrorEventRepositoryProtocol):
    """Append-only repository with a bounded read surface."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def append(self, event: SystemErrorEvent) -> SystemErrorEvent:
        safe_event = sanitize_event(event)
        row = SystemErrorEventModel(
            id=str(safe_event.id),
            created_at=_utc(safe_event.created_at),
            category=safe_event.category,
            severity=_severity_value(safe_event.severity),
            error_code=safe_event.error_code,
            summary=safe_event.summary,
            generation_id=str(safe_event.generation_id) if safe_event.generation_id else None,
            job_id=str(safe_event.job_id) if safe_event.job_id else None,
            retryable=safe_event.retryable,
            details=safe_event.details,
        )
        try:
            with session_scope(self._session_factory) as session:
                session.add(row)
                session.flush()
            return safe_event
        except SQLAlchemyError as exc:
            raise SystemErrorEventRepositoryError("system error could not be saved") from exc

    def record(
        self,
        *,
        category: str,
        severity: ErrorSeverity | str,
        error_code: str,
        summary: str,
        generation_id: UUID | None = None,
        job_id: UUID | None = None,
        retryable: bool = False,
        details: str | None = None,
        created_at: datetime | None = None,
    ) -> SystemErrorEvent:
        return self.append(
            SystemErrorEvent(
                id=uuid4(),
                created_at=created_at or datetime.now(UTC),
                category=category,
                severity=severity,
                error_code=error_code,
                summary=summary,
                generation_id=generation_id,
                job_id=job_id,
                retryable=retryable,
                details=details,
            )
        )

    def list_recent(self, limit: int = 100) -> tuple[SystemErrorEvent, ...]:
        bounded_limit = min(max(1, limit), 100)
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(SystemErrorEventModel)
                    .order_by(
                        SystemErrorEventModel.created_at.desc(),
                        SystemErrorEventModel.id.desc(),
                    )
                    .limit(bounded_limit)
                ).all()
                return tuple(_to_domain(row) for row in rows)
        except (SQLAlchemyError, ValueError) as exc:
            raise SystemErrorEventRepositoryError("system errors could not be listed") from exc


def sanitize_error_text(value: str | None, *, max_length: int) -> str | None:
    """Remove secrets, absolute paths, ANSI escapes, and unbounded text."""

    if value is None:
        return None
    normalized = _ANSI_ESCAPE_RE.sub("", str(value))
    normalized = _CONTROL_RE.sub(" ", normalized)
    normalized = _SECRET_RE.sub("<redacted-secret>", normalized)
    normalized = _ABSOLUTE_PATH_RE.sub("<redacted-path>", normalized)
    normalized = " ".join(normalized.split())
    if not normalized:
        return None
    return normalized[:max_length]


def sanitize_event(event: SystemErrorEvent) -> SystemErrorEvent:
    """Return the only representation allowed to cross into persistence."""

    return SystemErrorEvent(
        id=event.id,
        created_at=_utc(event.created_at),
        category=_bounded_identifier(event.category, 64, "system"),
        severity=_severity_value(event.severity),
        error_code=_bounded_identifier(event.error_code, 64, "system_error"),
        summary=sanitize_error_text(event.summary, max_length=MAX_ERROR_SUMMARY_LENGTH)
        or "system error",
        generation_id=event.generation_id,
        job_id=event.job_id,
        retryable=bool(event.retryable),
        details=sanitize_error_text(event.details, max_length=MAX_ERROR_DETAILS_LENGTH),
    )


def _to_domain(row: SystemErrorEventModel) -> SystemErrorEvent:
    return SystemErrorEvent(
        id=UUID(row.id),
        created_at=_utc(row.created_at),
        category=row.category,
        severity=ErrorSeverity(row.severity),
        error_code=row.error_code,
        summary=row.summary,
        generation_id=UUID(row.generation_id) if row.generation_id else None,
        job_id=UUID(row.job_id) if row.job_id else None,
        retryable=row.retryable,
        details=row.details,
    )


def _severity_value(value: ErrorSeverity | str) -> str:
    try:
        return ErrorSeverity(value).value
    except ValueError:
        return ErrorSeverity.ERROR.value


def _bounded_identifier(value: str, max_length: int, fallback: str) -> str:
    normalized = sanitize_error_text(value, max_length=max_length)
    return normalized or fallback


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "MAX_ERROR_DETAILS_LENGTH",
    "MAX_ERROR_SUMMARY_LENGTH",
    "SystemErrorEventRepository",
    "SystemErrorEventRepositoryError",
    "SystemErrorEventRepositoryProtocol",
    "sanitize_error_text",
]
