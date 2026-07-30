"""Atomic Generation and Job progress persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import GenerationJobModel, GenerationModel
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepositoryError,
    _require_generation,
    _require_job,
)
from runpod_sdxl_image_studio.domain.generation import GenerationStatus


class GenerationProgressRepositoryProtocol(Protocol):
    """契約 for atomic running transitions and progress updates."""

    def update_progress(
        self,
        generation_id: UUID,
        job_id: UUID,
        *,
        state: GenerationStatus,
        value: int | None,
        maximum: int | None,
        current_node: str | None,
        updated_at: datetime,
    ) -> None: ...


class GenerationProgressRepository(GenerationProgressRepositoryProtocol):
    """Update the Generation/Job pair without leaving a partial transition."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def update_progress(
        self,
        generation_id: UUID,
        job_id: UUID,
        *,
        state: GenerationStatus,
        value: int | None,
        maximum: int | None,
        current_node: str | None,
        updated_at: datetime,
    ) -> None:
        _validate_progress(value, maximum)
        timestamp = _utc(updated_at)
        if state not in {GenerationStatus.QUEUED, GenerationStatus.RUNNING}:
            raise GenerationRepositoryError("progress repository accepts queued or running only")
        try:
            with session_scope(self._session_factory) as session:
                generation = _require_generation(session, generation_id)
                job = _require_job(session, job_id)
                if job.generation_id != str(generation_id):
                    raise GenerationRepositoryError("job generation does not match")

                generation_status = GenerationStatus(generation.status)
                job_status = GenerationStatus(job.status)
                if generation_status is not job_status:
                    raise GenerationRepositoryError("generation and job statuses are inconsistent")

                if state is GenerationStatus.QUEUED:
                    if generation_status is not GenerationStatus.QUEUED:
                        raise GenerationRepositoryError("generation and job are not both queued")
                    _update_job_progress(job, value, maximum, current_node, timestamp)
                else:
                    if generation_status not in {
                        GenerationStatus.QUEUED,
                        GenerationStatus.RUNNING,
                    }:
                        raise GenerationRepositoryError("generation and job cannot become running")
                    _mark_generation_running(generation, timestamp)
                    _mark_job_running(job, value, maximum, current_node, timestamp)
                session.flush()
        except GenerationRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationRepositoryError("generation progress could not be persisted") from exc


def _mark_generation_running(row: GenerationModel, updated_at: datetime) -> None:
    row.status = GenerationStatus.RUNNING.value
    if row.started_at is None:
        row.started_at = updated_at
    row.updated_at = updated_at


def _mark_job_running(
    row: GenerationJobModel,
    value: int | None,
    maximum: int | None,
    current_node: str | None,
    updated_at: datetime,
) -> None:
    row.status = GenerationStatus.RUNNING.value
    if row.started_at is None:
        row.started_at = updated_at
    _update_job_progress(row, value, maximum, current_node, updated_at)


def _update_job_progress(
    row: GenerationJobModel,
    value: int | None,
    maximum: int | None,
    current_node: str | None,
    updated_at: datetime,
) -> None:
    row.progress_value = value
    row.progress_maximum = maximum
    row.current_node = current_node
    row.updated_at = updated_at


def _validate_progress(value: int | None, maximum: int | None) -> None:
    if value is not None and value < 0:
        raise GenerationRepositoryError("progress value must not be negative")
    if maximum is not None and maximum < 0:
        raise GenerationRepositoryError("progress maximum must not be negative")
    if value is not None and maximum is not None and value > maximum:
        raise GenerationRepositoryError("progress value must not exceed maximum")


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "GenerationProgressRepository",
    "GenerationProgressRepositoryProtocol",
    "GenerationRepositoryError",
]
