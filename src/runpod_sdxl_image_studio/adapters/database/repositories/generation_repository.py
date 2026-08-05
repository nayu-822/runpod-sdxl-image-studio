"""Repositories for persisted generations, artifacts, and jobs."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import (
    GenerationArtifactModel,
    GenerationJobModel,
    GenerationLoraModel,
    GenerationModel,
)
from runpod_sdxl_image_studio.domain.generation import (
    Generation,
    GenerationKind,
    GenerationStatus,
    is_valid_status_transition,
)
from runpod_sdxl_image_studio.domain.generation_artifact import (
    ArtifactType,
    GenerationArtifact,
)
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationHistoryFilter,
    GenerationHistoryPage,
    GenerationHistoryQuery,
    GenerationHistorySort,
)
from runpod_sdxl_image_studio.domain.generation_queue import OptionalArtifactRepairCandidate
from runpod_sdxl_image_studio.domain.generation_snapshot import (
    GenerationSettingsSnapshot,
    SnapshotError,
)
from runpod_sdxl_image_studio.domain.job import GenerationJob


class GenerationRepositoryError(RuntimeError):
    """Safe persistence error for generation records."""


class GenerationRepositoryProtocol(Protocol):
    def create_pending(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        kind: GenerationKind = GenerationKind.STANDARD,
        parent_generation_id: UUID | None = None,
        generation_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> Generation: ...

    def mark_queued(self, generation_id: UUID, prompt_id: str) -> Generation: ...

    def mark_running(self, generation_id: UUID) -> Generation: ...

    def mark_completed(
        self, generation_id: UUID, completed_at: datetime | None = None
    ) -> Generation: ...

    def mark_failed(
        self,
        generation_id: UUID,
        error_code: str,
        error_summary: str,
        completed_at: datetime | None = None,
    ) -> Generation: ...

    def get_by_id(self, generation_id: UUID) -> Generation | None: ...

    def get_by_prompt_id(self, prompt_id: str) -> Generation | None: ...

    def list_completed_optional_artifact_repairs(
        self,
        limit: int = 50,
        *,
        after_completed_at: datetime | None = None,
        after_generation_id: UUID | None = None,
    ) -> tuple[OptionalArtifactRepairCandidate, ...]: ...

    def list_history(self, query: GenerationHistoryFilter) -> GenerationHistoryPage: ...

    def set_favorite(self, generation_id: UUID, favorite: bool) -> Generation: ...

    def update_note(self, generation_id: UUID, note: str | None) -> Generation: ...


class GenerationRepository(GenerationRepositoryProtocol):
    """SQLAlchemy repository with explicit transaction and transition guards."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_pending(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        kind: GenerationKind = GenerationKind.STANDARD,
        parent_generation_id: UUID | None = None,
        generation_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> Generation:
        generation_id = generation_id or uuid4()
        timestamp = _utc(created_at or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                if (
                    parent_generation_id is not None
                    and session.get(GenerationModel, str(parent_generation_id)) is None
                ):
                    raise GenerationRepositoryError("parent generation was not found")
                row = GenerationModel(
                    id=str(generation_id),
                    kind=kind.value,
                    status=GenerationStatus.PENDING.value,
                    parent_generation_id=(
                        str(parent_generation_id) if parent_generation_id is not None else None
                    ),
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
                session.add(row)
                session.flush()
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
                return _generation_domain(row)
        except GenerationRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError) as exc:
            raise GenerationRepositoryError("generation could not be created") from exc

    def mark_queued(self, generation_id: UUID, prompt_id: str) -> Generation:
        return self._transition(generation_id, GenerationStatus.QUEUED, prompt_id=prompt_id)

    def mark_running(self, generation_id: UUID) -> Generation:
        return self._transition(generation_id, GenerationStatus.RUNNING)

    def mark_completed(
        self, generation_id: UUID, completed_at: datetime | None = None
    ) -> Generation:
        return self._transition(
            generation_id,
            GenerationStatus.COMPLETED,
            completed_at=_utc(completed_at or datetime.now(UTC)),
        )

    def mark_failed(
        self,
        generation_id: UUID,
        error_code: str,
        error_summary: str,
        completed_at: datetime | None = None,
    ) -> Generation:
        return self._transition(
            generation_id,
            GenerationStatus.FAILED,
            completed_at=_utc(completed_at or datetime.now(UTC)),
            error_code=error_code,
            error_summary=error_summary[:1000],
        )

    def get_by_id(self, generation_id: UUID) -> Generation | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(GenerationModel, str(generation_id))
                return _generation_domain(row) if row is not None else None
        except (SQLAlchemyError, SnapshotError) as exc:
            raise GenerationRepositoryError("generation could not be read") from exc

    def get_by_prompt_id(self, prompt_id: str) -> Generation | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(GenerationModel).where(GenerationModel.comfy_prompt_id == prompt_id)
                )
                return _generation_domain(row) if row is not None else None
        except (SQLAlchemyError, SnapshotError) as exc:
            raise GenerationRepositoryError("generation could not be read") from exc

    def list_completed_optional_artifact_repairs(
        self,
        limit: int = 50,
        *,
        after_completed_at: datetime | None = None,
        after_generation_id: UUID | None = None,
    ) -> tuple[OptionalArtifactRepairCandidate, ...]:
        """Find completed pairs with a primary image and a missing optional artifact."""

        bounded_limit = min(max(1, limit), 100)
        primary_exists = exists().where(
            GenerationArtifactModel.generation_id == GenerationModel.id,
            GenerationArtifactModel.artifact_type == ArtifactType.IMAGE.value,
        )
        metadata_exists = exists().where(
            GenerationArtifactModel.generation_id == GenerationModel.id,
            GenerationArtifactModel.artifact_type == ArtifactType.METADATA.value,
        )
        thumbnail_exists = exists().where(
            GenerationArtifactModel.generation_id == GenerationModel.id,
            GenerationArtifactModel.artifact_type == ArtifactType.THUMBNAIL.value,
        )
        try:
            with session_scope(self._session_factory) as session:
                statement = (
                    select(GenerationModel.id, GenerationModel.completed_at)
                    .join(
                        GenerationJobModel,
                        GenerationJobModel.generation_id == GenerationModel.id,
                    )
                    .where(
                        GenerationModel.status == GenerationStatus.COMPLETED.value,
                        GenerationJobModel.status == GenerationStatus.COMPLETED.value,
                        GenerationModel.completed_at.is_not(None),
                        GenerationJobModel.completed_at.is_not(None),
                        GenerationModel.completed_at == GenerationJobModel.completed_at,
                        primary_exists,
                        or_(~metadata_exists, ~thumbnail_exists),
                    )
                )
                if after_completed_at is not None and after_generation_id is not None:
                    cursor_time = _utc(after_completed_at)
                    statement = statement.where(
                        or_(
                            GenerationModel.completed_at > cursor_time,
                            and_(
                                GenerationModel.completed_at == cursor_time,
                                GenerationModel.id > str(after_generation_id),
                            ),
                        )
                    )
                elif after_completed_at is not None or after_generation_id is not None:
                    raise ValueError("optional artifact repair cursor must be complete")
                rows = session.execute(
                    statement.order_by(
                        GenerationModel.completed_at.asc(), GenerationModel.id.asc()
                    ).limit(bounded_limit)
                ).all()
                return tuple(
                    OptionalArtifactRepairCandidate(
                        generation_id=UUID(row.id),
                        completed_at=_utc(row.completed_at) or datetime.now(UTC),
                    )
                    for row in rows
                )
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationRepositoryError(
                "completed optional artifact repairs could not be read"
            ) from exc

    def list_history(self, query: GenerationHistoryFilter) -> GenerationHistoryPage:
        offset = max(0, query.offset)
        limit = min(max(1, query.limit), 100)
        try:
            with session_scope(self._session_factory) as session:
                normalized = _as_history_query(query)
                statement = select(GenerationModel)
                count_statement = select(func.count()).select_from(GenerationModel)
                filters = _history_filters(normalized)
                if filters:
                    statement = statement.where(*filters)
                    count_statement = count_statement.where(*filters)
                ordering = _history_ordering(normalized.sort)
                rows = session.scalars(
                    statement.order_by(*ordering).offset(offset).limit(limit)
                ).all()
                total = int(session.scalar(count_statement) or 0)
                generations = tuple(_generation_domain(row) for row in rows)
                return GenerationHistoryPage(
                    generations=generations,
                    page=offset // limit + 1,
                    page_size=limit,
                    total_count=total,
                    has_next=offset + len(generations) < total,
                )
        except (SQLAlchemyError, SnapshotError) as exc:
            raise GenerationRepositoryError("generation history could not be read") from exc

    def set_favorite(self, generation_id: UUID, favorite: bool) -> Generation:
        try:
            with session_scope(self._session_factory) as session:
                row = _require_generation(session, generation_id)
                row.favorite = favorite
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _generation_domain(row)
        except GenerationRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("favorite could not be saved") from exc

    def update_note(self, generation_id: UUID, note: str | None) -> Generation:
        normalized = note.strip() if note is not None else None
        if normalized == "":
            normalized = None
        if normalized is not None and len(normalized) > 2000:
            raise GenerationRepositoryError("note is too long")
        try:
            with session_scope(self._session_factory) as session:
                row = _require_generation(session, generation_id)
                row.user_note = normalized
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _generation_domain(row)
        except GenerationRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("note could not be saved") from exc

    def _transition(
        self,
        generation_id: UUID,
        target: GenerationStatus,
        *,
        prompt_id: str | None = None,
        completed_at: datetime | None = None,
        error_code: str | None = None,
        error_summary: str | None = None,
    ) -> Generation:
        try:
            with session_scope(self._session_factory) as session:
                row = _require_generation(session, generation_id)
                current = GenerationStatus(row.status)
                if not is_valid_status_transition(current, target):
                    raise GenerationRepositoryError("invalid generation status transition")
                if prompt_id is not None:
                    row.comfy_prompt_id = prompt_id
                row.status = target.value
                if target is GenerationStatus.RUNNING and row.started_at is None:
                    row.started_at = datetime.now(UTC)
                if completed_at is not None:
                    row.completed_at = completed_at
                if error_code is not None:
                    row.error_code = error_code
                    row.error_summary = error_summary
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _generation_domain(row)
        except GenerationRepositoryError:
            raise
        except (SQLAlchemyError, ValueError, SnapshotError) as exc:
            raise GenerationRepositoryError("generation status could not be updated") from exc


class GenerationQueueRepositoryProtocol(Protocol):
    """ComfyUI promptをGeneration/Jobへ原子的に関連付ける契約。"""

    def mark_queued(self, generation_id: UUID, job_id: UUID, prompt_id: str) -> None: ...


class GenerationQueueRepository(GenerationQueueRepositoryProtocol):
    """Generation/Jobのqueued状態とprompt IDを原子的に保存する。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def mark_queued(self, generation_id: UUID, job_id: UUID, prompt_id: str) -> None:
        normalized_prompt_id = prompt_id.strip()
        if not normalized_prompt_id:
            raise GenerationRepositoryError("prompt id must not be empty")
        try:
            with session_scope(self._session_factory) as session:
                generation = _require_generation(session, generation_id)
                job = _require_job(session, job_id)
                if job.generation_id != str(generation_id):
                    raise GenerationRepositoryError("job generation does not match")

                generation_status = GenerationStatus(generation.status)
                job_status = GenerationStatus(job.status)
                generation_prompt_id = generation.comfy_prompt_id
                job_prompt_id = job.comfy_prompt_id

                if (
                    generation_status is GenerationStatus.QUEUED
                    and job_status is GenerationStatus.QUEUED
                    and generation_prompt_id == normalized_prompt_id
                    and job_prompt_id == normalized_prompt_id
                ):
                    return
                if generation_status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED,
                } or job_status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED,
                }:
                    raise GenerationRepositoryError("cannot queue a terminal generation or job")
                if (
                    generation_status is not GenerationStatus.PENDING
                    or job_status is not GenerationStatus.PENDING
                ):
                    raise GenerationRepositoryError("generation and job are not both pending")
                if generation_prompt_id is not None or job_prompt_id is not None:
                    raise GenerationRepositoryError("prompt ID is already assigned inconsistently")

                _mark_generation_queued(generation, normalized_prompt_id)
                _mark_job_queued(job, normalized_prompt_id)
                session.flush()
        except GenerationRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError) as exc:
            raise GenerationRepositoryError("prompt ID could not be persisted") from exc


class GenerationFailureRepositoryProtocol(Protocol):
    """Generation/Jobのfailed状態を原子的に保存する契約。"""

    def fail_generation(
        self,
        generation_id: UUID,
        job_id: UUID,
        *,
        error_code: str,
        error_summary: str,
        failed_at: datetime,
    ) -> None: ...


class GenerationFailureRepository(GenerationFailureRepositoryProtocol):
    """両方のfailed状態を1つのSQLiteトランザクションで保存する。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def fail_generation(
        self,
        generation_id: UUID,
        job_id: UUID,
        *,
        error_code: str,
        error_summary: str,
        failed_at: datetime,
    ) -> None:
        if re.fullmatch(r"[a-z0-9_]+", error_code) is None:
            raise GenerationRepositoryError("error code is invalid")
        normalized_summary = error_summary[:1000]
        timestamp = _utc(failed_at)
        if timestamp is None:
            raise GenerationRepositoryError("failed time is required")
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
                if (
                    generation_status is GenerationStatus.FAILED
                    and job_status is GenerationStatus.FAILED
                ):
                    if (
                        generation.error_code == error_code
                        and generation.error_summary == normalized_summary
                        and job.error_code == error_code
                        and job.error_summary == normalized_summary
                    ):
                        return
                    raise GenerationRepositoryError("failure information is already finalized")
                if GenerationStatus.FAILED in {generation_status, job_status}:
                    raise GenerationRepositoryError(
                        "generation and job failure states are inconsistent"
                    )
                if generation_status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.CANCELLED,
                } or job_status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.CANCELLED,
                }:
                    raise GenerationRepositoryError("cannot fail a terminal generation or job")
                if generation_status not in {
                    GenerationStatus.PENDING,
                    GenerationStatus.QUEUED,
                    GenerationStatus.RUNNING,
                } or job_status not in {
                    GenerationStatus.PENDING,
                    GenerationStatus.QUEUED,
                    GenerationStatus.RUNNING,
                }:
                    raise GenerationRepositoryError("generation and job states are invalid")

                _mark_generation_failed(
                    generation,
                    error_code=error_code,
                    error_summary=normalized_summary,
                    failed_at=timestamp,
                )
                _mark_job_failed(
                    job,
                    error_code=error_code,
                    error_summary=normalized_summary,
                    failed_at=timestamp,
                )
                session.flush()
        except GenerationRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationRepositoryError("generation failure could not be persisted") from exc


class GenerationCancellationRepositoryProtocol(Protocol):
    """Persist a Generation/Job cancellation without requiring a queue row."""

    def cancel_generation(
        self,
        generation_id: UUID,
        job_id: UUID,
        *,
        cancelled_at: datetime,
        error_code: str,
        error_summary: str,
    ) -> None: ...


class GenerationCancellationRepository(GenerationCancellationRepositoryProtocol):
    """Atomically persist an interruption as a terminal cancelled pair."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def cancel_generation(
        self,
        generation_id: UUID,
        job_id: UUID,
        *,
        cancelled_at: datetime,
        error_code: str,
        error_summary: str,
    ) -> None:
        if re.fullmatch(r"[a-z0-9_]+", error_code) is None:
            raise GenerationRepositoryError("error code is invalid")
        timestamp = _utc(cancelled_at)
        if timestamp is None:
            raise GenerationRepositoryError("cancelled time is required")
        normalized_summary = error_summary[:1000]
        try:
            with session_scope(self._session_factory) as session:
                generation = _require_generation(session, generation_id)
                job = _require_job(session, job_id)
                if job.generation_id != str(generation_id):
                    raise GenerationRepositoryError("job generation does not match")
                generation_status = GenerationStatus(generation.status)
                job_status = GenerationStatus(job.status)
                if GenerationStatus.COMPLETED in {generation_status, job_status} or (
                    GenerationStatus.FAILED in {generation_status, job_status}
                ):
                    return
                if (
                    generation_status is GenerationStatus.CANCELLED
                    and job_status is GenerationStatus.CANCELLED
                ):
                    return
                allowed = {
                    GenerationStatus.PENDING,
                    GenerationStatus.QUEUED,
                    GenerationStatus.RUNNING,
                    GenerationStatus.CANCELLED,
                }
                if generation_status not in allowed or job_status not in allowed:
                    raise GenerationRepositoryError("generation and job states are invalid")
                generation.status = GenerationStatus.CANCELLED.value
                generation.completed_at = generation.completed_at or timestamp
                generation.error_code = error_code
                generation.error_summary = normalized_summary
                generation.updated_at = timestamp
                job.status = GenerationStatus.CANCELLED.value
                job.cancelled_at = job.cancelled_at or timestamp
                job.completed_at = job.completed_at or timestamp
                job.error_code = error_code
                job.error_summary = normalized_summary
                job.worker_id = None
                job.claimed_at = None
                job.lease_expires_at = None
                job.updated_at = timestamp
                session.flush()
        except GenerationRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValueError) as exc:
            raise GenerationRepositoryError(
                "generation cancellation could not be persisted"
            ) from exc


class GenerationArtifactRepositoryProtocol(Protocol):
    def add(self, artifact: GenerationArtifact) -> GenerationArtifact: ...

    def list_by_generation(self, generation_id: UUID) -> tuple[GenerationArtifact, ...]: ...

    def get_primary_image(self, generation_id: UUID) -> GenerationArtifact | None: ...


class GenerationCompletionRepositoryProtocol(Protocol):
    def complete_generation(
        self,
        generation_id: UUID,
        job_id: UUID,
        image_artifact: GenerationArtifact,
        completed_at: datetime | None = None,
    ) -> None: ...

    def complete_existing_artifact(
        self,
        generation_id: UUID,
        job_id: UUID,
        completed_at: datetime | None = None,
    ) -> None: ...


class GenerationArtifactRepository(GenerationArtifactRepositoryProtocol):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add(self, artifact: GenerationArtifact) -> GenerationArtifact:
        if _unsafe_relative_path(artifact.local_path):
            raise GenerationRepositoryError("artifact path must be relative")
        try:
            with session_scope(self._session_factory) as session:
                if session.get(GenerationModel, str(artifact.generation_id)) is None:
                    raise GenerationRepositoryError("generation was not found")
                existing = session.scalar(
                    select(GenerationArtifactModel).where(
                        GenerationArtifactModel.generation_id == str(artifact.generation_id),
                        GenerationArtifactModel.artifact_type == artifact.artifact_type.value,
                        GenerationArtifactModel.sha256 == artifact.sha256,
                    )
                )
                if existing is not None:
                    return _artifact_domain(existing)
                row = GenerationArtifactModel(
                    id=str(artifact.id),
                    generation_id=str(artifact.generation_id),
                    artifact_type=artifact.artifact_type.value,
                    local_path=artifact.local_path,
                    sha256=artifact.sha256,
                    size_bytes=artifact.size_bytes,
                    width=artifact.width,
                    height=artifact.height,
                    mime_type=artifact.mime_type,
                    created_at=_utc(artifact.created_at),
                )
                session.add(row)
                session.flush()
                return _artifact_domain(row)
        except GenerationRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("artifact could not be saved") from exc

    def list_by_generation(self, generation_id: UUID) -> tuple[GenerationArtifact, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(GenerationArtifactModel)
                    .where(GenerationArtifactModel.generation_id == str(generation_id))
                    .order_by(GenerationArtifactModel.created_at.asc())
                ).all()
                return tuple(_artifact_domain(row) for row in rows)
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("artifacts could not be read") from exc

    def get_primary_image(self, generation_id: UUID) -> GenerationArtifact | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(GenerationArtifactModel)
                    .where(
                        GenerationArtifactModel.generation_id == str(generation_id),
                        GenerationArtifactModel.artifact_type == ArtifactType.IMAGE.value,
                    )
                    .order_by(GenerationArtifactModel.created_at.asc())
                )
                return _artifact_domain(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("artifact could not be read") from exc


class GenerationCompletionRepository(GenerationCompletionRepositoryProtocol):
    """Persist the primary artifact and both completion states atomically."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def complete_generation(
        self,
        generation_id: UUID,
        job_id: UUID,
        image_artifact: GenerationArtifact,
        completed_at: datetime | None = None,
    ) -> None:
        if image_artifact.generation_id != generation_id:
            raise GenerationRepositoryError("image artifact generation does not match")
        if _unsafe_relative_path(image_artifact.local_path):
            raise GenerationRepositoryError("artifact path must be relative")
        timestamp = _utc(completed_at or datetime.now(UTC)) or datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                generation = _require_generation(session, generation_id)
                job = _require_job(session, job_id)
                if job.generation_id != str(generation_id):
                    raise GenerationRepositoryError("job generation does not match")
                self._add_or_validate_primary_artifact(session, image_artifact)
                _mark_generation_completed(generation, timestamp)
                _mark_job_completed(job, timestamp)
                session.flush()
        except GenerationRepositoryError:
            raise
        except (SQLAlchemyError, ValueError, SnapshotError) as exc:
            raise GenerationRepositoryError("generation completion could not be persisted") from exc

    def complete_existing_artifact(
        self,
        generation_id: UUID,
        job_id: UUID,
        completed_at: datetime | None = None,
    ) -> None:
        timestamp = _utc(completed_at or datetime.now(UTC)) or datetime.now(UTC)
        try:
            with session_scope(self._session_factory) as session:
                generation = _require_generation(session, generation_id)
                job = _require_job(session, job_id)
                if job.generation_id != str(generation_id):
                    raise GenerationRepositoryError("job generation does not match")
                existing = session.scalar(
                    select(GenerationArtifactModel).where(
                        GenerationArtifactModel.generation_id == str(generation_id),
                        GenerationArtifactModel.artifact_type == ArtifactType.IMAGE.value,
                    )
                )
                if existing is None:
                    raise GenerationRepositoryError("primary image artifact was not found")
                _mark_generation_completed(generation, timestamp)
                _mark_job_completed(job, timestamp)
                session.flush()
        except GenerationRepositoryError:
            raise
        except (SQLAlchemyError, ValueError, SnapshotError) as exc:
            raise GenerationRepositoryError(
                "existing generation completion could not be persisted"
            ) from exc

    @staticmethod
    def _add_or_validate_primary_artifact(session: Session, artifact: GenerationArtifact) -> None:
        generation = session.get(GenerationModel, str(artifact.generation_id))
        if generation is None:
            raise GenerationRepositoryError("generation was not found")
        existing = session.scalar(
            select(GenerationArtifactModel).where(
                GenerationArtifactModel.generation_id == str(artifact.generation_id),
                GenerationArtifactModel.artifact_type == ArtifactType.IMAGE.value,
            )
        )
        if existing is not None:
            if existing.sha256 != artifact.sha256 or existing.local_path != artifact.local_path:
                raise GenerationRepositoryError("primary image artifact does not match")
            return
        session.add(
            GenerationArtifactModel(
                id=str(artifact.id),
                generation_id=str(artifact.generation_id),
                artifact_type=ArtifactType.IMAGE.value,
                local_path=artifact.local_path,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                width=artifact.width,
                height=artifact.height,
                mime_type=artifact.mime_type,
                created_at=_utc(artifact.created_at),
            )
        )


def _require_job(session: Session, job_id: UUID) -> GenerationJobModel:
    row = session.get(GenerationJobModel, str(job_id))
    if row is None:
        raise GenerationRepositoryError("job was not found")
    return row


def _mark_generation_queued(row: GenerationModel, prompt_id: str) -> None:
    current = GenerationStatus(row.status)
    if not is_valid_status_transition(current, GenerationStatus.QUEUED):
        raise GenerationRepositoryError("invalid generation queue transition")
    row.comfy_prompt_id = prompt_id
    row.status = GenerationStatus.QUEUED.value
    row.updated_at = datetime.now(UTC)


def _mark_job_queued(row: GenerationJobModel, prompt_id: str) -> None:
    current = GenerationStatus(row.status)
    if not is_valid_status_transition(current, GenerationStatus.QUEUED):
        raise GenerationRepositoryError("invalid job queue transition")
    row.comfy_prompt_id = prompt_id
    row.status = GenerationStatus.QUEUED.value
    row.updated_at = datetime.now(UTC)


def _mark_generation_failed(
    row: GenerationModel,
    *,
    error_code: str,
    error_summary: str,
    failed_at: datetime,
) -> None:
    current = GenerationStatus(row.status)
    if not is_valid_status_transition(current, GenerationStatus.FAILED):
        raise GenerationRepositoryError("invalid generation failure transition")
    row.status = GenerationStatus.FAILED.value
    row.completed_at = failed_at
    row.error_code = error_code
    row.error_summary = error_summary
    row.updated_at = datetime.now(UTC)


def _mark_job_failed(
    row: GenerationJobModel,
    *,
    error_code: str,
    error_summary: str,
    failed_at: datetime,
) -> None:
    current = GenerationStatus(row.status)
    if not is_valid_status_transition(current, GenerationStatus.FAILED):
        raise GenerationRepositoryError("invalid job failure transition")
    row.status = GenerationStatus.FAILED.value
    row.completed_at = failed_at
    row.error_code = error_code
    row.error_summary = error_summary
    row.updated_at = datetime.now(UTC)


def _mark_generation_completed(row: GenerationModel, completed_at: datetime) -> None:
    current = GenerationStatus(row.status)
    if not is_valid_status_transition(current, GenerationStatus.COMPLETED):
        raise GenerationRepositoryError("invalid generation completion transition")
    row.status = GenerationStatus.COMPLETED.value
    row.completed_at = completed_at
    row.updated_at = datetime.now(UTC)


def _mark_job_completed(row: GenerationJobModel, completed_at: datetime) -> None:
    current = GenerationStatus(row.status)
    if not is_valid_status_transition(current, GenerationStatus.COMPLETED):
        raise GenerationRepositoryError("invalid job completion transition")
    row.status = GenerationStatus.COMPLETED.value
    row.completed_at = completed_at
    row.updated_at = datetime.now(UTC)


class GenerationJobRepositoryProtocol(Protocol):
    def create(self, job: GenerationJob) -> GenerationJob: ...

    def get_by_generation(self, generation_id: UUID) -> GenerationJob | None: ...

    def update_prompt_id(self, job_id: UUID, prompt_id: str) -> GenerationJob: ...

    def update_progress(
        self, job_id: UUID, value: int | None, maximum: int | None, node: str | None
    ) -> GenerationJob: ...

    def mark_completed(
        self, job_id: UUID, completed_at: datetime | None = None
    ) -> GenerationJob: ...

    def mark_failed(self, job_id: UUID, error_code: str, error_summary: str) -> GenerationJob: ...

    def list_recoverable(self, limit: int = 50) -> tuple[GenerationJob, ...]: ...


class GenerationJobRepository(GenerationJobRepositoryProtocol):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, job: GenerationJob) -> GenerationJob:
        timestamp = _utc(job.created_at or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                if session.get(GenerationModel, str(job.generation_id)) is None:
                    raise GenerationRepositoryError("generation was not found")
                row = GenerationJobModel(
                    id=str(job.id),
                    generation_id=str(job.generation_id),
                    status=job.status.value,
                    comfy_prompt_id=job.prompt_id,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(row)
                session.flush()
                return _job_domain(row)
        except GenerationRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("job could not be created") from exc

    def get_by_generation(self, generation_id: UUID) -> GenerationJob | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(GenerationJobModel).where(
                        GenerationJobModel.generation_id == str(generation_id)
                    )
                )
                return _job_domain(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("job could not be read") from exc

    def update_prompt_id(self, job_id: UUID, prompt_id: str) -> GenerationJob:
        return self._update(job_id, prompt_id=prompt_id, status=GenerationStatus.QUEUED)

    def update_progress(
        self, job_id: UUID, value: int | None, maximum: int | None, node: str | None
    ) -> GenerationJob:
        return self._update(
            job_id,
            value=value,
            maximum=maximum,
            node=node,
            status=GenerationStatus.RUNNING,
        )

    def mark_completed(self, job_id: UUID, completed_at: datetime | None = None) -> GenerationJob:
        return self._update(
            job_id,
            status=GenerationStatus.COMPLETED,
            completed_at=_utc(completed_at or datetime.now(UTC)),
        )

    def mark_failed(self, job_id: UUID, error_code: str, error_summary: str) -> GenerationJob:
        return self._update(
            job_id,
            status=GenerationStatus.FAILED,
            completed_at=datetime.now(UTC),
            error_code=error_code,
            error_summary=error_summary[:1000],
        )

    def list_recoverable(self, limit: int = 50) -> tuple[GenerationJob, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(GenerationJobModel)
                    .where(
                        GenerationJobModel.status.in_(
                            [
                                GenerationStatus.PENDING.value,
                                GenerationStatus.QUEUED.value,
                                GenerationStatus.RUNNING.value,
                            ]
                        )
                    )
                    .order_by(GenerationJobModel.created_at.asc())
                    .limit(min(max(1, limit), 100))
                ).all()
                return tuple(_job_domain(row) for row in rows)
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("recoverable jobs could not be read") from exc

    def _update(self, job_id: UUID, **values: object) -> GenerationJob:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(GenerationJobModel, str(job_id))
                if row is None:
                    raise GenerationRepositoryError("job was not found")
                status = values.get("status")
                if isinstance(status, GenerationStatus):
                    current = GenerationStatus(row.status)
                    if not is_valid_status_transition(current, status):
                        raise GenerationRepositoryError("invalid job status transition")
                    row.status = status.value
                    if status is GenerationStatus.RUNNING and row.started_at is None:
                        row.started_at = datetime.now(UTC)
                if "prompt_id" in values:
                    row.comfy_prompt_id = values["prompt_id"]
                if "value" in values:
                    row.progress_value = values["value"]
                    row.progress_maximum = values["maximum"]
                    row.current_node = values["node"]
                if "completed_at" in values:
                    row.completed_at = values["completed_at"]
                if "error_code" in values:
                    row.error_code = values["error_code"]
                    row.error_summary = values["error_summary"]
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _job_domain(row)
        except GenerationRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise GenerationRepositoryError("job could not be updated") from exc


def _require_generation(session: Session, generation_id: UUID) -> GenerationModel:
    row = session.get(GenerationModel, str(generation_id))
    if row is None:
        raise GenerationRepositoryError("generation was not found")
    return row


def _generation_domain(row: GenerationModel) -> Generation:
    try:
        snapshot = GenerationSettingsSnapshot.from_json(row.settings_snapshot_json)
        return Generation(
            id=UUID(row.id),
            kind=GenerationKind(row.kind),
            status=GenerationStatus(row.status),
            parent_generation_id=(
                UUID(row.parent_generation_id) if row.parent_generation_id is not None else None
            ),
            settings_snapshot=snapshot,
            workflow_template_id=row.workflow_template_id,
            workflow_template_version=row.workflow_template_version,
            comfy_prompt_id=row.comfy_prompt_id,
            favorite=row.favorite,
            user_note=row.user_note,
            error_code=row.error_code,
            error_summary=row.error_summary,
            retry_of_generation_id=(
                UUID(row.retry_of_generation_id) if row.retry_of_generation_id is not None else None
            ),
            retry_attempt=row.retry_attempt,
            created_at=_utc(row.created_at) or datetime.now(UTC),
            started_at=_utc(row.started_at),
            completed_at=_utc(row.completed_at),
            updated_at=_utc(row.updated_at) or datetime.now(UTC),
        )
    except (SnapshotError, ValueError) as exc:
        raise GenerationRepositoryError("stored generation data is invalid") from exc


def _artifact_domain(row: GenerationArtifactModel) -> GenerationArtifact:
    return GenerationArtifact(
        id=UUID(row.id),
        generation_id=UUID(row.generation_id),
        artifact_type=ArtifactType(row.artifact_type),
        local_path=row.local_path,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        width=row.width,
        height=row.height,
        mime_type=row.mime_type,
        created_at=_utc(row.created_at) or datetime.now(UTC),
    )


def _job_domain(row: GenerationJobModel) -> GenerationJob:
    return GenerationJob(
        generation_id=UUID(row.generation_id),
        status=GenerationStatus(row.status),
        id=UUID(row.id),
        prompt_id=row.comfy_prompt_id,
        progress_value=row.progress_value,
        progress_maximum=row.progress_maximum,
        current_node=row.current_node,
        created_at=_utc(row.created_at),
        started_at=_utc(row.started_at),
        completed_at=_utc(row.completed_at),
        updated_at=_utc(row.updated_at),
        error_code=row.error_code,
        error_summary=row.error_summary,
        error_message=row.error_summary,
        worker_id=row.worker_id,
        claimed_at=_utc(row.claimed_at),
        lease_expires_at=_utc(row.lease_expires_at),
        cancel_requested_at=_utc(row.cancel_requested_at),
        cancelled_at=_utc(row.cancelled_at),
    )


def _as_history_query(
    query: GenerationHistoryFilter | GenerationHistoryQuery,
) -> GenerationHistoryQuery:
    if isinstance(query, GenerationHistoryQuery):
        return query
    return GenerationHistoryQuery(
        date=query.date,
        status=query.status,
        favorite=query.favorite,
        kind=query.kind,
        offset=query.offset,
        limit=query.limit,
        start_utc=query.start_utc,
        end_utc=query.end_utc,
    )


def _history_filters(query: GenerationHistoryQuery) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    statuses = query.statuses or ((query.status,) if query.status is not None else ())
    kinds = query.kinds or ((query.kind,) if query.kind is not None else ())
    if statuses:
        filters.append(GenerationModel.status.in_([value.value for value in statuses]))
    if query.favorite_only or query.favorite is True:
        filters.append(GenerationModel.favorite.is_(True))
    if query.favorite is False:
        filters.append(GenerationModel.favorite.is_(False))
    if kinds:
        filters.append(GenerationModel.kind.in_([value.value for value in kinds]))
    if query.checkpoint_names:
        filters.append(GenerationModel.checkpoint_name.in_(query.checkpoint_names))
    if query.vae_names:
        filters.append(GenerationModel.vae_name.in_(query.vae_names))
    if query.seed is not None:
        filters.append(GenerationModel.seed == query.seed)
    if query.width is not None:
        filters.append(GenerationModel.width == query.width)
    if query.height is not None:
        filters.append(GenerationModel.height == query.height)
    if query.error_codes:
        filters.append(GenerationModel.error_code.in_(query.error_codes))
    if query.parent_generation_id is not None:
        filters.append(GenerationModel.parent_generation_id == str(query.parent_generation_id))
    if query.lora_names:
        if query.lora_search_mode.value == "all":
            filters.extend(
                cast(
                    ColumnElement[bool],
                    exists().where(
                        and_(
                            GenerationLoraModel.generation_id == GenerationModel.id,
                            GenerationLoraModel.lora_name == lora_name,
                        )
                    ),
                )
                for lora_name in query.lora_names
            )
        else:
            filters.append(
                exists().where(
                    and_(
                        GenerationLoraModel.generation_id == GenerationModel.id,
                        GenerationLoraModel.lora_name.in_(query.lora_names),
                    )
                )
            )
    if query.text is not None:
        pattern = f"%{_escape_like(query.text.lower())}%"
        text_columns = (
            GenerationModel.positive_prompt_search,
            GenerationModel.negative_prompt_search,
            GenerationModel.user_note,
            GenerationModel.checkpoint_name,
            GenerationModel.vae_name,
            GenerationModel.error_summary,
        )
        text_match: list[ColumnElement[bool]] = [
            func.lower(column).like(pattern, escape="\\") for column in text_columns
        ]
        text_match.append(
            cast(
                ColumnElement[bool],
                exists().where(
                    and_(
                        GenerationLoraModel.generation_id == GenerationModel.id,
                        func.lower(GenerationLoraModel.lora_name).like(pattern, escape="\\"),
                    )
                ),
            )
        )
        filters.append(or_(*text_match))
    start_utc = query.date_from or query.start_utc
    end_utc = query.date_to or query.end_utc
    if start_utc is not None:
        filters.append(GenerationModel.created_at >= _utc(start_utc))
    if end_utc is not None:
        filters.append(GenerationModel.created_at < _utc(end_utc))
    return filters


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _history_ordering(sort: GenerationHistorySort) -> tuple[ColumnElement[object], ...]:
    if sort is GenerationHistorySort.OLDEST:
        return cast(
            tuple[ColumnElement[object], ...],
            (GenerationModel.created_at.asc(), GenerationModel.id.asc()),
        )
    if sort is GenerationHistorySort.SEED_ASC:
        return cast(
            tuple[ColumnElement[object], ...],
            (GenerationModel.seed.asc(), GenerationModel.id.asc()),
        )
    if sort is GenerationHistorySort.SEED_DESC:
        return cast(
            tuple[ColumnElement[object], ...],
            (GenerationModel.seed.desc(), GenerationModel.id.desc()),
        )
    if sort is GenerationHistorySort.RESOLUTION_DESC:
        return cast(
            tuple[ColumnElement[object], ...],
            ((GenerationModel.width * GenerationModel.height).desc(), GenerationModel.id.desc()),
        )
    if sort is GenerationHistorySort.RECENTLY_COMPLETED:
        return cast(
            tuple[ColumnElement[object], ...],
            (GenerationModel.completed_at.desc(), GenerationModel.id.desc()),
        )
    return cast(
        tuple[ColumnElement[object], ...],
        (GenerationModel.created_at.desc(), GenerationModel.id.desc()),
    )


def _unsafe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        not normalized
        or normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    )


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
