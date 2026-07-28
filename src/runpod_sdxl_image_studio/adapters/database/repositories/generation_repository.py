"""Repositories for persisted generations, artifacts, and jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import (
    GenerationArtifactModel,
    GenerationJobModel,
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
)
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
                    workflow_template_id=snapshot.workflow_template_id,
                    workflow_template_version=snapshot.workflow_template_version,
                    favorite=False,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(row)
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

    def list_history(self, query: GenerationHistoryFilter) -> GenerationHistoryPage:
        offset = max(0, query.offset)
        limit = min(max(1, query.limit), 100)
        try:
            with session_scope(self._session_factory) as session:
                statement = select(GenerationModel)
                count_statement = select(func.count()).select_from(GenerationModel)
                filters = _history_filters(query)
                if filters:
                    statement = statement.where(*filters)
                    count_statement = count_statement.where(*filters)
                rows = session.scalars(
                    statement.order_by(GenerationModel.created_at.desc(), GenerationModel.id.desc())
                    .offset(offset)
                    .limit(limit)
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
                    row.comfy_prompt_id = values["prompt_id"]  # type: ignore[assignment]
                if "value" in values:
                    row.progress_value = values["value"]  # type: ignore[assignment]
                    row.progress_maximum = values["maximum"]  # type: ignore[assignment]
                    row.current_node = values["node"]  # type: ignore[assignment]
                if "completed_at" in values:
                    row.completed_at = values["completed_at"]  # type: ignore[assignment]
                if "error_code" in values:
                    row.error_code = values["error_code"]  # type: ignore[assignment]
                    row.error_summary = values["error_summary"]  # type: ignore[assignment]
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
    )


def _history_filters(query: GenerationHistoryFilter) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if query.status is not None:
        filters.append(GenerationModel.status == query.status.value)
    if query.favorite is not None:
        filters.append(GenerationModel.favorite.is_(query.favorite))
    if query.kind is not None:
        filters.append(GenerationModel.kind == query.kind.value)
    if query.start_utc is not None:
        filters.append(GenerationModel.created_at >= _utc(query.start_utc))
    if query.end_utc is not None:
        filters.append(GenerationModel.created_at < _utc(query.end_utc))
    return filters


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
