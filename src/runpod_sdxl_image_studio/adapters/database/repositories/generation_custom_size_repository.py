"""SQLite repository for persisted custom generation sizes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import GenerationCustomSizeModel
from runpod_sdxl_image_studio.domain.generation_custom_size import GenerationCustomSize


class GenerationCustomSizeRepositoryError(RuntimeError):
    """A safe persistence error for custom size preferences."""


class GenerationCustomSizeRepositoryProtocol(Protocol):
    def list(self, *, limit: int = 100) -> tuple[GenerationCustomSize, ...]: ...

    def get_by_dimensions(self, width: int, height: int) -> GenerationCustomSize | None: ...

    def add(
        self,
        width: int,
        height: int,
        *,
        created_at: datetime | None = None,
    ) -> GenerationCustomSize: ...

    def delete(self, size_id: UUID) -> None: ...


class GenerationCustomSizeRepository(GenerationCustomSizeRepositoryProtocol):
    """Keep the unique(width, height) constraint as the concurrency guard."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def list(self, *, limit: int = 100) -> tuple[GenerationCustomSize, ...]:
        normalized_limit = min(max(1, limit), 500)
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(GenerationCustomSizeModel)
                    .order_by(
                        GenerationCustomSizeModel.created_at.asc(),
                        GenerationCustomSizeModel.id.asc(),
                    )
                    .limit(normalized_limit)
                ).all()
                return tuple(_to_domain(row) for row in rows)
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationCustomSizeRepositoryError(
                "custom generation sizes could not be read"
            ) from exc

    def get_by_dimensions(self, width: int, height: int) -> GenerationCustomSize | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(GenerationCustomSizeModel).where(
                        GenerationCustomSizeModel.width == width,
                        GenerationCustomSizeModel.height == height,
                    )
                )
                return _to_domain(row) if row is not None else None
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationCustomSizeRepositoryError(
                "custom generation size could not be read"
            ) from exc

    def add(
        self,
        width: int,
        height: int,
        *,
        created_at: datetime | None = None,
    ) -> GenerationCustomSize:
        identifier = uuid4()
        timestamp = _utc(created_at or datetime.now(UTC))
        try:
            with session_scope(self._session_factory) as session:
                row = GenerationCustomSizeModel(
                    id=str(identifier),
                    width=width,
                    height=height,
                    created_at=timestamp,
                )
                session.add(row)
                session.flush()
                return _to_domain(row)
        except IntegrityError:
            existing = self.get_by_dimensions(width, height)
            if existing is not None:
                return existing
            raise GenerationCustomSizeRepositoryError(
                "custom generation size could not be created"
            ) from None
        except (SQLAlchemyError, ValueError) as exc:
            raise GenerationCustomSizeRepositoryError(
                "custom generation size could not be created"
            ) from exc

    def delete(self, size_id: UUID) -> None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(GenerationCustomSizeModel, str(size_id))
                if row is not None:
                    session.delete(row)
        except SQLAlchemyError as exc:
            raise GenerationCustomSizeRepositoryError(
                "custom generation size could not be deleted"
            ) from exc


def _to_domain(row: GenerationCustomSizeModel) -> GenerationCustomSize:
    return GenerationCustomSize(
        id=UUID(row.id),
        width=row.width,
        height=row.height,
        created_at=_utc(row.created_at),
    )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "GenerationCustomSizeRepository",
    "GenerationCustomSizeRepositoryError",
    "GenerationCustomSizeRepositoryProtocol",
]
