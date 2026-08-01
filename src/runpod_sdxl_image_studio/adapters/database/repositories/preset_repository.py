"""SQLAlchemy repository for typed presets."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import PresetModel
from runpod_sdxl_image_studio.domain.preset import Preset
from runpod_sdxl_image_studio.domain.preset_payload import (
    PresetKind,
    PresetPayloadError,
    parse_payload,
)


class PresetRepositoryError(RuntimeError):
    """Preset永続化で安全に扱えるエラー。"""


class PresetRepositoryProtocol(Protocol):
    """Preset Repositoryのアプリケーション境界。"""

    def create(self, preset: Preset) -> Preset: ...

    def update(self, preset: Preset) -> Preset: ...

    def get_by_id(self, preset_id: UUID) -> Preset | None: ...

    def list(
        self, *, kind: PresetKind | None = None, favorite_only: bool = False, limit: int = 100
    ) -> tuple[Preset, ...]: ...

    def search(
        self,
        text: str | None = None,
        *,
        kind: PresetKind | None = None,
        favorite_only: bool = False,
        limit: int = 100,
    ) -> tuple[Preset, ...]: ...

    def delete(self, preset_id: UUID) -> None: ...

    def set_favorite(self, preset_id: UUID, favorite: bool) -> Preset: ...

    def record_usage(self, preset_id: UUID, used_at: datetime | None = None) -> Preset: ...


class PresetRepository(PresetRepositoryProtocol):
    """PresetのJSONを型付きPayloadへ変換して返すRepository。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, preset: Preset) -> Preset:
        try:
            with session_scope(self._session_factory) as session:
                row = _from_domain(preset)
                session.add(row)
                session.flush()
                return _to_domain(row)
        except PresetRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, PresetPayloadError) as exc:
            raise PresetRepositoryError("preset could not be created") from exc

    def update(self, preset: Preset) -> Preset:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(PresetModel, str(preset.id))
                if row is None:
                    raise PresetRepositoryError("preset was not found")
                row.kind = preset.kind.value
                row.name = preset.name
                row.description = preset.description
                row.payload_json = preset.payload.model_dump_json()
                row.schema_version = preset.schema_version
                row.favorite = preset.favorite
                row.updated_at = _required_utc(preset.updated_at)
                session.flush()
                return _to_domain(row)
        except PresetRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, PresetPayloadError) as exc:
            raise PresetRepositoryError("preset could not be updated") from exc

    def get_by_id(self, preset_id: UUID) -> Preset | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(PresetModel, str(preset_id))
                return _to_domain(row) if row is not None else None
        except (SQLAlchemyError, PresetPayloadError) as exc:
            raise PresetRepositoryError("preset could not be read") from exc

    def list(
        self, *, kind: PresetKind | None = None, favorite_only: bool = False, limit: int = 100
    ) -> tuple[Preset, ...]:
        return self.search(kind=kind, favorite_only=favorite_only, limit=limit)

    def search(
        self,
        text: str | None = None,
        *,
        kind: PresetKind | None = None,
        favorite_only: bool = False,
        limit: int = 100,
    ) -> tuple[Preset, ...]:
        normalized_limit = min(max(1, limit), 100)
        try:
            with session_scope(self._session_factory) as session:
                statement = select(PresetModel)
                if kind is not None:
                    statement = statement.where(PresetModel.kind == kind.value)
                if favorite_only:
                    statement = statement.where(PresetModel.favorite.is_(True))
                if text and text.strip():
                    pattern = f"%{_escape_like(text.strip().lower())}%"
                    statement = statement.where(
                        func.lower(PresetModel.name).like(pattern, escape="\\")
                    )
                rows = session.scalars(
                    statement.order_by(
                        PresetModel.last_used_at.desc(), PresetModel.updated_at.desc()
                    ).limit(normalized_limit)
                ).all()
                return tuple(_to_domain(row) for row in rows)
        except (SQLAlchemyError, PresetPayloadError) as exc:
            raise PresetRepositoryError("presets could not be searched") from exc

    def delete(self, preset_id: UUID) -> None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(PresetModel, str(preset_id))
                if row is not None:
                    session.delete(row)
        except SQLAlchemyError as exc:
            raise PresetRepositoryError("preset could not be deleted") from exc

    def set_favorite(self, preset_id: UUID, favorite: bool) -> Preset:
        try:
            with session_scope(self._session_factory) as session:
                row = _require(session, preset_id)
                row.favorite = favorite
                row.updated_at = datetime.now(UTC)
                session.flush()
                return _to_domain(row)
        except PresetRepositoryError:
            raise
        except (SQLAlchemyError, PresetPayloadError) as exc:
            raise PresetRepositoryError("preset favorite could not be saved") from exc

    def record_usage(self, preset_id: UUID, used_at: datetime | None = None) -> Preset:
        try:
            with session_scope(self._session_factory) as session:
                row = _require(session, preset_id)
                row.usage_count += 1
                row.last_used_at = _required_utc(used_at or datetime.now(UTC))
                row.updated_at = _required_utc(used_at or datetime.now(UTC))
                session.flush()
                return _to_domain(row)
        except PresetRepositoryError:
            raise
        except (SQLAlchemyError, PresetPayloadError) as exc:
            raise PresetRepositoryError("preset usage could not be recorded") from exc


def _from_domain(preset: Preset) -> PresetModel:
    return PresetModel(
        id=str(preset.id),
        kind=preset.kind.value,
        name=preset.name,
        description=preset.description,
        payload_json=preset.payload.model_dump_json(),
        schema_version=preset.schema_version,
        favorite=preset.favorite,
        usage_count=preset.usage_count,
        last_used_at=_utc(preset.last_used_at),
        created_at=_utc(preset.created_at),
        updated_at=_utc(preset.updated_at),
    )


def _to_domain(row: PresetModel) -> Preset:
    kind = PresetKind(row.kind)
    return Preset(
        id=UUID(row.id),
        kind=kind,
        name=row.name,
        description=row.description,
        payload=parse_payload(kind, row.payload_json),
        schema_version=row.schema_version,
        favorite=row.favorite,
        usage_count=row.usage_count,
        last_used_at=_utc(row.last_used_at),
        created_at=_required_utc(row.created_at),
        updated_at=_required_utc(row.updated_at),
    )


def _require(session: Session, preset_id: UUID) -> PresetModel:
    row = session.get(PresetModel, str(preset_id))
    if row is None:
        raise PresetRepositoryError("preset was not found")
    return row


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _required_utc(value: datetime) -> datetime:
    result = _utc(value)
    assert result is not None
    return result


__all__ = ["PresetRepository", "PresetRepositoryError", "PresetRepositoryProtocol"]
