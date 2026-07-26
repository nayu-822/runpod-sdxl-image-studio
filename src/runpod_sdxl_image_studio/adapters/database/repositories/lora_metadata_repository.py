"""Repository for LoRA metadata with explicit transaction boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import LoraMetadataModel
from runpod_sdxl_image_studio.domain.lora import normalize_relative_lora_name
from runpod_sdxl_image_studio.domain.lora_metadata import LoraMetadata, LoraMetadataUpdate
from runpod_sdxl_image_studio.domain.lora_search import LoraSearchQuery, LoraSort


class LoraMetadataRepository:
    """Return domain objects and never expose ORM or Gradio objects."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_by_file_name(self, file_name: str) -> LoraMetadata | None:
        name = normalize_relative_lora_name(file_name)
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(LoraMetadataModel).where(LoraMetadataModel.file_name == name)
            )
            return row.to_domain() if row is not None else None

    def get_by_id(self, metadata_id: UUID) -> LoraMetadata | None:
        with session_scope(self._session_factory) as session:
            row = session.get(LoraMetadataModel, str(metadata_id))
            return row.to_domain() if row is not None else None

    def list_all(self, query: LoraSearchQuery | None = None) -> tuple[LoraMetadata, ...]:
        search = query or LoraSearchQuery()
        with session_scope(self._session_factory) as session:
            statement = self._query_statement(search)
            return tuple(row.to_domain() for row in session.scalars(statement).all())

    def list_categories(self) -> tuple[str, ...]:
        with session_scope(self._session_factory) as session:
            values = session.scalars(
                select(LoraMetadataModel.category)
                .where(LoraMetadataModel.category.is_not(None))
                .distinct()
                .order_by(LoraMetadataModel.category.asc())
            ).all()
            return tuple(value for value in values if value)

    def upsert_discovered_loras(self, file_names: Iterable[str]) -> tuple[LoraMetadata, ...]:
        names = tuple(dict.fromkeys(normalize_relative_lora_name(name) for name in file_names))
        now = datetime.now(UTC)
        with session_scope(self._session_factory) as session:
            rows = session.scalars(select(LoraMetadataModel)).all()
            by_name = {row.file_name: row for row in rows}
            for row in rows:
                row.is_missing = True
                row.updated_at = now
            for name in names:
                existing = by_name.get(name)
                if existing is None:
                    row = LoraMetadataModel.from_domain(
                        LoraMetadata(
                            file_name=name,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    session.add(row)
                else:
                    existing.is_missing = False
                    existing.updated_at = now
            session.flush()
            result: list[LoraMetadata] = []
            for name in names:
                fetched = session.scalar(
                    select(LoraMetadataModel).where(LoraMetadataModel.file_name == name)
                )
                if fetched is not None:
                    result.append(fetched.to_domain())
            return tuple(result)

    def update_metadata(
        self, metadata_id: UUID, metadata: LoraMetadataUpdate
    ) -> LoraMetadata | None:
        with session_scope(self._session_factory) as session:
            row = session.get(LoraMetadataModel, str(metadata_id))
            if row is None:
                return None
            row.display_name = metadata.display_name
            row.category = metadata.category
            row.is_favorite = metadata.is_favorite
            row.trigger_words_json = _json(metadata.trigger_words)
            row.recommended_model_strength = metadata.recommended_model_strength
            row.recommended_clip_strength = metadata.recommended_clip_strength
            row.notes = metadata.notes
            row.compatible_models_json = _json(metadata.compatible_models)
            row.updated_at = datetime.now(UTC)
            session.flush()
            return row.to_domain()

    def set_favorite(self, metadata_id: UUID, is_favorite: bool) -> LoraMetadata | None:
        with session_scope(self._session_factory) as session:
            row = session.get(LoraMetadataModel, str(metadata_id))
            if row is None:
                return None
            row.is_favorite = is_favorite
            row.updated_at = datetime.now(UTC)
            session.flush()
            return row.to_domain()

    def set_thumbnail_path(
        self, metadata_id: UUID, thumbnail_path: str | None
    ) -> LoraMetadata | None:
        with session_scope(self._session_factory) as session:
            row = session.get(LoraMetadataModel, str(metadata_id))
            if row is None:
                return None
            row.thumbnail_path = thumbnail_path
            row.updated_at = datetime.now(UTC)
            session.flush()
            return row.to_domain()

    def update_usage(
        self, file_names: Iterable[str], completed_at: datetime | None = None
    ) -> tuple[LoraMetadata, ...]:
        names = tuple(dict.fromkeys(normalize_relative_lora_name(name) for name in file_names))
        timestamp = completed_at or datetime.now(UTC)
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(LoraMetadataModel).where(LoraMetadataModel.file_name.in_(names))
            ).all()
            for row in rows:
                row.usage_count += 1
                row.last_used_at = timestamp
                row.updated_at = timestamp
            session.flush()
            return tuple(row.to_domain() for row in rows)

    def _query_statement(self, query: LoraSearchQuery) -> Select[tuple[LoraMetadataModel]]:
        statement = select(LoraMetadataModel)
        if query.text:
            pattern = f"%{_escape_like(query.text)}%"
            statement = statement.where(
                or_(
                    LoraMetadataModel.file_name.ilike(pattern, escape="\\"),
                    LoraMetadataModel.display_name.ilike(pattern, escape="\\"),
                    LoraMetadataModel.trigger_words_json.ilike(pattern, escape="\\"),
                    LoraMetadataModel.notes.ilike(pattern, escape="\\"),
                )
            )
        if query.category:
            statement = statement.where(LoraMetadataModel.category == query.category)
        if query.favorites_only:
            statement = statement.where(LoraMetadataModel.is_favorite.is_(True))
        if not query.include_missing:
            statement = statement.where(LoraMetadataModel.is_missing.is_(False))
        if query.sort is LoraSort.NAME:
            statement = statement.order_by(func.lower(LoraMetadataModel.file_name).asc())
        elif query.sort is LoraSort.USAGE:
            statement = statement.order_by(
                LoraMetadataModel.usage_count.desc(), LoraMetadataModel.file_name.asc()
            )
        elif query.sort is LoraSort.RECENT:
            statement = statement.order_by(
                LoraMetadataModel.last_used_at.desc(), LoraMetadataModel.file_name.asc()
            )
        else:
            statement = statement.order_by(
                LoraMetadataModel.is_favorite.desc(),
                LoraMetadataModel.last_used_at.desc(),
                LoraMetadataModel.file_name.asc(),
            )
        return statement


def _json(values: Iterable[str]) -> str:
    import json

    return json.dumps(tuple(values), ensure_ascii=False)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
