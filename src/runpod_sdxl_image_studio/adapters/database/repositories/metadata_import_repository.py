"""SQLite repository for external metadata import records."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import MetadataImportModel
from runpod_sdxl_image_studio.domain.metadata_import import (
    ImportedImage,
    MetadataImportCandidate,
    MetadataImportRecord,
    MetadataImportStatus,
    MetadataModelMapping,
    MetadataRawSource,
    MetadataSourceKind,
)


class MetadataImportRepositoryError(RuntimeError):
    """Safe persistence error for metadata import records."""


class MetadataImportRepositoryProtocol(Protocol):
    def create(self, record: MetadataImportRecord) -> MetadataImportRecord: ...

    def save(self, record: MetadataImportRecord) -> MetadataImportRecord: ...

    def get_by_id(self, import_id: UUID) -> MetadataImportRecord | None: ...

    def get_by_source_image_sha256(
        self, source_image_sha256: str
    ) -> MetadataImportRecord | None: ...

    def list_recent(self, limit: int = 20) -> tuple[MetadataImportRecord, ...]: ...


class MetadataImportRepository(MetadataImportRepositoryProtocol):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, record: MetadataImportRecord) -> MetadataImportRecord:
        return self.save(record)

    def save(self, record: MetadataImportRecord) -> MetadataImportRecord:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(MetadataImportModel, str(record.id))
                values = _row_values(record)
                if row is None:
                    session.add(MetadataImportModel(**values))
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
                session.flush()
                return record
        except (IntegrityError, SQLAlchemyError, ValueError, TypeError) as exc:
            raise MetadataImportRepositoryError("metadata import could not be saved") from exc

    def get_by_id(self, import_id: UUID) -> MetadataImportRecord | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(MetadataImportModel, str(import_id))
                return _record(row) if row is not None else None
        except (SQLAlchemyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise MetadataImportRepositoryError("metadata import could not be read") from exc

    def get_by_source_image_sha256(self, source_image_sha256: str) -> MetadataImportRecord | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.scalar(
                    select(MetadataImportModel)
                    .where(MetadataImportModel.source_image_sha256 == source_image_sha256)
                    .order_by(MetadataImportModel.created_at.desc())
                    .limit(1)
                )
                return _record(row) if row is not None else None
        except (SQLAlchemyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise MetadataImportRepositoryError("metadata import could not be read") from exc

    def list_recent(self, limit: int = 20) -> tuple[MetadataImportRecord, ...]:
        if limit < 1:
            return ()
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(MetadataImportModel)
                    .order_by(MetadataImportModel.created_at.desc())
                    .limit(min(limit, 100))
                ).all()
                return tuple(_record(row) for row in rows)
        except (SQLAlchemyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise MetadataImportRepositoryError("metadata imports could not be listed") from exc


def _row_values(record: MetadataImportRecord) -> dict[str, object]:
    raw_payload = {
        "schema_version": 1,
        "sources": [source.model_dump(mode="json") for source in record.raw_sources],
    }
    raw_json = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "id": str(record.id),
        "original_filename": record.imported_image.original_filename,
        "stored_image_path": record.imported_image.stored_image_path,
        "source_image_sha256": record.imported_image.source_image_sha256,
        "stored_image_sha256": record.imported_image.stored_image_sha256,
        "image_width": record.imported_image.image_width,
        "image_height": record.imported_image.image_height,
        "image_mime_type": record.imported_image.image_mime_type,
        "metadata_source": record.metadata_source.value,
        "metadata_status": record.metadata_status.value,
        "raw_metadata_json": raw_json,
        "raw_metadata_sha256": hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
        "candidate_json": (
            record.candidate.model_dump_json() if record.candidate is not None else None
        ),
        "candidate_options_json": json.dumps(
            [candidate.model_dump(mode="json") for candidate in record.candidates],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "selected_metadata_source": (
            record.selected_metadata_source.value
            if record.selected_metadata_source is not None
            else None
        ),
        "sidecar_hash_confirmed": record.sidecar_hash_confirmed,
        "normalized_snapshot_json": record.normalized_snapshot_json,
        "normalized_snapshot_schema_version": record.normalized_snapshot_schema_version,
        "manual_mapping_json": json.dumps(
            [mapping.model_dump(mode="json") for mapping in record.manual_mappings],
            ensure_ascii=False,
        ),
        "warnings_json": json.dumps(record.warnings, ensure_ascii=False),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _record(row: MetadataImportModel) -> MetadataImportRecord:
    raw_payload = json.loads(row.raw_metadata_json)
    sources = tuple(MetadataRawSource.model_validate(source) for source in raw_payload["sources"])
    candidate = (
        MetadataImportCandidate.model_validate(json.loads(row.candidate_json))
        if row.candidate_json
        else None
    )
    candidates = tuple(
        MetadataImportCandidate.model_validate(candidate_value)
        for candidate_value in json.loads(getattr(row, "candidate_options_json", "[]") or "[]")
    )
    if not candidates and candidate is not None:
        candidates = (candidate,)
    mappings = tuple(
        MetadataModelMapping.model_validate(mapping)
        for mapping in json.loads(row.manual_mapping_json or "[]")
    )
    warnings = tuple(str(value) for value in json.loads(row.warnings_json or "[]"))
    image = ImportedImage(
        id=UUID(row.id),
        original_filename=row.original_filename,
        stored_image_path=row.stored_image_path,
        source_image_sha256=row.source_image_sha256,
        stored_image_sha256=row.stored_image_sha256,
        image_width=row.image_width,
        image_height=row.image_height,
        image_mime_type=row.image_mime_type,
        created_at=_utc(row.created_at),
    )
    return MetadataImportRecord(
        id=UUID(row.id),
        imported_image=image,
        metadata_source=MetadataSourceKind(row.metadata_source),
        metadata_status=MetadataImportStatus(row.metadata_status),
        raw_sources=sources,
        candidate=candidate,
        candidates=candidates,
        selected_metadata_source=(
            MetadataSourceKind(row.selected_metadata_source)
            if getattr(row, "selected_metadata_source", None)
            else None
        ),
        sidecar_hash_confirmed=bool(getattr(row, "sidecar_hash_confirmed", False)),
        normalized_snapshot_json=row.normalized_snapshot_json,
        normalized_snapshot_schema_version=row.normalized_snapshot_schema_version,
        manual_mappings=mappings,
        warnings=warnings,
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
    )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "MetadataImportRepository",
    "MetadataImportRepositoryError",
    "MetadataImportRepositoryProtocol",
]
