"""Persistence boundary for the separate upscale settings snapshot."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import GenerationUpscaleSettingsModel
from runpod_sdxl_image_studio.domain.upscale_snapshot import (
    UpscaleSettingsSnapshot,
    UpscaleSnapshotError,
    UpscaleSourceKind,
)


class UpscaleSettingsRepositoryError(RuntimeError):
    """Safe persistence error for upscale settings."""


class UpscaleSettingsRepositoryProtocol(Protocol):
    def get_by_generation(self, generation_id: UUID) -> UpscaleSettingsSnapshot | None: ...

    def get_by_source_artifact(
        self, source_artifact_id: UUID
    ) -> tuple[UpscaleSettingsSnapshot, ...]: ...

    def get_by_source_import(
        self, source_import_id: UUID
    ) -> tuple[UpscaleSettingsSnapshot, ...]: ...


class UpscaleSettingsRepository(UpscaleSettingsRepositoryProtocol):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_by_generation(self, generation_id: UUID) -> UpscaleSettingsSnapshot | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(GenerationUpscaleSettingsModel, str(generation_id))
                return _snapshot(row) if row is not None else None
        except (SQLAlchemyError, UpscaleSnapshotError) as exc:
            raise UpscaleSettingsRepositoryError("upscale settings could not be read") from exc

    def get_by_source_artifact(
        self, source_artifact_id: UUID
    ) -> tuple[UpscaleSettingsSnapshot, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(GenerationUpscaleSettingsModel)
                    .where(
                        GenerationUpscaleSettingsModel.source_artifact_id == str(source_artifact_id)
                    )
                    .order_by(GenerationUpscaleSettingsModel.created_at.asc())
                ).all()
                return tuple(_snapshot(row) for row in rows)
        except (SQLAlchemyError, UpscaleSnapshotError) as exc:
            raise UpscaleSettingsRepositoryError("upscale settings could not be read") from exc

    def get_by_source_import(self, source_import_id: UUID) -> tuple[UpscaleSettingsSnapshot, ...]:
        try:
            with session_scope(self._session_factory) as session:
                rows = session.scalars(
                    select(GenerationUpscaleSettingsModel)
                    .where(GenerationUpscaleSettingsModel.source_import_id == str(source_import_id))
                    .order_by(GenerationUpscaleSettingsModel.created_at.asc())
                ).all()
                return tuple(_snapshot(row) for row in rows)
        except (SQLAlchemyError, UpscaleSnapshotError) as exc:
            raise UpscaleSettingsRepositoryError("upscale settings could not be read") from exc


def _snapshot(row: GenerationUpscaleSettingsModel) -> UpscaleSettingsSnapshot:
    snapshot = UpscaleSettingsSnapshot.from_json(row.settings_snapshot_json)
    row_source_kind = getattr(row, "source_kind", UpscaleSourceKind.GENERATION_ARTIFACT)
    if snapshot.source_kind.value != row_source_kind:
        raise UpscaleSnapshotError("upscale settings source kind does not match its row")
    if snapshot.source_kind is UpscaleSourceKind.GENERATION_ARTIFACT:
        if row.source_artifact_id is None or snapshot.source_artifact_id != UUID(
            row.source_artifact_id
        ):
            raise UpscaleSnapshotError("upscale settings source does not match its row")
    elif row.source_import_id is None or snapshot.source_import_id != UUID(row.source_import_id):
        raise UpscaleSnapshotError("upscale settings import source does not match its row")
    return snapshot


__all__ = [
    "UpscaleSettingsRepository",
    "UpscaleSettingsRepositoryError",
    "UpscaleSettingsRepositoryProtocol",
]
