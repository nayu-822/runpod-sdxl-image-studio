"""SQLite persistence for the singleton last-generation form snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import GenerationFormStateModel
from runpod_sdxl_image_studio.domain.generation_form_state import GenerationFormStateSnapshot


class GenerationFormStateRepositoryError(RuntimeError):
    """Safe persistence error for the form state boundary."""


class GenerationFormStateRepositoryProtocol(Protocol):
    def get(self) -> GenerationFormStateSnapshot | None: ...

    def save(self, snapshot: GenerationFormStateSnapshot) -> GenerationFormStateSnapshot: ...


class GenerationFormStateRepository(GenerationFormStateRepositoryProtocol):
    """Persist one validated JSON snapshot under a stable singleton key."""

    _ROW_ID = "current"

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get(self) -> GenerationFormStateSnapshot | None:
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(GenerationFormStateModel, self._ROW_ID)
                if row is None:
                    return None
                return GenerationFormStateSnapshot.from_json(row.snapshot_json)
        except Exception as exc:  # noqa: BLE001 - do not leak SQLite details to the UI
            if isinstance(exc, GenerationFormStateRepositoryError):
                raise
            raise GenerationFormStateRepositoryError(
                "generation form state could not be read"
            ) from exc

    def save(self, snapshot: GenerationFormStateSnapshot) -> GenerationFormStateSnapshot:
        timestamp = _utc(snapshot.updated_at)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(GenerationFormStateModel, self._ROW_ID)
                if row is None:
                    row = GenerationFormStateModel(
                        id=self._ROW_ID,
                        schema_version=snapshot.schema_version,
                        snapshot_json=snapshot.to_json(),
                        updated_at=timestamp,
                    )
                    session.add(row)
                else:
                    row.schema_version = snapshot.schema_version
                    row.snapshot_json = snapshot.to_json()
                    row.updated_at = timestamp
                session.flush()
                return snapshot
        except SQLAlchemyError as exc:
            raise GenerationFormStateRepositoryError(
                "generation form state could not be saved"
            ) from exc


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "GenerationFormStateRepository",
    "GenerationFormStateRepositoryError",
    "GenerationFormStateRepositoryProtocol",
]
