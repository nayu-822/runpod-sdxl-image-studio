"""Consistent SQLite snapshot creation for stateless state backup."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from runpod_sdxl_image_studio.adapters.database.engine import sqlite_database_path
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.state_sync import StateSnapshotMetadata


class StateSnapshotError(RuntimeError):
    """Safe snapshot creation or integrity failure."""


@dataclass(frozen=True)
class StateSnapshot:
    path: Path
    metadata: StateSnapshotMetadata


class StateSnapshotService:
    """Create a point-in-time SQLite copy without copying the live file directly."""

    def __init__(
        self,
        settings: Settings,
        *,
        now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._now_factory = now_factory

    def create_snapshot(self) -> StateSnapshot:
        source_path = sqlite_database_path(self._settings)
        if source_path is None:
            raise StateSnapshotError("state backup requires a file-backed SQLite database")
        source_path = source_path if source_path.is_absolute() else source_path.resolve()
        if not source_path.is_file():
            raise StateSnapshotError("state database does not exist")

        work_dir = self._settings.data_dir / ".state-sync"
        work_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        source: sqlite3.Connection | None = None
        target: sqlite3.Connection | None = None
        succeeded = False
        try:
            with NamedTemporaryFile(
                mode="wb", prefix="state-", suffix=".sqlite3", dir=work_dir, delete=False
            ) as file:
                temporary = Path(file.name)
            os.chmod(temporary, 0o600)
            source = sqlite3.connect(str(source_path), timeout=30.0)
            target = sqlite3.connect(str(temporary), timeout=30.0)
            source.backup(target, pages=100, sleep=0.01)
            self._check_integrity(target)
            target.close()
            target = None
            source.close()
            source = None
            size_bytes = temporary.stat().st_size
            metadata = StateSnapshotMetadata(
                schema_version=1,
                filename=temporary.name,
                sha256=_sha256(temporary),
                size_bytes=size_bytes,
                created_at=_utc(self._now_factory()),
            )
            succeeded = True
            return StateSnapshot(temporary, metadata)
        except (OSError, sqlite3.Error) as exc:
            raise StateSnapshotError("SQLite state snapshot could not be created") from exc
        finally:
            if target is not None:
                target.close()
            if source is not None:
                source.close()
            if temporary is not None and not succeeded:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def verify_snapshot(path: Path) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(str(path), timeout=30.0)
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise StateSnapshotError("SQLite state snapshot failed integrity check")
        except sqlite3.Error as exc:
            raise StateSnapshotError("SQLite state snapshot could not be verified") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _check_integrity(connection: sqlite3.Connection) -> None:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise StateSnapshotError("SQLite state snapshot failed integrity check")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["StateSnapshot", "StateSnapshotError", "StateSnapshotService"]
