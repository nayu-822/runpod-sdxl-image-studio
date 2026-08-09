"""Startup-only SQLite state restore with hash, integrity, and atomic placement."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from runpod_sdxl_image_studio.adapters.database.engine import sqlite_database_path
from runpod_sdxl_image_studio.adapters.rclone.state_backup_storage import StateBackupStorage
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.state_sync import (
    StateRestoreResult,
    StateRestoreStatus,
    StateSnapshotMetadata,
)
from runpod_sdxl_image_studio.services.state_snapshot_service import (
    StateSnapshotError,
    StateSnapshotService,
)

logger = logging.getLogger(__name__)


class StateRestoreService:
    """Restore only when the local database is absent; never overwrite local state."""

    def __init__(
        self,
        settings: Settings,
        storage: StateBackupStorage | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage or StateBackupStorage(settings)

    def restore_if_missing(self) -> StateRestoreResult:
        return asyncio.run(self.restore_if_missing_async())

    async def restore_if_missing_async(self) -> StateRestoreResult:
        if (
            not self._settings.state_sync_enabled
            or not self._settings.state_sync_restore_on_startup
        ):
            return StateRestoreResult(StateRestoreStatus.DISABLED, message="state sync is disabled")
        if not self._storage.is_configured:
            return StateRestoreResult(
                StateRestoreStatus.DISABLED, message="Drive is not configured"
            )
        database_path = sqlite_database_path(self._settings)
        if database_path is None:
            return StateRestoreResult(
                StateRestoreStatus.FAILED,
                message="state restore requires a file-backed SQLite database",
            )
        database_path = database_path if database_path.is_absolute() else database_path.resolve()
        if database_path.exists():
            return StateRestoreResult(
                StateRestoreStatus.SKIPPED_LOCAL, message="local database exists"
            )

        work_dir = self._settings.data_dir / ".state-sync"
        work_dir.mkdir(parents=True, exist_ok=True)
        pointer_path = work_dir / "latest.json"
        temporary_db: Path | None = None
        try:
            await self._storage.download("latest.json", pointer_path)
        except Exception as exc:  # noqa: BLE001 - missing/unavailable remote is non-fatal
            logger.info("state restore pointer unavailable error=%s", type(exc).__name__)
            pointer_path.unlink(missing_ok=True)
            return StateRestoreResult(
                StateRestoreStatus.NO_BACKUP, message="no remote state backup"
            )

        try:
            metadata = _read_pointer(pointer_path)
            with NamedTemporaryFile(
                mode="wb", prefix="restore-", suffix=".sqlite3", dir=work_dir, delete=False
            ) as file:
                temporary_db = Path(file.name)
            os.chmod(temporary_db, 0o600)
            await self._storage.download(metadata.filename, temporary_db)
            if temporary_db.stat().st_size != metadata.size_bytes:
                raise StateRestoreError("state backup size does not match latest pointer")
            if _sha256(temporary_db) != metadata.sha256:
                raise StateRestoreError("state backup hash does not match latest pointer")
            StateSnapshotService.verify_snapshot(temporary_db)
            if database_path.exists():
                return StateRestoreResult(
                    StateRestoreStatus.SKIPPED_LOCAL,
                    metadata=metadata,
                    message="local database appeared during restore",
                )
            database_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_db, database_path)
            temporary_db = None
            return StateRestoreResult(
                StateRestoreStatus.RESTORED,
                metadata=metadata,
                message="state database restored",
            )
        except (OSError, ValueError, StateRestoreError, StateSnapshotError) as exc:
            logger.warning("state restore validation failed error=%s", type(exc).__name__)
            return StateRestoreResult(StateRestoreStatus.FAILED, message="state restore failed")
        except Exception as exc:  # noqa: BLE001 - remote transfer failures fail closed
            logger.warning("state restore transfer failed error=%s", type(exc).__name__)
            return StateRestoreResult(StateRestoreStatus.FAILED, message="state restore failed")
        finally:
            pointer_path.unlink(missing_ok=True)
            if temporary_db is not None:
                temporary_db.unlink(missing_ok=True)


class StateRestoreError(RuntimeError):
    """Safe restore validation error."""


def _read_pointer(path: Path) -> StateSnapshotMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise StateRestoreError("latest state pointer is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise StateRestoreError("latest state pointer schema is unsupported")
    filename = payload.get("filename")
    sha256 = payload.get("sha256")
    size_bytes = payload.get("size_bytes")
    created_at = payload.get("created_at")
    if (
        not isinstance(filename, str)
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256.lower())
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
        or not isinstance(created_at, str)
    ):
        raise StateRestoreError("latest state pointer fields are invalid")
    relative = PurePosixPath(filename.replace("\\", "/"))
    if (
        relative.is_absolute()
        or relative.parts[:1] != ("backups",)
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix != ".sqlite3"
    ):
        raise StateRestoreError("latest state filename is unsafe")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StateRestoreError("latest state timestamp is invalid") from exc
    return StateSnapshotMetadata(
        schema_version=1,
        filename=relative.as_posix(),
        sha256=sha256.lower(),
        size_bytes=size_bytes,
        created_at=timestamp.replace(tzinfo=UTC)
        if timestamp.tzinfo is None
        else timestamp.astimezone(UTC),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["StateRestoreError", "StateRestoreService"]
