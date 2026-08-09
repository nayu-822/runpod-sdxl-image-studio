"""Manual and debounced background synchronization of SQLite application state."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from runpod_sdxl_image_studio.adapters.rclone.state_backup_storage import StateBackupStorage
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.state_sync import (
    StateSyncStatus,
    StateSyncView,
)
from runpod_sdxl_image_studio.services.state_snapshot_service import (
    StateSnapshot,
    StateSnapshotService,
)

logger = logging.getLogger(__name__)


class StateSyncService:
    """Coordinate snapshot creation and remote upload outside DB transactions."""

    def __init__(
        self,
        settings: Settings,
        *,
        storage: StateBackupStorage | None = None,
        snapshot_service: StateSnapshotService | None = None,
        now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._storage = storage or StateBackupStorage(settings)
        self._snapshot_service = snapshot_service or StateSnapshotService(
            settings, now_factory=now_factory
        )
        self._now_factory = now_factory
        self._timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()
        self._backup_lock = threading.Lock()
        self._last_hash: str | None = None
        self._view = StateSyncView(
            StateSyncStatus.IDLE
            if settings.state_sync_enabled and self._storage.is_configured
            else StateSyncStatus.DISABLED
        )

    @property
    def enabled(self) -> bool:
        return self._settings.state_sync_enabled and self._storage.is_configured

    def get_status(self) -> StateSyncView:
        with self._timer_lock:
            return self._view

    async def backup(self) -> StateSyncView:
        if not self.enabled:
            self._set_view(StateSyncStatus.DISABLED, "状態バックアップは無効です。")
            return self.get_status()
        if not self._backup_lock.acquire(blocking=False):
            return self.get_status()
        snapshot: StateSnapshot | None = None
        self._set_view(StateSyncStatus.RUNNING, "状態バックアップを実行中です。")
        try:
            snapshot = self._snapshot_service.create_snapshot()
            if snapshot.metadata.sha256 == self._last_hash:
                self._set_view(
                    StateSyncStatus.SYNCED,
                    "状態バックアップは変更がないためアップロードを省略しました。",
                    success_at=snapshot.metadata.created_at,
                    remote_sha256=snapshot.metadata.sha256,
                    remote_size_bytes=snapshot.metadata.size_bytes,
                )
                return self.get_status()
            backup_name = _backup_name(snapshot.metadata.created_at)
            remote_filename = f"backups/{backup_name}.sqlite3"
            metadata_filename = f"{remote_filename}.metadata.json"
            pointer_path = self._write_pointer(snapshot, remote_filename)
            metadata_path = self._write_metadata(snapshot, remote_filename, metadata_filename)
            try:
                await self._storage.upload(snapshot.path, remote_filename)
                await self._storage.upload(metadata_path, metadata_filename)
                await self._storage.upload(pointer_path, "latest.json")
            finally:
                pointer_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            self._last_hash = snapshot.metadata.sha256
            self._set_view(
                StateSyncStatus.SYNCED,
                "状態バックアップが完了しました。",
                success_at=snapshot.metadata.created_at,
                remote_sha256=snapshot.metadata.sha256,
                remote_size_bytes=snapshot.metadata.size_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - remote backup is non-fatal to the app
            logger.warning("state backup failed error=%s", type(exc).__name__)
            self._set_view(
                StateSyncStatus.FAILED,
                "状態バックアップに失敗しました。",
                failure_at=_utc(self._now_factory()),
            )
        finally:
            if snapshot is not None:
                snapshot.path.unlink(missing_ok=True)
            self._backup_lock.release()
        return self.get_status()

    def backup_sync(self) -> StateSyncView:
        return asyncio.run(self.backup())

    def mark_dirty(self) -> None:
        """Debounce a state change and run its backup on a daemon thread."""

        if not self.enabled:
            return
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                self._settings.state_sync_debounce_seconds,
                self._run_debounced_backup,
            )
            self._timer.daemon = True
            self._timer.start()

    def close(self) -> None:
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _run_debounced_backup(self) -> None:
        try:
            asyncio.run(self.backup())
        except Exception:  # noqa: BLE001 - background backup must never crash the process
            logger.warning("debounced state backup failed", exc_info=True)

    def _write_metadata(
        self, snapshot: StateSnapshot, remote_filename: str, local_name: str
    ) -> Path:
        return _write_json(
            self._settings.data_dir / ".state-sync",
            local_name.rsplit("/", 1)[-1],
            {
                "schema_version": snapshot.metadata.schema_version,
                "filename": remote_filename,
                "sha256": snapshot.metadata.sha256,
                "size_bytes": snapshot.metadata.size_bytes,
                "created_at": snapshot.metadata.created_at.isoformat().replace("+00:00", "Z"),
            },
        )

    def _write_pointer(self, snapshot: StateSnapshot, filename: str) -> Path:
        return _write_json(
            self._settings.data_dir / ".state-sync",
            "latest.json",
            {
                "schema_version": snapshot.metadata.schema_version,
                "filename": filename,
                "sha256": snapshot.metadata.sha256,
                "size_bytes": snapshot.metadata.size_bytes,
                "created_at": snapshot.metadata.created_at.isoformat().replace("+00:00", "Z"),
            },
        )

    def _set_view(
        self,
        status: StateSyncStatus,
        message: str,
        *,
        success_at: datetime | None = None,
        failure_at: datetime | None = None,
        remote_sha256: str | None = None,
        remote_size_bytes: int | None = None,
    ) -> None:
        with self._timer_lock:
            current = self._view
            self._view = StateSyncView(
                status=status,
                last_success_at=success_at or current.last_success_at,
                last_failure_at=failure_at or current.last_failure_at,
                last_message=message,
                remote_sha256=remote_sha256 or current.remote_sha256,
                remote_size_bytes=remote_size_bytes or current.remote_size_bytes,
            )


def _backup_name(created_at: datetime) -> str:
    return created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(directory: Path, name: str, value: dict[str, object]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    target = directory / name
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="state-", suffix=".json", dir=directory, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(value, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")
            file.flush()
        temporary.replace(target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["StateSyncService"]
