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
        initial_remote_sha256: str | None = None,
        remote_write_protected: bool = False,
    ) -> None:
        self._settings = settings
        self._storage = storage or StateBackupStorage(settings)
        self._snapshot_service = snapshot_service or StateSnapshotService(
            settings, now_factory=now_factory
        )
        self._now_factory = now_factory
        self._timer: threading.Timer | None = None
        self._timer_token = 0
        self._timer_lock = threading.Lock()
        self._backup_lock = threading.Lock()
        self._last_hash = initial_remote_sha256
        self._dirty_version = 0
        self._backed_up_version = 0
        self._closed = False
        self._remote_write_protected = remote_write_protected
        initial_status = (
            StateSyncStatus.FAILED
            if remote_write_protected
            else StateSyncStatus.IDLE
            if settings.state_sync_enabled and self._storage.is_configured
            else StateSyncStatus.DISABLED
        )
        initial_message = (
            "状態バックアップはリモート復旧失敗のため無効です。" if remote_write_protected else ""
        )
        self._view = StateSyncView(
            initial_status,
            last_message=initial_message,
            remote_sha256=initial_remote_sha256,
        )

    @property
    def enabled(self) -> bool:
        return self._is_configured and not self._remote_write_protected and not self._closed

    @property
    def _is_configured(self) -> bool:
        return self._settings.state_sync_enabled and self._storage.is_configured

    def get_status(self) -> StateSyncView:
        with self._timer_lock:
            return self._view

    async def backup(self, *, wait_for_clean: bool = True) -> StateSyncView:
        if self._remote_write_protected:
            self._set_view(
                StateSyncStatus.FAILED,
                "状態バックアップはリモート復旧失敗のため無効です。",
            )
            return self.get_status()
        if not self._is_configured or self._closed:
            self._set_view(StateSyncStatus.DISABLED, "状態バックアップは無効です。")
            return self.get_status()
        while True:
            if not self._backup_lock.acquire(blocking=False):
                if not wait_for_clean:
                    self._schedule_follow_up_backup()
                    return self.get_status()
                await asyncio.sleep(0.01)
                continue
            try:
                snapshot_version = self._current_dirty_version()
                view, succeeded = await self._backup_once(
                    snapshot_version,
                    schedule_follow_up=not wait_for_clean,
                )
            finally:
                self._backup_lock.release()
            if not wait_for_clean or not succeeded or self._is_clean(snapshot_version):
                return view

    def backup_sync(self) -> StateSyncView:
        return asyncio.run(self.backup())

    def mark_dirty(self) -> None:
        """Debounce a state change and run its backup on a daemon thread."""

        if not self.enabled:
            return
        with self._timer_lock:
            self._dirty_version += 1
            self._schedule_timer_locked(reset=True)

    def close(self) -> None:
        should_flush = False
        with self._timer_lock:
            if self._closed:
                return
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._timer_token += 1
            should_flush = self._dirty_version > self._backed_up_version
        if should_flush and self.enabled:
            try:
                self.backup_sync()
            except Exception:  # noqa: BLE001 - stop must continue after best-effort flush
                logger.warning("state backup flush during shutdown failed", exc_info=True)
        with self._timer_lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._timer_token += 1

    async def _backup_once(
        self,
        snapshot_version: int,
        *,
        schedule_follow_up: bool,
    ) -> tuple[StateSyncView, bool]:
        snapshot: StateSnapshot | None = None
        self._set_view(StateSyncStatus.RUNNING, "状態バックアップを実行中です。")
        try:
            snapshot = self._snapshot_service.create_snapshot()
            if snapshot.metadata.sha256 == self._current_last_hash():
                self._record_success(snapshot.metadata.sha256, snapshot_version)
                self._set_view(
                    StateSyncStatus.SYNCED,
                    "状態バックアップは変更がないためアップロードを省略しました。",
                    success_at=snapshot.metadata.created_at,
                    remote_sha256=snapshot.metadata.sha256,
                    remote_size_bytes=snapshot.metadata.size_bytes,
                )
                if schedule_follow_up and not self._is_clean(snapshot_version):
                    self._schedule_follow_up_backup()
                return self.get_status(), True
            backup_name = _backup_name(
                snapshot.metadata.created_at,
                snapshot.metadata.sha256,
            )
            remote_filename = f"backups/{backup_name}.sqlite3"
            metadata_filename = f"{remote_filename}.metadata.json"
            pointer_path: Path | None = None
            metadata_path: Path | None = None
            try:
                pointer_path = self._write_pointer(snapshot, remote_filename)
                metadata_path = self._write_metadata(snapshot, remote_filename, metadata_filename)
                await self._storage.upload(snapshot.path, remote_filename)
                assert metadata_path is not None
                assert pointer_path is not None
                await self._storage.upload(metadata_path, metadata_filename)
                await self._storage.upload(pointer_path, "latest.json")
            finally:
                if pointer_path is not None:
                    pointer_path.unlink(missing_ok=True)
                if metadata_path is not None:
                    metadata_path.unlink(missing_ok=True)
            self._record_success(snapshot.metadata.sha256, snapshot_version)
            self._set_view(
                StateSyncStatus.SYNCED,
                "状態バックアップが完了しました。",
                success_at=snapshot.metadata.created_at,
                remote_sha256=snapshot.metadata.sha256,
                remote_size_bytes=snapshot.metadata.size_bytes,
            )
            if schedule_follow_up and not self._is_clean(snapshot_version):
                self._schedule_follow_up_backup()
            return self.get_status(), True
        except Exception as exc:  # noqa: BLE001 - remote backup is non-fatal to the app
            logger.warning("state backup failed error=%s", type(exc).__name__)
            self._set_view(
                StateSyncStatus.FAILED,
                "状態バックアップに失敗しました。",
                failure_at=_utc(self._now_factory()),
            )
            if schedule_follow_up:
                self._schedule_follow_up_backup()
            return self.get_status(), False
        finally:
            if snapshot is not None:
                snapshot.path.unlink(missing_ok=True)

    def _run_debounced_backup(self, timer_token: int) -> None:
        with self._timer_lock:
            if timer_token != self._timer_token:
                return
            self._timer = None
            if self._closed:
                return
        try:
            asyncio.run(self.backup(wait_for_clean=False))
        except Exception:  # noqa: BLE001 - background backup must never crash the process
            logger.warning("debounced state backup failed", exc_info=True)

    def _schedule_timer_locked(self, *, reset: bool) -> None:
        if self._closed or not self._is_configured or self._remote_write_protected:
            return
        if reset and self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not reset and self._timer is not None:
            return
        self._timer_token += 1
        token = self._timer_token
        self._timer = threading.Timer(
            self._settings.state_sync_debounce_seconds,
            self._run_debounced_backup,
            args=(token,),
        )
        self._timer.daemon = True
        self._timer.start()

    def _schedule_follow_up_backup(self) -> None:
        with self._timer_lock:
            self._schedule_timer_locked(reset=False)

    def _current_dirty_version(self) -> int:
        with self._timer_lock:
            return self._dirty_version

    def _current_last_hash(self) -> str | None:
        with self._timer_lock:
            return self._last_hash

    def _record_success(self, remote_sha256: str, snapshot_version: int) -> None:
        with self._timer_lock:
            self._last_hash = remote_sha256
            self._backed_up_version = max(self._backed_up_version, snapshot_version)

    def _is_clean(self, snapshot_version: int) -> bool:
        with self._timer_lock:
            return (
                self._backed_up_version >= self._dirty_version
                and self._backed_up_version >= snapshot_version
            )

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
                remote_sha256=(
                    remote_sha256 if remote_sha256 is not None else current.remote_sha256
                ),
                remote_size_bytes=(
                    remote_size_bytes
                    if remote_size_bytes is not None
                    else current.remote_size_bytes
                ),
            )


def _backup_name(created_at: datetime, sha256: str) -> str:
    """Build a content-addressed immutable backup basename."""

    timestamp = created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{sha256}"


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
