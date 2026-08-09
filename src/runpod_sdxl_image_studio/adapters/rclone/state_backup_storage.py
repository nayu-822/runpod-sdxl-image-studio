"""Safe remote file storage for the SQLite state snapshot protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from runpod_sdxl_image_studio.adapters.drive.google_drive_adapter import (
    GoogleDriveAdapter,
    GoogleDriveAdapterError,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveDestination,
    validate_remote_relative_path,
)


class StateBackupStorageError(RuntimeError):
    """Safe state backup transfer failure."""


class StateBackupNotFound(StateBackupStorageError):
    """The requested remote state object was confirmed to be absent."""


class StateBackupUnavailable(StateBackupStorageError):
    """The remote state object could not be read because the service was unavailable."""


class StateBackupTransferAdapter(Protocol):
    async def copy_file(
        self,
        local_path: Path,
        destination: DriveDestination,
        relative_remote_path: str,
        *,
        total_bytes: int = 0,
        current_artifact: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None: ...

    async def copy_from_remote(
        self,
        destination: DriveDestination,
        relative_remote_path: str,
        local_path: Path,
    ) -> None: ...


class StateBackupStorage:
    """Map state-relative paths below the configured Drive state directory."""

    def __init__(
        self,
        settings: Settings,
        adapter: StateBackupTransferAdapter | None = None,
    ) -> None:
        self._settings = settings
        self._adapter = adapter or GoogleDriveAdapter(settings)

    @property
    def is_configured(self) -> bool:
        return bool(self._settings.rclone_remote)

    async def upload(self, local_path: Path, relative_path: str) -> None:
        if not self.is_configured:
            raise StateBackupStorageError("state backup remote is not configured")
        await self._adapter.copy_file(
            local_path,
            self._destination(),
            self._remote_state_path(relative_path),
            total_bytes=local_path.stat().st_size,
            current_artifact="state",
            timeout_seconds=self._settings.state_sync_upload_timeout_seconds,
        )

    async def download(self, relative_path: str, local_path: Path) -> None:
        if not self.is_configured:
            raise StateBackupStorageError("state backup remote is not configured")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._adapter.copy_from_remote(
                self._destination(),
                self._remote_state_path(relative_path),
                local_path,
            )
        except GoogleDriveAdapterError as exc:
            if exc.code == "remote_not_found":
                raise StateBackupNotFound("remote state object was not found") from exc
            raise StateBackupUnavailable("remote state transfer was unavailable") from exc
        except StateBackupStorageError:
            raise
        except Exception as exc:  # noqa: BLE001 - adapter failures stay fail-closed
            raise StateBackupUnavailable("remote state transfer was unavailable") from exc

    def _destination(self) -> DriveDestination:
        return DriveDestination(self._settings.rclone_remote, self._settings.rclone_base_path)

    def _remote_state_path(self, relative_path: str) -> str:
        normalized = validate_remote_relative_path(relative_path)
        return validate_remote_relative_path(f"{self._settings.state_sync_subdir}/{normalized}")


__all__ = [
    "StateBackupNotFound",
    "StateBackupStorage",
    "StateBackupStorageError",
    "StateBackupTransferAdapter",
    "StateBackupUnavailable",
]
