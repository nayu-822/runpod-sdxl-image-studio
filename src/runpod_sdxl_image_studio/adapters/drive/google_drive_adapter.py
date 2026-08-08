"""Safe rclone subprocess adapter for Google Drive copy operations."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveConnectionResult,
    DriveConnectionStatus,
    DriveSyncErrorCode,
    DriveSyncProgress,
    validate_remote_relative_path,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[DriveSyncProgress], Awaitable[None] | None]


class GoogleDriveAdapterError(RuntimeError):
    """Safe rclone error carrying a stable code but no raw command or stderr."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GoogleDriveAdapter:
    """Translate typed Drive operations into argument-array rclone calls."""

    def __init__(self, settings: Settings, *, executable: str = "rclone") -> None:
        self._settings = settings
        self._executable = executable

    def build_connection_command(self) -> tuple[str, ...]:
        return (*self._global_args(), "lsd", self._remote_root())

    def build_copy_command(self, local_path: Path, relative_remote_path: str) -> tuple[str, ...]:
        safe_relative = validate_remote_relative_path(relative_remote_path)
        return (
            *self._global_args(),
            "copyto",
            str(local_path),
            self._remote_path(safe_relative),
            "--stats-one-line-json",
            "--stats",
            "1s",
        )

    async def check_connection(self) -> DriveConnectionResult:
        if not self._settings.rclone_remote:
            return DriveConnectionResult(
                DriveConnectionStatus.NOT_CONFIGURED,
                "Google Drive未設定",
                DriveSyncErrorCode.NOT_CONFIGURED.value,
            )
        try:
            await self._run(self.build_connection_command())
        except GoogleDriveAdapterError as exc:
            if exc.code == DriveSyncErrorCode.RCLONE_NOT_FOUND.value:
                return DriveConnectionResult(
                    DriveConnectionStatus.RCLONE_NOT_FOUND,
                    "rcloneが見つかりません",
                    exc.code,
                )
            if exc.code == "drive_authentication_failed":
                return DriveConnectionResult(
                    DriveConnectionStatus.AUTH_FAILED,
                    "Google Drive認証に失敗しました",
                    exc.code,
                )
            return DriveConnectionResult(
                DriveConnectionStatus.FAILED,
                "Google Drive接続に失敗しました",
                exc.code,
            )
        return DriveConnectionResult(DriveConnectionStatus.CONNECTED, "接続済み")

    async def copy_file(
        self,
        local_path: Path,
        relative_remote_path: str,
        *,
        progress_callback: ProgressCallback | None = None,
        total_bytes: int = 0,
    ) -> None:
        try:
            stdout, _ = await self._run(self.build_copy_command(local_path, relative_remote_path))
        except GoogleDriveAdapterError:
            raise
        parsed = _progress_from_output(stdout, total_bytes)
        if progress_callback is not None and parsed is not None:
            result = progress_callback(parsed)
            if asyncio.iscoroutine(result):
                await result

    async def _run(self, command: tuple[str, ...]) -> tuple[str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise GoogleDriveAdapterError(
                DriveSyncErrorCode.RCLONE_NOT_FOUND.value, "rclone executable was not found"
            ) from exc
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self._settings.rclone_connection_timeout_seconds
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise GoogleDriveAdapterError(
                DriveSyncErrorCode.CONNECTION_FAILED.value, "rclone operation timed out"
            ) from exc
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            code = _classify_rclone_failure(stderr)
            logger.warning("rclone operation failed code=%s", code)
            raise GoogleDriveAdapterError(code, "rclone operation failed")
        return stdout, stderr

    def _global_args(self) -> tuple[str, ...]:
        args = [self._executable]
        if self._settings.rclone_config is not None:
            args.extend(("--config", str(self._settings.rclone_config)))
        return tuple(args)

    def _remote_root(self) -> str:
        return self._remote_path("")

    def _remote_path(self, relative_path: str) -> str:
        base = self._settings.rclone_base_path.strip().strip("/")
        if relative_path:
            return (
                f"{self._settings.rclone_remote}:{base}/{relative_path}"
                if base
                else f"{self._settings.rclone_remote}:{relative_path}"
            )
        return (
            f"{self._settings.rclone_remote}:{base}" if base else f"{self._settings.rclone_remote}:"
        )


def _classify_rclone_failure(stderr: str) -> str:
    normalized = stderr.lower()
    if any(token in normalized for token in ("auth", "unauthorized", "forbidden", "401", "403")):
        return "drive_authentication_failed"
    return DriveSyncErrorCode.TRANSFER_FAILED.value


def _progress_from_output(output: str, total_bytes: int) -> DriveSyncProgress | None:
    latest: dict[str, object] | None = None
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "bytes" in value:
            latest = value
    if latest is None:
        return None
    raw_bytes = latest.get("bytes")
    raw_total = latest.get("totalBytes")
    raw_percentage = latest.get("percentage")
    if not isinstance(raw_bytes, (int, float)):
        return None
    resolved_total = int(raw_total) if isinstance(raw_total, (int, float)) else total_bytes
    if resolved_total <= 0:
        resolved_total = max(total_bytes, int(raw_bytes))
    percentage = (
        float(raw_percentage)
        if isinstance(raw_percentage, (int, float))
        else min(100.0, int(raw_bytes) * 100.0 / resolved_total)
    )
    return DriveSyncProgress(int(raw_bytes), resolved_total, max(0.0, min(100.0, percentage)), None)


__all__ = ["GoogleDriveAdapter", "GoogleDriveAdapterError", "ProgressCallback"]
