"""Safe rclone subprocess adapter for Google Drive copy operations."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path, PurePosixPath

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveConnectionResult,
    DriveConnectionStatus,
    DriveDestination,
    DriveSyncErrorCode,
    DriveSyncProgress,
    validate_remote_relative_path,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[DriveSyncProgress], Awaitable[None] | None]
ProcessStartedCallback = Callable[[int], Awaitable[None] | None]
ProcessFinishedCallback = Callable[[], Awaitable[None] | None]


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

    def build_connection_command(
        self, destination: DriveDestination | None = None
    ) -> tuple[str, ...]:
        resolved = destination or self._settings_destination()
        return (*self._global_args(), "lsd", self._remote_root(resolved))

    def build_copy_command(
        self,
        local_path: Path,
        destination: DriveDestination,
        relative_remote_path: str,
    ) -> tuple[str, ...]:
        safe_relative = validate_remote_relative_path(relative_remote_path)
        return (
            *self._global_args(),
            "copyto",
            str(local_path),
            self._remote_path(destination, safe_relative),
            "--use-json-log",
            "--stats",
            "1s",
        )

    def build_copy_from_remote_command(
        self,
        destination: DriveDestination,
        relative_remote_path: str,
        local_path: Path,
    ) -> tuple[str, ...]:
        safe_relative = validate_remote_relative_path(relative_remote_path)
        return (
            *self._global_args(),
            "copyto",
            self._remote_path(destination, safe_relative),
            str(local_path),
        )

    async def check_connection(self) -> DriveConnectionResult:
        if not self._settings.rclone_remote:
            return DriveConnectionResult(
                DriveConnectionStatus.NOT_CONFIGURED,
                "Google Drive未設定",
                DriveSyncErrorCode.NOT_CONFIGURED.value,
            )
        try:
            await self._run(
                self.build_connection_command(),
                timeout_seconds=self._settings.rclone_connection_timeout_seconds,
                timeout_code=DriveSyncErrorCode.CONNECTION_FAILED.value,
                operation="connection",
            )
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
        destination: DriveDestination,
        relative_remote_path: str,
        *,
        progress_callback: ProgressCallback | None = None,
        total_bytes: int = 0,
        current_artifact: str | None = None,
        process_started_callback: ProcessStartedCallback | None = None,
        process_finished_callback: ProcessFinishedCallback | None = None,
        log_path: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        await self._run(
            self.build_copy_command(local_path, destination, relative_remote_path),
            timeout_seconds=(
                self._settings.rclone_transfer_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            ),
            timeout_code=DriveSyncErrorCode.TRANSFER_FAILED.value,
            progress_callback=progress_callback,
            total_bytes=total_bytes,
            current_artifact=current_artifact,
            process_started_callback=process_started_callback,
            process_finished_callback=process_finished_callback,
            log_path=log_path,
            operation="copyto",
        )

    async def copy_from_remote(
        self,
        destination: DriveDestination,
        relative_remote_path: str,
        local_path: Path,
    ) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run(
            self.build_copy_from_remote_command(destination, relative_remote_path, local_path),
            timeout_seconds=self._settings.state_sync_download_timeout_seconds,
            timeout_code=DriveSyncErrorCode.TRANSFER_FAILED.value,
            operation="copyfrom",
        )

    async def _run(
        self,
        command: tuple[str, ...],
        *,
        timeout_seconds: float | None,
        timeout_code: str,
        progress_callback: ProgressCallback | None = None,
        total_bytes: int = 0,
        current_artifact: str | None = None,
        process_started_callback: ProcessStartedCallback | None = None,
        process_finished_callback: ProcessFinishedCallback | None = None,
        log_path: str | None = None,
        operation: str,
    ) -> tuple[str, str]:
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

        log_target = self._resolve_log_path(log_path)
        self._append_log(log_target, f"operation_started={_safe_operation(operation)}")
        stdout_task = asyncio.create_task(
            self._consume_stream(
                process.stdout,
                progress_callback=progress_callback,
                total_bytes=total_bytes,
                current_artifact=current_artifact,
                log_target=log_target,
            )
        )
        stderr_task = asyncio.create_task(
            self._consume_stream(
                process.stderr,
                progress_callback=progress_callback,
                total_bytes=total_bytes,
                current_artifact=current_artifact,
                log_target=log_target,
            )
        )
        callback_error: Exception | None = None
        timed_out = False
        try:
            self._append_log(log_target, "pid_started=true")
            if process_started_callback is not None:
                try:
                    await _invoke(process_started_callback, process.pid)
                except Exception as exc:  # noqa: BLE001 - untracked processes must be stopped
                    callback_error = exc
                    process.kill()
            try:
                if timeout_seconds is None:
                    await process.wait()
                else:
                    await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except TimeoutError:
                timed_out = True
                process.kill()
                await process.wait()
            stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        finally:
            if not stdout_task.done():
                stdout_task.cancel()
            if not stderr_task.done():
                stderr_task.cancel()
            for task in (stdout_task, stderr_task):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            if process_finished_callback is not None:
                try:
                    await _invoke(process_finished_callback)
                except Exception:  # noqa: BLE001 - best effort PID cleanup
                    logger.warning("rclone process-finished callback failed", exc_info=True)
            self._append_log(log_target, f"operation_finished={_safe_operation(operation)}")

        if callback_error is not None:
            self._append_log(
                log_target,
                f"operation_error={DriveSyncErrorCode.PERSISTENCE_FAILED.value}",
            )
            raise GoogleDriveAdapterError(
                DriveSyncErrorCode.PERSISTENCE_FAILED.value,
                "rclone process state could not be persisted",
            ) from callback_error
        if timed_out:
            self._append_log(log_target, f"operation_error={_safe_error_code(timeout_code)}")
            raise GoogleDriveAdapterError(timeout_code, "rclone operation timed out")
        if process.returncode != 0:
            code = _classify_rclone_failure(stderr)
            logger.warning("rclone operation failed code=%s", code)
            self._append_log(log_target, f"operation_error={_safe_error_code(code)}")
            raise GoogleDriveAdapterError(code, "rclone operation failed")
        return stdout, stderr

    async def _consume_stream(
        self,
        stream: asyncio.StreamReader | None,
        *,
        progress_callback: ProgressCallback | None,
        total_bytes: int,
        current_artifact: str | None,
        log_target: Path | None,
    ) -> str:
        if stream is None:
            return ""
        chunks: list[str] = []
        while True:
            raw_line = await stream.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            chunks.append(line)
            progress = _progress_from_line(line, total_bytes, current_artifact)
            if progress is not None:
                self._append_log(log_target, _safe_progress_event(progress))
                if progress_callback is not None:
                    try:
                        await _invoke(progress_callback, progress)
                    except Exception:  # noqa: BLE001 - progress persistence is best effort
                        logger.warning("rclone progress callback failed", exc_info=True)
        return "\n".join(chunks)

    def _settings_destination(self) -> DriveDestination:
        return DriveDestination(self._settings.rclone_remote, self._settings.rclone_base_path)

    def _global_args(self) -> tuple[str, ...]:
        args = [self._executable]
        if self._settings.rclone_config is not None:
            args.extend(("--config", str(self._settings.rclone_config)))
        return tuple(args)

    @staticmethod
    def _remote_root(destination: DriveDestination) -> str:
        return GoogleDriveAdapter._remote_path(destination, "")

    @staticmethod
    def _remote_path(destination: DriveDestination, relative_path: str) -> str:
        base = destination.base_path.strip().strip("/")
        if relative_path:
            return (
                f"{destination.remote_name}:{base}/{relative_path}"
                if base
                else f"{destination.remote_name}:{relative_path}"
            )
        return f"{destination.remote_name}:{base}" if base else f"{destination.remote_name}:"

    def _resolve_log_path(self, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        normalized = relative_path.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            return None
        root = self._settings.data_dir.resolve()
        candidate = root.joinpath(*path.parts)
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError):
            return None
        return resolved

    @staticmethod
    def _append_log(path: Path | None, line: str) -> None:
        if path is None:
            return
        try:
            with path.open("a", encoding="utf-8") as file:
                file.write(line[:4000] + "\n")
        except OSError:
            logger.warning("rclone subprocess log could not be written", exc_info=True)


async def _invoke(callback: Callable[..., Awaitable[None] | None], *args: object) -> None:
    result = callback(*args)
    if asyncio.iscoroutine(result):
        await result


def _classify_rclone_failure(stderr: str) -> str:
    normalized = stderr.lower()
    if any(token in normalized for token in ("auth", "unauthorized", "forbidden", "401", "403")):
        return "drive_authentication_failed"
    if any(
        token in normalized
        for token in (
            "not found",
            "no such file",
            "doesn't exist",
            "does not exist",
            "directory not found",
            "file not found",
            "object not found",
        )
    ):
        return "remote_not_found"
    return DriveSyncErrorCode.TRANSFER_FAILED.value


def _progress_from_line(
    line: str, total_bytes: int, current_artifact: str | None
) -> DriveSyncProgress | None:
    try:
        value = json.loads(line)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None

    raw_stats = value.get("stats")
    if raw_stats is None:
        stats: Mapping[str, object] = value
    elif isinstance(raw_stats, Mapping):
        stats = raw_stats
    else:
        return None

    raw_bytes = stats.get("bytes")
    progress_bytes = _coerce_counter(raw_bytes)
    if progress_bytes is None:
        return None

    raw_total = stats.get("totalBytes")
    resolved_total_from_stats: int | None = None
    if isinstance(raw_total, (int, float)) and not isinstance(raw_total, bool):
        resolved_total_from_stats = _coerce_counter(raw_total)
        if resolved_total_from_stats is None:
            return None

    raw_percentage = stats.get("percentage")
    percentage: float | None = None
    if isinstance(raw_percentage, (int, float)) and not isinstance(raw_percentage, bool):
        try:
            percentage = float(raw_percentage)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(percentage):
            return None

    resolved_total = resolved_total_from_stats or max(total_bytes, 0)
    resolved_total = max(resolved_total, progress_bytes)
    if percentage is None:
        percentage = progress_bytes * 100.0 / resolved_total if resolved_total else 0.0
    return DriveSyncProgress(
        progress_bytes,
        resolved_total,
        max(0.0, min(100.0, percentage)),
        current_artifact,
    )


def _coerce_counter(value: object) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if converted < 0 or converted != value:
        return None
    return converted


_SAFE_ERROR_CODES = frozenset(
    {error.value for error in DriveSyncErrorCode}
    | {"drive_authentication_failed", "remote_not_found", "operation_timeout", "operation_failed"}
)
_SAFE_OPERATIONS = frozenset({"connection", "copyto", "copyfrom"})
_SAFE_ARTIFACTS = frozenset({"image", "metadata", "manifest"})


def _safe_error_code(value: str) -> str:
    return value if value in _SAFE_ERROR_CODES else "operation_failed"


def _safe_operation(value: str) -> str:
    return value if value in _SAFE_OPERATIONS else "operation"


def _safe_progress_event(progress: DriveSyncProgress) -> str:
    artifact = (
        progress.current_artifact if progress.current_artifact in _SAFE_ARTIFACTS else "unknown"
    )
    return (
        f"progress_bytes={progress.progress_bytes} "
        f"total_bytes={progress.total_bytes} "
        f"percentage={progress.progress_percentage:.1f} "
        f"current_artifact={artifact}"
    )


__all__ = [
    "GoogleDriveAdapter",
    "GoogleDriveAdapterError",
    "ProcessFinishedCallback",
    "ProcessStartedCallback",
    "ProgressCallback",
]
