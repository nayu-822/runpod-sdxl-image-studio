"""Safe rclone adapter for remote model catalogs and temporary downloads."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import validate_remote_relative_path
from runpod_sdxl_image_studio.domain.model_transfer import (
    ModelTransferErrorCode,
    ModelTransferProgress,
    RemoteModelCatalog,
    RemoteModelEntry,
    RemoteModelKind,
    is_supported_model_filename,
    normalize_model_relative_path,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[ModelTransferProgress], Awaitable[None] | None]
ProcessStartedCallback = Callable[[int], Awaitable[None] | None]
ProcessFinishedCallback = Callable[[], Awaitable[None] | None]
CancelCheck = Callable[[], Awaitable[bool] | bool]
ShutdownCheck = Callable[[], Awaitable[bool] | bool]
_REMOTE_CATEGORY_NOT_FOUND = "remote_model_category_not_found"


class RemoteModelAdapterError(RuntimeError):
    """Stable adapter error without raw command, path, or stderr details."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RemoteModelCatalogAdapter:
    """List configured Drive model roots and copy one model to a temp path."""

    def __init__(self, settings: Settings, *, executable: str = "rclone") -> None:
        self._settings = settings
        self._executable = executable

    def build_list_command(self, kind: RemoteModelKind) -> tuple[str, ...]:
        return (
            *self._global_args(),
            "lsjson",
            self._remote_path(kind),
            "--recursive",
            "--files-only",
            "--hash",
        )

    def build_download_command(self, entry: RemoteModelEntry, destination: Path) -> tuple[str, ...]:
        relative = normalize_model_relative_path(entry.relative_path)
        return (
            *self._global_args(),
            "copyto",
            self._remote_file_path(entry.kind, relative),
            str(destination),
            "--stats-one-line-json",
            "--stats",
            "1s",
        )

    async def list_catalog(self) -> RemoteModelCatalog:
        if not self._settings.rclone_remote:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.NOT_CONFIGURED.value,
                "Google Drive model catalog is not configured",
            )
        entries: list[RemoteModelEntry] = []
        try:
            for kind in RemoteModelKind:
                try:
                    stdout = await self._run_json(
                        self.build_list_command(kind),
                        timeout=self._settings.remote_model_list_timeout_seconds,
                        error_code=ModelTransferErrorCode.CATALOG_UNAVAILABLE.value,
                    )
                except RemoteModelAdapterError as exc:
                    if exc.code == _REMOTE_CATEGORY_NOT_FOUND:
                        continue
                    raise
                entries.extend(self._parse_entries(kind, stdout))
        except RemoteModelAdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 - safe catalog boundary
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.CATALOG_UNAVAILABLE.value,
                "Google Drive model catalog is unavailable",
            ) from exc
        if len(entries) > self._settings.remote_model_max_catalog_items:
            entries = entries[: self._settings.remote_model_max_catalog_items]
        entries.sort(key=lambda item: (item.kind.value, item.relative_path.casefold()))
        return RemoteModelCatalog(tuple(entries), datetime.now(UTC))

    async def download(
        self,
        entry: RemoteModelEntry,
        destination: Path,
        *,
        progress_callback: ProgressCallback | None = None,
        process_started_callback: ProcessStartedCallback | None = None,
        process_finished_callback: ProcessFinishedCallback | None = None,
        cancel_check: CancelCheck | None = None,
        shutdown_check: ShutdownCheck | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        command = self.build_download_command(entry, destination)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.DOWNLOAD_FAILED.value,
                "rclone executable was not found",
            ) from exc

        callback_error: Exception | None = None
        finished_callback_error: Exception | None = None
        cancelled = False
        interrupted = False
        timed_out = False
        cleanup_error: Exception | None = None
        stream_tasks: dict[asyncio.Task[bytes], asyncio.StreamReader] = {}
        try:
            if process_started_callback is not None:
                try:
                    await _invoke(process_started_callback, process.pid)
                except Exception as exc:  # noqa: BLE001 - cannot track an unpersisted process
                    callback_error = exc

            if callback_error is None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream_tasks[asyncio.create_task(stream.readline())] = stream
                started_at = asyncio.get_running_loop().time()
                while stream_tasks or process.returncode is None:
                    if (
                        shutdown_check is not None
                        and await _invoke(shutdown_check)
                        and process.returncode is None
                    ):
                        interrupted = True
                        await _terminate_process(process)
                    elif (
                        cancel_check is not None
                        and await _invoke(cancel_check)
                        and process.returncode is None
                    ):
                        cancelled = True
                        await _terminate_process(process)
                    if (
                        timeout_seconds is not None
                        and asyncio.get_running_loop().time() - started_at > timeout_seconds
                        and process.returncode is None
                    ):
                        timed_out = True
                        await _terminate_process(process)
                    if stream_tasks:
                        done, _ = await asyncio.wait(
                            stream_tasks,
                            timeout=0.25,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            stream = stream_tasks.pop(task, None)
                            raw_line = task.result()
                            if raw_line:
                                progress = _progress_from_line(
                                    raw_line.decode("utf-8", errors="replace"),
                                    entry.size_bytes,
                                )
                                if progress is not None and progress_callback is not None:
                                    await _invoke(progress_callback, progress)
                                if stream is not None:
                                    stream_tasks[asyncio.create_task(stream.readline())] = stream
                            else:
                                # An EOF task is intentionally not re-added.
                                continue
                    elif process.returncode is None:
                        await process.wait()
        finally:
            # Any callback, stream, or internal error must still reap rclone before
            # the PID cleanup callback is allowed to run.
            if process.returncode is None:
                try:
                    await _terminate_process(process)
                except Exception as exc:  # noqa: BLE001 - preserve cleanup failure safely
                    cleanup_error = exc
            for task in stream_tasks:
                task.cancel()
            for task in stream_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            if process_finished_callback is not None and process.returncode is not None:
                try:
                    await _invoke(process_finished_callback)
                except Exception as exc:  # noqa: BLE001 - report callback failures safely
                    finished_callback_error = exc

        if cleanup_error is not None and process.returncode is None:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.DOWNLOAD_FAILED.value,
                "model transfer process could not be stopped",
            ) from cleanup_error

        if callback_error is not None:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                "model transfer process state could not be persisted",
            ) from callback_error
        if finished_callback_error is not None:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.PERSISTENCE_FAILED.value,
                "model transfer process state could not be persisted",
            ) from finished_callback_error
        if cancelled:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.CANCELLED.value,
                "model transfer was cancelled",
            )
        if interrupted:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value,
                "model transfer was interrupted during application shutdown",
            )
        if timed_out:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.DOWNLOAD_TIMEOUT.value,
                "model transfer timed out",
            )
        if process.returncode != 0:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.DOWNLOAD_FAILED.value,
                "model transfer failed",
            )

    async def _run_json(
        self,
        command: tuple[str, ...],
        *,
        timeout: float,
        error_code: str,
    ) -> Any:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except FileNotFoundError as exc:
            raise RemoteModelAdapterError(error_code, "rclone executable was not found") from exc
        except TimeoutError as exc:
            if process is not None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await process.wait()
            raise RemoteModelAdapterError(
                error_code, "remote model catalog request timed out"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RemoteModelAdapterError(
                error_code, "remote model catalog request failed"
            ) from exc
        if process.returncode != 0:
            failure_code = _classify_list_failure(stderr)
            if failure_code == _REMOTE_CATEGORY_NOT_FOUND:
                raise RemoteModelAdapterError(
                    _REMOTE_CATEGORY_NOT_FOUND,
                    "remote model category was not found",
                )
            raise RemoteModelAdapterError(error_code, "remote model catalog request failed")
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteModelAdapterError(
                error_code, "remote model catalog response was invalid"
            ) from exc
        if not isinstance(payload, list):
            raise RemoteModelAdapterError(error_code, "remote model catalog response was invalid")
        return payload

    def _parse_entries(self, kind: RemoteModelKind, payload: Any) -> list[RemoteModelEntry]:
        if not isinstance(payload, list):
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.CATALOG_UNAVAILABLE.value,
                "remote model catalog response was invalid",
            )
        entries: list[RemoteModelEntry] = []
        for item in payload:
            if not isinstance(item, Mapping) or item.get("IsDir") is True:
                continue
            # rclone's Name is basename-only for nested entries.  Path is the
            # category-relative identity and must win whenever the backend sent it.
            raw_name = item.get("Path") if "Path" in item else item.get("Name")
            if not isinstance(raw_name, str):
                continue
            try:
                relative = normalize_model_relative_path(raw_name)
            except ValueError:
                logger.warning("unsafe remote model entry ignored kind=%s", kind.value)
                continue
            if not is_supported_model_filename(relative, kind):
                continue
            size = item.get("Size", 0)
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                continue
            hashes = item.get("Hashes", item.get("hashes", {}))
            algorithm, digest = _select_hash(hashes)
            modified = _parse_modified(item.get("ModTime", item.get("modtime")))
            try:
                entries.append(
                    RemoteModelEntry(
                        kind=kind,
                        relative_path=validate_remote_relative_path(relative),
                        display_name=relative,
                        size_bytes=size,
                        modified_at=modified,
                        remote_hash_algorithm=algorithm,
                        remote_hash=digest,
                    )
                )
            except ValueError:
                logger.warning("invalid remote model entry ignored kind=%s", kind.value)
        return entries

    def _global_args(self) -> tuple[str, ...]:
        args = [self._executable]
        if self._settings.rclone_config is not None:
            args.extend(("--config", str(self._settings.rclone_config)))
        return tuple(args)

    def _remote_path(self, kind: RemoteModelKind) -> str:
        base = self._settings.remote_model_base_path
        subdir = {
            RemoteModelKind.CHECKPOINT: self._settings.remote_checkpoint_subdir,
            RemoteModelKind.LORA: self._settings.remote_lora_subdir,
            RemoteModelKind.VAE: self._settings.remote_vae_subdir,
            RemoteModelKind.UPSCALER: self._settings.remote_upscaler_subdir,
        }[kind]
        return f"{self._settings.rclone_remote}:{base}/{subdir}"

    def _remote_file_path(self, kind: RemoteModelKind, relative: str) -> str:
        validate_remote_relative_path(relative)
        return f"{self._remote_path(kind)}/{relative}"


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Stop a transfer process without allowing a timeout to orphan it."""

    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()


async def _invoke(callback: Callable[..., Any], *args: Any) -> Any:
    result = callback(*args)
    if hasattr(result, "__await__"):
        return await result
    return result


def _select_hash(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, Mapping):
        return None, None
    for key in ("sha-256", "sha256", "SHA-256", "md5", "MD5"):
        digest = value.get(key)
        if isinstance(digest, str) and digest.strip():
            return key.casefold().replace("_", "-"), digest.strip().casefold()
    return None, None


def _parse_modified(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _classify_list_failure(stderr: bytes) -> str:
    """Classify rclone output into a stable internal category without exposing it."""

    normalized = stderr.decode("utf-8", errors="replace").casefold()
    if any(
        token in normalized
        for token in (
            "auth",
            "unauthorized",
            "forbidden",
            "401",
            "403",
        )
    ):
        return ModelTransferErrorCode.CATALOG_UNAVAILABLE.value
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
        return _REMOTE_CATEGORY_NOT_FOUND
    return ModelTransferErrorCode.CATALOG_UNAVAILABLE.value


_PROGRESS_RE = re.compile(r"(?P<percent>\d+(?:\.\d+)?)%")


def _progress_from_line(line: str, total_bytes: int) -> ModelTransferProgress | None:
    text = line.strip()
    if not text:
        return None
    progress_bytes: int | None = None
    percentage: float | None = None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        progress_bytes = _int_value(value, "bytes", "bytesTransferred", "transferred")
        percentage_value = value.get("percent", value.get("percentage"))
        if isinstance(percentage_value, (int, float)) and not isinstance(percentage_value, bool):
            percentage = float(percentage_value)
        total_value = _int_value(value, "totalBytes", "total")
        if total_value is not None and total_value > 0:
            total_bytes = total_value
    if percentage is None:
        match = _PROGRESS_RE.search(text)
        if match:
            percentage = float(match.group("percent"))
    if percentage is None and progress_bytes is None:
        return None
    if percentage is None and total_bytes > 0 and progress_bytes is not None:
        percentage = progress_bytes * 100 / total_bytes
    if progress_bytes is None and total_bytes > 0 and percentage is not None:
        progress_bytes = int(total_bytes * min(max(percentage, 0.0), 100.0) / 100)
    if progress_bytes is None or percentage is None:
        return None
    return ModelTransferProgress(
        min(max(progress_bytes, 0), total_bytes) if total_bytes else max(progress_bytes, 0),
        max(total_bytes, 0),
        min(max(percentage, 0.0), 100.0),
    )


def _int_value(value: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        if isinstance(candidate, float) and candidate.is_integer():
            return int(candidate)
    return None


__all__ = ["RemoteModelAdapterError", "RemoteModelCatalogAdapter"]
