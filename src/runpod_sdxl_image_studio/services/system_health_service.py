"""Application service that aggregates operational state for the System tab."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from runpod_sdxl_image_studio.adapters.storage.disk_usage import (
    DiskUsageAdapterProtocol,
    LocalDiskUsageAdapter,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.system_status import (
    ComfyUIHealthView,
    ComfyUIStatus,
    DriveHealthView,
    ErrorSeverity,
    GenerationQueueHealthView,
    ModelHealthView,
    StorageHealthView,
    SystemErrorEvent,
    SystemHealthStatus,
    SystemHealthView,
)

logger = logging.getLogger(__name__)


class ComfyUIHealthProvider(Protocol):
    async def get_status(self) -> ComfyUIStatus: ...


class QueueHealthProvider(Protocol):
    def list_jobs(self, *, limit: int = 200) -> tuple[object, ...]: ...


class DriveHealthProvider(Protocol):
    @property
    def is_configured(self) -> bool: ...

    async def check_connection(self) -> object: ...

    def status_counts(self) -> Any: ...

    def list_jobs(self, limit: int = 50) -> Sequence[object]: ...

    def capacity(self) -> object: ...


class ErrorHistoryProvider(Protocol):
    def list_recent(self, limit: int = 100) -> tuple[SystemErrorEvent, ...]: ...


class SystemHealthService:
    """Read system health through application-service boundaries only."""

    def __init__(
        self,
        comfyui_service: ComfyUIHealthProvider,
        queue_service: QueueHealthProvider,
        drive_sync_service: DriveHealthProvider | None,
        settings: Settings,
        *,
        disk_usage_adapter: DiskUsageAdapterProtocol | None = None,
        error_history_repository: ErrorHistoryProvider | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._comfyui_service = comfyui_service
        self._queue_service = queue_service
        self._drive_sync_service = drive_sync_service
        self._settings = settings
        self._disk_usage_adapter = disk_usage_adapter or LocalDiskUsageAdapter()
        self._error_history_repository = error_history_repository
        self._now_factory = now_factory or (lambda: datetime.now(UTC))

    async def get_health(self) -> SystemHealthView:
        """Collect one bounded snapshot; no continuous polling is performed here."""

        checked_at = _utc(self._now_factory())
        comfy_status, models, comfy_warning, comfy_error = await self._read_comfyui()
        queue_items = self._read_queue()
        storage = self._read_storage()
        drive, drive_unsynced = await self._read_drive()
        if drive_unsynced is not None:
            storage = StorageHealthView(
                storage.local_total_bytes,
                storage.local_free_bytes,
                storage.local_used_bytes,
                drive_unsynced,
            )
        errors = self._read_error_history(queue_items, drive)
        return SystemHealthView(
            checked_at=checked_at,
            overall_status=_overall_status(
                comfy_status,
                storage,
                queue_items,
                drive,
                self._settings,
                comfy_warning,
                comfy_error,
            ),
            comfyui=comfy_status,
            queue=_queue_health(queue_items),
            storage=storage,
            drive=drive,
            models=models,
            recent_errors=errors,
        )

    async def aggregate(self) -> SystemHealthView:
        """Alias for callers that name the operation after its role."""

        return await self.get_health()

    async def get_status(self) -> SystemHealthView:
        """Alias used by status-oriented application code."""

        return await self.get_health()

    async def _read_comfyui(
        self,
    ) -> tuple[ComfyUIHealthView, ModelHealthView, bool, bool]:
        try:
            status = await self._comfyui_service.get_status()
        except Exception:  # noqa: BLE001 - health view fails closed
            logger.warning("System health could not read ComfyUI status", exc_info=True)
            return (
                ComfyUIHealthView(False, "ComfyUI status unavailable", None, None, None, None),
                ModelHealthView(0, 0, 0, 0),
                False,
                True,
            )
        stats = status.system_stats
        device = stats.devices[0] if stats is not None and stats.devices else None
        return (
            ComfyUIHealthView(
                connected=status.is_connected,
                message=status.message,
                version=stats.comfyui_version if stats is not None else None,
                gpu_name=device.name if device is not None else None,
                vram_total=device.vram_total if device is not None else None,
                vram_free=device.vram_free if device is not None else None,
            ),
            ModelHealthView(
                checkpoint_count=len(status.capabilities.checkpoints)
                if status.capabilities is not None
                else 0,
                lora_count=len(status.capabilities.loras) if status.capabilities is not None else 0,
                vae_count=len(status.capabilities.vaes) if status.capabilities is not None else 0,
                upscaler_count=(
                    len(status.capabilities.upscale_models)
                    if status.capabilities is not None
                    else 0
                ),
            ),
            bool(status.warnings),
            not status.is_connected
            or status.capabilities is None
            or status.error_summary is not None,
        )

    def _read_queue(self) -> tuple[object, ...]:
        try:
            return self._queue_service.list_jobs(limit=500)
        except Exception:  # noqa: BLE001 - a missing queue count must not crash UI
            logger.warning("System health could not read generation queue", exc_info=True)
            return ()

    def _read_storage(self) -> StorageHealthView:
        try:
            usage = self._disk_usage_adapter.usage(self._settings.data_dir)
            return StorageHealthView(
                usage.total_bytes,
                usage.free_bytes,
                usage.used_bytes,
                0,
            )
        except Exception:  # noqa: BLE001 - health view fails closed
            logger.warning("System health could not read local disk usage", exc_info=True)
            return StorageHealthView(0, 0, 0, 0)

    async def _read_drive(self) -> tuple[DriveHealthView, int | None]:
        service = self._drive_sync_service
        if service is None:
            return DriveHealthView(False, False, None, 0, 0), 0
        configured = bool(service.is_configured)
        connected = False
        if configured:
            try:
                result = await service.check_connection()
                connected = bool(getattr(result, "connected", False)) or (
                    getattr(getattr(result, "status", None), "value", None) == "connected"
                )
            except Exception:  # noqa: BLE001 - Drive is an independent subsystem
                logger.warning("System health could not check Google Drive", exc_info=True)
        pending = 0
        failed = 0
        try:
            counts = service.status_counts()
            for key, count in counts.items():
                value = getattr(key, "value", key)
                if value in {"pending", "syncing"}:
                    pending += int(count)
                elif value == "failed":
                    failed += int(count)
        except Exception:  # noqa: BLE001
            logger.warning("System health could not read Drive sync counts", exc_info=True)
        last_sync_at: datetime | None = None
        try:
            jobs = service.list_jobs(100)
            for job in jobs:
                value = getattr(getattr(job, "status", None), "value", None)
                completed_at = getattr(job, "completed_at", None)
                if (
                    value == "synced"
                    and completed_at is not None
                    and (last_sync_at is None or completed_at > last_sync_at)
                ):
                    last_sync_at = completed_at
        except Exception:  # noqa: BLE001
            logger.warning("System health could not read last Drive sync", exc_info=True)
        unsynced: int | None = None
        try:
            capacity = service.capacity()
            unsynced_value = getattr(capacity, "unsynced_bytes", None)
            if isinstance(unsynced_value, int):
                unsynced = max(0, unsynced_value)
        except Exception:  # noqa: BLE001
            logger.warning("System health could not calculate unsynced bytes", exc_info=True)
        return (
            DriveHealthView(configured, connected, last_sync_at, pending, failed),
            unsynced,
        )

    def _read_error_history(
        self,
        queue_items: tuple[object, ...],
        drive: DriveHealthView,
    ) -> tuple[SystemErrorEvent, ...]:
        events: list[SystemErrorEvent] = []
        if self._error_history_repository is not None:
            try:
                events.extend(self._error_history_repository.list_recent(100))
            except Exception:  # noqa: BLE001
                logger.warning("System error history could not be listed", exc_info=True)
        for item in queue_items:
            generation = getattr(item, "generation", None)
            job = getattr(item, "job", None)
            if (
                generation is None
                or getattr(generation, "status", None) is not GenerationStatus.FAILED
            ):
                continue
            generation_id = getattr(generation, "id", None)
            if not isinstance(generation_id, UUID):
                continue
            events.append(
                SystemErrorEvent(
                    id=uuid5(
                        NAMESPACE_URL,
                        "generation-failure:"
                        f"{generation_id}:{getattr(generation, 'updated_at', '')}",
                    ),
                    created_at=_event_time(generation, self._now_factory()),
                    category="generation",
                    severity=ErrorSeverity.ERROR,
                    error_code=getattr(generation, "error_code", None) or "generation_failed",
                    summary=getattr(generation, "error_summary", None) or "Generation failed",
                    generation_id=generation_id,
                    job_id=getattr(job, "id", None),
                    retryable=True,
                )
            )
        if drive.failed_sync_count:
            events.append(
                SystemErrorEvent(
                    id=uuid5(NAMESPACE_URL, "drive-sync-failed"),
                    created_at=_utc(self._now_factory()),
                    category="drive_sync",
                    severity=ErrorSeverity.ERROR,
                    error_code="drive_sync_failed",
                    summary=f"{drive.failed_sync_count} Drive synchronization job(s) failed",
                    generation_id=None,
                    job_id=None,
                    retryable=True,
                )
            )
        events.sort(key=lambda event: event.created_at, reverse=True)
        return tuple(events[:100])


def _queue_health(items: tuple[object, ...]) -> GenerationQueueHealthView:
    pending = running = failed = 0
    for item in items:
        status = getattr(getattr(item, "generation", None), "status", None)
        value = getattr(status, "value", status)
        if value in {GenerationStatus.PENDING.value, GenerationStatus.QUEUED.value}:
            pending += 1
        elif value == GenerationStatus.RUNNING.value:
            running += 1
        elif value == GenerationStatus.FAILED.value:
            failed += 1
    return GenerationQueueHealthView(pending, running, failed)


def _overall_status(
    comfyui: ComfyUIHealthView,
    storage: StorageHealthView,
    queue_items: tuple[object, ...],
    drive: DriveHealthView,
    settings: Settings,
    comfy_warning: bool,
    comfy_error: bool,
) -> SystemHealthStatus:
    if comfy_error or storage.local_free_bytes < settings.min_free_disk_bytes:
        return SystemHealthStatus.ERROR
    queue = _queue_health(queue_items)
    if (
        queue.failed_count > 0
        or drive.failed_sync_count > 0
        or (drive.configured and not drive.connected)
        or storage.local_free_bytes < settings.warning_free_disk_bytes
        or comfy_warning
    ):
        return SystemHealthStatus.WARNING
    return SystemHealthStatus.HEALTHY


def _event_time(value: object, fallback: datetime) -> datetime:
    candidate = getattr(value, "updated_at", None) or getattr(value, "completed_at", None)
    return _utc(candidate if isinstance(candidate, datetime) else fallback)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["SystemHealthService"]
