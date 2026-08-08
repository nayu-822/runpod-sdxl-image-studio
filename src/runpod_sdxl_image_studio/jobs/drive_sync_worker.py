"""One-worker runtime for the independent Google Drive synchronization queue."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepositoryError,
    DriveSyncRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveSyncErrorCode,
    DriveSyncJob,
)
from runpod_sdxl_image_studio.services.drive_sync_service import (
    DriveSyncService,
    DriveSyncServiceError,
)

logger = logging.getLogger(__name__)


class DriveSyncWorker:
    """Run one leased Drive job at a time without depending on browser requests."""

    def __init__(
        self,
        repository: DriveSyncRepositoryProtocol,
        service: DriveSyncService,
        settings: Settings,
        *,
        worker_id: str | None = None,
    ) -> None:
        self._repository = repository
        self._service = service
        self._settings = settings
        self.worker_id = worker_id or f"drive-worker-{uuid4()}"
        self._stop_requested = threading.Event()
        self._wake_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_reconciled_at = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._wake_requested.clear()
        self._last_reconciled_at = 0.0
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self.run()),
            name="drive-sync-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested.set()
        self._wake_requested.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self._settings.drive_sync_lease_seconds))

    def wake(self) -> None:
        self._wake_requested.set()

    async def run(self) -> None:
        logger.info("drive sync worker started worker_id=%s", self.worker_id)
        await self.startup_reconcile()
        try:
            while not self._stop_requested.is_set():
                if time.monotonic() - self._last_reconciled_at >= max(
                    1.0, self._settings.drive_sync_poll_interval_seconds
                ):
                    await self.reconcile()
                if await self.run_once():
                    continue
                await asyncio.to_thread(
                    self._wake_requested.wait,
                    self._settings.drive_sync_poll_interval_seconds,
                )
                self._wake_requested.clear()
        finally:
            logger.info("drive sync worker stopped worker_id=%s", self.worker_id)

    async def startup_reconcile(self) -> None:
        await self.reconcile()
        if not self._settings.rclone_remote:
            return
        try:
            discovered = self._service.discover_missing(self._settings.drive_discovery_batch_size)
            if discovered:
                logger.info("drive sync startup discovery count=%s", len(discovered))
        except Exception as exc:  # noqa: BLE001 - discovery must not stop the worker
            logger.warning(
                "drive sync startup discovery failed error=%s",
                type(exc).__name__,
                exc_info=True,
            )

    async def reconcile(self) -> None:
        try:
            count = self._repository.reconcile_stale()
            if count:
                logger.info("drive sync stale jobs reconciled count=%s", count)
        except DriveSyncRepositoryError:
            logger.warning("drive sync stale reconciliation failed", exc_info=True)
        self._last_reconciled_at = time.monotonic()

    async def run_once(self) -> bool:
        if self._stop_requested.is_set():
            return False
        try:
            job = self._repository.claim_next(
                self.worker_id,
                lease_seconds=self._settings.drive_sync_lease_seconds,
            )
        except DriveSyncRepositoryError:
            logger.warning("drive sync job claim failed", exc_info=True)
            return False
        if job is None:
            return False
        heartbeat = asyncio.create_task(self._heartbeat(job.id))
        try:
            await self._service.process_job(job, self.worker_id)
        except DriveSyncServiceError as exc:
            self._mark_failed(job, exc.code, str(exc), retryable=exc.retryable)
        except Exception:  # noqa: BLE001 - one Drive failure must not stop the worker
            logger.error(
                "drive sync job failed generation=%s job=%s",
                job.generation_id,
                job.id,
                exc_info=True,
            )
            self._mark_failed(
                job,
                DriveSyncErrorCode.TRANSFER_FAILED.value,
                "Drive synchronization failed",
                retryable=True,
            )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        return True

    async def _heartbeat(self, job_id: UUID) -> None:
        while True:
            await asyncio.sleep(self._settings.drive_sync_heartbeat_seconds)
            try:
                self._repository.renew_lease(
                    job_id,
                    self.worker_id,
                    lease_seconds=self._settings.drive_sync_lease_seconds,
                )
            except DriveSyncRepositoryError:
                logger.warning("drive sync lease renewal failed", exc_info=True)

    def _mark_failed(
        self,
        job: DriveSyncJob,
        error_code: str,
        error_summary: str,
        *,
        retryable: bool,
    ) -> None:
        try:
            self._repository.mark_failed(
                job.id,
                self.worker_id,
                error_code,
                error_summary,
                retryable=retryable,
            )
        except DriveSyncRepositoryError:
            logger.warning("drive sync failure could not be persisted", exc_info=True)


class DriveSyncRuntime:
    """Own exactly one Drive sync worker for the application process."""

    def __init__(self, worker: DriveSyncWorker) -> None:
        self.worker = worker

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    def wake(self) -> None:
        self.worker.wake()


__all__ = ["DriveSyncRuntime", "DriveSyncWorker"]
