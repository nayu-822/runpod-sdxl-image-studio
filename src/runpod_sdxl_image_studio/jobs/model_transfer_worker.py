"""One-worker runtime for persistent remote model preparation jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.adapters.database.repositories.model_transfer_repository import (
    ModelTransferRepositoryError,
    ModelTransferRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.model_transfer import (
    ModelTransferErrorCode,
    ModelTransferJob,
)
from runpod_sdxl_image_studio.services.model_preparation_service import (
    ModelPreparationService,
    ModelPreparationServiceError,
)

logger = logging.getLogger(__name__)


class ModelTransferWorker:
    """Continue model transfers after browser disconnects without duplicate claims."""

    def __init__(
        self,
        repository: ModelTransferRepositoryProtocol,
        service: ModelPreparationService,
        settings: Settings,
        *,
        worker_id: str | None = None,
        state_changed_callback: Callable[[], None] | None = None,
    ) -> None:
        self._repository = repository
        self._service = service
        self._settings = settings
        self.worker_id = worker_id or f"model-worker-{uuid4()}"
        self._state_changed_callback = state_changed_callback
        self._stop_requested = threading.Event()
        self._wake_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_job_id: UUID | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._wake_requested.clear()
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self.run()),
            name="model-transfer-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Request a graceful subprocess stop and wait for the worker to observe it."""

        self._stop_requested.set()
        self._wake_requested.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(6.0, self._settings.drive_sync_lease_seconds))
            if self._thread.is_alive():
                logger.warning(
                    "model transfer worker did not stop within the graceful shutdown window "
                    "active_job_id=%s",
                    self._active_job_id,
                )

    def wake(self) -> None:
        self._wake_requested.set()

    async def run(self) -> None:
        logger.info("model transfer worker started worker_id=%s", self.worker_id)
        await self.startup_reconcile()
        try:
            while not self._stop_requested.is_set():
                if await self.run_once():
                    continue
                await asyncio.to_thread(
                    self._wake_requested.wait,
                    self._settings.drive_sync_poll_interval_seconds,
                )
                self._wake_requested.clear()
        finally:
            logger.info("model transfer worker stopped worker_id=%s", self.worker_id)

    async def startup_reconcile(self) -> None:
        try:
            interrupted = self._repository.reconcile_interrupted()
            count = self._repository.reconcile_stale()
            repaired = await self._service.reconcile_files()
            if interrupted or count or repaired:
                logger.info(
                    "model transfer startup reconciliation interrupted=%s stale=%s repaired=%s",
                    interrupted,
                    count,
                    repaired,
                )
                self._notify_state_changed()
        except (ModelTransferRepositoryError, ModelPreparationServiceError):
            logger.warning("model transfer startup reconciliation failed", exc_info=True)

    async def run_once(self) -> bool:
        if self._stop_requested.is_set() or not self._settings.remote_model_enabled:
            return False
        try:
            job = self._repository.claim_next(
                self.worker_id,
                lease_seconds=self._settings.drive_sync_lease_seconds,
            )
        except ModelTransferRepositoryError:
            logger.warning("model transfer job claim failed", exc_info=True)
            return False
        if job is None:
            return False
        self._active_job_id = job.id
        self._notify_state_changed()
        heartbeat = asyncio.create_task(self._heartbeat(job.id))
        try:
            await self._service.process_job(
                job,
                self.worker_id,
                shutdown_check=self._stop_requested.is_set,
            )
        except ModelPreparationServiceError as exc:
            if exc.code == ModelTransferErrorCode.CANCELLED.value:
                self._mark_cancelled(job)
            else:
                self._mark_failed(job, exc)
        except Exception:  # noqa: BLE001 - one transfer must not stop the worker
            logger.error("model transfer job failed job=%s", job.id, exc_info=True)
            self._mark_failed(
                job,
                ModelPreparationServiceError(
                    ModelTransferErrorCode.DOWNLOAD_FAILED.value,
                    "モデル準備に失敗しました。",
                ),
            )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            self._active_job_id = None
            self._notify_state_changed()
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
            except ModelTransferRepositoryError:
                logger.warning("model transfer lease renewal failed", exc_info=True)

    def _mark_cancelled(self, job: ModelTransferJob) -> None:
        try:
            self._repository.mark_cancelled(job.id, self.worker_id)
        except ModelTransferRepositoryError:
            logger.warning("model transfer cancellation could not be persisted", exc_info=True)

    def _mark_failed(self, job: ModelTransferJob, error: ModelPreparationServiceError) -> None:
        try:
            self._repository.mark_failed(
                job.id,
                self.worker_id,
                error.code,
                str(error),
                retryable=error.retryable,
            )
        except ModelTransferRepositoryError:
            logger.warning("model transfer failure could not be persisted", exc_info=True)

    def _notify_state_changed(self) -> None:
        if self._state_changed_callback is None:
            return
        try:
            self._state_changed_callback()
        except Exception:  # noqa: BLE001 - backup notification must not stop worker
            logger.warning("model transfer state backup notification failed", exc_info=True)


class ModelTransferRuntime:
    """Own exactly one model transfer worker for the application process."""

    def __init__(self, worker: ModelTransferWorker) -> None:
        self.worker = worker

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    def wake(self) -> None:
        self.worker.wake()


__all__ = ["ModelTransferRuntime", "ModelTransferWorker"]
