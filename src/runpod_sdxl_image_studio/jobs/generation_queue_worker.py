"""Single-worker background runner for the persistent generation queue."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryError,
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationProgress, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import GenerationQueueItem
from runpod_sdxl_image_studio.services.generation_execution_service import (
    GenerationExecutionService,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationCancelledError

logger = logging.getLogger(__name__)
ProgressReporter = Callable[[GenerationProgress], None]
ReconcileHandler = Callable[[GenerationQueueItem], Awaitable[bool]]


class GenerationQueueWorker:
    """A cooperative single worker with polling fallback and lease heartbeat."""

    def __init__(
        self,
        repository: GenerationDispatchQueueRepositoryProtocol,
        execution_service: GenerationExecutionService,
        settings: Settings,
        *,
        worker_id: str | None = None,
        reconcile_handler: ReconcileHandler | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self._repository = repository
        self._execution_service = execution_service
        self._settings = settings
        self.worker_id = worker_id or f"worker-{uuid4()}"
        self._reconcile_handler = reconcile_handler
        self._progress_reporter = progress_reporter
        self._stop_requested = threading.Event()
        self._wake_requested = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start one daemon thread; callers own the worker lifecycle explicitly."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self.run()),
            name="generation-queue-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested.set()
        self._wake_requested.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self._settings.queue_lease_seconds))

    def wake(self) -> None:
        self._wake_requested.set()

    async def run(self) -> None:
        logger.info("generation worker started worker_id=%s", self.worker_id)
        await self.reconcile()
        try:
            while not self._stop_requested.is_set():
                claimed = await self.run_once()
                if claimed:
                    continue
                await asyncio.to_thread(
                    self._wake_requested.wait, self._settings.queue_poll_interval_seconds
                )
                self._wake_requested.clear()
        finally:
            logger.info("generation worker stopped worker_id=%s", self.worker_id)

    async def run_once(self) -> bool:
        if self._stop_requested.is_set():
            return False
        try:
            item = self._repository.claim_next(
                self.worker_id,
                lease_seconds=self._settings.queue_lease_seconds,
            )
        except GenerationDispatchQueueRepositoryError:
            logger.warning(
                "generation worker claim failed worker_id=%s", self.worker_id, exc_info=True
            )
            return False
        if item is None:
            return False
        logger.info(
            "generation worker claim worker_id=%s sequence=%s generation_id=%s "
            "job_id=%s batch_id=%s batch_index=%s",
            self.worker_id,
            item.entry.sequence,
            item.entry.generation_id,
            item.entry.job_id,
            item.entry.batch_id,
            item.entry.batch_index,
        )
        heartbeat = asyncio.create_task(self._heartbeat(item.entry.sequence))
        try:
            if item.entry.cancel_requested_at is not None:
                self._repository.mark_cancelled(item.entry.generation_id)
                return True
            if item.generation.comfy_prompt_id or item.job.prompt_id:
                # A prompt ID means this is recovery work, never a resend candidate.
                if self._reconcile_handler is not None:
                    await self._reconcile_handler(item)
                else:
                    self._repository.release_claim(item.entry.sequence, self.worker_id)
                return True
            await self._execution_service.execute_persisted(
                item.entry.generation_id,
                item.entry.job_id,
                self._progress_reporter,
                lambda: self._is_cancel_requested(item.entry.generation_id),
            )
            return True
        except GenerationCancelledError:
            self._repository.mark_cancelled(item.entry.generation_id)
            return True
        except Exception:  # noqa: BLE001 - one failed job must not stop the FIFO worker
            logger.error(
                "generation worker execution failed worker_id=%s sequence=%s generation_id=%s",
                self.worker_id,
                item.entry.sequence,
                item.entry.generation_id,
                exc_info=True,
            )
            return True
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            try:
                self._repository.release_claim(item.entry.sequence, self.worker_id)
            except GenerationDispatchQueueRepositoryError:
                logger.warning("generation worker lease release failed", exc_info=True)

    async def reconcile(self) -> None:
        try:
            released = self._repository.reconcile_expired_claims()
            if released:
                logger.info("generation worker reconciled expired claims count=%s", released)
            now = datetime.now(UTC)
            items = self._repository.list_queue(
                statuses=(GenerationStatus.QUEUED, GenerationStatus.RUNNING), limit=500
            )
            for item in items:
                if item.job.prompt_id or item.generation.comfy_prompt_id:
                    if self._reconcile_handler is None:
                        continue
                    recovered = await self._reconcile_handler(item)
                    if not recovered and self._is_stale(item, now):
                        self._repository.mark_reconciliation_failed(
                            item.generation.id,
                            "ComfyUI prompt could not be reconciled within the grace period",
                            now=now,
                        )
                elif self._is_stale(item, now):
                    self._repository.mark_reconciliation_failed(
                        item.generation.id,
                        "persisted queue item has no ComfyUI prompt ID",
                        now=now,
                    )
        except GenerationDispatchQueueRepositoryError:
            logger.warning("generation worker reconciliation failed", exc_info=True)

    async def _heartbeat(self, sequence: int) -> None:
        while True:
            await asyncio.sleep(self._settings.queue_heartbeat_seconds)
            try:
                self._repository.renew_lease(
                    sequence,
                    self.worker_id,
                    lease_seconds=self._settings.queue_lease_seconds,
                )
            except GenerationDispatchQueueRepositoryError:
                logger.warning("generation worker lease renewal failed sequence=%s", sequence)

    def _is_cancel_requested(self, generation_id: UUID) -> bool:
        item = self._repository.get_queue_item(generation_id)
        return item is None or item.entry.cancel_requested_at is not None

    def _is_stale(self, item: GenerationQueueItem, now: datetime) -> bool:
        grace = timedelta(seconds=self._settings.reconciliation_grace_seconds)
        return item.entry.updated_at + grace <= now


class GenerationQueueRuntime:
    """Own exactly one worker thread for the application process."""

    def __init__(self, worker: GenerationQueueWorker) -> None:
        self.worker = worker

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    def wake(self) -> None:
        self.worker.wake()


__all__ = ["GenerationQueueRuntime", "GenerationQueueWorker"]
