"""Single-worker background runner for the persistent generation queue."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryError,
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationProgress, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import (
    CancellationOutcome,
    GenerationQueueItem,
    ReconciliationOutcome,
    SubmissionState,
)
from runpod_sdxl_image_studio.services.generation_errors import PromptPersistenceError
from runpod_sdxl_image_studio.services.generation_execution_service import (
    GenerationExecutionService,
)
from runpod_sdxl_image_studio.services.generation_queue_service import CancellationResult
from runpod_sdxl_image_studio.services.generation_service import (
    GenerationCancelledError,
    PromptSubmissionCoordinator,
)

logger = logging.getLogger(__name__)
ProgressReporter = Callable[[GenerationProgress], None]
ReconcileHandler = Callable[[GenerationQueueItem], Awaitable[ReconciliationOutcome]]
CompletedOptionalArtifactMaintenanceHandler = Callable[[], Awaitable[tuple[str, ...]]]


class CancellationAdapter(Protocol):
    async def cancel_prompt(self, prompt_id: str) -> CancellationResult: ...


class _QueueSubmissionCoordinator(PromptSubmissionCoordinator):
    def __init__(
        self,
        repository: GenerationDispatchQueueRepositoryProtocol,
        sequence: int,
        worker_id: str,
    ) -> None:
        self._repository = repository
        self._sequence = sequence
        self._worker_id = worker_id
        self.started = False
        self.submitted = False

    def begin(self) -> str:
        item = self._repository.begin_submission(self._sequence, self._worker_id)
        token = item.entry.submission_token
        if not token:
            raise PromptPersistenceError("submission token was not persisted")
        self.started = True
        return token

    def mark_submitted(self, prompt_id: str, submission_token: str) -> None:
        try:
            self._repository.mark_submitted(
                self._sequence,
                self._worker_id,
                submission_token,
                prompt_id,
            )
        except GenerationDispatchQueueRepositoryError as exc:
            raise PromptPersistenceError("prompt submission state could not be persisted") from exc
        self.submitted = True


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
        cancellation_adapter: CancellationAdapter | None = None,
        progress_reporter: ProgressReporter | None = None,
        completed_optional_artifact_handler: CompletedOptionalArtifactMaintenanceHandler
        | None = None,
    ) -> None:
        self._repository = repository
        self._execution_service = execution_service
        self._settings = settings
        self.worker_id = worker_id or f"worker-{uuid4()}"
        self._reconcile_handler = reconcile_handler
        self._cancellation_adapter = cancellation_adapter
        self._progress_reporter = progress_reporter
        self._completed_optional_artifact_handler = completed_optional_artifact_handler
        self._stop_requested = threading.Event()
        self._wake_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._fail_closed = False
        self._last_reconciled_at = 0.0
        self._optional_artifact_maintenance_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start one daemon thread; callers own the worker lifecycle explicitly."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._fail_closed = False
        self._last_reconciled_at = 0.0
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
                if self._fail_closed:
                    logger.error("generation worker stopped fail-closed after ambiguous submission")
                    break
                if time.monotonic() - self._last_reconciled_at >= max(
                    1.0, self._settings.queue_poll_interval_seconds
                ):
                    await self.reconcile()
                claimed = await self.run_once()
                if claimed:
                    continue
                await asyncio.to_thread(
                    self._wake_requested.wait, self._settings.queue_poll_interval_seconds
                )
                self._wake_requested.clear()
        finally:
            await self._stop_optional_artifact_maintenance()
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
        submission = _QueueSubmissionCoordinator(
            self._repository,
            item.entry.sequence,
            self.worker_id,
        )
        release_claim = True
        try:
            if item.entry.cancel_requested_at is not None:
                await self._cancel_item(item)
                return True
            if _has_prompt_mismatch(item):
                self._repository.mark_prompt_id_mismatch(item.generation.id)
                return True
            if _prompt_id(item):
                # A prompt ID means this is recovery work, never a resend candidate.
                await self._reconcile_item(item)
                return True
            if item.entry.submission_state is not SubmissionState.READY:
                return True
            await self._execution_service.execute_persisted(
                item.entry.generation_id,
                item.entry.job_id,
                self._progress_reporter,
                lambda: self._is_cancel_requested(item.entry.generation_id),
                submission_coordinator=submission,
            )
            return True
        except GenerationCancelledError:
            if submission.submitted:
                await self._cancel_item(item)
            elif submission.started:
                release_claim = self._quarantine_submission(item, release_claim)
            else:
                self._repository.mark_cancelled(item.entry.generation_id)
            return True
        except Exception:  # noqa: BLE001 - one failed job must not stop the FIFO worker
            if submission.started and not submission.submitted:
                release_claim = self._quarantine_submission(item, release_claim)
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
            if release_claim:
                try:
                    self._repository.release_claim(item.entry.sequence, self.worker_id)
                except GenerationDispatchQueueRepositoryError:
                    logger.warning("generation worker lease release failed", exc_info=True)

    async def _cancel_item(self, item: GenerationQueueItem) -> None:
        current = self._repository.get_queue_item(item.generation.id)
        if current is None:
            return
        if _has_prompt_mismatch(current):
            self._repository.mark_prompt_id_mismatch(current.generation.id)
            return
        prompt_id = _prompt_id(current)
        if not prompt_id:
            if current.entry.submission_state is SubmissionState.READY:
                self._repository.mark_cancelled(current.generation.id)
            return
        if self._cancellation_adapter is None:
            return
        result = await self._cancellation_adapter.cancel_prompt(prompt_id)
        if result.outcome is CancellationOutcome.CANCELLED:
            self._repository.mark_cancelled(current.generation.id)
        elif result.outcome in {
            CancellationOutcome.COMPLETED,
            CancellationOutcome.FAILED,
        }:
            await self._reconcile_item(current)
        elif result.outcome is CancellationOutcome.NOT_FOUND and self._is_stale(
            current, datetime.now(UTC)
        ):
            self._repository.mark_reconciliation_failed(
                current.generation.id,
                "キャンセル要求後のComfyUI promptが猶予期間を過ぎても見つかりません。",
            )

    async def _reconcile_item(self, item: GenerationQueueItem) -> ReconciliationOutcome:
        """Run normal prompt reconciliation and apply interruption cancellation."""

        if self._reconcile_handler is None:
            return ReconciliationOutcome.UNAVAILABLE
        outcome = await self._reconcile_handler(item)
        if outcome is ReconciliationOutcome.CANCELLED:
            self._repository.mark_cancelled(item.generation.id)
        return outcome

    def _quarantine_submission(self, item: GenerationQueueItem, release_claim: bool) -> bool:
        try:
            self._repository.mark_submission_ambiguous(
                item.entry.sequence,
                self.worker_id,
                "prompt request outcome could not be determined",
            )
            return release_claim
        except GenerationDispatchQueueRepositoryError:
            self._fail_closed = True
            logger.critical(
                "worker fail-closed: ambiguous prompt state could not be persisted sequence=%s",
                item.entry.sequence,
                exc_info=True,
            )
            return False

    async def reconcile(self) -> None:
        try:
            released = self._repository.reconcile_expired_claims()
            if released:
                logger.info("generation worker reconciled expired claims count=%s", released)
            self._last_reconciled_at = time.monotonic()
            now = datetime.now(UTC)
            items = self._repository.list_queue(
                statuses=(
                    GenerationStatus.PENDING,
                    GenerationStatus.QUEUED,
                    GenerationStatus.RUNNING,
                ),
                limit=500,
            )
            for item in items:
                if item.generation.status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED,
                } or item.job.status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED,
                }:
                    continue
                if item.entry.cancel_requested_at is not None:
                    try:
                        await self._cancel_item(item)
                    except Exception:  # noqa: BLE001 - transient adapter failures are retried
                        logger.warning(
                            "queue cancellation reconciliation failed generation_id=%s",
                            item.generation.id,
                            exc_info=True,
                        )
                    continue
                if _has_prompt_mismatch(item):
                    self._repository.mark_prompt_id_mismatch(item.generation.id, now=now)
                    continue
                if _prompt_id(item) is None:
                    # A submitting/ambiguous item is deliberately isolated. It must not be
                    # treated as a fresh prompt candidate after a process restart.
                    continue
                if self._reconcile_handler is None:
                    continue
                try:
                    outcome = await self._reconcile_item(item)
                except Exception:  # noqa: BLE001 - reconciliation must not stop the worker
                    logger.warning("queue item reconciliation handler failed", exc_info=True)
                    outcome = ReconciliationOutcome.UNAVAILABLE
                if outcome is ReconciliationOutcome.NOT_FOUND and self._is_stale(item, now):
                    self._repository.mark_reconciliation_failed(
                        item.generation.id,
                        "ComfyUI prompt was not found after the reconciliation grace period",
                        now=now,
                    )
        except GenerationDispatchQueueRepositoryError:
            logger.warning("generation worker reconciliation failed", exc_info=True)
        self._schedule_optional_artifact_maintenance()

    def _schedule_optional_artifact_maintenance(self) -> None:
        if self._completed_optional_artifact_handler is None:
            return
        if (
            self._optional_artifact_maintenance_task is not None
            and not self._optional_artifact_maintenance_task.done()
        ):
            return
        self._optional_artifact_maintenance_task = asyncio.create_task(
            self._run_optional_artifact_maintenance()
        )

    async def _run_optional_artifact_maintenance(self) -> None:
        assert self._completed_optional_artifact_handler is not None
        try:
            await self._completed_optional_artifact_handler()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - maintenance must not stop the worker
            logger.warning(
                "completed optional artifact maintenance failed worker_id=%s error=%s",
                self.worker_id,
                type(exc).__name__,
                exc_info=True,
            )

    async def _stop_optional_artifact_maintenance(self) -> None:
        task = self._optional_artifact_maintenance_task
        self._optional_artifact_maintenance_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

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


def _has_prompt_mismatch(item: GenerationQueueItem) -> bool:
    return (
        item.generation.comfy_prompt_id is not None
        and item.job.prompt_id is not None
        and item.generation.comfy_prompt_id != item.job.prompt_id
    )


def _prompt_id(item: GenerationQueueItem) -> str | None:
    if _has_prompt_mismatch(item):
        return None
    return item.job.prompt_id or item.generation.comfy_prompt_id


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
