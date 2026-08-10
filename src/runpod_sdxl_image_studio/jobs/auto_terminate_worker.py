"""Background grace-period worker for safe RunPod self-termination."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.pod_lifecycle import AutoTerminateState, TerminateReadiness
from runpod_sdxl_image_studio.services.pod_lifecycle_service import (
    PodLifecycleError,
    PodLifecycleService,
)


class AutoTerminateCoordinator:
    """Run one non-blocking readiness/grace/termination decision."""

    def __init__(
        self,
        lifecycle_service: PodLifecycleService,
        settings: Settings,
        *,
        now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
        startup_restore_ready: Callable[[], bool] | None = None,
    ) -> None:
        self._service = lifecycle_service
        self._settings = settings
        self._now_factory = now_factory
        self._startup_restore_ready = startup_restore_ready
        self._ready_since: datetime | None = None
        self.last_readiness: TerminateReadiness | None = None

    async def run_once(self) -> TerminateReadiness | None:
        if self._startup_restore_ready is not None and not self._startup_restore_ready():
            self._ready_since = None
            return None
        session = self._service.session or self._service.initialize_session()
        if session is None or not session.auto_terminate_enabled or not session.is_armed:
            self._ready_since = None
            return None
        if session.status in {
            AutoTerminateState.DRAINING,
            AutoTerminateState.TERMINATION_REQUESTING,
            AutoTerminateState.TERMINATION_AMBIGUOUS,
            AutoTerminateState.TERMINATION_FAILED,
        }:
            return self.last_readiness

        readiness = await self._service.check_readiness()
        self.last_readiness = readiness
        if not readiness.is_safe:
            self._ready_since = None
            if (self._service.session or session).status in {
                AutoTerminateState.WAITING,
                AutoTerminateState.READY,
            }:
                self._service.abort_draining()
            return readiness

        now = self._now_factory()
        if (
            self._ready_since is None
            or (self._service.session or session).status is AutoTerminateState.ARMED
        ):
            self._ready_since = now
            if not self._service.set_transient_state(AutoTerminateState.WAITING):
                # A manual drain may have won the CAS race after readiness
                # was checked.  Preserve the durable DRAINING state.
                self._ready_since = None
            return readiness
        elapsed = (now - self._ready_since).total_seconds()
        if elapsed < self._settings.auto_terminate_grace_seconds:
            if (self._service.session or session).status is not AutoTerminateState.READY:
                self._service.set_transient_state(AutoTerminateState.READY)
            return readiness

        try:
            final_readiness = await self._service.drain_backup_and_terminate(
                require_armed=True,
            )
            self.last_readiness = final_readiness
            return final_readiness
        except PodLifecycleError:
            self._ready_since = None
            return self.last_readiness
        except Exception:
            self._ready_since = None
            with contextlib.suppress(Exception):
                self._service.abort_draining()
            return self.last_readiness


class AutoTerminateRuntime:
    """Own the coordinator's daemon thread without blocking Gradio startup."""

    def __init__(self, coordinator: AutoTerminateCoordinator, settings: Settings) -> None:
        self._coordinator = coordinator
        self._settings = settings
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="auto-terminate", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(
                timeout=min(5.0, self._settings.auto_terminate_check_interval_seconds + 1.0)
            )
        self._thread = None

    def wake(self) -> None:
        """The next interval observes state changes; this method is a safe hook."""

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with contextlib.suppress(Exception):
                asyncio.run(self._coordinator.run_once())
            self._stop_event.wait(self._settings.auto_terminate_check_interval_seconds)


__all__ = ["AutoTerminateCoordinator", "AutoTerminateRuntime"]
