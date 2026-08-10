"""Fail-closed lifecycle readiness and self-termination coordination."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Protocol

from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.model_transfer_repository import (
    ModelTransferRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.pod_lifecycle_repository import (
    PodLifecycleRepositoryError,
    PodLifecycleRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.runpod.pod_lifecycle import (
    RunPodIdentity,
    RunPodLifecycleAdapter,
    RunPodTerminateError,
    RunPodTerminateResult,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveDestination,
    DriveManifestState,
    DriveSyncStatus,
)
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.model_transfer import ModelTransferStatus
from runpod_sdxl_image_studio.domain.pod_lifecycle import (
    AutoTerminateState,
    PodLifecycleSession,
    TerminateBlockReason,
    TerminateReadiness,
)
from runpod_sdxl_image_studio.domain.state_sync import StateSyncStatus


class PodLifecycleError(RuntimeError):
    """Safe lifecycle error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PodLifecycleWorkBlockedError(PodLifecycleError):
    """Raised when a new operation races with the draining boundary."""

    def __init__(self) -> None:
        super().__init__(
            "pod_lifecycle_draining", "Pod is preparing to terminate; new work is blocked"
        )


class PodLifecycleTransitionError(PodLifecycleError):
    """Raised when a persisted lifecycle transition is no longer valid."""

    def __init__(self, message: str = "invalid pod lifecycle transition") -> None:
        super().__init__("pod_lifecycle_invalid_transition", message)


class ComfyQueueProvider(Protocol):
    async def __call__(self) -> object: ...


class LifecycleGate(Protocol):
    def ensure_work_allowed(self) -> None: ...

    def admit_work(self) -> AbstractContextManager[None]: ...

    def admit_persistent_mutation(self) -> AbstractContextManager[None]: ...


class PodLifecycleService:
    """Coordinate one process-safe, persisted lifecycle session."""

    _termination_lock = threading.Lock()
    _work_admission_lock = threading.RLock()

    def __init__(
        self,
        repository: PodLifecycleRepositoryProtocol,
        generation_repository: GenerationRepositoryProtocol,
        dispatch_queue_repository: GenerationDispatchQueueRepositoryProtocol,
        model_transfer_repository: ModelTransferRepositoryProtocol,
        drive_sync_repository: DriveSyncRepositoryProtocol,
        state_sync_service: Any,
        runpod_adapter: RunPodLifecycleAdapter | Any,
        *,
        settings: Settings | None = None,
        comfyui_queue_provider: ComfyQueueProvider | None = None,
        now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
        state_changed_callback: Callable[[], None] | None = None,
    ) -> None:
        self._repository = repository
        self._generation_repository = generation_repository
        self._dispatch_queue_repository = dispatch_queue_repository
        self._model_transfer_repository = model_transfer_repository
        self._drive_sync_repository = drive_sync_repository
        self._state_sync_service = state_sync_service
        self._runpod_adapter = runpod_adapter
        self._settings = settings
        self._comfyui_queue_provider = comfyui_queue_provider
        self._now_factory = now_factory
        self._state_changed_callback = state_changed_callback
        self._session: PodLifecycleSession | None = None

    @property
    def session(self) -> PodLifecycleSession | None:
        return self._session

    @property
    def current_pod_id(self) -> str | None:
        identity = self._identity()
        return identity.pod_id if identity is not None else None

    def initialize_session(self) -> PodLifecycleSession | None:
        identity = self._identity()
        if identity is None or not identity.pod_id:
            self._session = None
            return None
        enabled = bool(self._settings and self._settings.auto_terminate_enabled)
        try:
            self._session = self._repository.get_or_create(
                identity.pod_id,
                auto_terminate_enabled=enabled,
                now=self._now_factory(),
            )
        except PodLifecycleRepositoryError as exc:
            raise PodLifecycleError("pod_lifecycle_persistence_failed", str(exc)) from exc
        return self._session

    def refresh_session(self) -> PodLifecycleSession | None:
        pod_id = self.current_pod_id
        if not pod_id:
            self._session = None
            return None
        try:
            self._session = self._repository.get_by_pod_id(pod_id)
        except PodLifecycleRepositoryError:
            self._session = None
        return self._session

    def ensure_work_allowed(self) -> None:
        with self._work_admission_lock:
            session = self._session or self.initialize_session()
            if session is None:
                return
            if session.status in {
                AutoTerminateState.DRAINING,
                AutoTerminateState.TERMINATION_REQUESTING,
                AutoTerminateState.TERMINATION_AMBIGUOUS,
            }:
                raise PodLifecycleWorkBlockedError()

    @contextmanager
    def admit_work(self) -> Iterator[None]:
        """Serialize work admission with the durable draining transition."""

        with self._work_admission_lock:
            self.ensure_work_allowed()
            yield

    @contextmanager
    def admit_persistent_mutation(self) -> Iterator[None]:
        """Admit a repository write and its StateSync dirty mark atomically."""

        with self._work_admission_lock:
            self.ensure_work_allowed()
            yield

    def arm_on_generation_enqueue(self) -> PodLifecycleSession | None:
        """Arm only after a new Generation or successful batch was persisted."""

        with self._work_admission_lock:
            session = self._session or self.initialize_session()
            if session is None:
                return None
            self.ensure_work_allowed()
            if session.status is AutoTerminateState.TERMINATION_FAILED:
                # A confirmed DELETE failure is recoverable for normal work,
                # but a new generation must not silently re-enable automatic
                # termination.  The user can explicitly reopen it with the
                # lifecycle toggle.
                return session
            timestamp = self._now_factory()
            updated = replace(
                session,
                auto_terminate_armed_at=session.auto_terminate_armed_at or timestamp,
                status=AutoTerminateState.ARMED,
                last_activity_at=timestamp,
                last_error_code=None,
                last_error_summary=None,
                updated_at=timestamp,
            )
            self._session = self._save(updated)
            return self._session

    def set_auto_terminate_enabled(self, enabled: bool) -> PodLifecycleSession | None:
        with self._work_admission_lock:
            session = self._session or self.initialize_session()
            if session is None:
                return None
            if session.status in {
                AutoTerminateState.DRAINING,
                AutoTerminateState.TERMINATION_REQUESTING,
                AutoTerminateState.TERMINATION_AMBIGUOUS,
            }:
                # Stale browser toggles must never reopen or rewrite a
                # request whose delivery state is still ambiguous.
                return session
            timestamp = self._now_factory()
            updated = replace(
                session,
                auto_terminate_enabled=enabled,
                status=(
                    AutoTerminateState.ARMED
                    if enabled and session.auto_terminate_armed_at is not None
                    else AutoTerminateState.IDLE
                    if not enabled
                    else session.status
                ),
                last_activity_at=timestamp,
                updated_at=timestamp,
            )
            self._session = self._save(updated)
            return self._session

    async def check_readiness(self) -> TerminateReadiness:
        checked_at = self._now_factory()
        session = self._session or self.initialize_session()
        reasons: list[str] = []
        if session is None:
            reasons.append(TerminateBlockReason.RUNPOD_IDENTITY_MISSING.value)
        elif session.status in {
            AutoTerminateState.TERMINATION_REQUESTING,
            AutoTerminateState.TERMINATION_AMBIGUOUS,
        }:
            reasons.append(TerminateBlockReason.TERMINATION_ALREADY_REQUESTED.value)

        identity_ready = self._identity_ready()
        if not identity_ready:
            reasons.append(TerminateBlockReason.RUNPOD_IDENTITY_MISSING.value)

        generation_ready = False
        current_generations: tuple[Any, ...] = ()
        if session is not None:
            try:
                current_generations = self._current_generations(session)
                statuses: set[GenerationStatus] = {
                    generation.status for generation in current_generations
                }
                generation_ready = (
                    any(status is GenerationStatus.COMPLETED for status in statuses)
                    and not any(status is GenerationStatus.FAILED for status in statuses)
                    and not any(status is GenerationStatus.CANCELLED for status in statuses)
                    and not any(
                        status
                        in {
                            GenerationStatus.PENDING,
                            GenerationStatus.QUEUED,
                            GenerationStatus.RUNNING,
                        }
                        for status in statuses
                    )
                )
                if not generation_ready:
                    reasons.append(TerminateBlockReason.GENERATION_NOT_COMPLETED.value)
                if any(status is GenerationStatus.FAILED for status in statuses):
                    reasons.append(TerminateBlockReason.GENERATION_FAILED.value)
                if any(status is GenerationStatus.CANCELLED for status in statuses):
                    reasons.append(TerminateBlockReason.GENERATION_CANCELLED.value)
            except Exception:  # noqa: BLE001 - readiness is fail-closed
                reasons.append(TerminateBlockReason.GENERATION_NOT_COMPLETED.value)
        else:
            reasons.append(TerminateBlockReason.GENERATION_NOT_COMPLETED.value)

        if session is not None:
            try:
                if self._dispatch_queue_repository.has_active_generation_work_since(
                    session.started_at
                ):
                    reasons.append(TerminateBlockReason.GENERATION_WORK_ACTIVE.value)
            except Exception:  # noqa: BLE001
                reasons.append(TerminateBlockReason.GENERATION_WORK_ACTIVE.value)

        comfyui_ready = await self._check_comfyui(reasons)
        model_transfer_ready = self._check_model_transfer(reasons)
        drive_sync_ready = self._check_drive_sync(current_generations, reasons)
        manifest_ready = self._check_manifest(current_generations, reasons)
        state_backup_ready = self._check_state_backup(reasons)
        if not identity_ready:
            reasons.append(TerminateBlockReason.RUNPOD_IDENTITY_MISSING.value)

        safe = (
            all(
                (
                    generation_ready,
                    comfyui_ready,
                    model_transfer_ready,
                    drive_sync_ready,
                    manifest_ready,
                    state_backup_ready,
                    identity_ready,
                )
            )
            and not reasons
        )
        return TerminateReadiness(
            is_safe=safe,
            checked_at=checked_at,
            generation_ready=generation_ready,
            comfyui_ready=comfyui_ready,
            model_transfer_ready=model_transfer_ready,
            drive_sync_ready=drive_sync_ready,
            manifest_ready=manifest_ready,
            state_backup_ready=state_backup_ready,
            runpod_identity_ready=identity_ready,
            block_reasons=tuple(dict.fromkeys(reasons)),
        )

    async def request_terminate(
        self,
        *,
        readiness: TerminateReadiness | None = None,
        require_armed: bool = False,
    ) -> RunPodTerminateResult:
        session = self._session or self.initialize_session()
        if session is None:
            raise PodLifecycleError(
                TerminateBlockReason.RUNPOD_IDENTITY_MISSING.value,
                "RunPod self-termination identity is unavailable",
            )
        # ``readiness`` is accepted for compatibility with existing callers,
        # but it is only a display snapshot.  DELETE authorization always uses
        # a fresh check at the final boundary.
        del readiness
        readiness = await self.check_readiness()
        if not readiness.is_safe:
            raise PodLifecycleError("pod_not_safe_to_terminate", "Pod is not safe to terminate")

        # Serialize the durable DRAINING -> REQUESTING transition with every
        # persistent mutation, then release before the network await.
        with self._work_admission_lock:
            session = self._session or self.initialize_session()
            if session is None:
                raise PodLifecycleError(
                    TerminateBlockReason.RUNPOD_IDENTITY_MISSING.value,
                    "RunPod self-termination identity is unavailable",
                )
            if session.status is not AutoTerminateState.DRAINING:
                if session.status in {
                    AutoTerminateState.TERMINATION_REQUESTING,
                    AutoTerminateState.TERMINATION_AMBIGUOUS,
                }:
                    raise PodLifecycleError(
                        TerminateBlockReason.TERMINATION_ALREADY_REQUESTED.value,
                        "termination request has already been sent",
                    )
                raise PodLifecycleTransitionError(
                    "termination request requires a durable draining state"
                )
            if require_armed and not session.is_armed:
                raise PodLifecycleError(
                    TerminateBlockReason.NOT_ARMED.value,
                    "auto-terminate is not armed for this lifecycle session",
                )
            if not self._termination_lock.acquire(blocking=False):
                raise PodLifecycleError(
                    TerminateBlockReason.TERMINATION_ALREADY_REQUESTED.value,
                    "termination request has already been sent",
                )
            try:
                timestamp = self._now_factory()
                self._session = self._save(
                    replace(
                        session,
                        status=AutoTerminateState.TERMINATION_REQUESTING,
                        last_activity_at=timestamp,
                        updated_at=timestamp,
                    )
                )
            finally:
                self._termination_lock.release()

        try:
            result = await self._runpod_adapter.terminate_self()
        except RunPodTerminateError as exc:
            failure_state = (
                AutoTerminateState.TERMINATION_AMBIGUOUS
                if exc.ambiguous
                else AutoTerminateState.TERMINATION_FAILED
            )
            with self._work_admission_lock:
                session = self._session
                if (
                    session is not None
                    and session.status is AutoTerminateState.TERMINATION_REQUESTING
                ):
                    self._session = self._save(
                        replace(
                            session,
                            status=failure_state,
                            last_error_code=exc.code,
                            last_error_summary=str(exc),
                            last_activity_at=self._now_factory(),
                            updated_at=self._now_factory(),
                        )
                    )
            raise
        # A successful 204 means this process is about to disappear.  Do not
        # issue another DB mutation or schedule a backup after this point.
        return result

    async def final_state_backup(self) -> object:
        """Flush the current SQLite snapshot before the final readiness check."""

        return await self._state_sync_service.backup(wait_for_clean=True)

    async def drain_backup_and_terminate(
        self, *, require_armed: bool = False
    ) -> TerminateReadiness:
        """Use one guarded drain/backup/readiness path for auto and manual termination."""

        if not require_armed:
            initial_readiness = await self.check_readiness()
            if not initial_readiness.is_safe:
                return initial_readiness
        self.begin_draining()
        try:
            readiness = await self.check_readiness()
            if not readiness.is_safe:
                raise PodLifecycleError("pod_not_safe_to_terminate", "Pod is not safe to terminate")

            await self.final_state_backup()
            final_readiness = await self.check_readiness()
            if not final_readiness.is_safe:
                raise PodLifecycleError("pod_not_safe_to_terminate", "Pod is not safe to terminate")

            await self.request_terminate(
                readiness=final_readiness,
                require_armed=require_armed,
            )
            return final_readiness
        except Exception:
            # A failed readiness or backup check must reopen admission.  Once
            # request_terminate has durably entered REQUESTING/AMBIGUOUS, the
            # session is intentionally left in that terminal-request state.
            if self._session is not None and self._session.status is AutoTerminateState.DRAINING:
                self.abort_draining()
            raise

    async def manual_drain_backup_and_terminate(self) -> TerminateReadiness:
        """Check readiness before opening DRAINING for a manual request."""

        return await self.drain_backup_and_terminate(require_armed=False)

    def begin_draining(self) -> None:
        with self._work_admission_lock:
            session = self._session or self.initialize_session()
            if session is None:
                raise PodLifecycleError(
                    TerminateBlockReason.RUNPOD_IDENTITY_MISSING.value,
                    "RunPod self-termination identity is unavailable",
                )
            if session.status is AutoTerminateState.DRAINING:
                return
            if session.status not in {
                AutoTerminateState.IDLE,
                AutoTerminateState.ARMED,
                AutoTerminateState.WAITING,
                AutoTerminateState.READY,
                AutoTerminateState.TERMINATION_FAILED,
            }:
                if session.status in {
                    AutoTerminateState.TERMINATION_REQUESTING,
                    AutoTerminateState.TERMINATION_AMBIGUOUS,
                }:
                    raise PodLifecycleError(
                        TerminateBlockReason.TERMINATION_ALREADY_REQUESTED.value,
                        "termination request has already been sent",
                    )
                raise PodLifecycleTransitionError(
                    f"cannot enter draining from {session.status.value}"
                )
            self._session = self._save(
                replace(
                    session,
                    status=AutoTerminateState.DRAINING,
                    last_activity_at=self._now_factory(),
                    updated_at=self._now_factory(),
                )
            )

    def abort_draining(self) -> None:
        with self._work_admission_lock:
            session = self._session
            if session is None or session.status is not AutoTerminateState.DRAINING:
                return
            status = AutoTerminateState.ARMED if session.is_armed else AutoTerminateState.IDLE
            self._session = self._save(
                replace(
                    session,
                    status=status,
                    last_activity_at=self._now_factory(),
                    updated_at=self._now_factory(),
                )
            )

    def set_transient_state(self, state: AutoTerminateState) -> bool:
        with self._work_admission_lock:
            session = self._session or self.initialize_session()
            if session is None:
                return False
            if state is AutoTerminateState.WAITING:
                allowed = {AutoTerminateState.ARMED}
            elif state is AutoTerminateState.READY:
                allowed = {AutoTerminateState.WAITING}
            else:
                raise PodLifecycleTransitionError(
                    "only WAITING and READY are transient lifecycle states"
                )
            if session.status not in allowed:
                return False
            self._session = self._save(
                replace(
                    session,
                    status=state,
                    last_activity_at=self._now_factory(),
                    updated_at=self._now_factory(),
                )
            )
            return True

    def reset_grace_to_armed(self) -> bool:
        """Reset only a WAITING/READY grace state after readiness became unsafe."""

        with self._work_admission_lock:
            session = self._session
            if session is None or session.status not in {
                AutoTerminateState.WAITING,
                AutoTerminateState.READY,
            }:
                return False
            timestamp = self._now_factory()
            self._session = self._save(
                replace(
                    session,
                    status=AutoTerminateState.ARMED,
                    last_activity_at=timestamp,
                    updated_at=timestamp,
                )
            )
            return True

    def _current_generations(self, session: PodLifecycleSession) -> tuple[Any, ...]:
        return self._generation_repository.list_since_unbounded(session.started_at)

    async def _check_comfyui(self, reasons: list[str]) -> bool:
        if self._comfyui_queue_provider is None:
            reasons.append(TerminateBlockReason.COMFYUI_UNAVAILABLE.value)
            return False
        try:
            queue = await self._comfyui_queue_provider()
            pending = getattr(queue, "pending_prompt_ids", None)
            running = getattr(queue, "running_prompt_ids", None)
            ready = (
                isinstance(pending, Sequence)
                and not isinstance(pending, (str, bytes, bytearray))
                and isinstance(running, Sequence)
                and not isinstance(running, (str, bytes, bytearray))
                and not pending
                and not running
            )
            if not ready:
                reasons.append(TerminateBlockReason.COMFYUI_QUEUE_ACTIVE.value)
            return ready
        except Exception:  # noqa: BLE001 - unknown remote queue is unsafe
            reasons.append(TerminateBlockReason.COMFYUI_UNAVAILABLE.value)
            return False

    def _check_model_transfer(self, reasons: list[str]) -> bool:
        try:
            counts = self._model_transfer_repository.status_counts()
            active = sum(
                counts.get(status, 0)
                for status in (
                    ModelTransferStatus.PENDING,
                    ModelTransferStatus.DOWNLOADING,
                    ModelTransferStatus.CANCEL_REQUESTED,
                )
            )
            if active:
                reasons.append(TerminateBlockReason.MODEL_TRANSFER_ACTIVE.value)
                return False
            return True
        except Exception:  # noqa: BLE001
            reasons.append(TerminateBlockReason.MODEL_TRANSFER_ACTIVE.value)
            return False

    def _check_drive_sync(self, generations: Sequence[Any], reasons: list[str]) -> bool:
        completed = [item for item in generations if item.status is GenerationStatus.COMPLETED]
        if not completed:
            reasons.append(TerminateBlockReason.DRIVE_NOT_SYNCED.value)
            return False
        try:
            counts = self._drive_sync_repository.status_counts()
            if counts.get(DriveSyncStatus.PENDING, 0) or counts.get(DriveSyncStatus.SYNCING, 0):
                reasons.append(TerminateBlockReason.DRIVE_SYNC_ACTIVE.value)
                return False
            for generation in completed:
                record = self._drive_sync_repository.get_by_generation(generation.id)
                if record is None or record.status is not DriveSyncStatus.SYNCED:
                    if record is not None and record.status is DriveSyncStatus.FAILED:
                        reasons.append(TerminateBlockReason.DRIVE_SYNC_FAILED.value)
                    else:
                        reasons.append(TerminateBlockReason.DRIVE_NOT_SYNCED.value)
                    return False
            return True
        except Exception:  # noqa: BLE001
            reasons.append(TerminateBlockReason.DRIVE_NOT_SYNCED.value)
            return False

    def _check_manifest(self, generations: Sequence[Any], reasons: list[str]) -> bool:
        completed = [item for item in generations if item.status is GenerationStatus.COMPLETED]
        try:
            if self._drive_sync_repository.has_active_manifest_jobs():
                reasons.append(TerminateBlockReason.MANIFEST_ACTIVE.value)
                return False
            for generation in completed:
                record = self._drive_sync_repository.get_by_generation(generation.id)
                if record is None:
                    reasons.append(TerminateBlockReason.MANIFEST_NOT_SYNCED.value)
                    return False
                local_date = record.remote_image_path.split("/", 1)[0]
                state = self._drive_sync_repository.manifest_state_for_destination(
                    local_date,
                    DriveDestination(record.remote_name, record.remote_base_path),
                )
                if state is not DriveManifestState.SYNCED:
                    reasons.append(
                        TerminateBlockReason.MANIFEST_FAILED.value
                        if state is DriveManifestState.FAILED
                        else TerminateBlockReason.MANIFEST_NOT_SYNCED.value
                    )
                    return False
            return bool(completed)
        except Exception:  # noqa: BLE001
            reasons.append(TerminateBlockReason.MANIFEST_NOT_SYNCED.value)
            return False

    def _check_state_backup(self, reasons: list[str]) -> bool:
        try:
            if not bool(self._state_sync_service.enabled):
                reasons.append(TerminateBlockReason.STATE_BACKUP_DISABLED.value)
                return False
            if bool(self._state_sync_service.backup_in_progress):
                reasons.append(TerminateBlockReason.STATE_BACKUP_ACTIVE.value)
                return False
            view = self._state_sync_service.get_status()
            if view.status is StateSyncStatus.FAILED:
                reasons.append(TerminateBlockReason.STATE_BACKUP_FAILED.value)
                return False
            if not bool(self._state_sync_service.is_clean):
                reasons.append(TerminateBlockReason.STATE_BACKUP_DIRTY.value)
                return False
            if not bool(self._state_sync_service.has_latest_remote_backup):
                reasons.append(TerminateBlockReason.STATE_BACKUP_DIRTY.value)
                return False
            return True
        except Exception:  # noqa: BLE001
            reasons.append(TerminateBlockReason.STATE_BACKUP_FAILED.value)
            return False

    def _identity(self) -> RunPodIdentity | None:
        try:
            identity = self._runpod_adapter.identity()
            return identity if isinstance(identity, RunPodIdentity) else identity
        except Exception:  # noqa: BLE001
            return None

    def _identity_ready(self) -> bool:
        try:
            return bool(self._runpod_adapter.identity_ready)
        except Exception:  # noqa: BLE001
            identity = self._identity()
            return bool(identity and identity.is_ready)

    def _save(self, session: PodLifecycleSession) -> PodLifecycleSession:
        try:
            result = self._repository.save(session)
        except PodLifecycleRepositoryError as exc:
            raise PodLifecycleError("pod_lifecycle_persistence_failed", str(exc)) from exc
        # Lifecycle session state is local control metadata for this Pod.  It
        # must survive an application restart, but it is deliberately excluded
        # from the cross-Pod StateSync dirty stream.
        return result


__all__ = [
    "LifecycleGate",
    "PodLifecycleError",
    "PodLifecycleService",
    "PodLifecycleTransitionError",
    "PodLifecycleWorkBlockedError",
]
