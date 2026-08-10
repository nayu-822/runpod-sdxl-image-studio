"""Domain state for a RunPod lifecycle session and safe termination checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4


class AutoTerminateState(StrEnum):
    IDLE = "idle"
    ARMED = "armed"
    WAITING = "waiting"
    READY = "ready"
    DRAINING = "draining"
    TERMINATION_REQUESTING = "termination_requesting"
    TERMINATION_AMBIGUOUS = "termination_ambiguous"
    TERMINATION_FAILED = "termination_failed"


class TerminateBlockReason(StrEnum):
    NOT_ARMED = "not_armed"
    AUTO_TERMINATE_DISABLED = "auto_terminate_disabled"
    GENERATION_NOT_COMPLETED = "generation_not_completed"
    GENERATION_WORK_ACTIVE = "generation_work_active"
    GENERATION_FAILED = "generation_failed"
    GENERATION_CANCELLED = "generation_cancelled"
    COMFYUI_QUEUE_ACTIVE = "comfyui_queue_active"
    COMFYUI_UNAVAILABLE = "comfyui_unavailable"
    MODEL_TRANSFER_ACTIVE = "model_transfer_active"
    DRIVE_SYNC_ACTIVE = "drive_sync_active"
    DRIVE_SYNC_FAILED = "drive_sync_failed"
    DRIVE_NOT_SYNCED = "drive_not_synced"
    MANIFEST_ACTIVE = "manifest_active"
    MANIFEST_FAILED = "manifest_failed"
    MANIFEST_NOT_SYNCED = "manifest_not_synced"
    STATE_BACKUP_DISABLED = "state_backup_disabled"
    STATE_BACKUP_ACTIVE = "state_backup_active"
    STATE_BACKUP_DIRTY = "state_backup_dirty"
    STATE_BACKUP_FAILED = "state_backup_failed"
    RUNPOD_IDENTITY_MISSING = "runpod_identity_missing"
    TERMINATION_ALREADY_REQUESTED = "termination_already_requested"
    DRAINING = "draining"


@dataclass(frozen=True)
class PodLifecycleSession:
    id: UUID = field(default_factory=uuid4)
    pod_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    auto_terminate_enabled: bool = False
    auto_terminate_armed_at: datetime | None = None
    status: AutoTerminateState = AutoTerminateState.IDLE
    last_activity_at: datetime | None = None
    last_error_code: str | None = None
    last_error_summary: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_armed(self) -> bool:
        return self.auto_terminate_armed_at is not None


@dataclass(frozen=True)
class TerminateReadiness:
    """All conditions must be true before a self-terminate request is allowed."""

    is_safe: bool
    checked_at: datetime
    generation_ready: bool
    comfyui_ready: bool
    model_transfer_ready: bool
    drive_sync_ready: bool
    manifest_ready: bool
    state_backup_ready: bool
    runpod_identity_ready: bool
    block_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        checked_at = self.checked_at
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=UTC)
        else:
            checked_at = checked_at.astimezone(UTC)
        object.__setattr__(self, "checked_at", checked_at)
        expected = all(
            (
                self.generation_ready,
                self.comfyui_ready,
                self.model_transfer_ready,
                self.drive_sync_ready,
                self.manifest_ready,
                self.state_backup_ready,
                self.runpod_identity_ready,
            )
        )
        if self.is_safe and not expected:
            raise ValueError("safe readiness requires every readiness condition")

    @classmethod
    def blocked(
        cls,
        *,
        checked_at: datetime | None = None,
        generation_ready: bool = False,
        comfyui_ready: bool = False,
        model_transfer_ready: bool = False,
        drive_sync_ready: bool = False,
        manifest_ready: bool = False,
        state_backup_ready: bool = False,
        runpod_identity_ready: bool = False,
        block_reasons: tuple[str, ...] = (),
    ) -> TerminateReadiness:
        return cls(
            is_safe=False,
            checked_at=checked_at or datetime.now(UTC),
            generation_ready=generation_ready,
            comfyui_ready=comfyui_ready,
            model_transfer_ready=model_transfer_ready,
            drive_sync_ready=drive_sync_ready,
            manifest_ready=manifest_ready,
            state_backup_ready=state_backup_ready,
            runpod_identity_ready=runpod_identity_ready,
            block_reasons=tuple(dict.fromkeys(block_reasons)),
        )


@dataclass(frozen=True)
class TerminationAttempt:
    status: AutoTerminateState
    code: str | None = None
    summary: str | None = None


__all__ = [
    "AutoTerminateState",
    "PodLifecycleSession",
    "TerminateBlockReason",
    "TerminateReadiness",
    "TerminationAttempt",
]
