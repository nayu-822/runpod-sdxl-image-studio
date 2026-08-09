"""Typed state backup and restore values independent from adapters and UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class StateSyncStatus(StrEnum):
    """Current lifecycle state of the optional remote SQLite backup."""

    DISABLED = "disabled"
    IDLE = "idle"
    RUNNING = "running"
    SYNCED = "synced"
    FAILED = "failed"


class StateRestoreStatus(StrEnum):
    """Outcome of the startup-only restore attempt."""

    DISABLED = "disabled"
    SKIPPED_LOCAL = "skipped_local"
    NO_BACKUP = "no_backup"
    RESTORED = "restored"
    FAILED = "failed"


@dataclass(frozen=True)
class StateSnapshotMetadata:
    schema_version: int
    filename: str
    sha256: str
    size_bytes: int
    created_at: datetime


@dataclass(frozen=True)
class StateSyncView:
    status: StateSyncStatus
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_message: str = ""
    remote_sha256: str | None = None
    remote_size_bytes: int | None = None


@dataclass(frozen=True)
class StateRestoreResult:
    status: StateRestoreStatus
    metadata: StateSnapshotMetadata | None = None
    message: str = ""


__all__ = [
    "StateBackupStatus",
    "StateRestoreResult",
    "StateRestoreStatus",
    "StateSnapshotMetadata",
    "StateSyncStatus",
    "StateSyncView",
]

# Backwards-friendly alias for callers that describe backup rather than sync.
StateBackupStatus = StateSyncStatus
