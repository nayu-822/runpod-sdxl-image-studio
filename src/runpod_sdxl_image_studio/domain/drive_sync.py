"""Typed domain models and safe path helpers for Google Drive synchronization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from uuid import UUID
from zoneinfo import ZoneInfo


class DriveSyncStatus(StrEnum):
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"


class DriveConnectionStatus(StrEnum):
    NOT_CONFIGURED = "not_configured"
    CONNECTED = "connected"
    RCLONE_NOT_FOUND = "rclone_not_found"
    AUTH_FAILED = "authentication_failed"
    FAILED = "connection_failed"


class DriveSyncErrorCode(StrEnum):
    NOT_CONFIGURED = "drive_not_configured"
    RCLONE_NOT_FOUND = "drive_rclone_not_found"
    CONNECTION_FAILED = "drive_connection_failed"
    SOURCE_MISSING = "drive_source_missing"
    SOURCE_CHANGED = "drive_source_changed"
    METADATA_MISSING = "drive_metadata_missing"
    METADATA_INVALID = "drive_metadata_invalid"
    TRANSFER_FAILED = "drive_transfer_failed"
    REMOTE_VERIFICATION_FAILED = "drive_remote_verification_failed"
    STALE = "drive_sync_stale"
    PERSISTENCE_FAILED = "drive_persistence_failed"
    MANIFEST_FAILED = "drive_manifest_failed"


@dataclass(frozen=True)
class DriveRemotePaths:
    local_date: str
    image_path: str
    metadata_path: str
    manifest_path: str


@dataclass(frozen=True)
class SyncRecord:
    id: UUID
    generation_id: UUID
    status: DriveSyncStatus
    remote_name: str
    remote_base_path: str
    remote_image_path: str
    remote_metadata_path: str
    image_artifact_id: UUID
    metadata_artifact_id: UUID | None
    image_sha256: str
    metadata_sha256: str | None
    image_size_bytes: int
    metadata_size_bytes: int | None
    attempt_count: int
    last_attempt_at: datetime | None
    synced_at: datetime | None
    error_code: str | None
    error_summary: str | None
    created_at: datetime
    updated_at: datetime


DriveSyncRecord = SyncRecord


@dataclass(frozen=True)
class DriveSyncJob:
    id: UUID
    sync_record_id: UUID
    generation_id: UUID
    status: DriveSyncStatus
    queue_sequence: int
    progress_bytes: int
    total_bytes: int
    progress_percentage: float
    current_artifact: str | None
    worker_id: str | None
    pid: int | None
    claimed_at: datetime | None
    lease_expires_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    error_summary: str | None
    retryable: bool
    log_path: str | None
    image_artifact_id: UUID
    metadata_artifact_id: UUID | None
    image_sha256: str
    metadata_sha256: str | None
    image_size_bytes: int
    metadata_size_bytes: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DriveSyncProgress:
    progress_bytes: int
    total_bytes: int
    progress_percentage: float
    current_artifact: str | None

    def __post_init__(self) -> None:
        if self.progress_bytes < 0 or self.total_bytes < 0:
            raise ValueError("progress byte values must not be negative")
        if self.progress_bytes > self.total_bytes and self.total_bytes > 0:
            raise ValueError("progress bytes must not exceed total bytes")
        if not 0.0 <= self.progress_percentage <= 100.0:
            raise ValueError("progress percentage must be between 0 and 100")


@dataclass(frozen=True)
class DriveCapacity:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    unsynced_bytes: int
    synced_cache_bytes: int


@dataclass(frozen=True)
class DriveCacheCandidate:
    generation_id: UUID
    kind: str
    created_at: datetime
    local_size_bytes: int


@dataclass(frozen=True)
class DriveConnectionResult:
    status: DriveConnectionStatus
    message: str
    error_code: str | None = None


_SAFE_REMOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_remote_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return ""
    if not _SAFE_REMOTE_NAME.fullmatch(normalized) or normalized in {".", ".."}:
        raise ValueError("rclone remote name is unsafe")
    return normalized


def validate_remote_base_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        return ""
    if "\x00" in normalized or normalized.startswith("-"):
        raise ValueError("rclone base path is unsafe")
    if normalized in {".", "..", "/"} or normalized.startswith("/"):
        raise ValueError("rclone base path must be relative")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("rclone base path contains an unsafe component")
    return normalized


def build_remote_paths(
    generation_id: UUID,
    kind: str,
    created_at: datetime,
    *,
    timezone_name: str = "Asia/Tokyo",
) -> DriveRemotePaths:
    try:
        local_datetime = utc(created_at).astimezone(ZoneInfo(timezone_name))
    except Exception as exc:
        raise ValueError("configured timezone is invalid") from exc
    local_date = local_datetime.date().isoformat()
    folder = "upscaled" if kind == "upscale" else "generated"
    stem = f"{local_datetime:%Y%m%d_%H%M%S}_{generation_id.hex[:8]}"
    return DriveRemotePaths(
        local_date=local_date,
        image_path=f"{local_date}/{folder}/{stem}.png",
        metadata_path=f"{local_date}/{folder}/{stem}.json",
        manifest_path=f"{local_date}/manifests/manifest.jsonl",
    )


def validate_remote_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in normalized
    ):
        raise ValueError("remote path is unsafe")
    return path.as_posix()


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "DriveCacheCandidate",
    "DriveCapacity",
    "DriveConnectionResult",
    "DriveConnectionStatus",
    "DriveRemotePaths",
    "DriveSyncErrorCode",
    "DriveSyncJob",
    "DriveSyncProgress",
    "DriveSyncRecord",
    "DriveSyncStatus",
    "SyncRecord",
    "build_remote_paths",
    "validate_remote_base_path",
    "validate_remote_name",
    "validate_remote_relative_path",
]
