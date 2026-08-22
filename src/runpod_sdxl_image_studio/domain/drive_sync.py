"""Typed domain models and safe path helpers for Google Drive synchronization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from math import isclose
from pathlib import PurePosixPath
from uuid import UUID
from zoneinfo import ZoneInfo


class DriveSyncStatus(StrEnum):
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"


class DriveManifestState(StrEnum):
    MISSING = "missing"
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
    MANIFEST_REBUILD_REQUIRED = "drive_manifest_rebuild_required"
    DESTINATION_CHANGED = "drive_destination_changed"


@dataclass(frozen=True)
class DriveDestination:
    """Immutable rclone destination captured when work is queued."""

    remote_name: str
    base_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "remote_name", validate_remote_name(self.remote_name))
        object.__setattr__(self, "base_path", validate_remote_base_path(self.base_path))


@dataclass(frozen=True)
class DriveRemotePaths:
    local_date: str
    image_path: str
    metadata_path: str
    manifest_path: str


@dataclass(frozen=True)
class DriveSyncArtifact:
    """Durable transfer plan and per-image progress for one Generation."""

    display_order: int
    image_artifact_id: UUID
    remote_image_path: str
    image_sha256: str
    image_size_bytes: int
    metadata_artifact_id: UUID | None
    remote_metadata_path: str | None
    metadata_sha256: str | None
    metadata_size_bytes: int | None
    image_synced: bool = False
    metadata_synced: bool = False

    def __post_init__(self) -> None:
        if self.display_order < 0:
            raise ValueError("display order must not be negative")
        if self.image_size_bytes < 0:
            raise ValueError("image size must not be negative")
        if self.metadata_size_bytes is not None and self.metadata_size_bytes < 0:
            raise ValueError("metadata size must not be negative")


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
    artifacts: tuple[DriveSyncArtifact, ...] = ()


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
    artifacts: tuple[DriveSyncArtifact, ...] = ()


@dataclass(frozen=True)
class DriveManifestJob:
    """Durable request for rebuilding and uploading one destination manifest."""

    id: UUID
    local_date: str
    status: DriveSyncStatus
    remote_name: str
    remote_base_path: str
    remote_manifest_path: str
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
    created_at: datetime
    updated_at: datetime

    @property
    def destination(self) -> DriveDestination:
        return DriveDestination(self.remote_name, self.remote_base_path)


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
    completed_at: datetime | None = None,
    final_upscale: bool = False,
    requested_scale_factor: float | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
) -> DriveRemotePaths:
    """Build deterministic remote paths from persisted generation evidence.

    The folder date is based on the terminal completion time when available, while
    the existing file-name timestamp continues to use ``created_at``.  The
    ``x1``/``x4`` classification is deliberately derived from persisted snapshot
    values supplied by the caller, never from current UI state.
    """

    try:
        created_local_datetime = utc(created_at).astimezone(ZoneInfo(timezone_name))
        folder_local_datetime = utc(completed_at or created_at).astimezone(ZoneInfo(timezone_name))
    except Exception as exc:
        raise ValueError("configured timezone is invalid") from exc
    local_date = folder_local_datetime.date().isoformat()
    folder = build_google_drive_output_folder(
        kind=kind,
        created_at=created_at,
        completed_at=completed_at,
        final_upscale=final_upscale,
        requested_scale_factor=requested_scale_factor,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        timezone_name=timezone_name,
    )
    stem = f"{created_local_datetime:%Y%m%d_%H%M%S}_{generation_id.hex[:8]}"
    return DriveRemotePaths(
        local_date=local_date,
        image_path=f"{folder}/{stem}.png",
        metadata_path=f"{folder}/{stem}.json",
        manifest_path=f"{local_date}/manifests/manifest.jsonl",
    )


def build_google_drive_output_folder(
    *,
    kind: str,
    created_at: datetime,
    completed_at: datetime | None = None,
    final_upscale: bool = False,
    requested_scale_factor: float | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    timezone_name: str = "Asia/Tokyo",
) -> str:
    """Return the flat ``YYYYMMDD_x1`` or ``YYYYMMDD_x4`` folder name.

    Standard/derived generations use the persisted ``final_upscale`` flag,
    whose product contract is Final 4x.  Upscale generations use only their
    durable factor or source/target dimensions; their inherited generation
    snapshot flag is intentionally ignored.  Missing or non-4x evidence is
    classified as ``x1`` rather than guessing from a current form value.
    """

    try:
        local_datetime = utc(completed_at or created_at).astimezone(ZoneInfo(timezone_name))
    except Exception as exc:
        raise ValueError("configured timezone is invalid") from exc
    kind_value = getattr(kind, "value", kind)
    if kind_value == "upscale":
        is_x4 = _is_four_x_upscale(
            requested_scale_factor,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )
    else:
        is_x4 = bool(final_upscale)
    suffix = "x4" if is_x4 else "x1"
    return f"{local_datetime:%Y%m%d}_{suffix}"


def remote_path_local_date(path: str) -> str:
    """Extract an ISO local date from old and current remote path formats."""

    prefix = path.replace("\\", "/").split("/", 1)[0]
    if re.fullmatch(r"\d{8}_x[14]", prefix):
        compact_date = f"{prefix[:4]}-{prefix[4:6]}-{prefix[6:8]}"
        try:
            return date.fromisoformat(compact_date).isoformat()
        except ValueError as exc:
            raise ValueError("remote path date is invalid") from exc
    try:
        return date.fromisoformat(prefix).isoformat()
    except ValueError as exc:
        raise ValueError("remote path date is invalid") from exc


def _is_four_x_upscale(
    requested_scale_factor: float | None,
    *,
    source_width: int | None,
    source_height: int | None,
    target_width: int | None,
    target_height: int | None,
) -> bool:
    if requested_scale_factor is not None:
        return isclose(requested_scale_factor, 4.0, rel_tol=0.0, abs_tol=1e-9)
    if None in {source_width, source_height, target_width, target_height}:
        return False
    assert source_width is not None
    assert source_height is not None
    assert target_width is not None
    assert target_height is not None
    if source_width <= 0 or source_height <= 0:
        return False
    return isclose(target_width / source_width, 4.0, rel_tol=0.0, abs_tol=1e-9) and isclose(
        target_height / source_height, 4.0, rel_tol=0.0, abs_tol=1e-9
    )


def build_remote_image_path(paths: DriveRemotePaths, display_order: int, image_count: int) -> str:
    """Return a collision-free image path while preserving old single-image paths."""

    if display_order < 0 or image_count < 1 or display_order >= image_count:
        raise ValueError("remote image path indexes are invalid")
    if image_count == 1:
        return paths.image_path
    suffix = f"_{display_order + 1:06d}"
    stem, extension = paths.image_path.rsplit(".", 1)
    return f"{stem}{suffix}.{extension}"


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
    "DriveDestination",
    "DriveManifestJob",
    "DriveManifestState",
    "DriveConnectionStatus",
    "DriveRemotePaths",
    "DriveSyncArtifact",
    "DriveSyncErrorCode",
    "DriveSyncJob",
    "DriveSyncProgress",
    "DriveSyncRecord",
    "DriveSyncStatus",
    "SyncRecord",
    "build_google_drive_output_folder",
    "build_remote_paths",
    "build_remote_image_path",
    "remote_path_local_date",
    "validate_remote_base_path",
    "validate_remote_name",
    "validate_remote_relative_path",
]
