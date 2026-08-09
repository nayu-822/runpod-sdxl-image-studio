"""Typed values for remote model discovery and preparation jobs."""

from __future__ import annotations

import ntpath
import posixpath
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from uuid import UUID

from runpod_sdxl_image_studio.domain.drive_sync import validate_remote_relative_path


class RemoteModelKind(StrEnum):
    """ComfyUI model directory categories exposed by the remote catalog."""

    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    UPSCALER = "upscaler"


class ModelTransferStatus(StrEnum):
    """Durable lifecycle states for one selected remote model."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ModelTransferStatus.COMPLETED,
            ModelTransferStatus.FAILED,
            ModelTransferStatus.CANCELLED,
        }


class ModelTransferErrorCode(StrEnum):
    """Stable safe error codes shown in the model preparation UI."""

    NOT_CONFIGURED = "remote_model_not_configured"
    CATALOG_UNAVAILABLE = "remote_model_catalog_unavailable"
    INVALID_REMOTE_ENTRY = "invalid_remote_model_entry"
    INVALID_LOCAL_PATH = "invalid_model_destination"
    DOWNLOAD_FAILED = "model_download_failed"
    DOWNLOAD_TIMEOUT = "model_download_timeout"
    SIZE_MISMATCH = "model_size_mismatch"
    HASH_MISMATCH = "model_hash_mismatch"
    MODEL_NOT_VISIBLE = "model_not_visible_to_comfyui"
    PERSISTENCE_FAILED = "model_transfer_persistence_failed"
    CANCELLED = "model_transfer_cancelled"
    APP_RESTART_INTERRUPTED = "model_transfer_interrupted"
    STATELESS_RESTORE_INTERRUPTED = "stateless_restore_interrupted"


_MODEL_EXTENSIONS = frozenset({".safetensors", ".ckpt", ".pt", ".pth", ".bin"})


def normalize_model_relative_path(value: str) -> str:
    """Validate a category-relative model path without flattening directories."""

    normalized = value.replace("\\", "/")
    if not normalized or posixpath.isabs(normalized) or ntpath.isabs(normalized):
        raise ValueError("model path must be relative")
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("model path contains an unsafe component")
    return validate_remote_relative_path(path.as_posix())


def is_supported_model_filename(value: str) -> bool:
    """Use the common model extension set for remote and local catalogs."""

    return PurePosixPath(value).suffix.casefold() in _MODEL_EXTENSIONS


@dataclass(frozen=True)
class RemoteModelEntry:
    """A safe, category-relative remote model snapshot."""

    kind: RemoteModelKind
    relative_path: str
    display_name: str
    size_bytes: int
    modified_at: datetime | None = None
    remote_hash_algorithm: str | None = None
    remote_hash: str | None = None

    def __post_init__(self) -> None:
        normalized = normalize_model_relative_path(self.relative_path)
        if not is_supported_model_filename(normalized):
            raise ValueError("remote model extension is not supported")
        if self.size_bytes < 0:
            raise ValueError("model size must not be negative")
        if not self.display_name.strip():
            raise ValueError("model display name must not be empty")
        object.__setattr__(self, "relative_path", normalized)
        object.__setattr__(self, "display_name", self.display_name.strip())
        if self.modified_at is not None and self.modified_at.tzinfo is None:
            object.__setattr__(self, "modified_at", self.modified_at.replace(tzinfo=UTC))
        if self.remote_hash is not None:
            object.__setattr__(self, "remote_hash", self.remote_hash.strip().casefold())
        if self.remote_hash_algorithm is not None:
            object.__setattr__(
                self,
                "remote_hash_algorithm",
                self.remote_hash_algorithm.strip().casefold().replace("_", "-"),
            )

    @property
    def identity(self) -> str:
        """Return a stable version identity used by the active-job constraint."""

        modified = self.modified_at.astimezone(UTC).isoformat() if self.modified_at else ""
        return (
            f"{self.remote_hash_algorithm or ''}:{self.remote_hash or ''}:"
            f"{self.size_bytes}:{modified}"
        )


@dataclass(frozen=True)
class RemoteModelCatalog:
    """Remote catalog snapshot separated from locally usable capabilities."""

    entries: tuple[RemoteModelEntry, ...]
    fetched_at: datetime
    is_available: bool = True
    message: str = ""

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            object.__setattr__(self, "fetched_at", self.fetched_at.replace(tzinfo=UTC))

    def by_kind(self, kind: RemoteModelKind) -> tuple[RemoteModelEntry, ...]:
        return tuple(entry for entry in self.entries if entry.kind is kind)

    def find(self, kind: RemoteModelKind, relative_path: str) -> RemoteModelEntry | None:
        normalized = normalize_model_relative_path(relative_path)
        return next(
            (
                entry
                for entry in self.entries
                if entry.kind is kind and entry.relative_path == normalized
            ),
            None,
        )


@dataclass(frozen=True)
class ModelTransferProgress:
    """Persisted progress projection safe for UI display."""

    progress_bytes: int
    total_bytes: int
    progress_percentage: float

    def __post_init__(self) -> None:
        if self.progress_bytes < 0 or self.total_bytes < 0:
            raise ValueError("model transfer byte values must not be negative")
        if self.total_bytes and self.progress_bytes > self.total_bytes:
            raise ValueError("model transfer progress exceeds total bytes")
        if not 0.0 <= self.progress_percentage <= 100.0:
            raise ValueError("model transfer percentage must be between 0 and 100")


@dataclass(frozen=True)
class ModelTransferJob:
    """Durable transfer job without secrets or absolute local paths."""

    id: UUID
    kind: RemoteModelKind
    remote_relative_path: str
    local_relative_path: str
    remote_size_bytes: int
    remote_hash_algorithm: str | None
    remote_hash: str | None
    remote_modified_at: datetime | None
    remote_identity: str
    local_sha256: str | None
    status: ModelTransferStatus
    progress_bytes: int
    total_bytes: int
    progress_percentage: float
    worker_id: str | None
    pid: int | None
    claimed_at: datetime | None
    lease_expires_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    cancelled_at: datetime | None
    error_code: str | None
    error_summary: str | None
    retryable: bool
    created_at: datetime
    updated_at: datetime

    @property
    def progress(self) -> ModelTransferProgress:
        return ModelTransferProgress(
            self.progress_bytes,
            self.total_bytes,
            self.progress_percentage,
        )


__all__ = [
    "ModelTransferErrorCode",
    "ModelTransferJob",
    "ModelTransferProgress",
    "ModelTransferStatus",
    "RemoteModelCatalog",
    "RemoteModelEntry",
    "RemoteModelKind",
    "is_supported_model_filename",
    "normalize_model_relative_path",
]
