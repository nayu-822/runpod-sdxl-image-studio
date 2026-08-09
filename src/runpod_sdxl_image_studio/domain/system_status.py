"""Application-facing system status DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUISystemStats,
)


@dataclass(frozen=True)
class CapabilityRefreshResult:
    """Service result for a capability refresh operation."""

    is_success: bool
    message: str
    capabilities: ComfyUICapabilities | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComfyUIStatus:
    """Aggregated status data suitable for a UI view model."""

    is_connected: bool
    message: str
    checked_at: datetime | None
    system_stats: ComfyUISystemStats | None
    capabilities: ComfyUICapabilities | None
    warnings: tuple[str, ...]
    error_summary: str | None


class SystemHealthStatus(StrEnum):
    """Overall status used by the operational health view."""

    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"


class ErrorSeverity(StrEnum):
    """Severity values persisted for operational error history."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ComfyUIHealthView:
    """Safe ComfyUI health projection for the application and UI layers."""

    connected: bool
    message: str
    version: str | None
    gpu_name: str | None
    vram_total: int | None
    vram_free: int | None


@dataclass(frozen=True)
class GenerationQueueHealthView:
    """Bounded queue counters shown in the system health card."""

    pending_count: int
    running_count: int
    failed_count: int


@dataclass(frozen=True)
class StorageHealthView:
    """Local filesystem capacity and unsynchronized artifact size."""

    local_total_bytes: int
    local_free_bytes: int
    local_used_bytes: int
    unsynced_bytes: int


@dataclass(frozen=True)
class DriveHealthView:
    """Google Drive connection and synchronization counters."""

    configured: bool
    connected: bool
    last_sync_at: datetime | None
    pending_sync_count: int
    failed_sync_count: int


@dataclass(frozen=True)
class ModelHealthView:
    """Available model counts from the current ComfyUI capability snapshot."""

    checkpoint_count: int
    lora_count: int
    vae_count: int
    upscaler_count: int


@dataclass(frozen=True)
class SystemErrorEvent:
    """A bounded, sanitized operational error history entry."""

    id: UUID
    created_at: datetime
    category: str
    severity: ErrorSeverity | str
    error_code: str
    summary: str
    generation_id: UUID | None
    job_id: UUID | None
    retryable: bool
    details: str | None = None


@dataclass(frozen=True)
class SystemHealthView:
    """Application-level aggregation consumed by the System tab."""

    checked_at: datetime
    overall_status: SystemHealthStatus | str
    comfyui: ComfyUIHealthView
    queue: GenerationQueueHealthView
    storage: StorageHealthView
    drive: DriveHealthView
    models: ModelHealthView
    recent_errors: tuple[SystemErrorEvent, ...] = ()

    # Flat aliases keep the DTO convenient for callers that do not need the
    # nested presentation objects and preserve the field vocabulary from the
    # Phase 9 implementation brief.
    @property
    def comfyui_connected(self) -> bool:
        return self.comfyui.connected

    @property
    def comfyui_message(self) -> str:
        return self.comfyui.message

    @property
    def comfyui_version(self) -> str | None:
        return self.comfyui.version

    @property
    def gpu_name(self) -> str | None:
        return self.comfyui.gpu_name

    @property
    def vram_total(self) -> int | None:
        return self.comfyui.vram_total

    @property
    def vram_free(self) -> int | None:
        return self.comfyui.vram_free

    @property
    def pending_count(self) -> int:
        return self.queue.pending_count

    @property
    def running_count(self) -> int:
        return self.queue.running_count

    @property
    def failed_count(self) -> int:
        return self.queue.failed_count

    @property
    def local_total_bytes(self) -> int:
        return self.storage.local_total_bytes

    @property
    def local_free_bytes(self) -> int:
        return self.storage.local_free_bytes

    @property
    def local_used_bytes(self) -> int:
        return self.storage.local_used_bytes

    @property
    def unsynced_bytes(self) -> int:
        return self.storage.unsynced_bytes

    @property
    def drive_configured(self) -> bool:
        return self.drive.configured

    @property
    def drive_connected(self) -> bool:
        return self.drive.connected

    @property
    def last_sync_at(self) -> datetime | None:
        return self.drive.last_sync_at

    @property
    def pending_sync_count(self) -> int:
        return self.drive.pending_sync_count

    @property
    def failed_sync_count(self) -> int:
        return self.drive.failed_sync_count

    @property
    def checkpoint_count(self) -> int:
        return self.models.checkpoint_count

    @property
    def lora_count(self) -> int:
        return self.models.lora_count

    @property
    def vae_count(self) -> int:
        return self.models.vae_count

    @property
    def upscaler_count(self) -> int:
        return self.models.upscaler_count
