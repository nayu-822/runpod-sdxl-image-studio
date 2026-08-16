"""Typed data transfer models for ComfyUI responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class PromptHistoryStatus(StrEnum):
    """Normalized status values returned by ComfyUI history."""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    IN_PROGRESS = "in_progress"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


class RemotePromptStatus(StrEnum):
    """Typed status returned by the remote prompt reconciliation adapter."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ComfyUIDeviceInfo:
    """A device reported by ComfyUI's system stats endpoint."""

    name: str | None
    device_type: str | None
    index: int | None
    vram_total: int | None
    vram_free: int | None
    torch_vram_total: int | None
    torch_vram_free: int | None


@dataclass(frozen=True)
class ComfyUISystemStats:
    """The supported, normalized subset of ``/system_stats``."""

    system_os: str | None
    python_version: str | None
    embedded_python: bool | None
    comfyui_version: str | None
    devices: tuple[ComfyUIDeviceInfo, ...]


@dataclass(frozen=True)
class ComfyUIObjectInfo:
    """A typed container for node definitions from ``/object_info``.

    Node input schemas vary by ComfyUI version, so their internals remain an
    intentionally narrow mapping. The parser is the only consumer of this
    version-dependent data.
    """

    nodes: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ComfyUIConnectionResult:
    """The result of an explicit connection check."""

    is_connected: bool
    message: str
    checked_at: datetime
    system_stats: ComfyUISystemStats | None


@dataclass(frozen=True)
class ComfyUICapabilities:
    """Selectable capabilities extracted from ComfyUI node definitions."""

    checkpoints: tuple[str, ...]
    vaes: tuple[str, ...]
    samplers: tuple[str, ...]
    schedulers: tuple[str, ...]
    loras: tuple[str, ...]
    upscale_models: tuple[str, ...]
    available_node_classes: frozenset[str]
    warnings: tuple[str, ...]
    detector_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueuedPrompt:
    """The safe subset of a ``/prompt`` response."""

    prompt_id: str
    number: int | None
    node_errors: Mapping[str, object]


@dataclass(frozen=True)
class ComfyUIOutputImage:
    """An image reference returned by ComfyUI history."""

    filename: str
    subfolder: str
    output_type: str


@dataclass(frozen=True)
class PromptHistory:
    """Normalized completion and output state for one prompt."""

    prompt_id: str
    is_completed: bool
    is_failed: bool
    outputs: tuple[ComfyUIOutputImage, ...]
    error_message: str | None
    exists: bool = True
    status: PromptHistoryStatus = PromptHistoryStatus.UNKNOWN

    def __post_init__(self) -> None:
        """Infer the status for backwards-compatible test and adapter fixtures."""

        if self.status is not PromptHistoryStatus.UNKNOWN:
            return
        inferred = (
            PromptHistoryStatus.NOT_FOUND
            if not self.exists
            else PromptHistoryStatus.COMPLETED
            if self.is_completed
            else PromptHistoryStatus.FAILED
            if self.is_failed
            else PromptHistoryStatus.UNKNOWN
        )
        object.__setattr__(self, "status", inferred)

    @property
    def is_interrupted(self) -> bool:
        """Whether ComfyUI reported an execution interruption."""

        return self.status is PromptHistoryStatus.INTERRUPTED


@dataclass(frozen=True)
class RemotePromptState:
    """A normalized remote state with no ComfyUI JSON exposed to the domain."""

    prompt_id: str
    status: RemotePromptStatus


@dataclass(frozen=True)
class ComfyUIQueueStatus:
    """The prompt IDs currently pending or running in ComfyUI."""

    pending_prompt_ids: tuple[str, ...]
    running_prompt_ids: tuple[str, ...]
