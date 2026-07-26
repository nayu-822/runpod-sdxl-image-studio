"""Application-facing system status DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUISystemStats,
)


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
