"""Injectable local filesystem capacity adapter."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DiskUsage:
    """Normalized ``shutil.disk_usage`` result."""

    total_bytes: int
    used_bytes: int
    free_bytes: int


class DiskUsageAdapterProtocol(Protocol):
    """Filesystem boundary used by health and preflight services."""

    def usage(self, path: Path) -> DiskUsage: ...


class LocalDiskUsageAdapter(DiskUsageAdapterProtocol):
    """Read capacity from the local filesystem without any RunPod API call."""

    def __init__(
        self,
        disk_usage: Callable[[str | bytes | Path], shutil._ntuple_diskusage] = shutil.disk_usage,
    ) -> None:
        self._disk_usage = disk_usage

    def usage(self, path: Path) -> DiskUsage:
        target = _nearest_existing_path(path)
        usage = self._disk_usage(target)
        return DiskUsage(
            total_bytes=max(0, int(usage.total)),
            used_bytes=max(0, int(usage.used)),
            free_bytes=max(0, int(usage.free)),
        )


def _nearest_existing_path(path: Path) -> Path:
    """Allow first-run deployments whose data directory is not created yet."""

    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else Path.cwd()


DiskUsageAdapter = LocalDiskUsageAdapter

__all__ = ["DiskUsage", "DiskUsageAdapter", "DiskUsageAdapterProtocol", "LocalDiskUsageAdapter"]
