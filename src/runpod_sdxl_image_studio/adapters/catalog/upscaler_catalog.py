"""Safe catalog of acquired ComfyUI upscaler models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpscalerCatalog:
    """``models`` is None when the configured directory is unavailable."""

    models: tuple[str, ...] | None

    @classmethod
    def scan(cls, root_dir: Path) -> UpscalerCatalog:
        try:
            root = root_dir.resolve(strict=True)
        except OSError:
            return cls(None)
        if not root.is_dir():
            return cls(None)
        found: list[str] = []
        for candidate in root.rglob("*"):
            if candidate.suffix.lower() not in {".safetensors", ".pth", ".pt", ".bin"}:
                continue
            try:
                resolved = candidate.resolve(strict=True)
                relative = resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if candidate.is_file() and all(part not in {"", ".", ".."} for part in relative.parts):
                found.append(relative.as_posix())
        return cls(tuple(sorted(set(found))))

    def contains(self, name: str) -> bool:
        return self.models is not None and name.replace("\\", "/") in self.models


__all__ = ["UpscalerCatalog"]
