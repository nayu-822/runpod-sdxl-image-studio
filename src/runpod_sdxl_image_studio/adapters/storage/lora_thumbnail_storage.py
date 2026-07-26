"""Validated, atomic storage for user-provided LoRA thumbnails."""

from __future__ import annotations

import os
import uuid
import warnings
from io import BytesIO
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageOps

from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError


class LoraThumbnailStorage:
    """Store thumbnails under an internal UUID filename only."""

    def __init__(self, root: Path, max_bytes: int, max_edge: int) -> None:
        self._root = root
        self._max_bytes = max_bytes
        self._max_edge = max_edge

    def save(self, metadata_id: UUID, payload: bytes) -> str:
        if len(payload) > self._max_bytes:
            raise StorageError("thumbnail exceeds the configured size limit")
        try:
            image = self._decode(payload)
            image = ImageOps.exif_transpose(image)
            if max(image.size) > self._max_edge:
                image.thumbnail((self._max_edge, self._max_edge), Image.Resampling.LANCZOS)
            converted = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            self._root.mkdir(parents=True, exist_ok=True)
            final_path = self._final_path(metadata_id)
            temporary_path = self._root / f".{metadata_id}.{uuid.uuid4().hex}.tmp"
            try:
                converted.save(temporary_path, format="WEBP")
                os.replace(temporary_path, final_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            return f"lora_thumbnails/{metadata_id}.webp"
        except StorageError:
            raise
        except Exception as exc:  # noqa: BLE001 - storage boundary hides decoder details
            raise StorageError("thumbnail could not be validated or saved") from exc

    def read(self, relative_path: str | None) -> bytes | None:
        path = self._safe_relative_path(relative_path)
        if path is None or not path.exists():
            return None
        return path.read_bytes()

    def delete(self, relative_path: str | None) -> None:
        path = self._safe_relative_path(relative_path)
        if path is not None:
            path.unlink(missing_ok=True)

    def path_for(self, metadata_id: UUID) -> Path | None:
        path = self._final_path(metadata_id)
        return path if path.exists() else None

    def _decode(self, payload: bytes) -> Image.Image:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source:
                source.verify()
            with Image.open(BytesIO(payload)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise StorageError("only PNG, JPEG, and WebP thumbnails are supported")
                image = source.copy()
        return image

    def _final_path(self, metadata_id: UUID) -> Path:
        return self._root / f"{metadata_id}.webp"

    def _safe_relative_path(self, relative_path: str | None) -> Path | None:
        if relative_path is None:
            return None
        candidate = Path(relative_path)
        if candidate.name != f"{candidate.stem}.webp" or candidate.suffix.lower() != ".webp":
            raise StorageError("invalid thumbnail path")
        try:
            UUID(candidate.stem)
        except ValueError as exc:
            raise StorageError("invalid thumbnail path") from exc
        resolved = (self._root / candidate.name).resolve()
        root = self._root.resolve()
        if resolved.parent != root:
            raise StorageError("invalid thumbnail path")
        return resolved
