"""Atomic thumbnail storage for generation history."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings, get_settings


class HistoryThumbnailStorage:
    """Create metadata-free WebP thumbnails without modifying original images."""

    def __init__(self, settings: Settings | None = None) -> None:
        app_settings = settings or get_settings()
        self._data_dir = app_settings.data_dir
        self._max_edge = min(app_settings.history_thumbnail_max_edge, 2048)
        self._timezone = ZoneInfo(app_settings.timezone)

    def save(
        self,
        image_path: Path,
        generation_id: UUID,
        created_at: datetime,
        *,
        display_order: int = 0,
    ) -> Path:
        if display_order < 0:
            raise StorageError("Thumbnail display order must not be negative")
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image.thumbnail((self._max_edge, self._max_edge), Image.Resampling.LANCZOS)
                output = BytesIO()
                image.save(output, format="WEBP", method=6)
                payload = output.getvalue()
        except (OSError, UnidentifiedImageError) as exc:
            raise StorageError("History thumbnail could not be created") from exc
        local_date = created_at.astimezone(self._timezone).date().isoformat()
        target_dir = self._data_dir / "generations" / local_date / "thumbnails"
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if display_order == 0 else f"_{display_order:06d}"
        target = target_dir / f"{generation_id}{suffix}.webp"
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", prefix=f".{generation_id}.", suffix=".tmp", dir=target_dir, delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            raise StorageError("History thumbnail could not be stored") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    def relative_path(self, path: Path) -> str:
        return LocalStorageAdapter.relative_path_from_data_dir(path, self._data_dir)

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
