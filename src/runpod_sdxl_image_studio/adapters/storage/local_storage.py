"""Atomic, validated local image storage."""

from __future__ import annotations

import hashlib
import os
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import StoredImage

from .exceptions import StorageError


class LocalStorageAdapter:
    """Store generated images below the configured data directory."""

    def __init__(self, settings: Settings | None = None) -> None:
        app_settings = settings or get_settings()
        self._data_dir = app_settings.data_dir
        try:
            self._timezone = ZoneInfo(app_settings.timezone)
        except Exception as exc:
            raise StorageError("Configured timezone is invalid") from exc

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @staticmethod
    def relative_path_from_data_dir(path: Path, data_dir: Path) -> str:
        try:
            relative = path.resolve().relative_to(data_dir.resolve())
        except ValueError as exc:
            raise StorageError("Stored path is outside the data directory") from exc
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise StorageError("Stored path is unsafe")
        return relative.as_posix()

    def relative_path(self, path: Path) -> str:
        return self.relative_path_from_data_dir(path, self._data_dir)

    def store_image(
        self,
        image_bytes: bytes,
        generation_id: UUID,
        created_at: datetime,
    ) -> StoredImage:
        """Validate and atomically save an image without overwriting an existing file."""

        if not image_bytes:
            raise StorageError("Image data is empty")
        width, height = _validate_image(image_bytes)
        local_datetime = created_at.astimezone(self._timezone)
        local_date = local_datetime.date().isoformat()
        target_dir = self._data_dir / "generations" / local_date / "generated"
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = local_datetime.strftime("%Y%m%d_%H%M%S")
        final_path = target_dir / f"{timestamp}_{generation_id.hex[:8]}.png"
        if final_path.exists():
            raise StorageError("Generation image already exists")

        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                prefix=f".{timestamp}_{generation_id.hex[:8]}.",
                suffix=".tmp",
                dir=target_dir,
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)
                temporary.write(_canonical_png(image_bytes))
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.link(temp_path, final_path)
            except FileExistsError as exc:
                raise StorageError("Generation image already exists") from exc
            temp_path.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError("Could not store generated image") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        stored_bytes = final_path.read_bytes()
        return StoredImage(
            path=final_path,
            sha256=hashlib.sha256(stored_bytes).hexdigest(),
            size_bytes=len(stored_bytes),
            width=width,
            height=height,
            mime_type="image/png",
        )


def _validate_image(image_bytes: bytes) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"PNG", "WEBP"}:
                    raise StorageError("Only PNG and WebP images are supported")
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                return image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise StorageError("Generated image is invalid") from exc


def _canonical_png(image_bytes: bytes) -> bytes:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except (UnidentifiedImageError, OSError) as exc:
        raise StorageError("Generated image could not be normalized") from exc


__all__ = ["LocalStorageAdapter", "StorageError"]
