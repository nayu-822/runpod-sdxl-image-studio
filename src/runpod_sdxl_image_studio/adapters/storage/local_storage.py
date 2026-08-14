"""Atomic, validated local image storage."""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import GenerationKind, StoredImage

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
        *,
        kind: GenerationKind = GenerationKind.STANDARD,
        client_local_date: str | None = None,
    ) -> StoredImage:
        """Validate and atomically save an image without overwriting an existing file."""

        if not image_bytes:
            raise StorageError("Image data is empty")
        width, height = _validate_image(image_bytes)
        local_datetime = created_at.astimezone(self._timezone)
        local_date = _validated_client_date(client_local_date) or local_datetime.date().isoformat()
        folder = "upscaled" if kind is GenerationKind.UPSCALE else "generated"
        target_dir = self._data_dir / "generations" / local_date
        _ensure_safe_storage_directory(target_dir, self._data_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if client_local_date is not None:
            return self._store_with_client_sequence(
                image_bytes,
                width,
                height,
                target_dir,
            )
        target_dir = target_dir / folder
        _ensure_safe_storage_directory(target_dir, self._data_dir)
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

    def _store_with_client_sequence(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
        target_dir: Path,
    ) -> StoredImage:
        """Create a browser-date filename with an exclusive, race-safe allocation."""

        canonical = _canonical_png(image_bytes)
        digest = hashlib.sha256(canonical).hexdigest()
        for sequence in range(1, 1_000_000):
            final_path = target_dir / f"{sequence:06d}.png"
            file_descriptor: int | None = None
            try:
                file_descriptor = os.open(
                    final_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
                with os.fdopen(file_descriptor, "wb") as output:
                    file_descriptor = None
                    output.write(canonical)
                    output.flush()
                    os.fsync(output.fileno())
                return StoredImage(
                    path=final_path,
                    sha256=digest,
                    size_bytes=len(canonical),
                    width=width,
                    height=height,
                    mime_type="image/png",
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise StorageError("Could not store generated image") from exc
            finally:
                if file_descriptor is not None:
                    os.close(file_descriptor)
        raise StorageError("No available six-digit image sequence remains")


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


def _validated_client_date(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized) is None:
        raise StorageError("Client local date must use YYYY-MM-DD")
    try:
        datetime.strptime(normalized, "%Y-%m-%d")
    except ValueError as exc:
        raise StorageError("Client local date is invalid") from exc
    return normalized


def _ensure_safe_storage_directory(target_dir: Path, data_dir: Path) -> None:
    """Reject a date/folder path that escapes the configured data root via a symlink."""

    resolved_root = _normalized_windows_path(os.path.realpath(os.fspath(data_dir)))
    resolved_target = _normalized_windows_path(os.path.realpath(os.fspath(target_dir)))
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise StorageError("Storage directory is outside the data directory") from exc

    absolute_root = _normalized_windows_path(os.path.abspath(os.fspath(data_dir)))
    absolute_target = _normalized_windows_path(os.path.abspath(os.fspath(target_dir)))
    try:
        relative_parts = absolute_target.relative_to(absolute_root).parts
    except ValueError as exc:
        raise StorageError("Storage directory is outside the data directory") from exc
    current = absolute_root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise StorageError("Storage directory must not contain symlinks")


def _normalized_windows_path(value: str) -> Path:
    """Normalize Windows extended-length paths returned during concurrent mkdir."""

    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(os.path.normcase(value))


__all__ = ["LocalStorageAdapter", "StorageError"]
