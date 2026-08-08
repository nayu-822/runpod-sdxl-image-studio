"""Safe canonical storage and revalidation for externally imported images."""

from __future__ import annotations

import hashlib
import ntpath
import os
import warnings
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.metadata_import import ImportedImage


class ImportedImageStorageError(RuntimeError):
    """Safe storage/revalidation error with a stable import error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ImportedImageStorage:
    """Store upload bytes as a metadata-free canonical PNG below ``data_dir``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        try:
            self._timezone = ZoneInfo(settings.timezone)
        except Exception as exc:
            raise ImportedImageStorageError(
                "metadata_import_invalid_image", "configured timezone is invalid"
            ) from exc

    @property
    def data_dir(self) -> Path:
        return self._settings.data_dir

    def store(
        self,
        image_bytes: bytes,
        original_filename: str | None = None,
        *,
        imported_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> ImportedImage:
        if not image_bytes:
            raise ImportedImageStorageError(
                "metadata_import_invalid_image", "image bytes are empty"
            )
        if len(image_bytes) > self._settings.max_metadata_import_image_bytes:
            raise ImportedImageStorageError(
                "metadata_import_too_large", "image exceeds the configured size limit"
            )
        width, height, canonical = _canonical_png(image_bytes)
        image_id = imported_id or uuid4()
        timestamp = created_at or datetime.now(UTC)
        local_date = timestamp.astimezone(self._timezone).date().isoformat()
        relative_path = Path("imports") / local_date / "images" / f"{image_id}.png"
        data_root, safe_parent = _prepare_safe_directory(
            self._settings.data_dir, relative_path.parent
        )
        target = safe_parent / relative_path.name
        try:
            target.resolve(strict=False).relative_to(data_root)
        except (OSError, ValueError) as exc:
            raise ImportedImageStorageError(
                "metadata_import_storage_failed", "import image path is outside the data root"
            ) from exc
        if target.is_symlink() or target.exists():
            raise ImportedImageStorageError(
                "metadata_import_storage_failed", "import image already exists"
            )
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", prefix=f".{image_id}.", suffix=".tmp", dir=target.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(canonical)
                file.flush()
                os.fsync(file.fileno())
            os.link(temporary, target)
            target.resolve(strict=True).relative_to(data_root)
        except FileExistsError as exc:
            raise ImportedImageStorageError(
                "metadata_import_storage_failed", "import image already exists"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ImportedImageStorageError(
                "metadata_import_storage_failed", "import image could not be stored"
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        safe_name = _safe_original_filename(original_filename)
        return ImportedImage(
            id=image_id,
            original_filename=safe_name,
            stored_image_path=relative_path.as_posix(),
            source_image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            stored_image_sha256=hashlib.sha256(canonical).hexdigest(),
            image_width=width,
            image_height=height,
            image_mime_type="image/png",
            created_at=timestamp,
        )

    store_image = store

    def verify(self, imported_image: ImportedImage) -> ImportedImage:
        """Revalidate path, symlink containment, hash, dimensions, and image format."""

        relative = _safe_relative_path(imported_image.stored_image_path)
        data_root = self._settings.data_dir.resolve()
        path = self._settings.data_dir.joinpath(*relative.parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(data_root)
        except (OSError, ValueError) as exc:
            raise ImportedImageStorageError(
                "metadata_import_source_changed", "imported image path is unavailable"
            ) from exc
        if not resolved.is_file():
            raise ImportedImageStorageError(
                "metadata_import_source_changed", "imported image file is unavailable"
            )
        try:
            image_bytes = resolved.read_bytes()
        except OSError as exc:
            raise ImportedImageStorageError(
                "metadata_import_source_changed", "imported image could not be read"
            ) from exc
        if len(image_bytes) > self._settings.max_metadata_import_image_bytes:
            raise ImportedImageStorageError(
                "metadata_import_source_changed", "imported image exceeds the size limit"
            )
        try:
            width, height, canonical = _canonical_png(image_bytes)
        except ImportedImageStorageError as exc:
            raise ImportedImageStorageError(
                "metadata_import_source_changed", "imported image contents no longer match"
            ) from exc
        stored_hash = hashlib.sha256(image_bytes).hexdigest()
        if (
            stored_hash != imported_image.stored_image_sha256
            or width != imported_image.image_width
            or height != imported_image.image_height
            or imported_image.image_mime_type != "image/png"
            or canonical != image_bytes
        ):
            raise ImportedImageStorageError(
                "metadata_import_source_changed", "imported image no longer matches its record"
            )
        return imported_image

    def read_verified(self, imported_image: ImportedImage) -> bytes:
        self.verify(imported_image)
        relative = _safe_relative_path(imported_image.stored_image_path)
        return (self._settings.data_dir.joinpath(*relative.parts)).read_bytes()

    def absolute_path(self, imported_image: ImportedImage) -> Path:
        self.verify(imported_image)
        relative = _safe_relative_path(imported_image.stored_image_path)
        return self._settings.data_dir.joinpath(*relative.parts)

    def cleanup_uncommitted(self, imported_image: ImportedImage) -> bool:
        """Remove only a newly-created canonical image whose DB row never committed.

        This is deliberately stricter than normal file cleanup.  A path must
        still be the UUID-based canonical import path, resolve below the data
        root without a symlink, and contain exactly the bytes recorded by the
        store operation.  Any uncertainty leaves the file in place.
        """

        try:
            relative = _safe_relative_path(imported_image.stored_image_path)
            if (
                len(relative.parts) != 4
                or relative.parts[0] != "imports"
                or relative.parts[2] != "images"
                or relative.name != f"{imported_image.id}.png"
            ):
                return False
            root = self._settings.data_dir.resolve()
            path = self._settings.data_dir.joinpath(*relative.parts)
            if path.is_symlink():
                return False
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file():
                return False
            contents = resolved.read_bytes()
            if hashlib.sha256(contents).hexdigest() != imported_image.stored_image_sha256:
                return False
            resolved.unlink()
            return True
        except (OSError, ValueError, ImportedImageStorageError):
            return False


def _canonical_png(image_bytes: bytes) -> tuple[int, int, bytes]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"PNG", "WEBP"}:
                    raise ImportedImageStorageError(
                        "metadata_import_invalid_image", "only PNG and WebP images are supported"
                    )
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                image.load()
                output = BytesIO()
                image.save(output, format="PNG")
                return width, height, output.getvalue()
    except ImportedImageStorageError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise ImportedImageStorageError(
            "metadata_import_invalid_image", "image contents could not be verified"
        ) from exc


def _safe_original_filename(filename: str | None) -> str:
    value = ntpath.basename((filename or "import.png").replace("/", "\\")).strip()
    if not value or value in {".", ".."} or "\x00" in value:
        return "import.png"
    return value[:500]


def _safe_relative_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ntpath.isabs(value)
        or "\x00" in normalized
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ImportedImageStorageError(
            "metadata_import_source_changed", "imported image path is unsafe"
        )
    return Path(*normalized.split("/"))


def _prepare_safe_directory(data_dir: Path, relative: Path) -> tuple[Path, Path]:
    """Create a relative directory while rejecting symlink escapes."""

    try:
        data_root = data_dir.resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        if not data_root.is_dir():
            raise OSError("configured data root is not a directory")
        current = data_root
        for component in relative.parts:
            candidate = current / component
            if candidate.is_symlink():
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(data_root)
                if not resolved.is_dir():
                    raise OSError("storage parent is not a directory")
                current = resolved
                continue
            if candidate.exists():
                if not candidate.is_dir():
                    raise OSError("storage parent is not a directory")
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(data_root)
                current = resolved
                continue
            candidate.mkdir()
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(data_root)
            current = resolved
        current.resolve(strict=True).relative_to(data_root)
        return data_root, current
    except (OSError, ValueError) as exc:
        raise ImportedImageStorageError(
            "metadata_import_storage_failed", "import image storage path is unsafe"
        ) from exc


__all__ = ["ImportedImageStorage", "ImportedImageStorageError"]
