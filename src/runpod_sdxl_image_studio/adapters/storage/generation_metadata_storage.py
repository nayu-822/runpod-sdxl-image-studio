"""Atomic UTF-8 sidecar JSON storage for generation records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter


class GenerationMetadataStorage:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def save_for_image(self, image_path: Path, payload: dict[str, object]) -> Path:
        target = image_path.with_suffix(".json")
        temporary: Path | None = None
        try:
            encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            with NamedTemporaryFile(
                mode="wb", prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
            return target
        except (OSError, TypeError, ValueError) as exc:
            raise StorageError("Generation metadata could not be stored") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def relative_path(self, path: Path) -> str:
        return LocalStorageAdapter.relative_path_from_data_dir(path, self._data_dir)

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
