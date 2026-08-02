"""Validation shared by live and restart-time upscale execution."""

from __future__ import annotations

from io import BytesIO
from uuid import UUID

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepositoryProtocol,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSettingsSnapshot
from runpod_sdxl_image_studio.services.upscale_enqueue_service import UpscaleEnqueueError


def validate_upscale_output(
    generation_id: UUID,
    image_bytes: bytes,
    settings_repository: UpscaleSettingsRepositoryProtocol,
) -> UpscaleSettingsSnapshot:
    """Validate output bytes against the durable target dimensions."""

    snapshot = settings_repository.get_by_generation(generation_id)
    if snapshot is None:
        raise UpscaleEnqueueError("upscale_settings_missing", "upscale settings were not found")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise UpscaleEnqueueError(
            "upscale_output_invalid", "ComfyUI returned an invalid upscale image"
        ) from exc
    if (width, height) != (snapshot.target_width, snapshot.target_height):
        raise UpscaleEnqueueError(
            "upscale_output_dimension_mismatch",
            "upscale output dimensions do not match the persisted target",
        )
    return snapshot


__all__ = ["validate_upscale_output"]
