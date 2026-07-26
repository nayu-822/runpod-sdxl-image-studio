"""Application boundary for creating and parsing immutable snapshots."""

from __future__ import annotations

from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import (
    GenerationSettingsSnapshot,
)


class GenerationSnapshotService:
    @staticmethod
    def create(settings: GenerationSettings) -> GenerationSettingsSnapshot:
        return GenerationSettingsSnapshot.from_settings(settings)

    @staticmethod
    def parse(payload: str | bytes | bytearray) -> GenerationSettingsSnapshot:
        return GenerationSettingsSnapshot.from_json(payload)
