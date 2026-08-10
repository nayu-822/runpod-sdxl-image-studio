from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.model_transfer import (
    RemoteModelCatalog,
    RemoteModelEntry,
    RemoteModelKind,
)
from runpod_sdxl_image_studio.services.model_preparation_service import ModelPreparationService


class _CatalogAdapter:
    def __init__(self, catalog: RemoteModelCatalog) -> None:
        self.catalog = catalog

    async def list_catalog(self) -> RemoteModelCatalog:
        return self.catalog


class _PreparationService(ModelPreparationService):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.prepared: list[str] = []

    async def prepare_entry(self, entry: RemoteModelEntry) -> SimpleNamespace:
        self.prepared.append(f"{entry.kind.value}:{entry.relative_path}")
        return SimpleNamespace(remote_relative_path=entry.relative_path)


@pytest.mark.asyncio
async def test_startup_prepares_exact_available_models_and_reports_missing(
    tmp_path: Path,
) -> None:
    catalog = RemoteModelCatalog(
        entries=(
            RemoteModelEntry(
                RemoteModelKind.CHECKPOINT,
                "checkpoints/A.safetensors",
                "A",
                1,
            ),
            RemoteModelEntry(
                RemoteModelKind.LORA,
                "loras/B.safetensors",
                "B",
                1,
            ),
        ),
        fetched_at=datetime.now(UTC),
    )
    settings = Settings(
        remote_model_enabled=True,
        rclone_remote="drive",
        checkpoint_dir=tmp_path / "checkpoints",
        lora_dir=tmp_path / "loras",
        vae_dir=tmp_path / "vae",
        upscaler_dir=tmp_path / "upscalers",
    )
    service = _PreparationService(
        SimpleNamespace(),
        _CatalogAdapter(catalog),
        settings,
        lambda: None,
    )
    result = await service.prepare_previous_models(
        "checkpoints/A.safetensors",
        None,
        ("loras/B.safetensors", "loras/MISSING.safetensors"),
    )
    assert service.prepared == ["checkpoint:checkpoints/A.safetensors", "lora:loras/B.safetensors"]
    assert result.missing == ("lora:loras/MISSING.safetensors",)
    assert "substitute" in result.message
