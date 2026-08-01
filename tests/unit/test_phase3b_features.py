"""Unit coverage for Phase 3B domain and local persistence features."""

from __future__ import annotations

from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.preset_repository import (
    PresetRepository,
)
from runpod_sdxl_image_studio.domain.generation_diff import ChangeType
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationHistoryQuery,
    LoraSearchMode,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.services.generation_diff_service import GenerationDiffService
from runpod_sdxl_image_studio.services.preset_service import PresetService


def _database():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine, create_session_factory(engine)


def _settings(prompt: str, loras: tuple[LoraSetting, ...] = ()) -> GenerationSettings:
    return GenerationSettings(
        positive_prompt=prompt,
        negative_prompt="low quality",
        seed=123,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="base.safetensors",
        vae_name="vae.safetensors",
        loras=loras,
    )


def test_advanced_history_search_uses_text_and_lora_all_filters() -> None:
    engine, factory = _database()
    repository = GenerationRepository(factory)
    repository.create_pending(
        GenerationSettingsSnapshot.from_settings(
            _settings(
                "cat, blue eyes",
                (
                    LoraSetting(name="cat.safetensors", order=0),
                    LoraSetting(name="style.safetensors", order=1),
                ),
            )
        )
    )
    page = repository.list_history(
        GenerationHistoryQuery(
            text="blue",
            lora_names=("cat.safetensors", "style.safetensors"),
            lora_search_mode=LoraSearchMode.ALL,
        )
    )
    assert len(page.generations) == 1
    engine.dispose()


def test_preset_round_trip_and_apply_does_not_generate() -> None:
    engine, factory = _database()
    service = PresetService(PresetRepository(factory))
    settings = _settings("cat")
    preset = service.create_from_current_settings("  favorite  ", settings)
    result = service.apply(preset.id)
    assert preset.name == "favorite"
    assert result.settings == settings
    engine.dispose()


def test_prompt_diff_marks_reorder_and_setting_change() -> None:
    source = GenerationSettingsSnapshot.from_settings(_settings("cat, blue, eyes"))
    target = GenerationSettingsSnapshot.from_settings(
        _settings("eyes, cat, blue").model_copy(update={"seed": 456})
    )
    diff = GenerationDiffService().compare_snapshots(
        __import__("uuid").uuid4(), source, __import__("uuid").uuid4(), target
    )
    assert any(item.change_type is ChangeType.REORDERED for item in diff.positive_prompt_changes)
    assert any(item.field_name == "seed" for item in diff.setting_changes)
