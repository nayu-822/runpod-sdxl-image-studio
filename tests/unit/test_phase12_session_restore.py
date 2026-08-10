from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base, GenerationFormStateModel
from runpod_sdxl_image_studio.adapters.database.repositories import (
    generation_form_state_repository,
)
from runpod_sdxl_image_studio.domain.generation import Generation, GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_form_state import (
    FormSeedMode,
    GenerationFormStateError,
    GenerationFormStateSnapshot,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import (
    GenerationSettingsSnapshot,
    LoraSettingSnapshot,
)
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.services.generation_form_state_service import (
    GenerationFormStateService,
)


def _snapshot() -> GenerationFormStateSnapshot:
    return GenerationFormStateSnapshot.from_ui(
        positive_prompt="positive",
        negative_prompt="negative",
        seed_mode="Fixed",
        seed=123,
        width=1024,
        height=832,
        steps=28,
        cfg_scale=5.5,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="checkpoints/model.safetensors",
        vae_name="vae/clear.vae.safetensors",
        loras=(
            LoraSetting(
                name="style/one.safetensors",
                model_strength=0.7,
                clip_strength=0.8,
                order=0,
            ),
            LoraSetting(
                name="style/two.safetensors",
                model_strength=0.4,
                clip_strength=0.5,
                order=1,
            ),
        ),
    )


def _generation() -> Generation:
    execution = GenerationSettingsSnapshot(
        positive_prompt="old positive",
        negative_prompt="old negative",
        seed=77,
        width=768,
        height=1024,
        steps=20,
        cfg_scale=6.0,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="old.safetensors",
        vae_name=None,
        loras=(
            LoraSettingSnapshot(
                name="old_lora.safetensors",
                model_strength=1.2,
                clip_strength=0.9,
                order=0,
            ),
        ),
        workflow_template_id="sdxl_txt2img",
        workflow_template_version="1",
    )
    timestamp = datetime(2026, 8, 10, tzinfo=UTC)
    return Generation(
        id=uuid4(),
        kind=GenerationKind.STANDARD,
        status=GenerationStatus.COMPLETED,
        parent_generation_id=None,
        settings_snapshot=execution,
        workflow_template_id="sdxl_txt2img",
        workflow_template_version="1",
        comfy_prompt_id="prompt",
        favorite=False,
        user_note=None,
        error_code=None,
        error_summary=None,
        created_at=timestamp,
        started_at=timestamp,
        completed_at=timestamp,
        updated_at=timestamp,
    )


def test_form_state_round_trips_schema_seed_mode_and_lora_order() -> None:
    snapshot = _snapshot()
    assert snapshot.seed_mode is FormSeedMode.FIXED
    assert [item.name for item in snapshot.loras] == [
        "style/one.safetensors",
        "style/two.safetensors",
    ]
    assert snapshot.loras[0].model_strength == 0.7
    restored = GenerationFormStateSnapshot.from_json(snapshot.to_json())
    assert restored == snapshot
    assert restored.ui_seed_mode == "Fixed"


def test_form_state_rejects_unknown_schema_without_breaking_restore() -> None:
    with pytest.raises(GenerationFormStateError):
        GenerationFormStateSnapshot.from_json('{"schema_version": 999, "seed_mode": "fixed"}')

    repository = _repository()
    service = GenerationFormStateService(repository, lambda: _generation())
    result = service.restore()
    assert result.source == "generation_snapshot"
    assert result.snapshot is not None
    assert result.snapshot.seed_mode is FormSeedMode.FIXED
    assert result.snapshot.seed == 77


def test_form_state_repository_persists_only_explicitly_saved_state() -> None:
    repository = _repository()
    assert repository.get() is None
    service = GenerationFormStateService(repository)
    snapshot = _snapshot()
    service.save(snapshot)
    assert repository.get() == snapshot


def test_legacy_generation_fallback_includes_optional_upscaler() -> None:
    repository = _repository()
    service = GenerationFormStateService(
        repository,
        lambda: _generation(),
        upscaler_provider=lambda generation_id: "upscale-model.pth",
    )
    result = service.restore()
    assert result.snapshot is not None
    assert result.snapshot.upscaler_name == "upscale-model.pth"


def test_invalid_persisted_form_state_is_safe_warning() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        session.add(
            GenerationFormStateModel(
                id="current",
                schema_version=999,
                snapshot_json='{"schema_version": 999}',
                updated_at=datetime.now(UTC),
            )
        )
        session.commit()

    result = GenerationFormStateService(
        generation_form_state_repository.GenerationFormStateRepository(session_factory),
        lambda: _generation(),
    ).restore()
    assert result.snapshot is not None
    assert result.source == "generation_snapshot"
    assert result.warning is not None


def _repository() -> generation_form_state_repository.GenerationFormStateRepository:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return generation_form_state_repository.GenerationFormStateRepository(
        create_session_factory(engine)
    )
