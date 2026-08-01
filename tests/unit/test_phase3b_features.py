"""Unit coverage for Phase 3B domain and local persistence features."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.preset_repository import (
    PresetRepository,
    PresetRepositoryError,
)
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_diff import ChangeType
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationHistoryQuery,
    GenerationHistorySort,
    LoraSearchMode,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.preset import Preset
from runpod_sdxl_image_studio.domain.preset_payload import LoraPresetPayload, PresetKind
from runpod_sdxl_image_studio.services.generation_diff_service import GenerationDiffService
from runpod_sdxl_image_studio.services.preset_service import PresetService
from runpod_sdxl_image_studio.services.recent_settings_service import RecentSettingsService
from runpod_sdxl_image_studio.ui.tabs.history_tab import seed_copy_value
from runpod_sdxl_image_studio.ui.tabs.preset_tab import (
    make_preset_apply_handler,
    make_preset_save_handler,
    make_recent_checkpoint_handler,
    make_recent_lora_add_handler,
    make_recent_vae_handler,
    preset_apply_output_count,
)


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


def test_history_search_supports_special_text_parent_and_multiple_filters() -> None:
    engine, factory = _database()
    repository = GenerationRepository(factory)
    now = datetime.now(UTC)
    parent = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings("Unicode 猫 100% _ \\")),
        created_at=now - timedelta(days=1),
    )
    repository.update_note(parent.id, "note 100% _ \\")
    repository.set_favorite(parent.id, True)
    child = repository.create_pending(
        GenerationSettingsSnapshot.from_settings(
            _settings(
                "child",
                (LoraSetting(name="cat.safetensors", order=0),),
            )
        ),
        parent_generation_id=parent.id,
        kind=GenerationKind.DERIVED,
    )
    assert repository.list_history(GenerationHistoryQuery(text="100%")).total_count == 1
    assert repository.list_history(GenerationHistoryQuery(text="_")).total_count == 1
    assert repository.list_history(GenerationHistoryQuery(text="\\")).total_count == 1
    assert repository.list_history(
        GenerationHistoryQuery(
            parent_generation_id=parent.id,
            statuses=(GenerationStatus.PENDING,),
            kinds=(GenerationKind.DERIVED,),
            favorite_only=False,
            sort=GenerationHistorySort.OLDEST,
        )
    ).generations == (child,)
    engine.dispose()


def test_preset_repository_service_crud_apply_and_ui_handlers() -> None:
    engine, factory = _database()
    service = PresetService(PresetRepository(factory))
    settings = _settings("cat")
    generation = service.create_from_current_settings("generation", settings)
    prompt = service.create_prompt_preset("prompt", "blue", "bad")
    result = service.apply(prompt.id, current_settings=settings, prompt_mode="append")
    assert result.settings.positive_prompt == "cat, blue"
    assert service.set_favorite(generation.id, True).favorite is True
    assert service.duplicate(generation.id).name.startswith("generation (copy)")
    assert service.apply(generation.id, current_settings=settings).settings == settings

    save = make_preset_save_handler(service, 2)
    saved = save(
        "prompt",
        "ui prompt",
        "description",
        False,
        "positive",
        "negative",
        1024,
        1024,
        "Fixed",
        123,
        28,
        5.5,
        "euler",
        "normal",
        "base.safetensors",
        None,
        [],
        "replace",
        "replace",
        "",
        "",
        False,
    )
    assert len(saved) == 10
    selected_id = str(service.search("ui prompt")[0].id)
    apply = make_preset_apply_handler(service, 2)
    applied = apply(
        selected_id,
        "append",
        "replace",
        "current",
        "negative",
        1024,
        1024,
        "Random",
        -1,
        28,
        5.5,
        "euler",
        "normal",
        "base.safetensors",
        None,
        [],
        None,
        None,
        None,
    )
    assert len(applied) == 17 + 7 * 2
    assert applied[3] == "current, positive"
    engine.dispose()


def test_recent_settings_are_bounded_and_seed_copy_uses_resolved_integer() -> None:
    engine, factory = _database()
    repository = GenerationRepository(factory)
    for index in range(3):
        repository.create_pending(
            GenerationSettingsSnapshot.from_settings(
                _settings(f"prompt-{index}").model_copy(update={"seed": index + 1})
            )
        )
    service = RecentSettingsService(repository, PresetRepository(factory), limit=2)
    recent = service.get_recent()
    assert len(recent.checkpoints) <= 2
    assert len(recent.recent_generation_ids) <= 2
    assert seed_copy_value(987654321) == "987654321"
    engine.dispose()


def test_phase3b_migration_backfills_and_preserves_phase2_tables(tmp_path: Path) -> None:
    database = tmp_path / "migration.sqlite3"
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option("script_location", str(Path(__file__).parents[2] / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    command.upgrade(config, "0002_generation_history")
    engine = create_engine(f"sqlite:///{database.as_posix()}")
    valid_id = str(uuid4())
    broken_id = str(uuid4())
    valid_snapshot = {
        "schema_version": 1,
        "positive_prompt": "Unicode 猫",
        "negative_prompt": "bad",
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg_scale": 5.5,
        "sampler_name": "euler",
        "scheduler_name": "normal",
        "checkpoint_name": "base.safetensors",
        "vae_name": None,
        "loras": [
            {"name": "style.safetensors", "order": 0, "model_strength": 0.8, "clip_strength": 0.7}
        ],
        "workflow_template_id": "sdxl_txt2img",
        "workflow_template_version": "1.0",
    }
    with engine.begin() as connection:
        for generation_id, payload in (
            (valid_id, json.dumps(valid_snapshot)),
            (broken_id, "not-json"),
        ):
            connection.execute(
                text(
                    """INSERT INTO generations
                    (id, kind, status, parent_generation_id, settings_snapshot_json,
                     snapshot_schema_version, workflow_template_id, workflow_template_version,
                     favorite, created_at, updated_at)
                    VALUES (:id, 'standard', 'pending', NULL, :payload, 1,
                            'sdxl_txt2img', '1.0', 0, :created, :created)"""
                ),
                {"id": generation_id, "payload": payload, "created": datetime.now(UTC)},
            )
    engine.dispose()
    command.upgrade(config, "head")
    upgraded = create_engine(f"sqlite:///{database.as_posix()}")
    inspector = inspect(upgraded)
    assert inspector.has_table("generation_loras")
    assert inspector.has_table("presets")
    assert "ix_generations_checkpoint" in {
        item["name"] for item in inspector.get_indexes("generations")
    }
    with upgraded.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM generation_loras")).scalar_one() == 1
        assert connection.execute(
            text("SELECT checkpoint_name, positive_prompt_search FROM generations WHERE id=:id"),
            {"id": valid_id},
        ).one() == ("base.safetensors", "Unicode 猫")
    upgraded.dispose()
    command.downgrade(config, "0002_generation_history")
    downgraded = create_engine(f"sqlite:///{database.as_posix()}")
    assert not inspect(downgraded).has_table("presets")
    assert inspect(downgraded).has_table("generations")
    downgraded.dispose()
    command.upgrade(config, "head")


def test_preset_apply_success_and_all_error_paths_keep_exact_output_count() -> None:
    engine, factory = _database()
    service = PresetService(PresetRepository(factory))
    generation = service.create_from_current_settings("apply", _settings("current"))
    duplicate = Preset.create(
        PresetKind.LORA,
        "duplicate",
        LoraPresetPayload(
            loras=(
                LoraSetting(name="same.safetensors", order=0),
                LoraSetting(name="same.safetensors", order=1),
            )
        ),
    )
    too_many = Preset.create(
        PresetKind.LORA,
        "too-many",
        LoraPresetPayload(
            loras=(
                LoraSetting(name="one.safetensors", order=0),
                LoraSetting(name="two.safetensors", order=1),
            )
        ),
    )
    service._repository.create(duplicate)  # type: ignore[attr-defined]
    service._repository.create(too_many)  # type: ignore[attr-defined]

    def call(
        handler: object,
        selected: str | None,
        *,
        prompt_mode: str = "replace",
        lora_mode: str = "replace",
    ) -> tuple[object, ...]:
        return handler(  # type: ignore[operator]
            selected,
            prompt_mode,
            lora_mode,
            "current",
            "negative",
            1024,
            1024,
            "Random",
            -1,
            28,
            5.5,
            "euler",
            "normal",
            "base.safetensors",
            None,
            [],
            ("base.safetensors",),
            (),
            (),
        )

    for max_loras in (1, 2, 8, 12):
        handler = make_preset_apply_handler(service, max_loras)
        expected = 17 + 7 * max_loras
        assert preset_apply_output_count(max_loras) == expected
        assert len(call(handler, str(generation.id))) == expected
        assert len(call(handler, None)) == expected
        assert len(call(handler, "not-a-uuid")) == expected
        assert len(call(handler, str(uuid4()))) == expected
        assert len(call(handler, str(generation.id), prompt_mode="invalid")) == expected
        assert len(call(handler, str(generation.id), lora_mode="invalid")) == expected
        assert len(call(handler, str(duplicate.id))) == expected
        assert len(call(handler, str(too_many.id))) == expected

    class BrokenRepository:
        def get_by_id(self, preset_id: object) -> None:
            raise PresetRepositoryError("database unavailable")

    broken_service = PresetService(BrokenRepository())  # type: ignore[arg-type]
    broken_handler = make_preset_apply_handler(broken_service, 1)
    assert len(call(broken_handler, str(generation.id))) == 24
    engine.dispose()


def test_recent_model_shortcuts_require_capability_and_preserve_other_state() -> None:
    checkpoint_handler = make_recent_checkpoint_handler()
    updated_checkpoint, message = checkpoint_handler("new.safetensors", ("new.safetensors",))
    assert getattr(updated_checkpoint, "value", None) == "new.safetensors"
    assert "反映" in message
    preserved, warning = checkpoint_handler("missing.safetensors", ("new.safetensors",))
    assert isinstance(preserved, dict)
    assert "利用できません" in warning

    vae_handler = make_recent_vae_handler()
    updated_vae, _ = vae_handler("vae.safetensors", ("vae.safetensors",))
    assert getattr(updated_vae, "value", None) == "vae.safetensors"
    embedded, _ = vae_handler(None, None)
    assert getattr(embedded, "value", "missing") is None
    missing_vae, warning = vae_handler("missing.vae", ("vae.safetensors",))
    assert isinstance(missing_vae, dict)
    assert "利用できません" in warning


def test_recent_lora_shortcut_appends_and_rejects_duplicate_limit_and_missing() -> None:
    handler = make_recent_lora_add_handler(2)
    choices = ("one.safetensors", "two.safetensors")
    empty = [
        {
            "row_id": "empty",
            "lora_name": None,
            "model_strength": 1.0,
            "clip_strength": 1.0,
        }
    ]
    added = handler("one.safetensors", empty, choices)
    assert added[-1] == "最近使ったLoRAを末尾へ追加しました。"
    assert added[0][0]["lora_name"] == "one.safetensors"
    one = added[0]
    duplicate = handler("one.safetensors", one, choices)
    assert "重複" in duplicate[-1]
    full = [
        {
            "row_id": "one",
            "lora_name": "one.safetensors",
            "model_strength": 1.0,
            "clip_strength": 1.0,
        },
        {
            "row_id": "two",
            "lora_name": "two.safetensors",
            "model_strength": 0.8,
            "clip_strength": 0.7,
        },
    ]
    missing = handler("missing.safetensors", full, choices)
    assert "利用できません" in missing[-1]
    over_limit = handler("three.safetensors", full, choices + ("three.safetensors",))
    assert "上限" in over_limit[-1]
    duplicate_again = handler("one.safetensors", full, choices)
    assert "重複" in duplicate_again[-1]
    assert len(over_limit) == 3 + 7 * 2
