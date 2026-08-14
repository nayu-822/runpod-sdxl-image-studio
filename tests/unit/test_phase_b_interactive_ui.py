from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.generation_custom_size_repository import (  # noqa: E501
    GenerationCustomSizeRepository,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.services.generation_custom_size_service import (
    GenerationCustomSizeError,
    GenerationCustomSizeService,
)
from runpod_sdxl_image_studio.ui.tabs.system_tab import interactive_action_updates

ROOT = Path(__file__).resolve().parents[2]


def test_custom_generation_sizes_are_idempotent_and_delete_only_the_preference() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    service = GenerationCustomSizeService(
        GenerationCustomSizeRepository(create_session_factory(engine)),
        Settings(_env_file=None, max_width=2048, max_height=2048, max_pixels=4_194_304),
    )

    first = service.add(768, 1024)
    duplicate = service.add(768, 1024)

    assert duplicate.id == first.id
    assert [(item.width, item.height) for item in service.list()] == [(768, 1024)]
    assert service.selector_options() == [("Custom 768 × 1024", f"custom:{first.id}")]

    service.delete(first.id)
    assert service.list() == ()
    engine.dispose()


def test_custom_generation_size_validation_does_not_register_invalid_values() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    service = GenerationCustomSizeService(
        GenerationCustomSizeRepository(create_session_factory(engine)),
        Settings(_env_file=None, max_width=2048, max_height=2048, max_pixels=4_194_304),
    )

    for dimensions in ((770, 1024), (0, 1024), (2048, 2112), (2048, 2048)):
        if dimensions == (2048, 2048):
            continue
        try:
            service.add(*dimensions)
        except GenerationCustomSizeError:
            pass
        else:
            raise AssertionError(f"invalid dimensions were accepted: {dimensions}")
    assert service.list() == ()
    engine.dispose()


def test_phase_b_custom_size_migration_round_trip(tmp_path: Path) -> None:
    database = tmp_path / "phase-b.sqlite3"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")

    command.upgrade(config, "0019_phase_a_multi_image_drive_sync")
    command.upgrade(config, "0020_phase_b_custom_generation_sizes")
    upgraded = inspect(create_engine(f"sqlite:///{database.as_posix()}"))
    assert upgraded.has_table("generation_custom_sizes")
    assert {column["name"] for column in upgraded.get_columns("generation_custom_sizes")} == {
        "id",
        "width",
        "height",
        "created_at",
    }

    command.downgrade(config, "0019_phase_a_multi_image_drive_sync")
    downgraded = inspect(create_engine(f"sqlite:///{database.as_posix()}"))
    assert not downgraded.has_table("generation_custom_sizes")


def test_phase_b_action_buttons_follow_durable_run_status() -> None:
    idle_generate, idle_cancel = interactive_action_updates("対話的生成: 待機中")
    active_generate, active_cancel = interactive_action_updates(
        "Interactive run: active; Batch 1 / 2; current Generation=running"
    )
    cancelling_generate, cancelling_cancel = interactive_action_updates(
        "Interactive run: cancelling; Batch 1 / 2"
    )
    completed_generate, completed_cancel = interactive_action_updates(
        "Interactive run: completed; Batch 2 / 2"
    )

    assert idle_generate.interactive is True
    assert idle_cancel.interactive is False
    assert active_generate.interactive is False
    assert active_cancel.interactive is True
    assert cancelling_generate.interactive is False
    assert cancelling_cancel.interactive is False
    assert completed_generate.interactive is True
    assert completed_cancel.interactive is False


def test_custom_size_migration_has_expected_revision_chain() -> None:
    migration = (
        ROOT / "alembic" / "versions" / "0020_phase_b_custom_generation_sizes.py"
    ).read_text(encoding="utf-8")
    assert 'revision = "0020_phase_b_custom_generation_sizes"' in migration
    assert 'down_revision = "0019_phase_a_multi_image_drive_sync"' in migration
    assert "UniqueConstraint" in migration
