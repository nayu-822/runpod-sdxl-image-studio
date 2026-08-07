"""Phase 6 migration and SQLite persistence integration checks."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def _alembic_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("script_location", str(Path("alembic").resolve()))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.resolve().as_posix()}")
    return config


def test_phase6_migration_roundtrip_and_safe_external_downgrade(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.sqlite3"
    config = _alembic_config(database_path)

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    assert "metadata_imports" in inspect(engine).get_table_names()
    assert "source_import_id" in {
        column["name"] for column in inspect(engine).get_columns("generation_upscale_settings")
    }

    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO generation_upscale_settings "
                "(generation_id, source_kind, source_artifact_id, source_import_id, method, "
                "sizing_mode, scale_factor, target_width, target_height, upscaler_name, denoise, "
                "settings_snapshot_json, snapshot_schema_version, created_at, updated_at) "
                "VALUES (:generation_id, 'metadata_import', NULL, :source_import_id, 'image', "
                "'factor', 2, 1024, 1024, '4x.pth', NULL, '{}', 2, :created_at, :updated_at)"
            ),
            {
                "generation_id": "external-generation",
                "source_import_id": "external-import",
                "created_at": now,
                "updated_at": now,
            },
        )

    with pytest.raises(RuntimeError, match="metadata_import upscale sources"):
        command.downgrade(config, "-1")

    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM generation_upscale_settings "
                "WHERE generation_id = 'external-generation'"
            )
        )
    command.downgrade(config, "-1")
    assert "metadata_imports" not in inspect(engine).get_table_names()
    command.upgrade(config, "head")
    assert "metadata_imports" in inspect(engine).get_table_names()
