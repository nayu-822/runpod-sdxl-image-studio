"""Phase 7 Alembic boundary checks."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

ROOT = Path(__file__).parents[2]


def _config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_phase7_migration_adds_only_sync_tables_and_downgrades_cleanly(tmp_path: Path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'phase7.sqlite3').as_posix()}"
    command.upgrade(_config(database_url), "0012_phase6_legacy_metadata_candidates")
    engine = create_engine(database_url)
    before = inspect(engine)
    assert before.has_table("generations")
    assert before.has_table("metadata_imports")
    assert not before.has_table("drive_sync_records")
    assert not before.has_table("drive_sync_jobs")

    command.upgrade(_config(database_url), "0013_phase7_drive_sync")
    upgraded = inspect(engine)
    assert upgraded.has_table("drive_sync_records")
    assert upgraded.has_table("drive_sync_jobs")
    assert {column["name"] for column in upgraded.get_columns("drive_sync_records")} >= {
        "generation_id",
        "status",
        "remote_image_path",
        "remote_metadata_path",
        "image_sha256",
        "metadata_sha256",
    }
    assert {column["name"] for column in upgraded.get_columns("drive_sync_jobs")} >= {
        "sync_record_id",
        "queue_sequence",
        "progress_bytes",
        "total_bytes",
        "worker_id",
        "lease_expires_at",
    }

    command.downgrade(_config(database_url), "0012_phase6_legacy_metadata_candidates")
    downgraded = inspect(engine)
    assert not downgraded.has_table("drive_sync_records")
    assert not downgraded.has_table("drive_sync_jobs")
    assert downgraded.has_table("generations")
    assert downgraded.has_table("metadata_imports")
    engine.dispose()
