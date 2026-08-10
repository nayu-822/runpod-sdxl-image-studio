from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def _config(database_path: Path) -> Config:
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    return config


def test_phase12_upgrade_and_downgrade_preserve_previous_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "phase12.sqlite3"
    config = _config(database_path)
    command.upgrade(config, "0016_phase11_model_transfer_jobs")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_transfer_jobs "
                "(id, kind, remote_relative_path, local_relative_path, remote_size_bytes, "
                "remote_identity, status, progress_bytes, total_bytes, progress_percentage, "
                "retryable, created_at, updated_at) VALUES "
                "('job', 'checkpoint', 'model.safetensors', 'model.safetensors', 1, 'id', "
                "'completed', 1, 1, 100, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )

    command.upgrade(config, "head")
    assert {"generation_form_state", "pod_lifecycle_sessions", "model_transfer_jobs"}.issubset(
        set(inspect(engine).get_table_names())
    )

    command.downgrade(config, "0016_phase11_model_transfer_jobs")
    tables = set(inspect(engine).get_table_names())
    assert "generation_form_state" not in tables
    assert "pod_lifecycle_sessions" not in tables
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM model_transfer_jobs WHERE id = 'job'")
            ).scalar_one()
            == 1
        )

    command.upgrade(config, "head")
