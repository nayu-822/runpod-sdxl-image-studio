"""Phase A migration preserves ordered artifacts and reverses its own schema."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parents[2]


def test_phase_a_migration_round_trip_adds_and_removes_phase_a_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "phase-a-migration.sqlite3"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "0017_phase12_session_lifecycle")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    before = inspect(engine)
    assert not before.has_table("interactive_generation_runs")
    assert "display_order" not in {
        column["name"] for column in before.get_columns("generation_artifacts")
    }

    command.upgrade(config, "head")
    after = inspect(engine)
    assert after.has_table("interactive_generation_runs")
    assert "display_order" in {
        column["name"] for column in after.get_columns("generation_artifacts")
    }
    with engine.connect() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'index' AND name = 'uq_interactive_run_active'"
                )
            ).scalar_one()
            == "uq_interactive_run_active"
        )
    assert any(
        constraint["name"] == "uq_generation_artifact_order"
        for constraint in after.get_unique_constraints("generation_artifacts")
    )

    command.downgrade(config, "0017_phase12_session_lifecycle")
    restored = inspect(engine)
    assert not restored.has_table("interactive_generation_runs")
    assert "display_order" not in {
        column["name"] for column in restored.get_columns("generation_artifacts")
    }
    assert restored.has_table("generation_form_state")
    assert restored.has_table("pod_lifecycle_sessions")
    engine.dispose()
