"""Phase 11 migration preserves existing state and reverses only its table."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from runpod_sdxl_image_studio.adapters.database.models import GenerationModel


def test_phase11_migration_round_trip_preserves_existing_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "phase11-migration.sqlite3"
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(config, "0015_phase9_system_error_events")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    generation_id = str(uuid4())
    timestamp = datetime.now(UTC)
    with Session(engine) as session:
        session.add(
            GenerationModel(
                id=generation_id,
                kind="txt2img",
                status="completed",
                settings_snapshot_json="{}",
                snapshot_schema_version=1,
                workflow_template_id="phase11",
                workflow_template_version="1",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        session.commit()

    command.upgrade(config, "head")
    assert inspect(engine).has_table("model_transfer_jobs")
    with Session(engine) as session:
        assert session.scalar(select(GenerationModel.id).where(GenerationModel.id == generation_id))

    command.downgrade(config, "-1")
    assert inspect(engine).has_table("model_transfer_jobs")
    assert not inspect(engine).has_table("generation_form_state")
    assert not inspect(engine).has_table("pod_lifecycle_sessions")
    assert inspect(engine).has_table("generations")
    with Session(engine) as session:
        assert session.scalar(select(GenerationModel.id).where(GenerationModel.id == generation_id))

    command.upgrade(config, "head")
    assert inspect(engine).has_table("model_transfer_jobs")
    assert inspect(engine).has_table("pod_lifecycle_sessions")
    engine.dispose()
