"""Phase A migration preserves ordered artifacts and reverses its own schema."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

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
    assert "artifacts_json" in {
        column["name"] for column in after.get_columns("drive_sync_records")
    }
    assert "artifacts_json" in {column["name"] for column in after.get_columns("drive_sync_jobs")}
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

    command.downgrade(config, "0018_phase_a_interactive_runs_and_artifact_order")
    phase_a = inspect(engine)
    assert phase_a.has_table("interactive_generation_runs")
    assert "artifacts_json" not in {
        column["name"] for column in phase_a.get_columns("drive_sync_records")
    }
    assert "artifacts_json" not in {
        column["name"] for column in phase_a.get_columns("drive_sync_jobs")
    }

    command.downgrade(config, "0017_phase12_session_lifecycle")
    restored = inspect(engine)
    assert not restored.has_table("interactive_generation_runs")
    assert "display_order" not in {
        column["name"] for column in restored.get_columns("generation_artifacts")
    }
    assert restored.has_table("generation_form_state")
    assert restored.has_table("pod_lifecycle_sessions")
    engine.dispose()


def test_phase_a_multi_image_migration_preserves_legacy_drive_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "phase-a-legacy-drive.sqlite3"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "0018_phase_a_interactive_runs_and_artifact_order")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    generation_id = uuid4()
    job_id = uuid4()
    artifact_id = uuid4()
    record_id = uuid4()
    sync_job_id = uuid4()
    timestamp = datetime(2026, 8, 14, 12, tzinfo=UTC)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO generations "
                "(id, kind, status, settings_snapshot_json, snapshot_schema_version, "
                "workflow_template_id, workflow_template_version, created_at, updated_at) "
                "VALUES (:id, 'txt2img', 'completed', '{}', 1, 'phase-a-test', '1', "
                ":created_at, :updated_at)"
            ),
            {"id": str(generation_id), "created_at": timestamp, "updated_at": timestamp},
        )
        connection.execute(
            text(
                "INSERT INTO generation_jobs "
                "(id, generation_id, status, created_at, updated_at) "
                "VALUES (:id, :generation_id, 'completed', :created_at, :updated_at)"
            ),
            {
                "id": str(job_id),
                "generation_id": str(generation_id),
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO generation_artifacts "
                "(id, generation_id, artifact_type, local_path, sha256, size_bytes, "
                "mime_type, created_at, display_order) VALUES "
                "(:id, :generation_id, 'image', 'generations/legacy.png', :sha256, "
                "3, 'image/png', :created_at, 0)"
            ),
            {
                "id": str(artifact_id),
                "generation_id": str(generation_id),
                "sha256": "a" * 64,
                "created_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO drive_sync_records "
                "(id, generation_id, status, remote_name, remote_base_path, "
                "remote_image_path, remote_metadata_path, image_artifact_id, image_sha256, "
                "image_size_bytes, created_at, updated_at) VALUES "
                "(:id, :generation_id, 'failed', 'gdrive', 'RunPod/Images', "
                "'2026-08-14/generated/legacy.png', '2026-08-14/generated/legacy.json', "
                ":artifact_id, :sha256, 3, :created_at, :updated_at)"
            ),
            {
                "id": str(record_id),
                "generation_id": str(generation_id),
                "artifact_id": str(artifact_id),
                "sha256": "a" * 64,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO drive_sync_jobs "
                "(id, sync_record_id, generation_id, queue_sequence, status, "
                "image_artifact_id, image_sha256, image_size_bytes, created_at, updated_at) "
                "VALUES (:id, :record_id, :generation_id, 1, 'failed', :artifact_id, "
                ":sha256, 3, :created_at, :updated_at)"
            ),
            {
                "id": str(sync_job_id),
                "record_id": str(record_id),
                "generation_id": str(generation_id),
                "artifact_id": str(artifact_id),
                "sha256": "a" * 64,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )

    def legacy_values() -> tuple[tuple[object, ...], tuple[object, ...]]:
        with engine.connect() as connection:
            record = connection.execute(
                text(
                    "SELECT id, generation_id, status, remote_image_path, "
                    "remote_metadata_path, image_artifact_id, image_sha256, image_size_bytes "
                    "FROM drive_sync_records"
                )
            ).all()
            job = connection.execute(
                text(
                    "SELECT id, sync_record_id, generation_id, status, image_artifact_id, "
                    "image_sha256, image_size_bytes FROM drive_sync_jobs"
                )
            ).all()
        return tuple(record), tuple(job)

    before = legacy_values()
    command.upgrade(config, "head")
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT artifacts_json FROM drive_sync_records")).scalar_one()
            is None
        )
        assert (
            connection.execute(text("SELECT artifacts_json FROM drive_sync_jobs")).scalar_one()
            is None
        )
    assert legacy_values() == before

    command.downgrade(config, "-1")
    assert legacy_values() == before
    engine.dispose()
