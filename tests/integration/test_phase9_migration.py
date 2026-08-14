from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from runpod_sdxl_image_studio.adapters.database.models import (
    DriveSyncJobModel,
    DriveSyncRecordModel,
    GenerationArtifactModel,
    GenerationJobModel,
    GenerationModel,
)


def test_phase9_migration_is_reversible_without_changing_existing_rows(tmp_path: Path) -> None:
    """Exercise 0015 through Alembic against a real SQLite migration chain."""

    database_path = tmp_path / "phase9-migration.sqlite3"
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "0014_phase7_drive_sync_hardening")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    generation_id = uuid4()
    job_id = uuid4()
    artifact_id = uuid4()
    drive_record_id = uuid4()
    drive_job_id = uuid4()
    timestamp = datetime(2026, 8, 9, 12, tzinfo=UTC)

    with Session(engine) as session:
        generation = GenerationModel(
            id=str(generation_id),
            kind="txt2img",
            status="completed",
            settings_snapshot_json="{}",
            snapshot_schema_version=1,
            workflow_template_id="phase9-test",
            workflow_template_version="1",
            created_at=timestamp,
            updated_at=timestamp,
        )
        job = GenerationJobModel(
            id=str(job_id),
            generation_id=str(generation_id),
            status="completed",
            created_at=timestamp,
            updated_at=timestamp,
        )
        session.add_all((generation, job))
        session.flush()
        session.execute(
            text(
                "INSERT INTO generation_artifacts "
                "(id, generation_id, artifact_type, local_path, sha256, size_bytes, "
                "mime_type, created_at) VALUES "
                "(:id, :generation_id, 'image', 'images/existing.png', :sha256, "
                "3, 'image/png', :created_at)"
            ),
            {
                "id": str(artifact_id),
                "generation_id": str(generation_id),
                "sha256": "a" * 64,
                "created_at": timestamp,
            },
        )
        session.add(
            DriveSyncRecordModel(
                id=str(drive_record_id),
                generation_id=str(generation_id),
                status="failed",
                remote_name="gdrive",
                remote_base_path="RunPod/Images",
                remote_image_path="RunPod/Images/existing.png",
                remote_metadata_path="RunPod/Images/existing.json",
                image_artifact_id=str(artifact_id),
                image_sha256="a" * 64,
                image_size_bytes=3,
                error_code="drive_copy_failed",
                error_summary="test failure",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        session.flush()
        session.add(
            DriveSyncJobModel(
                id=str(drive_job_id),
                sync_record_id=str(drive_record_id),
                generation_id=str(generation_id),
                queue_sequence=1,
                status="failed",
                error_code="drive_copy_failed",
                error_summary="test failure",
                retryable=True,
                image_artifact_id=str(artifact_id),
                image_sha256="a" * 64,
                image_size_bytes=3,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        session.commit()

    def snapshot() -> dict[str, tuple[tuple[object, ...], ...]]:
        with engine.connect() as connection:
            return {
                "generations": tuple(
                    connection.execute(
                        select(
                            GenerationModel.id,
                            GenerationModel.status,
                            GenerationModel.updated_at,
                        )
                    ).all()
                ),
                "jobs": tuple(
                    connection.execute(
                        select(
                            GenerationJobModel.id,
                            GenerationJobModel.status,
                            GenerationJobModel.updated_at,
                        )
                    ).all()
                ),
                "artifacts": tuple(
                    connection.execute(
                        select(GenerationArtifactModel.id, GenerationArtifactModel.local_path)
                    ).all()
                ),
                "drive_records": tuple(
                    connection.execute(
                        select(DriveSyncRecordModel.id, DriveSyncRecordModel.status)
                    ).all()
                ),
                "drive_jobs": tuple(
                    connection.execute(select(DriveSyncJobModel.id, DriveSyncJobModel.status)).all()
                ),
            }

    before = snapshot()
    command.upgrade(config, "0015_phase9_system_error_events")
    assert inspect(engine).has_table("system_error_events")
    assert snapshot() == before

    command.downgrade(config, "-1")
    assert not inspect(engine).has_table("system_error_events")
    assert snapshot() == before

    command.upgrade(config, "0015_phase9_system_error_events")
    assert inspect(engine).has_table("system_error_events")
    assert snapshot() == before
    engine.dispose()
