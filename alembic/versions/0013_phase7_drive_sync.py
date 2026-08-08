"""Add independent Google Drive synchronization records and jobs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0013_phase7_drive_sync"
down_revision = "0012_phase6_legacy_metadata_candidates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drive_sync_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("generation_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("remote_name", sa.String(length=128), nullable=False),
        sa.Column("remote_base_path", sa.String(length=1000), nullable=False),
        sa.Column("remote_image_path", sa.String(length=1000), nullable=False),
        sa.Column("remote_metadata_path", sa.String(length=1000), nullable=False),
        sa.Column("image_artifact_id", sa.String(length=36), nullable=False),
        sa.Column("metadata_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata_sha256", sa.String(length=64), nullable=True),
        sa.Column("image_size_bytes", sa.Integer(), nullable=False),
        sa.Column("metadata_size_bytes", sa.Integer(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["image_artifact_id"], ["generation_artifacts.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["metadata_artifact_id"], ["generation_artifacts.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("generation_id", name="uq_drive_sync_record_generation"),
        sa.CheckConstraint(
            "status IN ('pending', 'syncing', 'synced', 'failed')",
            name="ck_drive_sync_record_status",
        ),
    )
    op.create_index("ix_drive_sync_records_status", "drive_sync_records", ["status"])
    op.create_index("ix_drive_sync_records_generation", "drive_sync_records", ["generation_id"])

    op.create_table(
        "drive_sync_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("sync_record_id", sa.String(length=36), nullable=False),
        sa.Column("generation_id", sa.String(length=36), nullable=False),
        sa.Column("queue_sequence", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("progress_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress_percentage", sa.Float(), nullable=False, server_default="0"),
        sa.Column("current_artifact", sa.String(length=32), nullable=True),
        sa.Column("worker_id", sa.String(length=200), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("log_path", sa.String(length=1000), nullable=True),
        sa.Column("image_artifact_id", sa.String(length=36), nullable=False),
        sa.Column("metadata_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata_sha256", sa.String(length=64), nullable=True),
        sa.Column("image_size_bytes", sa.Integer(), nullable=False),
        sa.Column("metadata_size_bytes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["sync_record_id"], ["drive_sync_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("queue_sequence", name="uq_drive_sync_job_sequence"),
        sa.CheckConstraint(
            "status IN ('pending', 'syncing', 'synced', 'failed')",
            name="ck_drive_sync_job_status",
        ),
    )
    op.create_index(
        "ix_drive_sync_jobs_status_sequence",
        "drive_sync_jobs",
        ["status", "queue_sequence"],
    )
    op.create_index("ix_drive_sync_jobs_record", "drive_sync_jobs", ["sync_record_id"])
    op.create_index("ix_drive_sync_jobs_lease", "drive_sync_jobs", ["lease_expires_at"])
    op.create_index(
        "uq_drive_sync_active_record",
        "drive_sync_jobs",
        ["sync_record_id"],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'syncing')"),
    )


def downgrade() -> None:
    op.drop_index("uq_drive_sync_active_record", table_name="drive_sync_jobs")
    op.drop_index("ix_drive_sync_jobs_lease", table_name="drive_sync_jobs")
    op.drop_index("ix_drive_sync_jobs_record", table_name="drive_sync_jobs")
    op.drop_index("ix_drive_sync_jobs_status_sequence", table_name="drive_sync_jobs")
    op.drop_table("drive_sync_jobs")
    op.drop_index("ix_drive_sync_records_generation", table_name="drive_sync_records")
    op.drop_index("ix_drive_sync_records_status", table_name="drive_sync_records")
    op.drop_table("drive_sync_records")
