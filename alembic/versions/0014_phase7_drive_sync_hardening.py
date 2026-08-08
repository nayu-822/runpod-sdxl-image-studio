"""Add durable destination-scoped manifest rebuild jobs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014_phase7_drive_sync_hardening"
down_revision = "0013_phase7_drive_sync"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drive_manifest_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("local_date", sa.String(length=10), nullable=False),
        sa.Column("remote_name", sa.String(length=128), nullable=False),
        sa.Column("remote_base_path", sa.String(length=1000), nullable=False),
        sa.Column("remote_manifest_path", sa.String(length=1000), nullable=False),
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
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("queue_sequence", name="uq_drive_manifest_job_sequence"),
        sa.CheckConstraint(
            "status IN ('pending', 'syncing', 'synced', 'failed')",
            name="ck_drive_manifest_job_status",
        ),
    )
    op.create_index(
        "ix_drive_manifest_jobs_status_sequence",
        "drive_manifest_jobs",
        ["status", "queue_sequence"],
    )
    op.create_index("ix_drive_manifest_jobs_lease", "drive_manifest_jobs", ["lease_expires_at"])
    op.create_index(
        "uq_drive_manifest_active_destination",
        "drive_manifest_jobs",
        ["local_date", "remote_name", "remote_base_path"],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'syncing')"),
    )


def downgrade() -> None:
    op.drop_index("uq_drive_manifest_active_destination", table_name="drive_manifest_jobs")
    op.drop_index("ix_drive_manifest_jobs_lease", table_name="drive_manifest_jobs")
    op.drop_index("ix_drive_manifest_jobs_status_sequence", table_name="drive_manifest_jobs")
    op.drop_table("drive_manifest_jobs")
