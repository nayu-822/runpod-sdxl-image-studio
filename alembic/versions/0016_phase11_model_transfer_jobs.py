"""Add durable Google Drive model preparation jobs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016_phase11_model_transfer_jobs"
down_revision = "0015_phase9_system_error_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_transfer_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("remote_relative_path", sa.String(length=1000), nullable=False),
        sa.Column("local_relative_path", sa.String(length=1000), nullable=False),
        sa.Column("remote_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("remote_hash_algorithm", sa.String(length=32), nullable=True),
        sa.Column("remote_hash", sa.String(length=128), nullable=True),
        sa.Column("remote_modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("remote_identity", sa.String(length=500), nullable=False),
        sa.Column("local_sha256", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("progress_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("progress_percentage", sa.Float(), nullable=False, server_default="0"),
        sa.Column("worker_id", sa.String(length=200), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "kind IN ('checkpoint', 'lora', 'vae', 'upscaler')",
            name="ck_model_transfer_job_kind",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'downloading', 'completed', 'failed', "
            "'cancel_requested', 'cancelled')",
            name="ck_model_transfer_job_status",
        ),
    )
    op.create_index(
        "ix_model_transfer_jobs_status_created",
        "model_transfer_jobs",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_model_transfer_jobs_lease",
        "model_transfer_jobs",
        ["lease_expires_at"],
    )
    op.create_index(
        "ix_model_transfer_jobs_remote",
        "model_transfer_jobs",
        ["kind", "remote_relative_path"],
    )
    op.create_index(
        "uq_model_transfer_active_remote",
        "model_transfer_jobs",
        ["kind", "remote_relative_path", "remote_identity"],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'downloading', 'cancel_requested')"),
    )


def downgrade() -> None:
    op.drop_index("uq_model_transfer_active_remote", table_name="model_transfer_jobs")
    op.drop_index("ix_model_transfer_jobs_remote", table_name="model_transfer_jobs")
    op.drop_index("ix_model_transfer_jobs_lease", table_name="model_transfer_jobs")
    op.drop_index("ix_model_transfer_jobs_status_created", table_name="model_transfer_jobs")
    op.drop_table("model_transfer_jobs")
