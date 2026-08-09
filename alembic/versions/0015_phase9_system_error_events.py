"""Add sanitized append-only system error history."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015_phase9_system_error_events"
down_revision = "0014_phase7_drive_sync_hardening"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_error_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("generation_id", sa.String(length=36), nullable=True),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("details", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["job_id"], ["generation_jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name="ck_system_error_event_severity",
        ),
    )
    op.create_index(
        "ix_system_error_events_created_at",
        "system_error_events",
        ["created_at"],
    )
    op.create_index(
        "ix_system_error_events_category",
        "system_error_events",
        ["category"],
    )
    op.create_index(
        "ix_system_error_events_generation",
        "system_error_events",
        ["generation_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_system_error_events_generation", table_name="system_error_events")
    op.drop_index("ix_system_error_events_category", table_name="system_error_events")
    op.drop_index("ix_system_error_events_created_at", table_name="system_error_events")
    op.drop_table("system_error_events")
