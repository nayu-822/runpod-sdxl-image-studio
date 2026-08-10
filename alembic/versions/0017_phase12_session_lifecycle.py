"""Add Phase 12 form state and pod lifecycle session tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0017_phase12_session_lifecycle"
down_revision = "0016_phase11_model_transfer_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generation_form_state",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("snapshot_json", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "pod_lifecycle_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("pod_id", sa.String(length=200), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "auto_terminate_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("auto_terminate_armed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="idle"),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_code", sa.String(length=100), nullable=True),
        sa.Column("last_error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pod_id", name="uq_pod_lifecycle_session_pod"),
    )
    op.create_index(
        "ix_pod_lifecycle_sessions_status_updated",
        "pod_lifecycle_sessions",
        ["status", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_pod_lifecycle_sessions_status_updated",
        table_name="pod_lifecycle_sessions",
    )
    op.drop_table("pod_lifecycle_sessions")
    op.drop_table("generation_form_state")
