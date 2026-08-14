"""Add Phase A interactive runs and ordered image artifacts."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0018_phase_a_interactive_runs_and_artifact_order"
down_revision = "0017_phase12_session_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("generation_artifacts") as batch:
        batch.add_column(
            sa.Column("display_order", sa.Integer(), nullable=False, server_default="0")
        )
        batch.drop_constraint("uq_generation_artifact", type_="unique")
        batch.create_unique_constraint(
            "uq_generation_artifact_order",
            ["generation_id", "artifact_type", "display_order"],
        )

    op.create_table(
        "interactive_generation_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("batch_count", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("settings_snapshot_json", sa.Text(), nullable=False),
        sa.Column("snapshot_schema_version", sa.Integer(), nullable=False),
        sa.Column("client_local_date", sa.String(length=10), nullable=False),
        sa.Column("generation_ids_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("completed_generation_ids_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("current_generation_id", sa.String(length=36), nullable=True),
        sa.Column("last_completed_generation_id", sa.String(length=36), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'cancelling', 'completed', 'failed', 'cancelled')",
            name="ck_interactive_run_status",
        ),
        sa.CheckConstraint(
            "batch_count > 0 AND batch_count <= 100", name="ck_interactive_run_count"
        ),
        sa.CheckConstraint("batch_size > 0 AND batch_size <= 4", name="ck_interactive_run_size"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_interactive_run_active",
        "interactive_generation_runs",
        [sa.text("1")],
        unique=True,
        sqlite_where=sa.text("status IN ('active', 'cancelling')"),
    )
    op.create_index(
        "ix_interactive_run_updated",
        "interactive_generation_runs",
        ["updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_interactive_run_updated", table_name="interactive_generation_runs")
    op.drop_index("uq_interactive_run_active", table_name="interactive_generation_runs")
    op.drop_table("interactive_generation_runs")

    with op.batch_alter_table("generation_artifacts") as batch:
        batch.drop_constraint("uq_generation_artifact_order", type_="unique")
        batch.create_unique_constraint(
            "uq_generation_artifact",
            ["generation_id", "artifact_type", "sha256"],
        )
        batch.drop_column("display_order")
