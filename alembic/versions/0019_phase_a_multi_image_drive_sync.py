"""Persist the complete multi-image Drive transfer plan."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019_phase_a_multi_image_drive_sync"
down_revision = "0018_phase_a_interactive_runs_and_artifact_order"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable JSON plans while preserving all legacy scalar columns."""

    with op.batch_alter_table("drive_sync_records") as batch:
        batch.add_column(sa.Column("artifacts_json", sa.Text(), nullable=True))
    with op.batch_alter_table("drive_sync_jobs") as batch:
        batch.add_column(sa.Column("artifacts_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("drive_sync_jobs") as batch:
        batch.drop_column("artifacts_json")
    with op.batch_alter_table("drive_sync_records") as batch:
        batch.drop_column("artifacts_json")
