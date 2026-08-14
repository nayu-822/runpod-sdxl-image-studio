"""Add persisted custom generation-size preferences."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0020_phase_b_custom_generation_sizes"
down_revision = "0019_phase_a_multi_image_drive_sync"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generation_custom_sizes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "width > 0 AND height > 0",
            name="ck_custom_generation_size_positive",
        ),
        sa.CheckConstraint(
            "width % 64 = 0 AND height % 64 = 0",
            name="ck_custom_generation_size_multiple_of_64",
        ),
        sa.CheckConstraint("width * height > 0", name="ck_custom_generation_size_pixels"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "width",
            "height",
            name="uq_custom_generation_size_dimensions",
        ),
    )


def downgrade() -> None:
    op.drop_table("generation_custom_sizes")
