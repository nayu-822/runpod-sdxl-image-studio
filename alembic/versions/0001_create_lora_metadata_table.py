"""create lora metadata table

Revision ID: 0001_lora_metadata
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_lora_metadata"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lora_metadata",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("file_name", sa.String(length=500), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=True),
        sa.Column("category", sa.String(length=50), nullable=True),
        sa.Column("is_favorite", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("trigger_words_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("recommended_model_strength", sa.Float(), nullable=True),
        sa.Column("recommended_clip_strength", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("compatible_models_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("thumbnail_path", sa.String(length=500), nullable=True),
        sa.Column("is_missing", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_name"),
    )
    op.create_index("ix_lora_metadata_category", "lora_metadata", ["category"])
    op.create_index("ix_lora_metadata_favorite", "lora_metadata", ["is_favorite"])
    op.create_index("ix_lora_metadata_missing", "lora_metadata", ["is_missing"])
    op.create_index("ix_lora_metadata_last_used", "lora_metadata", ["last_used_at"])


def downgrade() -> None:
    op.drop_index("ix_lora_metadata_last_used", table_name="lora_metadata")
    op.drop_index("ix_lora_metadata_missing", table_name="lora_metadata")
    op.drop_index("ix_lora_metadata_favorite", table_name="lora_metadata")
    op.drop_index("ix_lora_metadata_category", table_name="lora_metadata")
    op.drop_table("lora_metadata")
