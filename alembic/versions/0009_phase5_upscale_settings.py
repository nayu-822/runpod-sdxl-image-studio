"""Persist Phase 5 upscale request settings."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_phase5_upscale_settings"
down_revision = "0008_phase4_recovery_correction"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generation_upscale_settings",
        sa.Column("generation_id", sa.String(length=36), nullable=False),
        sa.Column("source_artifact_id", sa.String(length=36), nullable=False),
        sa.Column("method", sa.String(length=16), nullable=False),
        sa.Column("sizing_mode", sa.String(length=16), nullable=False),
        sa.Column("scale_factor", sa.Float(), nullable=True),
        sa.Column("target_width", sa.Integer(), nullable=False),
        sa.Column("target_height", sa.Integer(), nullable=False),
        sa.Column("upscaler_name", sa.String(length=500), nullable=True),
        sa.Column("denoise", sa.Float(), nullable=True),
        sa.Column("settings_snapshot_json", sa.Text(), nullable=False),
        sa.Column("snapshot_schema_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_artifact_id"], ["generation_artifacts.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("generation_id"),
        sa.CheckConstraint("method IN ('image', 'latent')", name="ck_upscale_method"),
        sa.CheckConstraint(
            "sizing_mode IN ('factor', 'dimensions')", name="ck_upscale_sizing_mode"
        ),
        sa.CheckConstraint(
            "(sizing_mode='factor' AND scale_factor IS NOT NULL AND scale_factor > 1) OR "
            "(sizing_mode='dimensions' AND scale_factor IS NULL)",
            name="ck_upscale_sizing_values",
        ),
        sa.CheckConstraint(
            "target_width > 0 AND target_height > 0", name="ck_upscale_target_positive"
        ),
        sa.CheckConstraint(
            "denoise IS NULL OR (denoise >= 0 AND denoise <= 1)", name="ck_upscale_denoise"
        ),
    )
    op.create_index(
        "ix_generation_upscale_source_artifact",
        "generation_upscale_settings",
        ["source_artifact_id"],
    )
    op.create_index("ix_generation_upscale_method", "generation_upscale_settings", ["method"])


def downgrade() -> None:
    op.drop_index("ix_generation_upscale_method", table_name="generation_upscale_settings")
    op.drop_index("ix_generation_upscale_source_artifact", table_name="generation_upscale_settings")
    op.drop_table("generation_upscale_settings")
