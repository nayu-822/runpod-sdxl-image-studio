"""Add external metadata imports and external upscale source provenance."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_phase6_metadata_imports"
down_revision = "0009_phase5_upscale_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "metadata_imports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("original_filename", sa.String(length=500), nullable=False),
        sa.Column("stored_image_path", sa.String(length=1000), nullable=False),
        sa.Column("source_image_sha256", sa.String(length=64), nullable=False),
        sa.Column("stored_image_sha256", sa.String(length=64), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("image_mime_type", sa.String(length=100), nullable=False),
        sa.Column("metadata_source", sa.String(length=32), nullable=False),
        sa.Column("metadata_status", sa.String(length=32), nullable=False),
        sa.Column("raw_metadata_json", sa.Text(), nullable=False),
        sa.Column("raw_metadata_sha256", sa.String(length=64), nullable=False),
        sa.Column("candidate_json", sa.Text(), nullable=True),
        sa.Column("normalized_snapshot_json", sa.Text(), nullable=True),
        sa.Column("normalized_snapshot_schema_version", sa.Integer(), nullable=True),
        sa.Column("manual_mapping_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("warnings_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "metadata_status IN ('ready', 'needs_mapping', 'metadata_missing', 'invalid_metadata')",
            name="ck_metadata_import_status",
        ),
        sa.CheckConstraint(
            "metadata_source IN ('comfyui_prompt', 'app_sidecar', 'workflow', 'none')",
            name="ck_metadata_import_source",
        ),
    )
    op.create_index("ix_metadata_imports_created", "metadata_imports", ["created_at"])
    op.create_index("ix_metadata_imports_status", "metadata_imports", ["metadata_status"])
    op.create_index("ix_metadata_imports_source_hash", "metadata_imports", ["source_image_sha256"])

    with op.batch_alter_table("generation_upscale_settings", recreate="always") as batch:
        batch.alter_column(
            "source_artifact_id",
            existing_type=sa.String(length=36),
            nullable=True,
        )
        batch.add_column(
            sa.Column(
                "source_kind",
                sa.String(length=32),
                nullable=False,
                server_default="generation_artifact",
            )
        )
        batch.add_column(sa.Column("source_import_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key(
            "fk_generation_upscale_source_import",
            "metadata_imports",
            ["source_import_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_check_constraint(
            "ck_upscale_source_kind",
            "(source_kind='generation_artifact' AND source_artifact_id IS NOT NULL "
            "AND source_import_id IS NULL) OR "
            "(source_kind='metadata_import' AND source_artifact_id IS NULL "
            "AND source_import_id IS NOT NULL)",
        )
    op.create_index(
        "ix_generation_upscale_source_import",
        "generation_upscale_settings",
        ["source_import_id"],
    )
    op.execute(
        sa.text(
            "UPDATE generation_upscale_settings "
            "SET source_kind = 'generation_artifact' "
            "WHERE source_kind IS NULL"
        )
    )


def downgrade() -> None:
    connection = op.get_bind()
    external_count = connection.execute(
        sa.text(
            "SELECT COUNT(*) FROM generation_upscale_settings WHERE source_kind = 'metadata_import'"
        )
    ).scalar_one()
    if external_count:
        raise RuntimeError("cannot downgrade Phase 6 while metadata_import upscale sources exist")

    op.drop_index("ix_generation_upscale_source_import", table_name="generation_upscale_settings")
    with op.batch_alter_table("generation_upscale_settings", recreate="always") as batch:
        batch.drop_constraint("ck_upscale_source_kind", type_="check")
        batch.drop_constraint("fk_generation_upscale_source_import", type_="foreignkey")
        batch.drop_column("source_import_id")
        batch.drop_column("source_kind")
        batch.alter_column(
            "source_artifact_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
    op.drop_index("ix_metadata_imports_source_hash", table_name="metadata_imports")
    op.drop_index("ix_metadata_imports_status", table_name="metadata_imports")
    op.drop_index("ix_metadata_imports_created", table_name="metadata_imports")
    op.drop_table("metadata_imports")
