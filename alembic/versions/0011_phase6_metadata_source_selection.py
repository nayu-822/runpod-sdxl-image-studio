"""Persist explicit PNG/sidecar candidate selection for Phase 6 imports."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_phase6_metadata_source_selection"
down_revision = "0010_phase6_metadata_imports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "metadata_imports",
        sa.Column("candidate_options_json", sa.Text(), nullable=True),
    )
    op.add_column(
        "metadata_imports",
        sa.Column("selected_metadata_source", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "metadata_imports",
        sa.Column("sidecar_hash_confirmed", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.execute(
        sa.text(
            "UPDATE metadata_imports SET candidate_options_json = "
            "CASE WHEN candidate_json IS NULL THEN '[]' "
            "ELSE '[' || candidate_json || ']' END"
        )
    )
    op.execute(
        sa.text(
            "UPDATE metadata_imports SET selected_metadata_source = "
            "CASE WHEN candidate_json IS NULL "
            "OR metadata_source NOT IN ('comfyui_prompt', 'app_sidecar') "
            "OR warnings_json LIKE '%metadata_import_ambiguous%' THEN NULL "
            "ELSE metadata_source END"
        )
    )
    with op.batch_alter_table("metadata_imports", recreate="always") as batch:
        batch.alter_column(
            "candidate_options_json",
            existing_type=sa.Text(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("metadata_imports", recreate="always") as batch:
        batch.drop_column("sidecar_hash_confirmed")
        batch.drop_column("selected_metadata_source")
        batch.drop_column("candidate_options_json")
