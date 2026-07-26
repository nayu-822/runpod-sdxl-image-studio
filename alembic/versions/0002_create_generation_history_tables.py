"""create generation history, artifact, and job tables

Revision ID: 0002_generation_history
Revises: 0001_lora_metadata
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_generation_history"
down_revision = "0001_lora_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("parent_generation_id", sa.String(length=36), nullable=True),
        sa.Column("settings_snapshot_json", sa.Text(), nullable=False),
        sa.Column("snapshot_schema_version", sa.Integer(), nullable=False),
        sa.Column("workflow_template_id", sa.String(length=200), nullable=False),
        sa.Column("workflow_template_version", sa.String(length=100), nullable=False),
        sa.Column("comfy_prompt_id", sa.String(length=100), nullable=True),
        sa.Column("favorite", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("user_note", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["parent_generation_id"], ["generations.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("comfy_prompt_id"),
    )
    op.create_index("ix_generations_created_at", "generations", ["created_at"])
    op.create_index("ix_generations_status", "generations", ["status"])
    op.create_index("ix_generations_kind", "generations", ["kind"])
    op.create_index("ix_generations_favorite", "generations", ["favorite"])
    op.create_index("ix_generations_parent", "generations", ["parent_generation_id"])
    op.create_index("ix_generations_prompt", "generations", ["comfy_prompt_id"])

    op.create_table(
        "generation_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("generation_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_type", sa.String(length=32), nullable=False),
        sa.Column("local_path", sa.String(length=1000), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "generation_id", "artifact_type", "sha256", name="uq_generation_artifact"
        ),
    )
    op.create_index("ix_generation_artifacts_generation", "generation_artifacts", ["generation_id"])

    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("generation_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("comfy_prompt_id", sa.String(length=100), nullable=True),
        sa.Column("progress_value", sa.Integer(), nullable=True),
        sa.Column("progress_maximum", sa.Integer(), nullable=True),
        sa.Column("current_node", sa.String(length=200), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("generation_id", name="uq_generation_job_generation"),
        sa.UniqueConstraint("comfy_prompt_id"),
    )
    op.create_index("ix_generation_jobs_generation", "generation_jobs", ["generation_id"])
    op.create_index("ix_generation_jobs_prompt", "generation_jobs", ["comfy_prompt_id"])


def downgrade() -> None:
    op.drop_index("ix_generation_jobs_prompt", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_generation", table_name="generation_jobs")
    op.drop_table("generation_jobs")
    op.drop_index("ix_generation_artifacts_generation", table_name="generation_artifacts")
    op.drop_table("generation_artifacts")
    op.drop_index("ix_generations_prompt", table_name="generations")
    op.drop_index("ix_generations_parent", table_name="generations")
    op.drop_index("ix_generations_favorite", table_name="generations")
    op.drop_index("ix_generations_kind", table_name="generations")
    op.drop_index("ix_generations_status", table_name="generations")
    op.drop_index("ix_generations_created_at", table_name="generations")
    op.drop_table("generations")
