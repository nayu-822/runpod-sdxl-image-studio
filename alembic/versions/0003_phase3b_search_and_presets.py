"""Add Phase 3B search indexes and preset storage."""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op

revision = "0003_phase3b_search_and_presets"
down_revision = "0002_generation_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for _name, column in (
        ("checkpoint_name", sa.Column("checkpoint_name", sa.String(length=500), nullable=True)),
        ("vae_name", sa.Column("vae_name", sa.String(length=500), nullable=True)),
        ("seed", sa.Column("seed", sa.BigInteger(), nullable=True)),
        ("width", sa.Column("width", sa.Integer(), nullable=True)),
        ("height", sa.Column("height", sa.Integer(), nullable=True)),
        ("positive_prompt_search", sa.Column("positive_prompt_search", sa.Text(), nullable=True)),
        ("negative_prompt_search", sa.Column("negative_prompt_search", sa.Text(), nullable=True)),
    ):
        op.add_column("generations", column)

    op.create_table(
        "generation_loras",
        sa.Column("generation_id", sa.String(length=36), nullable=False),
        sa.Column("lora_name", sa.String(length=500), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("model_strength", sa.Float(), nullable=False),
        sa.Column("clip_strength", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("generation_id", "lora_name"),
        sa.UniqueConstraint("generation_id", "lora_name", name="uq_generation_lora_name"),
    )
    op.create_index("ix_generation_loras_name", "generation_loras", ["lora_name"])
    for name, columns in (
        ("ix_generations_checkpoint", ["checkpoint_name"]),
        ("ix_generations_vae", ["vae_name"]),
        ("ix_generations_seed", ["seed"]),
        ("ix_generations_resolution", ["width", "height"]),
    ):
        op.create_index(name, "generations", columns)

    op.create_table(
        "presets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("favorite", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("kind", "name", name="uq_preset_kind_name"),
    )
    for name, columns in (
        ("ix_presets_kind", ["kind"]),
        ("ix_presets_name", ["name"]),
        ("ix_presets_favorite", ["favorite"]),
        ("ix_presets_last_used", ["last_used_at"]),
        ("ix_presets_updated", ["updated_at"]),
    ):
        op.create_index(name, "presets", columns)
    _backfill_generation_search_data()


def _backfill_generation_search_data() -> None:
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, settings_snapshot_json FROM generations")).mappings()
    for row in rows:
        try:
            snapshot = json.loads(row["settings_snapshot_json"])
            loras = snapshot.get("loras", ())
            bind.execute(
                sa.text(
                    """UPDATE generations SET checkpoint_name=:checkpoint, vae_name=:vae,
                    seed=:seed, width=:width, height=:height,
                    positive_prompt_search=:positive, negative_prompt_search=:negative
                    WHERE id=:id"""
                ),
                {
                    "id": row["id"],
                    "checkpoint": snapshot.get("checkpoint_name"),
                    "vae": snapshot.get("vae_name"),
                    "seed": snapshot.get("seed"),
                    "width": snapshot.get("width"),
                    "height": snapshot.get("height"),
                    "positive": snapshot.get("positive_prompt", ""),
                    "negative": snapshot.get("negative_prompt", ""),
                },
            )
            for index, lora in enumerate(loras):
                bind.execute(
                    sa.text(
                        """INSERT OR IGNORE INTO generation_loras
                        (generation_id, lora_name, order_index, model_strength, clip_strength)
                        VALUES (:generation_id, :lora_name, :order_index,
                        :model_strength, :clip_strength)"""
                    ),
                    {
                        "generation_id": row["id"],
                        "lora_name": lora["name"],
                        "order_index": lora.get("order", index),
                        "model_strength": lora["model_strength"],
                        "clip_strength": lora["clip_strength"],
                    },
                )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue


def downgrade() -> None:
    for name in (
        "ix_presets_updated",
        "ix_presets_last_used",
        "ix_presets_favorite",
        "ix_presets_name",
        "ix_presets_kind",
    ):
        op.drop_index(name, table_name="presets")
    op.drop_table("presets")
    op.drop_index("ix_generation_loras_name", table_name="generation_loras")
    op.drop_table("generation_loras")
    for name in (
        "ix_generations_resolution",
        "ix_generations_seed",
        "ix_generations_vae",
        "ix_generations_checkpoint",
    ):
        op.drop_index(name, table_name="generations")
    with op.batch_alter_table("generations") as batch:
        for name in (
            "negative_prompt_search",
            "positive_prompt_search",
            "height",
            "width",
            "seed",
            "vae_name",
            "checkpoint_name",
        ):
            batch.drop_column(name)
