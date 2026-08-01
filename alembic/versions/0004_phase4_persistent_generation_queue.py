"""Add the Phase 4 persistent FIFO generation queue."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_phase4_persistent_generation_queue"
down_revision = "0003_phase3b_search_and_presets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("generations") as batch:
        batch.add_column(sa.Column("retry_of_generation_id", sa.String(length=36), nullable=True))
        batch.add_column(
            sa.Column("retry_attempt", sa.Integer(), nullable=False, server_default="0")
        )
        batch.create_foreign_key(
            "fk_generations_retry_of", "generations", ["retry_of_generation_id"], ["id"]
        )

    with op.batch_alter_table("generation_jobs") as batch:
        batch.add_column(sa.Column("worker_id", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(
            sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "generation_batches",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("seed_strategy", sa.String(length=32), nullable=False),
        sa.Column("start_seed", sa.BigInteger(), nullable=True),
        sa.Column("seed_step", sa.BigInteger(), nullable=False),
        sa.Column("retry_of_batch_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("item_count > 0", name="ck_generation_batch_item_count_positive"),
        sa.CheckConstraint("seed_step > 0", name="ck_generation_batch_seed_step_positive"),
        sa.ForeignKeyConstraint(
            ["retry_of_batch_id"], ["generation_batches.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_generation_batches_created", "generation_batches", ["created_at"])

    op.create_table(
        "generation_queue_entries",
        sa.Column("sequence", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("generation_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=True),
        sa.Column("batch_index", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(length=200), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("batch_index >= 0", name="ck_generation_queue_batch_index_nonnegative"),
        sa.ForeignKeyConstraint(["batch_id"], ["generation_batches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["generation_jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("sequence"),
        sa.UniqueConstraint("generation_id"),
        sa.UniqueConstraint("job_id"),
        sa.UniqueConstraint("batch_id", "batch_index", name="uq_generation_queue_batch_index"),
    )
    op.create_index("ix_generation_queue_batch", "generation_queue_entries", ["batch_id"])
    op.create_index("ix_generation_queue_lease", "generation_queue_entries", ["lease_expires_at"])

    # Existing unfinished records remain durable and become queue work. Terminal history is not
    # queued. A generation/job mismatch is intentionally skipped rather than creating an orphan.
    op.execute(
        sa.text(
            """INSERT INTO generation_queue_entries
            (generation_id, job_id, batch_id, batch_index, worker_id, claimed_at,
             lease_expires_at, cancel_requested_at, enqueued_at, updated_at)
            SELECT g.id, j.id, NULL, 0, NULL, NULL, NULL, NULL, g.created_at, g.updated_at
            FROM generations AS g
            JOIN generation_jobs AS j ON j.generation_id = g.id
            WHERE g.status IN ('pending', 'queued', 'running')
              AND j.status IN ('pending', 'queued', 'running')"""
        )
    )


def downgrade() -> None:
    op.drop_index("ix_generation_queue_lease", table_name="generation_queue_entries")
    op.drop_index("ix_generation_queue_batch", table_name="generation_queue_entries")
    op.drop_table("generation_queue_entries")
    op.drop_index("ix_generation_batches_created", table_name="generation_batches")
    op.drop_table("generation_batches")

    with op.batch_alter_table("generation_jobs") as batch:
        for name in (
            "cancelled_at",
            "cancel_requested_at",
            "lease_expires_at",
            "claimed_at",
            "worker_id",
        ):
            batch.drop_column(name)
    with op.batch_alter_table("generations") as batch:
        batch.drop_constraint("fk_generations_retry_of", type_="foreignkey")
        batch.drop_column("retry_attempt")
        batch.drop_column("retry_of_generation_id")
