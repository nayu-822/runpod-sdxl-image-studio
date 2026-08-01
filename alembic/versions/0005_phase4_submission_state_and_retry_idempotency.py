"""Harden Phase 4 submission, retry, and migration recovery state."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_phase4_submission_state_and_retry_idempotency"
down_revision = "0004_phase4_persistent_generation_queue"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("generation_queue_entries") as batch:
        batch.add_column(
            sa.Column(
                "submission_state",
                sa.String(length=32),
                nullable=False,
                server_default="ready",
            )
        )
        batch.add_column(sa.Column("submission_token", sa.String(length=100), nullable=True))
        batch.add_column(
            sa.Column("submission_started_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.create_check_constraint(
            "ck_generation_queue_submission_state",
            "submission_state IN ('ready', 'submitting', 'submitted', 'ambiguous')",
        )

    op.execute(
        sa.text(
            """UPDATE generation_queue_entries
            SET submission_state='submitted',
                submission_started_at=COALESCE(submission_started_at, updated_at)
            WHERE generation_id IN (
                SELECT g.id FROM generations AS g
                JOIN generation_jobs AS j ON j.generation_id = g.id
                WHERE g.comfy_prompt_id IS NOT NULL OR j.comfy_prompt_id IS NOT NULL
            )"""
        )
    )
    op.execute(
        sa.text(
            """UPDATE generation_queue_entries
            SET submission_state='ambiguous',
                submission_started_at=COALESCE(submission_started_at, updated_at)
            WHERE generation_id IN (
                SELECT g.id FROM generations AS g
                JOIN generation_jobs AS j ON j.generation_id = g.id
                WHERE g.comfy_prompt_id IS NULL
                  AND j.comfy_prompt_id IS NULL
                  AND (
                    g.status IN ('queued', 'running')
                    OR j.status IN ('queued', 'running')
                    OR g.status <> j.status
                  )
            )"""
        )
    )
    op.execute(
        sa.text(
            """UPDATE generations
            SET status='failed',
                error_code='migration_status_mismatch',
                error_summary='Generation and Job state had no prompt ID and could not be resumed',
                completed_at=COALESCE(completed_at, updated_at),
                updated_at=updated_at
            WHERE id IN (
                SELECT g.id FROM generations AS g
                JOIN generation_jobs AS j ON j.generation_id = g.id
                WHERE g.comfy_prompt_id IS NULL
                  AND j.comfy_prompt_id IS NULL
                  AND (
                    g.status IN ('queued', 'running')
                    OR j.status IN ('queued', 'running')
                    OR g.status <> j.status
                  )
            )
              AND status NOT IN ('completed', 'failed', 'cancelled')"""
        )
    )
    op.execute(
        sa.text(
            """UPDATE generation_jobs
            SET status='failed',
                error_code='migration_status_mismatch',
                error_summary='Generation and Job state had no prompt ID and could not be resumed',
                completed_at=COALESCE(completed_at, updated_at),
                updated_at=updated_at
            WHERE generation_id IN (
                SELECT g.id FROM generations AS g
                JOIN generation_jobs AS j ON j.generation_id = g.id
                WHERE g.comfy_prompt_id IS NULL
                  AND j.comfy_prompt_id IS NULL
                  AND (
                    g.status IN ('queued', 'running')
                    OR j.status IN ('queued', 'running')
                    OR g.status <> j.status
                  )
            )
              AND status NOT IN ('completed', 'failed', 'cancelled')"""
        )
    )

    op.create_index(
        "uq_generations_retry_of_generation",
        "generations",
        ["retry_of_generation_id"],
        unique=True,
        sqlite_where=sa.text("retry_of_generation_id IS NOT NULL"),
    )
    op.create_index(
        "uq_generation_batches_retry_of_batch",
        "generation_batches",
        ["retry_of_batch_id"],
        unique=True,
        sqlite_where=sa.text("retry_of_batch_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_generation_batches_retry_of_batch", table_name="generation_batches")
    op.drop_index("uq_generations_retry_of_generation", table_name="generations")
    with op.batch_alter_table("generation_queue_entries") as batch:
        batch.drop_constraint("ck_generation_queue_submission_state", type_="check")
        batch.drop_column("submission_started_at")
        batch.drop_column("submission_token")
        batch.drop_column("submission_state")
