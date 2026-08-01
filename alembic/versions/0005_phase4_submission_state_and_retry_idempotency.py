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

    bind = op.get_bind()
    _normalize_generation_job_pairs(bind)
    _deduplicate_retry_links(bind)
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


def _deduplicate_retry_links(bind: sa.Connection) -> None:
    for table, column in (
        ("generations", "retry_of_generation_id"),
        ("generation_batches", "retry_of_batch_id"),
    ):
        rows = bind.execute(
            sa.text(
                f"SELECT id, {column}, created_at FROM {table} "
                f"WHERE {column} IS NOT NULL ORDER BY {column}, created_at, id"
            )
        ).mappings()
        seen: set[str] = set()
        for row in rows:
            source_id = str(row[column])
            if source_id in seen:
                bind.execute(
                    sa.text(f"UPDATE {table} SET {column}=NULL WHERE id=:id"),
                    {"id": row["id"]},
                )
            else:
                seen.add(source_id)


def _normalize_generation_job_pairs(bind: sa.Connection) -> None:
    rows = bind.execute(
        sa.text(
            """SELECT g.id AS generation_id, j.id AS job_id,
                      g.status AS generation_status, j.status AS job_status,
                      g.comfy_prompt_id AS generation_prompt,
                      j.comfy_prompt_id AS job_prompt,
                      g.error_code AS generation_error_code,
                      g.error_summary AS generation_error_summary,
                      j.error_code AS job_error_code,
                      j.error_summary AS job_error_summary,
                      j.cancel_requested_at AS job_cancel_requested,
                      q.cancel_requested_at AS queue_cancel_requested,
                      q.submission_state AS submission_state,
                      q.updated_at AS queue_updated_at
               FROM generations AS g
               JOIN generation_jobs AS j ON j.generation_id = g.id
               LEFT JOIN generation_queue_entries AS q
                 ON q.generation_id = g.id"""
        )
    ).mappings()
    for row in rows:
        generation_status = str(row["generation_status"])
        job_status = str(row["job_status"])
        generation_prompt = row["generation_prompt"]
        job_prompt = row["job_prompt"]
        prompt_id = generation_prompt or job_prompt
        has_primary_artifact = bool(
            bind.execute(
                sa.text(
                    "SELECT 1 FROM generation_artifacts "
                    "WHERE generation_id=:generation_id AND artifact_type='image' LIMIT 1"
                ),
                {"generation_id": row["generation_id"]},
            ).first()
        )
        cancelled = (
            generation_status == "cancelled"
            or job_status == "cancelled"
            or row["job_cancel_requested"] is not None
            or row["queue_cancel_requested"] is not None
        )
        if has_primary_artifact or generation_status == job_status == "completed":
            target = "completed"
        elif prompt_id is not None:
            target = "queued"
        elif cancelled:
            target = "cancelled"
        elif "failed" in {generation_status, job_status}:
            target = "failed"
        elif generation_status == job_status == "pending":
            target = "pending"
        else:
            target = "failed"

        mismatch_failed = target == "failed" and (
            generation_status != "failed"
            or job_status != "failed"
            or generation_status != job_status
        )
        error_code = (
            "migration_status_mismatch"
            if mismatch_failed
            else (row["generation_error_code"] or row["job_error_code"])
        )
        error_summary = (
            "Generation and Job state had no prompt ID and could not be resumed"
            if mismatch_failed
            else (row["generation_error_summary"] or row["job_error_summary"])
        )
        completed_at = (
            row["queue_updated_at"] if target in {"completed", "failed", "cancelled"} else None
        )
        bind.execute(
            sa.text(
                """UPDATE generations
                   SET status=:status,
                       comfy_prompt_id=:prompt_id,
                       error_code=:error_code,
                       error_summary=:error_summary,
                       completed_at=CASE WHEN :completed_at IS NOT NULL
                                         THEN COALESCE(completed_at, :completed_at)
                                         ELSE completed_at END
                 WHERE id=:generation_id"""
            ),
            {
                "status": target,
                "prompt_id": prompt_id,
                "error_code": error_code,
                "error_summary": error_summary,
                "completed_at": completed_at,
                "generation_id": row["generation_id"],
            },
        )
        bind.execute(
            sa.text(
                """UPDATE generation_jobs
                   SET status=:status,
                       comfy_prompt_id=:prompt_id,
                       error_code=:error_code,
                       error_summary=:error_summary,
                       completed_at=CASE WHEN :completed_at IS NOT NULL
                                         THEN COALESCE(completed_at, :completed_at)
                                         ELSE completed_at END,
                       cancelled_at=CASE WHEN :status='cancelled'
                                         THEN COALESCE(cancelled_at, :completed_at)
                                         ELSE cancelled_at END
                 WHERE id=:job_id"""
            ),
            {
                "status": target,
                "prompt_id": prompt_id,
                "error_code": error_code,
                "error_summary": error_summary,
                "completed_at": completed_at,
                "job_id": row["job_id"],
            },
        )
        if row["submission_state"] is not None:
            queue_state = (
                "submitted"
                if prompt_id is not None
                else (
                    "ambiguous"
                    if target == "failed"
                    and (
                        generation_status != job_status
                        or generation_status in {"queued", "running"}
                    )
                    else "ready"
                )
            )
            bind.execute(
                sa.text(
                    """UPDATE generation_queue_entries
                       SET submission_state=:submission_state,
                           submission_started_at=CASE WHEN :submission_state IN
                                                      ('submitted', 'ambiguous')
                                                      THEN COALESCE(
                                                          submission_started_at, updated_at
                                                      )
                                                      ELSE submission_started_at END,
                           worker_id=NULL, claimed_at=NULL, lease_expires_at=NULL
                     WHERE generation_id=:generation_id"""
                ),
                {
                    "submission_state": queue_state,
                    "generation_id": row["generation_id"],
                },
            )
