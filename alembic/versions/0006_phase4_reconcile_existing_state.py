"""Repair databases where the original Phase 4 migration was already applied."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_phase4_reconcile_existing_state"
down_revision = "0005_phase4_submission_state_and_retry_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """SELECT g.id AS generation_id, j.id AS job_id,
                      g.status AS generation_status, j.status AS job_status,
                      g.comfy_prompt_id AS generation_prompt,
                      j.comfy_prompt_id AS job_prompt,
                      g.updated_at AS generation_updated_at,
                      j.updated_at AS job_updated_at,
                      j.cancel_requested_at AS job_cancel_requested,
                      q.cancel_requested_at AS queue_cancel_requested,
                      q.submission_state AS submission_state
               FROM generations AS g
               JOIN generation_jobs AS j ON j.generation_id=g.id
               LEFT JOIN generation_queue_entries AS q ON q.generation_id=g.id"""
        )
    ).mappings()
    for row in rows:
        has_artifact = bool(
            bind.execute(
                sa.text(
                    "SELECT 1 FROM generation_artifacts "
                    "WHERE generation_id=:generation_id AND artifact_type='image' LIMIT 1"
                ),
                {"generation_id": row["generation_id"]},
            ).first()
        )
        generation_status = str(row["generation_status"])
        job_status = str(row["job_status"])
        prompt_id = row["generation_prompt"] or row["job_prompt"]
        cancelled = (
            generation_status == "cancelled"
            or job_status == "cancelled"
            or row["job_cancel_requested"] is not None
            or row["queue_cancel_requested"] is not None
        )
        if has_artifact or generation_status == job_status == "completed":
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
        error_code = "migration_status_mismatch" if mismatch_failed else None
        error_summary = (
            "Generation and Job state had no prompt ID and could not be resumed"
            if mismatch_failed
            else None
        )
        terminal_at = (
            row["generation_updated_at"] or row["job_updated_at"]
            if target in {"completed", "failed", "cancelled"}
            else None
        )
        bind.execute(
            sa.text(
                """UPDATE generations
                   SET status=:status, comfy_prompt_id=:prompt_id,
                       error_code=COALESCE(:error_code, error_code),
                       error_summary=COALESCE(:error_summary, error_summary),
                       completed_at=CASE WHEN :terminal_at IS NOT NULL
                                         THEN COALESCE(completed_at, :terminal_at)
                                         ELSE completed_at END
                 WHERE id=:generation_id"""
            ),
            {
                "status": target,
                "prompt_id": prompt_id,
                "error_code": error_code,
                "error_summary": error_summary,
                "terminal_at": terminal_at,
                "generation_id": row["generation_id"],
            },
        )
        bind.execute(
            sa.text(
                """UPDATE generation_jobs
                   SET status=:status, comfy_prompt_id=:prompt_id,
                       error_code=COALESCE(:error_code, error_code),
                       error_summary=COALESCE(:error_summary, error_summary),
                       completed_at=CASE WHEN :terminal_at IS NOT NULL
                                         THEN COALESCE(completed_at, :terminal_at)
                                         ELSE completed_at END,
                       cancelled_at=CASE WHEN :status='cancelled'
                                         THEN COALESCE(cancelled_at, :terminal_at)
                                         ELSE cancelled_at END
                 WHERE id=:job_id"""
            ),
            {
                "status": target,
                "prompt_id": prompt_id,
                "error_code": error_code,
                "error_summary": error_summary,
                "terminal_at": terminal_at,
                "job_id": row["job_id"],
            },
        )
        if row["submission_state"] is not None:
            queue_state = (
                "submitted"
                if prompt_id is not None
                else ("ambiguous" if mismatch_failed else "ready")
            )
            bind.execute(
                sa.text(
                    "UPDATE generation_queue_entries SET submission_state=:state "
                    "WHERE generation_id=:generation_id"
                ),
                {"state": queue_state, "generation_id": row["generation_id"]},
            )


def downgrade() -> None:
    """The repair is intentionally data-preserving and has no inverse operation."""
