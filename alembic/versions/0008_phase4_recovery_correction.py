"""Quarantine rows that the original 0007 classified as migration failures."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_phase4_recovery_correction"
down_revision = "0007_phase4_terminal_state_repair"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Keep uncertain legacy rows visible for manual resolution, never resend them."""

    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """SELECT g.id AS generation_id, j.id AS job_id,
                      g.status AS generation_status, j.status AS job_status,
                      g.comfy_prompt_id AS generation_prompt,
                      j.comfy_prompt_id AS job_prompt,
                      g.error_code AS generation_error_code,
                      j.error_code AS job_error_code,
                      g.retry_of_generation_id AS retry_of_generation_id,
                      j.cancel_requested_at AS job_cancel_requested,
                      j.cancelled_at AS job_cancelled_at,
                      q.cancel_requested_at AS queue_cancel_requested,
                      q.submission_state AS submission_state
               FROM generations AS g
               JOIN generation_jobs AS j ON j.generation_id=g.id
               LEFT JOIN generation_queue_entries AS q
                 ON q.generation_id=g.id
              WHERE g.status='failed' AND j.status='failed'
                AND g.error_code='migration_status_mismatch'
                AND (j.error_code='migration_status_mismatch' OR j.error_code IS NULL)"""
        )
    ).mappings()
    for row in rows:
        if not _is_legacy_pending_candidate(bind, row):
            continue
        _quarantine(bind, row)


def downgrade() -> None:
    """The quarantine is durable evidence and is not reversed automatically."""


def _is_legacy_pending_candidate(bind: sa.Connection, row: sa.RowMapping) -> bool:
    if row["generation_prompt"] is not None or row["job_prompt"] is not None:
        return False
    if row["retry_of_generation_id"] is not None:
        return False
    if any(
        value is not None
        for value in (
            row["job_cancel_requested"],
            row["job_cancelled_at"],
            row["queue_cancel_requested"],
        )
    ):
        return False
    if row["submission_state"] not in {None, "ambiguous"}:
        return False
    artifact = bind.execute(
        sa.text(
            "SELECT 1 FROM generation_artifacts "
            "WHERE generation_id=:generation_id AND artifact_type='image' LIMIT 1"
        ),
        {"generation_id": row["generation_id"]},
    ).first()
    return artifact is None


def _quarantine(bind: sa.Connection, row: sa.RowMapping) -> None:
    summary = (
        "Legacy 0007 could not prove the original pending state; manual resolution is required."
    )
    bind.execute(
        sa.text(
            """UPDATE generations
                  SET error_code='migration_status_ambiguous',
                      error_summary=:summary,
                      updated_at=COALESCE(updated_at, CURRENT_TIMESTAMP)
                WHERE id=:generation_id"""
        ),
        {"summary": summary, "generation_id": row["generation_id"]},
    )
    bind.execute(
        sa.text(
            """UPDATE generation_jobs
                  SET error_code='migration_status_ambiguous',
                      error_summary=:summary,
                      updated_at=COALESCE(updated_at, CURRENT_TIMESTAMP)
                WHERE id=:job_id"""
        ),
        {"summary": summary, "job_id": row["job_id"]},
    )
    if row["submission_state"] is not None:
        bind.execute(
            sa.text(
                """UPDATE generation_queue_entries
                      SET submission_state='ambiguous',
                          worker_id=NULL, claimed_at=NULL, lease_expires_at=NULL,
                          updated_at=COALESCE(updated_at, CURRENT_TIMESTAMP)
                    WHERE generation_id=:generation_id"""
            ),
            {"generation_id": row["generation_id"]},
        )
