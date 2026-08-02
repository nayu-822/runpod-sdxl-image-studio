"""Repair rows already changed by the original Phase 4 normalization migrations."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_phase4_terminal_state_repair"
down_revision = "0006_phase4_reconcile_existing_state"
branch_labels = None
depends_on = None

TERMINAL_STATES = {"completed", "failed", "cancelled"}


def upgrade() -> None:
    bind = op.get_bind()
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
                      j.cancelled_at AS job_cancelled_at,
                      q.cancel_requested_at AS queue_cancel_requested,
                      q.submission_state AS submission_state,
                      q.updated_at AS queue_updated_at,
                      g.updated_at AS generation_updated_at,
                      j.updated_at AS job_updated_at
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
        result = _repair_row(row, has_artifact)
        _write_row(bind, row, result)


def downgrade() -> None:
    """Data repair is intentionally retained when the revision is downgraded."""


def _repair_row(row: sa.RowMapping, has_artifact: bool) -> dict[str, object]:
    generation_status = str(row["generation_status"])
    job_status = str(row["job_status"])
    generation_prompt = _prompt(row["generation_prompt"])
    job_prompt = _prompt(row["job_prompt"])
    queue_state = str(row["submission_state"] or "ready")
    prompt = generation_prompt or job_prompt
    has_cancel_request = (
        row["job_cancel_requested"] is not None or row["queue_cancel_requested"] is not None
    )
    if generation_prompt is not None and job_prompt is not None and generation_prompt != job_prompt:
        return _result(
            generation_status,
            job_status,
            None,
            "ambiguous",
            "migration_prompt_id_mismatch",
            generation_prompt=generation_prompt,
            job_prompt=job_prompt,
        )
    if has_artifact and not {generation_status, job_status} & {"failed", "cancelled"}:
        return _result("completed", "completed", prompt, "submitted")
    if row["job_cancelled_at"] is not None:
        return _result("cancelled", "cancelled", prompt, "submitted" if prompt else queue_state)
    if generation_status == job_status == "completed":
        return _result("completed", "completed", prompt, "submitted" if prompt else queue_state)
    if generation_status == job_status == "cancelled":
        return _result("cancelled", "cancelled", prompt, "submitted" if prompt else queue_state)
    if generation_status == job_status == "failed":
        return _result("failed", "failed", prompt, "submitted" if prompt else queue_state)
    if (
        any(value is not None for value in (row["generation_error_code"], row["job_error_code"]))
        and not has_artifact
    ):
        return _result("failed", "failed", prompt, "ambiguous", "migration_status_mismatch")
    if (
        has_cancel_request
        and generation_status == job_status == "pending"
        and queue_state == "ready"
        and prompt is None
    ):
        return _result("cancelled", "cancelled", None, "ready")
    if has_cancel_request or queue_state in {"submitting", "ambiguous"}:
        return _result(
            generation_status,
            job_status,
            prompt,
            "ambiguous",
            "migration_status_ambiguous",
        )
    if prompt is not None:
        target = "running" if "running" in {generation_status, job_status} else "queued"
        return _result(target, target, prompt, "submitted")
    if generation_status in TERMINAL_STATES or job_status in TERMINAL_STATES:
        target = "cancelled" if "cancelled" in {generation_status, job_status} else "failed"
        return _result(target, target, prompt, queue_state)
    return _result("failed", "failed", None, "ambiguous", "migration_status_mismatch")


def _result(
    generation_status: str,
    job_status: str,
    prompt: str | None,
    queue_state: str,
    error_code: str | None = None,
    *,
    generation_prompt: str | None = None,
    job_prompt: str | None = None,
) -> dict[str, object]:
    return {
        "generation_status": generation_status,
        "job_status": job_status,
        "prompt": prompt,
        "generation_prompt": prompt if generation_prompt is None else generation_prompt,
        "job_prompt": prompt if job_prompt is None else job_prompt,
        "queue_state": queue_state,
        "error_code": error_code,
    }


def _write_row(bind: sa.Connection, row: sa.RowMapping, result: dict[str, object]) -> None:
    terminal_at = row["generation_updated_at"] or row["job_updated_at"] or row["queue_updated_at"]
    error_code = result["error_code"] or row["generation_error_code"] or row["job_error_code"]
    error_summary = (
        "GenerationとJobのprompt IDが一致しません。"
        if result["error_code"] == "migration_prompt_id_mismatch"
        else "GenerationとJobの状態を安全に復元できません。"
        if result["error_code"] in {"migration_status_mismatch", "migration_status_ambiguous"}
        else row["generation_error_summary"] or row["job_error_summary"]
    )
    for table, row_key, status_key in (
        ("generations", "generation_id", "generation_status"),
        ("generation_jobs", "job_id", "job_status"),
    ):
        prompt_key = "generation_prompt" if table == "generations" else "job_prompt"
        cancelled_sql = (
            ", cancelled_at=CASE WHEN :status='cancelled' "
            "THEN COALESCE(cancelled_at, :terminal_at) ELSE cancelled_at END"
            if table == "generation_jobs"
            else ""
        )
        bind.execute(
            sa.text(
                f"""UPDATE {table}
                   SET status=:status, comfy_prompt_id=:prompt_id,
                       error_code=:error_code, error_summary=:error_summary,
                       completed_at=CASE WHEN :is_terminal=1
                                         THEN COALESCE(completed_at, :terminal_at)
                                         ELSE completed_at END{cancelled_sql}
                 WHERE id=:row_id"""
            ),
            {
                "status": result[status_key],
                "prompt_id": result[prompt_key],
                "error_code": error_code,
                "error_summary": error_summary,
                "is_terminal": int(str(result[status_key]) in TERMINAL_STATES),
                "terminal_at": terminal_at,
                "row_id": row[row_key],
            },
        )
    if row["submission_state"] is not None:
        bind.execute(
            sa.text(
                """UPDATE generation_queue_entries
                   SET submission_state=:submission_state,
                       submission_started_at=CASE WHEN :submission_state IN
                                                  ('submitted', 'ambiguous')
                                                  THEN COALESCE(submission_started_at, updated_at)
                                                  ELSE submission_started_at END,
                       worker_id=NULL, claimed_at=NULL, lease_expires_at=NULL
                 WHERE generation_id=:generation_id"""
            ),
            {
                "submission_state": result["queue_state"],
                "generation_id": row["generation_id"],
            },
        )


def _prompt(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
