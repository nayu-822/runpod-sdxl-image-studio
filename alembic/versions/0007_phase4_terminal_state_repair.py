"""Repair Phase 4 state without treating ordinary pending work as failed."""

from __future__ import annotations

from collections.abc import Mapping

import sqlalchemy as sa
from alembic import op

revision = "0007_phase4_terminal_state_repair"
down_revision = "0006_phase4_reconcile_existing_state"
branch_labels = None
depends_on = None

TERMINAL_STATES = {"completed", "failed", "cancelled"}
KNOWN_FAILURE_CODES = {
    "validation_error",
    "workflow_error",
    "comfyui_connection_error",
    "comfyui_prompt_error",
    "comfyui_execution_error",
    "history_timeout",
    "output_not_found",
    "image_download_error",
    "image_validation_error",
    "storage_error",
    "database_error",
    "recovery_error",
}
AMBIGUOUS_CODES = {
    "prompt_submission_ambiguous",
    "prompt_submission_ambiguous_resolved",
    "migration_status_ambiguous",
    "migration_prompt_id_mismatch",
}


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
                      g.completed_at AS generation_completed_at,
                      j.completed_at AS job_completed_at,
                      j.cancel_requested_at AS job_cancel_requested,
                      j.cancelled_at AS job_cancelled_at,
                      q.cancel_requested_at AS queue_cancel_requested,
                      q.submission_state AS submission_state,
                      q.updated_at AS queue_updated_at,
                      g.updated_at AS generation_updated_at,
                      j.updated_at AS job_updated_at,
                      g.retry_of_generation_id AS retry_of_generation_id
               FROM generations AS g
               JOIN generation_jobs AS j ON j.generation_id=g.id
               LEFT JOIN generation_queue_entries AS q
                 ON q.generation_id=g.id"""
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
        result = _classify(row, has_artifact)
        _write_pair(bind, row, result)


def downgrade() -> None:
    """State repair is intentionally retained when the revision is downgraded."""


def _classify(row: Mapping[str, object], has_artifact: bool) -> dict[str, object]:
    generation_status = str(row["generation_status"])
    job_status = str(row["job_status"])
    generation_prompt = _prompt(row["generation_prompt"])
    job_prompt = _prompt(row["job_prompt"])
    queue_state = str(row["submission_state"] or "ready")
    has_cancel_request = (
        row["job_cancel_requested"] is not None
        or row["queue_cancel_requested"] is not None
        or row["job_cancelled_at"] is not None
    )
    prompt = generation_prompt or job_prompt
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

    completion_evidence = (
        generation_status == "completed"
        or job_status == "completed"
        or row["generation_completed_at"] is not None
        or row["job_completed_at"] is not None
    )
    if (
        has_artifact
        and completion_evidence
        and not {generation_status, job_status}.intersection({"failed", "cancelled"})
    ):
        return _result("completed", "completed", prompt, "submitted" if prompt else queue_state)

    if row["job_cancelled_at"] is not None or (
        generation_status == "cancelled" and job_status == "cancelled"
    ):
        return _result("cancelled", "cancelled", prompt, "submitted" if prompt else queue_state)

    if generation_status == "failed" and job_status == "failed":
        return _result("failed", "failed", prompt, "submitted" if prompt else queue_state)
    if _has_failure_evidence(row, generation_status, job_status):
        return _result("failed", "failed", prompt, "submitted" if prompt else queue_state)
    if generation_status == "completed" and job_status == "completed":
        return _result("completed", "completed", prompt, "submitted" if prompt else queue_state)

    # This branch must precede the fallback ambiguity/failure classification.
    if (
        generation_status == job_status == "pending"
        and queue_state == "ready"
        and prompt is None
        and not has_cancel_request
    ):
        return _result("pending", "pending", None, "ready")

    if queue_state in {"submitting", "ambiguous"}:
        return _result(
            generation_status,
            job_status,
            prompt,
            "ambiguous",
            _existing_ambiguous_code(row),
        )

    if (
        prompt is not None
        and generation_status not in TERMINAL_STATES
        and job_status not in TERMINAL_STATES
    ):
        target = "running" if "running" in {generation_status, job_status} else "queued"
        return _result(target, target, prompt, "submitted")

    if has_cancel_request and prompt is None:
        return _result(
            generation_status,
            job_status,
            None,
            "ambiguous",
            "migration_status_ambiguous",
        )

    # Preserve terminal information; never demote a terminal pair to resendable work.
    if generation_status in TERMINAL_STATES or job_status in TERMINAL_STATES:
        return _result(
            generation_status,
            job_status,
            prompt,
            "ambiguous",
            "migration_status_ambiguous",
        )

    return _result(
        generation_status,
        job_status,
        prompt,
        "ambiguous",
        "migration_status_ambiguous",
    )


def _has_failure_evidence(
    row: Mapping[str, object], generation_status: str, job_status: str
) -> bool:
    if "failed" not in {generation_status, job_status}:
        return False
    failed_code = row["generation_error_code"] or row["job_error_code"]
    if not isinstance(failed_code, str) or failed_code in AMBIGUOUS_CODES:
        return False
    if failed_code not in KNOWN_FAILURE_CODES:
        return False
    return row["generation_completed_at"] is not None or row["job_completed_at"] is not None


def _existing_ambiguous_code(row: Mapping[str, object]) -> str:
    for key in ("generation_error_code", "job_error_code"):
        value = row[key]
        if isinstance(value, str) and value in AMBIGUOUS_CODES:
            return value
    return "migration_status_ambiguous"


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


def _write_pair(
    bind: sa.Connection, row: Mapping[str, object], result: Mapping[str, object]
) -> None:
    error_code = result["error_code"] or row["generation_error_code"] or row["job_error_code"]
    error_summary = (
        "Generation and Job prompt IDs differ; manual resolution is required."
        if result["error_code"] == "migration_prompt_id_mismatch"
        else "Migration could not prove a safe terminal state; manual review is required."
        if result["error_code"] == "migration_status_ambiguous"
        else row["generation_error_summary"] or row["job_error_summary"]
    )
    terminal_at = row["generation_updated_at"] or row["job_updated_at"] or row["queue_updated_at"]
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
