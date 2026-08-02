"""Add durable submission state and safely normalize existing Phase 4 pairs."""

from __future__ import annotations

from collections.abc import Mapping

import sqlalchemy as sa
from alembic import op

revision = "0005_phase4_submission_state_and_retry_idempotency"
down_revision = "0004_phase4_persistent_generation_queue"
branch_labels = None
depends_on = None

TERMINAL_STATES = {"completed", "failed", "cancelled"}


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
                      j.cancelled_at AS job_cancelled_at,
                      q.cancel_requested_at AS queue_cancel_requested,
                      q.submission_state AS submission_state,
                      q.updated_at AS queue_updated_at,
                      g.updated_at AS generation_updated_at,
                      j.updated_at AS job_updated_at
               FROM generations AS g
               JOIN generation_jobs AS j ON j.generation_id=g.id
               LEFT JOIN generation_queue_entries AS q
                 ON q.generation_id=g.id"""
        )
    ).mappings()
    for row in rows:
        has_primary_artifact = bool(
            bind.execute(
                sa.text(
                    "SELECT 1 FROM generation_artifacts "
                    "WHERE generation_id=:generation_id AND artifact_type='image' LIMIT 1"
                ),
                {"generation_id": row["generation_id"]},
            ).first()
        )
        normalized = _classify_pair(row, has_primary_artifact)
        _write_pair(bind, row, normalized)


def _classify_pair(row: Mapping[str, object], has_primary_artifact: bool) -> dict[str, object]:
    generation_status = str(row["generation_status"])
    job_status = str(row["job_status"])
    generation_prompt = _prompt(row["generation_prompt"])
    job_prompt = _prompt(row["job_prompt"])
    queue_state = str(row["submission_state"] or "ready")
    has_cancel_request = (
        row["job_cancel_requested"] is not None
        or row["job_cancelled_at"] is not None
        or row["queue_cancel_requested"] is not None
    )
    prompt_mismatch = (
        generation_prompt is not None and job_prompt is not None and generation_prompt != job_prompt
    )
    if prompt_mismatch:
        return {
            "generation_status": generation_status,
            "job_status": job_status,
            "prompt": None,
            "generation_prompt": generation_prompt,
            "job_prompt": job_prompt,
            "queue_state": "ambiguous",
            "error_code": "migration_prompt_id_mismatch",
            "error_summary": "GenerationとJobのprompt IDが一致しません。",
        }

    prompt = generation_prompt or job_prompt
    if has_primary_artifact and not {generation_status, job_status} & {"failed", "cancelled"}:
        return _classified("completed", prompt, "submitted" if prompt else queue_state, None, None)
    if generation_status == job_status and generation_status in TERMINAL_STATES:
        return _classified(
            generation_status,
            prompt,
            "submitted" if prompt else queue_state,
            None,
            None,
        )
    terminal_states = {
        status for status in (generation_status, job_status) if status in TERMINAL_STATES
    }
    if terminal_states:
        terminal = _safe_terminal(terminal_states)
        return _classified(terminal, prompt, "submitted" if prompt else queue_state, None, None)
    if row["job_cancelled_at"] is not None:
        return _classified("cancelled", prompt, queue_state, None, None)
    if (
        has_cancel_request
        and generation_status == job_status == "pending"
        and queue_state == "ready"
        and prompt is None
    ):
        return _classified("cancelled", None, queue_state, None, None)
    if queue_state in {"submitting", "ambiguous"} or (has_cancel_request and prompt is None):
        return _classified(
            generation_status,
            prompt,
            "ambiguous",
            "migration_status_ambiguous",
            "送信結果またはキャンセル結果を安全に確定できません。",
        )
    if prompt is not None:
        target = "running" if "running" in {generation_status, job_status} else "queued"
        return _classified(target, prompt, "submitted", None, None)
    if generation_status == job_status == "pending" and queue_state == "ready":
        return _classified("pending", None, queue_state, None, None)
    return _classified(
        "failed",
        None,
        "ambiguous",
        "migration_status_mismatch",
        "GenerationとJobの状態を安全に復元できません。",
    )


def _classified(
    status: str,
    prompt: str | None,
    queue_state: str,
    error_code: str | None,
    error_summary: str | None,
) -> dict[str, object]:
    return {
        "generation_status": status,
        "job_status": status,
        "prompt": prompt,
        "queue_state": queue_state,
        "error_code": error_code,
        "error_summary": error_summary,
    }


def _write_pair(
    bind: sa.Connection,
    row: Mapping[str, object],
    normalized: Mapping[str, object],
) -> None:
    target_generation_status = str(normalized["generation_status"])
    target_job_status = str(normalized["job_status"])
    error_code = normalized["error_code"] or row["generation_error_code"] or row["job_error_code"]
    error_summary = (
        normalized["error_summary"] or row["generation_error_summary"] or row["job_error_summary"]
    )
    terminal_at = row["generation_updated_at"] or row["job_updated_at"] or row["queue_updated_at"]
    bind.execute(
        sa.text(
            """UPDATE generations
               SET status=:status, comfy_prompt_id=:prompt_id,
                   error_code=:error_code, error_summary=:error_summary,
                   completed_at=CASE WHEN :is_terminal=1
                                     THEN COALESCE(completed_at, :terminal_at)
                                     ELSE completed_at END
             WHERE id=:generation_id"""
        ),
        {
            "status": target_generation_status,
            "prompt_id": normalized.get("generation_prompt", normalized["prompt"]),
            "error_code": error_code,
            "error_summary": error_summary,
            "is_terminal": int(target_generation_status in TERMINAL_STATES),
            "terminal_at": terminal_at,
            "generation_id": row["generation_id"],
        },
    )
    bind.execute(
        sa.text(
            """UPDATE generation_jobs
               SET status=:status, comfy_prompt_id=:prompt_id,
                   error_code=:error_code, error_summary=:error_summary,
                   completed_at=CASE WHEN :is_terminal=1
                                     THEN COALESCE(completed_at, :terminal_at)
                                     ELSE completed_at END,
                   cancelled_at=CASE WHEN :status='cancelled'
                                     THEN COALESCE(cancelled_at, :terminal_at)
                                     ELSE cancelled_at END
             WHERE id=:job_id"""
        ),
        {
            "status": target_job_status,
            "prompt_id": normalized.get("job_prompt", normalized["prompt"]),
            "error_code": error_code,
            "error_summary": error_summary,
            "is_terminal": int(target_job_status in TERMINAL_STATES),
            "terminal_at": terminal_at,
            "job_id": row["job_id"],
        },
    )
    if row["submission_state"] is not None:
        queue_state = str(normalized["queue_state"])
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
                "submission_state": queue_state,
                "generation_id": row["generation_id"],
            },
        )


def _prompt(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_terminal(states: set[str]) -> str:
    if "cancelled" in states:
        return "cancelled"
    if "failed" in states:
        return "failed"
    return "completed"
