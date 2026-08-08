"""Rebuild both metadata candidates for legacy ambiguous Phase 6 rows."""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op

from runpod_sdxl_image_studio.adapters.metadata.comfyui_prompt_metadata_adapter import (
    parse_comfyui_prompt_metadata,
)
from runpod_sdxl_image_studio.adapters.metadata.sidecar_metadata_adapter import (
    parse_sidecar_metadata,
)
from runpod_sdxl_image_studio.domain.metadata_import import MAX_METADATA_RAW_BYTES

revision = "0012_phase6_legacy_metadata_candidates"
down_revision = "0011_phase6_metadata_source_selection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    rows = (
        connection.execute(
            sa.text(
                "SELECT id, raw_metadata_json, candidate_json, candidate_options_json, "
                "source_image_sha256, warnings_json "
                "FROM metadata_imports "
                "WHERE warnings_json LIKE '%metadata_import_ambiguous%'"
            )
        )
        .mappings()
        .all()
    )
    for row in rows:
        try:
            options = _rebuild_candidates(
                row["raw_metadata_json"],
                row["candidate_json"],
                row["candidate_options_json"],
                row["source_image_sha256"],
            )
        except Exception:  # noqa: BLE001 - one corrupt row must not block others
            # A corrupt legacy row must remain inspectable and must not prevent
            # other imports from being repaired during the same migration.
            continue
        if not options:
            # Do not blank a legacy candidate when its raw sources cannot be
            # reconstructed.  Keeping candidate_json is safer than guessing.
            continue
        connection.execute(
            sa.text(
                "UPDATE metadata_imports SET candidate_json = NULL, "
                "candidate_options_json = :candidate_options_json, "
                "selected_metadata_source = NULL, metadata_status = 'needs_mapping' "
                "WHERE id = :id"
            ),
            {
                "id": row["id"],
                "candidate_options_json": json.dumps(
                    options, ensure_ascii=False, separators=(",", ":")
                ),
            },
        )


def downgrade() -> None:
    connection = op.get_bind()
    # 0012 intentionally moves the legacy candidate out of candidate_json and
    # reconstructs one or more source candidates in candidate_options_json.
    # The old row does not record which reconstructed option was the original
    # candidate, so deleting the options during 0011's downgrade could lose
    # data.  Refuse the downgrade before changing either the row or schema.
    protected = connection.execute(
        sa.text(
            "SELECT id FROM metadata_imports "
            "WHERE warnings_json LIKE '%metadata_import_ambiguous%' "
            "AND candidate_json IS NULL "
            "AND candidate_options_json IS NOT NULL "
            "LIMIT 1"
        )
    ).first()
    if protected is not None:
        raise RuntimeError(
            "cannot downgrade Phase 6 legacy metadata candidates without losing candidate data"
        )


def _rebuild_candidates(
    raw_metadata_json: str,
    candidate_json: str | None,
    candidate_options_json: str | None,
    source_image_sha256: str,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    raw_payload = json.loads(raw_metadata_json or "{}")
    sources = raw_payload.get("sources", []) if isinstance(raw_payload, dict) else []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("raw_text"), str):
                continue
            kind = source.get("kind")
            raw_text = source["raw_text"]
            try:
                if kind == "comfyui_prompt":
                    prompt = json.loads(raw_text)
                    if isinstance(prompt, dict):
                        candidate = parse_comfyui_prompt_metadata(prompt).candidate
                        _append_candidate(candidates, candidate.model_dump(mode="json"))
                elif kind == "app_sidecar":
                    candidate = parse_sidecar_metadata(
                        raw_text,
                        source_image_sha256=source_image_sha256,
                        max_raw_bytes=MAX_METADATA_RAW_BYTES,
                    ).candidate
                    _append_candidate(candidates, candidate.model_dump(mode="json"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

    for raw_options in (candidate_options_json,):
        try:
            parsed_options = json.loads(raw_options or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed_options = []
        if isinstance(parsed_options, list):
            for option in parsed_options:
                if isinstance(option, dict):
                    _append_candidate(candidates, option)
    try:
        legacy_candidate = json.loads(candidate_json) if candidate_json else None
    except (TypeError, ValueError, json.JSONDecodeError):
        legacy_candidate = None
    if isinstance(legacy_candidate, dict):
        _append_candidate(candidates, legacy_candidate)
    return candidates


def _append_candidate(candidates: list[dict[str, object]], candidate: dict[str, object]) -> None:
    if any(existing == candidate for existing in candidates):
        return
    candidates.append(candidate)
