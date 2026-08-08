"""Application service coordinating safe metadata import and preview decisions."""

from __future__ import annotations

import hashlib
import logging
import ntpath
import posixpath
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import UUID

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.database.repositories.metadata_import_repository import (
    MetadataImportRepositoryError,
    MetadataImportRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.metadata.comfyui_prompt_metadata_adapter import (
    parse_comfyui_prompt_metadata,
)
from runpod_sdxl_image_studio.adapters.metadata.png_metadata_adapter import parse_png_metadata
from runpod_sdxl_image_studio.adapters.metadata.sidecar_metadata_adapter import (
    SidecarMetadataError,
    parse_sidecar_metadata,
)
from runpod_sdxl_image_studio.adapters.storage.imported_image_storage import ImportedImageStorage
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.metadata_import import (
    ImportedImage,
    MetadataImportCandidate,
    MetadataImportError,
    MetadataImportPreview,
    MetadataImportRecord,
    MetadataImportStatus,
    MetadataModelMapping,
    MetadataRawSource,
    MetadataSourceKind,
)

logger = logging.getLogger(__name__)

_INVALID_METADATA_WARNINGS = frozenset(
    {
        "metadata_import_invalid_json",
        "metadata_import_invalid_utf8",
        "metadata_import_unsupported_schema",
        "metadata_import_too_large",
        "metadata_import_parse_failed",
        "metadata_import_png_prompt_invalid",
    }
)
_BLOCKING_WARNINGS = frozenset(
    {
        "metadata_import_sidecar_hash_mismatch",
        "metadata_import_ambiguous",
        "metadata_import_model_missing",
        "metadata_import_model_catalog_unavailable",
        "metadata_import_mapping_invalid",
        "metadata_import_unresolved",
    }
    | _INVALID_METADATA_WARNINGS
)
_CAPABILITY_WARNINGS = frozenset(
    {
        "metadata_import_model_missing",
        "metadata_import_model_catalog_unavailable",
    }
)


class MetadataImportService:
    """Import, preview, map, and validate external image metadata."""

    def __init__(
        self,
        repository: MetadataImportRepositoryProtocol,
        storage: ImportedImageStorage,
        settings: Settings,
        *,
        capabilities: ComfyUICapabilities | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._settings = settings
        self._capabilities = capabilities
        self._import_lock = Lock()

    def import_image(
        self,
        image_bytes: bytes,
        original_filename: str | None = None,
        *,
        sidecar_bytes: bytes | str | bytearray | None = None,
    ) -> MetadataImportPreview:
        # Gradio disables the button client-side, but the server-side lock and
        # source hash lookup make a repeated request idempotent as well.
        with self._import_lock:
            source_hash = hashlib.sha256(image_bytes).hexdigest()
            finder = getattr(self._repository, "get_by_source_image_sha256", None)
            if callable(finder):
                existing = finder(source_hash)
                if existing is not None and _same_import_inputs(existing, sidecar_bytes):
                    try:
                        self._storage.verify(existing.imported_image)
                    except Exception:  # noqa: BLE001 - stale rows are re-imported safely
                        pass
                    else:
                        return _preview(existing)
            return self._import_image(
                image_bytes,
                original_filename,
                sidecar_bytes=sidecar_bytes,
            )

    def _import_image(
        self,
        image_bytes: bytes,
        original_filename: str | None = None,
        *,
        sidecar_bytes: bytes | str | bytearray | None = None,
    ) -> MetadataImportPreview:
        imported = self._storage.store(image_bytes, original_filename)
        committed = False
        try:
            png_result = parse_png_metadata(
                image_bytes,
                max_raw_bytes=self._settings.max_metadata_raw_bytes,
            )
            warnings: list[str] = [*png_result.warnings]
            raw_sources: list[MetadataRawSource] = [*png_result.raw_sources]
            sidecar_result = None
            sidecar_error_source: MetadataRawSource | None = None
            if sidecar_bytes is not None:
                try:
                    sidecar_result = parse_sidecar_metadata(
                        sidecar_bytes,
                        source_image_sha256=imported.source_image_sha256,
                        max_raw_bytes=self._settings.max_metadata_sidecar_bytes,
                    )
                except SidecarMetadataError as exc:
                    warnings.append(exc.code)
                    sidecar_error_source = _sidecar_error_source(exc, sidecar_bytes)
            if sidecar_result is not None:
                raw_sources.append(sidecar_result.raw_source)
                warnings.extend(sidecar_result.warnings)
            elif sidecar_error_source is not None:
                raw_sources.append(sidecar_error_source)

            candidates: list[MetadataImportCandidate] = []
            if png_result.prompt is not None:
                parsed_prompt = parse_comfyui_prompt_metadata(png_result.prompt)
                candidates.append(parsed_prompt.candidate)
                warnings.extend(parsed_prompt.warnings)
            if sidecar_result is not None:
                candidates.append(sidecar_result.candidate)
                warnings.extend(sidecar_result.warnings)

            candidate, source_kind = _select_candidate(candidates, warnings)
            model_warnings: tuple[str, ...] = ()
            if candidate is not None:
                candidate, model_warnings = self._check_models(candidate)
            warnings.extend(model_warnings)
            candidate_warnings = candidate.warnings if candidate is not None else ()
            all_warnings = tuple(dict.fromkeys((*warnings, *candidate_warnings)))
            status = _status_for(candidate, all_warnings)
            selection_blocked = bool(
                set(all_warnings)
                & (_INVALID_METADATA_WARNINGS | {"metadata_import_sidecar_hash_mismatch"})
            )
            record = MetadataImportRecord(
                id=imported.id,
                imported_image=imported,
                metadata_source=source_kind,
                selected_metadata_source=(
                    source_kind if candidate is not None and not selection_blocked else None
                ),
                metadata_status=status,
                raw_sources=tuple(raw_sources),
                candidate=candidate,
                candidates=tuple(candidates),
                warnings=all_warnings,
                created_at=imported.created_at,
                updated_at=datetime.now(UTC),
            )
            self._repository.create(record)
            committed = True
            return _preview(record)
        except SidecarMetadataError as exc:
            raise MetadataImportError(exc.code, "sidecar metadata could not be parsed") from exc
        except MetadataImportError:
            raise
        except Exception as exc:  # noqa: BLE001 - cleanup boundary for all post-store failures
            logger.exception("Metadata import failed after canonical storage")
            raise MetadataImportError(
                getattr(exc, "code", "metadata_import_parse_failed"),
                "metadata import could not be persisted",
            ) from exc
        finally:
            if not committed:
                self._cleanup_uncommitted_if_unpersisted(imported)

    def get_preview(self, import_id: UUID) -> MetadataImportPreview:
        record = self._record(import_id)
        return _preview(record)

    def select_metadata_source(
        self,
        import_id: UUID,
        source_kind: MetadataSourceKind | str,
        *,
        confirm_sidecar_hash_mismatch: bool = False,
    ) -> MetadataImportPreview:
        """Persist an explicit PNG/sidecar choice before execution."""

        record = self._record(import_id)
        try:
            selected = (
                source_kind
                if isinstance(source_kind, MetadataSourceKind)
                else MetadataSourceKind(source_kind)
            )
        except ValueError as exc:
            raise MetadataImportError(
                "metadata_import_source_invalid", "metadata source selection is invalid"
            ) from exc
        candidate = _candidate_for_source(record, selected)
        if candidate is None:
            raise MetadataImportError(
                "metadata_import_source_invalid", "selected metadata source is unavailable"
            )
        hash_mismatch = "metadata_import_sidecar_hash_mismatch" in record.warnings
        if (
            selected is MetadataSourceKind.APP_SIDECAR
            and hash_mismatch
            and not confirm_sidecar_hash_mismatch
        ):
            raise MetadataImportError(
                "metadata_import_sidecar_hash_confirmation_required",
                "sidecar image hash mismatch requires explicit confirmation",
            )
        candidate, model_warnings = self._check_models(candidate)
        candidate = candidate.model_copy(
            update={
                "warnings": tuple(
                    warning
                    for warning in candidate.warnings
                    if warning != "metadata_import_sidecar_hash_mismatch"
                )
            }
        )
        retained = tuple(
            warning
            for warning in record.warnings
            if warning
            not in {
                "metadata_import_ambiguous",
                "metadata_import_sidecar_hash_mismatch",
                "metadata_import_sidecar_hash_mismatch_confirmed",
                "metadata_import_sidecar_hash_mismatch_ignored",
                "metadata_import_model_missing",
                "metadata_import_model_catalog_unavailable",
            }
        )
        if selected is MetadataSourceKind.APP_SIDECAR and hash_mismatch:
            retained = (*retained, "metadata_import_sidecar_hash_mismatch_confirmed")
        elif selected is MetadataSourceKind.COMFYUI_PROMPT and hash_mismatch:
            retained = (*retained, "metadata_import_sidecar_hash_mismatch_ignored")
        warnings = tuple(
            dict.fromkeys(
                (
                    *retained,
                    *model_warnings,
                    *tuple(
                        warning
                        for warning in candidate.warnings
                        if warning != "metadata_import_sidecar_hash_mismatch"
                    ),
                )
            )
        )
        status = _status_for(candidate, warnings)
        updated = _with_normalized_snapshot(
            record.model_copy(
                update={
                    "metadata_source": selected,
                    "selected_metadata_source": selected,
                    "candidate": candidate,
                    "candidates": _replace_candidate(record.candidates, candidate),
                    "metadata_status": status,
                    "sidecar_hash_confirmed": (
                        confirm_sidecar_hash_mismatch
                        if selected is MetadataSourceKind.APP_SIDECAR and hash_mismatch
                        else record.sidecar_hash_confirmed
                    ),
                    "warnings": warnings,
                    "updated_at": datetime.now(UTC),
                }
            )
        )
        self._repository.save(updated)
        return _preview(updated)

    def apply_model_mapping(
        self,
        import_id: UUID,
        mappings: Sequence[MetadataModelMapping],
    ) -> MetadataImportPreview:
        record = self._record(import_id)
        candidate = record.candidate
        if candidate is None:
            raise MetadataImportError(
                "metadata_import_mapping_invalid",
                "select a metadata source before applying a mapping",
            )
        mapping_tuple = tuple(mappings)
        _validate_mapping_targets(mapping_tuple)
        candidate = candidate.with_mappings(mapping_tuple)
        candidate, model_warnings = self._check_models(candidate)
        candidate = candidate.model_copy(
            update={
                "warnings": tuple(
                    warning
                    for warning in candidate.warnings
                    if warning != "metadata_import_sidecar_hash_mismatch"
                )
            }
        )
        retained_warnings = tuple(
            warning
            for warning in record.warnings
            if warning
            not in {
                "metadata_import_model_missing",
                "metadata_import_model_catalog_unavailable",
            }
        )
        warnings = tuple(
            dict.fromkeys(
                (
                    *retained_warnings,
                    *model_warnings,
                    *tuple(
                        warning
                        for warning in candidate.warnings
                        if warning != "metadata_import_sidecar_hash_mismatch"
                    ),
                )
            )
        )
        status = _status_for(candidate, warnings)
        normalized_json: str | None = None
        snapshot_version: int | None = None
        if status is MetadataImportStatus.READY:
            settings = candidate.to_generation_settings()
            snapshot = GenerationSettingsSnapshot.from_settings(settings)
            normalized_json = snapshot.to_json()
            snapshot_version = snapshot.schema_version
        updated = record.model_copy(
            update={
                "candidate": candidate,
                "candidates": _replace_candidate(record.candidates, candidate),
                "metadata_status": status,
                "manual_mappings": mapping_tuple,
                "warnings": warnings,
                "normalized_snapshot_json": normalized_json,
                "normalized_snapshot_schema_version": snapshot_version,
                "updated_at": datetime.now(UTC),
            }
        )
        self._repository.save(updated)
        return _preview(updated)

    def build_generation_settings(self, import_id: UUID) -> GenerationSettings:
        record = self._record(import_id)
        if record.candidate is None:
            raise MetadataImportError(
                "metadata_import_unresolved",
                "metadata must be confirmed and fully resolved before execution",
            )
        candidate, model_warnings = self._check_models(record.candidate)
        blocking = (
            (set(record.warnings) - _CAPABILITY_WARNINGS)
            | set(model_warnings)
            | (set(candidate.warnings) if candidate is not None else set())
        ) & _BLOCKING_WARNINGS
        if model_warnings or not candidate.is_generation_ready:
            blocking.add("metadata_import_unresolved")
        if blocking:
            raise MetadataImportError(
                "metadata_import_unresolved", "metadata contains blocking warnings"
            )
        settings = candidate.to_generation_settings()
        # The import preview is never allowed to choose a workflow template.
        return settings.model_copy(
            update={"workflow_template_id": "sdxl_txt2img", "workflow_template_version": "1.0"}
        )

    def get_upscale_source(self, import_id: UUID) -> ImportedImage:
        record = self._record(import_id)
        self._storage.verify(record.imported_image)
        return record.imported_image

    def get_upscale_source_path(self, import_id: UUID) -> Path:
        record = self._record(import_id)
        return self._storage.absolute_path(record.imported_image)

    def set_capabilities(self, capabilities: ComfyUICapabilities | None) -> None:
        self._capabilities = capabilities

    def _record(self, import_id: UUID) -> MetadataImportRecord:
        try:
            record = self._repository.get_by_id(import_id)
        except MetadataImportRepositoryError as exc:
            raise MetadataImportError(
                "metadata_import_parse_failed", "import record unavailable"
            ) from exc
        if record is None:
            raise MetadataImportError("metadata_import_parse_failed", "import record not found")
        return record

    def _cleanup_uncommitted_if_unpersisted(self, imported: ImportedImage) -> None:
        """Delete a canonical file only when the import row is known absent."""

        try:
            if self._repository.get_by_id(imported.id) is not None:
                return
        except Exception:  # noqa: BLE001 - unknown DB state must retain the file
            logger.warning(
                "Metadata import cleanup deferred because row state is unavailable import=%s",
                imported.id,
            )
            return
        self._storage.cleanup_uncommitted(imported)

    def _check_models(
        self,
        candidate: MetadataImportCandidate,
    ) -> tuple[MetadataImportCandidate, tuple[str, ...]]:
        capabilities = self._capabilities
        if capabilities is None:
            return candidate, ("metadata_import_model_catalog_unavailable",)
        warnings: list[str] = []
        if (
            candidate.checkpoint_name is not None
            and candidate.checkpoint_name not in capabilities.checkpoints
        ):
            warnings.append("metadata_import_model_missing")
        if candidate.vae_name is not None and candidate.vae_name not in capabilities.vaes:
            warnings.append("metadata_import_model_missing")
        if any(lora.name not in capabilities.loras for lora in candidate.loras):
            warnings.append("metadata_import_model_missing")
        if (
            candidate.sampler_name is not None
            and candidate.sampler_name not in capabilities.samplers
        ):
            warnings.append("metadata_import_model_missing")
        if (
            candidate.scheduler_name is not None
            and candidate.scheduler_name not in capabilities.schedulers
        ):
            warnings.append("metadata_import_model_missing")
        return candidate, tuple(dict.fromkeys(warnings))


def _select_candidate(
    candidates: Sequence[MetadataImportCandidate],
    warnings: list[str],
) -> tuple[MetadataImportCandidate | None, MetadataSourceKind]:
    if not candidates:
        return None, MetadataSourceKind.NONE
    png_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.source_kind is MetadataSourceKind.COMFYUI_PROMPT
        ),
        None,
    )
    sidecar_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.source_kind is MetadataSourceKind.APP_SIDECAR
        ),
        None,
    )
    if png_candidate is not None and sidecar_candidate is not None:
        if _candidate_values(png_candidate) != _candidate_values(sidecar_candidate):
            warnings.append("metadata_import_ambiguous")
            return None, MetadataSourceKind.NONE
        return png_candidate, MetadataSourceKind.COMFYUI_PROMPT
    candidate = png_candidate or sidecar_candidate or candidates[0]
    return candidate, candidate.source_kind


def _same_import_inputs(
    record: MetadataImportRecord,
    sidecar_bytes: bytes | str | bytearray | None,
) -> bool:
    existing_sidecars = tuple(
        source for source in record.raw_sources if source.kind is MetadataSourceKind.APP_SIDECAR
    )
    if sidecar_bytes is None:
        return not existing_sidecars
    raw_bytes = (
        sidecar_bytes.encode("utf-8") if isinstance(sidecar_bytes, str) else bytes(sidecar_bytes)
    )
    if len(existing_sidecars) != 1:
        return False
    return existing_sidecars[0].sha256 == hashlib.sha256(raw_bytes).hexdigest()


def _sidecar_error_source(
    error: SidecarMetadataError,
    payload: bytes | str | bytearray,
) -> MetadataRawSource:
    raw_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    return MetadataRawSource(
        kind=MetadataSourceKind.APP_SIDECAR,
        raw_text=error.raw_text,
        sha256=error.raw_sha256 or hashlib.sha256(raw_bytes).hexdigest(),
    )


def _candidate_for_source(
    record: MetadataImportRecord,
    source_kind: MetadataSourceKind,
) -> MetadataImportCandidate | None:
    candidates = record.candidates or ((record.candidate,) if record.candidate is not None else ())
    return next(
        (candidate for candidate in candidates if candidate.source_kind is source_kind),
        None,
    )


def _replace_candidate(
    candidates: Sequence[MetadataImportCandidate],
    replacement: MetadataImportCandidate,
) -> tuple[MetadataImportCandidate, ...]:
    if not candidates:
        return (replacement,)
    return tuple(
        replacement if candidate.source_kind is replacement.source_kind else candidate
        for candidate in candidates
    )


def _with_normalized_snapshot(record: MetadataImportRecord) -> MetadataImportRecord:
    if record.metadata_status is not MetadataImportStatus.READY or record.candidate is None:
        return record.model_copy(
            update={
                "normalized_snapshot_json": None,
                "normalized_snapshot_schema_version": None,
            }
        )
    settings = record.candidate.to_generation_settings()
    snapshot = GenerationSettingsSnapshot.from_settings(settings)
    return record.model_copy(
        update={
            "normalized_snapshot_json": snapshot.to_json(),
            "normalized_snapshot_schema_version": snapshot.schema_version,
        }
    )


def _candidate_values(candidate: MetadataImportCandidate) -> tuple[object, ...]:
    return (
        candidate.positive_prompt,
        candidate.negative_prompt,
        candidate.seed,
        candidate.width,
        candidate.height,
        candidate.steps,
        candidate.cfg_scale,
        candidate.sampler_name,
        candidate.scheduler_name,
        candidate.checkpoint_name,
        candidate.vae_name,
        tuple(lora.model_dump(mode="json") for lora in candidate.loras),
    )


def _status_for(
    candidate: MetadataImportCandidate | None,
    warnings: Sequence[str],
) -> MetadataImportStatus:
    if set(warnings) & _INVALID_METADATA_WARNINGS:
        return MetadataImportStatus.INVALID_METADATA
    if candidate is None:
        if "metadata_import_ambiguous" in warnings:
            return MetadataImportStatus.NEEDS_MAPPING
        return MetadataImportStatus.METADATA_MISSING
    if (
        "metadata_import_parse_failed" in warnings or "metadata_import_invalid_json" in warnings
    ) and not candidate.is_generation_ready:
        return MetadataImportStatus.INVALID_METADATA
    if not candidate.is_generation_ready or set(warnings) & _BLOCKING_WARNINGS:
        return MetadataImportStatus.NEEDS_MAPPING
    return MetadataImportStatus.READY


def _validate_mapping_targets(mappings: Sequence[MetadataModelMapping]) -> None:
    allowed = {"checkpoint", "vae", "lora"}
    for mapping in mappings:
        if mapping.model_kind not in allowed:
            raise MetadataImportError(
                "metadata_import_mapping_invalid", "model mapping kind is invalid"
            )
        if not _is_safe_relative_model_name(
            mapping.target_name
        ) or not _is_safe_relative_model_name(mapping.source_name):
            raise MetadataImportError(
                "metadata_import_mapping_invalid", "model mapping target is invalid"
            )


def _is_safe_relative_model_name(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    return (
        bool(normalized)
        and "\x00" not in normalized
        and not (
            posixpath.isabs(normalized)
            or ntpath.isabs(normalized)
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
        )
    )


def _preview(record: MetadataImportRecord) -> MetadataImportPreview:
    return MetadataImportPreview(
        id=record.id,
        imported_image=record.imported_image,
        status=record.metadata_status,
        metadata_source=record.metadata_source,
        candidate=record.candidate,
        candidates=record.candidates,
        selected_metadata_source=record.selected_metadata_source,
        sidecar_hash_confirmed=record.sidecar_hash_confirmed,
        raw_sources=record.raw_sources,
        warnings=record.warnings,
        unresolved_fields=(record.candidate.unresolved_fields if record.candidate else ()),
        model_mappings=record.manual_mappings,
        created_at=record.created_at,
    )


__all__ = ["MetadataImportService"]
