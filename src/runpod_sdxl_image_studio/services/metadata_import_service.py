"""Application service coordinating safe metadata import and preview decisions."""

from __future__ import annotations

import logging
import ntpath
import posixpath
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.database.repositories.metadata_import_repository import (
    MetadataImportRepositoryError,
    MetadataImportRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.metadata.comfyui_prompt_metadata_adapter import (
    parse_comfyui_prompt_metadata,
)
from runpod_sdxl_image_studio.adapters.metadata.png_metadata_adapter import (
    PngMetadataResult,
    parse_png_metadata,
)
from runpod_sdxl_image_studio.adapters.metadata.sidecar_metadata_adapter import (
    SidecarMetadataError,
    SidecarMetadataResult,
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

_BLOCKING_WARNINGS = frozenset(
    {
        "metadata_import_sidecar_hash_mismatch",
        "metadata_import_ambiguous",
        "metadata_import_model_missing",
        "metadata_import_model_catalog_unavailable",
        "metadata_import_mapping_invalid",
        "metadata_import_unresolved",
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

    def import_image(
        self,
        image_bytes: bytes,
        original_filename: str | None = None,
        *,
        sidecar_bytes: bytes | str | bytearray | None = None,
    ) -> MetadataImportPreview:
        imported = self._storage.store(image_bytes, original_filename)
        png_result: PngMetadataResult | None = None
        sidecar_result: SidecarMetadataResult | None = None
        warnings: list[str] = []
        raw_sources: list[MetadataRawSource] = []
        try:
            png_result = parse_png_metadata(
                image_bytes,
                max_raw_bytes=self._settings.max_metadata_raw_bytes,
            )
            raw_sources.extend(png_result.raw_sources)
            warnings.extend(png_result.warnings)
        except ValueError:
            warnings.append("metadata_import_parse_failed")
        if sidecar_bytes is not None:
            try:
                sidecar_result = parse_sidecar_metadata(
                    sidecar_bytes,
                    source_image_sha256=imported.source_image_sha256,
                    max_raw_bytes=self._settings.max_metadata_sidecar_bytes,
                )
                raw_sources.append(sidecar_result.raw_source)
                warnings.extend(sidecar_result.warnings)
            except SidecarMetadataError as exc:
                warnings.append(exc.code)

        candidates: list[MetadataImportCandidate] = []
        if png_result is not None and png_result.prompt is not None:
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
        record = MetadataImportRecord(
            id=imported.id,
            imported_image=imported,
            metadata_source=source_kind,
            metadata_status=status,
            raw_sources=tuple(raw_sources),
            candidate=candidate,
            warnings=all_warnings,
            created_at=imported.created_at,
            updated_at=datetime.now(UTC),
        )
        self._repository.create(record)
        return _preview(record)

    def get_preview(self, import_id: UUID) -> MetadataImportPreview:
        record = self._record(import_id)
        return _preview(record)

    def apply_model_mapping(
        self,
        import_id: UUID,
        mappings: Sequence[MetadataModelMapping],
    ) -> MetadataImportPreview:
        record = self._record(import_id)
        if record.candidate is None:
            raise MetadataImportError(
                "metadata_import_mapping_invalid", "metadata candidate is absent"
            )
        mapping_tuple = tuple(mappings)
        _validate_mapping_targets(mapping_tuple)
        candidate = record.candidate.with_mappings(mapping_tuple)
        candidate, model_warnings = self._check_models(candidate)
        retained_warnings = tuple(
            warning
            for warning in record.warnings
            if warning
            not in {
                "metadata_import_model_missing",
                "metadata_import_model_catalog_unavailable",
            }
        )
        warnings = tuple(dict.fromkeys((*retained_warnings, *model_warnings, *candidate.warnings)))
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
        if record.metadata_status is not MetadataImportStatus.READY or record.candidate is None:
            raise MetadataImportError(
                "metadata_import_unresolved",
                "metadata must be confirmed and fully resolved before execution",
            )
        blocking = set(record.warnings) & _BLOCKING_WARNINGS
        if blocking:
            raise MetadataImportError(
                "metadata_import_unresolved", "metadata contains blocking warnings"
            )
        settings = record.candidate.to_generation_settings()
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
        if not png_candidate.is_generation_ready and sidecar_candidate.is_generation_ready:
            return sidecar_candidate, MetadataSourceKind.APP_SIDECAR
        if _candidate_values(png_candidate) != _candidate_values(sidecar_candidate):
            warnings.append("metadata_import_ambiguous")
        return png_candidate, MetadataSourceKind.COMFYUI_PROMPT
    candidate = png_candidate or sidecar_candidate or candidates[0]
    return candidate, candidate.source_kind


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
    if candidate is None:
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
        raw_sources=record.raw_sources,
        warnings=record.warnings,
        unresolved_fields=(record.candidate.unresolved_fields if record.candidate else ()),
        model_mappings=record.manual_mappings,
        created_at=record.created_at,
    )


__all__ = ["MetadataImportService"]
