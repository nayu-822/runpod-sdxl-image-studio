"""Application service for validating and enqueuing a reproducible upscale."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from uuid import UUID
from warnings import catch_warnings, simplefilter

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryError,
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationRepositoryError,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.metadata_import_repository import (
    MetadataImportRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepositoryError,
    UpscaleSettingsRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.storage.imported_image_storage import ImportedImageStorage
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_artifact import GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationHistoryQuery,
    GenerationHistorySort,
)
from runpod_sdxl_image_studio.domain.generation_queue import GenerationQueueItem
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.metadata_import import (
    ImportedImage,
    MetadataImportCandidate,
    MetadataImportError,
)
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleLoadLevel,
    UpscaleSettings,
    estimate_load_level,
    resolve_output_size,
    validate_upscaler_name,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import (
    UpscaleSettingsSnapshot,
    UpscaleSourceKind,
)


class UpscaleEnqueueError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_CAPABILITY_WARNINGS = frozenset(
    {
        "metadata_import_model_missing",
        "metadata_import_model_catalog_unavailable",
    }
)
_BLOCKING_METADATA_WARNINGS = frozenset(
    {
        "metadata_import_sidecar_hash_mismatch",
        "metadata_import_ambiguous",
        "metadata_import_model_missing",
        "metadata_import_model_catalog_unavailable",
        "metadata_import_mapping_invalid",
        "metadata_import_unresolved",
        "metadata_import_invalid_json",
        "metadata_import_invalid_utf8",
        "metadata_import_unsupported_schema",
        "metadata_import_too_large",
        "metadata_import_parse_failed",
        "metadata_import_png_prompt_invalid",
    }
)


@dataclass(frozen=True)
class VerifiedUpscaleSource:
    artifact: GenerationArtifact
    path: Path
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class UpscalePlan:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    load_level: UpscaleLoadLevel


@dataclass(frozen=True)
class UpscaleParentSelection:
    generation_id: UUID
    preview_path: Path


@dataclass(frozen=True)
class UpscaleComparison:
    parent_generation_id: UUID | None
    result_generation_id: UUID
    result_path: Path
    gallery: tuple[tuple[str, str], ...]


class UpscaleEnqueueService:
    def __init__(
        self,
        generation_repository: GenerationRepositoryProtocol,
        artifact_repository: GenerationArtifactRepositoryProtocol,
        queue_repository: GenerationDispatchQueueRepositoryProtocol,
        settings: Settings,
        *,
        catalog: UpscalerCatalog | None = None,
        metadata_import_repository: MetadataImportRepositoryProtocol | None = None,
        imported_image_storage: ImportedImageStorage | None = None,
        upscale_settings_repository: UpscaleSettingsRepositoryProtocol | None = None,
        capabilities: ComfyUICapabilities | None = None,
    ) -> None:
        self._generations = generation_repository
        self._artifacts = artifact_repository
        self._queue = queue_repository
        self._settings = settings
        self._catalog = catalog or UpscalerCatalog.scan(settings.upscaler_dir)
        self._metadata_import_repository = metadata_import_repository
        self._imported_image_storage = imported_image_storage
        self._upscale_settings_repository = upscale_settings_repository
        self._capabilities = capabilities

    def set_capabilities(self, capabilities: ComfyUICapabilities | None) -> None:
        self._capabilities = capabilities

    def latest_completed_generation_id(self) -> UUID | None:
        page = self._generations.list_history(
            GenerationHistoryQuery(
                statuses=(GenerationStatus.COMPLETED,),
                sort=GenerationHistorySort.RECENTLY_COMPLETED,
                page_size=20,
            )
        )
        for generation in page.generations:
            if self._artifacts.get_primary_image(generation.id) is not None:
                return generation.id
        return None

    def select_parent(self, generation_id: UUID) -> UpscaleParentSelection:
        """Resolve a completed parent and its persisted image for UI preview."""

        try:
            generation = self._generations.get_by_id(generation_id)
            if generation is None:
                raise UpscaleEnqueueError(
                    "upscale_parent_not_found", "the selected Generation was not found"
                )
            if generation.status is not GenerationStatus.COMPLETED:
                raise UpscaleEnqueueError(
                    "upscale_parent_not_completed", "only completed Generations can be selected"
                )
            artifact = self._artifacts.get_primary_image(generation_id)
            if artifact is None:
                raise UpscaleEnqueueError(
                    "upscale_source_artifact_missing", "the selected Generation has no image"
                )
            source = verify_source_artifact(artifact, self._settings)
            return UpscaleParentSelection(generation_id, source.path)
        except UpscaleEnqueueError:
            raise
        except (GenerationRepositoryError, ValueError, OSError) as exc:
            raise UpscaleEnqueueError(
                "upscale_parent_unavailable", "the selected parent could not be read"
            ) from exc

    def get_parent_generation_settings(self, generation_id: UUID) -> GenerationSettings:
        """Read the persisted parent snapshot required by latent-upscale preflight."""

        try:
            generation = self._generations.get_by_id(generation_id)
            if generation is None:
                raise UpscaleEnqueueError(
                    "upscale_parent_not_found", "the selected Generation was not found"
                )
            if generation.status is not GenerationStatus.COMPLETED:
                raise UpscaleEnqueueError(
                    "upscale_parent_not_completed",
                    "only completed Generations can be used for latent upscale",
                )
            return generation.settings_snapshot.to_generation_settings()
        except UpscaleEnqueueError:
            raise
        except (GenerationRepositoryError, ValueError, OSError) as exc:
            raise UpscaleEnqueueError(
                "upscale_parent_unavailable",
                "the parent Generation snapshot could not be read",
            ) from exc

    def get_import_generation_settings(self, import_id: UUID) -> GenerationSettings:
        """Read the validated imported snapshot used by latent-upscale preflight."""

        if self._metadata_import_repository is None:
            raise UpscaleEnqueueError(
                "metadata_import_unavailable", "the metadata import is unavailable"
            )
        try:
            record = self._metadata_import_repository.get_by_id(import_id)
            if (
                record is None
                or record.candidate is None
                or not record.candidate.is_generation_ready
            ):
                raise UpscaleEnqueueError(
                    "metadata_import_unresolved",
                    "latent upscale requires complete imported metadata",
                )
            return record.candidate.to_generation_settings()
        except UpscaleEnqueueError:
            raise
        except (MetadataImportError, ValueError, OSError) as exc:
            raise UpscaleEnqueueError(
                "metadata_import_unresolved",
                "latent upscale metadata could not be read",
            ) from exc

    def comparison_for_generation(self, generation_id: UUID) -> UpscaleComparison:
        """Return two persisted files only for a completed upscale Generation."""

        try:
            generation = self._generations.get_by_id(generation_id)
            if generation is None or generation.kind is not GenerationKind.UPSCALE:
                raise UpscaleEnqueueError(
                    "upscale_result_unavailable", "the selected Generation is not an upscale"
                )
            if generation.status is not GenerationStatus.COMPLETED:
                raise UpscaleEnqueueError(
                    "upscale_result_not_completed", "the upscale result is not completed"
                )
            result_artifact = self._artifacts.get_primary_image(generation_id)
            if result_artifact is None:
                raise UpscaleEnqueueError(
                    "upscale_result_artifact_missing", "the comparison images are unavailable"
                )
            result = verify_source_artifact(result_artifact, self._settings)
            if self._upscale_settings_repository is not None:
                snapshot = self._upscale_settings_repository.get_by_generation(generation_id)
            else:
                snapshot = None
            if snapshot is not None and snapshot.source_kind is UpscaleSourceKind.METADATA_IMPORT:
                if (
                    snapshot.source_import_id is None
                    or self._metadata_import_repository is None
                    or self._imported_image_storage is None
                ):
                    raise UpscaleEnqueueError(
                        "upscale_result_unavailable", "the metadata import source is unavailable"
                    )
                imported = self._metadata_import_repository.get_by_id(snapshot.source_import_id)
                if imported is None:
                    raise UpscaleEnqueueError(
                        "upscale_result_unavailable", "the metadata import source was not found"
                    )
                source_path = self._imported_image_storage.absolute_path(imported.imported_image)
                return UpscaleComparison(
                    None,
                    generation_id,
                    result.path,
                    ((str(source_path), "source"), (str(result.path), "upscaled")),
                )
            if generation.parent_generation_id is None:
                raise UpscaleEnqueueError(
                    "upscale_parent_not_found", "the upscale parent was not found"
                )
            parent_artifact = self._artifacts.get_primary_image(generation.parent_generation_id)
            if parent_artifact is None:
                raise UpscaleEnqueueError(
                    "upscale_result_artifact_missing", "the comparison images are unavailable"
                )
            parent = verify_source_artifact(parent_artifact, self._settings)
            return UpscaleComparison(
                generation.parent_generation_id,
                generation_id,
                result.path,
                ((str(parent.path), "parent"), (str(result.path), "upscaled")),
            )
        except UpscaleEnqueueError:
            raise
        except (
            GenerationRepositoryError,
            UpscaleSettingsRepositoryError,
            ValueError,
            OSError,
        ) as exc:
            raise UpscaleEnqueueError(
                "upscale_result_unavailable", "the comparison images could not be read"
            ) from exc

    def plan(self, parent_generation_id: UUID, upscale_settings: UpscaleSettings) -> UpscalePlan:
        parent = self._generations.get_by_id(parent_generation_id)
        if parent is None:
            raise UpscaleEnqueueError("upscale_parent_not_found", "親Generationが見つかりません。")
        if parent.status is not GenerationStatus.COMPLETED:
            raise UpscaleEnqueueError(
                "upscale_parent_not_completed", "完了済みGenerationだけ指定できます。"
            )
        artifact = self._artifacts.get_primary_image(parent_generation_id)
        if artifact is None:
            raise UpscaleEnqueueError(
                "upscale_source_artifact_missing", "親の一次画像Artifactがありません。"
            )
        source = verify_source_artifact(artifact, self._settings)
        if upscale_settings.method.value == "image":
            model_name = validate_upscaler_name(upscale_settings.upscaler_name)
            if not self._catalog.contains(model_name):
                raise UpscaleEnqueueError(
                    "upscale_model_missing", "指定されたupscalerが取得済みではありません。"
                )
        try:
            size = resolve_output_size(
                upscale_settings,
                source.width,
                source.height,
                max_width=self._settings.max_width,
                max_height=self._settings.max_height,
                max_pixels=self._settings.max_pixels,
                max_upscale_factor=self._settings.max_upscale_factor,
            )
        except ValueError as exc:
            raise UpscaleEnqueueError("upscale_limit_exceeded", str(exc)) from exc
        return UpscalePlan(
            source.width,
            source.height,
            size.width,
            size.height,
            estimate_load_level(
                upscale_settings.method,
                source.width,
                source.height,
                size.width,
                size.height,
            ),
        )

    def enqueue(
        self, parent_generation_id: UUID, upscale_settings: UpscaleSettings
    ) -> GenerationQueueItem:
        parent = self._generations.get_by_id(parent_generation_id)
        if parent is None:
            raise UpscaleEnqueueError("upscale_parent_not_found", "親Generationが見つかりません。")
        if parent.status is not GenerationStatus.COMPLETED:
            raise UpscaleEnqueueError(
                "upscale_parent_not_completed", "完了済みGenerationだけ指定できます。"
            )
        artifact = self._artifacts.get_primary_image(parent_generation_id)
        if artifact is None:
            raise UpscaleEnqueueError(
                "upscale_source_artifact_missing", "親の一次画像Artifactがありません。"
            )
        source = verify_source_artifact(artifact, self._settings)
        if upscale_settings.method.value == "image":
            model_name = validate_upscaler_name(upscale_settings.upscaler_name)
            if not self._catalog.contains(model_name):
                raise UpscaleEnqueueError(
                    "upscale_model_missing", "指定されたupscalerが取得済みではありません。"
                )
        try:
            size = resolve_output_size(
                upscale_settings,
                source.width,
                source.height,
                max_width=self._settings.max_width,
                max_height=self._settings.max_height,
                max_pixels=self._settings.max_pixels,
                max_upscale_factor=self._settings.max_upscale_factor,
            )
        except ValueError as exc:
            raise UpscaleEnqueueError("upscale_limit_exceeded", str(exc)) from exc
        workflow_id = (
            "sdxl_image_upscale"
            if upscale_settings.method.value == "image"
            else "sdxl_latent_upscale"
        )
        effective_upscale = upscale_settings.model_copy(
            update={"workflow_template_id": workflow_id, "workflow_template_version": "1.0"}
        )
        upscale_snapshot = UpscaleSettingsSnapshot.from_settings(
            effective_upscale,
            source_generation_id=parent_generation_id,
            source_artifact_id=source.artifact.id,
            source_sha256=source.sha256,
            source_width=source.width,
            source_height=source.height,
            target_width=size.width,
            target_height=size.height,
        )
        generation_snapshot = parent.settings_snapshot.model_copy(
            update={
                "width": size.width,
                "height": size.height,
                "workflow_template_id": workflow_id,
                "workflow_template_version": "1.0",
            }
        )
        try:
            return self._queue.enqueue_upscale(
                generation_snapshot,
                upscale_snapshot,
                parent_generation_id=parent_generation_id,
                source_artifact_id=source.artifact.id,
                pending_limit=self._settings.queue_max_pending_jobs,
            )
        except GenerationDispatchQueueRepositoryError as exc:
            raise UpscaleEnqueueError("upscale_workflow_error", str(exc)) from exc

    enqueue_upscale = enqueue

    def plan_import(self, import_id: UUID, upscale_settings: UpscaleSettings) -> UpscalePlan:
        imported = self._verified_import(import_id)
        self._validate_import_method(imported.id, upscale_settings)
        return self._plan_for_dimensions(
            upscale_settings, imported.image_width, imported.image_height
        )

    def enqueue_import(
        self, import_id: UUID, upscale_settings: UpscaleSettings
    ) -> GenerationQueueItem:
        imported = self._verified_import(import_id)
        self._validate_import_method(imported.id, upscale_settings)
        size = self._plan_for_dimensions(
            upscale_settings, imported.image_width, imported.image_height
        )
        workflow_id = (
            "sdxl_image_upscale"
            if upscale_settings.method.value == "image"
            else "sdxl_latent_upscale"
        )
        effective_upscale = upscale_settings.model_copy(
            update={"workflow_template_id": workflow_id, "workflow_template_version": "1.0"}
        )
        if upscale_settings.method.value == "latent":
            record = self._metadata_import_repository.get_by_id(import_id)  # type: ignore[union-attr]
            if (
                record is None
                or record.candidate is None
                or not record.candidate.is_generation_ready
            ):
                raise UpscaleEnqueueError(
                    "metadata_import_unresolved",
                    "latent upscale requires complete imported metadata",
                )
            if (set(record.warnings) - _CAPABILITY_WARNINGS) & _BLOCKING_METADATA_WARNINGS:
                raise UpscaleEnqueueError(
                    "metadata_import_unresolved",
                    "latent upscale metadata has blocking warnings",
                )
            self._validate_import_capabilities(record.candidate)
            try:
                source_settings = record.candidate.to_generation_settings()
            except MetadataImportError as exc:
                raise UpscaleEnqueueError(exc.code, str(exc)) from exc
        else:
            source_settings = GenerationSettings(
                positive_prompt="",
                negative_prompt="",
                seed=0,
                width=size.source_width,
                height=size.source_height,
                steps=1,
                cfg_scale=0.0,
                sampler_name="euler",
                scheduler_name="normal",
                checkpoint_name="__metadata_import_image_source__",
            )
        upscale_snapshot = UpscaleSettingsSnapshot.from_settings(
            effective_upscale,
            source_kind=UpscaleSourceKind.METADATA_IMPORT,
            source_import_id=import_id,
            source_sha256=imported.stored_image_sha256,
            source_width=imported.image_width,
            source_height=imported.image_height,
            target_width=size.target_width,
            target_height=size.target_height,
        )
        generation_snapshot = GenerationSettingsSnapshot.from_settings(
            source_settings.model_copy(
                update={
                    "width": size.target_width,
                    "height": size.target_height,
                    "workflow_template_id": workflow_id,
                    "workflow_template_version": "1.0",
                }
            )
        )
        try:
            return self._queue.enqueue_upscale(
                generation_snapshot,
                upscale_snapshot,
                parent_generation_id=None,
                source_artifact_id=None,
                source_import_id=import_id,
                pending_limit=self._settings.queue_max_pending_jobs,
            )
        except GenerationDispatchQueueRepositoryError as exc:
            raise UpscaleEnqueueError("upscale_workflow_error", str(exc)) from exc

    def _verified_import(self, import_id: UUID) -> ImportedImage:
        if self._metadata_import_repository is None or self._imported_image_storage is None:
            raise UpscaleEnqueueError(
                "metadata_import_unavailable", "metadata import source is unavailable"
            )
        imported_record = self._metadata_import_repository.get_by_id(import_id)
        if imported_record is None:
            raise UpscaleEnqueueError(
                "metadata_import_source_changed", "metadata import source was not found"
            )
        try:
            self._imported_image_storage.verify(imported_record.imported_image)
        except Exception as exc:  # noqa: BLE001 - stable enqueue error boundary
            code = getattr(exc, "code", "metadata_import_source_changed")
            raise UpscaleEnqueueError(code, "metadata import source could not be verified") from exc
        return imported_record.imported_image

    def _validate_import_method(self, import_id: UUID, upscale_settings: UpscaleSettings) -> None:
        if upscale_settings.method.value != "latent":
            return
        record = self._metadata_import_repository.get_by_id(import_id)  # type: ignore[union-attr]
        if record is None or record.candidate is None or not record.candidate.is_generation_ready:
            raise UpscaleEnqueueError(
                "metadata_import_unresolved",
                "latent upscale requires complete imported metadata",
            )
        if (set(record.warnings) - _CAPABILITY_WARNINGS) & _BLOCKING_METADATA_WARNINGS:
            raise UpscaleEnqueueError(
                "metadata_import_unresolved",
                "latent upscale metadata has blocking warnings",
            )
        self._validate_import_capabilities(record.candidate)

    def _validate_import_capabilities(self, candidate: MetadataImportCandidate) -> None:
        capabilities = self._capabilities
        if capabilities is None:
            raise UpscaleEnqueueError(
                "metadata_import_model_catalog_unavailable",
                "latent upscale capabilities are unavailable",
            )
        missing = []
        checkpoint = candidate.checkpoint_name
        if checkpoint not in capabilities.checkpoints:
            missing.append("checkpoint")
        vae = candidate.vae_name
        if vae is not None and vae not in capabilities.vaes:
            missing.append("vae")
        if any(lora.name not in capabilities.loras for lora in candidate.loras):
            missing.append("lora")
        if candidate.sampler_name not in capabilities.samplers:
            missing.append("sampler")
        if candidate.scheduler_name not in capabilities.schedulers:
            missing.append("scheduler")
        if missing:
            raise UpscaleEnqueueError(
                "metadata_import_model_missing",
                "latent upscale metadata references unavailable capabilities",
            )

    def _plan_for_dimensions(
        self, upscale_settings: UpscaleSettings, source_width: int, source_height: int
    ) -> UpscalePlan:
        if upscale_settings.method.value == "image":
            model_name = validate_upscaler_name(upscale_settings.upscaler_name)
            if not self._catalog.contains(model_name):
                raise UpscaleEnqueueError(
                    "upscale_model_missing", "指定されたupscalerが取得済みではありません。"
                )
        try:
            size = resolve_output_size(
                upscale_settings,
                source_width,
                source_height,
                max_width=self._settings.max_width,
                max_height=self._settings.max_height,
                max_pixels=self._settings.max_pixels,
                max_upscale_factor=self._settings.max_upscale_factor,
            )
        except ValueError as exc:
            raise UpscaleEnqueueError("upscale_limit_exceeded", str(exc)) from exc
        return UpscalePlan(
            source_width,
            source_height,
            size.width,
            size.height,
            estimate_load_level(
                upscale_settings.method,
                source_width,
                source_height,
                size.width,
                size.height,
            ),
        )


def verify_source_artifact(
    artifact: GenerationArtifact, settings: Settings
) -> VerifiedUpscaleSource:
    """Revalidate the file immediately before enqueue; this never mutates the source."""

    relative = PurePosixPath(artifact.local_path.replace("\\", "/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise UpscaleEnqueueError("upscale_source_invalid", "Artifact pathが安全ではありません。")
    data_root = settings.data_dir.resolve()
    path = data_root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(data_root)
    except (OSError, ValueError) as exc:
        raise UpscaleEnqueueError(
            "upscale_source_file_missing", "親画像ファイルがありません。"
        ) from exc
    if not resolved.is_file():
        raise UpscaleEnqueueError("upscale_source_file_missing", "親画像ファイルがありません。")
    try:
        if (
            resolved.stat().st_size <= 0
            or resolved.stat().st_size > settings.max_upscale_input_image_bytes
        ):
            raise UpscaleEnqueueError("upscale_source_invalid", "親画像サイズが許可範囲外です。")
        image_bytes = resolved.read_bytes()
        sha256 = hashlib.sha256(image_bytes).hexdigest()
        if sha256 != artifact.sha256:
            raise UpscaleEnqueueError(
                "upscale_source_changed", "親画像がArtifact登録後に変更されています。"
            )
        with catch_warnings():
            simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"PNG", "WEBP"}:
                    raise ValueError("unsupported source format")
                image.verify()
                actual_mime = "image/png" if image.format == "PNG" else "image/webp"
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
    except UpscaleEnqueueError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise UpscaleEnqueueError("upscale_source_invalid", "親画像を検証できません。") from exc
    if artifact.width != width or artifact.height != height:
        raise UpscaleEnqueueError(
            "upscale_source_changed", "親画像の寸法がArtifactと一致しません。"
        )
    if artifact.size_bytes != len(image_bytes) or artifact.mime_type != actual_mime:
        raise UpscaleEnqueueError(
            "upscale_source_changed", "親画像のArtifact情報が実体と一致しません。"
        )
    return VerifiedUpscaleSource(artifact, resolved, sha256, width, height)


__all__ = [
    "UpscaleEnqueueError",
    "UpscaleEnqueueService",
    "UpscaleComparison",
    "UpscalePlan",
    "UpscaleParentSelection",
    "VerifiedUpscaleSource",
    "verify_source_artifact",
]
