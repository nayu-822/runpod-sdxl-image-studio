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
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryError,
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_artifact import GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationHistoryQuery,
    GenerationHistorySort,
)
from runpod_sdxl_image_studio.domain.generation_queue import GenerationQueueItem
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleLoadLevel,
    UpscaleSettings,
    estimate_load_level,
    resolve_output_size,
    validate_upscaler_name,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSettingsSnapshot


class UpscaleEnqueueError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


class UpscaleEnqueueService:
    def __init__(
        self,
        generation_repository: GenerationRepositoryProtocol,
        artifact_repository: GenerationArtifactRepositoryProtocol,
        queue_repository: GenerationDispatchQueueRepositoryProtocol,
        settings: Settings,
        *,
        catalog: UpscalerCatalog | None = None,
    ) -> None:
        self._generations = generation_repository
        self._artifacts = artifact_repository
        self._queue = queue_repository
        self._settings = settings
        self._catalog = catalog or UpscalerCatalog.scan(settings.upscaler_dir)

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
    "UpscalePlan",
    "VerifiedUpscaleSource",
    "verify_source_artifact",
]
