"""Application service for durable, one-way Google Drive synchronization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import warnings
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Protocol
from uuid import UUID, uuid4

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveManifestRecord,
    DriveSyncRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveCacheCandidate,
    DriveCapacity,
    DriveConnectionResult,
    DriveRemotePaths,
    DriveSyncErrorCode,
    DriveSyncJob,
    DriveSyncProgress,
    DriveSyncRecord,
    DriveSyncStatus,
    build_remote_paths,
    utc,
    validate_remote_relative_path,
)
from runpod_sdxl_image_studio.domain.generation import Generation, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[DriveSyncProgress], Awaitable[None] | None]
MetadataRepairHandler = Callable[[UUID], object]


class DriveAdapterProtocol(Protocol):
    async def check_connection(self) -> DriveConnectionResult: ...

    async def copy_file(
        self,
        local_path: Path,
        relative_remote_path: str,
        *,
        progress_callback: ProgressCallback | None = None,
        total_bytes: int = 0,
    ) -> None: ...


class DriveSyncServiceError(RuntimeError):
    """Safe error returned by Drive synchronization operations."""

    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class DriveSyncService:
    """Coordinate source verification, copy ordering, and durable sync state."""

    def __init__(
        self,
        repository: DriveSyncRepositoryProtocol,
        generation_repository: GenerationRepositoryProtocol,
        artifact_repository: GenerationArtifactRepositoryProtocol,
        settings: Settings,
        adapter: DriveAdapterProtocol | None = None,
        *,
        rclone_adapter: DriveAdapterProtocol | None = None,
        metadata_repair_handler: MetadataRepairHandler | None = None,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if adapter is not None and rclone_adapter is not None:
            raise ValueError("adapter and rclone_adapter cannot both be configured")
        selected_adapter = adapter or rclone_adapter
        if selected_adapter is None:
            raise ValueError("a Drive adapter is required")
        self._repository = repository
        self._generation_repository = generation_repository
        self._artifact_repository = artifact_repository
        self._settings = settings
        self._adapter = selected_adapter
        self._metadata_repair_handler = metadata_repair_handler
        self._id_factory = id_factory

    async def check_connection(self) -> DriveConnectionResult:
        return await self._adapter.check_connection()

    @property
    def is_configured(self) -> bool:
        """Return only whether a remote name is configured, never the secret path."""

        return bool(self._settings.rclone_remote)

    def enqueue_generation(self, generation_id: UUID) -> DriveSyncRecord | None:
        """Create one persistent sync record after generation completion."""

        generation = self._get_completed_generation(generation_id)
        existing = self._repository.get_by_generation(generation_id)
        if existing is not None and existing.status in {
            DriveSyncStatus.PENDING,
            DriveSyncStatus.SYNCING,
            DriveSyncStatus.SYNCED,
        }:
            return existing
        artifacts = self._artifacts_with_repair(generation_id)
        image, metadata = _select_required_artifacts(artifacts)
        paths = build_remote_paths(
            generation.id,
            generation.kind.value,
            generation.created_at,
            timezone_name=self._settings.timezone,
        )
        record, job = self._build_record_and_job(
            generation,
            image,
            metadata,
            paths,
            existing=existing,
        )
        saved_record, saved_job = self._repository.enqueue(record, job)
        if saved_job is None:
            return saved_record
        if metadata is None:
            return self._mark_enqueued_failed(
                saved_job,
                DriveSyncErrorCode.METADATA_MISSING.value,
                "metadata sidecar is not available",
            )
        if not self._settings.rclone_remote:
            return self._mark_enqueued_failed(
                saved_job,
                DriveSyncErrorCode.NOT_CONFIGURED.value,
                "Google Drive is not configured",
            )
        return saved_record

    def retry_generation(
        self, generation_id: UUID, *, resync: bool = False
    ) -> tuple[DriveSyncRecord, DriveSyncJob | None]:
        """Retry a failed record or explicitly resync a synced record."""

        generation = self._get_completed_generation(generation_id)
        existing = self._repository.get_by_generation(generation_id)
        if existing is None:
            record = self.enqueue_generation(generation_id)
            if record is None:
                raise DriveSyncServiceError(
                    DriveSyncErrorCode.PERSISTENCE_FAILED.value,
                    "Drive synchronization could not be queued",
                )
            return record, None
        if existing.status is DriveSyncStatus.SYNCED and not resync:
            raise DriveSyncServiceError(
                "drive_resync_requires_confirmation",
                "a synced generation requires explicit resync confirmation",
                retryable=False,
            )
        if existing.status in {DriveSyncStatus.PENDING, DriveSyncStatus.SYNCING}:
            active_job = next(
                (
                    candidate
                    for candidate in self._repository.list_jobs(100)
                    if candidate.generation_id == generation_id
                    and candidate.status in {DriveSyncStatus.PENDING, DriveSyncStatus.SYNCING}
                ),
                None,
            )
            return existing, active_job

        artifacts = self._artifacts_with_repair(generation_id)
        image, metadata = _select_required_artifacts(artifacts)
        paths = _paths_from_record_or_generation(existing, generation, self._settings.timezone)
        updated_record, job = self._build_record_and_job(
            generation,
            image,
            metadata,
            paths,
            existing=existing,
        )
        saved_record, saved_job = self._repository.retry(updated_record, job)
        if metadata is None:
            return (
                self._mark_enqueued_failed(
                    saved_job,
                    DriveSyncErrorCode.METADATA_MISSING.value,
                    "metadata sidecar is not available",
                ),
                saved_job,
            )
        if not self._settings.rclone_remote:
            return (
                self._mark_enqueued_failed(
                    saved_job,
                    DriveSyncErrorCode.NOT_CONFIGURED.value,
                    "Google Drive is not configured",
                ),
                saved_job,
            )
        return saved_record, saved_job

    def retry_failed(self, limit: int = 100) -> tuple[UUID, ...]:
        generation_ids: list[UUID] = []
        seen: set[UUID] = set()
        for job in self._repository.list_jobs(limit):
            if job.status is not DriveSyncStatus.FAILED or not job.retryable:
                continue
            if job.generation_id in seen:
                continue
            seen.add(job.generation_id)
            try:
                self.retry_generation(job.generation_id)
            except Exception as exc:  # noqa: BLE001 - one manual retry must not stop the batch
                logger.warning(
                    "Drive retry could not be queued generation=%s error=%s",
                    job.generation_id,
                    type(exc).__name__,
                )
                continue
            generation_ids.append(job.generation_id)
        return tuple(generation_ids)

    def resync_synced(self, limit: int = 100) -> tuple[UUID, ...]:
        generation_ids: list[UUID] = []
        seen: set[UUID] = set()
        for job in self._repository.list_jobs(limit):
            if job.status is not DriveSyncStatus.SYNCED or job.generation_id in seen:
                continue
            seen.add(job.generation_id)
            try:
                self.retry_generation(job.generation_id, resync=True)
            except Exception as exc:  # noqa: BLE001 - one manual resync must not stop the batch
                logger.warning(
                    "Drive resync could not be queued generation=%s error=%s",
                    job.generation_id,
                    type(exc).__name__,
                )
                continue
            generation_ids.append(job.generation_id)
        return tuple(generation_ids)

    def discover_missing(self, limit: int | None = None) -> tuple[UUID, ...]:
        """Discover completed primary images without a sync record, once per call."""

        if not self._settings.rclone_remote:
            return ()
        bounded = limit or self._settings.drive_discovery_batch_size
        discovered: list[UUID] = []
        for candidate in self._repository.list_discovery_candidates(bounded):
            try:
                record = self.enqueue_generation(candidate.generation_id)
            except Exception as exc:  # noqa: BLE001 - discovery is best effort
                logger.warning(
                    "Drive discovery enqueue failed generation=%s error=%s",
                    candidate.generation_id,
                    type(exc).__name__,
                )
                continue
            if record is not None:
                discovered.append(candidate.generation_id)
        return tuple(discovered)

    async def process_job(self, job: DriveSyncJob, worker_id: str) -> DriveSyncRecord | None:
        """Verify both sources, copy image then metadata, and commit synced state."""

        try:
            record = self._repository.get_by_generation(job.generation_id)
            generation = self._generation_repository.get_by_id(job.generation_id)
            artifacts = self._artifact_repository.list_by_generation(job.generation_id)
            if record is None or generation is None:
                raise DriveSyncServiceError(
                    DriveSyncErrorCode.PERSISTENCE_FAILED.value,
                    "Drive synchronization source records are missing",
                )
            if generation.status is not GenerationStatus.COMPLETED:
                raise DriveSyncServiceError(
                    "drive_generation_not_completed",
                    "generation is not completed",
                )
            image, metadata = _select_required_artifacts(artifacts)
            if metadata is None or record.metadata_artifact_id is None:
                raise DriveSyncServiceError(
                    DriveSyncErrorCode.METADATA_MISSING.value,
                    "metadata sidecar is not available",
                )
            if image.id != job.image_artifact_id or image.id != record.image_artifact_id:
                raise DriveSyncServiceError(
                    DriveSyncErrorCode.SOURCE_CHANGED.value,
                    "image artifact identity changed",
                )
            if (
                metadata.id != job.metadata_artifact_id
                or metadata.id != record.metadata_artifact_id
            ):
                raise DriveSyncServiceError(
                    DriveSyncErrorCode.SOURCE_CHANGED.value,
                    "metadata artifact identity changed",
                )
            image_path = _verify_image_source(image, self._settings)
            metadata_path = _verify_metadata_source(metadata, self._settings, generation.id)
            if image.sha256 != job.image_sha256 or image.size_bytes != job.image_size_bytes:
                raise DriveSyncServiceError(
                    DriveSyncErrorCode.SOURCE_CHANGED.value,
                    "image artifact snapshot changed",
                )
            if (
                metadata.sha256 != job.metadata_sha256
                or metadata.size_bytes != job.metadata_size_bytes
            ):
                raise DriveSyncServiceError(
                    DriveSyncErrorCode.SOURCE_CHANGED.value,
                    "metadata artifact snapshot changed",
                )
            validate_remote_relative_path(record.remote_image_path)
            validate_remote_relative_path(record.remote_metadata_path)
            total_bytes = image.size_bytes + metadata.size_bytes
        except DriveSyncServiceError as exc:
            return self._failed(job, worker_id, exc)
        except Exception as exc:  # noqa: BLE001 - fail closed before any copy
            logger.warning(
                "Drive source verification failed generation=%s error=%s",
                job.generation_id,
                type(exc).__name__,
                exc_info=True,
            )
            return self._failed(
                job,
                worker_id,
                DriveSyncServiceError(
                    DriveSyncErrorCode.SOURCE_MISSING.value,
                    "Drive source verification failed",
                ),
            )

        await self._update_progress(job, worker_id, 0, total_bytes, "image")
        try:
            await self._copy_with_progress(
                job,
                worker_id,
                image_path,
                record.remote_image_path,
                image.size_bytes,
                total_bytes,
                completed_before=0,
                artifact_name="image",
            )
            await self._update_progress(job, worker_id, image.size_bytes, total_bytes, "metadata")
            await self._copy_with_progress(
                job,
                worker_id,
                metadata_path,
                record.remote_metadata_path,
                metadata.size_bytes,
                total_bytes,
                completed_before=image.size_bytes,
                artifact_name="metadata",
            )
            await self._update_progress(job, worker_id, total_bytes, total_bytes, None)
        except Exception as exc:  # noqa: BLE001 - remote failures never delete local files
            code = getattr(exc, "code", DriveSyncErrorCode.TRANSFER_FAILED.value)
            if not isinstance(code, str):
                code = DriveSyncErrorCode.TRANSFER_FAILED.value
            return self._failed(
                job,
                worker_id,
                DriveSyncServiceError(code, "Drive file copy failed"),
            )

        try:
            synced = self._repository.mark_synced(job.id, worker_id, datetime.now(UTC))
        except Exception as exc:  # noqa: BLE001 - remote success cannot be undone
            logger.error(
                "Drive sync persistence failed after copy generation=%s error=%s",
                job.generation_id,
                type(exc).__name__,
                exc_info=True,
            )
            return self._failed(
                job,
                worker_id,
                DriveSyncServiceError(
                    DriveSyncErrorCode.PERSISTENCE_FAILED.value,
                    "Drive sync result could not be saved",
                ),
            )
        try:
            await self.rebuild_manifest_async(
                _local_date(generation.created_at, self._settings.timezone)
            )
        except Exception as exc:  # noqa: BLE001 - manifest is outside the sync transaction
            logger.warning(
                "Drive manifest update failed generation=%s error=%s",
                job.generation_id,
                type(exc).__name__,
                exc_info=True,
            )
            try:
                self._repository.mark_manifest_warning(synced.id, "Drive manifest update failed")
            except Exception:
                logger.warning("Drive manifest warning could not be persisted", exc_info=True)
        return synced

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        return self._repository.status_counts()

    def list_jobs(self, limit: int = 50) -> tuple[DriveSyncJob, ...]:
        return self._repository.list_jobs(limit)

    def capacity(self) -> DriveCapacity:
        usage = shutil.disk_usage(self._settings.data_dir)
        return self._repository.capacity(
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
        )

    def cache_candidates(self, limit: int = 100) -> tuple[DriveCacheCandidate, ...]:
        return self._repository.cache_candidates(limit)

    def rebuild_manifest(self, local_date: str | None = None) -> Path:
        if not self._settings.rclone_remote:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.NOT_CONFIGURED.value,
                "Google Drive is not configured",
            )
        normalized_date = local_date or _local_date(datetime.now(UTC), self._settings.timezone)
        try:
            date.fromisoformat(normalized_date)
        except ValueError as exc:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.MANIFEST_FAILED.value,
                "manifest date is invalid",
                retryable=False,
            ) from exc
        target = self._write_manifest(normalized_date)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.MANIFEST_FAILED.value,
                "manifest copy requires an asynchronous boundary",
            )
        asyncio.run(self._copy_manifest(target, f"{normalized_date}/manifests/manifest.jsonl"))
        return target

    async def rebuild_manifest_async(self, local_date: str | None = None) -> Path:
        if not self._settings.rclone_remote:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.NOT_CONFIGURED.value,
                "Google Drive is not configured",
            )
        normalized_date = local_date or _local_date(datetime.now(UTC), self._settings.timezone)
        target = self._write_manifest(normalized_date)
        await self._copy_manifest(target, f"{normalized_date}/manifests/manifest.jsonl")
        return target

    async def _copy_manifest(self, target: Path, relative_path: str) -> None:
        await self._adapter.copy_file(target, relative_path, total_bytes=target.stat().st_size)

    def _write_manifest(self, normalized_date: str) -> Path:
        try:
            date.fromisoformat(normalized_date)
        except ValueError as exc:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.MANIFEST_FAILED.value,
                "manifest date is invalid",
                retryable=False,
            ) from exc
        records = self._repository.list_manifest_records(normalized_date)
        target_dir = self._settings.data_dir / ".drive-sync-manifests" / normalized_date
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "manifest.jsonl"
        temporary: Path | None = None
        try:
            lines = "".join(
                json.dumps(
                    _manifest_line(record, normalized_date),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for record in sorted(
                    records, key=lambda item: (item.created_at, item.generation_id.hex)
                )
            ).encode("utf-8")
            with NamedTemporaryFile(
                mode="wb", prefix=".manifest.", suffix=".tmp", dir=target_dir, delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(lines)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
            return target
        except OSError as exc:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.MANIFEST_FAILED.value,
                "manifest could not be written",
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _get_completed_generation(self, generation_id: UUID) -> Generation:
        generation = self._generation_repository.get_by_id(generation_id)
        if generation is None:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.SOURCE_MISSING.value,
                "generation was not found",
            )
        if generation.status is not GenerationStatus.COMPLETED:
            raise DriveSyncServiceError(
                "drive_generation_not_completed",
                "generation is not completed",
                retryable=False,
            )
        return generation

    def _artifacts_with_repair(self, generation_id: UUID) -> tuple[GenerationArtifact, ...]:
        artifacts = self._artifact_repository.list_by_generation(generation_id)
        if _find_artifact(artifacts, ArtifactType.METADATA) is not None:
            return artifacts
        if self._metadata_repair_handler is not None:
            try:
                self._metadata_repair_handler(generation_id)
            except Exception as exc:  # noqa: BLE001 - retry state is persisted below
                logger.warning(
                    "Drive metadata repair failed generation=%s error=%s",
                    generation_id,
                    type(exc).__name__,
                    exc_info=True,
                )
            try:
                artifacts = self._artifact_repository.list_by_generation(generation_id)
            except Exception:
                logger.warning("Drive metadata recheck failed generation=%s", generation_id)
        return artifacts

    def _build_record_and_job(
        self,
        generation: Generation,
        image: GenerationArtifact,
        metadata: GenerationArtifact | None,
        paths: DriveRemotePaths,
        *,
        existing: DriveSyncRecord | None,
    ) -> tuple[DriveSyncRecord, DriveSyncJob]:
        now = datetime.now(UTC)
        record_id = existing.id if existing is not None else self._id_factory()
        record = DriveSyncRecord(
            id=record_id,
            generation_id=generation.id,
            status=DriveSyncStatus.PENDING,
            remote_name=self._settings.rclone_remote,
            remote_base_path=self._settings.rclone_base_path,
            remote_image_path=paths.image_path,
            remote_metadata_path=paths.metadata_path,
            image_artifact_id=image.id,
            metadata_artifact_id=metadata.id if metadata is not None else None,
            image_sha256=image.sha256,
            metadata_sha256=metadata.sha256 if metadata is not None else None,
            image_size_bytes=image.size_bytes,
            metadata_size_bytes=metadata.size_bytes if metadata is not None else None,
            attempt_count=existing.attempt_count if existing is not None else 0,
            last_attempt_at=existing.last_attempt_at if existing is not None else None,
            synced_at=None,
            error_code=None,
            error_summary=None,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        return record, DriveSyncJob(
            id=self._id_factory(),
            sync_record_id=record.id,
            generation_id=generation.id,
            status=DriveSyncStatus.PENDING,
            queue_sequence=0,
            progress_bytes=0,
            total_bytes=image.size_bytes + (metadata.size_bytes if metadata is not None else 0),
            progress_percentage=0.0,
            current_artifact=None,
            worker_id=None,
            pid=None,
            claimed_at=None,
            lease_expires_at=None,
            started_at=None,
            completed_at=None,
            error_code=None,
            error_summary=None,
            retryable=True,
            log_path=f"logs/drive_sync/{generation.id}.log",
            image_artifact_id=image.id,
            metadata_artifact_id=metadata.id if metadata is not None else None,
            image_sha256=image.sha256,
            metadata_sha256=metadata.sha256 if metadata is not None else None,
            image_size_bytes=image.size_bytes,
            metadata_size_bytes=metadata.size_bytes if metadata is not None else None,
            created_at=now,
            updated_at=now,
        )

    def _mark_enqueued_failed(self, job: DriveSyncJob, code: str, summary: str) -> DriveSyncRecord:
        return self._repository.mark_failed(job.id, None, code, summary, retryable=True)

    def _failed(
        self, job: DriveSyncJob, worker_id: str, error: DriveSyncServiceError
    ) -> DriveSyncRecord | None:
        try:
            return self._repository.mark_failed(
                job.id,
                worker_id,
                error.code,
                str(error),
                retryable=error.retryable,
            )
        except Exception as exc:  # noqa: BLE001 - preserve original safe failure in logs only
            logger.error(
                "Drive sync failure could not be persisted generation=%s error=%s",
                job.generation_id,
                type(exc).__name__,
                exc_info=True,
            )
            return None

    async def _copy_with_progress(
        self,
        job: DriveSyncJob,
        worker_id: str,
        local_path: Path,
        relative_remote_path: str,
        artifact_size: int,
        total_bytes: int,
        *,
        completed_before: int,
        artifact_name: str,
    ) -> None:
        async def report(progress: DriveSyncProgress) -> None:
            current = min(artifact_size, max(0, progress.progress_bytes))
            total_progress = min(total_bytes, completed_before + current)
            percentage = total_progress * 100.0 / total_bytes if total_bytes else 100.0
            await self._update_progress(
                job,
                worker_id,
                total_progress,
                total_bytes,
                artifact_name,
                percentage=percentage,
            )

        await self._adapter.copy_file(
            local_path,
            relative_remote_path,
            progress_callback=report,
            total_bytes=artifact_size,
        )
        await self._update_progress(
            job,
            worker_id,
            completed_before + artifact_size,
            total_bytes,
            artifact_name,
        )

    async def _update_progress(
        self,
        job: DriveSyncJob,
        worker_id: str,
        progress_bytes: int,
        total_bytes: int,
        current_artifact: str | None,
        *,
        percentage: float | None = None,
    ) -> None:
        progress = DriveSyncProgress(
            progress_bytes=min(total_bytes, max(0, progress_bytes)),
            total_bytes=total_bytes,
            progress_percentage=(
                percentage
                if percentage is not None
                else (progress_bytes * 100.0 / total_bytes if total_bytes else 100.0)
            ),
            current_artifact=current_artifact,
        )
        self._repository.update_progress(job.id, worker_id, progress)


def _select_required_artifacts(
    artifacts: tuple[GenerationArtifact, ...],
) -> tuple[GenerationArtifact, GenerationArtifact | None]:
    image = _find_artifact(artifacts, ArtifactType.IMAGE)
    if image is None:
        raise DriveSyncServiceError(
            DriveSyncErrorCode.SOURCE_MISSING.value,
            "primary image artifact is missing",
        )
    return image, _find_artifact(artifacts, ArtifactType.METADATA)


def _find_artifact(
    artifacts: tuple[GenerationArtifact, ...], artifact_type: ArtifactType
) -> GenerationArtifact | None:
    return next(
        (artifact for artifact in artifacts if artifact.artifact_type is artifact_type),
        None,
    )


def _paths_from_record_or_generation(
    record: DriveSyncRecord, generation: Generation, timezone_name: str
) -> DriveRemotePaths:
    return DriveRemotePaths(
        local_date=_local_date(generation.created_at, timezone_name),
        image_path=record.remote_image_path,
        metadata_path=record.remote_metadata_path,
        manifest_path=(
            f"{_local_date(generation.created_at, timezone_name)}/manifests/manifest.jsonl"
        ),
    )


def _resolve_artifact_path(artifact: GenerationArtifact, settings: Settings) -> Path:
    normalized = artifact.local_path.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\x00" in normalized
    ):
        raise DriveSyncServiceError(
            DriveSyncErrorCode.SOURCE_MISSING.value,
            "artifact path is unsafe",
        )
    root = settings.data_dir.resolve()
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise DriveSyncServiceError(
            DriveSyncErrorCode.SOURCE_MISSING.value,
            "artifact file is not available",
        ) from exc
    if not resolved.is_file():
        raise DriveSyncServiceError(
            DriveSyncErrorCode.SOURCE_MISSING.value,
            "artifact file is not available",
        )
    return resolved


def _verify_image_source(artifact: GenerationArtifact, settings: Settings) -> Path:
    if artifact.mime_type != "image/png":
        raise DriveSyncServiceError(
            DriveSyncErrorCode.SOURCE_CHANGED.value,
            "primary image is not a PNG",
        )
    path = _resolve_artifact_path(artifact, settings)
    try:
        data = path.read_bytes()
        if len(data) != artifact.size_bytes or len(data) > settings.max_output_image_bytes:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.SOURCE_CHANGED.value,
                "primary image size changed",
            )
        if hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.SOURCE_CHANGED.value,
                "primary image hash changed",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format != "PNG":
                    raise ValueError("image is not PNG")
                image.verify()
            with Image.open(BytesIO(data)) as image:
                dimensions = image.size
        if artifact.width != dimensions[0] or artifact.height != dimensions[1]:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.SOURCE_CHANGED.value,
                "primary image dimensions changed",
            )
    except DriveSyncServiceError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise DriveSyncServiceError(
            DriveSyncErrorCode.SOURCE_CHANGED.value,
            "primary image validation failed",
        ) from exc
    return path


def _verify_metadata_source(
    artifact: GenerationArtifact, settings: Settings, generation_id: UUID
) -> Path:
    if artifact.mime_type != "application/json":
        raise DriveSyncServiceError(
            DriveSyncErrorCode.METADATA_INVALID.value,
            "metadata artifact type is invalid",
        )
    path = _resolve_artifact_path(artifact, settings)
    try:
        data = path.read_bytes()
        if len(data) != artifact.size_bytes or len(data) > settings.max_metadata_sidecar_bytes:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.SOURCE_CHANGED.value,
                "metadata size changed",
            )
        if hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise DriveSyncServiceError(
                DriveSyncErrorCode.SOURCE_CHANGED.value,
                "metadata hash changed",
            )
        payload = json.loads(data.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("generation_id") != str(generation_id)
        ):
            raise ValueError("metadata schema or generation does not match")
    except DriveSyncServiceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DriveSyncServiceError(
            DriveSyncErrorCode.METADATA_INVALID.value,
            "metadata validation failed",
        ) from exc
    return path


def _local_date(value: datetime, timezone_name: str) -> str:
    from zoneinfo import ZoneInfo

    try:
        return utc(value).astimezone(ZoneInfo(timezone_name)).date().isoformat()
    except Exception as exc:
        raise DriveSyncServiceError(
            DriveSyncErrorCode.MANIFEST_FAILED.value,
            "configured timezone is invalid",
            retryable=False,
        ) from exc


def _manifest_line(record: DriveManifestRecord, local_date: str) -> dict[str, object]:
    return {
        "local_date": local_date,
        "generation_id": str(record.generation_id),
        "kind": record.kind,
        "created_at": utc(record.created_at).isoformat(),
        "remote_image_path": record.remote_image_path,
        "remote_metadata_path": record.remote_metadata_path,
        "image_sha256": record.image_sha256,
        "metadata_sha256": record.metadata_sha256,
        "image_size_bytes": record.image_size_bytes,
        "metadata_size_bytes": record.metadata_size_bytes,
        "synced_at": utc(record.synced_at).isoformat(),
    }


__all__ = [
    "DriveAdapterProtocol",
    "DriveSyncService",
    "DriveSyncServiceError",
]
