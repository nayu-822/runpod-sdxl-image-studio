"""Application service for the Phase A interactive generation session."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.interactive_run_repository import (
    InteractiveRunRepositoryError,
    InteractiveRunRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import BatchSeedStrategy, GenerationQueueItem
from runpod_sdxl_image_studio.domain.generation_settings import (
    MAX_SEED,
    RANDOM_SEED,
    GenerationSettings,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.interactive_run import (
    InteractiveGenerationRun,
    InteractiveRunStatus,
    InteractiveRunView,
)
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)


class InteractiveGenerationError(RuntimeError):
    """Safe error at the interactive run UI boundary."""


logger = logging.getLogger(__name__)


class InteractiveGenerationService:
    """Own the active-run invariant while delegating each Generation to the FIFO queue."""

    def __init__(
        self,
        repository: InteractiveRunRepositoryProtocol,
        queue_service: GenerationQueueService,
        settings: Settings,
        *,
        artifact_repository: GenerationArtifactRepositoryProtocol | None = None,
        state_changed_callback: Callable[[], None] | None = None,
    ) -> None:
        self._repository = repository
        self._queue_service = queue_service
        self._settings = settings
        self._artifact_repository = artifact_repository
        self._state_changed_callback = state_changed_callback

    def start(
        self,
        settings: GenerationSettings,
        *,
        batch_count: int,
        batch_size: int,
        client_local_date: str,
        name: str = "Interactive run",
    ) -> InteractiveRunView:
        if not 1 <= batch_count <= min(100, self._settings.max_batch_count):
            raise InteractiveGenerationError("interactive batch count is outside the allowed range")
        if not 1 <= batch_size <= 4:
            raise InteractiveGenerationError("batch size must be between 1 and 4")
        if not client_local_date or not client_local_date.strip():
            raise InteractiveGenerationError("client local date is required")
        try:
            run_settings = GenerationSettings.model_validate(
                {
                    **settings.model_dump(mode="python"),
                    "batch_size": batch_size,
                    "client_local_date": client_local_date,
                }
            )
        except (InteractiveRunRepositoryError, ValueError) as exc:
            raise InteractiveGenerationError(str(exc)) from exc
        normalized_client_local_date = run_settings.client_local_date
        if normalized_client_local_date is None:
            raise InteractiveGenerationError("client local date is required")

        strategy = (
            BatchSeedStrategy.RANDOM
            if settings.seed == RANDOM_SEED
            else BatchSeedStrategy.SEQUENTIAL
        )
        try:
            seeds = (
                tuple(secrets.randbelow(MAX_SEED + 1) for _ in range(batch_count))
                if strategy is BatchSeedStrategy.RANDOM
                else tuple(settings.seed + index for index in range(batch_count))
            )
            snapshots = tuple(
                GenerationSettingsSnapshot.from_settings(
                    run_settings.model_copy(update={"seed": seed})
                )
                for seed in seeds
            )
            run, _, _ = self._queue_service.run_with_enqueue_admission(
                lambda: self._repository.create_active_with_batch(
                    snapshots,
                    batch_count=batch_count,
                    batch_size=batch_size,
                    client_local_date=normalized_client_local_date,
                    name=name or "Interactive run",
                    seed_strategy=strategy,
                    start_seed=None if strategy is BatchSeedStrategy.RANDOM else settings.seed,
                    seed_step=1,
                    pending_limit=self._settings.queue_max_pending_jobs,
                )
            )
        except (
            GenerationQueueServiceError,
            InteractiveRunRepositoryError,
            ValueError,
        ) as exc:
            raise InteractiveGenerationError("interactive generations could not be queued") from exc
        self._notify_state_change()
        return self.refresh(run.id) or self._view(run, (), None)

    def get_active(self) -> InteractiveRunView | None:
        try:
            run = self._repository.get_active()
        except InteractiveRunRepositoryError as exc:
            raise InteractiveGenerationError("active interactive run could not be read") from exc
        return self._refresh_run(run) if run is not None else None

    def reconcile_after_queue_change(self) -> None:
        """Reconcile the active run after the single worker changes a Generation."""

        try:
            run = self._repository.get_active()
            if run is not None:
                self._refresh_run(run)
        except InteractiveGenerationError:
            logger.warning("interactive run reconciliation failed", exc_info=True)

    def restore(self) -> InteractiveRunView | None:
        """Reload the DB-backed active run and reconcile its queue projections."""
        try:
            run = self._repository.get_active()
            if run is None:
                run = self._repository.get_latest_completed()
        except InteractiveRunRepositoryError as exc:
            raise InteractiveGenerationError("interactive run could not be restored") from exc
        return self._refresh_run(run)

    def refresh(self, run_id: UUID | None = None) -> InteractiveRunView | None:
        try:
            run = self._repository.get_by_id(run_id) if run_id is not None else self._latest_run()
        except InteractiveRunRepositoryError as exc:
            raise InteractiveGenerationError("interactive run could not be read") from exc
        return self._refresh_run(run) if run is not None else None

    async def cancel(self, run_id: UUID | None = None) -> InteractiveRunView | None:
        try:
            run = (
                self._repository.get_by_id(run_id)
                if run_id is not None
                else self._repository.get_active()
            )
            if run is None:
                return None
            run = self._repository.request_cancel(run.id)
        except InteractiveRunRepositoryError as exc:
            raise InteractiveGenerationError(
                "interactive run cancellation could not start"
            ) from exc

        for generation_id in run.generation_ids:
            try:
                item = self._queue_service.get_job_detail(generation_id)
                if item is None or item.generation.status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED,
                }:
                    continue
                await self._queue_service.cancel(generation_id)
            except (GenerationQueueServiceError, ValueError):
                # Keep the durable run in cancelling until a later reconciliation can decide.
                continue
        return self.refresh(run.id)

    def _refresh_run(self, run: InteractiveGenerationRun | None) -> InteractiveRunView | None:
        if run is None:
            return None
        items: list[GenerationQueueItem] = []
        for generation_id in run.generation_ids:
            try:
                item = self._queue_service.get_job_detail(generation_id)
            except GenerationQueueServiceError:
                item = None
            if item is not None:
                items.append(item)

        completed_ids = tuple(
            generation_id
            for generation_id in run.generation_ids
            if any(
                item.generation.id == generation_id
                and item.generation.status is GenerationStatus.COMPLETED
                for item in items
            )
        )
        current_item = next(
            (
                item
                for item in items
                if item.generation.status
                not in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED,
                }
            ),
            None,
        )
        terminal_count = sum(
            1
            for item in items
            if item.generation.status
            in {
                GenerationStatus.COMPLETED,
                GenerationStatus.FAILED,
                GenerationStatus.CANCELLED,
            }
        )
        failed = next(
            (item for item in items if item.generation.status is GenerationStatus.FAILED), None
        )
        target_status: InteractiveRunStatus | None = None
        error_code: str | None = None
        error_summary: str | None = None
        completed_at: datetime | None = None
        if failed is not None and run.status is not InteractiveRunStatus.FAILED:
            for generation_id in run.generation_ids:
                if generation_id in completed_ids or generation_id == failed.generation.id:
                    continue
                try:
                    self._queue_service.cancel_pending(generation_id)
                except GenerationQueueServiceError:
                    logger.warning(
                        "interactive remaining generation cancellation failed generation_id=%s",
                        generation_id,
                        exc_info=True,
                    )
            try:
                run = self._repository.update_progress(
                    run.id,
                    completed_generation_ids=completed_ids,
                    current_generation_id=None,
                    status=InteractiveRunStatus.FAILED,
                    error_code=failed.generation.error_code or "generation_failed",
                    error_summary=failed.generation.error_summary
                    or "one interactive generation failed",
                    completed_at=datetime.now(UTC),
                )
            except InteractiveRunRepositoryError as exc:
                raise InteractiveGenerationError(
                    "interactive run progress could not be saved"
                ) from exc
            self._notify_state_change()
            return self._refresh_run(run)

        if failed is not None and run.status is InteractiveRunStatus.FAILED:
            return self._view(run, completed_ids, None)

        if failed is None and (
            run.generation_ids
            and terminal_count == len(run.generation_ids)
            and run.status
            not in {
                InteractiveRunStatus.COMPLETED,
                InteractiveRunStatus.CANCELLED,
                InteractiveRunStatus.FAILED,
            }
        ):
            target_status = (
                InteractiveRunStatus.CANCELLED
                if run.status is InteractiveRunStatus.CANCELLING
                or any(item.generation.status is GenerationStatus.CANCELLED for item in items)
                else InteractiveRunStatus.COMPLETED
            )
            completed_at = datetime.now(UTC)
        if (
            completed_ids != run.completed_generation_ids
            or (current_item.generation.id if current_item else None) != run.current_generation_id
            or target_status is not None
        ):
            try:
                run = self._repository.update_progress(
                    run.id,
                    completed_generation_ids=completed_ids,
                    current_generation_id=current_item.generation.id if current_item else None,
                    status=target_status,
                    error_code=error_code,
                    error_summary=error_summary,
                    completed_at=completed_at,
                )
            except InteractiveRunRepositoryError as exc:
                raise InteractiveGenerationError(
                    "interactive run progress could not be saved"
                ) from exc
            self._notify_state_change()
        current_status = current_item.generation.status.value if current_item else None
        return self._view(run, completed_ids, current_status)

    def _view(
        self,
        run: InteractiveGenerationRun,
        completed_ids: tuple[UUID, ...],
        current_status: str | None,
    ) -> InteractiveRunView:
        paths: list[Path] = []
        if self._artifact_repository is not None:
            last_generation_id = run.last_completed_generation_id
            if last_generation_id is None and completed_ids:
                last_generation_id = completed_ids[-1]
            result_generation_ids = (
                (last_generation_id,)
                if last_generation_id is not None and last_generation_id in completed_ids
                else ()
            )
            for generation_id in result_generation_ids:
                try:
                    artifacts = self._artifact_repository.list_by_generation(generation_id)
                except Exception:
                    continue
                for artifact in artifacts:
                    if artifact.artifact_type.value != "image":
                        continue
                    candidate = (self._settings.data_dir / artifact.local_path).resolve()
                    try:
                        candidate.relative_to(self._settings.data_dir.resolve())
                    except ValueError:
                        continue
                    if candidate.exists():
                        paths.append(candidate)
        return InteractiveRunView(run, len(completed_ids), current_status, tuple(paths))

    def _notify_state_change(self) -> None:
        if self._state_changed_callback is not None:
            self._state_changed_callback()

    def _latest_run(self) -> InteractiveGenerationRun | None:
        run = self._repository.get_active()
        return run if run is not None else self._repository.get_latest_completed()


__all__ = ["InteractiveGenerationError", "InteractiveGenerationService"]
