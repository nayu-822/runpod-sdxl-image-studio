"""Application service for durable single-worker queue operations."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepositoryError,
    GenerationDispatchQueueRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import (
    BatchSeedStrategy,
    GenerationBatch,
    GenerationQueueItem,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot

MAX_SEED = 2**64 - 1


class GenerationCancellationAdapterProtocol(Protocol):
    async def cancel_prompt(self, prompt_id: str) -> CancellationResult: ...


@dataclass(frozen=True)
class CancellationResult:
    """Adapter-level result without leaking ComfyUI response details."""

    requested: bool
    confirmed: bool
    message: str = ""


class GenerationQueueServiceError(RuntimeError):
    """Safe user-facing queue service boundary error."""


@dataclass(frozen=True)
class QueueEnqueueResult:
    item: GenerationQueueItem
    queue_position: int


@dataclass(frozen=True)
class BatchEnqueueResult:
    batch: GenerationBatch
    items: tuple[GenerationQueueItem, ...]


class GenerationQueueService:
    """Validate settings and delegate durable queue mutations to a repository."""

    def __init__(
        self,
        repository: GenerationDispatchQueueRepositoryProtocol,
        settings: Settings,
        cancellation_adapter: GenerationCancellationAdapterProtocol | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._cancellation_adapter = cancellation_adapter
        self._wake_callback: Callable[[], None] | None = None

    def set_wake_callback(self, callback: Callable[[], None] | None) -> None:
        self._wake_callback = callback

    def enqueue(
        self,
        settings: GenerationSettings,
        *,
        parent_generation_id: UUID | None = None,
    ) -> QueueEnqueueResult:
        self._ensure_pending_capacity(1)
        resolved = settings.model_copy(update={"seed": _resolve_seed(settings.seed)})
        try:
            snapshot = GenerationSettingsSnapshot.from_settings(resolved)
            item = self._repository.enqueue_single(
                snapshot,
                parent_generation_id=parent_generation_id,
            )
            self._wake()
            return QueueEnqueueResult(item=item, queue_position=item.entry.sequence)
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError("生成をキューへ追加できませんでした。") from exc

    def enqueue_batch(
        self,
        settings: GenerationSettings,
        *,
        count: int,
        seed_strategy: BatchSeedStrategy | str,
        start_seed: int | None,
        seed_step: int,
        name: str,
    ) -> BatchEnqueueResult:
        strategy = _seed_strategy(seed_strategy)
        self._validate_batch(count, strategy, start_seed, seed_step, name)
        self._ensure_pending_capacity(count)
        seeds = _batch_seeds(count, strategy, start_seed, seed_step)
        try:
            snapshots = tuple(
                GenerationSettingsSnapshot.from_settings(settings.model_copy(update={"seed": seed}))
                for seed in seeds
            )
            batch, items = self._repository.enqueue_batch(
                snapshots,
                name=name,
                seed_strategy=strategy,
                start_seed=start_seed,
                seed_step=seed_step,
            )
            self._wake()
            return BatchEnqueueResult(batch=batch, items=items)
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError("バッチをキューへ追加できませんでした。") from exc

    def list_jobs(
        self,
        *,
        statuses: Sequence[GenerationStatus] | None = None,
        batch_id: UUID | None = None,
    ) -> tuple[GenerationQueueItem, ...]:
        try:
            return self._repository.list_queue(statuses=statuses, batch_id=batch_id)
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("キューを取得できませんでした。") from exc

    def get_job_detail(self, generation_id: UUID) -> GenerationQueueItem | None:
        try:
            return self._repository.get_queue_item(generation_id)
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("ジョブ詳細を取得できませんでした。") from exc

    async def cancel(self, generation_id: UUID) -> GenerationQueueItem:
        try:
            requested = self._repository.request_cancel(generation_id)
            if (
                requested.generation.status is GenerationStatus.PENDING
                or not requested.job.prompt_id
            ):
                return self._repository.mark_cancelled(generation_id)
            if self._cancellation_adapter is None:
                raise GenerationQueueServiceError("実行中ジョブのキャンセルAdapterが未設定です。")
            result = await self._cancellation_adapter.cancel_prompt(requested.job.prompt_id)
            if not result.confirmed:
                raise GenerationQueueServiceError("ComfyUIのキャンセル確認を取得できませんでした。")
            return self._repository.mark_cancelled(generation_id)
        except GenerationQueueServiceError:
            raise
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError("ジョブをキャンセルできませんでした。") from exc

        except Exception as exc:  # noqa: BLE001 - hide adapter details at the UI boundary
            raise GenerationQueueServiceError("キャンセルに失敗しました。") from exc

    def retry(self, generation_id: UUID) -> QueueEnqueueResult:
        try:
            item = self._repository.get_queue_item(generation_id)
            if item is None:
                raise GenerationQueueServiceError("対象ジョブが見つかりません。")
            if item.generation.status not in {GenerationStatus.FAILED, GenerationStatus.CANCELLED}:
                raise GenerationQueueServiceError(
                    "失敗またはキャンセル済みジョブだけ再試行できます。"
                )
            self._ensure_pending_capacity(1)
            new_item = self._repository.enqueue_single(
                item.generation.settings_snapshot,
                kind=item.generation.kind,
                parent_generation_id=item.generation.parent_generation_id,
                retry_of_generation_id=item.generation.id,
                retry_attempt=item.generation.retry_attempt + 1,
            )
            self._wake()
            return QueueEnqueueResult(new_item, new_item.entry.sequence)
        except GenerationQueueServiceError:
            raise
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError("ジョブを再試行できませんでした。") from exc

    def retry_failed_batch(self, batch_id: UUID) -> BatchEnqueueResult | None:
        try:
            source_items = self._repository.list_batch_items(batch_id)
            failed = [
                item for item in source_items if item.generation.status is GenerationStatus.FAILED
            ]
            if not failed:
                return None
            source_batch = next(
                (item.batch for item in source_items if item.batch is not None), None
            )
            if source_batch is None:
                raise GenerationQueueServiceError("対象バッチが見つかりません。")
            self._ensure_pending_capacity(len(failed))
            snapshots = tuple(item.generation.settings_snapshot for item in failed)
            batch, items = self._repository.enqueue_batch(
                snapshots,
                name=f"{source_batch.name} (failed retry)",
                seed_strategy=source_batch.seed_strategy,
                start_seed=source_batch.start_seed,
                seed_step=source_batch.seed_step,
                retry_of_batch_id=source_batch.id,
                retry_of_generations=tuple(item.generation.id for item in failed),
                retry_attempts=tuple(item.generation.retry_attempt + 1 for item in failed),
            )
            self._wake()
            return BatchEnqueueResult(batch, items)
        except GenerationQueueServiceError:
            raise
        except (GenerationDispatchQueueRepositoryError, ValueError) as exc:
            raise GenerationQueueServiceError("失敗ジョブを再実行できませんでした。") from exc

    def _ensure_pending_capacity(self, additional: int) -> None:
        if additional < 1:
            raise GenerationQueueServiceError("キュー追加数が不正です。")
        try:
            pending = self._repository.list_queue(
                statuses=(GenerationStatus.PENDING,),
                limit=self._settings.queue_max_pending_jobs + 1,
            )
        except GenerationDispatchQueueRepositoryError as exc:
            raise GenerationQueueServiceError("待機中ジョブ数を確認できませんでした。") from exc
        if len(pending) + additional > self._settings.queue_max_pending_jobs:
            raise GenerationQueueServiceError("待機中ジョブ数の上限に達しています。")

    def _wake(self) -> None:
        if self._wake_callback is not None:
            self._wake_callback()

    def _validate_batch(
        self,
        count: int,
        strategy: BatchSeedStrategy,
        start_seed: int | None,
        seed_step: int,
        name: str,
    ) -> None:
        if not 1 <= count <= self._settings.batch_max_items:
            raise GenerationQueueServiceError("バッチ枚数が範囲外です。")
        if not name.strip() or len(name.strip()) > 200:
            raise GenerationQueueServiceError("バッチ名を確認してください。")
        if seed_step <= 0:
            raise GenerationQueueServiceError("seed増分は1以上にしてください。")
        if strategy is BatchSeedStrategy.SEQUENTIAL and start_seed is None:
            raise GenerationQueueServiceError("連番seedには開始seedが必要です。")
        if start_seed is not None and not 0 <= start_seed <= MAX_SEED:
            raise GenerationQueueServiceError("開始seedが範囲外です。")
        if strategy is BatchSeedStrategy.SEQUENTIAL and start_seed is not None:
            final_seed = start_seed + (count - 1) * seed_step
            if final_seed > MAX_SEED:
                raise GenerationQueueServiceError("バッチseedが上限を超えます。")


def _resolve_seed(seed: int) -> int:
    return secrets.randbelow(MAX_SEED + 1) if seed == -1 else seed


def _batch_seeds(
    count: int,
    strategy: BatchSeedStrategy,
    start_seed: int | None,
    seed_step: int,
) -> tuple[int, ...]:
    if strategy is BatchSeedStrategy.RANDOM:
        return tuple(_resolve_seed(-1) for _ in range(count))
    if start_seed is None:
        raise GenerationQueueServiceError("連番seedには開始seedが必要です。")
    return tuple(start_seed + index * seed_step for index in range(count))


def _seed_strategy(value: BatchSeedStrategy | str) -> BatchSeedStrategy:
    try:
        return value if isinstance(value, BatchSeedStrategy) else BatchSeedStrategy(value)
    except ValueError as exc:
        raise GenerationQueueServiceError("seed方式が不正です。") from exc


__all__ = [
    "BatchEnqueueResult",
    "CancellationResult",
    "GenerationCancellationAdapterProtocol",
    "GenerationQueueService",
    "GenerationQueueServiceError",
    "MAX_SEED",
    "QueueEnqueueResult",
]
