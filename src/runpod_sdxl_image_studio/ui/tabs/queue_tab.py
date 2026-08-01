"""Queue tab handlers that expose only application-service view data."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import GenerationQueueItem
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)


def build_queue_tab() -> tuple[
    gr.Button,
    gr.Dropdown,
    gr.Textbox,
    gr.Dropdown,
    gr.Markdown,
    gr.Button,
    gr.Button,
    gr.Button,
    gr.Markdown,
]:
    gr.Markdown("## 生成キュー")
    with gr.Row():
        refresh = gr.Button("キューを更新", variant="primary")
        status = gr.Dropdown(
            [("すべて", ""), *((item.value, item.value) for item in GenerationStatus)],
            value="",
            label="状態",
        )
        batch_filter = gr.Textbox(label="Batch ID", placeholder="optional UUID")
    jobs = gr.Dropdown([], label="ジョブ一覧", allow_custom_value=False)
    detail = gr.Markdown("ジョブを選択してください。")
    with gr.Row():
        cancel = gr.Button("選択ジョブをキャンセル")
        retry = gr.Button("選択ジョブを再試行")
        retry_batch = gr.Button("失敗のみ再実行")
    message = gr.Markdown("")
    return refresh, status, batch_filter, jobs, detail, cancel, retry, retry_batch, message


def make_queue_refresh_handler(
    service: GenerationQueueService,
) -> Callable[[str | None, str | None], tuple[object, str]]:
    def handler(status: str | None, batch_filter: str | None) -> tuple[object, str]:
        try:
            statuses = (GenerationStatus(status),) if status else None
            batch_id = UUID(batch_filter.strip()) if batch_filter and batch_filter.strip() else None
            items = service.list_jobs(statuses=statuses, batch_id=batch_id)
            choices = [(_queue_label(item), str(item.generation.id)) for item in items]
            return gr.Dropdown(
                choices=choices, value=choices[0][1] if choices else None
            ), f"{len(choices)}件"
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.skip(), str(exc)

    return handler


def make_queue_detail_handler(service: GenerationQueueService) -> Callable[[str | None], str]:
    def handler(selected: str | None) -> str:
        if not selected:
            return "ジョブを選択してください。"
        try:
            item = service.get_job_detail(UUID(selected))
            return _queue_detail(item) if item is not None else "ジョブが見つかりません。"
        except (GenerationQueueServiceError, ValueError):
            return "ジョブ詳細を取得できませんでした。"

    return handler


def make_queue_cancel_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], object]:
    async def handler(selected: str | None) -> object:
        if not selected:
            return "キャンセルするジョブを選択してください。"
        try:
            await service.cancel(UUID(selected))
            return "キャンセル処理を受け付けました。"
        except (GenerationQueueServiceError, ValueError) as exc:
            return str(exc)

    return handler


def make_queue_retry_handler(service: GenerationQueueService) -> Callable[[str | None], str]:
    def handler(selected: str | None) -> str:
        if not selected:
            return "再試行するジョブを選択してください。"
        try:
            result = service.retry(UUID(selected))
            return f"再試行をキューへ追加しました。sequence={result.queue_position}"
        except (GenerationQueueServiceError, ValueError) as exc:
            return str(exc)

    return handler


def make_queue_retry_batch_handler(service: GenerationQueueService) -> Callable[[str | None], str]:
    def handler(selected: str | None) -> str:
        if not selected:
            return "対象バッチのジョブを選択してください。"
        try:
            item = service.get_job_detail(UUID(selected))
            if item is None or item.entry.batch_id is None:
                return "バッチジョブを選択してください。"
            result = service.retry_failed_batch(item.entry.batch_id)
            return (
                "失敗ジョブはありません。"
                if result is None
                else f"{len(result.items)}件を再実行キューへ追加しました。"
            )
        except (GenerationQueueServiceError, ValueError) as exc:
            return str(exc)

    return handler


def _queue_label(queue_item: GenerationQueueItem) -> str:
    return (
        f"#{queue_item.entry.sequence} {queue_item.generation.status.value} "
        f"seed={queue_item.generation.settings_snapshot.seed} "
        f"{queue_item.generation.id}"
    )


def _queue_detail(item: GenerationQueueItem | None) -> str:
    if item is None:
        return "ジョブが見つかりません。"
    queue_item = item
    snapshot = queue_item.generation.settings_snapshot
    batch = (
        f"\nBatch: `{queue_item.batch.name}` index={queue_item.entry.batch_index}"
        if queue_item.batch
        else ""
    )
    cancel = "\nキャンセル要求中" if queue_item.entry.cancel_requested_at else ""
    error = (
        f"\nError: `{queue_item.generation.error_summary}`"
        if queue_item.generation.error_summary
        else ""
    )
    return (
        f"**Sequence:** `{queue_item.entry.sequence}`  "
        f"**Status:** `{queue_item.generation.status.value}`\n"
        f"Generation: `{queue_item.generation.id}`  Job: `{queue_item.job.id}`{batch}{cancel}\n"
        f"Checkpoint: `{snapshot.checkpoint_name}`  Seed: `{snapshot.seed}`  "
        f"Size: `{snapshot.width}×{snapshot.height}`  LoRA: `{len(snapshot.loras)}`\n"
        f"Progress: `{queue_item.job.progress_value}/{queue_item.job.progress_maximum}`  "
        f"Node: `{queue_item.job.current_node or '-'}`{error}"
    )


__all__ = [
    "build_queue_tab",
    "make_queue_cancel_handler",
    "make_queue_detail_handler",
    "make_queue_refresh_handler",
    "make_queue_retry_batch_handler",
    "make_queue_retry_handler",
]
