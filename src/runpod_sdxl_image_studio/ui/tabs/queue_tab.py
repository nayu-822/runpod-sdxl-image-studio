"""Queue tab handlers that expose only application-service view data."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
    gr.Textbox,
    gr.Button,
    gr.Button,
]:
    gr.Markdown("## 生成キュー")
    with gr.Row():
        refresh = gr.Button("キューを更新", variant="primary")
        status = gr.Dropdown(
            [("すべて", ""), *((_status_label(item), item.value) for item in GenerationStatus)],
            value="",
            label="状態",
        )
        batch_filter = gr.Textbox(label="バッチ ID", placeholder="任意のUUID")
    jobs = gr.Dropdown([], label="ジョブ", allow_custom_value=False)
    detail = gr.Markdown("ジョブを選択してください")
    with gr.Row():
        cancel = gr.Button("選択ジョブをキャンセル")
        retry = gr.Button("選択ジョブを再試行")
        retry_batch = gr.Button("失敗バッチを再試行")
    ambiguous_prompt_id = gr.Textbox(label="曖昧なprompt ID", visible=False, interactive=False)
    with gr.Row():
        ambiguous_link = gr.Button("prompt IDを紐付け", visible=False, interactive=False)
        ambiguous_fail = gr.Button(
            "prompt不在を確認して失敗確定",
            visible=False,
            interactive=False,
            variant="stop",
        )
    message = gr.Markdown("")
    return (
        refresh,
        status,
        batch_filter,
        jobs,
        detail,
        cancel,
        retry,
        retry_batch,
        message,
        ambiguous_prompt_id,
        ambiguous_link,
        ambiguous_fail,
    )


def make_queue_refresh_handler(
    service: GenerationQueueService,
) -> Callable[[str | None, str | None], tuple[object, str]]:
    def handler(status: str | None, batch_filter: str | None) -> tuple[object, str]:
        try:
            statuses = (GenerationStatus(status),) if status else None
            batch_id = UUID(batch_filter.strip()) if batch_filter and batch_filter.strip() else None
            items = service.list_jobs(statuses=statuses, batch_id=batch_id)
            choices = [(_queue_label(item), str(item.generation.id)) for item in items]
            return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), (
                f"{len(choices)}件のジョブ"
            )
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.skip(), str(exc)

    return handler


def make_queue_detail_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], tuple[str, object, object, object]]:
    def handler(selected: str | None) -> tuple[str, object, object, object]:
        if not selected:
            return ("ジョブを選択してください", *_ambiguous_controls(None))
        try:
            item = service.get_job_detail(UUID(selected))
            return (
                _queue_detail(item) if item is not None else "ジョブが見つかりませんでした",
                *_ambiguous_controls(item),
            )
        except (GenerationQueueServiceError, ValueError):
            return ("ジョブ詳細を読み込めませんでした", *_ambiguous_controls(None))

    return handler


def make_queue_cancel_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], Awaitable[tuple[object, str]]]:
    async def handler(selected: str | None) -> tuple[object, str]:
        if not selected:
            return gr.Button(interactive=True), "キャンセルするジョブを選択してください"
        try:
            item = await service.cancel(UUID(selected))
            message = (
                "キャンセルが完了しました"
                if item.generation.status is GenerationStatus.CANCELLED
                else "キャンセルを要求しました。workerが状態を確認します"
            )
            return gr.Button(interactive=True), message
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.Button(interactive=True), str(exc)

    return handler


def make_queue_retry_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], tuple[object, str]]:
    def handler(selected: str | None) -> tuple[object, str]:
        if not selected:
            return gr.Button(interactive=True), "再試行するジョブを選択してください"
        try:
            result = service.retry(UUID(selected))
            return gr.Button(
                interactive=True
            ), f"再試行をキューへ追加しました: 順序={result.queue_position}"
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.Button(interactive=True), str(exc)

    return handler


def make_queue_retry_batch_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], tuple[object, str]]:
    def handler(selected: str | None) -> tuple[object, str]:
        if not selected:
            return gr.Button(interactive=True), "バッチのジョブを選択してください"
        try:
            item = service.get_job_detail(UUID(selected))
            if item is None or item.entry.batch_id is None:
                return gr.Button(interactive=True), "バッチのジョブを選択してください"
            result = service.retry_failed_batch(item.entry.batch_id)
            message = (
                "失敗ジョブはありません"
                if result is None
                else f"{len(result.items)}件の再試行をキューへ追加しました"
            )
            return gr.Button(interactive=True), message
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.Button(interactive=True), str(exc)

    return handler


def make_queue_ambiguous_link_handler(
    service: GenerationQueueService,
) -> Callable[[str | None, str | None], tuple[object, object, object, str]]:
    def handler(selected: str | None, prompt_id: str | None) -> tuple[object, object, object, str]:
        try:
            if not selected:
                raise GenerationQueueServiceError("曖昧なジョブを選択してください")
            item = service.link_ambiguous_prompt(UUID(selected), prompt_id or "")
            return (
                gr.Textbox(value="", visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                f"promptを紐付けました: `{item.job.prompt_id}`",
            )
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.skip(), gr.Button(interactive=True), gr.Button(interactive=True), str(exc)

    return handler


def make_queue_ambiguous_fail_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], tuple[object, object, object, str]]:
    def handler(selected: str | None) -> tuple[object, object, object, str]:
        try:
            if not selected:
                raise GenerationQueueServiceError("曖昧なジョブを選択してください")
            item = service.fail_ambiguous_prompt(UUID(selected))
            return (
                gr.Textbox(value="", visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                f"prompt不在を確認し、ジョブを失敗確定しました: `{item.generation.id}`",
            )
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.skip(), gr.Button(interactive=True), gr.Button(interactive=True), str(exc)

    return handler


def _queue_label(queue_item: GenerationQueueItem) -> str:
    return (
        f"#{queue_item.entry.sequence} {_status_label(queue_item.generation.status)} "
        f"seed={queue_item.generation.settings_snapshot.seed} {queue_item.generation.id}"
    )


def _status_label(status: GenerationStatus) -> str:
    return {
        GenerationStatus.PENDING: "待機中",
        GenerationStatus.QUEUED: "キュー済み",
        GenerationStatus.RUNNING: "実行中",
        GenerationStatus.COMPLETED: "完了",
        GenerationStatus.FAILED: "失敗",
        GenerationStatus.CANCELLED: "キャンセル済み",
    }[status]


def _queue_detail(item: GenerationQueueItem | None) -> str:
    if item is None:
        return "ジョブが見つかりませんでした"
    snapshot = item.generation.settings_snapshot
    batch = f"\nバッチ: `{item.batch.name}` 番号={item.entry.batch_index}" if item.batch else ""
    cancel = "\nキャンセル要求済み" if item.entry.cancel_requested_at else ""
    error = f"\nエラー: `{item.generation.error_summary}`" if item.generation.error_summary else ""
    ambiguous = (
        "\n**曖昧なpromptは自動再送信されません。明示的な解決操作を使用してください。**"
        if item.entry.submission_state.value == "ambiguous"
        else ""
    )
    return (
        f"**順序:** `{item.entry.sequence}` **状態:** `{item.generation.status.value}`\n"
        f"Generation: `{item.generation.id}` Job: `{item.job.id}`{batch}{cancel}\n"
        f"送信状態: `{item.entry.submission_state.value}` "
        f"token=`{item.entry.submission_token or '-'}` "
        f"開始=`{item.entry.submission_started_at or '-'}`\n"
        f"prompt ID: generation=`{item.generation.comfy_prompt_id or '-'}` "
        f"job=`{item.job.prompt_id or '-'}`{ambiguous}\n"
        f"Checkpoint: `{snapshot.checkpoint_name}` Seed: `{snapshot.seed}` "
        f"サイズ: `{snapshot.width}x{snapshot.height}` LoRA: `{len(snapshot.loras)}`\n"
        f"進捗: `{item.job.progress_value}/{item.job.progress_maximum}` "
        f"ノード: `{item.job.current_node or '-'}`{error}"
    )


def _ambiguous_controls(item: GenerationQueueItem | None) -> tuple[object, object, object]:
    terminal = {GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED}
    eligible = (
        item is not None
        and item.entry.submission_state.value == "ambiguous"
        and item.generation.comfy_prompt_id is None
        and item.job.prompt_id is None
        and item.generation.status not in terminal
        and item.job.status not in terminal
    )
    return (
        gr.Textbox(value="", visible=eligible, interactive=eligible),
        gr.Button(visible=eligible, interactive=eligible),
        gr.Button(visible=eligible, interactive=eligible),
    )


__all__ = [
    "build_queue_tab",
    "make_queue_ambiguous_fail_handler",
    "make_queue_ambiguous_link_handler",
    "make_queue_cancel_handler",
    "make_queue_detail_handler",
    "make_queue_refresh_handler",
    "make_queue_retry_batch_handler",
    "make_queue_retry_handler",
]
