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
    gr.Markdown("## Generation Queue")
    with gr.Row():
        refresh = gr.Button("Refresh queue", variant="primary")
        status = gr.Dropdown(
            [("all", ""), *((item.value, item.value) for item in GenerationStatus)],
            value="",
            label="Status",
        )
        batch_filter = gr.Textbox(label="Batch ID", placeholder="optional UUID")
    jobs = gr.Dropdown([], label="Jobs", allow_custom_value=False)
    detail = gr.Markdown("Select a job")
    with gr.Row():
        cancel = gr.Button("Cancel selected job")
        retry = gr.Button("Retry selected job")
        retry_batch = gr.Button("Retry failed batch")
    ambiguous_prompt_id = gr.Textbox(label="Ambiguous prompt ID", visible=False, interactive=False)
    with gr.Row():
        ambiguous_link = gr.Button("Link prompt ID", visible=False, interactive=False)
        ambiguous_fail = gr.Button(
            "Confirm prompt absent as failed",
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
                f"{len(choices)} jobs"
            )
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.skip(), str(exc)

    return handler


def make_queue_detail_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], tuple[str, object, object, object]]:
    def handler(selected: str | None) -> tuple[str, object, object, object]:
        if not selected:
            return ("Select a job", *_ambiguous_controls(None))
        try:
            item = service.get_job_detail(UUID(selected))
            return (
                _queue_detail(item) if item is not None else "Job was not found",
                *_ambiguous_controls(item),
            )
        except (GenerationQueueServiceError, ValueError):
            return ("Job detail could not be loaded", *_ambiguous_controls(None))

    return handler


def make_queue_cancel_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], Awaitable[tuple[object, str]]]:
    async def handler(selected: str | None) -> tuple[object, str]:
        if not selected:
            return gr.Button(interactive=True), "Select a job to cancel"
        try:
            item = await service.cancel(UUID(selected))
            message = (
                "Cancellation completed"
                if item.generation.status is GenerationStatus.CANCELLED
                else "Cancellation requested; the worker will confirm it"
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
            return gr.Button(interactive=True), "Select a job to retry"
        try:
            result = service.retry(UUID(selected))
            return gr.Button(interactive=True), f"Retry queued: sequence={result.queue_position}"
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.Button(interactive=True), str(exc)

    return handler


def make_queue_retry_batch_handler(
    service: GenerationQueueService,
) -> Callable[[str | None], tuple[object, str]]:
    def handler(selected: str | None) -> tuple[object, str]:
        if not selected:
            return gr.Button(interactive=True), "Select a batch job"
        try:
            item = service.get_job_detail(UUID(selected))
            if item is None or item.entry.batch_id is None:
                return gr.Button(interactive=True), "Select a batch job"
            result = service.retry_failed_batch(item.entry.batch_id)
            message = "No failed jobs" if result is None else f"Queued {len(result.items)} retries"
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
                raise GenerationQueueServiceError("Select an ambiguous job")
            item = service.link_ambiguous_prompt(UUID(selected), prompt_id or "")
            return (
                gr.Textbox(value="", visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                f"Prompt linked: `{item.job.prompt_id}`",
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
                raise GenerationQueueServiceError("Select an ambiguous job")
            item = service.fail_ambiguous_prompt(UUID(selected))
            return (
                gr.Textbox(value="", visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                f"Prompt absence confirmed; job failed: `{item.generation.id}`",
            )
        except (GenerationQueueServiceError, ValueError) as exc:
            return gr.skip(), gr.Button(interactive=True), gr.Button(interactive=True), str(exc)

    return handler


def _queue_label(queue_item: GenerationQueueItem) -> str:
    return (
        f"#{queue_item.entry.sequence} {queue_item.generation.status.value} "
        f"seed={queue_item.generation.settings_snapshot.seed} {queue_item.generation.id}"
    )


def _queue_detail(item: GenerationQueueItem | None) -> str:
    if item is None:
        return "Job was not found"
    snapshot = item.generation.settings_snapshot
    batch = f"\nBatch: `{item.batch.name}` index={item.entry.batch_index}" if item.batch else ""
    cancel = "\nCancellation requested" if item.entry.cancel_requested_at else ""
    error = f"\nError: `{item.generation.error_summary}`" if item.generation.error_summary else ""
    ambiguous = (
        "\n**Ambiguous prompt is never resent automatically; use an explicit resolution action.**"
        if item.entry.submission_state.value == "ambiguous"
        else ""
    )
    return (
        f"**Sequence:** `{item.entry.sequence}` **Status:** `{item.generation.status.value}`\n"
        f"Generation: `{item.generation.id}` Job: `{item.job.id}`{batch}{cancel}\n"
        f"Submission: `{item.entry.submission_state.value}` "
        f"token=`{item.entry.submission_token or '-'}` "
        f"started=`{item.entry.submission_started_at or '-'}`\n"
        f"Prompt IDs: generation=`{item.generation.comfy_prompt_id or '-'}` "
        f"job=`{item.job.prompt_id or '-'}`{ambiguous}\n"
        f"Checkpoint: `{snapshot.checkpoint_name}` Seed: `{snapshot.seed}` "
        f"Size: `{snapshot.width}x{snapshot.height}` LoRA: `{len(snapshot.loras)}`\n"
        f"Progress: `{item.job.progress_value}/{item.job.progress_maximum}` "
        f"Node: `{item.job.current_node or '-'}`{error}"
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
