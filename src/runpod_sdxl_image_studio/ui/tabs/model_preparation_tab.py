"""Gradio controls for remote model catalog and durable preparation jobs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.model_transfer import (
    ModelTransferJob,
    ModelTransferStatus,
    RemoteModelCatalog,
    RemoteModelKind,
)
from runpod_sdxl_image_studio.services.model_preparation_service import (
    ModelPreparationService,
    ModelPreparationServiceError,
)


@dataclass(frozen=True)
class ModelPreparationTabComponents:
    refresh_button: gr.Button
    checkpoint: gr.Dropdown
    vae: gr.Dropdown
    loras: gr.Dropdown
    upscaler: gr.Dropdown
    prepare_button: gr.Button
    jobs_refresh_button: gr.Button
    jobs: gr.Dropdown
    cancel_button: gr.Button
    retry_button: gr.Button
    status: gr.Markdown
    message: gr.Markdown


def build_model_preparation_tab(max_loras: int) -> ModelPreparationTabComponents:
    gr.Markdown("## Google Driveモデル")
    refresh_button = gr.Button("Remote一覧を更新", variant="secondary")
    with gr.Row(elem_classes=["model-preparation-selection"]):
        checkpoint = gr.Dropdown([], label="Checkpoint（0または1）", allow_custom_value=False)
        vae = gr.Dropdown([], label="VAE（0または1）", allow_custom_value=False)
    loras = gr.Dropdown(
        [],
        label=f"LoRA（0〜{max_loras}件）",
        multiselect=True,
        allow_custom_value=False,
    )
    upscaler = gr.Dropdown([], label="Upscaler（0または1）", allow_custom_value=False)
    prepare_button = gr.Button("選択モデルをPodへ準備", variant="primary")
    gr.Markdown("## 準備状況")
    with gr.Row(elem_classes=["model-preparation-actions"]):
        jobs_refresh_button = gr.Button("準備状況を更新")
        jobs = gr.Dropdown([], label="対象ジョブ", allow_custom_value=False)
        cancel_button = gr.Button("キャンセル", elem_classes=["mobile-tap-button"])
        retry_button = gr.Button("再試行", elem_classes=["mobile-tap-button"])
    status = gr.Markdown("モデル準備状況は未取得です。")
    message = gr.Markdown("")
    return ModelPreparationTabComponents(
        refresh_button,
        checkpoint,
        vae,
        loras,
        upscaler,
        prepare_button,
        jobs_refresh_button,
        jobs,
        cancel_button,
        retry_button,
        status,
        message,
    )


def make_model_catalog_refresh_handler(
    service: ModelPreparationService,
) -> Callable[[], Awaitable[tuple[object, object, object, object, str]]]:
    async def handler() -> tuple[object, object, object, object, str]:
        try:
            catalog = await service.refresh_catalog()
            return (
                *_catalog_updates(catalog),
                f"Remoteモデルを{len(catalog.entries)}件取得しました。",
            )
        except ModelPreparationServiceError as exc:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), _safe_message(exc)

    return handler


def make_model_prepare_handler(
    service: ModelPreparationService,
) -> Callable[[str | None, str | None, list[str] | None, str | None], Awaitable[tuple[str, str]]]:
    async def handler(
        checkpoint: str | None,
        vae: str | None,
        loras: list[str] | None,
        upscaler: str | None,
    ) -> tuple[str, str]:
        try:
            result = await service.prepare_selected(checkpoint, vae, loras, upscaler)
            return _jobs_markdown(result.jobs), result.message
        except ModelPreparationServiceError as exc:
            return gr.skip(), _safe_message(exc)

    return handler


def make_model_jobs_refresh_handler(
    service: ModelPreparationService,
) -> Callable[[], tuple[object, str]]:
    def handler() -> tuple[object, str]:
        try:
            jobs = service.list_jobs()
            choices = [(_job_choice(job), str(job.id)) for job in jobs]
            return gr.Dropdown(
                choices=choices, value=choices[0][1] if choices else None
            ), _jobs_markdown(jobs)
        except ModelPreparationServiceError as exc:
            return gr.skip(), _safe_message(exc)

    return handler


def make_model_cancel_handler(
    service: ModelPreparationService,
) -> Callable[[str | None], tuple[str, str]]:
    def handler(selected: str | None) -> tuple[str, str]:
        if not selected:
            return gr.skip(), "キャンセルするジョブを選択してください。"
        try:
            job = service.cancel(UUID(selected))
            return _jobs_markdown((job,)), "キャンセル要求を保存しました。"
        except (ValueError, ModelPreparationServiceError) as exc:
            return gr.skip(), _safe_message(exc)

    return handler


def make_model_retry_handler(
    service: ModelPreparationService,
) -> Callable[[str | None], Awaitable[tuple[str, str]]]:
    async def handler(selected: str | None) -> tuple[str, str]:
        if not selected:
            return gr.skip(), "再試行するジョブを選択してください。"
        try:
            job = await service.retry(UUID(selected))
            return _jobs_markdown((job,)), "モデル準備を再試行キューへ登録しました。"
        except (ValueError, ModelPreparationServiceError) as exc:
            return gr.skip(), _safe_message(exc)

    return handler


def _catalog_updates(catalog: RemoteModelCatalog) -> tuple[object, object, object, object]:
    return (
        gr.Dropdown(choices=_choices(catalog, RemoteModelKind.CHECKPOINT), value=None),
        gr.Dropdown(choices=_choices(catalog, RemoteModelKind.VAE), value=None),
        gr.Dropdown(choices=_choices(catalog, RemoteModelKind.LORA), value=[]),
        gr.Dropdown(choices=_choices(catalog, RemoteModelKind.UPSCALER), value=None),
    )


def _choices(catalog: RemoteModelCatalog, kind: RemoteModelKind) -> list[tuple[str, str]]:
    return [(entry.display_name, entry.relative_path) for entry in catalog.by_kind(kind)]


def _job_choice(job: ModelTransferJob) -> str:
    return f"{job.kind.value} {job.remote_relative_path} ({job.status.value})"


def _jobs_markdown(jobs: tuple[ModelTransferJob, ...] | list[ModelTransferJob]) -> str:
    if not jobs:
        return "準備ジョブはありません。"
    lines = ["| 種別 | モデル | 状態 | 進捗 | error_code |", "|---|---|---|---:|---|"]
    for job in jobs:
        lines.append(
            f"| {job.kind.value} | {job.remote_relative_path} | "
            f"{_status_label(job.status)} | {job.progress_percentage:.0f}% "
            f"({job.progress_bytes}/{job.total_bytes} bytes) | {job.error_code or ''} |"
        )
    return "\n".join(lines)


def _status_label(status: ModelTransferStatus) -> str:
    return {
        ModelTransferStatus.PENDING: "待機中",
        ModelTransferStatus.DOWNLOADING: "取得中",
        ModelTransferStatus.COMPLETED: "完了",
        ModelTransferStatus.FAILED: "失敗",
        ModelTransferStatus.CANCEL_REQUESTED: "キャンセル中",
        ModelTransferStatus.CANCELLED: "キャンセル済み",
    }[status]


def _safe_message(error: ModelPreparationServiceError | Exception) -> str:
    if isinstance(error, ModelPreparationServiceError):
        return str(error)
    return "モデル準備に失敗しました。"


__all__ = [
    "ModelPreparationTabComponents",
    "build_model_preparation_tab",
    "make_model_cancel_handler",
    "make_model_catalog_refresh_handler",
    "make_model_jobs_refresh_handler",
    "make_model_prepare_handler",
    "make_model_retry_handler",
]
