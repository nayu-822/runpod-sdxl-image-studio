"""Safe UI components and handlers for Google Drive synchronization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.drive_sync import (
    DriveConnectionStatus,
    DriveSyncStatus,
)
from runpod_sdxl_image_studio.services.drive_sync_service import (
    DriveSyncService,
    DriveSyncServiceError,
)


@dataclass(frozen=True)
class DriveSyncTabComponents:
    """Controls and output components used by the app builder."""

    refresh_button: gr.Button
    connection_button: gr.Button
    discovery_button: gr.Button
    retry_selected_button: gr.Button
    retry_failed_button: gr.Button
    resync_button: gr.Button
    manifest_button: gr.Button
    failed_manifest_button: gr.Button
    manifest_date: gr.Textbox
    selected_job: gr.Dropdown
    connection_status: gr.Markdown
    summary: gr.Markdown
    jobs: gr.Markdown
    message: gr.Markdown


def build_drive_sync_tab() -> DriveSyncTabComponents:
    gr.Markdown("## 同期・設定")
    connection_status = gr.Markdown("接続状態は未確認です")
    with gr.Row():
        connection_button = gr.Button("接続確認", variant="primary")
        refresh_button = gr.Button("同期状態を更新")
        discovery_button = gr.Button("完了済みを検出")
    summary = gr.Markdown("同期状態を更新してください")
    selected_job = gr.Dropdown([], label="対象Generation", allow_custom_value=False)
    with gr.Row():
        retry_selected_button = gr.Button("選択を再試行")
        retry_failed_button = gr.Button("失敗を再試行")
        resync_button = gr.Button("同期済みを再同期")
        manifest_button = gr.Button("Manifest再構築を登録")
        failed_manifest_button = gr.Button("失敗日付を再構築")
    manifest_date = gr.Textbox(
        label="Manifest日付 (YYYY-MM-DD、空欄は今日)",
        placeholder="2026-08-08",
    )
    jobs = gr.Markdown("")
    message = gr.Markdown("")
    return DriveSyncTabComponents(
        refresh_button=refresh_button,
        connection_button=connection_button,
        discovery_button=discovery_button,
        retry_selected_button=retry_selected_button,
        retry_failed_button=retry_failed_button,
        resync_button=resync_button,
        manifest_button=manifest_button,
        failed_manifest_button=failed_manifest_button,
        manifest_date=manifest_date,
        selected_job=selected_job,
        connection_status=connection_status,
        summary=summary,
        jobs=jobs,
        message=message,
    )


def make_drive_sync_refresh_handler(
    service: DriveSyncService,
) -> Callable[[], tuple[object, str, str, str]]:
    def handler() -> tuple[object, str, str, str]:
        try:
            choices, summary, jobs = render_drive_sync_outputs(service)
            return (
                gr.Dropdown(choices=choices, value=choices[0][1] if choices else None),
                summary,
                jobs,
                "",
            )
        except Exception:
            return gr.skip(), "同期状態を読み込めませんでした", "", "同期状態の取得に失敗しました"

    return handler


def make_drive_connection_handler(
    service: DriveSyncService,
) -> Callable[[], Awaitable[tuple[object, str, str]]]:
    async def handler() -> tuple[object, str, str]:
        try:
            result = await service.check_connection()
            return gr.Button(interactive=True), _connection_message(result.status), result.message
        except Exception:
            return (
                gr.Button(interactive=True),
                "接続状態を確認できませんでした",
                "接続確認に失敗しました",
            )

    return handler


def make_drive_discovery_handler(
    service: DriveSyncService,
) -> Callable[[], tuple[object, str]]:
    def handler() -> tuple[object, str]:
        try:
            count = len(service.discover_missing())
            return gr.Button(interactive=True), f"{count}件を同期キューへ追加しました"
        except Exception:
            return gr.Button(interactive=True), "完了済みGenerationの検出に失敗しました"

    return handler


def make_drive_retry_selected_handler(
    service: DriveSyncService,
) -> Callable[[str | None], tuple[object, str]]:
    def handler(selected: str | None) -> tuple[object, str]:
        if not selected:
            return gr.Button(interactive=True), "対象Generationを選択してください"
        try:
            service.retry_generation(UUID(selected))
            return gr.Button(interactive=True), "再試行を同期キューへ追加しました"
        except (DriveSyncServiceError, ValueError):
            return gr.Button(interactive=True), "選択したGenerationを再試行できませんでした"
        except Exception:
            return gr.Button(interactive=True), "再試行の登録に失敗しました"

    return handler


def make_drive_retry_failed_handler(
    service: DriveSyncService,
) -> Callable[[], tuple[object, str]]:
    def handler() -> tuple[object, str]:
        try:
            count = len(service.retry_failed())
            return gr.Button(interactive=True), f"{count}件の再試行を登録しました"
        except Exception:
            return gr.Button(interactive=True), "失敗Jobの再試行登録に失敗しました"

    return handler


def make_drive_resync_handler(
    service: DriveSyncService,
) -> Callable[[], tuple[object, str]]:
    def handler() -> tuple[object, str]:
        try:
            count = len(service.resync_synced())
            return gr.Button(interactive=True), f"{count}件の再同期を登録しました"
        except Exception:
            return gr.Button(interactive=True), "同期済みJobの再同期登録に失敗しました"

    return handler


def make_drive_manifest_handler(
    service: DriveSyncService,
) -> Callable[[str | None], tuple[object, str]]:
    def handler(local_date: str | None = None) -> tuple[object, str]:
        try:
            service.enqueue_manifest_rebuild(local_date or None)
            return gr.Button(interactive=True), "Manifest再構築をWorkerへ登録しました"
        except Exception:
            return gr.Button(interactive=True), "Manifest再構築の登録に失敗しました"

    return handler


def make_drive_failed_manifest_handler(
    service: DriveSyncService,
) -> Callable[[], tuple[object, str]]:
    def handler() -> tuple[object, str]:
        try:
            count = len(service.retry_failed_manifests())
            return gr.Button(interactive=True), f"{count}件の失敗日付を再構築キューへ登録しました"
        except Exception:
            return gr.Button(interactive=True), "失敗manifestの再構築登録に失敗しました"

    return handler


def render_drive_sync_outputs(
    service: DriveSyncService,
) -> tuple[list[tuple[str, str]], str, str]:
    counts = service.status_counts()
    capacity = service.capacity()
    jobs = service.list_jobs(50)
    manifest_jobs = service.list_manifest_jobs(50)
    manifest_failures = service.list_manifest_failure_targets(100)
    failure_dates = (
        ", ".join(sorted({target.local_date for target in manifest_failures})[:10]) or "-"
    )
    choices = [
        (f"{job.status.value} / {job.generation_id}", str(job.generation_id)) for job in jobs
    ]
    summary = (
        f"Drive設定: {'設定済み' if service.is_configured else '未設定'}\n\n"
        f"pending={counts.get(DriveSyncStatus.PENDING, 0)} / "
        f"syncing={counts.get(DriveSyncStatus.SYNCING, 0)} / "
        f"synced={counts.get(DriveSyncStatus.SYNCED, 0)} / "
        f"failed={counts.get(DriveSyncStatus.FAILED, 0)}\n\n"
        f"空き容量: {_bytes(capacity.free_bytes)} / "
        f"未同期: {_bytes(capacity.unsynced_bytes)} / "
        f"同期済みキャッシュ候補: {_bytes(capacity.synced_cache_bytes)}"
        f"\n\nmanifest失敗: {len(manifest_failures)}件"
        f" / 対象日付: {failure_dates}"
    )
    lines = ["| 状態 | Generation | 進捗 | エラー |", "|---|---|---:|---|"]
    for job in jobs:
        error = job.error_code or "-"
        lines.append(
            f"| `{job.status.value}` | `{job.generation_id}` | "
            f"{job.progress_percentage:.1f}% | `{error}` |"
        )
    for manifest_job in manifest_jobs[:10]:
        lines.append(
            f"| `manifest:{manifest_job.status.value}` | `{manifest_job.local_date}` | "
            f"{manifest_job.progress_percentage:.1f}% | `{manifest_job.error_code or '-'}:` "
            f"{manifest_job.remote_name}:{manifest_job.remote_base_path} |"
        )
    return choices, summary, "\n".join(lines)


def _connection_message(status: DriveConnectionStatus) -> str:
    return {
        DriveConnectionStatus.CONNECTED: "接続済み",
        DriveConnectionStatus.NOT_CONFIGURED: "未設定",
        DriveConnectionStatus.RCLONE_NOT_FOUND: "rclone未検出",
        DriveConnectionStatus.AUTH_FAILED: "認証失敗",
        DriveConnectionStatus.FAILED: "接続失敗",
    }[status]


def _bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.1f} KiB"
    if value < 1024**3:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value / 1024**3:.1f} GiB"


__all__ = [
    "DriveSyncTabComponents",
    "build_drive_sync_tab",
    "make_drive_connection_handler",
    "make_drive_discovery_handler",
    "make_drive_manifest_handler",
    "make_drive_failed_manifest_handler",
    "make_drive_resync_handler",
    "make_drive_retry_failed_handler",
    "make_drive_retry_selected_handler",
    "make_drive_sync_refresh_handler",
    "render_drive_sync_outputs",
]
