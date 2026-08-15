"""Formatting and component-update view models for the Gradio UI."""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.domain.preflight import PreflightResult
from runpod_sdxl_image_studio.domain.state_sync import StateSyncView
from runpod_sdxl_image_studio.domain.system_status import (
    ComfyUIStatus,
    DriveHealthAvailability,
    QueueHealthAvailability,
    SystemErrorEvent,
    SystemHealthView,
)


@dataclass(frozen=True)
class GenerationStatusCardView:
    """DBから取得した生成状態をスマホ向け表示へ変換したview model。"""

    generation_id: str | None
    status: str
    queue_position: int | None
    progress_percentage: float | None
    current_step: str | None
    message: str


def generation_status_card_markdown(view: GenerationStatusCardView) -> str:
    """Render a user-facing status without exposing operational identifiers."""

    status = view.status.strip().lower()
    if status == "idle":
        return "### 生成待機中"
    if status in {"pending", "queued"}:
        return "### 生成待機中"
    if status == "running":
        if view.progress_percentage is not None:
            percentage = min(100.0, max(0.0, view.progress_percentage))
            return f"### 生成中 · {percentage:.0f}%"
        return "### 生成中"
    if status == "completed":
        return "### 完了"
    if status == "cancelled":
        return "### キャンセルしました"
    if status == "failed":
        return "### 生成に失敗しました\n再度お試しください。"
    return "### 状態を確認中"


def selected_lora_summary_markdown(state: object, max_loras: int) -> str:
    """Render a compact selected-LoRA summary for the primary generation surface."""

    if max_loras <= 0 or not isinstance(state, (list, tuple)):
        return "**LoRA**\nLoRA なし"
    labels: list[str] = []
    for row in state[:max_loras]:
        if not isinstance(row, dict):
            continue
        name = row.get("lora_name")
        if not isinstance(name, str) or not name.strip():
            continue
        strength = row.get("model_strength", 1.0)
        try:
            strength_text = f"{float(strength):.1f}"
        except (TypeError, ValueError):
            strength_text = "1.0"
        labels.append(f"{html.escape(name.strip())} {strength_text}")
    if not labels:
        return "**LoRA**\nLoRA なし"
    visible = " ".join(f"`{label}`" for label in labels[:2])
    if len(labels) > 2:
        visible += f" `+{len(labels) - 2}`"
    return f"**LoRA**\n{visible}"


def _short_generation_id(value: str | None) -> str:
    if not value:
        return "-"
    return value.replace("\r", "").replace("\n", "")[:12]


def initial_status_markdown() -> str:
    """Render the external-connection-free initial status."""

    return "\n".join(
        (
            "### ComfyUI 状態: 未確認",
            "**最終確認日時:** -",
            "**checkpoint / VAE / LoRA / upscaler:** 未取得",
        )
    )


def status_markdown(status: ComfyUIStatus, timezone_name: str) -> str:
    """Render safe, bounded system information for display."""

    state_label = "接続" if status.is_connected else status.message
    lines = [f"### ComfyUI 状態: {state_label}"]
    lines.append(f"**最終確認日時:** {_format_datetime(status.checked_at, timezone_name)}")

    system_stats = status.system_stats
    if system_stats is not None:
        lines.extend(
            (
                f"**ComfyUIバージョン:** {system_stats.comfyui_version or '不明'}",
                f"**Python:** {system_stats.python_version or '不明'}",
                f"**OS:** {system_stats.system_os or '不明'}",
            )
        )
        if system_stats.devices:
            lines.append("**デバイス:**")
            for device in system_stats.devices:
                lines.append(
                    "- "
                    f"{device.name or '不明'} / {device.device_type or '不明'} / "
                    f"VRAM 合計 {_format_bytes(device.vram_total)} / "
                    f"空き {_format_bytes(device.vram_free)}"
                )
        else:
            lines.append("**デバイス:** 不明")
    else:
        lines.append("**システム情報:** 未取得")

    capabilities = status.capabilities
    if capabilities is not None:
        lines.extend(
            (
                f"**checkpoint件数:** {len(capabilities.checkpoints)}",
                f"**VAE件数:** {len(capabilities.vaes)}",
                f"**LoRA件数:** {len(capabilities.loras)}",
                f"**upscaler件数:** {len(capabilities.upscale_models)}",
            )
        )
    else:
        lines.append("**能力情報:** 未取得")

    if status.error_summary:
        lines.append(f"**エラー概要:** {status.error_summary}")
    if status.warnings:
        lines.append("**warning:**")
        lines.extend(f"- {warning}" for warning in status.warnings)
    return "\n".join(lines)


def capability_choices(capabilities: ComfyUICapabilities) -> dict[str, tuple[str, ...]]:
    """Return stable choices for each UI selector."""

    return {
        "checkpoint": capabilities.checkpoints,
        "vae": capabilities.vaes,
        "sampler": capabilities.samplers,
        "scheduler": capabilities.schedulers,
        "upscaler": capabilities.upscale_models,
    }


def lora_markdown(capabilities: ComfyUICapabilities | None) -> str:
    """Render the basic LoRA list without exposing the raw object info payload."""

    if capabilities is None:
        return "**LoRA一覧:** 未取得"
    if not capabilities.loras:
        return "**LoRA一覧:** 0件"
    return "**LoRA一覧:**\n" + "\n".join(f"- `{name}`" for name in capabilities.loras)


def system_health_markdown(view: SystemHealthView, timezone_name: str) -> str:
    """Render the aggregated health view as responsive, bounded Markdown."""

    status = html.escape(str(view.overall_status))
    gpu = html.escape(view.gpu_name or "-")
    comfy_state = "connected" if view.comfyui_connected else "disconnected"
    queue_unavailable = (
        getattr(view.queue_available, "value", view.queue_available)
        == QueueHealthAvailability.UNAVAILABLE.value
    )
    drive_connection = _drive_connection_label(view)
    drive_sync_status = _drive_metric(
        view.drive.configured,
        view.drive.sync_status_available,
        view.pending_sync_count,
    )
    drive_failed_status = _drive_metric(
        view.drive.configured,
        view.drive.sync_status_available,
        view.failed_sync_count,
    )
    drive_last_sync = _drive_datetime(
        view.drive.configured,
        view.drive.job_history_available,
        view.last_sync_at,
        timezone_name,
    )
    drive_last_failure = _drive_datetime(
        view.drive.configured,
        view.drive.job_history_available,
        view.drive.last_failure_at,
        timezone_name,
    )
    unsynced = (
        "unavailable"
        if view.drive.configured
        and view.drive.capacity_available is DriveHealthAvailability.UNAVAILABLE
        else _format_bytes(view.unsynced_bytes)
    )
    lines = [
        f"### System Health: `{status}`",
        f"**Checked:** {_format_datetime(view.checked_at, timezone_name)}",
        f"**ComfyUI:** `{comfy_state}` — {html.escape(view.comfyui_message or '-')}",
        f"**ComfyUI version:** `{html.escape(view.comfyui_version or '-')}`",
        f"**GPU:** `{gpu}`",
        (
            f"**VRAM:** `{_format_bytes(view.vram_total)}` total / "
            f"`{_format_bytes(view.vram_free)}` free"
        ),
        (
            "**Queue:** unavailable"
            if queue_unavailable
            else (
                "**Queue:** "
                f"pending `{view.pending_count}`, running `{view.running_count}`, "
                f"unresolved failed `{view.unresolved_failed_count}`, "
                f"historical failed `{view.historical_failed_count}`"
            )
        ),
        (
            "**Storage:** "
            f"total `{_format_bytes(view.local_total_bytes)}`, "
            f"used `{_format_bytes(view.local_used_bytes)}`, "
            f"free `{_format_bytes(view.local_free_bytes)}`, "
            f"unsynced `{unsynced}`"
        ),
        (
            "**Google Drive:** "
            f"configured `{view.drive_configured}`, connected `{drive_connection}`, "
            f"pending `{drive_sync_status}`, failed `{drive_failed_status}`, "
            f"last sync `{drive_last_sync}`, last failure `{drive_last_failure}`"
        ),
        (
            "**Models:** "
            f"checkpoint `{view.checkpoint_count}`, LoRA `{view.lora_count}`, "
            f"VAE `{view.vae_count}`, upscaler `{view.upscaler_count}`"
        ),
    ]
    return "\n".join(lines)


def state_sync_markdown(view: StateSyncView, timezone_name: str) -> str:
    """Render state backup status without exposing local paths or secrets."""

    remote = "-"
    if view.remote_sha256 is not None:
        remote = f"{view.remote_sha256[:12]} / {_format_bytes(view.remote_size_bytes)}"
    return "\n".join(
        (
            "### State backup status",
            f"**Status:** `{html.escape(view.status.value)}`",
            f"**Last success:** `{_format_datetime(view.last_success_at, timezone_name)}`",
            f"**Last failure:** `{_format_datetime(view.last_failure_at, timezone_name)}`",
            f"**Remote hash / size:** `{html.escape(remote)}`",
            html.escape(view.last_message or "-"),
        )
    )


def _drive_connection_label(view: SystemHealthView) -> str:
    if not view.drive.configured:
        return "not configured"
    if view.drive.connection_available is DriveHealthAvailability.UNAVAILABLE:
        return "unavailable"
    return "connected" if view.drive.connected else "disconnected"


def _drive_metric(configured: bool, availability: DriveHealthAvailability, value: object) -> str:
    if not configured or availability is DriveHealthAvailability.UNAVAILABLE:
        return "unavailable"
    return str(value) if value is not None else "-"


def _drive_datetime(
    configured: bool,
    availability: DriveHealthAvailability,
    value: datetime | None,
    timezone_name: str,
) -> str:
    if not configured or availability is DriveHealthAvailability.UNAVAILABLE:
        return "unavailable"
    return _format_datetime(value, timezone_name)


def system_error_history_markdown(
    events: tuple[SystemErrorEvent, ...],
    timezone_name: str,
) -> str:
    """Render at most the repository's bounded recent error history."""

    if not events:
        return "### Recent errors\nNo recent operational errors."
    lines = ["### Recent errors"]
    for event in events[:100]:
        severity = html.escape(str(event.severity))
        generation = _short_generation_id(str(event.generation_id) if event.generation_id else None)
        job = _short_generation_id(str(event.job_id) if event.job_id else None)
        retryable = "yes" if event.retryable else "no"
        lines.append(
            "- "
            f"`{_format_datetime(event.created_at, timezone_name)}` "
            f"`{severity}` `{html.escape(event.category)}` "
            f"`{html.escape(event.error_code)}` "
            f"{html.escape(event.summary)} "
            f"(Generation: `{html.escape(generation)}`, Job: `{html.escape(job)}`, "
            f"retryable: `{retryable}`)"
        )
    return "\n".join(lines)


def preflight_markdown(result: PreflightResult) -> str:
    """Render preflight errors and warnings without exposing internal paths."""

    if result.is_ready and not result.warnings:
        return "Preflight: ready"
    lines = ["Preflight: ready with warnings" if result.is_ready else "Preflight: blocked"]
    for issue in result.errors:
        lines.append(f"- ❌ `{html.escape(issue.code)}` {html.escape(issue.message)}")
    for issue in result.warnings:
        lines.append(f"- ⚠️ `{html.escape(issue.code)}` {html.escape(issue.message)}")
    return "\n".join(lines)


def preserve_selection(current: str | None, choices: tuple[str, ...]) -> str | None:
    """Keep a selected value only if it is still present after a refresh."""

    return current if current in choices else None


def _format_datetime(value: datetime | None, timezone_name: str) -> str:
    if value is None:
        return "未確認"
    try:
        localized = value.astimezone(ZoneInfo(timezone_name))
    except (ZoneInfoNotFoundError, ValueError):
        localized = value
    return localized.isoformat(timespec="seconds")


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "不明"
    gib = value / (1024**3)
    return f"{gib:.2f} GiB ({value:,} bytes)"
