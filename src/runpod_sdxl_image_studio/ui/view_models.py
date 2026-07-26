"""Formatting and component-update view models for the Gradio UI."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.domain.system_status import ComfyUIStatus


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
