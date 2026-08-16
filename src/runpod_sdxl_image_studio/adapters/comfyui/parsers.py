"""Pure parsers for ComfyUI response payloads."""

from __future__ import annotations

import ntpath
import posixpath
from collections.abc import Mapping, Sequence

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import ComfyUIParseError
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIDeviceInfo,
    ComfyUIObjectInfo,
    ComfyUIOutputImage,
    ComfyUISystemStats,
    PromptHistory,
    PromptHistoryStatus,
    QueuedPrompt,
    RemotePromptState,
    RemotePromptStatus,
)


def parse_system_stats(payload: Mapping[str, object]) -> ComfyUISystemStats:
    """Parse the supported fields from a ComfyUI system stats payload."""

    if not isinstance(payload, Mapping):
        raise ComfyUIParseError("ComfyUI system stats must be a JSON object")

    system_payload = _mapping_value(payload, "system") or payload
    devices_payload = payload.get("devices")
    if not isinstance(devices_payload, Sequence) or isinstance(devices_payload, str):
        devices_payload = ()

    devices: list[ComfyUIDeviceInfo] = []
    for device_payload in devices_payload:
        if not isinstance(device_payload, Mapping):
            continue
        devices.append(
            ComfyUIDeviceInfo(
                name=_optional_string(device_payload.get("name")),
                device_type=_optional_string(
                    device_payload.get("type", device_payload.get("device_type"))
                ),
                index=_optional_integer(device_payload.get("index")),
                vram_total=_optional_integer(device_payload.get("vram_total")),
                vram_free=_optional_integer(device_payload.get("vram_free")),
                torch_vram_total=_optional_integer(device_payload.get("torch_vram_total")),
                torch_vram_free=_optional_integer(device_payload.get("torch_vram_free")),
            )
        )

    return ComfyUISystemStats(
        system_os=_optional_string(system_payload.get("os")),
        python_version=_optional_string(system_payload.get("python_version")),
        embedded_python=_optional_boolean(system_payload.get("embedded_python")),
        comfyui_version=_optional_string(
            system_payload.get(
                "comfyui_version",
                payload.get("comfyui_version", payload.get("version")),
            )
        ),
        devices=tuple(devices),
    )


def parse_object_info(payload: Mapping[str, object]) -> ComfyUIObjectInfo:
    """Keep only node definitions with object-shaped values."""

    if not isinstance(payload, Mapping):
        raise ComfyUIParseError("ComfyUI object info must be a JSON object")

    nodes: dict[str, Mapping[str, object]] = {}
    for node_name, node_payload in payload.items():
        if isinstance(node_name, str) and isinstance(node_payload, Mapping):
            nodes[node_name] = dict(node_payload)
    return ComfyUIObjectInfo(nodes=nodes)


def parse_capabilities(object_info: ComfyUIObjectInfo) -> ComfyUICapabilities:
    """Extract model and sampler choices from supported node input schemas."""

    node_specs = (
        ("CheckpointLoaderSimple", "ckpt_name", "checkpoints", "checkpoint"),
        ("VAELoader", "vae_name", "vaes", "VAE"),
        ("KSampler", "sampler_name", "samplers", "sampler"),
        ("KSampler", "scheduler", "schedulers", "scheduler"),
        ("LoraLoader", "lora_name", "loras", "LoRA"),
        ("UpscaleModelLoader", "model_name", "upscale_models", "upscaler"),
        (
            "UltralyticsDetectorProvider",
            "model_name",
            "detector_models",
            "face detector",
        ),
    )
    extracted: dict[str, tuple[str, ...]] = {
        "checkpoints": (),
        "vaes": (),
        "samplers": (),
        "schedulers": (),
        "loras": (),
        "upscale_models": (),
        "detector_models": (),
    }
    warnings: list[str] = []

    for node_name, input_name, result_name, display_name in node_specs:
        node_payload = object_info.nodes.get(node_name)
        if node_payload is None:
            if result_name != "detector_models":
                warning = (
                    f"{node_name} ノードが /object_info にありません（{display_name}一覧は空です）"
                )
                if warning not in warnings:
                    warnings.append(warning)
            continue
        extracted[result_name] = _extract_choices(
            node_payload,
            input_name,
            warnings,
            node_name,
            strict_relative=result_name == "detector_models",
        )

    available_node_classes = frozenset(object_info.nodes)
    return ComfyUICapabilities(
        checkpoints=extracted["checkpoints"],
        vaes=extracted["vaes"],
        samplers=extracted["samplers"],
        schedulers=extracted["schedulers"],
        loras=extracted["loras"],
        upscale_models=extracted["upscale_models"],
        available_node_classes=available_node_classes,
        warnings=tuple(warnings),
        detector_models=extracted["detector_models"],
    )


def parse_queued_prompt(payload: Mapping[str, object]) -> QueuedPrompt:
    """Parse the small, typed subset of a ``/prompt`` response."""

    prompt_id = payload.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ComfyUIParseError("ComfyUI prompt response did not contain a prompt id")
    number = payload.get("number")
    safe_number = number if isinstance(number, int) and not isinstance(number, bool) else None
    node_errors = payload.get("node_errors", {})
    if not isinstance(node_errors, Mapping):
        node_errors = {}
    return QueuedPrompt(prompt_id=prompt_id, number=safe_number, node_errors=dict(node_errors))


def parse_prompt_history(payload: Mapping[str, object], prompt_id: str) -> PromptHistory:
    """Parse ComfyUI history without exposing its raw response structure."""

    prompt_entry = payload.get(prompt_id)
    if not isinstance(prompt_entry, Mapping):
        return PromptHistory(
            prompt_id,
            False,
            False,
            (),
            None,
            False,
            PromptHistoryStatus.NOT_FOUND,
        )

    status_payload = _mapping_value(prompt_entry, "status") or {}
    status_string = _optional_string(status_payload.get("status_str"))
    status = _parse_history_status(status_payload, status_string)
    is_failed = status is PromptHistoryStatus.FAILED
    is_completed = status is PromptHistoryStatus.COMPLETED
    outputs_payload = _mapping_value(prompt_entry, "outputs") or {}
    outputs: list[ComfyUIOutputImage] = []
    for node_output in outputs_payload.values():
        if not isinstance(node_output, Mapping):
            continue
        images = node_output.get("images")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
            continue
        for image_payload in images:
            if not isinstance(image_payload, Mapping):
                continue
            filename = _optional_string(image_payload.get("filename"))
            subfolder = _optional_string(image_payload.get("subfolder"))
            output_type = _optional_string(image_payload.get("type"))
            if filename is None or subfolder is None or output_type is None:
                continue
            outputs.append(ComfyUIOutputImage(filename, subfolder, output_type))

    error_message = "ComfyUIで画像生成に失敗しました" if is_failed else None
    return PromptHistory(
        prompt_id,
        is_completed,
        is_failed,
        tuple(outputs),
        error_message,
        True,
        status,
    )


def parse_remote_prompt_status(payload: Mapping[str, object], prompt_id: str) -> RemotePromptState:
    """Parse the supported status subset of the modern job endpoint."""

    if not isinstance(payload, Mapping):
        return RemotePromptState(prompt_id, RemotePromptStatus.UNAVAILABLE)
    if payload.get("not_found") is True or payload.get("exists") is False:
        return RemotePromptState(prompt_id, RemotePromptStatus.NOT_FOUND)
    if payload.get("cancelled") is True or payload.get("canceled") is True:
        return RemotePromptState(prompt_id, RemotePromptStatus.CANCELLED)
    if payload.get("completed") is True or payload.get("success") is True:
        return RemotePromptState(prompt_id, RemotePromptStatus.COMPLETED)
    if payload.get("failed") is True or payload.get("error") is True:
        return RemotePromptState(prompt_id, RemotePromptStatus.FAILED)

    status_payload = _mapping_value(payload, "status")
    job_payload = _mapping_value(payload, "job")
    candidates: list[object] = [
        payload.get("status"),
        payload.get("state"),
        payload.get("status_str"),
        payload.get("job_status"),
    ]
    if status_payload is not None:
        candidates.extend(
            (
                status_payload.get("status"),
                status_payload.get("status_str"),
                status_payload.get("state"),
            )
        )
    if job_payload is not None:
        candidates.extend(
            (job_payload.get("status"), job_payload.get("status_str"), job_payload.get("state"))
        )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        status = _remote_status_from_string(candidate)
        if status is not None:
            return RemotePromptState(prompt_id, status)
    return RemotePromptState(prompt_id, RemotePromptStatus.UNAVAILABLE)


def _remote_status_from_string(value: str) -> RemotePromptStatus | None:
    normalized = value.strip().casefold()
    mapping = {
        "pending": RemotePromptStatus.PENDING,
        "queued": RemotePromptStatus.PENDING,
        "waiting": RemotePromptStatus.PENDING,
        "running": RemotePromptStatus.IN_PROGRESS,
        "executing": RemotePromptStatus.IN_PROGRESS,
        "in_progress": RemotePromptStatus.IN_PROGRESS,
        "processing": RemotePromptStatus.IN_PROGRESS,
        "success": RemotePromptStatus.COMPLETED,
        "completed": RemotePromptStatus.COMPLETED,
        "complete": RemotePromptStatus.COMPLETED,
        "failed": RemotePromptStatus.FAILED,
        "error": RemotePromptStatus.FAILED,
        "cancelled": RemotePromptStatus.CANCELLED,
        "canceled": RemotePromptStatus.CANCELLED,
        "interrupted": RemotePromptStatus.CANCELLED,
        "execution_interrupted": RemotePromptStatus.CANCELLED,
        "not_found": RemotePromptStatus.NOT_FOUND,
        "missing": RemotePromptStatus.NOT_FOUND,
    }
    return mapping.get(normalized)


def _parse_history_status(
    status_payload: Mapping[str, object], status_string: str | None
) -> PromptHistoryStatus:
    """Map known ComfyUI status and message names without exposing raw JSON."""

    if _contains_interruption(status_payload.get("messages")):
        return PromptHistoryStatus.INTERRUPTED
    normalized = status_string.casefold() if status_string is not None else None
    if normalized in {"success", "completed"} or bool(status_payload.get("completed")):
        return PromptHistoryStatus.COMPLETED
    if normalized in {"error", "failed"}:
        return PromptHistoryStatus.FAILED
    if normalized in {"pending", "running", "executing", "in_progress"}:
        return PromptHistoryStatus.IN_PROGRESS
    if normalized in {"execution_interrupted", "interrupted", "cancelled", "canceled"}:
        return PromptHistoryStatus.INTERRUPTED
    return PromptHistoryStatus.UNKNOWN


def _contains_interruption(value: object) -> bool:
    if isinstance(value, str):
        return value.casefold() in {
            "execution_interrupted",
            "interrupted",
            "cancelled",
            "canceled",
        }
    if isinstance(value, Mapping):
        return any(_contains_interruption(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_interruption(child) for child in value)
    return False


def _extract_choices(
    node_payload: Mapping[str, object],
    input_name: str,
    warnings: list[str],
    node_name: str,
    *,
    strict_relative: bool = False,
) -> tuple[str, ...]:
    input_payload = _mapping_value(node_payload, "input")
    if input_payload is None:
        warnings.append(f"{node_name} ノードの input 情報を解釈できません")
        return ()

    required_payload = _mapping_value(input_payload, "required") or {}
    optional_payload = _mapping_value(input_payload, "optional") or {}
    raw_choices = required_payload.get(input_name, optional_payload.get(input_name))
    if raw_choices is None:
        warnings.append(f"{node_name}.{input_name} の選択肢がありません")
        return ()

    candidates = _candidate_strings(raw_choices)
    is_safe = _is_safe_relative_reference if strict_relative else _is_safe_reference
    safe_choice_set: set[str] = set()
    rejected_count = 0
    for candidate in candidates:
        normalized = candidate.strip().replace("\\", "/") if strict_relative else candidate.strip()
        if normalized and is_safe(normalized):
            safe_choice_set.add(normalized)
        elif candidate.strip():
            rejected_count += 1
    safe_choices = sorted(safe_choice_set, key=str.casefold)
    if rejected_count:
        warnings.append(f"{node_name}.{input_name} に不正または空の選択肢があります")
    return tuple(safe_choices)


def _candidate_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidates: list[str] = []
        for item in value:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
                candidates.extend(_candidate_strings(item))
        return candidates
    return []


def _is_safe_reference(value: str) -> bool:
    normalized = value.strip()
    return bool(normalized) and not posixpath.isabs(normalized) and not ntpath.isabs(normalized)


def _is_safe_relative_reference(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    return _is_safe_reference(normalized) and all(
        part not in {"", ".", ".."} for part in normalized.split("/")
    )


def _mapping_value(payload: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
