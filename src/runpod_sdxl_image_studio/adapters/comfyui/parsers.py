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
    ComfyUISystemStats,
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
    )
    extracted: dict[str, tuple[str, ...]] = {
        "checkpoints": (),
        "vaes": (),
        "samplers": (),
        "schedulers": (),
        "loras": (),
        "upscale_models": (),
    }
    warnings: list[str] = []

    for node_name, input_name, result_name, display_name in node_specs:
        node_payload = object_info.nodes.get(node_name)
        if node_payload is None:
            warning = (
                f"{node_name} ノードが /object_info にありません（{display_name}一覧は空です）"
            )
            if warning not in warnings:
                warnings.append(warning)
            continue
        extracted[result_name] = _extract_choices(node_payload, input_name, warnings, node_name)

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
    )


def _extract_choices(
    node_payload: Mapping[str, object],
    input_name: str,
    warnings: list[str],
    node_name: str,
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
    safe_choices = sorted(
        {candidate.strip() for candidate in candidates if _is_safe_reference(candidate)},
        key=str.casefold,
    )
    rejected_count = sum(
        1 for candidate in candidates if candidate.strip() and not _is_safe_reference(candidate)
    )
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
