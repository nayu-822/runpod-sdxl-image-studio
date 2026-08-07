"""Parse the small, known ComfyUI prompt graph subset used by this app."""

from __future__ import annotations

import ntpath
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass

from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.metadata_import import (
    MetadataFieldResolution,
    MetadataFieldStatus,
    MetadataImportCandidate,
    MetadataSourceKind,
)

SUPPORTED_NODE_CLASSES = frozenset(
    {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "KSampler",
        "EmptyLatentImage",
        "LoraLoader",
        "VAELoader",
        "VAEDecode",
    }
)


@dataclass(frozen=True)
class ComfyPromptMetadataResult:
    candidate: MetadataImportCandidate
    warnings: tuple[str, ...]
    unresolved_fields: tuple[str, ...]


def parse_comfyui_prompt_metadata(
    prompt: Mapping[str, object] | Mapping[object, object],
) -> ComfyPromptMetadataResult:
    """Return a normalized candidate; arbitrary workflow values are ignored."""

    nodes = _normalize_nodes(prompt)
    warnings: list[str] = []
    unresolved: list[str] = []
    unsupported = [
        payload.get("class_type")
        for payload in nodes.values()
        if isinstance(payload, Mapping)
        and isinstance(payload.get("class_type"), str)
        and payload.get("class_type") not in SUPPORTED_NODE_CLASSES
    ]
    if unsupported:
        warnings.append("metadata_import_unknown_node")

    samplers = [
        (node_id, node) for node_id, node in nodes.items() if node.get("class_type") == "KSampler"
    ]
    if len(samplers) != 1:
        unresolved.append("sampler_graph")
        sampler_node: Mapping[str, object] = {}
    else:
        sampler_node = samplers[0][1]
    inputs = _inputs(sampler_node)

    positive = _connected_text(nodes, inputs.get("positive"))
    negative = _connected_text(nodes, inputs.get("negative"))
    if positive is None:
        unresolved.append("positive_prompt")
    if negative is None:
        unresolved.append("negative_prompt")

    model_ref = _link(inputs.get("model"))
    checkpoint, loras, model_unresolved = _model_chain(nodes, model_ref)
    if checkpoint is None:
        unresolved.append("checkpoint")
    unresolved.extend(model_unresolved)

    latent_ref = _link(inputs.get("latent_image"))
    width, height = _latent_dimensions(nodes, latent_ref)
    if width is None:
        unresolved.append("width")
    if height is None:
        unresolved.append("height")

    vae_name = _external_vae(nodes, inputs)
    # No VAELoader is a valid checkpoint-internal VAE. An explicitly present but
    # disconnected external loader is not guessed as the execution VAE.
    if vae_name is _UNRESOLVED:
        unresolved.append("vae")
        vae_name = None

    values = {
        "positive_prompt": positive,
        "negative_prompt": negative,
        "seed": _int_value(inputs.get("seed")),
        "width": width,
        "height": height,
        "steps": _int_value(inputs.get("steps")),
        "cfg_scale": _float_value(inputs.get("cfg")),
        "sampler_name": _string_value(inputs.get("sampler_name")),
        "scheduler_name": _string_value(inputs.get("scheduler")),
        "checkpoint_name": checkpoint,
        "vae_name": vae_name,
        "loras": tuple(loras),
    }
    for field_name in (
        "seed",
        "steps",
        "cfg_scale",
        "sampler_name",
        "scheduler_name",
    ):
        if values[field_name] is None:
            unresolved.append(field_name)

    unique_unresolved = tuple(dict.fromkeys(unresolved))
    resolutions = tuple(
        MetadataFieldResolution(
            field_name=name,
            status=(
                MetadataFieldStatus.UNRESOLVED
                if name in unique_unresolved
                else MetadataFieldStatus.RESOLVED
            ),
            value=values.get(name),
        )
        for name in (
            "positive_prompt",
            "negative_prompt",
            "seed",
            "width",
            "height",
            "steps",
            "cfg_scale",
            "sampler_name",
            "scheduler_name",
            "checkpoint_name",
            "vae_name",
            "loras",
        )
    )
    candidate = MetadataImportCandidate(
        source_kind=MetadataSourceKind.COMFYUI_PROMPT,
        positive_prompt=positive,
        negative_prompt=negative,
        seed=values["seed"],
        width=width,
        height=height,
        steps=values["steps"],
        cfg_scale=values["cfg_scale"],
        sampler_name=values["sampler_name"],
        scheduler_name=values["scheduler_name"],
        checkpoint_name=checkpoint,
        vae_name=vae_name,
        loras=tuple(loras),
        unresolved_fields=unique_unresolved,
        warnings=tuple(dict.fromkeys(warnings)),
        resolutions=resolutions,
    )
    return ComfyPromptMetadataResult(candidate, candidate.warnings, unique_unresolved)


_UNRESOLVED = object()


def _normalize_nodes(
    prompt: Mapping[str, object] | Mapping[object, object],
) -> dict[str, Mapping[str, object]]:
    normalized: dict[str, Mapping[str, object]] = {}
    for node_id, raw_node in prompt.items():
        if not isinstance(node_id, (str, int)) or not isinstance(raw_node, Mapping):
            continue
        class_type = raw_node.get("class_type")
        inputs = raw_node.get("inputs")
        if not isinstance(class_type, str) or not isinstance(inputs, Mapping):
            continue
        normalized[str(node_id)] = {"class_type": class_type, "inputs": dict(inputs)}
    return normalized


def _inputs(node: Mapping[str, object]) -> Mapping[str, object]:
    value = node.get("inputs")
    return value if isinstance(value, Mapping) else {}


def _link(value: object) -> tuple[str, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 1 and isinstance(value[0], (str, int)):
        return str(value[0]), int(value[1]) if len(value) > 1 and isinstance(value[1], int) else 0
    return None


def _connected_text(nodes: Mapping[str, Mapping[str, object]], value: object) -> str | None:
    link = _link(value)
    if link is None:
        return None
    node = nodes.get(link[0])
    if node is None or node.get("class_type") != "CLIPTextEncode":
        return None
    text = _inputs(node).get("text")
    return text if isinstance(text, str) else None


def _model_chain(
    nodes: Mapping[str, Mapping[str, object]],
    model_ref: tuple[str, int] | None,
) -> tuple[str | None, tuple[LoraSetting, ...], tuple[str, ...]]:
    if model_ref is None:
        return None, (), ("checkpoint",)
    current = model_ref[0]
    lora_nodes: list[Mapping[str, object]] = []
    visited: set[str] = set()
    unresolved: list[str] = []
    checkpoint: str | None = None
    while current not in visited:
        visited.add(current)
        node = nodes.get(current)
        if node is None:
            unresolved.append("checkpoint")
            break
        class_type = node.get("class_type")
        node_inputs = _inputs(node)
        if class_type == "LoraLoader":
            lora_nodes.append(node)
            previous = _link(node_inputs.get("model"))
            if previous is None:
                unresolved.append("loras")
                break
            current = previous[0]
            continue
        if class_type == "CheckpointLoaderSimple":
            checkpoint = _safe_model_name(node_inputs.get("ckpt_name"))
            if checkpoint is None:
                unresolved.append("checkpoint")
            break
        unresolved.append("checkpoint")
        break
    if current in visited and nodes.get(current, {}).get("class_type") == "LoraLoader":
        unresolved.append("loras")

    loras: list[LoraSetting] = []
    for order, node in enumerate(reversed(lora_nodes)):
        node_inputs = _inputs(node)
        name = _safe_model_name(node_inputs.get("lora_name"))
        model_strength = _float_value(node_inputs.get("strength_model"))
        clip_strength = _float_value(node_inputs.get("strength_clip"))
        if name is None or model_strength is None or clip_strength is None:
            unresolved.append("loras")
            continue
        try:
            loras.append(
                LoraSetting(
                    name=name,
                    model_strength=model_strength,
                    clip_strength=clip_strength,
                    order=order,
                )
            )
        except ValueError:
            unresolved.append("loras")
    return checkpoint, tuple(loras), tuple(dict.fromkeys(unresolved))


def _latent_dimensions(
    nodes: Mapping[str, Mapping[str, object]],
    latent_ref: tuple[str, int] | None,
) -> tuple[int | None, int | None]:
    if latent_ref is None:
        return None, None
    node = nodes.get(latent_ref[0])
    if node is None or node.get("class_type") != "EmptyLatentImage":
        return None, None
    inputs = _inputs(node)
    return _int_value(inputs.get("width")), _int_value(inputs.get("height"))


def _external_vae(
    nodes: Mapping[str, Mapping[str, object]], sampler_inputs: Mapping[str, object]
) -> str | None | object:
    # Locate the VAEDecode node fed by the sampler and use only its VAE link.
    sample_ref = _link(sampler_inputs.get("latent_image"))
    del sample_ref
    for node in nodes.values():
        if node.get("class_type") != "VAEDecode":
            continue
        vae_ref = _link(_inputs(node).get("vae"))
        if vae_ref is None:
            return None
        vae_node = nodes.get(vae_ref[0])
        if vae_node is None:
            return _UNRESOLVED
        if vae_node.get("class_type") == "VAELoader":
            name = _safe_model_name(_inputs(vae_node).get("vae_name"))
            return name if name is not None else _UNRESOLVED
        if vae_node.get("class_type") == "CheckpointLoaderSimple":
            return None
        return _UNRESOLVED
    return None


def _safe_model_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("\\", "/")
    if (
        not normalized
        or "\x00" in normalized
        or posixpath.isabs(normalized)
        or ntpath.isabs(value)
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        return None
    return normalized


def _string_value(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


__all__ = [
    "ComfyPromptMetadataResult",
    "SUPPORTED_NODE_CLASSES",
    "parse_comfyui_prompt_metadata",
]
