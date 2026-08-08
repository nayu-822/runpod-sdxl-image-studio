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
    sampler_id: str | None = None
    if len(samplers) != 1:
        unresolved.append("sampler_graph")
        sampler_node: Mapping[str, object] = {}
    else:
        sampler_id, sampler_node = samplers[0]
    inputs = _inputs(sampler_node)

    positive, positive_clip_chain, positive_clip_ok = _connected_text(nodes, inputs.get("positive"))
    negative, negative_clip_chain, negative_clip_ok = _connected_text(nodes, inputs.get("negative"))
    if positive is None:
        unresolved.append("positive_prompt")
    if negative is None:
        unresolved.append("negative_prompt")
    if not positive_clip_ok or not negative_clip_ok:
        unresolved.append("clip_graph")

    model_ref = _link(inputs.get("model"))
    checkpoint, loras, model_unresolved, model_chain = _model_chain(nodes, model_ref)
    if checkpoint is None:
        unresolved.append("checkpoint")
    unresolved.extend(model_unresolved)
    if (
        model_chain is None
        or positive_clip_chain is None
        or negative_clip_chain is None
        or positive_clip_chain != model_chain
        or negative_clip_chain != model_chain
    ):
        unresolved.append("clip_graph")

    latent_ref = _link(inputs.get("latent_image"))
    width, height = _latent_dimensions(nodes, latent_ref)
    if width is None:
        unresolved.append("width")
    if height is None:
        unresolved.append("height")

    vae_name = _external_vae(nodes, sampler_id, model_chain)
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
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
        and value[1] >= 0
    ):
        return str(value[0]), value[1]
    return None


def _connected_text(
    nodes: Mapping[str, Mapping[str, object]], value: object
) -> tuple[str | None, tuple[str, ...] | None, bool]:
    link = _link(value)
    if link is None or link[1] != 0:
        return None, None, False
    node = nodes.get(link[0])
    if node is None or node.get("class_type") != "CLIPTextEncode":
        return None, None, False
    text = _inputs(node).get("text")
    clip_ref = _link(_inputs(node).get("clip"))
    chain, ok = _clip_chain(nodes, clip_ref)
    return (text if isinstance(text, str) else None), chain, ok


def _model_chain(
    nodes: Mapping[str, Mapping[str, object]],
    model_ref: tuple[str, int] | None,
) -> tuple[str | None, tuple[LoraSetting, ...], tuple[str, ...], tuple[str, ...] | None]:
    if model_ref is None:
        return None, (), ("checkpoint",), None
    current, output_index = model_ref
    lora_nodes: list[Mapping[str, object]] = []
    lora_ids: list[str] = []
    visited: set[str] = set()
    unresolved: list[str] = []
    checkpoint: str | None = None
    checkpoint_id: str | None = None
    while current not in visited:
        visited.add(current)
        node = nodes.get(current)
        if node is None:
            unresolved.append("checkpoint")
            break
        class_type = node.get("class_type")
        node_inputs = _inputs(node)
        if class_type == "LoraLoader":
            if output_index != 0:
                unresolved.append("checkpoint")
                break
            lora_nodes.append(node)
            lora_ids.append(current)
            previous = _link(node_inputs.get("model"))
            if previous is None:
                unresolved.append("loras")
                break
            current, output_index = previous
            continue
        if class_type == "CheckpointLoaderSimple":
            if output_index != 0:
                unresolved.append("checkpoint")
                break
            checkpoint_id = current
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
    chain = (checkpoint_id, *reversed(lora_ids)) if checkpoint_id is not None else None
    return checkpoint, tuple(loras), tuple(dict.fromkeys(unresolved)), chain


def _clip_chain(
    nodes: Mapping[str, Mapping[str, object]],
    clip_ref: tuple[str, int] | None,
) -> tuple[tuple[str, ...] | None, bool]:
    """Trace the CLIP output back through the exact model/LoRA chain."""

    if clip_ref is None:
        return None, False
    current, output_index = clip_ref
    lora_ids: list[str] = []
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        node = nodes.get(current)
        if node is None:
            return None, False
        class_type = node.get("class_type")
        if class_type == "CheckpointLoaderSimple":
            return ((current, *reversed(lora_ids)), True) if output_index == 1 else (None, False)
        if class_type != "LoraLoader":
            return None, False
        if output_index != 1:
            return None, False
        lora_ids.append(current)
        previous = _link(_inputs(node).get("clip"))
        if previous is None:
            return None, False
        current, output_index = previous
    return None, False


def _latent_dimensions(
    nodes: Mapping[str, Mapping[str, object]],
    latent_ref: tuple[str, int] | None,
) -> tuple[int | None, int | None]:
    if latent_ref is None:
        return None, None
    node = nodes.get(latent_ref[0])
    if node is None or node.get("class_type") != "EmptyLatentImage" or latent_ref[1] != 0:
        return None, None
    inputs = _inputs(node)
    return _int_value(inputs.get("width")), _int_value(inputs.get("height"))


def _external_vae(
    nodes: Mapping[str, Mapping[str, object]],
    sampler_id: str | None,
    model_chain: tuple[str, ...] | None,
) -> str | None | object:
    """Resolve only the VAE on the selected KSampler execution path."""

    if sampler_id is None or model_chain is None:
        return _UNRESOLVED
    matches = [
        node_id
        for node_id, node in nodes.items()
        if node.get("class_type") == "VAEDecode"
        and _link(_inputs(node).get("samples")) == (sampler_id, 0)
    ]
    if len(matches) != 1:
        return _UNRESOLVED
    vae_ref = _link(_inputs(nodes[matches[0]]).get("vae"))
    if vae_ref is None:
        return _UNRESOLVED
    vae_node = nodes.get(vae_ref[0])
    if vae_node is None:
        return _UNRESOLVED
    if vae_node.get("class_type") == "VAELoader":
        if vae_ref[1] != 0:
            return _UNRESOLVED
        name = _safe_model_name(_inputs(vae_node).get("vae_name"))
        return name if name is not None else _UNRESOLVED
    if vae_node.get("class_type") == "CheckpointLoaderSimple":
        # CheckpointLoaderSimple output 2 is its VAE.  A different checkpoint
        # or a non-VAE output is not equivalent to the execution model.
        return None if vae_ref[0] == model_chain[0] and vae_ref[1] == 2 else _UNRESOLVED
    return _UNRESOLVED


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
