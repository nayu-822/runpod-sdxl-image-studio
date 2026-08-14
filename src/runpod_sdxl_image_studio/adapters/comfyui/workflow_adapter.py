"""Build the fixed, API-format SDXL txt2img workflow."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    WorkflowBindingError,
    WorkflowTemplateError,
    WorkflowValidationError,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings

SAFE_FILENAME_PREFIX = "runpod_sdxl_image_studio"
CHECKPOINT_NODE_ID = "4"
KSampler_NODE_ID = "3"
POSITIVE_CLIP_NODE_ID = "6"
NEGATIVE_CLIP_NODE_ID = "7"
VAE_DECODE_NODE_ID = "8"
EXTERNAL_VAE_NODE_ID = "vae_external"
DEFAULT_REQUIRED_NODE_CLASSES = (
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
)
REQUIRED_BINDING_FIELDS = (
    "checkpoint_name",
    "positive_prompt",
    "negative_prompt",
    "seed",
    "steps",
    "cfg_scale",
    "sampler_name",
    "scheduler_name",
    "width",
    "height",
    "batch_size",
    "filename_prefix",
)


class WorkflowAdapter:
    """Apply typed settings to one repository-controlled workflow definition."""

    def __init__(self, template_definition: Mapping[str, object]) -> None:
        self._template_definition = copy.deepcopy(dict(template_definition))

    def build_txt2img_workflow(self, settings: GenerationSettings) -> dict[str, object]:
        return build_txt2img_workflow(self._template_definition, settings)


def build_txt2img_workflow(
    template: Mapping[str, object],
    settings: GenerationSettings,
) -> dict[str, object]:
    """Deep-copy and bind settings into a validated fixed API workflow."""

    workflow_payload = template.get("workflow")
    if workflow_payload is None:
        metadata_keys = {
            "template_id",
            "schema_version",
            "workflow_version",
            "workflow_file",
            "required_node_classes",
            "bindings",
        }
        workflow_payload = {
            key: value for key, value in template.items() if key not in metadata_keys
        }
    if not isinstance(workflow_payload, Mapping):
        raise WorkflowTemplateError("workflow payload must be a mapping")
    workflow = copy.deepcopy(dict(workflow_payload))

    required_classes = _required_classes(template.get("required_node_classes"))
    present_classes = {
        node_payload.get("class_type")
        for node_payload in workflow.values()
        if isinstance(node_payload, Mapping)
    }
    missing_classes = [
        node_class for node_class in required_classes if node_class not in present_classes
    ]
    if missing_classes:
        raise WorkflowTemplateError("required workflow node class is missing")

    bindings = template.get("bindings")
    if not isinstance(bindings, Mapping):
        raise WorkflowTemplateError("workflow bindings are missing")
    if any(field_name not in bindings for field_name in REQUIRED_BINDING_FIELDS):
        raise WorkflowTemplateError("workflow bindings are incomplete")
    values: dict[str, object] = {
        "checkpoint_name": settings.checkpoint_name,
        "positive_prompt": settings.positive_prompt,
        "negative_prompt": settings.negative_prompt,
        "seed": settings.seed,
        "steps": settings.steps,
        "cfg_scale": settings.cfg_scale,
        "sampler_name": settings.sampler_name,
        "scheduler_name": settings.scheduler_name,
        "width": settings.width,
        "height": settings.height,
        "batch_size": settings.batch_size,
        "filename_prefix": SAFE_FILENAME_PREFIX,
    }
    for field_name, binding in bindings.items():
        if field_name not in values:
            continue
        path = _binding_path(binding, field_name)
        _set_binding(workflow, path, values[field_name])

    _apply_external_vae(workflow, settings.vae_name)
    _apply_lora_chain(workflow, settings)
    _apply_clip_skip(workflow, settings.clip_skip)
    if settings.hires_fix:
        _apply_hires_fix(workflow, settings)
    if settings.final_upscale:
        _apply_final_upscale(workflow, settings.final_upscale_model)

    try:
        json.dumps(workflow)
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError("workflow is not JSON serializable") from exc
    return workflow


def _apply_external_vae(workflow: dict[str, object], vae_name: str | None) -> None:
    if vae_name is None:
        return
    if EXTERNAL_VAE_NODE_ID in workflow:
        raise WorkflowTemplateError("reserved external VAE node id is already in use")
    workflow[EXTERNAL_VAE_NODE_ID] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": vae_name},
    }
    _set_binding(
        workflow,
        (VAE_DECODE_NODE_ID, "inputs", "vae"),
        [EXTERNAL_VAE_NODE_ID, 0],
    )


def _apply_lora_chain(workflow: dict[str, object], settings: GenerationSettings) -> None:
    if not settings.loras:
        return
    model_link: list[object] = [CHECKPOINT_NODE_ID, 0]
    clip_link: list[object] = [CHECKPOINT_NODE_ID, 1]
    for index, lora in enumerate(sorted(settings.loras, key=lambda item: item.order)):
        node_id = f"lora_{index:03d}"
        if node_id in workflow:
            raise WorkflowTemplateError("reserved LoRA node id is already in use")
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora.name,
                "strength_model": lora.model_strength,
                "strength_clip": lora.clip_strength,
                "model": model_link,
                "clip": clip_link,
            },
        }
        model_link = [node_id, 0]
        clip_link = [node_id, 1]
    _set_binding(workflow, (KSampler_NODE_ID, "inputs", "model"), model_link)
    _set_binding(workflow, (POSITIVE_CLIP_NODE_ID, "inputs", "clip"), clip_link)
    _set_binding(workflow, (NEGATIVE_CLIP_NODE_ID, "inputs", "clip"), clip_link)


def _apply_clip_skip(workflow: dict[str, object], clip_skip: int) -> None:
    if clip_skip == 1:
        return
    node_id = "clip_skip"
    if node_id in workflow:
        raise WorkflowTemplateError("reserved CLIP skip node id is already in use")
    positive = _get_input(workflow, POSITIVE_CLIP_NODE_ID, "clip")
    workflow[node_id] = {
        "class_type": "CLIPSetLastLayer",
        "inputs": {"clip": positive, "stop_at_clip_layer": -clip_skip},
    }
    link = [node_id, 0]
    _set_binding(workflow, (POSITIVE_CLIP_NODE_ID, "inputs", "clip"), link)
    _set_binding(workflow, (NEGATIVE_CLIP_NODE_ID, "inputs", "clip"), link)


def _apply_hires_fix(workflow: dict[str, object], settings: GenerationSettings) -> None:
    if "hires_latent" in workflow or "hires_sampler" in workflow:
        raise WorkflowTemplateError("reserved Hires.fix node id is already in use")
    model = _get_input(workflow, KSampler_NODE_ID, "model")
    positive = _get_input(workflow, KSampler_NODE_ID, "positive")
    negative = _get_input(workflow, KSampler_NODE_ID, "negative")
    workflow["hires_latent"] = {
        "class_type": "LatentUpscale",
        "inputs": {
            "upscale_method": "nearest-exact",
            "width": int(settings.width * settings.hires_scale),
            "height": int(settings.height * settings.hires_scale),
            "crop": "disabled",
            "samples": [KSampler_NODE_ID, 0],
        },
    }
    workflow["hires_sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": settings.seed,
            "steps": settings.steps,
            "cfg": settings.cfg_scale,
            "sampler_name": settings.sampler_name,
            "scheduler": settings.scheduler_name,
            "denoise": settings.hires_denoise,
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent_image": ["hires_latent", 0],
        },
    }
    _set_binding(workflow, (VAE_DECODE_NODE_ID, "inputs", "samples"), ["hires_sampler", 0])


def _apply_final_upscale(workflow: dict[str, object], model_name: str | None) -> None:
    if "final_upscale_loader" in workflow or "final_upscale" in workflow:
        raise WorkflowTemplateError("reserved final upscale node id is already in use")
    workflow["final_upscale_loader"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": model_name or "4x-UltraSharp.pth"},
    }
    workflow["final_upscale"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["final_upscale_loader", 0],
            "image": [VAE_DECODE_NODE_ID, 0],
        },
    }
    _set_binding(workflow, ("9", "inputs", "images"), ["final_upscale", 0])


def _get_input(workflow: dict[str, object], node_id: str, input_name: str) -> object:
    node_payload = workflow.get(node_id)
    if not isinstance(node_payload, dict):
        raise WorkflowBindingError(f"workflow node {node_id} is missing")
    inputs = node_payload.get("inputs")
    if not isinstance(inputs, dict) or input_name not in inputs:
        raise WorkflowBindingError(f"workflow input {node_id}.inputs.{input_name} is missing")
    return inputs[input_name]


def _required_classes(value: object) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_REQUIRED_NODE_CLASSES
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WorkflowTemplateError("required_node_classes must be a list")
    classes = tuple(item for item in value if isinstance(item, str) and item)
    if not classes:
        raise WorkflowTemplateError("required_node_classes cannot be empty")
    return classes


def _binding_path(value: object, field_name: object) -> tuple[str, str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WorkflowBindingError(f"binding for {field_name} is invalid")
    path = tuple(item for item in value if isinstance(item, str))
    if len(path) != 3 or path[1] != "inputs":
        raise WorkflowBindingError(f"binding for {field_name} is invalid")
    return path


def _set_binding(workflow: dict[str, object], path: tuple[str, str, str], value: object) -> None:
    node_id, _, input_name = path
    node_payload = workflow.get(node_id)
    if not isinstance(node_payload, dict):
        raise WorkflowBindingError(f"workflow node {node_id} is missing")
    inputs = node_payload.get("inputs")
    if not isinstance(inputs, dict) or input_name not in inputs:
        raise WorkflowBindingError(f"workflow input {node_id}.inputs.{input_name} is missing")
    inputs[input_name] = value
