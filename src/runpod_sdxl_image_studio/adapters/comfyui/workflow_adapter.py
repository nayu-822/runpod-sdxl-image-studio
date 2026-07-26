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
DEFAULT_REQUIRED_NODE_CLASSES = (
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
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
        "filename_prefix": SAFE_FILENAME_PREFIX,
    }
    for field_name, binding in bindings.items():
        if field_name not in values:
            continue
        path = _binding_path(binding, field_name)
        _set_binding(workflow, path, values[field_name])

    try:
        json.dumps(workflow)
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError("workflow is not JSON serializable") from exc
    return workflow


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
