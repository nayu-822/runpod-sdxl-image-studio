"""Fixed, typed workflow builders for image and latent upscale jobs."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import WorkflowTemplateError
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.upscale import UpscaleMethod
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSettingsSnapshot


class UpscaleWorkflowAdapter:
    """Bind only known values into repository-controlled API workflows."""

    def __init__(
        self, image_template: Mapping[str, object], latent_template: Mapping[str, object]
    ) -> None:
        self._image_template = copy.deepcopy(dict(image_template))
        self._latent_template = copy.deepcopy(dict(latent_template))

    def build_image_upscale_workflow(
        self,
        source_image: str,
        settings: UpscaleSettingsSnapshot,
        *,
        filename_prefix: str = "runpod_sdxl_image_studio/upscaled",
    ) -> dict[str, object]:
        if settings.method is not UpscaleMethod.IMAGE or not settings.upscaler_name:
            raise WorkflowTemplateError("image workflow requires an image upscale snapshot")
        workflow = _workflow(self._image_template)
        _set(workflow, "1", "image", source_image)
        _set(workflow, "2", "model_name", settings.upscaler_name)
        _set(workflow, "5", "width", settings.target_width)
        _set(workflow, "5", "height", settings.target_height)
        _set(workflow, "4", "filename_prefix", filename_prefix)
        _validate_json(workflow)
        return workflow

    def build_latent_upscale_workflow(
        self,
        source_image: str,
        generation: GenerationSettingsSnapshot,
        settings: UpscaleSettingsSnapshot,
        *,
        filename_prefix: str = "runpod_sdxl_image_studio/upscaled",
    ) -> dict[str, object]:
        if settings.method is not UpscaleMethod.LATENT or settings.denoise is None:
            raise WorkflowTemplateError("latent workflow requires a latent upscale snapshot")
        workflow = _workflow(self._latent_template)
        values = {
            ("1", "image"): source_image,
            ("2", "ckpt_name"): generation.checkpoint_name,
            ("3", "text"): generation.positive_prompt,
            ("4", "text"): generation.negative_prompt,
            ("6", "width"): settings.target_width,
            ("6", "height"): settings.target_height,
            ("7", "seed"): generation.seed,
            ("7", "steps"): generation.steps,
            ("7", "cfg"): generation.cfg_scale,
            ("7", "sampler_name"): generation.sampler_name,
            ("7", "scheduler"): generation.scheduler_name,
            ("7", "denoise"): settings.denoise,
            ("10", "filename_prefix"): filename_prefix,
        }
        for (node_id, input_name), value in values.items():
            _set(workflow, node_id, input_name, value)
        model_link: list[object] = ["2", 0]
        clip_link: list[object] = ["2", 1]
        for index, lora in enumerate(sorted(generation.loras, key=lambda item: item.order)):
            node_id = f"lora_{index:03d}"
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
        _set(workflow, "3", "clip", clip_link)
        _set(workflow, "4", "clip", clip_link)
        _set(workflow, "7", "model", model_link)
        if generation.vae_name is not None:
            workflow["vae_external"] = {
                "class_type": "VAELoader",
                "inputs": {"vae_name": generation.vae_name},
            }
            _set(workflow, "5", "vae", ["vae_external", 0])
            _set(workflow, "8", "vae", ["vae_external", 0])
        _validate_json(workflow)
        return workflow


def _workflow(template: Mapping[str, object]) -> dict[str, object]:
    payload = template.get("workflow", template)
    if not isinstance(payload, Mapping):
        raise WorkflowTemplateError("workflow payload must be a mapping")
    result = copy.deepcopy(dict(payload))
    required = template.get("required_node_classes")
    if isinstance(required, list):
        classes = {node.get("class_type") for node in result.values() if isinstance(node, Mapping)}
        if any(node_class not in classes for node_class in required):
            raise WorkflowTemplateError("required upscale node class is missing")
    return result


def _set(workflow: dict[str, object], node_id: str, input_name: str, value: object) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
        raise WorkflowTemplateError("upscale workflow binding is missing")
    inputs = node["inputs"]
    assert isinstance(inputs, dict)
    if input_name not in inputs:
        raise WorkflowTemplateError("upscale workflow input is missing")
    inputs[input_name] = value


def _validate_json(workflow: Mapping[str, object]) -> None:
    try:
        json.dumps(workflow)
    except (TypeError, ValueError) as exc:
        raise WorkflowTemplateError("upscale workflow is not JSON serializable") from exc


__all__ = ["UpscaleWorkflowAdapter"]
