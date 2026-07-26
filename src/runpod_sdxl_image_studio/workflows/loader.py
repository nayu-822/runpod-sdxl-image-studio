"""Load versioned workflow definitions and API-format templates from Git."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import WorkflowTemplateError


@dataclass(frozen=True)
class LoadedWorkflowTemplate:
    definition: Mapping[str, object]
    workflow: Mapping[str, object]

    def as_mapping(self) -> dict[str, object]:
        return {**self.definition, "workflow": self.workflow}


def load_txt2img_template(root_dir: Path | None = None) -> LoadedWorkflowTemplate:
    """Load the repository-controlled SDXL txt2img definition and template."""

    repository_root = root_dir or Path(__file__).resolve().parents[3]
    definition_path = repository_root / "workflows" / "definitions" / "sdxl_txt2img.json"
    definition = _read_json_object(definition_path)
    workflow_file = definition.get("workflow_file")
    if not isinstance(workflow_file, str) or not workflow_file:
        raise WorkflowTemplateError("workflow definition does not specify a template file")
    workflow_path = repository_root / workflow_file
    workflow = _read_json_object(workflow_path)
    return LoadedWorkflowTemplate(definition=definition, workflow=workflow)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowTemplateError("workflow template could not be loaded") from exc
    if not isinstance(payload, dict):
        raise WorkflowTemplateError("workflow template must be a JSON object")
    return payload
