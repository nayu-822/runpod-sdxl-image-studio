from __future__ import annotations

import json

import pytest

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.comfyui.parsers import (
    parse_capabilities,
    parse_object_info,
)
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import (
    WorkflowAdapter,
    build_txt2img_workflow,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.detailer import (
    DEFAULT_DETAILER_REGISTRY,
    FACE_DEFAULT_DETECTOR_MODEL,
    DetailerKind,
    DetailerSettings,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.system_status import ComfyUIStatus
from runpod_sdxl_image_studio.services.generation_preflight_service import (
    GenerationPreflightService,
)
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template


def _settings(**updates: object) -> GenerationSettings:
    values: dict[str, object] = {
        "positive_prompt": "base prompt",
        "negative_prompt": "bad",
        "seed": 123,
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg_scale": 5.5,
        "sampler_name": "euler",
        "scheduler_name": "normal",
        "checkpoint_name": "checkpoints/base.safetensors",
        "workflow_template_version": "2.2",
    }
    values.update(updates)
    return GenerationSettings(**values)


def _workflow(settings: GenerationSettings) -> dict[str, object]:
    return build_txt2img_workflow(load_txt2img_template().as_mapping(), settings)


def _face(**updates: object) -> DetailerSettings:
    return DEFAULT_DETAILER_REGISTRY.default_settings(DetailerKind.FACE).model_copy(update=updates)


def _capabilities(
    *, nodes: set[str] | None = None, detectors: tuple[str, ...] = ()
) -> ComfyUICapabilities:
    return ComfyUICapabilities(
        checkpoints=("checkpoints/base.safetensors",),
        vaes=("vae/base.safetensors",),
        samplers=("euler", "euler_ancestral"),
        schedulers=("normal",),
        loras=("loras/style.safetensors",),
        upscale_models=("upscalers/4x.pth",),
        available_node_classes=frozenset(
            nodes
            or {
                "CheckpointLoaderSimple",
                "CLIPTextEncode",
                "EmptyLatentImage",
                "KSampler",
                "VAEDecode",
                "SaveImage",
            }
        ),
        warnings=(),
        detector_models=detectors,
    )


class _StatusProvider:
    def __init__(self, capabilities: ComfyUICapabilities) -> None:
        self.status = ComfyUIStatus(
            is_connected=True,
            message="connected",
            checked_at=None,
            system_stats=None,
            capabilities=capabilities,
            warnings=(),
            error_summary=None,
        )

    async def get_status(self) -> ComfyUIStatus:
        return self.status


def test_detailer_defaults_are_registry_owned_and_detector_path_is_safe() -> None:
    defaults = _face()
    assert defaults.detector_model == FACE_DEFAULT_DETECTOR_MODEL
    assert defaults.guide_size == 768
    assert defaults.denoise == 0.22
    assert defaults.scheduler_name == "normal"
    assert DEFAULT_DETAILER_REGISTRY.require(DetailerKind.FACE).node_class == "FaceDetailer"

    with pytest.raises(ValueError):
        DetailerSettings(detector_model="../face.pt")
    with pytest.raises(ValueError):
        DetailerSettings(detector_model="C:/models/face.pt")
    with pytest.raises(ValueError):
        DetailerSettings(detector_model="bbox//face.pt")


def test_capabilities_extract_detector_choices_and_reject_unsafe_paths() -> None:
    info = parse_object_info(
        {
            "UltralyticsDetectorProvider": {
                "input": {
                    "required": {
                        "model_name": [["bbox/face_yolov8m.pt", "bbox\\hand.pt", "../escape.pt"]]
                    }
                }
            }
        }
    )
    capabilities = parse_capabilities(info)
    assert capabilities.detector_models == ("bbox/face_yolov8m.pt", "bbox/hand.pt")
    assert "UltralyticsDetectorProvider" in capabilities.available_node_classes


@pytest.mark.parametrize(
    ("hires_fix", "final_upscale", "expected_image"),
    [
        (False, False, ["detailer_face", 0]),
        (True, False, ["detailer_face", 0]),
        (True, True, ["final_upscale", 0]),
    ],
)
def test_face_pipeline_is_ordered_before_final_upscale(
    hires_fix: bool,
    final_upscale: bool,
    expected_image: list[object],
) -> None:
    settings = _settings(
        hires_fix=hires_fix,
        final_upscale=final_upscale,
        final_upscale_model="upscalers/4x.pth" if final_upscale else None,
        detailers=(_face(),),
    )
    workflow = _workflow(settings)
    assert workflow["9"]["inputs"]["images"] == expected_image  # type: ignore[index]
    assert workflow["detailer_face"]["inputs"]["image"] == (  # type: ignore[index]
        ["hires_decode", 0] if hires_fix else ["8", 0]
    )
    if final_upscale:
        assert workflow["final_upscale"]["inputs"]["image"] == ["detailer_face", 0]  # type: ignore[index]
    else:
        assert "final_upscale" not in workflow


def test_face_uses_effective_lora_clip_skip_and_external_vae_and_preserves_batch() -> None:
    workflow = _workflow(
        _settings(
            batch_size=4,
            clip_skip=2,
            vae_name="vae/base.safetensors",
            loras=(LoraSetting(name="loras/style.safetensors", order=0),),
            detailers=(_face(),),
        )
    )
    inputs = workflow["detailer_face"]["inputs"]  # type: ignore[index]
    assert inputs["model"] == ["lora_000", 0]
    assert inputs["clip"] == ["clip_skip", 0]
    assert inputs["vae"] == ["vae_external", 0]
    assert workflow["detailer_face_positive"]["inputs"]["clip"] == ["clip_skip", 0]  # type: ignore[index]
    assert workflow["detailer_face_positive"]["inputs"]["text"] == _face().positive_prompt  # type: ignore[index]
    assert workflow["5"]["inputs"]["batch_size"] == 4  # type: ignore[index]
    assert "SAMLoader" not in {
        node["class_type"] for node in workflow.values() if isinstance(node, dict)
    }


def test_face_off_does_not_add_custom_detailer_nodes() -> None:
    workflow = _workflow(_settings(detailers=()))
    assert not any(
        isinstance(node, dict)
        and node.get("class_type") in {"FaceDetailer", "UltralyticsDetectorProvider"}
        for node in workflow.values()
    )
    assert workflow["9"]["inputs"]["images"] == ["8", 0]  # type: ignore[index]


@pytest.mark.asyncio
async def test_face_preflight_is_optional_but_fails_closed_when_enabled(tmp_path) -> None:
    base = _capabilities()
    service = GenerationPreflightService(
        _StatusProvider(base),
        Settings(_env_file=None, checkpoint_dir=tmp_path / "checkpoints"),
    )
    off = await service.check(_settings(detailers=()))
    assert off.is_ready

    missing_nodes = await service.check(_settings(detailers=(_face(),)))
    assert not missing_nodes.is_ready
    assert any(issue.code == "face_detailer_nodes_missing" for issue in missing_nodes.errors)

    ready_nodes = set(base.available_node_classes) | {
        "FaceDetailer",
        "UltralyticsDetectorProvider",
    }
    ready_service = GenerationPreflightService(
        _StatusProvider(_capabilities(nodes=ready_nodes, detectors=(FACE_DEFAULT_DETECTOR_MODEL,))),
        Settings(_env_file=None, checkpoint_dir=tmp_path / "checkpoints"),
    )
    ready = await ready_service.check(_settings(detailers=(_face(),)))
    assert ready.is_ready

    no_detector = await ready_service.check(
        _settings(detailers=(_face(detector_model="bbox/missing.pt"),))
    )
    assert not no_detector.is_ready
    assert any(issue.code == "face_detector_missing" for issue in no_detector.errors)


def test_generation_snapshot_schema_two_roundtrip_and_schema_one_compatibility() -> None:
    settings = _settings(detailers=(_face(),))
    snapshot = GenerationSettingsSnapshot.from_settings(settings)
    assert snapshot.schema_version == 2
    assert snapshot.detailers[0].detector_model == FACE_DEFAULT_DETECTOR_MODEL
    restored = GenerationSettingsSnapshot.from_json(snapshot.to_json()).to_generation_settings()
    assert restored.detailers == settings.detailers

    legacy_payload = snapshot.model_dump(mode="json")
    legacy_payload["schema_version"] = 1
    legacy_payload.pop("detailers")
    legacy = GenerationSettingsSnapshot.from_json(json.dumps(legacy_payload))
    assert legacy.schema_version == 1
    assert legacy.detailers == ()


def test_workflow_two_point_one_keeps_pixel_hires_compatibility() -> None:
    settings = _settings(hires_fix=True, workflow_template_version="2.1")
    workflow = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(
        settings
    )
    assert "hires_decode" in workflow
    assert workflow["9"]["inputs"]["images"] == ["hires_decode", 0]  # type: ignore[index]
