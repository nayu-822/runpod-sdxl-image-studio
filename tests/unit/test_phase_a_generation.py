"""Phase A unit coverage for workflow options and browser-date storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import gradio as gr
import pytest
from PIL import Image

from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import (
    WorkflowAdapter,
    WorkflowTemplateError,
    _apply_final_upscale,
)
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationKind
from runpod_sdxl_image_studio.domain.generation_settings import (
    CURRENT_WORKFLOW_TEMPLATE_VERSION,
    LEGACY_WORKFLOW_TEMPLATE_VERSION,
    GenerationSettings,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.ui.tabs.system_tab import build_generation_tab
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template


def _png(size: tuple[int, int] = (8, 8)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "blue").save(output, format="PNG")
    return output.getvalue()


def _settings(**updates: object) -> GenerationSettings:
    values: dict[str, object] = {
        "positive_prompt": "a test image",
        "negative_prompt": "blurry",
        "checkpoint_name": "sdxl.safetensors",
        "sampler_name": "euler",
        "scheduler_name": "normal",
        "seed": 123,
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg_scale": 5.5,
    }
    values.update(updates)
    return GenerationSettings(**values)


def test_phase_a_workflow_binds_batch_and_optional_nodes() -> None:
    settings = _settings(
        batch_size=3,
        clip_skip=2,
        hires_fix=True,
        hires_scale=1.5,
        hires_resize_method="lanczos",
        hires_steps=17,
        hires_cfg_scale=6.25,
        hires_sampler_name="heun",
        hires_scheduler_name="karras",
        hires_denoise=0.35,
        final_upscale=True,
        final_upscale_model="4x-UltraSharp.pth",
        loras=(
            LoraSetting(name="style-one.safetensors", order=0, model_strength=0.8),
            LoraSetting(name="style-two.safetensors", order=1, model_strength=0.6),
        ),
    )
    workflow = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(
        settings
    )

    assert workflow["5"]["inputs"]["batch_size"] == 3  # type: ignore[index]
    assert workflow["lora_001"]["inputs"]["model"] == ["lora_000", 0]  # type: ignore[index]
    assert workflow["6"]["inputs"]["clip"] == ["clip_skip", 0]  # type: ignore[index]
    assert workflow["hires_scale"]["inputs"]["upscale_method"] == "lanczos"  # type: ignore[index]
    assert workflow["hires_scale"]["inputs"]["scale_by"] == 1.5  # type: ignore[index]
    assert workflow["hires_sampler"]["inputs"]["steps"] == 17  # type: ignore[index]
    assert workflow["hires_sampler"]["inputs"]["cfg"] == 6.25  # type: ignore[index]
    assert workflow["hires_sampler"]["inputs"]["sampler_name"] == "heun"  # type: ignore[index]
    assert workflow["hires_sampler"]["inputs"]["scheduler"] == "karras"  # type: ignore[index]
    assert workflow["hires_sampler"]["inputs"]["denoise"] == 0.35  # type: ignore[index]
    assert workflow["hires_decode"]["inputs"]["samples"] == ["hires_sampler", 0]  # type: ignore[index]
    assert workflow["final_upscale_loader"]["inputs"]["model_name"] == "4x-UltraSharp.pth"  # type: ignore[index]
    assert workflow["9"]["inputs"]["images"] == ["final_upscale", 0]  # type: ignore[index]

    legacy_workflow = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(
        _settings(
            workflow_template_version=LEGACY_WORKFLOW_TEMPLATE_VERSION,
            hires_fix=True,
            hires_scale=1.5,
            hires_denoise=0.35,
        )
    )
    assert legacy_workflow["hires_latent"]["class_type"] == "LatentUpscale"  # type: ignore[index]
    assert legacy_workflow["hires_latent"]["inputs"]["width"] == 1536  # type: ignore[index]
    assert legacy_workflow["hires_latent"]["inputs"]["height"] == 1536  # type: ignore[index]
    assert legacy_workflow["hires_sampler"]["inputs"]["steps"] == 28  # type: ignore[index]
    assert legacy_workflow["8"]["inputs"]["samples"] == ["hires_sampler", 0]  # type: ignore[index]
    assert "hires_scale" not in legacy_workflow

    without_optional = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(
        _settings(batch_size=2)
    )
    assert "clip_skip" not in without_optional
    assert "hires_sampler" not in without_optional
    assert "final_upscale" not in without_optional
    assert without_optional["9"]["inputs"]["images"] == ["8", 0]  # type: ignore[index]


def test_final_upscale_requires_an_explicit_model_and_never_uses_a_default() -> None:
    assert _settings(final_upscale=False).final_upscale_model is None

    with pytest.raises(ValueError, match="final_upscale_model"):
        _settings(final_upscale=True)

    with pytest.raises(WorkflowTemplateError, match="final upscale model is required"):
        _apply_final_upscale({}, None)


def test_phase_a_snapshot_round_trip_preserves_new_generation_options() -> None:
    settings = _settings(
        batch_size=4,
        clip_skip=3,
        hires_fix=True,
        hires_scale=1.75,
        hires_resize_method="bicubic",
        hires_steps=19,
        hires_cfg_scale=7.25,
        hires_sampler_name="heun",
        hires_scheduler_name="karras",
        hires_denoise=0.3,
        final_upscale=True,
        final_upscale_model="4x-UltraSharp.pth",
        client_local_date="2026-08-13",
    )

    snapshot = GenerationSettingsSnapshot.from_settings(settings)
    restored = GenerationSettingsSnapshot.from_json(snapshot.to_json()).to_generation_settings()

    assert restored.batch_size == 4
    assert restored.clip_skip == 3
    assert restored.hires_fix is True
    assert restored.hires_scale == 1.75
    assert restored.hires_resize_method == "bicubic"
    assert restored.hires_steps == 19
    assert restored.hires_cfg_scale == 7.25
    assert restored.hires_sampler_name == "heun"
    assert restored.hires_scheduler_name == "karras"
    assert restored.hires_denoise == 0.3
    assert restored.final_upscale is True
    assert restored.final_upscale_model == "4x-UltraSharp.pth"
    assert restored.client_local_date == "2026-08-13"
    assert restored.workflow_template_version == CURRENT_WORKFLOW_TEMPLATE_VERSION


def test_phase_a_legacy_hires_version_survives_snapshot_reload_and_retry_rebuild() -> None:
    settings = _settings(
        workflow_template_version=LEGACY_WORKFLOW_TEMPLATE_VERSION,
        hires_fix=True,
        hires_scale=1.75,
        hires_denoise=0.3,
    )
    snapshot = GenerationSettingsSnapshot.from_settings(settings)
    reloaded = GenerationSettingsSnapshot.from_json(snapshot.to_json()).to_generation_settings()
    retry_snapshot = GenerationSettingsSnapshot.from_settings(reloaded)
    retried = GenerationSettingsSnapshot.from_json(
        retry_snapshot.to_json()
    ).to_generation_settings()

    assert reloaded.workflow_template_version == LEGACY_WORKFLOW_TEMPLATE_VERSION
    assert retried.workflow_template_version == LEGACY_WORKFLOW_TEMPLATE_VERSION
    workflow = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(retried)
    assert workflow["hires_latent"]["class_type"] == "LatentUpscale"  # type: ignore[index]


def test_phase_a_local_storage_uses_client_date_and_six_digit_exclusive_sequence(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    storage = LocalStorageAdapter(settings)
    created_at = datetime(2026, 8, 14, 23, 59, tzinfo=UTC)

    def store(index: int):
        return storage.store_image(
            _png(),
            UUID(int=index + 1),
            created_at,
            client_local_date="2026-08-13",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        stored = tuple(executor.map(store, range(8)))

    paths = sorted(item.path for item in stored)
    assert [path.name for path in paths] == [f"{index:06d}.png" for index in range(1, 9)]
    assert all(path.parent == tmp_path / "generations" / "2026-08-13" for path in paths)
    assert len({item.sha256 for item in stored}) == 1
    assert all(path.read_bytes() for path in paths)

    with pytest.raises(StorageError):
        storage.store_image(_png(), UUID(int=100), created_at, client_local_date="2026-02-30")


def test_phase_a_client_date_sequence_is_shared_by_generated_and_upscaled_images(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    storage = LocalStorageAdapter(settings)
    created_at = datetime(2026, 8, 14, 23, 59, tzinfo=UTC)

    generated = storage.store_image(
        _png(), UUID(int=101), created_at, client_local_date="2026-08-13"
    )
    upscaled = storage.store_image(
        _png(),
        UUID(int=102),
        created_at,
        kind=GenerationKind.UPSCALE,
        client_local_date="2026-08-13",
    )

    assert generated.path == tmp_path / "generations" / "2026-08-13" / "000001.png"
    assert upscaled.path == tmp_path / "generations" / "2026-08-13" / "000002.png"


def test_phase_a_generation_tab_exposes_batch_size_and_interactive_run_controls() -> None:
    with gr.Blocks():
        generation = build_generation_tab(max_loras=2)

    assert generation.batch_size.minimum == 1
    assert generation.batch_size.maximum == 4
    assert generation.interactive_poll_timer.value == 3.0
    assert generation.interactive_start_button is not None
    assert generation.interactive_cancel_button is not None
