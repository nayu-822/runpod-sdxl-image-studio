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

from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
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
    assert workflow["hires_sampler"]["inputs"]["denoise"] == 0.35  # type: ignore[index]
    assert workflow["final_upscale_loader"]["inputs"]["model_name"] == "4x-UltraSharp.pth"  # type: ignore[index]
    assert workflow["9"]["inputs"]["images"] == ["final_upscale", 0]  # type: ignore[index]

    without_optional = WorkflowAdapter(load_txt2img_template().as_mapping()).build_txt2img_workflow(
        _settings(batch_size=2)
    )
    assert "clip_skip" not in without_optional
    assert "hires_sampler" not in without_optional
    assert "final_upscale" not in without_optional
    assert without_optional["9"]["inputs"]["images"] == ["8", 0]  # type: ignore[index]


def test_phase_a_snapshot_round_trip_preserves_new_generation_options() -> None:
    settings = _settings(
        batch_size=4,
        clip_skip=3,
        hires_fix=True,
        final_upscale=True,
        final_upscale_model="4x-UltraSharp.pth",
        client_local_date="2026-08-13",
    )

    snapshot = GenerationSettingsSnapshot.from_settings(settings)
    restored = GenerationSettingsSnapshot.from_json(snapshot.to_json()).to_generation_settings()

    assert restored.batch_size == 4
    assert restored.clip_skip == 3
    assert restored.hires_fix is True
    assert restored.final_upscale is True
    assert restored.final_upscale_model == "4x-UltraSharp.pth"
    assert restored.client_local_date == "2026-08-13"
    assert restored.workflow_template_version == "2.0"


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
    assert all(
        path.parent == tmp_path / "generations" / "2026-08-13" / "generated" for path in paths
    )
    assert len({item.sha256 for item in stored}) == 1
    assert all(path.read_bytes() for path in paths)

    with pytest.raises(StorageError):
        storage.store_image(_png(), UUID(int=100), created_at, client_local_date="2026-02-30")


def test_phase_a_generation_tab_exposes_batch_size_and_interactive_run_controls() -> None:
    with gr.Blocks():
        generation = build_generation_tab(max_loras=2)

    assert generation.batch_size.minimum == 1
    assert generation.batch_size.maximum == 4
    assert generation.interactive_poll_timer.value == 3.0
    assert generation.interactive_start_button is not None
    assert generation.interactive_cancel_button is not None
