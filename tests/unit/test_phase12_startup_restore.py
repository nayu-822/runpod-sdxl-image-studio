from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr
import pytest

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation_form_state import GenerationFormStateSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.model_transfer import ModelTransferStatus
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.jobs.startup_model_restore import (
    StartupModelRestoreRuntime,
    StartupRestoreState,
)
from runpod_sdxl_image_studio.services.generation_form_state_service import FormStateRestoreResult
from runpod_sdxl_image_studio.services.model_preparation_service import ModelPreparationResult
from runpod_sdxl_image_studio.ui.components.lora_editor import component_outputs
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    build_generation_tab,
    capability_refresh_outputs,
    make_startup_restore_handler,
)


def _snapshot() -> GenerationFormStateSnapshot:
    return GenerationFormStateSnapshot.from_ui(
        positive_prompt="positive",
        negative_prompt="negative",
        seed_mode="Fixed",
        seed=123,
        width=1024,
        height=1024,
        steps=28,
        cfg_scale=5.5,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="checkpoints/A.safetensors",
        vae_name="vae/V.safetensors",
        upscaler_name="upscalers/U.pth",
        loras=(),
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        remote_model_enabled=True,
        rclone_remote="drive",
        checkpoint_dir=tmp_path / "checkpoints",
        lora_dir=tmp_path / "loras",
        vae_dir=tmp_path / "vae",
        upscaler_dir=tmp_path / "upscalers",
    )


class _FormState:
    def __init__(self, snapshot: GenerationFormStateSnapshot | None) -> None:
        self.snapshot = snapshot

    def restore(self) -> FormStateRestoreResult:
        return FormStateRestoreResult(self.snapshot, "form_state")


class _Preparation:
    def __init__(self, job: object | None, missing: tuple[str, ...] = ()) -> None:
        self.job = job
        self.missing = missing
        self.prepare_calls = 0
        self.arguments: tuple[object, ...] | None = None

    async def prepare_previous_models(
        self,
        checkpoint: str | None,
        vae: str | None,
        loras: tuple[str, ...],
        upscaler: str | None,
    ) -> ModelPreparationResult:
        self.prepare_calls += 1
        self.arguments = (checkpoint, vae, loras, upscaler)
        return ModelPreparationResult(
            (self.job,) if self.job is not None else (),
            "queued",
            missing=self.missing,
        )

    def list_jobs(self, limit: int = 500) -> tuple[object, ...]:
        return (self.job,) if self.job is not None else ()


def _capabilities() -> CapabilityRefreshResult:
    return CapabilityRefreshResult(
        True,
        "refreshed",
        ComfyUICapabilities(
            checkpoints=("checkpoints/A.safetensors",),
            vaes=("vae/V.safetensors",),
            samplers=("euler",),
            schedulers=("normal",),
            loras=("loras/one.safetensors", "loras/two.safetensors"),
            upscale_models=("upscalers/U.pth",),
            available_node_classes=frozenset(),
            warnings=(),
        ),
    )


def _wait_for(runtime: StartupModelRestoreRuntime, state: StartupRestoreState) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if runtime.status().state is state:
            return
        time.sleep(0.02)
    raise AssertionError(f"startup restore did not reach {state}")


def test_fresh_pod_reapplies_form_only_after_model_terminal_and_visibility(
    tmp_path: Path,
) -> None:
    job = SimpleNamespace(
        id=uuid4(),
        status=ModelTransferStatus.PENDING,
        kind=SimpleNamespace(value="checkpoint"),
        remote_relative_path="checkpoints/A.safetensors",
    )
    preparation = _Preparation(job)
    runtime = StartupModelRestoreRuntime(
        _FormState(_snapshot()),
        preparation,  # type: ignore[arg-type]
        _settings(tmp_path),
        capability_refresh=lambda: _async_capabilities(),
        poll_interval_seconds=0.01,
    )
    runtime.start()
    _wait_for(runtime, StartupRestoreState.PREPARING_MODELS)
    assert runtime.status().snapshot is not None
    assert not runtime.is_ready
    deadline = time.monotonic() + 3.0
    while preparation.prepare_calls < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert preparation.prepare_calls == 1

    job.status = ModelTransferStatus.COMPLETED
    _wait_for(runtime, StartupRestoreState.READY)
    status = runtime.status()
    assert runtime.is_ready
    assert status.capabilities is not None
    assert not status.applied
    assert preparation.arguments == (
        "checkpoints/A.safetensors",
        "vae/V.safetensors",
        (),
        "upscalers/U.pth",
    )
    runtime.mark_applied()
    assert runtime.status().applied
    runtime.stop()


def test_missing_model_is_incomplete_without_auto_substitution_or_generation(
    tmp_path: Path,
) -> None:
    preparation = _Preparation(None, ("checkpoint:checkpoints/MISSING.safetensors",))
    runtime = StartupModelRestoreRuntime(
        _FormState(_snapshot()),
        preparation,  # type: ignore[arg-type]
        _settings(tmp_path),
        capability_refresh=lambda: _async_capabilities(),
    )
    runtime.start()
    _wait_for(runtime, StartupRestoreState.INCOMPLETE)
    status = runtime.status()
    assert status.missing == ("checkpoint:checkpoints/MISSING.safetensors",)
    assert "MISSING" in status.message
    assert preparation.prepare_calls == 1
    runtime.stop()


async def _async_capabilities() -> CapabilityRefreshResult:
    return _capabilities()


@pytest.mark.asyncio
async def test_startup_handler_applies_exact_form_and_visible_lora_rows_after_restore(
    tmp_path: Path,
) -> None:
    snapshot = GenerationFormStateSnapshot.from_ui(
        positive_prompt="restored positive",
        negative_prompt="restored negative",
        seed_mode="Fixed",
        seed=321,
        width=832,
        height=1216,
        steps=30,
        cfg_scale=6.5,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="checkpoints/A.safetensors",
        vae_name="vae/V.safetensors",
        upscaler_name="upscalers/U.pth",
        loras=(
            LoraSetting(
                name="loras/one.safetensors",
                model_strength=0.7,
                clip_strength=0.8,
                order=0,
            ),
            LoraSetting(
                name="loras/two.safetensors",
                model_strength=0.4,
                clip_strength=0.5,
                order=1,
            ),
        ),
    )
    job = SimpleNamespace(
        id=uuid4(),
        status=ModelTransferStatus.COMPLETED,
        kind=SimpleNamespace(value="checkpoint"),
        remote_relative_path="checkpoints/A.safetensors",
    )
    runtime = StartupModelRestoreRuntime(
        _FormState(snapshot),
        _Preparation(job),  # type: ignore[arg-type]
        _settings(tmp_path),
        capability_refresh=lambda: _async_capabilities(),
        poll_interval_seconds=0.01,
    )
    runtime.start()
    _wait_for(runtime, StartupRestoreState.READY)

    with gr.Blocks():
        generation = build_generation_tab(max_loras=3)
        handler = make_startup_restore_handler(
            runtime,
            SimpleNamespace(refresh_capabilities=_async_capabilities),
            generation,
        )
        result = await handler(None, None, None, None, None)

    capability_count = len(capability_refresh_outputs(generation))
    component_count = len(component_outputs(generation.lora_editor))
    component_start = 2 + capability_count - component_count - 2
    first_row = component_start
    second_row = component_start + 7
    assert result[2].value == "checkpoints/A.safetensors"
    assert result[3].value == "vae/V.safetensors"
    assert result[first_row + 1].value == "loras/one.safetensors"
    assert result[first_row + 2].value == 0.7
    assert result[first_row + 3].value == 0.8
    assert result[second_row + 1].value == "loras/two.safetensors"
    assert result[second_row + 2].value == 0.4
    assert result[second_row + 3].value == 0.5
    assert result[2 + capability_count] == "restored positive"
    assert result[3 + capability_count] == "restored negative"
    assert runtime.status().applied
    runtime.stop()
