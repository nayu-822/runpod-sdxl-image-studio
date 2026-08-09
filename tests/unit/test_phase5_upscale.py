"""Phase 5 domain, source validation, queue atomicity, and workflow tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

import gradio as gr
import httpx
import pytest
import respx
from PIL import Image
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.pool import StaticPool

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.comfyui.upscale_workflow_adapter import (
    UpscaleWorkflowAdapter,
)
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base, GenerationModel
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
    GenerationFailureRepository,
    GenerationJobRepository,
    GenerationQueueRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepository,
    UpscaleSettingsRepositoryError,
)
from runpod_sdxl_image_studio.adapters.storage.generation_metadata_storage import (
    GenerationMetadataStorage,
)
from runpod_sdxl_image_studio.adapters.storage.history_thumbnail_storage import (
    HistoryThumbnailStorage,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_queue import (
    OptionalArtifactRepairCandidate,
    OptionalArtifactRepairOutcome,
)
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.job import GenerationJob
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleLoadLevel,
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
    estimate_load_level,
    resolve_output_size,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import (
    UpscaleSettingsSnapshot,
    UpscaleSnapshotError,
)
from runpod_sdxl_image_studio.jobs.generation_queue_worker import GenerationQueueWorker
from runpod_sdxl_image_studio.services import generation_recovery_service as recovery_module
from runpod_sdxl_image_studio.services import generation_service as generation_module
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
)
from runpod_sdxl_image_studio.services.generation_queue_service import GenerationQueueService
from runpod_sdxl_image_studio.services.generation_recovery_service import (
    GenerationRecoveryService,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.upscale_enqueue_service import (
    UpscaleEnqueueError,
    UpscaleEnqueueService,
)
from runpod_sdxl_image_studio.ui.tabs import upscale_tab as upscale_ui
from runpod_sdxl_image_studio.workflows.loader import load_workflow_template


def _settings() -> GenerationSettings:
    return GenerationSettings(
        positive_prompt="a cat",
        negative_prompt="blur",
        seed=42,
        width=512,
        height=512,
        steps=20,
        cfg_scale=7,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="sdxl.safetensors",
    )


def _png(size: tuple[int, int] = (512, 512)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def test_upscale_domain_resolves_source_artifact_size_and_load_levels() -> None:
    settings = UpscaleSettings(
        method=UpscaleMethod.IMAGE,
        sizing_mode=UpscaleSizingMode.FACTOR,
        scale_factor=2,
        upscaler_name="models/4x.pth",
    )
    size = resolve_output_size(settings, 513, 512, max_width=2048, max_height=2048)
    assert size.width == 1088
    assert size.height == 1024
    assert estimate_load_level(UpscaleMethod.IMAGE, 512, 512, 1024, 1024) is UpscaleLoadLevel.MEDIUM
    assert estimate_load_level(UpscaleMethod.LATENT, 512, 512, 2048, 2048) is UpscaleLoadLevel.HIGH
    assert estimate_load_level(UpscaleMethod.IMAGE, 100, 100, 100, 200) is UpscaleLoadLevel.LOW
    assert estimate_load_level(UpscaleMethod.IMAGE, 100, 100, 200, 200) is UpscaleLoadLevel.MEDIUM
    assert estimate_load_level(UpscaleMethod.IMAGE, 100, 100, 300, 200) is UpscaleLoadLevel.HIGH
    assert estimate_load_level(UpscaleMethod.LATENT, 100, 100, 100, 150) is UpscaleLoadLevel.LOW
    assert estimate_load_level(UpscaleMethod.LATENT, 100, 100, 100, 300) is UpscaleLoadLevel.MEDIUM
    with pytest.raises(ValueError):
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="C:\\outside.pth",
        )


def test_upscale_snapshot_rejects_unknown_schema_fields() -> None:
    with pytest.raises(UpscaleSnapshotError):
        UpscaleSettingsSnapshot.from_json(
            '{"schema_version": 1, "method": "image", "unexpected": true}'
        )


def test_phase5_migration_model_is_present_and_downgrade_safe() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert "generation_upscale_settings" in inspect(engine).get_table_names()
    engine.dispose()


def test_upscale_enqueue_creates_parent_relation_and_settings_atomically(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        upscaler_dir=tmp_path / "upscalers",
        max_width=2048,
        max_height=2048,
    )
    source_path = tmp_path / "generations" / "2026-08-02" / "generated" / "source.png"
    source_path.parent.mkdir(parents=True)
    source_bytes = _png()
    source_path.write_bytes(source_bytes)
    parent_id, parent_job_id = uuid4(), uuid4()
    parent_snapshot = GenerationSettingsSnapshot.from_settings(_settings())
    GenerationStartRepository(factory).create_pending(
        parent_snapshot,
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
    )
    artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/2026-08-02/generated/source.png",
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        size_bytes=len(source_bytes),
        width=512,
        height=512,
        mime_type="image/png",
        created_at=datetime.now(UTC),
    )
    artifacts = GenerationArtifactRepository(factory)
    artifacts.add(artifact)
    GenerationCompletionRepository(factory).complete_generation(
        parent_id, parent_job_id, artifact, datetime.now(UTC)
    )
    upscale_settings = UpscaleSettings(
        method=UpscaleMethod.IMAGE,
        sizing_mode=UpscaleSizingMode.FACTOR,
        scale_factor=2,
        upscaler_name="4x.pth",
    )
    service = UpscaleEnqueueService(
        GenerationRepository(factory),
        artifacts,
        GenerationDispatchQueueRepository(factory),
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
    )
    item = service.enqueue(parent_id, upscale_settings)
    assert item.generation.kind is GenerationKind.UPSCALE
    assert item.generation.parent_generation_id == parent_id
    assert item.generation.settings_snapshot.width == 1024
    persisted = UpscaleSettingsRepository(factory).get_by_generation(item.generation.id)
    assert persisted is not None
    assert persisted.source_artifact_id == artifact.id
    assert persisted.target_width == 1024
    engine.dispose()


def test_upscale_enqueue_rejects_source_mutation_before_generation_creation(tmp_path: Path) -> None:
    artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=uuid4(),
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/source.png",
        sha256="0" * 64,
        size_bytes=10,
        width=512,
        height=512,
        mime_type="image/png",
        created_at=datetime.now(UTC),
    )
    source = tmp_path / "generations" / "source.png"
    source.parent.mkdir()
    source.write_bytes(_png())
    settings = Settings(_env_file=None, data_dir=tmp_path)
    with pytest.raises(UpscaleEnqueueError) as error:
        from runpod_sdxl_image_studio.services.upscale_enqueue_service import verify_source_artifact

        verify_source_artifact(artifact, settings)
    assert error.value.code == "upscale_source_changed"
    source.unlink()
    with pytest.raises(UpscaleEnqueueError) as missing_error:
        verify_source_artifact(artifact, settings)
    assert missing_error.value.code == "upscale_source_file_missing"


def test_upscale_enqueue_missing_source_creates_no_generation_or_queue_entry(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        upscaler_dir=tmp_path / "upscalers",
        max_width=2048,
        max_height=2048,
    )
    parent_id, parent_job_id = uuid4(), uuid4()
    now = datetime.now(UTC)
    parent_snapshot = GenerationSettingsSnapshot.from_settings(_settings())
    GenerationStartRepository(factory).create_pending(
        parent_snapshot,
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=now,
    )
    missing_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/missing-source.png",
        sha256="a" * 64,
        size_bytes=10,
        width=512,
        height=512,
        mime_type="image/png",
        created_at=now,
    )
    GenerationCompletionRepository(factory).complete_generation(
        parent_id,
        parent_job_id,
        missing_artifact,
        now,
    )
    dispatch = GenerationDispatchQueueRepository(factory)
    service = UpscaleEnqueueService(
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        dispatch,
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
    )

    with pytest.raises(UpscaleEnqueueError) as error:
        service.enqueue(
            parent_id,
            UpscaleSettings(
                method=UpscaleMethod.IMAGE,
                sizing_mode=UpscaleSizingMode.FACTOR,
                scale_factor=2,
                upscaler_name="4x.pth",
            ),
        )

    assert error.value.code == "upscale_source_file_missing"
    with factory() as session:
        assert len(session.scalars(select(GenerationModel)).all()) == 1
    assert dispatch.list_queue() == ()
    engine.dispose()


def test_upscale_retry_preserves_parent_snapshots_and_does_not_resubmit_prompt(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    storage = LocalStorageAdapter(settings)
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    generations = GenerationRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    completion = GenerationCompletionRepository(factory)
    start = GenerationStartRepository(factory)
    dispatch = GenerationDispatchQueueRepository(factory)
    queue = GenerationQueueRepository(factory)
    failure = GenerationFailureRepository(factory)
    upscale_repository = UpscaleSettingsRepository(factory)
    parent_id, parent_job_id = uuid4(), uuid4()
    now = datetime.now(UTC)
    source = storage.store_image(_png(), parent_id, now)
    parent_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path=storage.relative_path(source.path),
        sha256=source.sha256,
        size_bytes=source.size_bytes,
        width=source.width,
        height=source.height,
        mime_type=source.mime_type,
        created_at=now,
    )
    start.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=now,
    )
    artifacts.add(parent_artifact)
    completion.complete_generation(parent_id, parent_job_id, parent_artifact, now)
    enqueue = UpscaleEnqueueService(
        generations,
        artifacts,
        dispatch,
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
    )
    original = enqueue.enqueue(
        parent_id,
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth",
        ),
    )
    queue.mark_queued(original.generation.id, original.job.id, "old-upscale-prompt")
    original_snapshot = original.generation.settings_snapshot
    original_upscale = upscale_repository.get_by_generation(original.generation.id)
    assert original_upscale is not None
    failure.fail_generation(
        original.generation.id,
        original.job.id,
        error_code="upscale_output_invalid",
        error_summary="アップスケール出力画像を検証できません。",
        failed_at=now,
    )

    service = GenerationQueueService(
        dispatch,
        settings,
        upscale_settings_repository=upscale_repository,
    )
    first = service.retry(original.generation.id).item
    second = service.retry(original.generation.id).item

    assert first.generation.id == second.generation.id
    assert first.generation.kind is GenerationKind.UPSCALE
    assert first.generation.parent_generation_id == parent_id
    assert first.generation.retry_of_generation_id == original.generation.id
    assert first.generation.retry_attempt == original.generation.retry_attempt + 1
    assert first.generation.settings_snapshot == original_snapshot
    retry_upscale = upscale_repository.get_by_generation(first.generation.id)
    assert retry_upscale is not None
    assert retry_upscale == original_upscale
    assert first.job.prompt_id is None
    assert dispatch.get_queue_item(original.generation.id).job.prompt_id == "old-upscale-prompt"  # type: ignore[union-attr]
    engine.dispose()


def test_upscale_ui_catalog_visibility_and_safe_handlers() -> None:
    with gr.Blocks():
        missing_catalog = upscale_ui.build_upscale_tab(None)
    with gr.Blocks():
        empty_catalog = upscale_ui.build_upscale_tab(())
    assert "未取得" in missing_catalog.catalog_message.value
    assert "0件" in empty_catalog.catalog_message.value

    visibility = upscale_ui.make_upscale_visibility_handler()
    image_updates = visibility(UpscaleMethod.IMAGE.value)
    latent_updates = visibility(UpscaleMethod.LATENT.value)
    assert image_updates[0]["visible"] is True
    assert image_updates[1]["visible"] is False
    assert latent_updates[0]["visible"] is False
    assert latent_updates[1]["visible"] is True

    generation_id = uuid4()
    service = Mock()
    service.latest_completed_generation_id.return_value = generation_id
    service.select_parent.return_value = Mock(
        generation_id=generation_id,
        preview_path=Path("generations/source.png"),
    )
    latest_result = upscale_ui.make_latest_parent_selection_handler(service)()
    assert latest_result[0] == str(generation_id)
    assert Path(latest_result[1]) == Path("generations/source.png")
    assert len(latest_result) == 3

    service.comparison_for_generation.return_value = Mock(
        result_generation_id=generation_id,
        result_path=Path("generations/upscaled.png"),
        gallery=(("generations/source.png", "parent"), ("generations/upscaled.png", "upscaled")),
    )
    comparison = upscale_ui.make_upscale_result_handler(service)(str(generation_id))
    assert Path(comparison[0]) == Path("generations/upscaled.png")
    assert len(comparison[1]) == 2
    assert len(comparison) == 3

    service.select_parent.side_effect = UpscaleEnqueueError(
        "upscale_parent_not_completed", "secret internal details"
    )
    parent_result = upscale_ui.make_parent_selection_handler(service)(str(generation_id))
    assert len(parent_result) == 3
    assert "secret" not in parent_result[2]
    assert "完了済み" in parent_result[2]

    service.select_parent.side_effect = RuntimeError("secret internal details")
    unexpected_parent_result = upscale_ui.make_parent_selection_handler(service)(str(generation_id))
    assert len(unexpected_parent_result) == 3
    assert "secret" not in unexpected_parent_result[2]
    assert "内部エラー" in unexpected_parent_result[2]

    service.comparison_for_generation.side_effect = UpscaleEnqueueError(
        "upscale_result_not_completed", "secret internal details"
    )
    result = upscale_ui.make_upscale_result_handler(service)(str(generation_id))
    assert len(result) == 3
    assert "secret" not in result[2]
    assert result[0] is None
    assert result[1] == []

    service.comparison_for_generation.side_effect = RuntimeError("secret internal details")
    unexpected_result = upscale_ui.make_upscale_result_handler(service)(str(generation_id))
    assert len(unexpected_result) == 3
    assert "secret" not in unexpected_result[2]
    assert "内部エラー" in unexpected_result[2]


def test_upscale_ui_enqueue_and_plan_handlers_restore_button_on_all_errors() -> None:
    service = Mock()
    inputs = (
        str(uuid4()),
        UpscaleMethod.IMAGE.value,
        UpscaleSizingMode.FACTOR.value,
        2,
        1024,
        1024,
        "4x.pth",
        0.35,
    )

    service.enqueue.side_effect = UpscaleEnqueueError("upscale_parent_not_completed", "secret")
    enqueue_result = upscale_ui.make_upscale_enqueue_handler(service)(*inputs)
    assert len(enqueue_result) == 2
    assert enqueue_result[0].interactive is True
    assert "secret" not in enqueue_result[1]

    service.enqueue.side_effect = RuntimeError("secret")
    unexpected_enqueue_result = upscale_ui.make_upscale_enqueue_details_handler(service)(*inputs)
    assert len(unexpected_enqueue_result) == 2
    assert unexpected_enqueue_result[0].interactive is True
    assert "secret" not in unexpected_enqueue_result[1]
    assert "内部エラー" in unexpected_enqueue_result[1]

    service.plan.side_effect = UpscaleEnqueueError("upscale_parent_not_completed", "secret")
    plan_result = upscale_ui.make_upscale_plan_handler(service)(*inputs)
    assert "secret" not in plan_result
    assert "入力内容" in plan_result

    service.plan.side_effect = RuntimeError("secret")
    unexpected_plan_result = upscale_ui.make_upscale_plan_handler(service)(*inputs)
    assert "secret" not in unexpected_plan_result
    assert "内部エラー" in unexpected_plan_result


@pytest.mark.parametrize(
    ("method", "missing_capability"),
    [
        (UpscaleMethod.IMAGE, "capabilities"),
        (UpscaleMethod.IMAGE, "required_node"),
        (UpscaleMethod.IMAGE, "remote_model"),
        (UpscaleMethod.IMAGE, "local_model"),
        (UpscaleMethod.LATENT, "checkpoint"),
        (UpscaleMethod.LATENT, "sampler"),
        (UpscaleMethod.LATENT, "scheduler"),
        (UpscaleMethod.LATENT, "lora"),
        (UpscaleMethod.LATENT, "lora_node"),
        (UpscaleMethod.LATENT, "vae"),
        (UpscaleMethod.LATENT, "vae_node"),
        (UpscaleMethod.LATENT, "latent_node"),
    ],
)
def test_upscale_preflight_rejects_missing_capabilities_before_submission(
    method: UpscaleMethod,
    missing_capability: str,
) -> None:
    loras = (LoraSetting(name="style.safetensors", order=0),)
    source_settings = _settings().model_copy(
        update={
            "vae_name": "vae.safetensors",
            "loras": loras,
        }
    )
    snapshot = UpscaleSettingsSnapshot.from_settings(
        UpscaleSettings(
            method=method,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth" if method is UpscaleMethod.IMAGE else None,
            denoise=0.35 if method is UpscaleMethod.LATENT else None,
            workflow_template_id=(
                "sdxl_image_upscale" if method is UpscaleMethod.IMAGE else "sdxl_latent_upscale"
            ),
        ),
        source_generation_id=uuid4(),
        source_artifact_id=uuid4(),
        source_sha256="a" * 64,
        source_width=512,
        source_height=512,
        target_width=1024,
        target_height=1024,
    )
    image_nodes = {
        "LoadImage",
        "UpscaleModelLoader",
        "ImageUpscaleWithModel",
        "ImageScale",
        "SaveImage",
    }
    latent_nodes = {
        "LoadImage",
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "VAEEncode",
        "LatentUpscale",
        "KSampler",
        "VAEDecode",
        "SaveImage",
        "LoraLoader",
        "VAELoader",
    }
    all_capabilities = ComfyUICapabilities(
        checkpoints=("sdxl.safetensors",),
        vaes=("vae.safetensors",),
        samplers=("euler",),
        schedulers=("normal",),
        loras=("style.safetensors",),
        upscale_models=("4x.pth",),
        available_node_classes=frozenset(image_nodes | latent_nodes),
        warnings=(),
    )
    capabilities = all_capabilities
    if missing_capability == "capabilities":
        capabilities_result = CapabilityRefreshResult(False, "unavailable", None)
    else:
        if missing_capability == "remote_model":
            capabilities = capabilities.__class__(**capabilities.__dict__ | {"upscale_models": ()})
        elif missing_capability == "checkpoint":
            capabilities = capabilities.__class__(**capabilities.__dict__ | {"checkpoints": ()})
        elif missing_capability == "sampler":
            capabilities = capabilities.__class__(**capabilities.__dict__ | {"samplers": ()})
        elif missing_capability == "scheduler":
            capabilities = capabilities.__class__(**capabilities.__dict__ | {"schedulers": ()})
        elif missing_capability == "lora":
            capabilities = capabilities.__class__(**capabilities.__dict__ | {"loras": ()})
        elif missing_capability == "vae":
            capabilities = capabilities.__class__(**capabilities.__dict__ | {"vaes": ()})
        elif missing_capability == "required_node":
            capabilities = capabilities.__class__(
                **capabilities.__dict__ | {"available_node_classes": frozenset()}
            )
        elif missing_capability in {"lora_node", "vae_node", "latent_node"}:
            nodes = set(capabilities.available_node_classes)
            nodes.remove(
                {
                    "lora_node": "LoraLoader",
                    "vae_node": "VAELoader",
                    "latent_node": "LatentUpscale",
                }[missing_capability]
            )
            capabilities = capabilities.__class__(
                **capabilities.__dict__ | {"available_node_classes": frozenset(nodes)}
            )
        elif missing_capability == "local_model":
            pass
        capabilities_result = CapabilityRefreshResult(True, "ok", capabilities)

    async def capability_provider() -> CapabilityRefreshResult:
        return capabilities_result

    service = GenerationService(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        capability_provider,
        Settings(_env_file=None),
        upscaler_catalog=UpscalerCatalog(())
        if missing_capability == "local_model"
        else UpscalerCatalog(("4x.pth",)),
    )

    with pytest.raises(UpscaleEnqueueError) as error:
        source_snapshot = GenerationSettingsSnapshot.from_settings(source_settings)
        asyncio.run(service._preflight_upscale(snapshot, source_snapshot))  # noqa: SLF001

    expected_codes = {
        "capabilities": "upscale_capabilities_unavailable",
        "required_node": "upscale_required_node_missing",
        "remote_model": "upscale_model_missing",
        "local_model": "upscale_model_missing",
        "checkpoint": "upscale_checkpoint_missing",
        "sampler": "upscale_sampler_missing",
        "scheduler": "upscale_scheduler_missing",
        "lora": "upscale_lora_missing",
        "lora_node": "upscale_lora_missing",
        "vae": "upscale_vae_missing",
        "vae_node": "upscale_required_node_missing",
        "latent_node": "upscale_required_node_missing",
    }
    assert error.value.code == expected_codes[missing_capability]


@pytest.mark.parametrize(
    "code",
    [
        "upscale_output_dimension_mismatch",
        "upscale_output_invalid",
        "upscale_settings_missing",
        "upscale_capabilities_unavailable",
        "upscale_required_node_missing",
        "upscale_checkpoint_missing",
        "upscale_sampler_missing",
        "upscale_scheduler_missing",
        "upscale_lora_missing",
        "upscale_vae_missing",
        "upscale_model_missing",
    ],
)
def test_upscale_failure_summaries_are_stable_and_do_not_expose_details(code: str) -> None:
    error = UpscaleEnqueueError(code, "secret path=/tmp/internal response body")
    first = generation_module._safe_generation_error(error)  # noqa: SLF001
    second = generation_module._safe_generation_error(error)  # noqa: SLF001
    assert first == second
    assert first
    assert "secret" not in first
    assert "/tmp" not in first


def test_upscale_workflows_are_fixed_and_bind_only_typed_values() -> None:
    adapter = UpscaleWorkflowAdapter(
        load_workflow_template("sdxl_image_upscale").as_mapping(),
        load_workflow_template("sdxl_latent_upscale").as_mapping(),
    )
    source_generation_id, artifact_id = uuid4(), uuid4()
    image_snapshot = UpscaleSettingsSnapshot.from_settings(
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth",
            workflow_template_id="sdxl_image_upscale",
        ),
        source_generation_id=source_generation_id,
        source_artifact_id=artifact_id,
        source_sha256="a" * 64,
        source_width=512,
        source_height=512,
        target_width=1024,
        target_height=1024,
    )
    workflow = adapter.build_image_upscale_workflow("uploaded.png", image_snapshot)
    assert workflow["1"]["inputs"]["image"] == "uploaded.png"
    assert workflow["4"]["inputs"]["images"] == ["5", 0]
    assert workflow["5"]["inputs"]["width"] == 1024


@pytest.mark.asyncio
@respx.mock
async def test_comfyui_upload_uses_application_owned_input_name() -> None:
    route = respx.post("http://comfy.test:8188/upload/image").mock(
        return_value=httpx.Response(
            200, json={"name": "staged.png", "subfolder": "", "type": "input"}
        )
    )
    client = ComfyUIClient(base_url="http://comfy.test:8188")
    uploaded = await client.upload_input_image(_png(), UUID(int=1), "a" * 64)
    assert route.called
    assert uploaded.filename == "staged.png"
    request = route.calls[0].request
    assert b"image-studio-" in request.content
    assert b"source.png" not in request.content
    await client.close()


def test_upscaled_storage_uses_separate_directory(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    stored = LocalStorageAdapter(settings).store_image(
        _png(), UUID(int=101), datetime(2026, 8, 2, tzinfo=UTC), kind=GenerationKind.UPSCALE
    )
    assert "/upscaled/" in stored.path.as_posix()


@pytest.mark.parametrize(
    "failure_kind",
    [
        "settings",
        "sidecar",
        "metadata_artifact",
        "thumbnail",
        "thumbnail_artifact",
        "artifact_lookup",
        "artifact_lookup_recovery",
        "metadata_relative",
        "metadata_sha",
    ],
)
def test_optional_upscale_artifact_failures_do_not_reopen_completed_state(
    tmp_path: Path, failure_kind: str
) -> None:
    """Optional artifact failures must occur after durable completion."""

    now = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    settings = Settings(_env_file=None, data_dir=tmp_path)
    storage = LocalStorageAdapter(settings)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)

    generations = GenerationRepository(factory)
    jobs = GenerationJobRepository(factory)
    actual_artifacts = GenerationArtifactRepository(factory)
    completion = GenerationCompletionRepository(factory)
    start = GenerationStartRepository(factory)

    parent_id, parent_job_id = uuid4(), uuid4()
    parent_snapshot = GenerationSettingsSnapshot.from_settings(_settings())
    start.create_pending(
        parent_snapshot,
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=now,
    )
    parent_path = storage.store_image(_png(), parent_id, now)
    parent_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path=storage.relative_path(parent_path.path),
        sha256=parent_path.sha256,
        size_bytes=parent_path.size_bytes,
        width=parent_path.width,
        height=parent_path.height,
        mime_type=parent_path.mime_type,
        created_at=now,
    )
    completion.complete_generation(parent_id, parent_job_id, parent_artifact, now)

    child_id, child_job_id = uuid4(), uuid4()
    child_settings = _settings()
    _, pending_job = start.create_pending(
        GenerationSettingsSnapshot.from_settings(child_settings),
        generation_id=child_id,
        job_id=child_job_id,
        kind=GenerationKind.UPSCALE,
        parent_generation_id=parent_id,
        created_at=now,
    )
    GenerationQueueRepository(factory).mark_queued(child_id, child_job_id, "upscale-prompt")
    output = storage.store_image(_png(), child_id, now, kind=GenerationKind.UPSCALE)
    job = GenerationJob(
        generation_id=child_id,
        id=child_job_id,
        status=pending_job.status,
        prompt_id="upscale-prompt",
        created_at=pending_job.created_at,
        stored_image=output,
    )

    counters = {
        "artifact_lookup": 0,
        "artifact_add": 0,
        "metadata_save": 0,
        "metadata_relative": 0,
        "metadata_sha": 0,
        "metadata_add": 0,
        "thumbnail_save": 0,
        "thumbnail_add": 0,
    }

    class ArtifactRepository:
        def list_by_generation(self, generation_id: UUID) -> tuple[GenerationArtifact, ...]:
            counters["artifact_lookup"] += 1
            if (
                failure_kind in {"artifact_lookup", "artifact_lookup_recovery"}
                and counters["artifact_lookup"] == 1
            ):
                raise RuntimeError("artifact lookup failure")
            return actual_artifacts.list_by_generation(generation_id)

        def add(self, artifact: GenerationArtifact) -> GenerationArtifact:
            counters["artifact_add"] += 1
            if artifact.artifact_type is ArtifactType.METADATA:
                counters["metadata_add"] += 1
                if failure_kind == "metadata_artifact" and counters["metadata_add"] == 1:
                    raise RuntimeError("metadata artifact failure")
            if artifact.artifact_type is ArtifactType.THUMBNAIL:
                counters["thumbnail_add"] += 1
                if failure_kind == "thumbnail_artifact" and counters["thumbnail_add"] == 1:
                    raise RuntimeError("thumbnail artifact failure")
            return actual_artifacts.add(artifact)

    class MetadataStorage:
        def __init__(self) -> None:
            self._storage = GenerationMetadataStorage(tmp_path)

        def save_for_image(self, image_path: Path, payload: dict[str, object]) -> Path:
            counters["metadata_save"] += 1
            if failure_kind == "sidecar":
                raise RuntimeError("sidecar failure")
            return self._storage.save_for_image(image_path, payload)

        def relative_path(self, path: Path) -> str:
            counters["metadata_relative"] += 1
            if failure_kind == "metadata_relative":
                raise RuntimeError("metadata relative path failure")
            return self._storage.relative_path(path)

        def sha256(self, path: Path) -> str:
            counters["metadata_sha"] += 1
            if failure_kind == "metadata_sha":
                raise RuntimeError("metadata sha256 failure")
            return self._storage.sha256(path)

    class ThumbnailStorage:
        def __init__(self) -> None:
            self._storage = HistoryThumbnailStorage(settings)

        def save(self, image_path: Path, generation_id: UUID, created_at: datetime) -> Path:
            counters["thumbnail_save"] += 1
            if failure_kind == "thumbnail":
                raise RuntimeError("thumbnail failure")
            return self._storage.save(image_path, generation_id, created_at)

        def relative_path(self, path: Path) -> str:
            return self._storage.relative_path(path)

        def sha256(self, path: Path) -> str:
            return self._storage.sha256(path)

    class UpscaleRepository:
        calls = 0

        snapshot = UpscaleSettingsSnapshot.from_settings(
            UpscaleSettings(
                method=UpscaleMethod.IMAGE,
                sizing_mode=UpscaleSizingMode.FACTOR,
                scale_factor=2,
                upscaler_name="4x.pth",
                workflow_template_id="sdxl_image_upscale",
            ),
            source_generation_id=parent_id,
            source_artifact_id=parent_artifact.id,
            source_sha256=parent_path.sha256,
            source_width=parent_path.width,
            source_height=parent_path.height,
            target_width=1024,
            target_height=1024,
        )

        def get_by_generation(self, generation_id: UUID) -> object | None:
            del generation_id
            self.calls += 1
            if failure_kind == "settings":
                raise UpscaleSettingsRepositoryError("settings failure")
            return self.snapshot

        def get_by_source_artifact(self, source_artifact_id: UUID) -> tuple[object, ...]:
            del source_artifact_id
            return ()

    class FailureRepository:
        calls = 0

        def fail_generation(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.calls += 1
            raise AssertionError("optional artifact failure must not persist generation failure")

    metadata_storage = None if failure_kind == "thumbnail" else MetadataStorage()
    thumbnail_storage = (
        ThumbnailStorage()
        if failure_kind
        in {
            "metadata_artifact",
            "thumbnail",
            "thumbnail_artifact",
            "artifact_lookup",
            "artifact_lookup_recovery",
        }
        else None
    )

    existing_optional_artifacts: tuple[GenerationArtifact, ...] = ()
    if failure_kind == "artifact_lookup":
        existing_metadata_path = output.path.with_suffix(".json")
        existing_metadata_path.write_text('{"existing": true}', encoding="utf-8")
        existing_thumbnail_path = output.path.with_name(f"{child_id}.webp")
        existing_thumbnail_path.write_bytes(b"existing thumbnail")
        existing_optional_artifacts = (
            GenerationArtifact(
                id=uuid4(),
                generation_id=child_id,
                artifact_type=ArtifactType.METADATA,
                local_path=storage.relative_path(existing_metadata_path),
                sha256=hashlib.sha256(existing_metadata_path.read_bytes()).hexdigest(),
                size_bytes=existing_metadata_path.stat().st_size,
                width=None,
                height=None,
                mime_type="application/json",
                created_at=now,
            ),
            GenerationArtifact(
                id=uuid4(),
                generation_id=child_id,
                artifact_type=ArtifactType.THUMBNAIL,
                local_path=storage.relative_path(existing_thumbnail_path),
                sha256=hashlib.sha256(existing_thumbnail_path.read_bytes()).hexdigest(),
                size_bytes=existing_thumbnail_path.stat().st_size,
                width=None,
                height=None,
                mime_type="image/webp",
                created_at=now,
            ),
        )
        for artifact in existing_optional_artifacts:
            actual_artifacts.add(artifact)

    artifact_repository = ArtifactRepository()
    upscale_repository = UpscaleRepository()
    failure_repository = FailureRepository()
    service = GenerationService(
        object(),
        object(),
        object(),
        storage,
        lambda: None,  # type: ignore[arg-type]
        settings,
        persistence=GenerationPersistenceRepositories(
            generation=generations,
            job=jobs,
            artifact=artifact_repository,  # type: ignore[arg-type]
            start=start,
            queue=GenerationQueueRepository(factory),
            progress=GenerationProgressRepository(factory),
            completion=completion,
            failure=failure_repository,  # type: ignore[arg-type]
        ),
        thumbnail_storage=thumbnail_storage,  # type: ignore[arg-type]
        metadata_storage=metadata_storage,  # type: ignore[arg-type]
        upscale_settings_repository=upscale_repository,  # type: ignore[arg-type]
    )

    try:
        try:
            service._complete_job(  # noqa: SLF001 - verify the durable completion boundary
                job,
                child_settings,
                now,
                GenerationKind.UPSCALE,
                parent_id,
            )
        except Exception as exc:  # noqa: BLE001 - optional failure must not escape the boundary
            pytest.fail(f"optional artifact failure leaked from _complete_job: {exc!r}")

        resolved_generation = generations.get_by_id(child_id)
        resolved_job = jobs.get_by_generation(child_id)
        assert resolved_generation is not None
        assert resolved_job is not None
        assert resolved_generation.status is GenerationStatus.COMPLETED
        assert resolved_job.status is GenerationStatus.COMPLETED
        assert resolved_generation.completed_at is not None
        assert resolved_job.completed_at is not None
        assert resolved_generation.completed_at == resolved_job.completed_at
        assert job.status is GenerationStatus.COMPLETED
        result = service._result_for_job(job, child_settings.seed, now)  # noqa: SLF001
        assert result.status is GenerationStatus.COMPLETED

        persisted_artifacts = actual_artifacts.list_by_generation(child_id)
        primary_artifacts = [
            artifact
            for artifact in persisted_artifacts
            if artifact.artifact_type is ArtifactType.IMAGE
        ]
        assert len(primary_artifacts) == 1
        assert output.path.exists()
        assert resolved_generation.parent_generation_id == parent_id
        assert "upscaled" in primary_artifacts[0].local_path
        assert failure_repository.calls == 0

        optional_types = {
            artifact.artifact_type
            for artifact in persisted_artifacts
            if artifact.artifact_type is not ArtifactType.IMAGE
        }
        expected_optional_types = {
            "metadata_artifact": {ArtifactType.THUMBNAIL},
            "thumbnail_artifact": {ArtifactType.METADATA},
            "artifact_lookup": {ArtifactType.METADATA, ArtifactType.THUMBNAIL},
        }.get(failure_kind, set())
        assert optional_types == expected_optional_types
        if failure_kind == "artifact_lookup":
            assert (
                len([a for a in persisted_artifacts if a.artifact_type is ArtifactType.METADATA])
                == 1
            )
            assert (
                len([a for a in persisted_artifacts if a.artifact_type is ArtifactType.THUMBNAIL])
                == 1
            )
        assert counters["artifact_lookup"] >= 1

        expected_counters = {
            "settings": (upscale_repository.calls >= 1, "upscale settings lookup"),
            "sidecar": (counters["metadata_save"] >= 1, "sidecar save"),
            "metadata_artifact": (counters["metadata_add"] >= 1, "metadata artifact registration"),
            "thumbnail": (counters["thumbnail_save"] >= 1, "thumbnail save"),
            "thumbnail_artifact": (
                counters["thumbnail_add"] >= 1,
                "thumbnail artifact registration",
            ),
            "artifact_lookup": (counters["artifact_lookup"] >= 1, "artifact lookup"),
            "artifact_lookup_recovery": (
                counters["artifact_lookup"] >= 1,
                "artifact lookup recovery",
            ),
            "metadata_relative": (
                counters["metadata_relative"] >= 1,
                "metadata relative path calculation",
            ),
            "metadata_sha": (counters["metadata_sha"] >= 1, "metadata sha256 calculation"),
        }
        failure_exercised, failure_description = expected_counters[failure_kind]
        assert failure_exercised, failure_description

        if failure_kind in {"metadata_artifact", "thumbnail_artifact"}:
            completed_at = resolved_generation.completed_at
            assert completed_at is not None
            first_sidecar_path: Path | None = None
            first_sidecar_payload: dict[str, object] | None = None
            first_sidecar_sha: str | None = None
            first_sidecar_path = output.path.with_suffix(".json")
            first_sidecar_payload = json.loads(first_sidecar_path.read_text(encoding="utf-8"))
            first_sidecar_sha = hashlib.sha256(first_sidecar_path.read_bytes()).hexdigest()
            assert (
                service.repair_optional_artifacts(child_id)
                is OptionalArtifactRepairOutcome.REPAIRED
            )
            assert (
                service.repair_optional_artifacts(child_id)
                is OptionalArtifactRepairOutcome.ALREADY_COMPLETE
            )
            retried_artifacts = actual_artifacts.list_by_generation(child_id)
            assert len([a for a in retried_artifacts if a.artifact_type is ArtifactType.IMAGE]) == 1
            metadata_artifacts = [
                a for a in retried_artifacts if a.artifact_type is ArtifactType.METADATA
            ]
            thumbnail_artifacts = [
                a for a in retried_artifacts if a.artifact_type is ArtifactType.THUMBNAIL
            ]
            assert len(metadata_artifacts) == 1
            assert len(thumbnail_artifacts) == 1
            sidecar_path = output.path.with_suffix(".json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            assert first_sidecar_path == sidecar_path
            assert first_sidecar_payload == sidecar
            assert first_sidecar_sha == hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
            assert datetime.fromisoformat(sidecar["completed_at"]) == completed_at
            assert (
                metadata_artifacts[0].sha256
                == hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
            )
            thumbnail_path = tmp_path / thumbnail_artifacts[0].local_path
            assert (
                thumbnail_artifacts[0].sha256
                == hashlib.sha256(thumbnail_path.read_bytes()).hexdigest()
            )
        elif failure_kind == "artifact_lookup":
            optional_save_counts = (counters["metadata_save"], counters["thumbnail_save"])
            artifact_add_count = counters["artifact_add"]
            assert optional_save_counts == (0, 0)
            assert artifact_add_count == 0
            assert (counters["metadata_save"], counters["thumbnail_save"]) == optional_save_counts
            assert counters["artifact_add"] == artifact_add_count
            recovered_artifacts = actual_artifacts.list_by_generation(child_id)
            assert (
                len([a for a in recovered_artifacts if a.artifact_type is ArtifactType.IMAGE]) == 1
            )
            assert (
                len([a for a in recovered_artifacts if a.artifact_type is ArtifactType.METADATA])
                == 1
            )
            assert (
                len([a for a in recovered_artifacts if a.artifact_type is ArtifactType.THUMBNAIL])
                == 1
            )
        elif failure_kind == "artifact_lookup_recovery":
            assert (
                service.repair_optional_artifacts(child_id)
                is OptionalArtifactRepairOutcome.REPAIRED
            )
            assert (
                service.repair_optional_artifacts(child_id)
                is OptionalArtifactRepairOutcome.ALREADY_COMPLETE
            )
            recovered_artifacts = actual_artifacts.list_by_generation(child_id)
            assert (
                len([a for a in recovered_artifacts if a.artifact_type is ArtifactType.IMAGE]) == 1
            )
            assert (
                len([a for a in recovered_artifacts if a.artifact_type is ArtifactType.METADATA])
                == 1
            )
            assert counters["metadata_save"] == 1
            assert counters["artifact_add"] == 2
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_repairs_completed_optional_artifacts_without_comfyui(
    tmp_path: Path,
) -> None:
    """The existing reconciliation entrypoint repairs completed records only."""

    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    settings = Settings(_env_file=None, data_dir=tmp_path)
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    generations = GenerationRepository(factory)
    jobs = GenerationJobRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    completion = GenerationCompletionRepository(factory)
    start = GenerationStartRepository(factory)
    generation_id, job_id = uuid4(), uuid4()
    generation, pending_job = start.create_pending(
        GenerationSettingsSnapshot.from_settings(_settings()),
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=now,
    )
    GenerationQueueRepository(factory).mark_queued(generation_id, job_id, "prompt-completed")
    stored = LocalStorageAdapter(settings).store_image(_png((8, 4)), generation_id, now)
    assert pending_job.id == job_id

    job = GenerationJob(
        generation_id=generation_id,
        id=job_id,
        status=pending_job.status,
        prompt_id="prompt-completed",
        created_at=pending_job.created_at,
        stored_image=stored,
    )

    class FailingFirstLookupArtifactRepository:
        calls = 0

        def list_by_generation(
            self, requested_generation_id: UUID
        ) -> tuple[GenerationArtifact, ...]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient artifact lookup failure")
            return artifacts.list_by_generation(requested_generation_id)

        def add(self, artifact: GenerationArtifact) -> GenerationArtifact:
            return artifacts.add(artifact)

    initial_artifacts = FailingFirstLookupArtifactRepository()

    def build_service(artifact_repository: object) -> GenerationService:
        return GenerationService(
            object(),
            object(),
            object(),
            LocalStorageAdapter(settings),
            lambda: None,  # type: ignore[arg-type]
            settings,
            persistence=GenerationPersistenceRepositories(
                generation=generations,
                job=jobs,
                artifact=artifact_repository,  # type: ignore[arg-type]
                start=start,
                queue=GenerationQueueRepository(factory),
                progress=GenerationProgressRepository(factory),
                completion=completion,
                failure=GenerationFailureRepository(factory),
            ),
            metadata_storage=GenerationMetadataStorage(tmp_path),
            thumbnail_storage=HistoryThumbnailStorage(settings),
        )

    initial_service = build_service(initial_artifacts)
    initial_service._complete_job(  # noqa: SLF001 - exercise completion before restart
        job,
        _settings(),
        now,
        GenerationKind.STANDARD,
        None,
    )
    assert initial_artifacts.calls == 1

    before_generation = generations.get_by_id(generation_id)
    before_job = jobs.get_by_generation(generation_id)
    assert before_generation is not None
    assert before_job is not None
    assert before_generation.status is GenerationStatus.COMPLETED
    assert before_job.status is GenerationStatus.COMPLETED
    assert before_generation.completed_at is not None
    assert before_job.completed_at == before_generation.completed_at
    assert len(artifacts.list_by_generation(generation_id)) == 1
    candidate_page = generations.list_completed_optional_artifact_repairs(1)
    assert len(candidate_page) == 1
    assert candidate_page[0].generation_id == generation_id
    assert candidate_page[0].completed_at == before_generation.completed_at
    assert (
        generations.list_completed_optional_artifact_repairs(
            1,
            after_completed_at=candidate_page[0].completed_at,
            after_generation_id=candidate_page[0].generation_id,
        )
        == ()
    )

    class Client:
        calls = 0

        async def get_remote_prompt_status(self, prompt_id: str) -> object:
            self.calls += 1
            raise AssertionError(f"completed repair must not query ComfyUI: {prompt_id}")

        async def get_prompt_history(self, prompt_id: str) -> object:
            self.calls += 1
            raise AssertionError(f"completed repair must not read history: {prompt_id}")

    service = build_service(artifacts)
    client = Client()
    recovery = GenerationRecoveryService(
        client,  # type: ignore[arg-type]
        generations,
        jobs,
        artifacts,
        settings,
        completed_optional_artifact_handler=service.repair_optional_artifacts,
    )

    worker = GenerationQueueWorker(
        GenerationDispatchQueueRepository(factory),
        object(),
        settings,
        completed_optional_artifact_handler=recovery.repair_completed_optional_artifacts,
    )
    await worker.reconcile()
    maintenance_task = worker._optional_artifact_maintenance_task  # noqa: SLF001
    assert maintenance_task is not None
    await maintenance_task

    first_messages = await recovery.recover(now)
    assert first_messages == ()
    assert client.calls == 0
    repaired = artifacts.list_by_generation(generation_id)
    assert len([item for item in repaired if item.artifact_type is ArtifactType.IMAGE]) == 1
    assert len([item for item in repaired if item.artifact_type is ArtifactType.METADATA]) == 1
    assert len([item for item in repaired if item.artifact_type is ArtifactType.THUMBNAIL]) == 1
    metadata = next(item for item in repaired if item.artifact_type is ArtifactType.METADATA)
    sidecar = json.loads((tmp_path / metadata.local_path).read_text(encoding="utf-8"))
    assert datetime.fromisoformat(sidecar["completed_at"]) == before_generation.completed_at
    assert stored.path.exists()

    second_messages = await recovery.recover(now)
    assert second_messages == ()
    assert client.calls == 0
    assert len(artifacts.list_by_generation(generation_id)) == 3
    after_generation = generations.get_by_id(generation_id)
    after_job = jobs.get_by_generation(generation_id)
    assert after_generation == before_generation
    assert after_job == before_job
    engine.dispose()


@pytest.mark.asyncio
async def test_optional_artifact_reconciliation_rotates_past_deferred_candidates(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(
        logging.WARNING,
        logger="runpod_sdxl_image_studio.services.generation_recovery_service",
    )
    warning = Mock()
    monkeypatch.setattr(recovery_module, "logger", warning)
    first_id, second_id = uuid4(), uuid4()
    completed_at = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    candidates = (
        OptionalArtifactRepairCandidate(first_id, completed_at),
        OptionalArtifactRepairCandidate(second_id, completed_at),
    )

    class CandidateRepository:
        calls: list[tuple[int, datetime | None, UUID | None]] = []

        def list_completed_optional_artifact_repairs(
            self,
            limit: int = 50,
            *,
            after_completed_at: datetime | None = None,
            after_generation_id: UUID | None = None,
        ) -> tuple[OptionalArtifactRepairCandidate, ...]:
            self.calls.append((limit, after_completed_at, after_generation_id))
            if after_generation_id is None:
                return candidates[:limit]
            if after_generation_id == first_id:
                return candidates[1:2]
            return ()

    class JobRepository:
        def list_recoverable(self, limit: int = 50) -> tuple[object, ...]:
            del limit
            return ()

    repository = CandidateRepository()
    handled: list[UUID] = []
    first_attempts = 0

    def handler(generation_id: UUID) -> OptionalArtifactRepairOutcome:
        nonlocal first_attempts
        handled.append(generation_id)
        if generation_id == first_id:
            first_attempts += 1
            if first_attempts == 1:
                return OptionalArtifactRepairOutcome.DEFERRED
            raise RuntimeError("unexpected maintenance failure")
        return OptionalArtifactRepairOutcome.REPAIRED

    recovery = GenerationRecoveryService(
        object(),  # type: ignore[arg-type]
        repository,  # type: ignore[arg-type]
        JobRepository(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        Settings(
            _env_file=None,
            recovery_max_items=1,
            optional_artifact_repair_batch_size=1,
        ),
        completed_optional_artifact_handler=handler,
    )

    first_messages = await recovery.repair_completed_optional_artifacts()
    second_messages = await recovery.repair_completed_optional_artifacts()
    third_messages = await recovery.repair_completed_optional_artifacts()

    assert first_messages == (f"{first_id}: optional artifacts deferred",)
    assert second_messages == (f"{second_id}: optional artifacts repaired",)
    assert third_messages == ()
    assert handled == [first_id, second_id, first_id]
    assert repository.calls[0] == (1, None, None)
    assert repository.calls[1] == (1, completed_at, first_id)
    assert repository.calls[2] == (1, completed_at, second_id)
    assert repository.calls[3] == (1, None, None)
    assert any(
        call.args[0] == "Completed optional artifact repair failed generation=%s error=%s"
        and call.args[1] == first_id
        and call.kwargs.get("exc_info") is True
        for call in warning.warning.call_args_list
    )


@pytest.mark.asyncio
async def test_optional_artifact_repair_gate_skips_busy_requests_and_recovers() -> None:
    generation_id = uuid4()
    candidate = OptionalArtifactRepairCandidate(
        generation_id,
        datetime(2026, 8, 5, 13, 0, tzinfo=UTC),
    )

    class CandidateRepository:
        def list_completed_optional_artifact_repairs(
            self,
            limit: int = 50,
            *,
            after_completed_at: datetime | None = None,
            after_generation_id: UUID | None = None,
        ) -> tuple[OptionalArtifactRepairCandidate, ...]:
            del limit, after_completed_at, after_generation_id
            return (candidate,)

    started = threading.Event()
    release = threading.Event()
    handled = 0

    def handler(requested_generation_id: UUID) -> OptionalArtifactRepairOutcome:
        nonlocal handled
        assert requested_generation_id == generation_id
        handled += 1
        started.set()
        release.wait(timeout=5)
        return OptionalArtifactRepairOutcome.REPAIRED

    recovery = GenerationRecoveryService(
        object(),  # type: ignore[arg-type]
        CandidateRepository(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        Settings(
            _env_file=None,
            optional_artifact_repair_batch_size=1,
        ),
        completed_optional_artifact_handler=handler,
    )

    first = asyncio.create_task(recovery.repair_completed_optional_artifacts())
    assert await asyncio.to_thread(started.wait, 5)

    busy = asyncio.create_task(recovery.repair_completed_optional_artifacts())
    assert await asyncio.wait_for(busy, timeout=0.2) == ()

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    third_messages: tuple[str, ...] = ()
    for _ in range(50):
        third_messages = await recovery.repair_completed_optional_artifacts()
        if third_messages:
            break
        await asyncio.sleep(0.01)
    assert third_messages == (f"{generation_id}: optional artifacts repaired",)
    assert handled == 2


@pytest.mark.asyncio
async def test_queue_claim_is_not_blocked_by_optional_artifact_maintenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    repository = GenerationDispatchQueueRepository(factory)
    item = repository.enqueue_single(GenerationSettingsSnapshot.from_settings(_settings()))
    events: list[str] = []
    original_claim = repository.claim_next

    def claim_next(*args: object, **kwargs: object) -> object:
        events.append("claim")
        return original_claim(*args, **kwargs)

    monkeypatch.setattr(repository, "claim_next", claim_next)

    async def maintain() -> tuple[str, ...]:
        events.append("maintenance")
        await asyncio.sleep(0.01)
        return ()

    class FakeExecution:
        async def execute_persisted(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("execute")

    worker = GenerationQueueWorker(
        repository,
        FakeExecution(),
        Settings(_env_file=None),
        completed_optional_artifact_handler=maintain,
    )

    await worker.reconcile()
    maintenance_task = worker._optional_artifact_maintenance_task  # noqa: SLF001
    assert maintenance_task is not None
    assert await worker.run_once() is True
    await maintenance_task

    assert events[0] == "claim"
    assert events.count("maintenance") == 1
    assert events.count("execute") == 1
    queued = repository.get_queue_item(item.generation.id)
    assert queued is not None
    assert queued.generation.id == item.generation.id
    engine.dispose()
