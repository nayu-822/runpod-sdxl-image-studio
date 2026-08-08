"""Local Phase 5 integration coverage using Alembic and fake ComfyUI objects."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from PIL import Image
from sqlalchemy import create_engine, inspect, text

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUICapabilities,
    ComfyUIOutputImage,
    PromptHistory,
    QueuedPrompt,
)
from runpod_sdxl_image_studio.adapters.comfyui.upscale_workflow_adapter import (
    UpscaleWorkflowAdapter,
)
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
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
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import (
    GenerationKind,
    GenerationProgress,
    GenerationStatus,
)
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_queue import ReconciliationOutcome
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.upscale_enqueue_service import UpscaleEnqueueService
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template, load_workflow_template

ROOT = Path(__file__).parents[2]


def _png(size: tuple[int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def _alembic_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_phase5_migration_runs_from_empty_and_phase4_database(tmp_path: Path) -> None:
    empty_url = f"sqlite:///{(tmp_path / 'empty.sqlite3').as_posix()}"
    command.upgrade(_alembic_config(empty_url), "head")
    engine = create_engine(empty_url)
    migration_inspector = inspect(engine)
    assert migration_inspector.has_table("generation_upscale_settings")
    columns = {
        column["name"] for column in migration_inspector.get_columns("generation_upscale_settings")
    }
    assert {
        "generation_id",
        "source_artifact_id",
        "method",
        "sizing_mode",
        "scale_factor",
        "target_width",
        "target_height",
        "upscaler_name",
        "denoise",
        "settings_snapshot_json",
        "snapshot_schema_version",
    } <= columns
    assert {
        index["name"] for index in migration_inspector.get_indexes("generation_upscale_settings")
    } >= {
        "ix_generation_upscale_source_artifact",
        "ix_generation_upscale_method",
    }
    foreign_keys = migration_inspector.get_foreign_keys("generation_upscale_settings")
    assert {tuple(key["constrained_columns"]) for key in foreign_keys} == {
        ("generation_id",),
        ("source_artifact_id",),
        ("source_import_id",),
    }
    assert len(migration_inspector.get_check_constraints("generation_upscale_settings")) >= 4
    engine.dispose()

    phase4_path = tmp_path / "phase4.sqlite3"
    phase4_url = f"sqlite:///{phase4_path.as_posix()}"
    command.upgrade(_alembic_config(phase4_url), "0008_phase4_recovery_correction")
    engine = create_engine(phase4_url)
    timestamp = "2026-08-02 00:00:00"
    snapshot = GenerationSettings(
        positive_prompt="a cat",
        negative_prompt="",
        seed=42,
        width=512,
        height=512,
        steps=20,
        cfg_scale=7,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="sdxl.safetensors",
    )
    snapshot_json = GenerationSettingsSnapshot.from_settings(snapshot).to_json()
    generation_id = str(uuid4())
    job_id = str(uuid4())
    artifact_id = str(uuid4())
    with engine.begin() as connection:
        connection.execute(
            text(
                """INSERT INTO generations
                (id, kind, status, parent_generation_id, retry_of_generation_id,
                 retry_attempt, settings_snapshot_json, snapshot_schema_version,
                 checkpoint_name, vae_name, seed, width, height,
                 positive_prompt_search, negative_prompt_search,
                 workflow_template_id, workflow_template_version, comfy_prompt_id,
                 favorite, user_note, error_code, error_summary, created_at,
                 started_at, completed_at, updated_at)
                VALUES (:id, 'standard', 'completed', NULL, NULL, 0, :snapshot, 1,
                 'sdxl.safetensors', NULL, 42, 512, 512, 'a cat', '',
                 'sdxl_txt2img', '1.0', NULL, 0, NULL, NULL, NULL,
                 :timestamp, NULL, :timestamp, :timestamp)"""
            ),
            {"id": generation_id, "snapshot": snapshot_json, "timestamp": timestamp},
        )
        connection.execute(
            text(
                """INSERT INTO generation_jobs
                (id, generation_id, status, comfy_prompt_id, progress_value,
                 progress_maximum, current_node, error_code, error_summary,
                 created_at, started_at, completed_at, updated_at,
                 worker_id, claimed_at, lease_expires_at, cancel_requested_at, cancelled_at)
                VALUES (:id, :generation_id, 'completed', NULL, NULL, NULL, NULL, NULL, NULL,
                 :timestamp, NULL, :timestamp, :timestamp, NULL, NULL, NULL, NULL, NULL)"""
            ),
            {"id": job_id, "generation_id": generation_id, "timestamp": timestamp},
        )
        connection.execute(
            text(
                """INSERT INTO generation_artifacts
                (id, generation_id, artifact_type, local_path, sha256, size_bytes,
                 width, height, mime_type, created_at)
                VALUES (:id, :generation_id, 'image', 'generations/source.png', :sha256,
                 4, 2, 2, 'image/png', :timestamp)"""
            ),
            {
                "id": artifact_id,
                "generation_id": generation_id,
                "sha256": "a" * 64,
                "timestamp": timestamp,
            },
        )
    command.upgrade(_alembic_config(phase4_url), "head")
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT id FROM generations WHERE id=:id"), {"id": generation_id}
            ).scalar_one()
            == generation_id
        )
        assert (
            connection.execute(
                text("SELECT id FROM generation_jobs WHERE id=:id"), {"id": job_id}
            ).scalar_one()
            == job_id
        )
    command.downgrade(_alembic_config(phase4_url), "0012_phase6_legacy_metadata_candidates")
    assert not inspect(engine).has_table("drive_sync_records")
    assert not inspect(engine).has_table("drive_sync_jobs")
    command.downgrade(_alembic_config(phase4_url), "-1")
    assert inspect(engine).has_table("metadata_imports")
    command.downgrade(_alembic_config(phase4_url), "-1")
    assert inspect(engine).has_table("metadata_imports")
    command.downgrade(_alembic_config(phase4_url), "-1")
    assert not inspect(engine).has_table("metadata_imports")
    assert inspect(engine).has_table("generation_upscale_settings")
    command.downgrade(_alembic_config(phase4_url), "-1")
    assert not inspect(engine).has_table("generation_upscale_settings")
    command.upgrade(_alembic_config(phase4_url), "head")
    assert inspect(engine).has_table("generation_upscale_settings")
    engine.dispose()


@pytest.mark.parametrize(
    ("method", "output_bytes", "expected_failure_code"),
    [
        (UpscaleMethod.IMAGE, _png((1024, 1024)), None),
        (UpscaleMethod.LATENT, _png((1024, 1024)), None),
        (UpscaleMethod.IMAGE, _png((768, 768)), "upscale_output_dimension_mismatch"),
        (UpscaleMethod.IMAGE, b"invalid image bytes", "upscale_output_invalid"),
    ],
)
def test_fake_comfyui_upscale_executes_and_completes(
    tmp_path: Path,
    method: UpscaleMethod,
    output_bytes: bytes,
    expected_failure_code: str | None,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'execution.sqlite3').as_posix()}"
    settings = Settings(
        _env_file=None,
        database_url=database_url,
        data_dir=tmp_path,
        upscaler_dir=tmp_path / "upscalers",
        history_poll_interval_seconds=0.001,
        generation_timeout_seconds=5,
    )
    settings.upscaler_dir.mkdir()
    (settings.upscaler_dir / "4x.pth").write_bytes(b"fake model")
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_engine(database_url)
    factory = create_session_factory(engine)
    generations = GenerationRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    jobs = GenerationJobRepository(factory)
    queue = GenerationQueueRepository(factory)
    dispatch = GenerationDispatchQueueRepository(factory)
    start = GenerationStartRepository(factory)
    completion = GenerationCompletionRepository(factory)
    failure = GenerationFailureRepository(factory)
    progress = GenerationProgressRepository(factory)
    parent_id, parent_job_id = uuid4(), uuid4()
    source_bytes = _png((512, 512))
    source_path = tmp_path / "generations" / "source.png"
    source_path.parent.mkdir()
    source_path.write_bytes(source_bytes)
    source_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/source.png",
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        size_bytes=len(source_bytes),
        width=512,
        height=512,
        mime_type="image/png",
        created_at=datetime.now(UTC),
    )
    parent_settings = GenerationSettings(
        positive_prompt="a cat",
        negative_prompt="bad anatomy",
        seed=42,
        width=512,
        height=512,
        steps=20,
        cfg_scale=7,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="sdxl.safetensors",
        vae_name="vae.safetensors" if method is UpscaleMethod.LATENT else None,
        loras=(
            LoraSetting(
                name="style.safetensors",
                model_strength=0.7,
                clip_strength=0.8,
                order=0,
            ),
        )
        if method is UpscaleMethod.LATENT
        else (),
    )
    start.create_pending(
        GenerationSettingsSnapshot.from_settings(parent_settings),
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
    )
    artifacts.add(source_artifact)
    completion.complete_generation(parent_id, parent_job_id, source_artifact)
    enqueue = UpscaleEnqueueService(
        generations,
        artifacts,
        dispatch,
        settings,
        catalog=UpscalerCatalog.scan(settings.upscaler_dir),
    )
    item = enqueue.enqueue(
        parent_id,
        UpscaleSettings(
            method=method,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth" if method is UpscaleMethod.IMAGE else None,
            denoise=0.35 if method is UpscaleMethod.LATENT else None,
        ),
    )

    class FakeClient:
        uploads = 0
        prompts = 0

        async def upload_input_image(
            self, image_bytes: bytes, generation_id: UUID, source_sha256: str
        ):
            assert image_bytes == source_bytes
            assert source_sha256 == source_artifact.sha256
            self.uploads += 1
            return ComfyUIOutputImage("uploaded.png", "", "input")

        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            assert workflow["1"]["inputs"]["image"] == "uploaded.png"  # type: ignore[index]
            if method is UpscaleMethod.LATENT:
                assert workflow["2"]["inputs"]["ckpt_name"] == "sdxl.safetensors"  # type: ignore[index]
                assert workflow["3"]["inputs"]["text"] == "a cat"  # type: ignore[index]
                assert workflow["4"]["inputs"]["text"] == "bad anatomy"  # type: ignore[index]
                assert workflow["7"]["inputs"]["seed"] == 42  # type: ignore[index]
                assert workflow["7"]["inputs"]["denoise"] == 0.35  # type: ignore[index]
                assert "lora_000" in workflow  # type: ignore[operator]
                assert "vae_external" in workflow  # type: ignore[operator]
            else:
                assert workflow["2"]["inputs"]["model_name"] == "4x.pth"  # type: ignore[index]
            self.prompts += 1
            return QueuedPrompt("upscale-prompt", 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                True,
                False,
                (ComfyUIOutputImage("upscaled.png", "", "output"),),
                None,
            )

        async def get_output_image(self, output: ComfyUIOutputImage) -> bytes:
            return output_bytes

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            del prompt_id, client_id
            if False:
                yield GenerationProgress(state=GenerationStatus.COMPLETED)

    capabilities = (
        ComfyUICapabilities(
            checkpoints=(),
            vaes=(),
            samplers=(),
            schedulers=(),
            loras=(),
            upscale_models=("4x.pth",),
            available_node_classes=frozenset(
                {
                    "LoadImage",
                    "UpscaleModelLoader",
                    "ImageUpscaleWithModel",
                    "ImageScale",
                    "SaveImage",
                }
            ),
            warnings=(),
        )
        if method is UpscaleMethod.IMAGE
        else ComfyUICapabilities(
            checkpoints=("sdxl.safetensors",),
            vaes=("vae.safetensors",),
            samplers=("euler",),
            schedulers=("normal",),
            loras=("style.safetensors",),
            upscale_models=(),
            available_node_classes=frozenset(
                {
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
            ),
            warnings=(),
        )
    )

    async def capability_provider() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    class CountingFailureRepository:
        calls = 0

        def fail_generation(self, *args: object, **kwargs: object) -> None:
            self.calls += 1
            failure.fail_generation(*args, **kwargs)  # type: ignore[arg-type]

    client = FakeClient()
    counting_failure = CountingFailureRepository()
    service = GenerationService(
        client,  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        FakeWebSocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capability_provider,
        settings,
        persistence=GenerationPersistenceRepositories(
            generation=generations,
            job=jobs,
            artifact=artifacts,
            start=start,
            queue=queue,
            progress=progress,
            completion=completion,
            failure=counting_failure,  # type: ignore[arg-type]
        ),
        upscale_settings_repository=UpscaleSettingsRepository(factory),
        upscale_workflow_adapter=UpscaleWorkflowAdapter(
            load_workflow_template("sdxl_image_upscale").as_mapping(),
            load_workflow_template("sdxl_latent_upscale").as_mapping(),
        ),
        upscaler_catalog=UpscalerCatalog.scan(settings.upscaler_dir),
    )
    result = asyncio.run(service.execute_persisted(item.generation.id, item.job.id))

    assert client.uploads == 1
    assert client.prompts == 1
    persisted_generation = generations.get_by_id(item.generation.id)
    persisted_job = jobs.get_by_generation(item.generation.id)
    assert persisted_generation is not None
    assert persisted_job is not None
    assert persisted_generation.comfy_prompt_id == "upscale-prompt"
    assert persisted_job.prompt_id == "upscale-prompt"
    if expected_failure_code is not None:
        assert result.status is GenerationStatus.FAILED
        assert result.stored_image is None
        assert persisted_generation.status is GenerationStatus.FAILED
        assert persisted_job.status is GenerationStatus.FAILED
        assert persisted_generation.error_code == expected_failure_code
        assert persisted_job.error_code == expected_failure_code
        assert persisted_generation.error_summary == persisted_job.error_summary
        assert persisted_generation.error_summary is not None
        assert "768" not in persisted_generation.error_summary
        assert counting_failure.calls == 1
        assert artifacts.get_primary_image(item.generation.id) is None
        engine.dispose()
        return

    assert result.status is GenerationStatus.COMPLETED
    assert result.stored_image is not None
    assert result.stored_image.path.parent.name == "upscaled"
    assert persisted_generation.status is GenerationStatus.COMPLETED
    assert persisted_job.status is GenerationStatus.COMPLETED
    assert counting_failure.calls == 0
    persisted_upscale = UpscaleSettingsRepository(factory).get_by_generation(item.generation.id)
    assert persisted_upscale is not None
    assert persisted_upscale.target_width == 1024
    assert persisted_upscale.target_height == 1024
    assert persisted_upscale.denoise == (0.35 if method is UpscaleMethod.LATENT else None)
    assert persisted_generation.settings_snapshot.positive_prompt == "a cat"
    assert persisted_generation.settings_snapshot.negative_prompt == "bad anatomy"
    assert persisted_generation.settings_snapshot.seed == 42
    assert persisted_generation.settings_snapshot.checkpoint_name == "sdxl.safetensors"
    assert persisted_generation.settings_snapshot.sampler_name == "euler"
    assert persisted_generation.settings_snapshot.scheduler_name == "normal"
    assert persisted_generation.settings_snapshot.steps == 20
    assert persisted_generation.settings_snapshot.cfg_scale == 7
    assert persisted_generation.parent_generation_id == parent_id
    assert artifacts.get_primary_image(item.generation.id) is not None
    engine.dispose()


def _build_reconciliation_harness(
    tmp_path: Path,
    output_bytes: bytes,
    settings_mode: str = "present",
) -> tuple[GenerationService, UUID, object, object, object, object, object]:
    database_url = f"sqlite:///{(tmp_path / 'reconcile.sqlite3').as_posix()}"
    settings = Settings(
        _env_file=None,
        database_url=database_url,
        data_dir=tmp_path,
        upscaler_dir=tmp_path / "upscalers",
    )
    settings.upscaler_dir.mkdir()
    (settings.upscaler_dir / "4x.pth").write_bytes(b"fake model")
    command.upgrade(_alembic_config(database_url), "head")
    engine = create_engine(database_url)
    factory = create_session_factory(engine)
    generations = GenerationRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    jobs = GenerationJobRepository(factory)
    queue = GenerationQueueRepository(factory)
    dispatch = GenerationDispatchQueueRepository(factory)
    start = GenerationStartRepository(factory)
    completion = GenerationCompletionRepository(factory)
    failure = GenerationFailureRepository(factory)
    progress = GenerationProgressRepository(factory)
    parent_id, parent_job_id = uuid4(), uuid4()
    source_bytes = _png((512, 512))
    source_path = tmp_path / "generations" / "source.png"
    source_path.parent.mkdir()
    source_path.write_bytes(source_bytes)
    source_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path="generations/source.png",
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        size_bytes=len(source_bytes),
        width=512,
        height=512,
        mime_type="image/png",
        created_at=datetime.now(UTC),
    )
    parent_settings = GenerationSettings(
        positive_prompt="a cat",
        negative_prompt="",
        seed=42,
        width=512,
        height=512,
        steps=20,
        cfg_scale=7,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="sdxl.safetensors",
    )
    start.create_pending(
        GenerationSettingsSnapshot.from_settings(parent_settings),
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
    )
    artifacts.add(source_artifact)
    completion.complete_generation(parent_id, parent_job_id, source_artifact)
    item = UpscaleEnqueueService(
        generations,
        artifacts,
        dispatch,
        settings,
        catalog=UpscalerCatalog.scan(settings.upscaler_dir),
    ).enqueue(
        parent_id,
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth",
        ),
    )
    queue.mark_queued(item.generation.id, item.job.id, "reconcile-prompt")

    class FakeClient:
        history_calls = 0
        output_calls = 0
        prompt_calls = 0

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            self.history_calls += 1
            return PromptHistory(
                prompt_id,
                True,
                False,
                (ComfyUIOutputImage("upscaled.png", "", "output"),),
                None,
            )

        async def get_output_image(self, output: ComfyUIOutputImage) -> bytes:
            del output
            self.output_calls += 1
            return output_bytes

        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del workflow, client_id
            self.prompt_calls += 1
            raise AssertionError("reconciliation must not submit a new prompt")

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            del prompt_id, client_id
            if False:
                yield GenerationProgress(state=GenerationStatus.COMPLETED)

    capabilities = ComfyUICapabilities(
        checkpoints=(),
        vaes=(),
        samplers=(),
        schedulers=(),
        loras=(),
        upscale_models=("4x.pth",),
        available_node_classes=frozenset(
            {"LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "SaveImage"}
        ),
        warnings=(),
    )

    async def capability_provider() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    upscale_repository: object = UpscaleSettingsRepository(factory)
    if settings_mode == "missing":

        class MissingSettingsRepository:
            def get_by_generation(self, generation_id: UUID) -> None:
                del generation_id
                return None

            def get_by_source_artifact(self, source_artifact_id: UUID) -> tuple[object, ...]:
                del source_artifact_id
                return ()

        upscale_repository = MissingSettingsRepository()
    elif settings_mode == "error":

        class ErrorSettingsRepository:
            def get_by_generation(self, generation_id: UUID) -> None:
                del generation_id
                raise UpscaleSettingsRepositoryError("temporary settings failure")

            def get_by_source_artifact(self, source_artifact_id: UUID) -> tuple[object, ...]:
                del source_artifact_id
                return ()

        upscale_repository = ErrorSettingsRepository()

    class CountingFailureRepository:
        calls = 0

        def fail_generation(self, *args: object, **kwargs: object) -> None:
            self.calls += 1
            failure.fail_generation(*args, **kwargs)  # type: ignore[arg-type]

    counting_failure = CountingFailureRepository()
    client = FakeClient()
    service = GenerationService(
        client,  # type: ignore[arg-type]
        WorkflowAdapter(load_txt2img_template().as_mapping()),
        FakeWebSocket(),  # type: ignore[arg-type]
        LocalStorageAdapter(settings),
        capability_provider,
        settings,
        persistence=GenerationPersistenceRepositories(
            generation=generations,
            job=jobs,
            artifact=artifacts,
            start=start,
            queue=queue,
            progress=progress,
            completion=completion,
            failure=counting_failure,  # type: ignore[arg-type]
        ),
        upscale_settings_repository=upscale_repository,  # type: ignore[arg-type]
        upscale_workflow_adapter=UpscaleWorkflowAdapter(
            load_workflow_template("sdxl_image_upscale").as_mapping(),
            load_workflow_template("sdxl_latent_upscale").as_mapping(),
        ),
        upscaler_catalog=UpscalerCatalog.scan(settings.upscaler_dir),
    )
    return service, item.generation.id, generations, jobs, artifacts, counting_failure, client


@pytest.mark.parametrize(
    ("output_bytes", "expected_code"),
    [
        (_png((768, 768)), "upscale_output_dimension_mismatch"),
        (b"invalid image bytes", "upscale_output_invalid"),
    ],
)
def test_fake_comfyui_reconciliation_confirms_deterministic_upscale_failures(
    tmp_path: Path,
    output_bytes: bytes,
    expected_code: str,
) -> None:
    service, generation_id, generations, jobs, artifacts, failure, client = (
        _build_reconciliation_harness(tmp_path, output_bytes)
    )

    outcome = asyncio.run(service.reconcile_prompt(generation_id, "reconcile-prompt"))

    assert outcome is ReconciliationOutcome.FAILED
    generation = generations.get_by_id(generation_id)
    job = jobs.get_by_generation(generation_id)
    assert generation is not None
    assert job is not None
    assert generation.status is GenerationStatus.FAILED
    assert job.status is GenerationStatus.FAILED
    assert generation.error_code == expected_code
    assert job.error_code == expected_code
    assert generation.error_summary == job.error_summary
    assert generation.completed_at is not None
    assert job.completed_at == generation.completed_at
    assert generation.comfy_prompt_id == "reconcile-prompt"
    assert job.prompt_id == "reconcile-prompt"
    assert failure.calls == 1
    assert artifacts.get_primary_image(generation_id) is None
    assert client.prompt_calls == 0

    second = asyncio.run(service.reconcile_prompt(generation_id, "reconcile-prompt"))
    assert second is ReconciliationOutcome.FAILED
    assert client.history_calls == 1
    assert client.output_calls == 1


@pytest.mark.parametrize(
    ("settings_mode", "expected_outcome", "expected_status", "expected_code", "calls"),
    [
        (
            "missing",
            ReconciliationOutcome.FAILED,
            GenerationStatus.FAILED,
            "upscale_settings_missing",
            1,
        ),
        (
            "error",
            ReconciliationOutcome.UNAVAILABLE,
            GenerationStatus.QUEUED,
            None,
            0,
        ),
    ],
)
def test_fake_comfyui_reconciliation_distinguishes_missing_and_unavailable_settings(
    tmp_path: Path,
    settings_mode: str,
    expected_outcome: ReconciliationOutcome,
    expected_status: GenerationStatus,
    expected_code: str | None,
    calls: int,
) -> None:
    service, generation_id, generations, jobs, _artifacts, failure, _client = (
        _build_reconciliation_harness(tmp_path, _png((1024, 1024)), settings_mode)
    )

    outcome = asyncio.run(service.reconcile_prompt(generation_id, "reconcile-prompt"))

    assert outcome is expected_outcome
    generation = generations.get_by_id(generation_id)
    job = jobs.get_by_generation(generation_id)
    assert generation is not None
    assert job is not None
    assert generation.status is expected_status
    assert job.status is expected_status
    assert generation.error_code == expected_code
    assert job.error_code == expected_code
    assert failure.calls == calls
