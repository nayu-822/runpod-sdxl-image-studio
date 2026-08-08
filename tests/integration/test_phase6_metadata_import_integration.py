"""Phase 6 migration and SQLite persistence integration checks."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from PIL import Image, PngImagePlugin
from sqlalchemy import inspect, text

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
from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
)
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
from runpod_sdxl_image_studio.adapters.database.repositories.metadata_import_repository import (
    MetadataImportRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepository,
)
from runpod_sdxl_image_studio.adapters.metadata.comfyui_prompt_metadata_adapter import (
    parse_comfyui_prompt_metadata,
)
from runpod_sdxl_image_studio.adapters.metadata.sidecar_metadata_adapter import (
    parse_sidecar_metadata,
)
from runpod_sdxl_image_studio.adapters.storage.imported_image_storage import ImportedImageStorage
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
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import (
    UpscaleSettingsSnapshot,
    UpscaleSourceKind,
)
from runpod_sdxl_image_studio.jobs.generation_queue_worker import GenerationQueueWorker
from runpod_sdxl_image_studio.services.generation_execution_service import (
    GenerationExecutionService,
)
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
)
from runpod_sdxl_image_studio.services.generation_queue_service import GenerationQueueService
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.metadata_import_service import MetadataImportService
from runpod_sdxl_image_studio.services.upscale_enqueue_service import UpscaleEnqueueService
from runpod_sdxl_image_studio.services.upscale_service import UpscaleService
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template, load_workflow_template


def _alembic_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("script_location", str(Path("alembic").resolve()))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.resolve().as_posix()}")
    return config


def _png_bytes(size: tuple[int, int] = (512, 512), color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _prompt_graph() -> dict[str, object]:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a cat", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blur", "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7,
                "sampler_name": "euler",
                "scheduler": "normal",
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
    }


def test_phase6_migration_roundtrip_and_safe_external_downgrade(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.sqlite3"
    config = _alembic_config(database_path)

    command.upgrade(config, "head")
    settings = Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path,
        database_url=f"sqlite:///{database_path.as_posix()}",
    )
    engine = create_image_studio_engine(settings)
    assert "metadata_imports" in inspect(engine).get_table_names()
    assert "source_import_id" in {
        column["name"] for column in inspect(engine).get_columns("generation_upscale_settings")
    }

    now = datetime.now(UTC)
    snapshot = GenerationSettingsSnapshot.from_settings(
        GenerationSettings(
            positive_prompt="a cat",
            negative_prompt="",
            seed=1,
            width=512,
            height=512,
            steps=20,
            cfg_scale=7,
            sampler_name="euler",
            scheduler_name="normal",
            checkpoint_name="model.safetensors",
        )
    )
    generation_id = uuid4()
    job_id = uuid4()
    import_id = uuid4()
    start = GenerationStartRepository(create_session_factory(engine))
    start.create_pending(
        snapshot,
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.UPSCALE,
        parent_generation_id=None,
        created_at=now,
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO metadata_imports "
                "(id, original_filename, stored_image_path, source_image_sha256, "
                "stored_image_sha256, image_width, image_height, image_mime_type, "
                "metadata_source, metadata_status, raw_metadata_json, raw_metadata_sha256, "
                "candidate_json, candidate_options_json, selected_metadata_source, "
                "sidecar_hash_confirmed, normalized_snapshot_json, "
                "normalized_snapshot_schema_version, manual_mapping_json, warnings_json, "
                "created_at, updated_at) VALUES (:id, 'import.png', 'imports/x.png', "
                ":hash, :hash, 512, 512, 'image/png', 'none', 'metadata_missing', "
                "'{}', :hash, NULL, '[]', NULL, 0, "
                "NULL, NULL, '[]', '[]', :created_at, :updated_at)"
            ),
            {
                "id": str(import_id),
                "hash": "0" * 64,
                "created_at": now,
                "updated_at": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO generation_upscale_settings "
                "(generation_id, source_kind, source_artifact_id, source_import_id, method, "
                "sizing_mode, scale_factor, target_width, target_height, upscaler_name, denoise, "
                "settings_snapshot_json, snapshot_schema_version, created_at, updated_at) "
                "VALUES (:generation_id, 'metadata_import', NULL, :source_import_id, 'image', "
                "'factor', 2, 1024, 1024, '4x.pth', NULL, :snapshot, 2, :created_at, :updated_at)"
            ),
            {
                "generation_id": str(generation_id),
                "source_import_id": str(import_id),
                "snapshot": snapshot.to_json(),
                "created_at": now,
                "updated_at": now,
            },
        )

    # 0011 can be removed while retaining the Phase 6 source rows.  The
    # destructive 0010 -> 0009 downgrade must reject external rows.
    command.downgrade(config, "0010_phase6_metadata_imports")
    with pytest.raises(RuntimeError, match="metadata_import upscale sources"):
        command.downgrade(config, "0009_phase5_upscale_settings")

    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM generation_upscale_settings WHERE generation_id = :generation_id"),
            {"generation_id": str(generation_id)},
        )
    command.downgrade(config, "0009_phase5_upscale_settings")
    assert "metadata_imports" not in inspect(engine).get_table_names()
    command.upgrade(config, "head")
    assert "metadata_imports" in inspect(engine).get_table_names()


def test_phase6_migration_backfills_legacy_artifact_sources_without_data_loss(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    config = _alembic_config(database_path)
    command.upgrade(config, "0009_phase5_upscale_settings")
    settings = Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path,
        database_url=f"sqlite:///{database_path.as_posix()}",
    )
    engine = create_image_studio_engine(settings)
    factory = create_session_factory(engine)
    now = datetime.now(UTC)
    generation_id = uuid4()
    job_id = uuid4()
    artifact_id = uuid4()
    generation_snapshot = GenerationSettingsSnapshot.from_settings(
        GenerationSettings(
            positive_prompt="legacy",
            negative_prompt="",
            seed=2,
            width=512,
            height=512,
            steps=20,
            cfg_scale=7,
            sampler_name="euler",
            scheduler_name="normal",
            checkpoint_name="model.safetensors",
        )
    )
    GenerationStartRepository(factory).create_pending(
        generation_snapshot,
        generation_id=generation_id,
        job_id=job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=now,
    )
    GenerationArtifactRepository(factory).add(
        GenerationArtifact(
            id=artifact_id,
            generation_id=generation_id,
            artifact_type=ArtifactType.IMAGE,
            local_path="generations/legacy.png",
            sha256="a" * 64,
            size_bytes=4,
            width=512,
            height=512,
            mime_type="image/png",
            created_at=now,
        )
    )
    upscale_snapshot = UpscaleSettingsSnapshot.from_settings(
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth",
            workflow_template_id="sdxl_image_upscale",
        ),
        source_generation_id=generation_id,
        source_artifact_id=artifact_id,
        source_sha256="a" * 64,
        source_width=512,
        source_height=512,
        target_width=1024,
        target_height=1024,
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO generation_upscale_settings "
                "(generation_id, source_artifact_id, method, sizing_mode, scale_factor, "
                "target_width, target_height, upscaler_name, denoise, settings_snapshot_json, "
                "snapshot_schema_version, created_at, updated_at) VALUES "
                "(:generation_id, :source_artifact_id, 'image', 'factor', 2, 1024, 1024, "
                "'4x.pth', NULL, :snapshot, 2, :created_at, :updated_at)"
            ),
            {
                "generation_id": str(generation_id),
                "source_artifact_id": str(artifact_id),
                "snapshot": upscale_snapshot.to_json(),
                "created_at": now,
                "updated_at": now,
            },
        )

    command.upgrade(config, "head")
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT source_kind, source_artifact_id, source_import_id "
                "FROM generation_upscale_settings WHERE generation_id=:generation_id"
            ),
            {"generation_id": str(generation_id)},
        ).one()
        assert row.source_kind == "generation_artifact"
        assert row.source_artifact_id == str(artifact_id)
        assert row.source_import_id is None
        assert connection.execute(
            text("SELECT id FROM generations WHERE id=:id"), {"id": str(generation_id)}
        ).scalar_one() == str(generation_id)
        assert connection.execute(
            text("SELECT id FROM generation_jobs WHERE id=:id"), {"id": str(job_id)}
        ).scalar_one() == str(job_id)


def test_phase6_migration_repairs_legacy_ambiguous_candidates_without_auto_selecting(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-ambiguous.sqlite3"
    config = _alembic_config(database_path)
    command.upgrade(config, "0010_phase6_metadata_imports")
    engine = create_image_studio_engine(
        Settings(
            _env_file=None,
            environment="test",
            data_dir=tmp_path,
            database_url=f"sqlite:///{database_path.as_posix()}",
        )
    )
    now = datetime.now(UTC)
    import_id = uuid4()
    source_hash = "a" * 64
    png_candidate = parse_comfyui_prompt_metadata(_prompt_graph()).candidate
    sidecar_payload = {
        "schema_version": 1,
        "settings": {
            "positive_prompt": "sidecar candidate",
            "negative_prompt": "blur",
            "seed": 42,
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7.0,
            "sampler_name": "euler",
            "scheduler_name": "normal",
            "checkpoint_name": "model.safetensors",
            "vae_name": None,
            "loras": [],
        },
    }
    sidecar_result = parse_sidecar_metadata(json.dumps(sidecar_payload))
    raw_metadata = json.dumps(
        {
            "schema_version": 1,
            "sources": [
                {
                    "kind": "comfyui_prompt",
                    "raw_text": json.dumps(_prompt_graph()),
                    "sha256": "1" * 64,
                },
                sidecar_result.raw_source.model_dump(mode="json"),
            ],
        },
        ensure_ascii=False,
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO metadata_imports "
                "(id, original_filename, stored_image_path, source_image_sha256, "
                "stored_image_sha256, image_width, image_height, image_mime_type, "
                "metadata_source, metadata_status, raw_metadata_json, raw_metadata_sha256, "
                "candidate_json, normalized_snapshot_json, normalized_snapshot_schema_version, "
                "manual_mapping_json, warnings_json, created_at, updated_at) "
                "VALUES (:id, 'legacy.png', 'imports/legacy.png', :source_hash, :source_hash, "
                "512, 512, 'image/png', 'none', 'needs_mapping', :raw_metadata, :raw_hash, "
                ":candidate_json, NULL, NULL, '[]', '[\"metadata_import_ambiguous\"]', "
                ":created_at, :updated_at)"
            ),
            {
                "id": str(import_id),
                "source_hash": source_hash,
                "raw_metadata": raw_metadata,
                "raw_hash": "b" * 64,
                "candidate_json": png_candidate.model_dump_json(),
                "created_at": now,
                "updated_at": now,
            },
        )

    original_raw_metadata = raw_metadata
    original_warnings = '["metadata_import_ambiguous"]'

    command.upgrade(config, "head")
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT candidate_json, candidate_options_json, selected_metadata_source, "
                "metadata_status FROM metadata_imports WHERE id=:id"
            ),
            {"id": str(import_id)},
        ).one()
    options = json.loads(row.candidate_options_json)
    assert row.candidate_json is None
    assert row.selected_metadata_source is None
    assert row.metadata_status == "needs_mapping"
    assert {option["source_kind"] for option in options} == {
        "comfyui_prompt",
        "app_sidecar",
    }

    with pytest.raises(RuntimeError, match="without losing candidate data"):
        command.downgrade(config, "0010_phase6_metadata_imports")
    with engine.connect() as connection:
        unchanged = connection.execute(
            text(
                "SELECT candidate_json, candidate_options_json, selected_metadata_source, "
                "raw_metadata_json, warnings_json, metadata_status "
                "FROM metadata_imports WHERE id=:id"
            ),
            {"id": str(import_id)},
        ).one()
        assert unchanged.candidate_json is None
        assert unchanged.candidate_options_json == row.candidate_options_json
        assert unchanged.selected_metadata_source is None
        assert unchanged.raw_metadata_json == original_raw_metadata
        assert unchanged.warnings_json == original_warnings
        assert unchanged.metadata_status == "needs_mapping"
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            "0012_phase6_legacy_metadata_candidates"
        )


def test_phase6_migration_empty_head_downgrade_upgrade_roundtrip(tmp_path: Path) -> None:
    database_path = tmp_path / "migration-empty-roundtrip.sqlite3"
    config = _alembic_config(database_path)

    command.upgrade(config, "head")
    command.downgrade(config, "-1")
    command.upgrade(config, "head")

    with create_image_studio_engine(
        Settings(
            _env_file=None,
            environment="test",
            data_dir=tmp_path,
            database_url=f"sqlite:///{database_path.as_posix()}",
        )
    ).connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            "0013_phase7_drive_sync"
        )


@pytest.mark.parametrize(
    ("method", "operation"),
    [
        (UpscaleMethod.IMAGE, "worker"),
        (UpscaleMethod.LATENT, "worker"),
        (UpscaleMethod.IMAGE, "mutated"),
        (UpscaleMethod.IMAGE, "reconcile"),
        (UpscaleMethod.LATENT, "reconcile"),
    ],
)
def test_external_upscale_runs_through_worker_with_import_provenance(
    tmp_path: Path, method: UpscaleMethod, operation: str
) -> None:
    database_path = tmp_path / f"external-{method.value}.sqlite3"
    config = _alembic_config(database_path)
    command.upgrade(config, "head")
    upscaler_dir = tmp_path / "upscalers"
    upscaler_dir.mkdir()
    (upscaler_dir / "4x.pth").write_bytes(b"fake")
    settings = Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path,
        database_url=f"sqlite:///{database_path.as_posix()}",
        upscaler_dir=upscaler_dir,
    )
    engine = create_image_studio_engine(settings)
    factory = create_session_factory(engine)
    metadata_repository = MetadataImportRepository(factory)
    imported_storage = ImportedImageStorage(settings)
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
            checkpoints=("model.safetensors",),
            vaes=(),
            samplers=("euler",),
            schedulers=("normal",),
            loras=(),
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
                }
            ),
            warnings=(),
        )
    )
    metadata_service = MetadataImportService(
        metadata_repository,
        imported_storage,
        settings,
        capabilities=capabilities if method is UpscaleMethod.LATENT else None,
    )
    source_bytes = _png_bytes()
    if method is UpscaleMethod.LATENT:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("prompt", json.dumps(_prompt_graph()))
        output = BytesIO()
        Image.new("RGB", (512, 512), "white").save(output, format="PNG", pnginfo=metadata)
        source_bytes = output.getvalue()
    preview = metadata_service.import_image(source_bytes, "external.png")
    dispatch = GenerationDispatchQueueRepository(factory)
    enqueue = UpscaleEnqueueService(
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        dispatch,
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
        metadata_import_repository=metadata_repository,
        imported_image_storage=imported_storage,
        upscale_settings_repository=UpscaleSettingsRepository(factory),
        capabilities=capabilities,
    )
    item = enqueue.enqueue_import(
        preview.id,
        UpscaleSettings(
            method=method,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth" if method is UpscaleMethod.IMAGE else None,
            denoise=0.35 if method is UpscaleMethod.LATENT else None,
        ),
    )
    if operation == "mutated":
        imported_storage.absolute_path(preview.imported_image).write_bytes(b"changed source")

    output_bytes = _png_bytes((1024, 1024), "blue")
    expected_prompt = (
        f"external-{method.value}-reconcile"
        if operation == "reconcile"
        else f"external-{method.value}-prompt"
    )

    class FakeClient:
        uploads = 0
        prompts = 0
        uploaded_bytes = b""
        uploaded_sha256 = ""
        last_workflow: object = None

        async def upload_input_image(
            self, image_bytes: bytes, generation_id: object, source_sha256: str
        ) -> ComfyUIOutputImage:
            self.uploaded_bytes = image_bytes
            self.uploaded_sha256 = source_sha256
            self.uploads += 1
            return ComfyUIOutputImage("uploaded.png", "", "input")

        async def queue_prompt(self, workflow: object, client_id: str) -> QueuedPrompt:
            del client_id
            self.last_workflow = workflow
            self.prompts += 1
            return QueuedPrompt(expected_prompt, 1, {})

        async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
            return PromptHistory(
                prompt_id,
                True,
                False,
                (ComfyUIOutputImage("upscaled.png", "", "output"),),
                None,
            )

        async def get_output_image(self, output: ComfyUIOutputImage) -> bytes:
            del output
            return output_bytes

    class FakeWebSocket:
        async def watch_prompt(self, prompt_id: str, client_id: str):
            del prompt_id, client_id
            if False:
                yield GenerationProgress()

    async def capability_provider() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    generations = GenerationRepository(factory)
    jobs = GenerationJobRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    start = GenerationStartRepository(factory)
    completion = GenerationCompletionRepository(factory)
    failure = GenerationFailureRepository(factory)
    progress = GenerationProgressRepository(factory)
    queue = GenerationQueueRepository(factory)
    client = FakeClient()
    if operation == "reconcile":
        claimed = dispatch.claim_next("phase6-test-worker", lease_seconds=60)
        assert claimed is not None
        submitting = dispatch.begin_submission(claimed.entry.sequence, "phase6-test-worker")
        token = submitting.entry.submission_token
        assert token is not None
        dispatch.mark_submitted(
            claimed.entry.sequence,
            "phase6-test-worker",
            token,
            expected_prompt,
        )
        dispatch.release_claim(claimed.entry.sequence, "phase6-test-worker")
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
            failure=failure,
        ),
        upscale_settings_repository=UpscaleSettingsRepository(factory),
        upscale_workflow_adapter=UpscaleWorkflowAdapter(
            load_workflow_template("sdxl_image_upscale").as_mapping(),
            load_workflow_template("sdxl_latent_upscale").as_mapping(),
        ),
        upscaler_catalog=UpscalerCatalog(("4x.pth",)),
        metadata_import_repository=metadata_repository,
        imported_image_storage=imported_storage,
    )
    execution = GenerationExecutionService(service, dispatch, UpscaleService(service))
    if operation == "reconcile":
        assert (
            asyncio.run(service.reconcile_prompt(item.generation.id, expected_prompt))
            is ReconciliationOutcome.COMPLETED
        )
    else:
        worker = GenerationQueueWorker(
            dispatch, execution, settings, worker_id="phase6-test-worker"
        )
        assert asyncio.run(worker.run_once()) is True

    persisted_generation = generations.get_by_id(item.generation.id)
    persisted_job = jobs.get_by_generation(item.generation.id)
    persisted_snapshot = UpscaleSettingsRepository(factory).get_by_generation(item.generation.id)
    assert persisted_generation is not None
    assert persisted_job is not None
    assert persisted_snapshot is not None
    assert persisted_generation.parent_generation_id is None
    if operation == "mutated":
        assert persisted_generation.status is GenerationStatus.FAILED
        assert persisted_generation.error_code == "metadata_import_source_changed"
        assert persisted_job.status is GenerationStatus.FAILED
        assert persisted_generation.comfy_prompt_id is None
        assert persisted_job.prompt_id is None
        assert client.uploads == 0
        assert client.prompts == 0
        return
    comparison = enqueue.comparison_for_generation(item.generation.id)
    assert comparison.parent_generation_id is None
    assert comparison.gallery[0][1] == "source"
    assert comparison.gallery[1][1] == "upscaled"
    assert Path(comparison.gallery[0][0]).exists()
    assert Path(comparison.gallery[1][0]).exists()
    assert persisted_generation.status is GenerationStatus.COMPLETED, (
        persisted_generation.error_code,
        persisted_generation.error_summary,
        client.uploads,
        client.prompts,
        client.uploaded_sha256,
        client.last_workflow,
    )
    assert persisted_job.status is GenerationStatus.COMPLETED
    assert persisted_generation.comfy_prompt_id == expected_prompt
    assert persisted_job.prompt_id == expected_prompt
    assert persisted_snapshot.source_kind is UpscaleSourceKind.METADATA_IMPORT
    assert persisted_snapshot.source_import_id == preview.id
    assert persisted_snapshot.source_sha256 == preview.imported_image.stored_image_sha256
    assert artifacts.get_primary_image(item.generation.id) is not None
    if operation == "reconcile":
        assert client.uploads == 0
        assert client.prompts == 0
    else:
        assert client.uploaded_bytes == imported_storage.read_verified(preview.imported_image)
        assert client.uploaded_sha256 == preview.imported_image.stored_image_sha256
        assert client.last_workflow["1"]["inputs"]["image"] == "uploaded.png"  # type: ignore[index]
        if method is UpscaleMethod.IMAGE:
            assert client.last_workflow["2"]["inputs"]["model_name"] == "4x.pth"  # type: ignore[index]
        else:
            assert client.last_workflow["2"]["inputs"]["ckpt_name"] == "model.safetensors"  # type: ignore[index]
            assert client.last_workflow["3"]["inputs"]["text"] == "a cat"  # type: ignore[index]
            assert client.last_workflow["4"]["inputs"]["text"] == "blur"  # type: ignore[index]
            assert "vae_external" not in client.last_workflow  # type: ignore[operator]
        assert client.uploads == 1
        assert client.prompts == 1


def test_external_upscale_retry_preserves_source_and_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "external-retry.sqlite3"
    config = _alembic_config(database_path)
    command.upgrade(config, "head")
    upscaler_dir = tmp_path / "upscalers"
    upscaler_dir.mkdir()
    (upscaler_dir / "4x.pth").write_bytes(b"fake")
    settings = Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path,
        database_url=f"sqlite:///{database_path.as_posix()}",
        upscaler_dir=upscaler_dir,
    )
    engine = create_image_studio_engine(settings)
    factory = create_session_factory(engine)
    metadata_repository = MetadataImportRepository(factory)
    imported_storage = ImportedImageStorage(settings)
    preview = MetadataImportService(metadata_repository, imported_storage, settings).import_image(
        _png_bytes(), "retry.png"
    )
    dispatch = GenerationDispatchQueueRepository(factory)
    enqueue = UpscaleEnqueueService(
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        dispatch,
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
        metadata_import_repository=metadata_repository,
        imported_image_storage=imported_storage,
    )
    original = enqueue.enqueue_import(
        preview.id,
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth",
        ),
    )
    GenerationFailureRepository(factory).fail_generation(
        original.generation.id,
        original.job.id,
        error_code="metadata_import_source_changed",
        error_summary="source changed",
        failed_at=datetime.now(UTC),
    )

    queue_service = GenerationQueueService(
        dispatch,
        settings,
        upscale_settings_repository=UpscaleSettingsRepository(factory),
    )
    first_retry = queue_service.retry(original.generation.id).item
    second_retry = queue_service.retry(original.generation.id).item
    retry_snapshot = UpscaleSettingsRepository(factory).get_by_generation(first_retry.generation.id)

    assert first_retry.generation.id != original.generation.id
    assert first_retry.generation.retry_of_generation_id == original.generation.id
    assert first_retry.generation.retry_attempt == 1
    assert second_retry.generation.id == first_retry.generation.id
    assert first_retry.generation.comfy_prompt_id is None
    assert first_retry.job.prompt_id is None
    assert retry_snapshot is not None
    assert retry_snapshot.source_kind is UpscaleSourceKind.METADATA_IMPORT
    assert retry_snapshot.source_import_id == preview.id
    assert retry_snapshot.source_sha256 == preview.imported_image.stored_image_sha256
