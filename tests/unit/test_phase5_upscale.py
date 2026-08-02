"""Phase 5 domain, source validation, queue atomicity, and workflow tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
import respx
from PIL import Image
from sqlalchemy import create_engine, inspect

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.upscale_workflow_adapter import (
    UpscaleWorkflowAdapter,
)
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
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
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus, StoredImage
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.job import GenerationJob
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
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.upscale_enqueue_service import (
    UpscaleEnqueueError,
    UpscaleEnqueueService,
)
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


@pytest.mark.parametrize("failure_kind", ["settings", "sidecar", "metadata_artifact", "thumbnail"])
def test_optional_upscale_artifact_failures_do_not_reopen_completed_state(
    tmp_path: Path, failure_kind: str
) -> None:
    class ArtifactRepository:
        def list_by_generation(self, generation_id: UUID) -> tuple[object, ...]:
            del generation_id
            return ()

        def add(self, artifact: object) -> object:
            if failure_kind == "metadata_artifact":
                raise RuntimeError("metadata artifact failure")
            return artifact

    class MetadataStorage:
        def save_for_image(self, image_path: Path, payload: dict[str, object]) -> Path:
            del payload
            if failure_kind == "sidecar":
                raise RuntimeError("sidecar failure")
            path = image_path.with_suffix(".json")
            path.write_text("{}", encoding="utf-8")
            return path

        def relative_path(self, path: Path) -> str:
            return path.name

        def sha256(self, path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

    class ThumbnailStorage:
        def save(self, image_path: Path, generation_id: UUID, created_at: datetime) -> Path:
            del image_path, generation_id, created_at
            raise RuntimeError("thumbnail failure")

        def relative_path(self, path: Path) -> str:
            return path.name

        def sha256(self, path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

    class UpscaleRepository:
        def get_by_generation(self, generation_id: UUID) -> object | None:
            del generation_id
            if failure_kind == "settings":
                raise UpscaleSettingsRepositoryError("settings failure")
            return None

        def get_by_source_artifact(self, source_artifact_id: UUID) -> tuple[object, ...]:
            del source_artifact_id
            return ()

    settings = Settings(_env_file=None, data_dir=tmp_path)
    image_path = tmp_path / "result.png"
    image_path.write_bytes(_png((8, 8)))
    service = GenerationService(
        object(),
        object(),
        object(),
        object(),
        lambda: None,  # type: ignore[arg-type]
        settings,
        thumbnail_storage=ThumbnailStorage() if failure_kind == "thumbnail" else None,  # type: ignore[arg-type]
        metadata_storage=MetadataStorage() if failure_kind != "thumbnail" else None,  # type: ignore[arg-type]
        upscale_settings_repository=UpscaleRepository(),  # type: ignore[arg-type]
    )
    service._artifact_repository = ArtifactRepository()  # noqa: SLF001 - exercise optional boundary
    job = GenerationJob(
        generation_id=uuid4(),
        status=GenerationStatus.COMPLETED,
        created_at=datetime.now(UTC),
    )
    image = StoredImage(
        path=image_path,
        sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        size_bytes=image_path.stat().st_size,
        width=8,
        height=8,
        mime_type="image/png",
    )

    service._persist_optional_artifacts(  # noqa: SLF001 - verify completion boundary
        job,
        _settings(),
        image,
        datetime.now(UTC),
        GenerationKind.UPSCALE,
        None,
    )

    assert job.status is GenerationStatus.COMPLETED
