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
from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
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
from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
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
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
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


@pytest.mark.parametrize(
    "failure_kind",
    [
        "settings",
        "sidecar",
        "metadata_artifact",
        "thumbnail",
        "thumbnail_artifact",
        "artifact_lookup",
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
            if failure_kind == "artifact_lookup":
                raise RuntimeError("artifact lookup failure")
            return actual_artifacts.list_by_generation(generation_id)

        def add(self, artifact: GenerationArtifact) -> GenerationArtifact:
            if artifact.artifact_type is ArtifactType.METADATA:
                counters["metadata_add"] += 1
                if failure_kind == "metadata_artifact" and counters["metadata_add"] == 1:
                    raise RuntimeError("metadata artifact failure")
            if artifact.artifact_type is ArtifactType.THUMBNAIL:
                counters["thumbnail_add"] += 1
                if failure_kind == "thumbnail_artifact":
                    raise RuntimeError("thumbnail artifact failure")
            return actual_artifacts.add(artifact)

    class MetadataStorage:
        def save_for_image(self, image_path: Path, payload: dict[str, object]) -> Path:
            counters["metadata_save"] += 1
            if failure_kind == "sidecar":
                raise RuntimeError("sidecar failure")
            del payload
            path = image_path.with_suffix(".json")
            path.write_text("{}", encoding="utf-8")
            return path

        def relative_path(self, path: Path) -> str:
            counters["metadata_relative"] += 1
            if failure_kind == "metadata_relative":
                raise RuntimeError("metadata relative path failure")
            return path.name

        def sha256(self, path: Path) -> str:
            counters["metadata_sha"] += 1
            if failure_kind == "metadata_sha":
                raise RuntimeError("metadata sha256 failure")
            return hashlib.sha256(path.read_bytes()).hexdigest()

    class ThumbnailStorage:
        def save(self, image_path: Path, generation_id: UUID, created_at: datetime) -> Path:
            counters["thumbnail_save"] += 1
            if failure_kind == "thumbnail":
                raise RuntimeError("thumbnail failure")
            path = image_path.with_name(f"{generation_id}.webp")
            path.write_bytes(b"thumbnail")
            del created_at
            return path

        def relative_path(self, path: Path) -> str:
            return path.name

        def sha256(self, path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

    class UpscaleRepository:
        calls = 0

        def get_by_generation(self, generation_id: UUID) -> object | None:
            del generation_id
            self.calls += 1
            if failure_kind == "settings":
                raise UpscaleSettingsRepositoryError("settings failure")
            return None

        def get_by_source_artifact(self, source_artifact_id: UUID) -> tuple[object, ...]:
            del source_artifact_id
            return ()

    class FailureRepository:
        calls = 0

        def fail_generation(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.calls += 1
            raise AssertionError("optional artifact failure must not persist generation failure")

    metadata_storage = (
        MetadataStorage()
        if failure_kind
        not in {
            "thumbnail",
            "thumbnail_artifact",
            "artifact_lookup",
        }
        else None
    )
    thumbnail_storage = (
        ThumbnailStorage()
        if failure_kind
        in {
            "thumbnail",
            "thumbnail_artifact",
        }
        else None
    )
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
        assert optional_types == set()
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
            "metadata_relative": (
                counters["metadata_relative"] >= 1,
                "metadata relative path calculation",
            ),
            "metadata_sha": (counters["metadata_sha"] >= 1, "metadata sha256 calculation"),
        }
        failure_exercised, failure_description = expected_counters[failure_kind]
        assert failure_exercised, failure_description

        if failure_kind == "metadata_artifact":
            service._persist_optional_artifacts(  # noqa: SLF001 - verify retry after optional failure
                job,
                child_settings,
                output,
                now,
                GenerationKind.UPSCALE,
                parent_id,
            )
            service._persist_optional_artifacts(  # noqa: SLF001 - verify retry idempotency
                job,
                child_settings,
                output,
                now,
                GenerationKind.UPSCALE,
                parent_id,
            )
            retried_artifacts = actual_artifacts.list_by_generation(child_id)
            assert len([a for a in retried_artifacts if a.artifact_type is ArtifactType.IMAGE]) == 1
            assert (
                len([a for a in retried_artifacts if a.artifact_type is ArtifactType.METADATA]) == 1
            )
    finally:
        engine.dispose()
