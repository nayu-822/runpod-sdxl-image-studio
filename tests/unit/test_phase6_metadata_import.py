"""Phase 6 external image validation, metadata parsing, and source provenance tests."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image, PngImagePlugin
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationRepository,
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
from runpod_sdxl_image_studio.adapters.metadata.png_metadata_adapter import parse_png_metadata
from runpod_sdxl_image_studio.adapters.storage.imported_image_storage import (
    ImportedImageStorage,
    ImportedImageStorageError,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.metadata_import import (
    MetadataImportStatus,
    MetadataModelMapping,
)
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSourceKind
from runpod_sdxl_image_studio.services.metadata_import_service import MetadataImportService
from runpod_sdxl_image_studio.services.upscale_enqueue_service import UpscaleEnqueueService


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path,
        database_url=f"sqlite:///{(tmp_path / 'image-studio.sqlite3').as_posix()}",
    )


def _capabilities(
    *,
    checkpoints: tuple[str, ...] = ("model.safetensors",),
) -> ComfyUICapabilities:
    return ComfyUICapabilities(
        checkpoints=checkpoints,
        vaes=(),
        samplers=("euler",),
        schedulers=("normal",),
        loras=(),
        upscale_models=(),
        available_node_classes=frozenset(
            {"CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "EmptyLatentImage"}
        ),
        warnings=(),
    )


def _prompt_graph(
    *, checkpoint: str = "model.safetensors", sampler_count: int = 1
) -> dict[str, object]:
    graph: dict[str, object] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
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
    }
    for index in range(sampler_count):
        node_id = str(5 + index)
        graph[node_id] = {
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
        }
    return graph


def _png(*, prompt: dict[str, object] | None = None, workflow: str | None = None) -> bytes:
    output = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
    if workflow is not None:
        metadata.add_text("workflow", workflow)
    Image.new("RGB", (512, 512), "white").save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


def _service(
    tmp_path: Path, capabilities: ComfyUICapabilities | None = None
) -> tuple[MetadataImportService, MetadataImportRepository, ImportedImageStorage]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = _settings(tmp_path)
    engine = create_engine(settings.database_url or "sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    repository = MetadataImportRepository(factory)
    storage = ImportedImageStorage(settings)
    return (
        MetadataImportService(repository, storage, settings, capabilities=capabilities),
        repository,
        storage,
    )


def test_external_image_is_canonicalized_and_raw_known_metadata_is_retained(tmp_path: Path) -> None:
    service, repository, storage = _service(tmp_path, _capabilities())
    payload = _png(prompt=_prompt_graph(), workflow='{"nodes": "raw only"}')

    preview = service.import_image(payload, "..\\external.png")
    record = repository.get_by_id(preview.id)

    assert preview.status is MetadataImportStatus.READY
    assert preview.imported_image.original_filename == "external.png"
    assert preview.imported_image.stored_image_path.startswith("imports/")
    assert preview.imported_image.stored_image_path.endswith(".png")
    assert preview.imported_image.source_image_sha256 == hashlib.sha256(payload).hexdigest()
    assert record is not None
    assert any(source.kind.value == "workflow" for source in record.raw_sources)
    assert storage.absolute_path(record.imported_image).exists()
    assert storage.absolute_path(record.imported_image).read_bytes() != payload


def test_webp_is_accepted_but_stored_as_png_and_metadata_can_be_missing(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path, _capabilities())
    output = BytesIO()
    Image.new("RGB", (512, 512), "blue").save(output, format="WEBP")

    preview = service.import_image(output.getvalue(), "image.webp")

    assert preview.status is MetadataImportStatus.METADATA_MISSING
    assert preview.imported_image.image_mime_type == "image/png"
    assert preview.imported_image.stored_image_path.endswith(".png")


def test_sidecar_model_mapping_is_explicit_and_exact(tmp_path: Path) -> None:
    service, repository, _ = _service(tmp_path, _capabilities())
    payload = _png()
    sidecar = {
        "schema_version": 1,
        "image": {"sha256": hashlib.sha256(payload).hexdigest()},
        "settings": {
            "positive_prompt": "a cat",
            "negative_prompt": "blur",
            "seed": 42,
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7,
            "sampler_name": "euler",
            "scheduler_name": "normal",
            "checkpoint_name": "old-model.safetensors",
            "loras": [],
        },
    }

    preview = service.import_image(payload, sidecar_bytes=json.dumps(sidecar))
    assert preview.status is MetadataImportStatus.NEEDS_MAPPING
    mapped = service.apply_model_mapping(
        preview.id,
        (
            MetadataModelMapping(
                model_kind="checkpoint",
                source_name="old-model.safetensors",
                target_name="model.safetensors",
            ),
        ),
    )

    assert mapped.status is MetadataImportStatus.READY
    assert service.build_generation_settings(mapped.id).checkpoint_name == "model.safetensors"
    assert repository.get_by_id(mapped.id).normalized_snapshot_schema_version == 1


def test_model_catalog_none_and_empty_are_distinct(tmp_path: Path) -> None:
    unavailable, _, _ = _service(tmp_path / "unavailable", None)
    unavailable_preview = unavailable.import_image(_png(prompt=_prompt_graph()))
    assert unavailable_preview.status is MetadataImportStatus.NEEDS_MAPPING
    assert "metadata_import_model_catalog_unavailable" in unavailable_preview.warnings

    empty, _, _ = _service(tmp_path / "empty", _capabilities(checkpoints=()))
    empty_preview = empty.import_image(_png(prompt=_prompt_graph()))
    assert empty_preview.status is MetadataImportStatus.NEEDS_MAPPING
    assert "metadata_import_model_missing" in empty_preview.warnings


def test_comfy_parser_uses_connections_and_rejects_ambiguous_sampler_graph() -> None:
    parsed = parse_comfyui_prompt_metadata(_prompt_graph(sampler_count=2))

    assert "sampler_graph" in parsed.unresolved_fields
    assert parsed.candidate.checkpoint_name is None
    assert parsed.candidate.loras == ()


def test_comfy_parser_preserves_connected_lora_order_and_strengths() -> None:
    graph = _prompt_graph()
    graph["6"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "first.safetensors",
            "strength_model": 0.7,
            "strength_clip": 0.8,
            "model": ["1", 0],
            "clip": ["1", 1],
        },
    }
    graph["7"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "second.safetensors",
            "strength_model": 0.4,
            "strength_clip": 0.5,
            "model": ["6", 0],
            "clip": ["6", 1],
        },
    }
    graph["5"]["inputs"]["model"] = ["7", 0]  # type: ignore[index]

    parsed = parse_comfyui_prompt_metadata(graph)

    assert [
        (lora.name, lora.model_strength, lora.clip_strength) for lora in parsed.candidate.loras
    ] == [
        ("first.safetensors", 0.7, 0.8),
        ("second.safetensors", 0.4, 0.5),
    ]


def test_workflow_metadata_is_raw_only_and_never_parsed_as_generation_input() -> None:
    result = parse_png_metadata(_png(workflow='{"class_type": "PythonNode", "code": "exec()"}'))

    assert result.workflow is not None
    assert all(source.kind.value != "comfyui_prompt" for source in result.raw_sources)
    assert result.prompt is None


def test_import_source_change_is_detected_before_upscale_enqueue(tmp_path: Path) -> None:
    service, repository, storage = _service(tmp_path, _capabilities())
    preview = service.import_image(_png(), "source.png")
    record = repository.get_by_id(preview.id)
    assert record is not None
    path = storage.absolute_path(record.imported_image)
    path.write_bytes(b"changed source")

    with pytest.raises(ImportedImageStorageError) as error:
        storage.verify(record.imported_image)
    assert error.value.code == "metadata_import_source_changed"


def test_external_image_upscale_persists_import_source_without_parent(tmp_path: Path) -> None:
    service, metadata_repository, storage = _service(tmp_path, _capabilities())
    preview = service.import_image(_png(), "source.png")
    settings = _settings(tmp_path)
    engine = create_engine(settings.database_url or "sqlite:///:memory:")
    factory = create_session_factory(engine)
    generations = GenerationRepository(factory)
    artifacts = GenerationArtifactRepository(factory)
    queue = GenerationDispatchQueueRepository(factory)
    upscale_settings = UpscaleSettingsRepository(factory)
    enqueue = UpscaleEnqueueService(
        generations,
        artifacts,
        queue,
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
        metadata_import_repository=metadata_repository,
        imported_image_storage=storage,
    )

    item = enqueue.enqueue_import(
        preview.id,
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth",
        ),
    )
    snapshot = upscale_settings.get_by_generation(item.generation.id)

    assert item.generation.parent_generation_id is None
    assert snapshot is not None
    assert snapshot.source_kind is UpscaleSourceKind.METADATA_IMPORT
    assert snapshot.source_import_id == UUID(str(preview.id))
    assert snapshot.source_artifact_id is None
