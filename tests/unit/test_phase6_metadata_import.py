"""Phase 6 external image validation, metadata parsing, and source provenance tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

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
from runpod_sdxl_image_studio.adapters.metadata.sidecar_metadata_adapter import (
    parse_sidecar_metadata,
)
from runpod_sdxl_image_studio.adapters.storage.imported_image_storage import (
    ImportedImageStorage,
    ImportedImageStorageError,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.metadata_import import (
    MetadataImportError,
    MetadataImportStatus,
    MetadataModelMapping,
    MetadataRawSource,
    MetadataSourceKind,
)
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.domain.upscale_snapshot import UpscaleSourceKind
from runpod_sdxl_image_studio.services.metadata_import_service import MetadataImportService
from runpod_sdxl_image_studio.services.upscale_enqueue_service import (
    UpscaleEnqueueError,
    UpscaleEnqueueService,
)
from runpod_sdxl_image_studio.ui.tabs.metadata_import_tab import (
    make_metadata_import_handler,
    make_metadata_mapping_handler,
    make_metadata_source_selection_handler,
)


def _settings(tmp_path: Path, *, max_metadata_sidecar_bytes: int = 4_000_000) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path,
        database_url=f"sqlite:///{(tmp_path / 'image-studio.sqlite3').as_posix()}",
        max_metadata_sidecar_bytes=max_metadata_sidecar_bytes,
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
    graph["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
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


def _sidecar_settings(**overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "positive_prompt": "a cat",
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
    }
    settings.update(overrides)
    return {"schema_version": 1, "settings": settings}


def _service(
    tmp_path: Path,
    capabilities: ComfyUICapabilities | None = None,
    *,
    max_metadata_sidecar_bytes: int = 4_000_000,
) -> tuple[MetadataImportService, MetadataImportRepository, ImportedImageStorage]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = _settings(tmp_path, max_metadata_sidecar_bytes=max_metadata_sidecar_bytes)
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


def test_sidecar_preserves_empty_and_whitespace_prompts_and_explicit_zero_values() -> None:
    payload = _sidecar_settings(
        positive_prompt="  ",
        negative_prompt="",
        loras=[
            {
                "name": "style.safetensors",
                "model_strength": 0.0,
                "clip_strength": 0.0,
                "order": 4,
            }
        ],
    )

    result = parse_sidecar_metadata(json.dumps(payload))

    assert result.candidate.is_generation_ready
    assert result.candidate.positive_prompt == "  "
    assert result.candidate.negative_prompt == ""
    assert result.candidate.loras[0].model_strength == 0.0
    assert result.candidate.loras[0].clip_strength == 0.0
    assert result.candidate.loras[0].order == 4


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"loras": None}, "loras"),
        (
            {"loras": [{"name": "style.safetensors", "model_strength": 1.0, "clip_strength": 1.0}]},
            "loras",
        ),
        (
            {
                "loras": [
                    {
                        "name": "style.safetensors",
                        "model_strength": 1.0,
                        "clip_strength": 1.0,
                        "order": 0,
                    },
                    {
                        "name": "other.safetensors",
                        "model_strength": 1.0,
                        "clip_strength": 1.0,
                        "order": 0,
                    },
                ]
            },
            "loras",
        ),
        (
            {
                "loras": [
                    {
                        "name": "style.safetensors",
                        "model_strength": 1.0,
                        "clip_strength": 1.0,
                        "order": -1,
                    }
                ]
            },
            "loras",
        ),
        ({"vae_name": "../outside.safetensors"}, "vae_name"),
    ],
)
def test_sidecar_invalid_or_ambiguous_fields_are_unresolved(
    override: dict[str, object], field: str
) -> None:
    result = parse_sidecar_metadata(json.dumps(_sidecar_settings(**override)))

    assert field in result.candidate.unresolved_fields
    assert not result.candidate.is_generation_ready


def test_sidecar_missing_loras_and_missing_vae_are_not_silently_defaulted() -> None:
    payload = _sidecar_settings()
    del payload["settings"]["loras"]  # type: ignore[index]
    del payload["settings"]["vae_name"]  # type: ignore[index]

    result = parse_sidecar_metadata(json.dumps(payload))

    assert {"loras", "vae_name"}.issubset(result.candidate.unresolved_fields)


def test_metadata_raw_source_accepts_one_mb_plus_below_byte_contract() -> None:
    source = MetadataRawSource(
        kind=MetadataSourceKind.APP_SIDECAR,
        raw_text="x" * 1_100_000,
        sha256="0" * 64,
    )

    assert len(source.raw_text) == 1_100_000


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


def _make_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


@pytest.mark.parametrize("symlink_level", ["imports", "images"])
def test_import_storage_rejects_symlink_escape_before_writing(
    tmp_path: Path, symlink_level: str
) -> None:
    service, repository, storage = _service(tmp_path, _capabilities())
    outside = tmp_path / "outside"
    outside.mkdir()
    local_date = datetime.now(UTC).astimezone(ZoneInfo("Asia/Tokyo")).date().isoformat()
    if symlink_level == "imports":
        _make_directory_symlink(tmp_path / "imports", outside)
    else:
        (tmp_path / "imports" / local_date).mkdir(parents=True)
        _make_directory_symlink(tmp_path / "imports" / local_date / "images", outside)

    with pytest.raises(ImportedImageStorageError) as error:
        service.import_image(_png(), "outside.png")

    assert error.value.code == "metadata_import_storage_failed"
    assert not list(outside.glob("*.png"))
    assert repository.list_recent() == ()
    assert storage.data_dir == tmp_path


def test_valid_png_can_be_selected_when_sidecar_is_invalid(tmp_path: Path) -> None:
    service, repository, _ = _service(tmp_path, _capabilities())
    preview = service.import_image(
        _png(prompt=_prompt_graph()),
        "valid-png-invalid-sidecar.png",
        sidecar_bytes=b"{malformed",
    )

    assert preview.status is MetadataImportStatus.INVALID_METADATA
    selected = service.select_metadata_source(preview.id, MetadataSourceKind.COMFYUI_PROMPT)

    assert selected.status is MetadataImportStatus.READY
    assert selected.selected_metadata_source is MetadataSourceKind.COMFYUI_PROMPT
    assert "metadata_import_sidecar_invalid_ignored" in selected.warnings
    assert "metadata_import_invalid_json" not in selected.warnings
    assert service.build_generation_settings(preview.id).positive_prompt == "a cat"
    persisted = repository.get_by_id(preview.id)
    assert persisted is not None
    assert persisted.metadata_status is MetadataImportStatus.READY


def test_valid_sidecar_can_be_selected_when_png_prompt_is_invalid(tmp_path: Path) -> None:
    output = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", "{malformed")
    Image.new("RGB", (512, 512), "white").save(output, format="PNG", pnginfo=metadata)
    service, repository, _ = _service(tmp_path, _capabilities())
    preview = service.import_image(
        output.getvalue(),
        "invalid-png-valid-sidecar.png",
        sidecar_bytes=json.dumps(_sidecar_settings()),
    )

    assert preview.status is MetadataImportStatus.INVALID_METADATA
    selected = service.select_metadata_source(preview.id, MetadataSourceKind.APP_SIDECAR)

    assert selected.status is MetadataImportStatus.READY
    assert selected.selected_metadata_source is MetadataSourceKind.APP_SIDECAR
    assert "metadata_import_png_prompt_invalid_ignored" in selected.warnings
    assert "metadata_import_png_prompt_invalid" not in selected.warnings
    assert service.build_generation_settings(preview.id).positive_prompt == "a cat"
    persisted = repository.get_by_id(preview.id)
    assert persisted is not None
    assert persisted.metadata_status is MetadataImportStatus.READY


def test_invalid_png_and_sidecar_keep_import_invalid_and_image_upscale_available(
    tmp_path: Path,
) -> None:
    output = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", "{malformed")
    Image.new("RGB", (512, 512), "white").save(output, format="PNG", pnginfo=metadata)
    service, _, storage = _service(tmp_path, _capabilities())

    preview = service.import_image(
        output.getvalue(), "invalid-both.png", sidecar_bytes=b"{malformed"
    )

    assert preview.status is MetadataImportStatus.INVALID_METADATA
    assert preview.candidate is None
    assert storage.absolute_path(preview.imported_image).exists()
    with pytest.raises(MetadataImportError):
        service.build_generation_settings(preview.id)


def test_import_storage_rejects_symlink_that_resolves_inside_data_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    storage = ImportedImageStorage(settings)
    internal = tmp_path / "internal-images"
    internal.mkdir()
    (tmp_path / "imports" / "2025-01-02").mkdir(parents=True)
    _make_directory_symlink(tmp_path / "imports" / "2025-01-02" / "images", internal)

    with pytest.raises(ImportedImageStorageError) as error:
        storage.store(_png(), "internal.png", created_at=datetime(2025, 1, 2, tzinfo=UTC))

    assert error.value.code == "metadata_import_storage_failed"
    assert not list(internal.glob("*.png"))


@pytest.mark.parametrize(
    "sidecar_bytes",
    [
        b"{malformed",
        b"\xff\xfe\x00\x01",
        json.dumps({"schema_version": 99, "settings": {}}).encode("utf-8"),
    ],
)
def test_invalid_sidecar_does_not_cancel_canonical_image_import(
    tmp_path: Path, sidecar_bytes: bytes
) -> None:
    service, repository, storage = _service(tmp_path, _capabilities())

    preview = service.import_image(_png(), "invalid-sidecar.png", sidecar_bytes=sidecar_bytes)
    record = repository.get_by_id(preview.id)

    assert record is not None
    assert preview.status is MetadataImportStatus.INVALID_METADATA
    assert storage.absolute_path(record.imported_image).exists()
    assert any(source.kind is MetadataSourceKind.APP_SIDECAR for source in record.raw_sources)
    assert any(
        warning in preview.warnings
        for warning in (
            "metadata_import_invalid_json",
            "metadata_import_invalid_utf8",
            "metadata_import_unsupported_schema",
        )
    )
    with pytest.raises(MetadataImportError):
        service.build_generation_settings(preview.id)


def test_invalid_sidecar_allows_image_upscale_but_rejects_latent_upscale(tmp_path: Path) -> None:
    service, metadata_repository, storage = _service(tmp_path, _capabilities())
    preview = service.import_image(_png(), sidecar_bytes=b"not-json")
    settings = _settings(tmp_path)
    engine = create_engine(settings.database_url or "sqlite:///:memory:")
    factory = create_session_factory(engine)
    dispatch = GenerationDispatchQueueRepository(factory)
    enqueue = UpscaleEnqueueService(
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        dispatch,
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
        metadata_import_repository=metadata_repository,
        imported_image_storage=storage,
        capabilities=_capabilities(),
    )

    image_item = enqueue.enqueue_import(
        preview.id,
        UpscaleSettings(
            method=UpscaleMethod.IMAGE,
            sizing_mode=UpscaleSizingMode.FACTOR,
            scale_factor=2,
            upscaler_name="4x.pth",
        ),
    )
    assert image_item.generation.parent_generation_id is None
    with pytest.raises(UpscaleEnqueueError) as error:
        enqueue.enqueue_import(
            preview.id,
            UpscaleSettings(
                method=UpscaleMethod.LATENT,
                sizing_mode=UpscaleSizingMode.FACTOR,
                scale_factor=2,
                denoise=0.35,
            ),
        )
    assert error.value.code == "metadata_import_unresolved"


def test_invalid_png_prompt_is_distinguished_from_missing_metadata(tmp_path: Path) -> None:
    output = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", "{malformed")
    Image.new("RGB", (512, 512), "white").save(output, format="PNG", pnginfo=metadata)
    service, repository, storage = _service(tmp_path, _capabilities())

    preview = service.import_image(output.getvalue(), "invalid-prompt.png")
    record = repository.get_by_id(preview.id)

    assert record is not None
    assert preview.status is MetadataImportStatus.INVALID_METADATA
    assert "metadata_import_png_prompt_invalid" in preview.warnings
    assert any(
        source.kind is MetadataSourceKind.COMFYUI_PROMPT and source.raw_text == "{malformed"
        for source in record.raw_sources
    )
    assert storage.absolute_path(record.imported_image).exists()


def test_oversized_sidecar_keeps_hash_warning_without_storing_raw_text(tmp_path: Path) -> None:
    service, repository, storage = _service(
        tmp_path,
        _capabilities(),
        max_metadata_sidecar_bytes=16,
    )
    sidecar_bytes = b"x" * 17

    preview = service.import_image(_png(), sidecar_bytes=sidecar_bytes)
    record = repository.get_by_id(preview.id)

    assert record is not None
    assert preview.status is MetadataImportStatus.INVALID_METADATA
    assert "metadata_import_too_large" in preview.warnings
    sidecar_sources = [
        source for source in record.raw_sources if source.kind is MetadataSourceKind.APP_SIDECAR
    ]
    assert len(sidecar_sources) == 1
    assert sidecar_sources[0].raw_text is None
    assert sidecar_sources[0].sha256 == hashlib.sha256(sidecar_bytes).hexdigest()
    assert storage.absolute_path(record.imported_image).exists()


def test_png_sidecar_conflict_requires_persisted_source_selection_and_hash_confirmation(
    tmp_path: Path,
) -> None:
    service, repository, _ = _service(tmp_path, _capabilities())
    payload = _png(prompt=_prompt_graph())
    sidecar = _sidecar_settings(positive_prompt="sidecar prompt")
    sidecar["image"] = {"sha256": "0" * 64}

    preview = service.import_image(payload, sidecar_bytes=json.dumps(sidecar))

    assert preview.status is MetadataImportStatus.NEEDS_MAPPING
    assert preview.selected_metadata_source is None
    assert len(preview.candidates) == 2
    with pytest.raises(MetadataImportError) as error:
        service.select_metadata_source(preview.id, MetadataSourceKind.APP_SIDECAR)
    assert error.value.code == "metadata_import_sidecar_hash_confirmation_required"

    selected = service.select_metadata_source(
        preview.id,
        MetadataSourceKind.APP_SIDECAR,
        confirm_sidecar_hash_mismatch=True,
    )
    assert selected.selected_metadata_source is MetadataSourceKind.APP_SIDECAR
    assert selected.sidecar_hash_confirmed is True
    assert "metadata_import_sidecar_hash_mismatch" not in selected.warnings
    persisted = repository.get_by_id(preview.id)
    assert persisted is not None
    assert persisted.selected_metadata_source is MetadataSourceKind.APP_SIDECAR
    assert service.build_generation_settings(preview.id).positive_prompt == "sidecar prompt"


def test_repeated_import_is_idempotent_and_does_not_create_two_rows(tmp_path: Path) -> None:
    service, repository, storage = _service(tmp_path, _capabilities())
    payload = _png(prompt=_prompt_graph())

    first = service.import_image(payload, "double-click.png")
    second = service.import_image(payload, "double-click.png")

    assert second.id == first.id
    assert len(repository.list_recent()) == 1
    assert len(tuple(storage.data_dir.glob("imports/*/*/*.png"))) == 1


def test_unexpected_parser_failure_cleans_uncommitted_canonical_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, storage = _service(tmp_path, _capabilities())
    monkeypatch.setattr(
        "runpod_sdxl_image_studio.services.metadata_import_service.parse_comfyui_prompt_metadata",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("unexpected parser validation")),
    )

    with pytest.raises(MetadataImportError):
        service.import_image(_png(prompt=_prompt_graph()))

    assert not list((storage.data_dir / "imports").rglob("*.png"))


def test_repository_failure_cleans_file_only_when_row_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, repository, storage = _service(tmp_path, _capabilities())
    monkeypatch.setattr(
        repository,
        "create",
        lambda record: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(MetadataImportError):
        service.import_image(_png(prompt=_prompt_graph()))

    assert not list((storage.data_dir / "imports").rglob("*.png"))


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
            "vae_name": None,
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


def test_build_generation_settings_rechecks_capabilities_after_initial_unavailable_catalog(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path, None)
    preview = service.import_image(_png(prompt=_prompt_graph()))
    service.set_capabilities(_capabilities())

    settings = service.build_generation_settings(preview.id)

    assert settings.checkpoint_name == "model.safetensors"


def test_removed_capability_blocks_latent_import_before_queue_creation(tmp_path: Path) -> None:
    service, metadata_repository, storage = _service(tmp_path, _capabilities())
    preview = service.import_image(_png(prompt=_prompt_graph()))
    settings = _settings(tmp_path)
    engine = create_engine(settings.database_url or "sqlite:///:memory:")
    factory = create_session_factory(engine)
    dispatch = GenerationDispatchQueueRepository(factory)
    enqueue = UpscaleEnqueueService(
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        dispatch,
        settings,
        catalog=UpscalerCatalog(("4x.pth",)),
        metadata_import_repository=metadata_repository,
        imported_image_storage=storage,
        capabilities=_capabilities(checkpoints=()),
    )

    with pytest.raises(UpscaleEnqueueError) as error:
        enqueue.enqueue_import(
            preview.id,
            UpscaleSettings(
                method=UpscaleMethod.LATENT,
                sizing_mode=UpscaleSizingMode.FACTOR,
                scale_factor=2,
                denoise=0.35,
            ),
        )

    assert error.value.code == "metadata_import_model_missing"
    assert dispatch.list_queue() == ()


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


def test_comfy_parser_resolves_vae_only_from_selected_sampler_output() -> None:
    graph = _prompt_graph()
    graph["8"]["inputs"]["samples"] = ["4", 0]  # type: ignore[index]

    parsed = parse_comfyui_prompt_metadata(graph)

    assert "vae" in parsed.unresolved_fields


def test_comfy_parser_accepts_external_vae_on_selected_decode_path() -> None:
    graph = _prompt_graph()
    graph["9"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "external.safetensors"},
    }
    graph["8"]["inputs"]["vae"] = ["9", 0]  # type: ignore[index]

    parsed = parse_comfyui_prompt_metadata(graph)

    assert parsed.candidate.vae_name == "external.safetensors"
    assert "vae" not in parsed.unresolved_fields


def test_comfy_parser_rejects_other_vae_branch_and_malformed_connection() -> None:
    graph = _prompt_graph()
    graph["9"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
    }
    parsed = parse_comfyui_prompt_metadata(graph)
    assert "vae" not in parsed.unresolved_fields

    duplicate_target = _prompt_graph()
    duplicate_target["9"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    }
    duplicate_result = parse_comfyui_prompt_metadata(duplicate_target)
    assert "vae" in duplicate_result.unresolved_fields

    malformed = _prompt_graph()
    malformed["5"]["inputs"]["model"] = ["1"]  # type: ignore[index]
    malformed_result = parse_comfyui_prompt_metadata(malformed)
    assert "checkpoint" in malformed_result.unresolved_fields


def test_comfy_parser_rejects_clip_chain_mismatch_and_model_cycle() -> None:
    mismatch = _prompt_graph()
    mismatch["6"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "style.safetensors",
            "strength_model": 0.5,
            "strength_clip": 0.5,
            "model": ["1", 0],
            "clip": ["1", 1],
        },
    }
    mismatch["5"]["inputs"]["model"] = ["6", 0]  # type: ignore[index]
    parsed_mismatch = parse_comfyui_prompt_metadata(mismatch)
    assert "clip_graph" in parsed_mismatch.unresolved_fields

    cycle = _prompt_graph()
    cycle["6"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "style.safetensors",
            "strength_model": 0.5,
            "strength_clip": 0.5,
            "model": ["7", 0],
            "clip": ["1", 1],
        },
    }
    cycle["7"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "other.safetensors",
            "strength_model": 0.5,
            "strength_clip": 0.5,
            "model": ["6", 0],
            "clip": ["1", 1],
        },
    }
    cycle["5"]["inputs"]["model"] = ["6", 0]  # type: ignore[index]
    parsed_cycle = parse_comfyui_prompt_metadata(cycle)
    assert "checkpoint" in parsed_cycle.unresolved_fields


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


def test_metadata_import_ui_reenables_parse_after_success_and_failure(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path, _capabilities())
    image_path = tmp_path / "input.png"
    image_path.write_bytes(_png(prompt=_prompt_graph()))
    handler = make_metadata_import_handler(service)

    success = handler(str(image_path), None)
    assert len(success) == 16
    assert success[0] is not None
    assert success[13].interactive is True
    assert success[14].interactive is True
    assert success[15].interactive is True

    failure = handler(None, None)
    assert len(failure) == 16
    assert failure[0] is None
    assert failure[7].interactive is False
    assert "image" in failure[8] or "metadata" in failure[8]
    assert failure[13].interactive is False
    assert failure[14].interactive is False
    assert failure[15].interactive is True


def test_metadata_source_selection_error_preserves_preview_and_parse_state(
    tmp_path: Path,
) -> None:
    service, _, _ = _service(tmp_path, _capabilities())
    sidecar = _sidecar_settings(positive_prompt="different")
    sidecar["image"] = {"sha256": "0" * 64}
    preview = service.import_image(_png(prompt=_prompt_graph()), sidecar_bytes=json.dumps(sidecar))
    handler = make_metadata_source_selection_handler(service)

    failed = handler(preview.id.hex, MetadataSourceKind.APP_SIDECAR.value, False)
    assert len(failed) == 16
    assert failed[0] == str(preview.id)
    assert "metadata_import_sidecar_hash_confirmation_required" in failed[9]
    assert failed[15].interactive is True

    selected = handler(preview.id.hex, MetadataSourceKind.APP_SIDECAR.value, True)
    assert len(selected) == 16
    assert selected[0] == str(preview.id)
    assert selected[13].interactive is True
    assert selected[14].interactive is True
    assert selected[15].interactive is True


def test_metadata_mapping_error_preserves_preview_and_parse_state(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path, _capabilities())
    preview = service.import_image(_png(prompt=_prompt_graph()))
    handler = make_metadata_mapping_handler(service)

    failed = handler(str(preview.id), "{malformed")

    assert len(failed) == 16
    assert failed[0] == str(preview.id)
    assert failed[1] is not None
    assert "metadata_import_mapping_invalid" in failed[9]
    assert failed[15].interactive is True
