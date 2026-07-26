from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image
from sqlalchemy import inspect

from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
)
from runpod_sdxl_image_studio.adapters.database.repositories.lora_metadata_repository import (
    LoraMetadataRepository,
)
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.lora_thumbnail_storage import (
    LoraThumbnailStorage,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database
from runpod_sdxl_image_studio.domain.lora_metadata import LoraMetadataUpdate
from runpod_sdxl_image_studio.domain.lora_search import (
    LoraSearchQuery,
    LoraSort,
    append_trigger_words,
)
from runpod_sdxl_image_studio.services.lora_catalog_service import LoraCatalogService


def _catalog(tmp_path: Path) -> tuple[Settings, LoraMetadataRepository, LoraThumbnailStorage]:
    database_path = tmp_path / "image-studio.sqlite3"
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_url=f"sqlite:///{database_path.as_posix()}",
    )
    upgrade_database(settings, Path(__file__).parents[2])
    engine = create_image_studio_engine(settings)
    repository = LoraMetadataRepository(create_session_factory(engine))
    thumbnails = LoraThumbnailStorage(
        tmp_path / "lora_thumbnails", settings.max_lora_thumbnail_bytes, 512
    )
    return settings, repository, thumbnails


def test_metadata_and_trigger_normalization() -> None:
    metadata = LoraMetadataUpdate(
        display_name="  Character  ",
        category="  キャラクター ",
        trigger_words="hero, blue eyes, hero, ,",
        compatible_models="SDXL 1.0, SDXL 1.0, Pony XL",
        notes="memo",
    )

    assert metadata.display_name == "Character"
    assert metadata.category == "キャラクター"
    assert metadata.trigger_words == ("hero", "blue eyes")
    assert metadata.compatible_models == ("SDXL 1.0", "Pony XL")

    assert append_trigger_words("1girl, outdoors", ("character", "blue eyes")) == (
        "1girl, outdoors, character, blue eyes"
    )
    assert append_trigger_words("1girl, outdoors, character", ("character",)) == (
        "1girl, outdoors, character"
    )


def test_repository_migration_sync_update_search_missing_and_usage(tmp_path: Path) -> None:
    _, repository, _ = _catalog(tmp_path)
    assert repository.upsert_discovered_loras(("style.safetensors", "character.safetensors"))
    assert len(repository.upsert_discovered_loras(("style.safetensors",))) == 1

    style = repository.get_by_file_name("style.safetensors")
    assert style is not None
    updated = repository.update_metadata(
        style.id,
        LoraMetadataUpdate(
            display_name="Style",
            category="画風",
            is_favorite=True,
            trigger_words=("style", "soft light"),
            recommended_model_strength=0.8,
            recommended_clip_strength=0.7,
            compatible_models=("SDXL 1.0",),
        ),
    )
    assert updated is not None and updated.is_favorite is True
    assert (
        repository.list_all(LoraSearchQuery(text="soft light"))[0].file_name == "style.safetensors"
    )
    assert (
        repository.list_all(LoraSearchQuery(favorites_only=True))[0].file_name
        == "style.safetensors"
    )
    assert repository.list_all(LoraSearchQuery(sort=LoraSort.NAME))

    repository.update_usage(("style.safetensors", "style.safetensors"), datetime.now(UTC))
    assert repository.get_by_file_name("style.safetensors").usage_count == 1  # type: ignore[union-attr]
    repository.upsert_discovered_loras(("character.safetensors",))
    missing = repository.get_by_file_name("style.safetensors")
    assert missing is not None and missing.is_missing is True
    assert (
        repository.list_all(LoraSearchQuery(include_missing=False))[0].file_name
        == "character.safetensors"
    )
    repository.upsert_discovered_loras(("style.safetensors",))
    assert repository.get_by_file_name("style.safetensors").is_missing is False  # type: ignore[union-attr]


def test_migration_creates_expected_table(tmp_path: Path) -> None:
    settings, _, _ = _catalog(tmp_path)
    engine = create_image_studio_engine(settings)
    assert inspect(engine).has_table("lora_metadata")


def test_thumbnail_storage_validates_and_atomically_writes_webp(tmp_path: Path) -> None:
    storage = LoraThumbnailStorage(tmp_path / "lora_thumbnails", 1024 * 1024, 512)
    source = BytesIO()
    Image.new("RGBA", (1024, 256), (255, 0, 0, 128)).save(source, format="PNG")
    metadata_id = UUID(int=100)

    relative = storage.save(metadata_id, source.getvalue())
    path = storage.path_for(metadata_id)
    assert relative == f"lora_thumbnails/{metadata_id}.webp"
    assert path is not None and path.suffix == ".webp"
    with Image.open(path) as saved:
        assert max(saved.size) <= 512
        assert saved.format == "WEBP"
    assert "style" not in path.name
    storage.delete(relative)
    assert storage.path_for(metadata_id) is None

    with pytest.raises(StorageError):
        storage.save(UUID(int=101), b"<svg><script>alert(1)</script></svg>")

    gif = BytesIO()
    Image.new("RGB", (4, 4), "blue").save(gif, format="GIF")
    with pytest.raises(StorageError):
        storage.save(UUID(int=102), gif.getvalue())
    with pytest.raises(StorageError):
        storage.read("lora_thumbnails/not-a-uuid.webp")


def test_catalog_service_sync_failure_does_not_mark_existing_metadata_missing(
    tmp_path: Path,
) -> None:
    _, repository, thumbnails = _catalog(tmp_path)
    repository.upsert_discovered_loras(("style.safetensors",))
    service = LoraCatalogService(repository, thumbnails)

    service.sync_with_capabilities((), capability_success=False)

    metadata = service.get_by_file_name("style.safetensors")
    assert metadata is not None and metadata.is_missing is False
