from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID

import gradio as gr
import pytest
from alembic import command
from alembic.config import Config
from PIL import Image
from sqlalchemy import inspect

from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
    resolved_database_url,
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
from runpod_sdxl_image_studio.services.lora_catalog_service import (
    LoraCatalogError,
    LoraCatalogService,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    build_lora_editor,
    component_outputs,
)
from runpod_sdxl_image_studio.ui.tabs.lora_management_tab import (
    build_catalog_list_updates,
    make_favorite_handler,
    make_save_handler,
    make_select_handler,
    metadata_save_preserve_updates,
)


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
    inspector = inspect(engine)
    assert inspector.has_table("lora_metadata")
    indexes = {index["name"] for index in inspector.get_indexes("lora_metadata")}
    assert {
        "ix_lora_metadata_category",
        "ix_lora_metadata_favorite",
        "ix_lora_metadata_missing",
        "ix_lora_metadata_last_used",
    } <= indexes
    engine.dispose()

    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option("script_location", str(Path(__file__).parents[2] / "alembic"))
    config.set_main_option("sqlalchemy.url", resolved_database_url(settings))
    command.downgrade(config, "base")
    downgraded_engine = create_image_studio_engine(settings)
    assert not inspect(downgraded_engine).has_table("lora_metadata")
    downgraded_engine.dispose()
    command.upgrade(config, "head")
    upgraded_engine = create_image_studio_engine(settings)
    assert inspect(upgraded_engine).has_table("lora_metadata")
    upgraded_engine.dispose()


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


def test_thumbnail_service_compensates_for_database_failures(tmp_path: Path) -> None:
    _, repository, thumbnails = _catalog(tmp_path)
    repository.upsert_discovered_loras(("style.safetensors", "new.safetensors"))
    service = LoraCatalogService(repository, thumbnails)
    style = service.get_by_file_name("style.safetensors")
    new = service.get_by_file_name("new.safetensors")
    assert style is not None and new is not None

    payload = BytesIO()
    Image.new("RGB", (8, 8), "red").save(payload, format="PNG")
    service.save_thumbnail(style.id, payload.getvalue())
    old_path = service.thumbnail_path(style.id)
    assert old_path is not None and old_path.exists()
    with Image.open(old_path) as old_image:
        old_pixel = old_image.getpixel((0, 0))

    def fail_thumbnail_update(metadata_id: UUID, thumbnail_path: str | None):
        del metadata_id, thumbnail_path
        raise RuntimeError("database failure")

    original_set_thumbnail_path = repository.set_thumbnail_path
    repository.set_thumbnail_path = fail_thumbnail_update  # type: ignore[method-assign]
    with pytest.raises(LoraCatalogError):
        service.save_thumbnail(new.id, payload.getvalue())
    assert service.thumbnail_path(new.id) is None

    with pytest.raises(LoraCatalogError):
        service.save_thumbnail(style.id, payload.getvalue())
    restored_path = service.thumbnail_path(style.id)
    assert restored_path is not None and restored_path.exists()
    with Image.open(restored_path) as restored:
        assert restored.getpixel((0, 0)) == old_pixel

    with pytest.raises(LoraCatalogError):
        service.delete_thumbnail(style.id)
    assert service.thumbnail_path(style.id) is not None
    repository.set_thumbnail_path = original_set_thumbnail_path  # type: ignore[method-assign]
    service.delete_thumbnail(new.id)
    assert service.thumbnail_path(new.id) is None


def test_catalog_service_sync_failure_does_not_mark_existing_metadata_missing(
    tmp_path: Path,
) -> None:
    _, repository, thumbnails = _catalog(tmp_path)
    repository.upsert_discovered_loras(("style.safetensors",))
    service = LoraCatalogService(repository, thumbnails)

    service.sync_with_capabilities((), capability_success=False)

    metadata = service.get_by_file_name("style.safetensors")
    assert metadata is not None and metadata.is_missing is False


def test_metadata_changes_refresh_catalog_and_generation_views(tmp_path: Path) -> None:
    _, repository, thumbnails = _catalog(tmp_path)
    repository.upsert_discovered_loras(("style.safetensors",))
    service = LoraCatalogService(repository, thumbnails)
    metadata = service.get_by_file_name("style.safetensors")
    assert metadata is not None

    with gr.Blocks():
        updates = build_catalog_list_updates(
            (metadata.model_copy(update={"display_name": "Style"}),),
            str(metadata.id),
            ("character", "style"),
            "style",
        )
    assert "Style" in updates[0]
    assert updates[1].value == str(metadata.id)
    assert updates[1].choices[0][0] == "Style — style.safetensors"
    assert updates[2].value == "style"

    save = make_save_handler(service, max_loras=2)
    save_result = save(
        str(metadata.id),
        "Updated Style",
        "character",
        True,
        "style",
        0.8,
        0.7,
        "SDXL 1.0",
        "memo",
        "",
        None,
        False,
        False,
        "favorites_recent",
        None,
        [],
    )
    assert len(save_result) == 7 + 7 * 2
    assert "Updated Style" in save_result[1]
    assert save_result[2].choices[0][0] == "Updated Style — style.safetensors"
    assert ("character", "character") in save_result[3].choices

    favorite = make_favorite_handler(service, max_loras=2)
    invalid = favorite("not-a-uuid", True)
    assert invalid[0].value is False
    assert "UUID" in invalid[1]


def test_metadata_save_preserve_updates_follow_editor_output_structure() -> None:
    with gr.Blocks():
        editor = build_lora_editor(2)
        updates = metadata_save_preserve_updates(lora_editor=editor)

    assert len(updates) == 6 + len(component_outputs(editor))
    assert 1 + len(updates) == 7 + 7 * 2


def _assert_skip_updates(updates: tuple[object, ...]) -> None:
    assert all(update == gr.skip() for update in updates)


@pytest.mark.parametrize("max_loras", [1, 2, 8, 12])
def test_save_handler_returns_matching_output_count_when_no_selection(
    max_loras: int,
) -> None:
    result = make_save_handler(object(), max_loras)(None, "", "", False, "", None, None, "", "")

    assert len(result) == 7 + 7 * max_loras
    assert result[0] == "LoRAを選択してください。"
    _assert_skip_updates(result[1:])


@pytest.mark.parametrize("max_loras", [1, 2, 8, 12])
def test_save_handler_returns_matching_output_count_on_validation_error(
    max_loras: int,
) -> None:
    class ValidationCatalog:
        def update_metadata(self, metadata_id: UUID, update: LoraMetadataUpdate) -> None:
            del metadata_id, update
            raise AssertionError("validation must happen before the catalog call")

    result = make_save_handler(ValidationCatalog(), max_loras)(
        str(UUID(int=1)),
        "",
        "",
        False,
        "",
        3.0,
        None,
        "",
        "",
    )

    assert len(result) == 7 + 7 * max_loras
    assert result[0] == "入力値を確認してください。"
    _assert_skip_updates(result[1:])
    assert "3.0" not in str(result)


@pytest.mark.parametrize("max_loras", [1, 2, 8, 12])
def test_save_handler_returns_matching_output_count_on_catalog_error(
    max_loras: int,
) -> None:
    class FailingCatalog:
        def update_metadata(self, metadata_id: UUID, update: LoraMetadataUpdate) -> None:
            del metadata_id, update
            raise LoraCatalogError("database details")

    result = make_save_handler(FailingCatalog(), max_loras)(
        str(UUID(int=1)),
        "",
        "",
        False,
        "",
        None,
        None,
        "",
        "",
    )

    assert len(result) == 7 + 7 * max_loras
    assert result[0] == "入力値を確認してください。"
    _assert_skip_updates(result[1:])
    assert "database details" not in str(result)


def test_select_handler_preserves_form_on_invalid_uuid() -> None:
    class UnexpectedCatalog:
        def get_metadata(self, metadata_id: UUID) -> None:
            del metadata_id
            raise AssertionError("invalid UUID must not reach the catalog")

    result = make_select_handler(UnexpectedCatalog())("not-a-uuid")

    assert len(result) == 9
    _assert_skip_updates(result)


def test_select_handler_preserves_form_on_catalog_error() -> None:
    class FailingCatalog:
        def get_metadata(self, metadata_id: UUID) -> None:
            del metadata_id
            raise LoraCatalogError("database details")

    result = make_select_handler(FailingCatalog())(str(UUID(int=1)))

    assert len(result) == 9
    _assert_skip_updates(result)
    assert "database details" not in str(result)


def test_select_handler_preserves_form_when_metadata_was_deleted() -> None:
    class EmptyCatalog:
        def get_metadata(self, metadata_id: UUID) -> None:
            del metadata_id
            return None

    result = make_select_handler(EmptyCatalog())(str(UUID(int=1)))

    assert len(result) == 9
    _assert_skip_updates(result)
