"""Application service for local LoRA metadata and usage information."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.lora_metadata_repository import (
    LoraMetadataRepository,
)
from runpod_sdxl_image_studio.adapters.storage.exceptions import StorageError
from runpod_sdxl_image_studio.adapters.storage.lora_thumbnail_storage import LoraThumbnailStorage
from runpod_sdxl_image_studio.domain.lora_metadata import LoraMetadata, LoraMetadataUpdate
from runpod_sdxl_image_studio.domain.lora_search import LoraSearchQuery, LoraSort

logger = logging.getLogger(__name__)


class LoraCatalogError(Exception):
    """Safe application-level catalog error."""


class LoraCatalogService:
    """Coordinate repository validation and thumbnail compensation."""

    def __init__(
        self,
        repository: LoraMetadataRepository,
        thumbnail_storage: LoraThumbnailStorage,
    ) -> None:
        self._repository = repository
        self._thumbnails = thumbnail_storage

    def sync_with_capabilities(
        self,
        file_names: Iterable[str],
        *,
        capability_success: bool = True,
    ) -> tuple[LoraMetadata, ...]:
        if not capability_success:
            return self._repository.list_all(LoraSearchQuery(include_missing=True))
        try:
            return self._repository.upsert_discovered_loras(file_names)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRA一覧を同期できませんでした。") from exc

    def search(self, query: LoraSearchQuery | None = None) -> tuple[LoraMetadata, ...]:
        try:
            return self._repository.list_all(query)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRA一覧を取得できませんでした。") from exc

    def categories(self) -> tuple[str, ...]:
        try:
            return self._repository.list_categories()
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRAカテゴリを取得できませんでした。") from exc

    def get_metadata(self, metadata_id: UUID) -> LoraMetadata | None:
        try:
            return self._repository.get_by_id(metadata_id)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRA metadataを取得できませんでした。") from exc

    def get_by_file_name(self, file_name: str) -> LoraMetadata | None:
        try:
            return self._repository.get_by_file_name(file_name)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRA metadataを取得できませんでした。") from exc

    def update_metadata(self, metadata_id: UUID, update: LoraMetadataUpdate) -> LoraMetadata:
        try:
            metadata = self._repository.update_metadata(metadata_id, update)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRA metadataを更新できませんでした。") from exc
        if metadata is None:
            raise LoraCatalogError("対象のLoRA metadataが見つかりません。")
        return metadata

    def set_favorite(self, metadata_id: UUID, is_favorite: bool) -> LoraMetadata:
        try:
            metadata = self._repository.set_favorite(metadata_id, is_favorite)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("お気に入りを更新できませんでした。") from exc
        if metadata is None:
            raise LoraCatalogError("対象のLoRA metadataが見つかりません。")
        return metadata

    def save_thumbnail(self, metadata_id: UUID, payload: bytes) -> LoraMetadata:
        try:
            current = self._repository.get_by_id(metadata_id)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRA metadataを取得できませんでした。") from exc
        if current is None:
            raise LoraCatalogError("対象のLoRA metadataが見つかりません。")
        try:
            old_payload = self._thumbnails.read(current.thumbnail_path)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("既存サムネイルを読み込めませんでした。") from exc
        try:
            relative_path = self._thumbnails.save(metadata_id, payload)
            updated = self._repository.set_thumbnail_path(metadata_id, relative_path)
            if updated is None:
                raise LoraCatalogError("対象のLoRA metadataが見つかりません。")
            return updated
        except Exception as exc:  # noqa: BLE001 - compensate DB/filesystem boundary
            try:
                if old_payload is not None:
                    self._thumbnails.save(metadata_id, old_payload)
                else:
                    self._thumbnails.delete(f"lora_thumbnails/{metadata_id}.webp")
            except Exception:  # noqa: BLE001 - preserve the original safe error
                logger.warning("Could not restore the previous LoRA thumbnail", exc_info=True)
            if isinstance(exc, LoraCatalogError):
                raise
            if isinstance(exc, StorageError):
                raise LoraCatalogError("サムネイルを保存できませんでした。") from exc
            raise LoraCatalogError("LoRA metadataを更新できませんでした。") from exc

    def delete_thumbnail(self, metadata_id: UUID) -> LoraMetadata:
        try:
            current = self._repository.get_by_id(metadata_id)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("LoRA metadataを取得できませんでした。") from exc
        if current is None:
            raise LoraCatalogError("対象のLoRA metadataが見つかりません。")
        try:
            old_payload = self._thumbnails.read(current.thumbnail_path)
        except Exception as exc:  # noqa: BLE001 - service boundary
            raise LoraCatalogError("既存サムネイルを読み込めませんでした。") from exc
        try:
            self._thumbnails.delete(current.thumbnail_path)
            updated = self._repository.set_thumbnail_path(metadata_id, None)
            if updated is None:
                raise LoraCatalogError("対象のLoRA metadataが見つかりません。")
            return updated
        except Exception as exc:  # noqa: BLE001 - compensate DB/filesystem boundary
            if old_payload is not None:
                try:
                    self._thumbnails.save(metadata_id, old_payload)
                except Exception:  # noqa: BLE001
                    logger.warning("Could not restore the deleted LoRA thumbnail", exc_info=True)
            raise LoraCatalogError("サムネイルを削除できませんでした。") from exc

    def thumbnail_path(self, metadata_id: UUID) -> Path | None:
        return self._thumbnails.path_for(metadata_id)

    def record_usage(self, file_names: Iterable[str], completed_at: datetime | None = None) -> None:
        names = tuple(dict.fromkeys(file_names))
        if not names:
            return
        try:
            self._repository.update_usage(names, completed_at or datetime.now(UTC))
        except Exception:  # noqa: BLE001 - usage is explicitly best effort
            logger.warning("LoRA usage statistics update failed", exc_info=True)

    def selector_options(self, category: str | None = None) -> tuple[tuple[str, str], ...]:
        query = LoraSearchQuery(category=category or None, include_missing=False)
        metadata = self.search(query)
        return tuple(
            (
                f"{item.display_name} — {item.file_name}" if item.display_name else item.file_name,
                item.file_name,
            )
            for item in metadata
        )

    def metadata_for_files(self, file_names: Iterable[str]) -> tuple[LoraMetadata | None, ...]:
        return tuple(self._repository.get_by_file_name(name) for name in file_names)

    def list_sort_values(self) -> tuple[tuple[str, str], ...]:
        return (
            ("お気に入り・最近使用", LoraSort.FAVORITES_RECENT.value),
            ("最近使用", LoraSort.RECENT.value),
            ("利用回数", LoraSort.USAGE.value),
            ("名前", LoraSort.NAME.value),
        )
