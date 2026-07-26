"""Application service for safe, paged generation history operations."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationRepositoryError,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import Generation
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationDetailView,
    GenerationHistoryFilter,
    GenerationHistoryItem,
    GenerationHistoryPage,
    RegenerationPlan,
    RestoreSettingsResult,
)


class GenerationHistoryError(RuntimeError):
    """Safe history service error."""


class GenerationHistoryService:
    def __init__(
        self,
        generation_repository: GenerationRepositoryProtocol,
        artifact_repository: GenerationArtifactRepositoryProtocol,
        settings: Settings | None = None,
    ) -> None:
        app_settings = settings or get_settings()
        self._generation_repository = generation_repository
        self._artifact_repository = artifact_repository
        self._data_dir = app_settings.data_dir.resolve()
        self._timezone = ZoneInfo(app_settings.timezone)
        self._page_size = app_settings.history_page_size

    @property
    def page_size(self) -> int:
        return self._page_size

    def to_item(self, generation: Generation) -> GenerationHistoryItem:
        return self._item(generation)

    def absolute_data_path(self, relative_path: str | None) -> Path | None:
        if relative_path is None:
            return None
        normalized = relative_path.replace("\\", "/")
        path = (self._data_dir / normalized).resolve()
        try:
            path.relative_to(self._data_dir)
        except ValueError:
            return None
        return path if path.exists() else None

    def list_history(self, query: GenerationHistoryFilter | None = None) -> GenerationHistoryPage:
        requested = query or GenerationHistoryFilter(limit=self._page_size)
        normalized = self._with_date_range(requested)
        try:
            page = self._generation_repository.list_history(normalized)
            return page
        except GenerationRepositoryError as exc:
            raise GenerationHistoryError("履歴を取得できませんでした。") from exc

    def list_items(
        self, query: GenerationHistoryFilter | None = None
    ) -> tuple[GenerationHistoryItem, ...]:
        page = self.list_history(query)
        return tuple(self._item(generation) for generation in page.generations)

    def get_detail(self, generation_id: UUID) -> GenerationDetailView:
        generation = self._get_generation(generation_id)
        try:
            artifacts = self._artifact_repository.list_by_generation(generation_id)
        except GenerationRepositoryError as exc:
            raise GenerationHistoryError("履歴のartifactを取得できませんでした。") from exc
        image = next((item for item in artifacts if item.artifact_type.value == "image"), None)
        thumbnail = next(
            (item for item in artifacts if item.artifact_type.value == "thumbnail"), None
        )
        return GenerationDetailView(
            generation_id=str(generation.id),
            image_path=self._safe_relative_existing(image.local_path) if image else None,
            thumbnail_path=self._safe_relative_existing(thumbnail.local_path)
            if thumbnail
            else None,
            kind_text=generation.kind.value,
            status_text=generation.status.value,
            parent_generation_id=(
                str(generation.parent_generation_id)
                if generation.parent_generation_id is not None
                else None
            ),
            created_at_text=self._display_time(generation.created_at) or "",
            started_at_text=self._display_time(generation.started_at),
            completed_at_text=self._display_time(generation.completed_at),
            snapshot=generation.settings_snapshot,
            comfy_prompt_id=generation.comfy_prompt_id,
            image_sha256=image.sha256 if image else None,
            image_size_bytes=image.size_bytes if image else None,
            favorite=generation.favorite,
            user_note=generation.user_note,
            error_summary=generation.error_summary,
            restore_warnings=(),
        )

    def restore_settings(
        self,
        generation_id: UUID,
        *,
        checkpoints: tuple[str, ...] = (),
        vaes: tuple[str, ...] = (),
        loras: tuple[str, ...] = (),
        max_loras: int | None = None,
    ) -> RestoreSettingsResult:
        generation = self._get_generation(generation_id)
        settings = generation.settings_snapshot.to_generation_settings()
        warnings: list[str] = []
        if checkpoints and settings.checkpoint_name not in checkpoints:
            warnings.append(f"checkpointが現在利用できません: {settings.checkpoint_name}")
        if settings.vae_name is not None and vaes and settings.vae_name not in vaes:
            warnings.append(f"VAEが現在利用できません: {settings.vae_name}")
        missing_loras = [lora.name for lora in settings.loras if lora.name not in loras]
        warnings.extend(f"LoRAが現在利用できません: {name}" for name in missing_loras)
        if max_loras is not None and len(settings.loras) > max_loras:
            warnings.append("保存時のLoRA数が現在の上限を超えています。")
        return RestoreSettingsResult(
            settings=settings, warnings=tuple(warnings), parent_generation_id=generation.id
        )

    def prepare_regeneration(self, generation_id: UUID) -> RegenerationPlan:
        restored = self.restore_settings(generation_id)
        if restored.warnings:
            raise GenerationHistoryError("利用できないモデルがあるため再生成できません。")
        return RegenerationPlan(
            settings=restored.settings,
            parent_generation_id=generation_id,
        )

    def set_favorite(self, generation_id: UUID, favorite: bool) -> GenerationDetailView:
        try:
            self._generation_repository.set_favorite(generation_id, favorite)
            return self.get_detail(generation_id)
        except (GenerationRepositoryError, ValueError) as exc:
            raise GenerationHistoryError("お気に入りを保存できませんでした。") from exc

    def update_note(self, generation_id: UUID, note: str | None) -> GenerationDetailView:
        try:
            self._generation_repository.update_note(generation_id, note)
            return self.get_detail(generation_id)
        except (GenerationRepositoryError, ValueError) as exc:
            raise GenerationHistoryError("メモを保存できませんでした。") from exc

    def _get_generation(self, generation_id: UUID) -> Generation:
        try:
            generation = self._generation_repository.get_by_id(generation_id)
        except (GenerationRepositoryError, ValueError) as exc:
            raise GenerationHistoryError("履歴を取得できませんでした。") from exc
        if generation is None:
            raise GenerationHistoryError("履歴が見つかりません。")
        return generation

    def _item(self, generation: Generation) -> GenerationHistoryItem:
        try:
            artifacts = self._artifact_repository.list_by_generation(generation.id)
        except GenerationRepositoryError as exc:
            raise GenerationHistoryError("履歴のartifactを取得できませんでした。") from exc
        thumbnail = next(
            (item for item in artifacts if item.artifact_type.value == "thumbnail"), None
        )
        return GenerationHistoryItem(
            generation_id=str(generation.id),
            created_at_text=self._display_time(generation.created_at) or "",
            status_text=generation.status.value,
            checkpoint_label=generation.settings_snapshot.checkpoint_name,
            lora_labels=tuple(lora.name for lora in generation.settings_snapshot.loras),
            seed_text=str(generation.settings_snapshot.seed),
            resolution_text=(
                f"{generation.settings_snapshot.width} × {generation.settings_snapshot.height}"
            ),
            thumbnail_path=(
                self._safe_relative_existing(thumbnail.local_path) if thumbnail else None
            ),
            favorite=generation.favorite,
            kind_text=generation.kind.value,
            error_summary=generation.error_summary,
        )

    def _with_date_range(self, query: GenerationHistoryFilter) -> GenerationHistoryFilter:
        if query.date is None or query.start_utc is not None or query.end_utc is not None:
            return query
        start_local = datetime.combine(query.date, time.min, self._timezone)
        end_local = start_local + timedelta(days=1)
        return GenerationHistoryFilter(
            date=query.date,
            status=query.status,
            favorite=query.favorite,
            kind=query.kind,
            offset=query.offset,
            limit=query.limit,
            start_utc=start_local.astimezone(UTC),
            end_utc=end_local.astimezone(UTC),
        )

    def _safe_relative_existing(self, value: str) -> str | None:
        normalized = value.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in normalized.split("/")):
            return None
        resolved = (self._data_dir / path).resolve()
        try:
            resolved.relative_to(self._data_dir)
        except ValueError:
            return None
        return normalized if resolved.exists() else None

    def _display_time(self, value: datetime | None) -> str | None:
        return value.astimezone(self._timezone).strftime("%Y-%m-%d %H:%M:%S") if value else None
