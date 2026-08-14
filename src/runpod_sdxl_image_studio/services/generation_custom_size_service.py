"""Application service for safe persisted custom-size preferences."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from contextlib import nullcontext
from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.generation_custom_size_repository import (  # noqa: E501
    GenerationCustomSizeRepositoryError,
    GenerationCustomSizeRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation_custom_size import GenerationCustomSize


class GenerationCustomSizeError(ValueError):
    """A safe validation or persistence error for custom sizes."""


class GenerationCustomSizeService:
    """Validate dimensions before delegating idempotent persistence."""

    def __init__(
        self,
        repository: GenerationCustomSizeRepositoryProtocol,
        settings: Settings,
        *,
        state_changed_callback: Callable[[], None] | None = None,
        work_gate: object | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._state_changed_callback = state_changed_callback
        self._work_gate = work_gate

    def list(self) -> tuple[GenerationCustomSize, ...]:
        try:
            return self._repository.list()
        except GenerationCustomSizeRepositoryError as exc:
            raise GenerationCustomSizeError("保存済みサイズを取得できませんでした。") from exc

    def add(self, width: int, height: int) -> GenerationCustomSize:
        normalized_width, normalized_height = self._validate(width, height)
        try:
            context = (
                self._work_gate.admit_persistent_mutation()  # type: ignore[attr-defined]
                if self._work_gate is not None
                else nullcontext()
            )
            with context:
                result = self._repository.add(normalized_width, normalized_height)
                self._notify_changed()
                return result
        except GenerationCustomSizeRepositoryError as exc:
            raise GenerationCustomSizeError("保存済みサイズを登録できませんでした。") from exc

    def delete(self, size_id: UUID | str) -> None:
        try:
            identifier = UUID(str(size_id))
        except (TypeError, ValueError) as exc:
            raise GenerationCustomSizeError("削除する保存済みサイズが不正です。") from exc
        try:
            context = (
                self._work_gate.admit_persistent_mutation()  # type: ignore[attr-defined]
                if self._work_gate is not None
                else nullcontext()
            )
            with context:
                self._repository.delete(identifier)
                self._notify_changed()
        except GenerationCustomSizeRepositoryError as exc:
            raise GenerationCustomSizeError("保存済みサイズを削除できませんでした。") from exc

    def selector_options(self) -> builtins.list[tuple[str, str]]:
        return [(item.label, item.selector_value) for item in self.list()]

    def resolve(self, value: str | None) -> GenerationCustomSize | None:
        if not value or not value.startswith("custom:"):
            return None
        try:
            identifier = UUID(value.removeprefix("custom:"))
        except ValueError as exc:
            raise GenerationCustomSizeError("保存済みサイズが不正です。") from exc
        item = next((candidate for candidate in self.list() if candidate.id == identifier), None)
        if item is None:
            raise GenerationCustomSizeError("保存済みサイズが見つかりません。")
        return item

    def _validate(self, width: int, height: int) -> tuple[int, int]:
        if isinstance(width, bool) or isinstance(height, bool):
            raise GenerationCustomSizeError("幅と高さは整数で指定してください。")
        try:
            normalized_width = int(width)
            normalized_height = int(height)
        except (TypeError, ValueError) as exc:
            raise GenerationCustomSizeError("幅と高さは整数で指定してください。") from exc
        if normalized_width != width or normalized_height != height:
            raise GenerationCustomSizeError("幅と高さは整数で指定してください。")
        if normalized_width <= 0 or normalized_height <= 0:
            raise GenerationCustomSizeError("幅と高さは正の値で指定してください。")
        if normalized_width % 64 or normalized_height % 64:
            raise GenerationCustomSizeError("幅と高さは64の倍数で指定してください。")
        if (
            normalized_width > self._settings.max_width
            or normalized_height > self._settings.max_height
        ):
            raise GenerationCustomSizeError("幅または高さが上限を超えています。")
        if normalized_width * normalized_height > self._settings.max_pixels:
            raise GenerationCustomSizeError("総ピクセル数が上限を超えています。")
        return normalized_width, normalized_height

    def _notify_changed(self) -> None:
        if self._state_changed_callback is not None:
            try:
                self._state_changed_callback()
            except Exception:  # noqa: BLE001 - backup notification is best effort
                return


__all__ = ["GenerationCustomSizeError", "GenerationCustomSizeService"]
