"""Application service for preset validation, application, and usage."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.preset_repository import (
    PresetRepositoryError,
    PresetRepositoryProtocol,
)
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.preset import Preset
from runpod_sdxl_image_studio.domain.preset_payload import (
    GenerationPresetPayload,
    LoraPresetPayload,
    PresetKind,
    PresetPayloadError,
    PresetPayloadValue,
    PromptApplyMode,
    PromptPresetPayload,
)


class PresetServiceError(RuntimeError):
    """UIへ返せるPresetエラー。"""


@dataclass(frozen=True)
class PresetApplyResult:
    """Preset適用後の設定と検証警告。"""

    preset: Preset
    settings: GenerationSettings
    warnings: tuple[str, ...] = ()


class PresetService:
    """Presetを保存・検索・適用するApplication Service。"""

    def __init__(
        self,
        repository: PresetRepositoryProtocol,
        settings: Settings | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()

    def create_from_current_settings(
        self,
        name: str,
        settings: GenerationSettings,
        *,
        description: str | None = None,
        favorite: bool = False,
    ) -> Preset:
        return self._create(
            PresetKind.GENERATION,
            name,
            GenerationPresetPayload.from_settings(settings),
            description,
            favorite,
        )

    def create_prompt_preset(
        self,
        name: str,
        positive_prompt: str,
        negative_prompt: str,
        *,
        description: str | None = None,
        favorite: bool = False,
        positive_mode: PromptApplyMode = PromptApplyMode.REPLACE,
        negative_mode: PromptApplyMode = PromptApplyMode.REPLACE,
    ) -> Preset:
        payload = PromptPresetPayload(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            positive_mode=positive_mode,
            negative_mode=negative_mode,
        )
        return self._create(PresetKind.PROMPT, name, payload, description, favorite)

    def create_lora_preset(
        self,
        name: str,
        loras: tuple[LoraSetting, ...],
        *,
        description: str | None = None,
        favorite: bool = False,
    ) -> Preset:
        return self._create(
            PresetKind.LORA, name, LoraPresetPayload(loras=loras), description, favorite
        )

    def update(self, preset: Preset) -> Preset:
        try:
            return self._repository.update(preset)
        except (PresetRepositoryError, PresetPayloadError, ValueError) as exc:
            raise PresetServiceError("Presetを更新できませんでした。") from exc

    def duplicate(self, preset_id: UUID, name: str | None = None) -> Preset:
        source = self._get(preset_id)
        requested = name.strip() if name else f"{source.name} (copy)"
        if name is None:
            existing = {item.name for item in self._repository.list(kind=source.kind, limit=100)}
            suffix = 2
            candidate = requested
            while candidate in existing:
                candidate = f"{requested} {suffix}"
                suffix += 1
            requested = candidate
        return self._create(
            source.kind, requested, source.payload, source.description, source.favorite
        )

    def delete(self, preset_id: UUID) -> None:
        try:
            self._repository.delete(preset_id)
        except PresetRepositoryError as exc:
            raise PresetServiceError("Presetを削除できませんでした。") from exc

    def search(
        self,
        text: str | None = None,
        *,
        kind: PresetKind | None = None,
        favorite_only: bool = False,
    ) -> tuple[Preset, ...]:
        try:
            return self._repository.search(text, kind=kind, favorite_only=favorite_only, limit=100)
        except PresetRepositoryError as exc:
            raise PresetServiceError("Presetを検索できませんでした。") from exc

    def set_favorite(self, preset_id: UUID, favorite: bool) -> Preset:
        try:
            return self._repository.set_favorite(preset_id, favorite)
        except PresetRepositoryError as exc:
            raise PresetServiceError("Presetのお気に入りを保存できませんでした。") from exc

    def apply(
        self,
        preset_id: UUID,
        *,
        current_settings: GenerationSettings | None = None,
        available_checkpoints: tuple[str, ...] | None = None,
        available_vaes: tuple[str, ...] | None = None,
        available_loras: tuple[str, ...] | None = None,
        max_loras: int | None = None,
        prompt_mode: PromptApplyMode | None = None,
        lora_mode: str = "replace",
    ) -> PresetApplyResult:
        preset = self._get(preset_id)
        warnings: list[str] = []
        try:
            if preset.kind is PresetKind.GENERATION:
                payload = preset.payload
                assert isinstance(payload, GenerationPresetPayload)
                settings = payload.to_settings()
                if (
                    available_checkpoints is not None
                    and settings.checkpoint_name not in available_checkpoints
                ):
                    warnings.append(f"checkpointが利用できません: {settings.checkpoint_name}")
                if (
                    settings.vae_name
                    and available_vaes is not None
                    and settings.vae_name not in available_vaes
                ):
                    warnings.append(f"VAEが利用できません: {settings.vae_name}")
                warnings.extend(_missing_loras(settings.loras, available_loras))
            elif preset.kind is PresetKind.PROMPT:
                if current_settings is None:
                    raise PresetServiceError("Prompt Presetの適用には現在の設定が必要です。")
                payload = preset.payload
                assert isinstance(payload, PromptPresetPayload)
                positive = _apply_text(
                    current_settings.positive_prompt,
                    payload.positive_prompt,
                    prompt_mode or payload.positive_mode,
                )
                negative = _apply_text(
                    current_settings.negative_prompt,
                    payload.negative_prompt,
                    prompt_mode or payload.negative_mode,
                )
                settings = current_settings.model_copy(
                    update={"positive_prompt": positive, "negative_prompt": negative}
                )
            else:
                if current_settings is None:
                    raise PresetServiceError("LoRA Presetの適用には現在の設定が必要です。")
                payload = preset.payload
                assert isinstance(payload, LoraPresetPayload)
                if lora_mode not in {"replace", "append"}:
                    raise PresetServiceError("LoRA Presetの適用方式が不正です。")
                loras = (
                    payload.loras
                    if lora_mode == "replace"
                    else current_settings.loras + payload.loras
                )
                if len({lora.name for lora in loras}) != len(loras):
                    raise PresetServiceError("同じLoRAを重複して適用できません。")
                if max_loras is not None and len(loras) > max_loras:
                    raise PresetServiceError("LoRA数が上限を超えています。")
                warnings.extend(_missing_loras(loras, available_loras))
                settings = current_settings.model_copy(update={"loras": loras})
            with suppress(PresetRepositoryError):
                self._repository.record_usage(preset.id)
            return PresetApplyResult(preset=preset, settings=settings, warnings=tuple(warnings))
        except PresetServiceError:
            raise
        except (PresetPayloadError, ValueError) as exc:
            raise PresetServiceError("Presetを適用できませんでした。") from exc

    def _create(
        self,
        kind: PresetKind,
        name: str,
        payload: PresetPayloadValue,
        description: str | None,
        favorite: bool,
    ) -> Preset:
        try:
            return self._repository.create(
                Preset.create(kind, name, payload, description=description, favorite=favorite)
            )
        except (PresetRepositoryError, PresetPayloadError, ValueError) as exc:
            raise PresetServiceError("Presetを保存できませんでした。") from exc

    def _get(self, preset_id: UUID) -> Preset:
        try:
            preset = self._repository.get_by_id(preset_id)
        except PresetRepositoryError as exc:
            raise PresetServiceError("Presetを取得できませんでした。") from exc
        if preset is None:
            raise PresetServiceError("Presetが見つかりません。")
        return preset


def _apply_text(current: str, value: str, mode: PromptApplyMode) -> str:
    if mode is PromptApplyMode.REPLACE:
        return value
    if mode is PromptApplyMode.PREPEND:
        return f"{value}, {current}" if current else value
    return f"{current}, {value}" if current else value


def _missing_loras(loras: tuple[LoraSetting, ...], available: tuple[str, ...] | None) -> list[str]:
    if available is None:
        return []
    return [f"LoRAが利用できません: {lora.name}" for lora in loras if lora.name not in available]
