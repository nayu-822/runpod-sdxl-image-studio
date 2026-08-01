"""Typed, versioned payloads used by Phase 3B presets."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.lora import LoraSetting

CURRENT_PRESET_SCHEMA_VERSION = 1


class PresetPayloadError(ValueError):
    """Preset payload cannot be safely validated or upgraded."""


class PresetKind(StrEnum):
    """保存できるPresetの種類。"""

    GENERATION = "generation"
    PROMPT = "prompt"
    LORA = "lora"


class SeedMode(StrEnum):
    """Generation Presetでのseedの扱い。"""

    RANDOM = "random"
    FIXED = "fixed"
    PREVIOUS = "previous"


class PromptApplyMode(StrEnum):
    """Promptを現在値へ適用する方法。"""

    REPLACE = "replace"
    APPEND = "append"
    PREPEND = "prepend"


class PresetPayload(BaseModel):
    """全Payloadが共有するschema version。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=CURRENT_PRESET_SCHEMA_VERSION, ge=1)


class GenerationPresetPayload(PresetPayload):
    """GenerationSettingsを再現するPayload。"""

    checkpoint_name: str = Field(min_length=1)
    vae_name: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    steps: int = Field(ge=1, le=150)
    cfg_scale: float = Field(ge=0, le=30)
    sampler_name: str = Field(min_length=1)
    scheduler_name: str = Field(min_length=1)
    seed_mode: SeedMode = SeedMode.RANDOM
    fixed_seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    positive_prompt: str = Field(default="", max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    loras: tuple[LoraSetting, ...] = ()

    @classmethod
    def from_settings(cls, settings: GenerationSettings) -> GenerationPresetPayload:
        mode = SeedMode.FIXED if settings.seed >= 0 else SeedMode.RANDOM
        return cls(
            checkpoint_name=settings.checkpoint_name,
            vae_name=settings.vae_name,
            width=settings.width,
            height=settings.height,
            steps=settings.steps,
            cfg_scale=settings.cfg_scale,
            sampler_name=settings.sampler_name,
            scheduler_name=settings.scheduler_name,
            seed_mode=mode,
            fixed_seed=settings.seed if settings.seed >= 0 else None,
            positive_prompt=settings.positive_prompt,
            negative_prompt=settings.negative_prompt,
            loras=settings.loras,
        )

    def to_settings(self, *, previous_seed: int | None = None) -> GenerationSettings:
        if self.seed_mode is SeedMode.FIXED:
            if self.fixed_seed is None:
                raise PresetPayloadError("fixed seed is required")
            seed = self.fixed_seed
        elif self.seed_mode is SeedMode.PREVIOUS:
            if previous_seed is None:
                raise PresetPayloadError("previous seed is unavailable")
            seed = previous_seed
        else:
            seed = -1
        return GenerationSettings(
            positive_prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            seed=seed,
            width=self.width,
            height=self.height,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler_name,
            scheduler_name=self.scheduler_name,
            checkpoint_name=self.checkpoint_name,
            vae_name=self.vae_name,
            loras=self.loras,
        )


class PromptPresetPayload(PresetPayload):
    """Positive/Negative promptを保存するPayload。"""

    positive_prompt: str = Field(default="", max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    positive_mode: PromptApplyMode = PromptApplyMode.REPLACE
    negative_mode: PromptApplyMode = PromptApplyMode.REPLACE


class LoraPresetPayload(PresetPayload):
    """順序付きLoRA設定を保存するPayload。"""

    loras: tuple[LoraSetting, ...] = ()


PresetPayloadValue: TypeAlias = GenerationPresetPayload | PromptPresetPayload | LoraPresetPayload


def payload_for_kind(kind: PresetKind, payload: PresetPayloadValue) -> PresetPayloadValue:
    """Validate that a Payload subtype matches its Preset kind."""

    expected = {
        PresetKind.GENERATION: GenerationPresetPayload,
        PresetKind.PROMPT: PromptPresetPayload,
        PresetKind.LORA: LoraPresetPayload,
    }[kind]
    if not isinstance(payload, expected):
        raise PresetPayloadError("preset kind and payload type do not match")
    if payload.schema_version != CURRENT_PRESET_SCHEMA_VERSION:
        raise PresetPayloadError("unsupported preset schema version")
    return payload


def parse_payload(kind: PresetKind, payload_json: str) -> PresetPayloadValue:
    """Parse a DB JSON payload without modifying the original string."""

    try:
        value: Any = json.loads(payload_json)
        if not isinstance(value, dict):
            raise PresetPayloadError("preset payload must be a JSON object")
        if kind is PresetKind.GENERATION:
            return payload_for_kind(kind, GenerationPresetPayload.model_validate(value))
        if kind is PresetKind.PROMPT:
            return payload_for_kind(kind, PromptPresetPayload.model_validate(value))
        return payload_for_kind(kind, LoraPresetPayload.model_validate(value))
    except PresetPayloadError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise PresetPayloadError("preset payload JSON is invalid") from exc


def upgrade_payload(kind: PresetKind, payload: dict[str, Any]) -> PresetPayloadValue:
    """Upgrade entry point; version 1 currently needs no transformation."""

    version = payload.get("schema_version")
    if version != CURRENT_PRESET_SCHEMA_VERSION:
        raise PresetPayloadError("unsupported preset schema version")
    return parse_payload(kind, json.dumps(payload, ensure_ascii=False))
