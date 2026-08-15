"""Versioned UI state for restoring the last generation form safely."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from runpod_sdxl_image_studio.domain.generation import Generation
from runpod_sdxl_image_studio.domain.generation_settings import MAX_SEED
from runpod_sdxl_image_studio.domain.generation_snapshot import LoraSettingSnapshot
from runpod_sdxl_image_studio.domain.lora import LoraSetting

CURRENT_FORM_STATE_SCHEMA_VERSION = 1


class GenerationFormStateError(ValueError):
    """Raised when a persisted form state cannot be used safely."""


class FormSeedMode(StrEnum):
    RANDOM = "random"
    FIXED = "fixed"
    PREVIOUS = "previous_seed"


_SEED_MODE_ALIASES = {
    "random": FormSeedMode.RANDOM,
    "fixed": FormSeedMode.FIXED,
    "previous": FormSeedMode.PREVIOUS,
    "previous_seed": FormSeedMode.PREVIOUS,
    "previous seed": FormSeedMode.PREVIOUS,
}


class GenerationFormStateSnapshot(BaseModel):
    """The last successfully queued values shown in the generation form.

    This is intentionally not a ``GenerationSettingsSnapshot``.  A form may
    contain an unresolved random seed and a seed mode, while an execution
    snapshot always contains the resolved seed used by ComfyUI.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=CURRENT_FORM_STATE_SCHEMA_VERSION, ge=1)
    positive_prompt: str = Field(default="", max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    seed_mode: FormSeedMode
    seed: int = Field(ge=-1, le=MAX_SEED)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    steps: int = Field(ge=1, le=150)
    cfg_scale: float = Field(ge=0.0, le=30.0)
    clip_skip: int = Field(default=1, ge=1, le=12)
    hires_fix: bool = False
    hires_scale: float = Field(default=1.5, ge=1.0, le=4.0)
    hires_resize_method: str = Field(default="lanczos", min_length=1)
    hires_steps: int = Field(default=20, ge=1, le=150)
    hires_cfg_scale: float = Field(default=5.5, ge=0.0, le=30.0)
    hires_sampler_name: str = Field(default="euler", min_length=1)
    hires_scheduler_name: str = Field(default="normal", min_length=1)
    hires_denoise: float = Field(default=0.4, ge=0.0, le=1.0)
    final_upscale: bool = False
    sampler_name: str = Field(min_length=1, max_length=200)
    scheduler_name: str = Field(min_length=1, max_length=200)
    checkpoint_name: str = Field(min_length=1, max_length=500)
    vae_name: str | None = Field(default=None, max_length=500)
    upscaler_name: str | None = Field(default=None, max_length=500)
    loras: tuple[LoraSettingSnapshot, ...] = ()
    auto_trigger_lora_names: tuple[str, ...] = ()
    updated_at: datetime

    @field_validator("seed_mode", mode="before")
    @classmethod
    def normalize_seed_mode(cls, value: object) -> FormSeedMode:
        if isinstance(value, FormSeedMode):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            try:
                return _SEED_MODE_ALIASES[normalized]
            except KeyError as exc:
                raise ValueError("seed_mode is unsupported") from exc
        raise ValueError("seed_mode is unsupported")

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @classmethod
    def from_ui(
        cls,
        *,
        positive_prompt: str | None,
        negative_prompt: str | None,
        seed_mode: str | FormSeedMode,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        scheduler_name: str,
        checkpoint_name: str,
        vae_name: str | None,
        upscaler_name: str | None = None,
        loras: tuple[LoraSetting, ...] | list[LoraSetting] = (),
        auto_trigger_lora_names: tuple[str, ...] | list[str] = (),
        clip_skip: int = 1,
        hires_fix: bool = False,
        hires_scale: float = 1.5,
        hires_resize_method: str = "lanczos",
        hires_steps: int = 20,
        hires_cfg_scale: float = 5.5,
        hires_sampler_name: str = "euler",
        hires_scheduler_name: str = "normal",
        hires_denoise: float = 0.4,
        final_upscale: bool = False,
        updated_at: datetime | None = None,
    ) -> GenerationFormStateSnapshot:
        return cls(
            positive_prompt=positive_prompt or "",
            negative_prompt=negative_prompt or "",
            seed_mode=seed_mode,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            clip_skip=clip_skip,
            hires_fix=hires_fix,
            hires_scale=hires_scale,
            hires_resize_method=hires_resize_method,
            hires_steps=hires_steps,
            hires_cfg_scale=hires_cfg_scale,
            hires_sampler_name=hires_sampler_name,
            hires_scheduler_name=hires_scheduler_name,
            hires_denoise=hires_denoise,
            final_upscale=final_upscale,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            checkpoint_name=checkpoint_name,
            vae_name=vae_name,
            upscaler_name=upscaler_name,
            loras=tuple(
                LoraSettingSnapshot(
                    name=item.name,
                    model_strength=item.model_strength,
                    clip_strength=item.clip_strength,
                    order=item.order,
                )
                for item in loras
            ),
            auto_trigger_lora_names=tuple(
                dict.fromkeys(
                    str(name).strip() for name in auto_trigger_lora_names if str(name).strip()
                )
            ),
            updated_at=updated_at or datetime.now(UTC),
        )

    @classmethod
    def from_generation(
        cls,
        generation: Generation,
        *,
        upscaler_name: str | None = None,
    ) -> GenerationFormStateSnapshot:
        """Build the safe old-DB fallback using the resolved execution seed."""

        snapshot = generation.settings_snapshot
        return cls(
            positive_prompt=snapshot.positive_prompt,
            negative_prompt=snapshot.negative_prompt,
            seed_mode=FormSeedMode.FIXED,
            seed=snapshot.seed,
            width=snapshot.width,
            height=snapshot.height,
            steps=snapshot.steps,
            cfg_scale=snapshot.cfg_scale,
            clip_skip=snapshot.clip_skip,
            hires_fix=snapshot.hires_fix,
            hires_scale=snapshot.hires_scale,
            hires_resize_method=snapshot.hires_resize_method,
            hires_steps=snapshot.hires_steps,
            hires_cfg_scale=snapshot.hires_cfg_scale,
            hires_sampler_name=snapshot.hires_sampler_name,
            hires_scheduler_name=snapshot.hires_scheduler_name,
            hires_denoise=snapshot.hires_denoise,
            final_upscale=snapshot.final_upscale,
            sampler_name=snapshot.sampler_name,
            scheduler_name=snapshot.scheduler_name,
            checkpoint_name=snapshot.checkpoint_name,
            vae_name=snapshot.vae_name,
            upscaler_name=upscaler_name,
            loras=snapshot.loras,
            auto_trigger_lora_names=(),
            updated_at=generation.updated_at,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> GenerationFormStateSnapshot:
        try:
            parsed: Any = json.loads(payload)
            if not isinstance(parsed, dict):
                raise GenerationFormStateError("form state JSON must be an object")
            if parsed.get("schema_version") != CURRENT_FORM_STATE_SCHEMA_VERSION:
                raise GenerationFormStateError("unsupported form state schema version")
            return cls.model_validate(parsed)
        except GenerationFormStateError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise GenerationFormStateError("form state JSON is invalid") from exc

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @property
    def ui_seed_mode(self) -> str:
        return {
            FormSeedMode.RANDOM: "Random",
            FormSeedMode.FIXED: "Fixed",
            FormSeedMode.PREVIOUS: "Previous seed",
        }[self.seed_mode]

    @property
    def lora_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.loras)


__all__ = [
    "CURRENT_FORM_STATE_SCHEMA_VERSION",
    "FormSeedMode",
    "GenerationFormStateError",
    "GenerationFormStateSnapshot",
    "LastGenerationFormState",
]

# Specification-friendly alias for callers that use the shorter name.
LastGenerationFormState = GenerationFormStateSnapshot
