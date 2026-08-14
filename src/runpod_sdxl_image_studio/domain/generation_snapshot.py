"""Immutable, versioned generation settings snapshots."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runpod_sdxl_image_studio.domain.generation_settings import MAX_SEED, GenerationSettings

CURRENT_SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotError(ValueError):
    """Raised when a stored snapshot cannot safely be used."""


class LoraSettingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    model_strength: float = Field(ge=-2.0, le=2.0)
    clip_strength: float = Field(ge=-2.0, le=2.0)
    order: int = Field(ge=0)


class GenerationSettingsSnapshot(BaseModel):
    """The complete resolved input used for one generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=CURRENT_SNAPSHOT_SCHEMA_VERSION, ge=1)
    positive_prompt: str = Field(default="", max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    seed: int = Field(ge=0, le=MAX_SEED)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    steps: int = Field(ge=1, le=150)
    cfg_scale: float = Field(ge=0.0, le=30.0)
    batch_size: int = Field(default=1, ge=1, le=4)
    clip_skip: int = Field(default=1, ge=1, le=12)
    hires_fix: bool = False
    hires_scale: float = Field(default=1.5, ge=1.0, le=4.0)
    hires_denoise: float = Field(default=0.4, ge=0.0, le=1.0)
    final_upscale: bool = False
    final_upscale_model: str | None = None
    client_local_date: str | None = None
    sampler_name: str = Field(min_length=1)
    scheduler_name: str = Field(min_length=1)
    checkpoint_name: str = Field(min_length=1)
    vae_name: str | None = None
    loras: tuple[LoraSettingSnapshot, ...] = ()
    workflow_template_id: str = Field(min_length=1)
    workflow_template_version: str = Field(min_length=1)

    @classmethod
    def from_settings(cls, settings: GenerationSettings) -> GenerationSettingsSnapshot:
        if settings.seed < 0:
            raise SnapshotError("snapshot requires a resolved seed")
        return cls(
            positive_prompt=settings.positive_prompt,
            negative_prompt=settings.negative_prompt,
            seed=settings.seed,
            width=settings.width,
            height=settings.height,
            steps=settings.steps,
            cfg_scale=settings.cfg_scale,
            batch_size=settings.batch_size,
            clip_skip=settings.clip_skip,
            hires_fix=settings.hires_fix,
            hires_scale=settings.hires_scale,
            hires_denoise=settings.hires_denoise,
            final_upscale=settings.final_upscale,
            final_upscale_model=settings.final_upscale_model,
            client_local_date=settings.client_local_date,
            sampler_name=settings.sampler_name,
            scheduler_name=settings.scheduler_name,
            checkpoint_name=settings.checkpoint_name,
            vae_name=settings.vae_name,
            loras=tuple(
                LoraSettingSnapshot(
                    name=lora.name,
                    model_strength=lora.model_strength,
                    clip_strength=lora.clip_strength,
                    order=lora.order,
                )
                for lora in settings.loras
            ),
            workflow_template_id=settings.workflow_template_id,
            workflow_template_version=settings.workflow_template_version,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> GenerationSettingsSnapshot:
        try:
            parsed: Any = json.loads(payload)
            if not isinstance(parsed, dict):
                raise SnapshotError("snapshot JSON must be an object")
            if parsed.get("schema_version") != CURRENT_SNAPSHOT_SCHEMA_VERSION:
                raise SnapshotError("unsupported snapshot schema version")
            return cls.model_validate(parsed)
        except SnapshotError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise SnapshotError("snapshot JSON is invalid") from exc

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_generation_settings(self) -> GenerationSettings:
        try:
            return GenerationSettings(
                positive_prompt=self.positive_prompt,
                negative_prompt=self.negative_prompt,
                seed=self.seed,
                width=self.width,
                height=self.height,
                steps=self.steps,
                cfg_scale=self.cfg_scale,
                batch_size=self.batch_size,
                clip_skip=self.clip_skip,
                hires_fix=self.hires_fix,
                hires_scale=self.hires_scale,
                hires_denoise=self.hires_denoise,
                final_upscale=self.final_upscale,
                final_upscale_model=self.final_upscale_model,
                client_local_date=self.client_local_date,
                sampler_name=self.sampler_name,
                scheduler_name=self.scheduler_name,
                checkpoint_name=self.checkpoint_name,
                vae_name=self.vae_name,
                loras=tuple(
                    {
                        "name": lora.name,
                        "model_strength": lora.model_strength,
                        "clip_strength": lora.clip_strength,
                        "order": lora.order,
                    }
                    for lora in self.loras
                ),
                workflow_template_id=self.workflow_template_id,
                workflow_template_version=self.workflow_template_version,
            )
        except ValidationError as exc:
            raise SnapshotError("snapshot settings are invalid") from exc
