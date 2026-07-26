"""Validated, reproducible settings for one txt2img request."""

from __future__ import annotations

import ntpath
import posixpath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from runpod_sdxl_image_studio.domain.lora import LoraSetting


class GenerationSettings(BaseModel):
    """Settings accepted by the fixed SDXL txt2img workflow."""

    model_config = ConfigDict(extra="forbid")

    positive_prompt: str = Field(default="", max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    seed: int = Field(default=-1, ge=-1, le=2**64 - 1)
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)
    steps: int = Field(default=28, ge=1, le=150)
    cfg_scale: float = Field(default=5.5, ge=0.0, le=30.0)
    sampler_name: str = Field(min_length=1)
    scheduler_name: str = Field(min_length=1)
    checkpoint_name: str = Field(min_length=1)
    vae_name: str | None = None
    loras: tuple[LoraSetting, ...] = ()
    workflow_template_id: str = Field(default="sdxl_txt2img", min_length=1)
    workflow_template_version: str = Field(default="1.0", min_length=1)

    @field_validator("sampler_name", "scheduler_name", "checkpoint_name")
    @classmethod
    def reject_blank_model_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sampler, scheduler, and checkpoint values cannot be blank")
        return value

    @field_validator("width", "height")
    @classmethod
    def validate_multiple_of_64(cls, value: int) -> int:
        if value % 64 != 0:
            raise ValueError("width and height must be multiples of 64")
        return value

    @field_validator("vae_name")
    @classmethod
    def validate_vae_name(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip().replace("\\", "/")
        if posixpath.isabs(normalized) or ntpath.isabs(normalized):
            raise ValueError("VAE name must be a relative path")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("VAE name contains an unsafe path segment")
        return normalized

    @model_validator(mode="after")
    def validate_lora_collection(self) -> GenerationSettings:
        names = [lora.name for lora in self.loras]
        if len(names) != len(set(names)):
            raise ValueError("The same LoRA cannot be selected more than once")
        orders = [lora.order for lora in self.loras]
        if len(orders) != len(set(orders)):
            raise ValueError("LoRA order values must be unique")
        return self
