"""Validated, reproducible settings for one txt2img request."""

from __future__ import annotations

import ntpath
import posixpath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from runpod_sdxl_image_studio.domain.lora import LoraSetting

RANDOM_SEED = -1
MAX_SEED = 2**63 - 1
LEGACY_WORKFLOW_TEMPLATE_VERSION = "2.0"
CURRENT_WORKFLOW_TEMPLATE_VERSION = "2.1"
LEGACY_DEFAULT_FINAL_UPSCALE_MODEL = "4x-UltraSharp.pth"


class GenerationSettings(BaseModel):
    """Settings accepted by the fixed SDXL txt2img workflow."""

    model_config = ConfigDict(extra="forbid")

    positive_prompt: str = Field(default="", max_length=10_000)
    negative_prompt: str = Field(default="", max_length=10_000)
    seed: int = Field(default=RANDOM_SEED, ge=RANDOM_SEED, le=MAX_SEED)
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)
    steps: int = Field(default=28, ge=1, le=150)
    cfg_scale: float = Field(default=5.5, ge=0.0, le=30.0)
    batch_size: int = Field(default=1, ge=1, le=4)
    clip_skip: int = Field(default=1, ge=1, le=12)
    hires_fix: bool = False
    hires_scale: float = Field(default=1.5, ge=1.0, le=4.0)
    hires_denoise: float = Field(default=0.4, ge=0.0, le=1.0)
    hires_resize_method: Literal["lanczos", "nearest-exact", "bilinear", "bicubic"] = "lanczos"
    hires_steps: int = Field(default=20, ge=1, le=150)
    hires_cfg_scale: float = Field(default=5.5, ge=0.0, le=30.0)
    hires_sampler_name: str = Field(default="euler", min_length=1)
    hires_scheduler_name: str = Field(default="normal", min_length=1)
    final_upscale: bool = False
    final_upscale_model: str | None = None
    client_local_date: str | None = None
    sampler_name: str = Field(min_length=1)
    scheduler_name: str = Field(min_length=1)
    checkpoint_name: str = Field(min_length=1)
    vae_name: str | None = None
    loras: tuple[LoraSetting, ...] = ()
    workflow_template_id: str = Field(default="sdxl_txt2img", min_length=1)
    workflow_template_version: str = Field(default=CURRENT_WORKFLOW_TEMPLATE_VERSION, min_length=1)

    @field_validator(
        "sampler_name",
        "scheduler_name",
        "checkpoint_name",
        "hires_sampler_name",
        "hires_scheduler_name",
    )
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

    @field_validator("final_upscale_model")
    @classmethod
    def validate_final_upscale_model(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip().replace("\\", "/")
        if posixpath.isabs(normalized) or ntpath.isabs(normalized):
            raise ValueError("upscale model must be a relative path")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("upscale model contains an unsafe path segment")
        return normalized

    @field_validator("client_local_date")
    @classmethod
    def validate_client_local_date(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized) is None:
            raise ValueError("client_local_date must use YYYY-MM-DD")
        year, month, day = (int(part) for part in normalized.split("-"))
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            raise ValueError("client_local_date is invalid")
        try:
            from datetime import date

            date(year, month, day)
        except ValueError as exc:
            raise ValueError("client_local_date is invalid") from exc
        return normalized

    @model_validator(mode="after")
    def validate_lora_collection(self) -> GenerationSettings:
        if self.final_upscale and self.final_upscale_model is None:
            raise ValueError("final_upscale_model is required when final_upscale is enabled")
        names = [lora.name for lora in self.loras]
        if len(names) != len(set(names)):
            raise ValueError("The same LoRA cannot be selected more than once")
        orders = [lora.order for lora in self.loras]
        if len(orders) != len(set(orders)):
            raise ValueError("LoRA order values must be unique")
        return self


__all__ = [
    "CURRENT_WORKFLOW_TEMPLATE_VERSION",
    "LEGACY_DEFAULT_FINAL_UPSCALE_MODEL",
    "LEGACY_WORKFLOW_TEMPLATE_VERSION",
    "MAX_SEED",
    "RANDOM_SEED",
    "GenerationSettings",
]
