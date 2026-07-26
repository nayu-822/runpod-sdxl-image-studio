"""Typed application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the application.

    Constructing this model is intentionally side-effect free: it does not create
    directories, open the database, or contact ComfyUI or rclone.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
        env_ignore_empty=True,
    )

    environment: str = Field("development", validation_alias="IMAGE_STUDIO_ENV")
    host: str = Field("127.0.0.1", validation_alias="IMAGE_STUDIO_HOST")
    port: int = Field(7860, validation_alias="IMAGE_STUDIO_PORT")
    timezone: str = Field("Asia/Tokyo", validation_alias="IMAGE_STUDIO_TIMEZONE")

    comfyui_base_url: str = Field("http://127.0.0.1:8188", validation_alias="COMFYUI_BASE_URL")
    comfyui_ws_url: str = Field("ws://127.0.0.1:8188/ws", validation_alias="COMFYUI_WS_URL")
    comfyui_output_dir: Path = Field(
        Path("/workspace/ComfyUI/output"), validation_alias="COMFYUI_OUTPUT_DIR"
    )
    comfyui_timeout_seconds: float = Field(30.0, validation_alias="COMFYUI_TIMEOUT_SECONDS")
    generation_timeout_seconds: float = Field(
        600.0, validation_alias="IMAGE_STUDIO_GENERATION_TIMEOUT_SECONDS"
    )
    history_poll_interval_seconds: float = Field(
        2.0, validation_alias="IMAGE_STUDIO_HISTORY_POLL_INTERVAL_SECONDS"
    )
    history_max_attempts: int = Field(10, validation_alias="IMAGE_STUDIO_HISTORY_MAX_ATTEMPTS")
    max_output_image_bytes: int = Field(
        52_428_800, validation_alias="IMAGE_STUDIO_MAX_OUTPUT_IMAGE_BYTES"
    )

    data_dir: Path = Field(
        Path("/workspace/image-studio-data"), validation_alias="IMAGE_STUDIO_DATA_DIR"
    )
    database_url: str = Field(
        "sqlite:////workspace/image-studio-data/app.db",
        validation_alias="IMAGE_STUDIO_DATABASE_URL",
    )
    workflow_dir: Path = Field(
        Path("/workspace/runpod-sdxl-image-studio/workflows"),
        validation_alias="IMAGE_STUDIO_WORKFLOW_DIR",
    )

    checkpoint_dir: Path = Field(
        Path("/workspace/ComfyUI/models/checkpoints"),
        validation_alias="IMAGE_STUDIO_CHECKPOINT_DIR",
    )
    lora_dir: Path = Field(
        Path("/workspace/ComfyUI/models/loras"), validation_alias="IMAGE_STUDIO_LORA_DIR"
    )
    vae_dir: Path = Field(
        Path("/workspace/ComfyUI/models/vae"), validation_alias="IMAGE_STUDIO_VAE_DIR"
    )
    upscaler_dir: Path = Field(
        Path("/workspace/ComfyUI/models/upscale_models"),
        validation_alias="IMAGE_STUDIO_UPSCALER_DIR",
    )

    max_width: int = Field(2048, validation_alias="IMAGE_STUDIO_MAX_WIDTH")
    max_height: int = Field(2048, validation_alias="IMAGE_STUDIO_MAX_HEIGHT")
    max_pixels: int = Field(4_194_304, validation_alias="IMAGE_STUDIO_MAX_PIXELS")
    max_batch_count: int = Field(8, validation_alias="IMAGE_STUDIO_MAX_BATCH_COUNT")
    max_loras: int = Field(8, validation_alias="IMAGE_STUDIO_MAX_LORAS")
    max_upscale_factor: float = Field(4.0, validation_alias="IMAGE_STUDIO_MAX_UPSCALE_FACTOR")
    thumbnail_size: int = Field(512, validation_alias="IMAGE_STUDIO_THUMBNAIL_SIZE")

    rclone_remote: str = Field("", validation_alias="RCLONE_REMOTE")
    rclone_base_path: str = Field("RunPodSDXLImageStudio", validation_alias="RCLONE_BASE_PATH")
    rclone_config: Path | None = Field(None, validation_alias="RCLONE_CONFIG")

    @field_validator("port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        return value

    @field_validator("comfyui_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("comfyui_timeout_seconds must be greater than zero")
        return value

    @field_validator("generation_timeout_seconds", "history_poll_interval_seconds")
    @classmethod
    def validate_generation_timing(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("generation timing values must be greater than zero")
        return value

    @field_validator(
        "max_width",
        "max_height",
        "max_pixels",
        "max_batch_count",
        "max_loras",
        "thumbnail_size",
        "history_max_attempts",
        "max_output_image_bytes",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("configured limits must be greater than zero")
        return value

    @field_validator("max_upscale_factor")
    @classmethod
    def validate_upscale_factor(cls, value: float) -> float:
        if value < 1.0:
            raise ValueError("max_upscale_factor must be at least 1.0")
        return value


def get_settings() -> Settings:
    """Create and return settings explicitly for the current process."""

    return Settings()
