"""Typed application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from runpod_sdxl_image_studio.domain.drive_sync import (
    validate_remote_base_path,
    validate_remote_name,
)
from runpod_sdxl_image_studio.domain.metadata_import import MAX_METADATA_RAW_BYTES


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
    max_upscale_input_image_bytes: int = Field(
        52_428_800, validation_alias="IMAGE_STUDIO_MAX_UPSCALE_INPUT_IMAGE_BYTES"
    )
    max_metadata_import_image_bytes: int = Field(
        52_428_800, validation_alias="IMAGE_STUDIO_MAX_METADATA_IMPORT_IMAGE_BYTES"
    )
    max_metadata_sidecar_bytes: int = Field(
        2_000_000, validation_alias="IMAGE_STUDIO_MAX_METADATA_SIDECAR_BYTES"
    )
    max_metadata_raw_bytes: int = Field(
        4_000_000, validation_alias="IMAGE_STUDIO_MAX_METADATA_RAW_BYTES"
    )

    data_dir: Path = Field(
        Path("/workspace/image-studio-data"), validation_alias="IMAGE_STUDIO_DATA_DIR"
    )
    database_url: str | None = Field(
        None,
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
    max_lora_thumbnail_bytes: int = Field(
        10_485_760, validation_alias="IMAGE_STUDIO_MAX_LORA_THUMBNAIL_BYTES"
    )
    lora_thumbnail_max_edge: int = Field(
        512, validation_alias="IMAGE_STUDIO_LORA_THUMBNAIL_MAX_EDGE"
    )
    history_page_size: int = Field(20, validation_alias="IMAGE_STUDIO_HISTORY_PAGE_SIZE")
    history_search_page_size: int = Field(
        20, validation_alias="IMAGE_STUDIO_HISTORY_SEARCH_PAGE_SIZE"
    )
    history_search_max_page_size: int = Field(
        100, validation_alias="IMAGE_STUDIO_HISTORY_SEARCH_MAX_PAGE_SIZE"
    )
    history_search_max_text_length: int = Field(
        500, validation_alias="IMAGE_STUDIO_HISTORY_SEARCH_MAX_TEXT_LENGTH"
    )
    preset_name_max_length: int = Field(100, validation_alias="IMAGE_STUDIO_PRESET_NAME_MAX_LENGTH")
    preset_description_max_length: int = Field(
        1000, validation_alias="IMAGE_STUDIO_PRESET_DESCRIPTION_MAX_LENGTH"
    )
    recent_settings_limit: int = Field(10, validation_alias="IMAGE_STUDIO_RECENT_SETTINGS_LIMIT")
    prompt_diff_max_length: int = Field(
        20_000, validation_alias="IMAGE_STUDIO_PROMPT_DIFF_MAX_LENGTH"
    )
    history_thumbnail_max_edge: int = Field(
        384, validation_alias="IMAGE_STUDIO_HISTORY_THUMBNAIL_MAX_EDGE"
    )
    stale_pending_seconds: int = Field(300, validation_alias="IMAGE_STUDIO_STALE_PENDING_SECONDS")
    recovery_max_items: int = Field(50, validation_alias="IMAGE_STUDIO_RECOVERY_MAX_ITEMS")
    optional_artifact_repair_batch_size: int = Field(
        2,
        validation_alias="IMAGE_STUDIO_OPTIONAL_ARTIFACT_REPAIR_BATCH_SIZE",
    )
    queue_poll_interval_seconds: float = Field(
        2.0, validation_alias="IMAGE_STUDIO_QUEUE_POLL_INTERVAL_SECONDS"
    )
    queue_lease_seconds: float = Field(60.0, validation_alias="IMAGE_STUDIO_QUEUE_LEASE_SECONDS")
    queue_heartbeat_seconds: float = Field(
        20.0, validation_alias="IMAGE_STUDIO_QUEUE_HEARTBEAT_SECONDS"
    )
    queue_max_pending_jobs: int = Field(200, validation_alias="IMAGE_STUDIO_QUEUE_MAX_PENDING_JOBS")
    batch_max_items: int = Field(50, validation_alias="IMAGE_STUDIO_BATCH_MAX_ITEMS")
    reconciliation_grace_seconds: float = Field(
        120.0, validation_alias="IMAGE_STUDIO_RECONCILIATION_GRACE_SECONDS"
    )
    queue_auto_refresh_seconds: float = Field(
        3.0, validation_alias="IMAGE_STUDIO_QUEUE_AUTO_REFRESH_SECONDS"
    )
    image_download_stale_after_seconds: float = Field(
        300.0, validation_alias="IMAGE_STUDIO_IMAGE_DOWNLOAD_STALE_AFTER_SECONDS"
    )
    metadata_request_max_wait_seconds: float = Field(
        60.0, validation_alias="IMAGE_STUDIO_METADATA_REQUEST_MAX_WAIT_SECONDS"
    )
    metadata_connect_timeout_seconds: float = Field(
        10.0, validation_alias="IMAGE_STUDIO_METADATA_CONNECT_TIMEOUT_SECONDS"
    )
    metadata_read_timeout_seconds: float = Field(
        30.0, validation_alias="IMAGE_STUDIO_METADATA_READ_TIMEOUT_SECONDS"
    )
    metadata_rate_limiter_wait_seconds: float = Field(
        5.0, validation_alias="IMAGE_STUDIO_METADATA_RATE_LIMITER_WAIT_SECONDS"
    )
    metadata_heartbeat_interval_seconds: float = Field(
        5.0, validation_alias="IMAGE_STUDIO_METADATA_HEARTBEAT_INTERVAL_SECONDS"
    )
    max_trigger_words: int = Field(50, validation_alias="IMAGE_STUDIO_MAX_TRIGGER_WORDS")
    max_compatible_models: int = Field(20, validation_alias="IMAGE_STUDIO_MAX_COMPATIBLE_MODELS")

    rclone_remote: str = Field("", validation_alias="RCLONE_REMOTE")
    rclone_base_path: str = Field("RunPodSDXLImageStudio", validation_alias="RCLONE_BASE_PATH")
    rclone_config: Path | None = Field(None, validation_alias="RCLONE_CONFIG")
    drive_sync_poll_interval_seconds: float = Field(
        5.0, validation_alias="IMAGE_STUDIO_DRIVE_SYNC_POLL_INTERVAL_SECONDS"
    )
    drive_sync_lease_seconds: float = Field(
        120.0, validation_alias="IMAGE_STUDIO_DRIVE_SYNC_LEASE_SECONDS"
    )
    drive_sync_heartbeat_seconds: float = Field(
        30.0, validation_alias="IMAGE_STUDIO_DRIVE_SYNC_HEARTBEAT_SECONDS"
    )
    drive_discovery_batch_size: int = Field(
        100, validation_alias="IMAGE_STUDIO_DRIVE_DISCOVERY_BATCH_SIZE"
    )
    rclone_connection_timeout_seconds: float = Field(
        20.0, validation_alias="IMAGE_STUDIO_RCLONE_CONNECTION_TIMEOUT_SECONDS"
    )
    rclone_transfer_timeout_seconds: float | None = Field(
        None, validation_alias="IMAGE_STUDIO_RCLONE_TRANSFER_TIMEOUT_SECONDS"
    )

    @field_validator("rclone_remote")
    @classmethod
    def validate_rclone_remote(cls, value: str) -> str:
        return validate_remote_name(value)

    @field_validator("rclone_base_path")
    @classmethod
    def validate_rclone_base(cls, value: str) -> str:
        return validate_remote_base_path(value)

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
        "max_upscale_input_image_bytes",
        "max_metadata_import_image_bytes",
        "max_metadata_sidecar_bytes",
        "max_metadata_raw_bytes",
        "max_lora_thumbnail_bytes",
        "lora_thumbnail_max_edge",
        "max_trigger_words",
        "max_compatible_models",
        "history_page_size",
        "history_search_page_size",
        "history_search_max_page_size",
        "history_search_max_text_length",
        "preset_name_max_length",
        "preset_description_max_length",
        "recent_settings_limit",
        "prompt_diff_max_length",
        "history_thumbnail_max_edge",
        "stale_pending_seconds",
        "recovery_max_items",
        "optional_artifact_repair_batch_size",
        "queue_max_pending_jobs",
        "batch_max_items",
        "drive_discovery_batch_size",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("configured limits must be greater than zero")
        return value

    @field_validator("history_page_size")
    @classmethod
    def validate_history_page_size(cls, value: int) -> int:
        if value > 100:
            raise ValueError("history_page_size must not exceed 100")
        return value

    @field_validator("history_thumbnail_max_edge")
    @classmethod
    def validate_history_thumbnail_edge(cls, value: int) -> int:
        if value > 2048:
            raise ValueError("history_thumbnail_max_edge must not exceed 2048")
        return value

    @field_validator("recovery_max_items")
    @classmethod
    def validate_recovery_max_items(cls, value: int) -> int:
        if value > 100:
            raise ValueError("recovery_max_items must not exceed 100")
        return value

    @field_validator("optional_artifact_repair_batch_size")
    @classmethod
    def validate_optional_artifact_repair_batch_size(cls, value: int) -> int:
        if value > 10:
            raise ValueError("optional_artifact_repair_batch_size must not exceed 10")
        return value

    @field_validator(
        "queue_poll_interval_seconds",
        "queue_lease_seconds",
        "queue_heartbeat_seconds",
        "image_download_stale_after_seconds",
        "metadata_request_max_wait_seconds",
        "metadata_connect_timeout_seconds",
        "metadata_read_timeout_seconds",
        "metadata_rate_limiter_wait_seconds",
        "metadata_heartbeat_interval_seconds",
        "drive_sync_poll_interval_seconds",
        "drive_sync_lease_seconds",
        "drive_sync_heartbeat_seconds",
        "rclone_connection_timeout_seconds",
    )
    @classmethod
    def validate_queue_timing(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("queue timing values must be greater than zero")
        return value

    @field_validator("rclone_transfer_timeout_seconds")
    @classmethod
    def validate_rclone_transfer_timeout(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("rclone_transfer_timeout_seconds must be zero or greater")
        return None if value == 0 else value

    @field_validator("reconciliation_grace_seconds", "queue_auto_refresh_seconds")
    @classmethod
    def validate_queue_optional_timing(cls, value: float) -> float:
        if value < 0:
            raise ValueError("queue optional timing values must not be negative")
        return value

    @model_validator(mode="after")
    def validate_queue_lease(self) -> Settings:
        if self.queue_lease_seconds <= self.queue_heartbeat_seconds:
            raise ValueError("queue_lease_seconds must be greater than queue_heartbeat_seconds")
        if self.drive_sync_lease_seconds <= self.drive_sync_heartbeat_seconds:
            raise ValueError(
                "drive_sync_lease_seconds must be greater than drive_sync_heartbeat_seconds"
            )
        if self.optional_artifact_repair_batch_size > self.recovery_max_items:
            raise ValueError(
                "optional_artifact_repair_batch_size must not exceed recovery_max_items"
            )
        metadata_wait = max(
            self.metadata_request_max_wait_seconds,
            self.metadata_connect_timeout_seconds
            + self.metadata_read_timeout_seconds
            + self.metadata_rate_limiter_wait_seconds
            + self.metadata_heartbeat_interval_seconds
            + 5.0,
        )
        if self.image_download_stale_after_seconds <= metadata_wait:
            raise ValueError(
                "image_download_stale_after_seconds must exceed the maximum metadata request wait"
            )
        if self.max_metadata_sidecar_bytes > MAX_METADATA_RAW_BYTES:
            raise ValueError("max_metadata_sidecar_bytes exceeds the raw metadata contract")
        if self.max_metadata_raw_bytes > MAX_METADATA_RAW_BYTES:
            raise ValueError("max_metadata_raw_bytes exceeds the raw metadata contract")
        return self

    @field_validator("max_upscale_factor")
    @classmethod
    def validate_upscale_factor(cls, value: float) -> float:
        if value < 1.0:
            raise ValueError("max_upscale_factor must be at least 1.0")
        return value


def get_settings() -> Settings:
    """Create and return settings explicitly for the current process."""

    return Settings()
