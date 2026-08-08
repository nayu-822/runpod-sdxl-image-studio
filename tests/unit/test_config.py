from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from runpod_sdxl_image_studio.config import Settings, get_settings


def test_default_settings_are_valid_and_path_values_are_paths() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.timezone == "Asia/Tokyo"
    assert settings.port == 7860
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.workflow_dir, Path)
    assert settings.max_upscale_factor == 4.0
    assert settings.optional_artifact_repair_batch_size == 2


def test_optional_artifact_repair_batch_is_small_and_bounded_by_recovery_limit() -> None:
    settings = Settings(_env_file=None, optional_artifact_repair_batch_size=10)
    assert settings.optional_artifact_repair_batch_size == 10

    with pytest.raises(ValidationError):
        Settings(_env_file=None, optional_artifact_repair_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, optional_artifact_repair_batch_size=11)
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            recovery_max_items=1,
            optional_artifact_repair_batch_size=2,
        )


def test_rclone_connection_and_transfer_timeouts_are_separate() -> None:
    settings = Settings(
        _env_file=None,
        rclone_connection_timeout_seconds=20,
        rclone_transfer_timeout_seconds=0,
    )
    assert settings.rclone_connection_timeout_seconds == 20
    assert settings.rclone_transfer_timeout_seconds is None

    with pytest.raises(ValidationError):
        Settings(_env_file=None, rclone_transfer_timeout_seconds=-1)


def test_environment_variables_override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_STUDIO_HOST", "0.0.0.0")
    monkeypatch.setenv("IMAGE_STUDIO_PORT", "9000")
    monkeypatch.setenv("IMAGE_STUDIO_DATA_DIR", "C:/tmp/image-studio")
    monkeypatch.setenv("IMAGE_STUDIO_MAX_WIDTH", "1024")

    settings = get_settings()

    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.data_dir == Path("C:/tmp/image-studio")
    assert settings.max_width == 1024


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("port", 0),
        ("port", 65_536),
        ("max_width", 0),
        ("max_height", 0),
        ("max_pixels", 0),
        ("max_batch_count", 0),
        ("max_loras", 0),
        ("thumbnail_size", 0),
        ("max_upscale_factor", 0.99),
        ("comfyui_timeout_seconds", 0),
    ],
)
def test_invalid_limits_are_rejected(field_name: str, invalid_value: int | float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field_name: invalid_value})


def test_settings_loading_does_not_create_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "not-created"
    monkeypatch.setenv("IMAGE_STUDIO_DATA_DIR", str(data_dir))

    Settings(_env_file=None)

    assert not data_dir.exists()


def test_metadata_request_wait_must_fit_inside_download_stale_threshold() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            image_download_stale_after_seconds=40,
            metadata_request_max_wait_seconds=40,
        )

    settings = Settings(
        _env_file=None,
        image_download_stale_after_seconds=120,
        metadata_request_max_wait_seconds=40,
    )
    assert settings.image_download_stale_after_seconds == 120
