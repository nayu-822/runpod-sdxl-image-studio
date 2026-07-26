"""Explicit application-start database migration entry point."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from runpod_sdxl_image_studio.adapters.database.engine import (
    ensure_database_directory,
    resolved_database_url,
)
from runpod_sdxl_image_studio.config import Settings


def upgrade_database(settings: Settings, project_root: Path | None = None) -> None:
    ensure_database_directory(settings)
    root = project_root or Path(__file__).resolve().parents[3]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", resolved_database_url(settings))
    command.upgrade(config, "head")
