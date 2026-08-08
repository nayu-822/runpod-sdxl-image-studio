"""Start a disposable local Gradio instance for the Phase 8 browser test."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database
from runpod_sdxl_image_studio.ui.app_builder import build_app

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    data_dir = Path(os.environ.get("IMAGE_STUDIO_BROWSER_DATA_DIR", tempfile.mkdtemp()))
    database_path = data_dir / "database" / "image_studio.sqlite3"
    settings = Settings(
        _env_file=None,
        environment="browser-test",
        data_dir=data_dir,
        database_url=f"sqlite:///{database_path.as_posix()}",
        workflow_dir=ROOT / "workflows",
        checkpoint_dir=data_dir / "models" / "checkpoints",
        lora_dir=data_dir / "models" / "loras",
        vae_dir=data_dir / "models" / "vae",
        upscaler_dir=data_dir / "models" / "upscale_models",
    )
    upgrade_database(settings, ROOT)
    demo = build_app(settings)
    demo.launch(
        server_name=os.environ.get("IMAGE_STUDIO_BROWSER_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("IMAGE_STUDIO_BROWSER_PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()
