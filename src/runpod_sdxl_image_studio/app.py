"""Gradio entry point for the application."""

from __future__ import annotations

from runpod_sdxl_image_studio.config import get_settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database
from runpod_sdxl_image_studio.domain.state_sync import StateRestoreStatus
from runpod_sdxl_image_studio.services.state_restore_service import StateRestoreService
from runpod_sdxl_image_studio.ui.app_builder import build_app, build_application_runtime

__all__ = ["build_app", "main"]


def main() -> None:
    """Launch the Gradio server using configured host and port."""

    settings = get_settings()
    restore_result = StateRestoreService(settings).restore_if_missing()
    if restore_result.status in {StateRestoreStatus.UNAVAILABLE, StateRestoreStatus.FAILED}:
        raise RuntimeError("remote write protection: state restore could not be verified")
    upgrade_database(settings)
    runtime = build_application_runtime(
        settings,
        run_stateless_reconciliation=restore_result.status is StateRestoreStatus.RESTORED,
        initial_remote_sha256=(
            restore_result.metadata.sha256 if restore_result.metadata is not None else None
        ),
    )
    try:
        runtime.start()
        runtime.demo.launch(
            server_name=settings.host,
            server_port=settings.port,
            share=False,
        )
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
