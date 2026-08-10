"""Non-blocking startup preparation for the exact last-used models."""

from __future__ import annotations

import asyncio
import logging
import threading

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.services.generation_form_state_service import (
    GenerationFormStateService,
)
from runpod_sdxl_image_studio.services.model_preparation_service import (
    ModelPreparationResult,
    ModelPreparationService,
)

logger = logging.getLogger(__name__)


class StartupModelRestoreRuntime:
    """Start model preparation in the background so Gradio is immediately usable."""

    def __init__(
        self,
        form_state_service: GenerationFormStateService,
        model_preparation_service: ModelPreparationService,
        settings: Settings,
    ) -> None:
        self._form_state_service = form_state_service
        self._model_preparation_service = model_preparation_service
        self._settings = settings
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self.result: ModelPreparationResult | None = None

    @property
    def is_ready(self) -> bool:
        """Return whether startup restore has finished deciding/queuing work."""

        return self._ready_event.is_set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="startup-model-restore",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._ready_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        self._thread = None

    def _run(self) -> None:
        try:
            if not self._settings.restore_last_settings_on_startup:
                return
            restored = self._form_state_service.restore()
            if not restored.is_restored or not self._settings.auto_prepare_last_models:
                return
            snapshot = restored.snapshot
            assert snapshot is not None
            try:
                self.result = asyncio.run(
                    self._model_preparation_service.prepare_previous_models(
                        snapshot.checkpoint_name,
                        snapshot.vae_name,
                        snapshot.lora_names,
                        snapshot.upscaler_name,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - model preparation is best effort at startup
                logger.warning("startup model preparation failed error=%s", type(exc).__name__)
        finally:
            self._ready_event.set()


__all__ = ["StartupModelRestoreRuntime"]
