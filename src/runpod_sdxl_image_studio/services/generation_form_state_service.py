"""Save and restore the last successfully enqueued generation form."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from runpod_sdxl_image_studio.adapters.database.repositories.generation_form_state_repository import (  # noqa: E501
    GenerationFormStateRepositoryError,
    GenerationFormStateRepositoryProtocol,
)
from runpod_sdxl_image_studio.domain.generation import Generation
from runpod_sdxl_image_studio.domain.generation_form_state import (
    GenerationFormStateError,
    GenerationFormStateSnapshot,
)

logger = logging.getLogger(__name__)


class LatestGenerationProvider(Protocol):
    def __call__(self) -> Generation | None: ...


class UpscalerProvider(Protocol):
    def __call__(self, generation_id: UUID) -> str | None: ...


@dataclass(frozen=True)
class FormStateRestoreResult:
    snapshot: GenerationFormStateSnapshot | None
    source: str
    warning: str | None = None

    @property
    def is_restored(self) -> bool:
        return self.snapshot is not None


class GenerationFormStateService:
    """Keep restore failures non-fatal while never inventing generation work."""

    def __init__(
        self,
        repository: GenerationFormStateRepositoryProtocol,
        latest_generation_provider: LatestGenerationProvider | None = None,
        *,
        upscaler_provider: UpscalerProvider | None = None,
        state_changed_callback: Callable[[], None] | None = None,
    ) -> None:
        self._repository = repository
        self._latest_generation_provider = latest_generation_provider
        self._upscaler_provider = upscaler_provider
        self._state_changed_callback = state_changed_callback

    def restore(self) -> FormStateRestoreResult:
        stored: GenerationFormStateSnapshot | None = None
        warning: str | None = None
        try:
            stored = self._repository.get()
        except GenerationFormStateRepositoryError:
            warning = "last generation form state could not be read"
            logger.warning("last generation form state read failed", exc_info=True)
        except GenerationFormStateError:
            warning = "last generation form state was invalid and was ignored"
            logger.warning("last generation form state validation failed")
        if stored is not None:
            return FormStateRestoreResult(stored, "form_state", warning)

        if self._latest_generation_provider is None:
            return FormStateRestoreResult(None, "none", warning)
        try:
            generation = self._latest_generation_provider()
        except Exception:  # noqa: BLE001 - startup restore must not break the app
            logger.warning("latest generation fallback could not be read", exc_info=True)
            return FormStateRestoreResult(
                None, "none", warning or "last generation fallback unavailable"
            )
        if generation is None:
            return FormStateRestoreResult(None, "none", warning)
        try:
            upscaler_name = None
            if self._upscaler_provider is not None:
                try:
                    upscaler_name = self._upscaler_provider(generation.id)
                except Exception:  # noqa: BLE001 - optional legacy fallback data
                    logger.warning("latest generation upscaler fallback could not be read")
            fallback = GenerationFormStateSnapshot.from_generation(
                generation,
                upscaler_name=upscaler_name,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("latest generation fallback was invalid: %s", type(exc).__name__)
            return FormStateRestoreResult(
                None, "none", warning or "last generation fallback invalid"
            )
        return FormStateRestoreResult(fallback, "generation_snapshot", warning)

    def save(self, snapshot: GenerationFormStateSnapshot) -> GenerationFormStateSnapshot:
        """Persist after a caller has already observed a successful enqueue."""

        result = self._repository.save(snapshot)
        if self._state_changed_callback is not None:
            self._state_changed_callback()
        return result

    def save_from_ui(self, **values: object) -> GenerationFormStateSnapshot:
        snapshot = GenerationFormStateSnapshot.from_ui(**values)  # type: ignore[arg-type]
        return self.save(snapshot)


__all__ = ["FormStateRestoreResult", "GenerationFormStateService"]
