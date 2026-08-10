"""Non-blocking startup preparation for the exact last-used models."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation_form_state import GenerationFormStateSnapshot
from runpod_sdxl_image_studio.domain.model_transfer import ModelTransferStatus
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.generation_form_state_service import (
    GenerationFormStateService,
)
from runpod_sdxl_image_studio.services.model_preparation_service import (
    ModelPreparationResult,
    ModelPreparationService,
)

logger = logging.getLogger(__name__)


class StartupRestoreState(StrEnum):
    RESTORING = "restoring"
    PREPARING_MODELS = "preparing_models"
    WAITING_VISIBILITY = "waiting_visibility"
    READY = "ready"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True)
class StartupRestoreStatus:
    state: StartupRestoreState
    snapshot: GenerationFormStateSnapshot | None
    missing: tuple[str, ...] = ()
    message: str = ""
    capabilities: object | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            StartupRestoreState.READY,
            StartupRestoreState.INCOMPLETE,
            StartupRestoreState.FAILED,
        }


class StartupModelRestoreRuntime:
    """Prepare exact model selections and expose a UI-pollable restore state."""

    def __init__(
        self,
        form_state_service: GenerationFormStateService,
        model_preparation_service: ModelPreparationService,
        settings: Settings,
        *,
        capability_refresh: Callable[[], Coroutine[Any, Any, CapabilityRefreshResult]]
        | None = None,
        poll_interval_seconds: float = 0.2,
    ) -> None:
        self._form_state_service = form_state_service
        self._model_preparation_service = model_preparation_service
        self._settings = settings
        self._capability_refresh = capability_refresh
        self._poll_interval_seconds = max(0.05, poll_interval_seconds)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._status_lock = threading.RLock()
        self._status = StartupRestoreStatus(
            StartupRestoreState.RESTORING,
            None,
            message="前回設定を確認しています。",
        )
        self.result: ModelPreparationResult | None = None

    @property
    def is_ready(self) -> bool:
        """Return whether startup restore reached a terminal decision."""

        return self._ready_event.is_set()

    def status(self) -> StartupRestoreStatus:
        """Return a snapshot safe for a Gradio poll handler."""

        with self._status_lock:
            return self._status

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ready_event.clear()
        with self._status_lock:
            self._status = StartupRestoreStatus(
                StartupRestoreState.RESTORING,
                None,
                message="前回設定を確認しています。",
            )
        self.result = None
        self._thread = threading.Thread(
            target=self._run,
            name="startup-model-restore",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None

    def _run(self) -> None:
        snapshot: GenerationFormStateSnapshot | None = None
        try:
            if not self._settings.restore_last_settings_on_startup:
                self._finish(StartupRestoreState.READY, None, "前回設定の復元は無効です。")
                return

            restored = self._form_state_service.restore()
            snapshot = restored.snapshot
            if snapshot is None:
                self._finish(
                    StartupRestoreState.READY,
                    None,
                    restored.warning or "復元する前回設定はありません。",
                )
                return
            self._set_state(
                StartupRestoreState.PREPARING_MODELS,
                snapshot,
                restored.warning or "前回設定のモデルを準備しています。",
            )
            if not self._settings.auto_prepare_last_models:
                self._finish(
                    StartupRestoreState.READY,
                    snapshot,
                    restored.warning or "モデルの自動準備は無効です。",
                )
                return

            self.result = asyncio.run(
                self._model_preparation_service.prepare_previous_models(
                    snapshot.checkpoint_name,
                    snapshot.vae_name,
                    snapshot.lora_names,
                    snapshot.upscaler_name,
                )
            )
            failures = self._wait_for_model_jobs(self.result)
            if self._stop_event.is_set():
                return
            missing = tuple(dict.fromkeys((*self.result.missing, *failures)))
            self._set_state(
                StartupRestoreState.WAITING_VISIBILITY,
                snapshot,
                "モデル取得後のComfyUI反映を確認しています。",
                missing=missing,
            )
            capabilities: object | None = None
            if self._capability_refresh is not None:
                refresh_result: CapabilityRefreshResult = asyncio.run(self._capability_refresh())
                if not refresh_result.is_success or refresh_result.capabilities is None:
                    self._finish(
                        StartupRestoreState.FAILED,
                        snapshot,
                        "モデル取得後のComfyUI能力情報を更新できませんでした。",
                        missing=missing,
                    )
                    return
                capabilities = refresh_result.capabilities
            if missing:
                self._finish(
                    StartupRestoreState.INCOMPLETE,
                    snapshot,
                    "不足model: " + ", ".join(missing),
                    missing=missing,
                    capabilities=capabilities,
                )
            else:
                self._finish(
                    StartupRestoreState.READY,
                    snapshot,
                    "前回設定をモデル取得後に復元できます。",
                    capabilities=capabilities,
                )
        except Exception as exc:  # noqa: BLE001 - startup restore must not stop the app
            logger.warning("startup model restoration failed error=%s", type(exc).__name__)
            self._finish(
                StartupRestoreState.FAILED,
                snapshot,
                "前回設定の自動復元に失敗しました。",
            )
        finally:
            if not self.status().is_terminal:
                self._finish(
                    StartupRestoreState.FAILED,
                    snapshot,
                    "前回設定の自動復元を完了できませんでした。",
                )
            self._ready_event.set()

    def _wait_for_model_jobs(self, result: ModelPreparationResult) -> tuple[str, ...]:
        tracked = tuple(result.jobs)
        while not self._stop_event.is_set():
            current = {job.id: job for job in self._model_preparation_service.list_jobs(500)}
            failures: list[str] = []
            all_terminal = True
            for original in tracked:
                original_id = getattr(original, "id", None)
                job = current.get(original_id, original) if original_id is not None else original
                status = getattr(job, "status", None)
                if status is None:
                    all_terminal = False
                    continue
                if not status.is_terminal:
                    all_terminal = False
                    continue
                if status is not ModelTransferStatus.COMPLETED:
                    kind = getattr(getattr(job, "kind", None), "value", "model")
                    path = getattr(job, "remote_relative_path", "unknown")
                    failures.append(f"{kind}:{path} ({status.value})")
            if all_terminal:
                return tuple(failures)
            self._stop_event.wait(self._poll_interval_seconds)
        return ()

    def _set_state(
        self,
        state: StartupRestoreState,
        snapshot: GenerationFormStateSnapshot | None,
        message: str,
        *,
        missing: tuple[str, ...] = (),
        capabilities: object | None = None,
    ) -> None:
        with self._status_lock:
            self._status = StartupRestoreStatus(
                state,
                snapshot,
                missing,
                message,
                capabilities,
            )

    def _finish(
        self,
        state: StartupRestoreState,
        snapshot: GenerationFormStateSnapshot | None,
        message: str,
        *,
        missing: tuple[str, ...] = (),
        capabilities: object | None = None,
    ) -> None:
        self._set_state(
            state,
            snapshot,
            message,
            missing=missing,
            capabilities=capabilities,
        )


__all__ = [
    "StartupModelRestoreRuntime",
    "StartupRestoreState",
    "StartupRestoreStatus",
]
