"""ComfyUI cancellation adapter kept outside the application service."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIError,
    ComfyUIResponseError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    PromptHistory,
    PromptHistoryStatus,
)
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation_queue import CancellationOutcome
from runpod_sdxl_image_studio.services.generation_queue_service import CancellationResult


class ComfyUICancellationAdapter:
    """Request cancellation and classify the resulting remote prompt state."""

    def __init__(
        self,
        client: ComfyUIClient,
        settings: Settings | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._settings = settings or get_settings()
        self._sleep = sleep

    async def cancel_prompt(self, prompt_id: str) -> CancellationResult:
        """Use modern cancellation first, then verify queue and history state."""

        try:
            accepted = await self._request_cancellation(prompt_id)
            return await self._confirm_outcome(prompt_id, accepted=accepted)
        except ComfyUIError:
            return CancellationResult(
                requested=True,
                outcome=CancellationOutcome.UNAVAILABLE,
                message="ComfyUIのキャンセル状態を確認できませんでした。",
            )

    async def _request_cancellation(self, prompt_id: str) -> bool:
        """Use legacy endpoints only after an explicit modern 404 or 405."""

        modern_cancel = getattr(self._client, "cancel_job", None)
        if modern_cancel is not None:
            try:
                # False means the API refused or did not find the prompt. It is
                # still followed by observation, never by a legacy resubmission.
                return bool(await modern_cancel(prompt_id))
            except ComfyUIResponseError as exc:
                if exc.status_code not in {404, 405}:
                    raise

        queue = await self._client.get_queue_status()
        if prompt_id in queue.pending_prompt_ids:
            await self._client.delete_queued_prompt(prompt_id)
            return True
        if prompt_id in queue.running_prompt_ids:
            await self._client.interrupt_prompt(prompt_id)
            return True
        return False

    async def _confirm_outcome(self, prompt_id: str, *, accepted: bool) -> CancellationResult:
        attempts = max(1, self._settings.history_max_attempts)
        last_outcome = CancellationOutcome.IN_PROGRESS
        for attempt in range(attempts):
            queue = await self._client.get_queue_status()
            history = await self._client.get_prompt_history(prompt_id)
            outcome = _history_outcome(history)
            if outcome is not None and outcome is not CancellationOutcome.IN_PROGRESS:
                return _cancellation_result(outcome)
            if (
                prompt_id in queue.pending_prompt_ids
                or prompt_id in queue.running_prompt_ids
                or history.status is PromptHistoryStatus.IN_PROGRESS
            ):
                last_outcome = CancellationOutcome.IN_PROGRESS
            elif history.status is PromptHistoryStatus.NOT_FOUND:
                last_outcome = (
                    CancellationOutcome.CANCELLED if accepted else CancellationOutcome.NOT_FOUND
                )
            else:
                last_outcome = CancellationOutcome.UNAVAILABLE
            if last_outcome in {
                CancellationOutcome.CANCELLED,
                CancellationOutcome.NOT_FOUND,
                CancellationOutcome.UNAVAILABLE,
            }:
                return _cancellation_result(last_outcome)
            if attempt + 1 < attempts:
                await self._sleep(self._settings.history_poll_interval_seconds)
        return _cancellation_result(last_outcome)


def _history_outcome(history: PromptHistory) -> CancellationOutcome | None:
    """Translate typed history state into a cancellation decision."""

    mapping = {
        PromptHistoryStatus.COMPLETED: CancellationOutcome.COMPLETED,
        PromptHistoryStatus.FAILED: CancellationOutcome.FAILED,
        PromptHistoryStatus.INTERRUPTED: CancellationOutcome.CANCELLED,
        PromptHistoryStatus.IN_PROGRESS: CancellationOutcome.IN_PROGRESS,
        PromptHistoryStatus.NOT_FOUND: None,
        PromptHistoryStatus.UNKNOWN: CancellationOutcome.UNAVAILABLE,
    }
    return mapping[history.status]


def _cancellation_result(outcome: CancellationOutcome) -> CancellationResult:
    messages = {
        CancellationOutcome.CANCELLED: "ComfyUIでキャンセルが確定しました。",
        CancellationOutcome.COMPLETED: "キャンセル前に生成が完了しました。",
        CancellationOutcome.FAILED: "キャンセル前に生成が失敗しました。",
        CancellationOutcome.IN_PROGRESS: "ComfyUIの処理は継続中です。",
        CancellationOutcome.NOT_FOUND: "ComfyUIでpromptが見つかりません。",
        CancellationOutcome.UNAVAILABLE: "ComfyUIの状態を判定できません。",
    }
    return CancellationResult(
        requested=True,
        outcome=outcome,
        message=messages[outcome],
    )


__all__ = ["ComfyUICancellationAdapter"]
