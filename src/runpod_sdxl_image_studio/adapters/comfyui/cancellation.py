"""ComfyUI cancellation adapter kept outside the application service."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIError,
    ComfyUIResponseError,
)
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.services.generation_queue_service import CancellationResult


class ComfyUICancellationAdapter:
    """Translate ComfyUI cancellation into a durable confirmation result."""

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
        """Request cancellation and confirm absence from queue and history."""

        try:
            accepted = await self._request_cancellation(prompt_id)
            if not accepted:
                return CancellationResult(
                    requested=True,
                    confirmed=False,
                    message="ComfyUI did not accept the cancellation request",
                )
            attempts = max(1, self._settings.history_max_attempts)
            for attempt in range(attempts):
                queue = await self._client.get_queue_status()
                if (
                    prompt_id not in queue.pending_prompt_ids
                    and prompt_id not in queue.running_prompt_ids
                ):
                    history = await self._client.get_prompt_history(prompt_id)
                    if not history.exists:
                        return CancellationResult(
                            requested=True,
                            confirmed=True,
                            message="Cancellation was confirmed by ComfyUI",
                        )
                    return CancellationResult(
                        requested=True,
                        confirmed=False,
                        message="ComfyUI history shows the prompt still exists",
                    )
                if attempt + 1 < attempts:
                    await self._sleep(self._settings.history_poll_interval_seconds)
            return CancellationResult(
                requested=True,
                confirmed=False,
                message="Cancellation could not be confirmed by ComfyUI",
            )
        except ComfyUIError:
            return CancellationResult(
                requested=True,
                confirmed=False,
                message="ComfyUI cancellation confirmation failed",
            )

    async def _request_cancellation(self, prompt_id: str) -> bool:
        """Use the modern API and fall back only for an explicit 404/405."""

        modern_cancel = getattr(self._client, "cancel_job", None)
        if modern_cancel is not None:
            try:
                # A false response is an explicit refusal/unknown prompt, not a
                # signal to retry with a different endpoint.
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


__all__ = ["ComfyUICancellationAdapter"]
