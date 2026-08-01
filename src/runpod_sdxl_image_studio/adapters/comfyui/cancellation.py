"""ComfyUI cancellation adapter kept outside the application service."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import ComfyUIError
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.services.generation_queue_service import CancellationResult


class ComfyUICancellationAdapter:
    """Translate the single-worker interrupt operation to the queue contract."""

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
        try:
            attempts = max(1, self._settings.history_max_attempts)
            for attempt in range(attempts):
                queue = await self._client.get_queue_status()
                if prompt_id in queue.pending_prompt_ids:
                    await self._client.delete_queued_prompt(prompt_id)
                elif prompt_id in queue.running_prompt_ids:
                    await self._client.interrupt_prompt(prompt_id)
                else:
                    history = await self._client.get_prompt_history(prompt_id)
                    if not history.exists:
                        return CancellationResult(
                            requested=True,
                            confirmed=True,
                            message="ComfyUI側で対象promptの停止を確認しました。",
                        )
                    if history.is_completed and history.outputs:
                        return CancellationResult(
                            requested=True,
                            confirmed=False,
                            message="ComfyUIで対象promptの完了画像を確認したためキャンセル未確定です。",
                        )
                    return CancellationResult(
                        requested=True,
                        confirmed=False,
                        message="ComfyUI側で対象promptの状態を確認できないためキャンセル未確定です。",
                    )
                if attempt + 1 < attempts:
                    await self._sleep(self._settings.history_poll_interval_seconds)
            return CancellationResult(
                requested=True,
                confirmed=False,
                message="ComfyUI側で対象promptの停止を確認できませんでした。",
            )
        except ComfyUIError:
            return CancellationResult(
                requested=True,
                confirmed=False,
                message="ComfyUIへのキャンセル確認に失敗しました。",
            )


__all__ = ["ComfyUICancellationAdapter"]
