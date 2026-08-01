"""ComfyUI cancellation adapter kept outside the application service."""

from __future__ import annotations

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.services.generation_queue_service import CancellationResult


class ComfyUICancellationAdapter:
    """Translate the single-worker interrupt operation to the queue contract."""

    def __init__(self, client: ComfyUIClient) -> None:
        self._client = client

    async def cancel_prompt(self, prompt_id: str) -> CancellationResult:
        await self._client.interrupt_prompt(prompt_id)
        return CancellationResult(
            requested=True,
            confirmed=True,
            message="ComfyUIへキャンセルを要求しました。",
        )


__all__ = ["ComfyUICancellationAdapter"]
