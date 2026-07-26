"""WebSocket progress adapter for ComfyUI prompt execution."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIWebSocketDisconnectedError,
    ComfyUIWebSocketError,
    ComfyUIWebSocketTimeoutError,
)
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation import (
    GenerationProgress,
    GenerationStatus,
)

logger = logging.getLogger(__name__)


class ComfyUIWebSocketClient:
    """Receive progress events for a single ComfyUI prompt."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        app_settings = settings or get_settings()
        self._url = app_settings.comfyui_ws_url
        self._timeout = app_settings.generation_timeout_seconds
        self._connect = connect or websockets.connect

    async def watch_prompt(
        self,
        prompt_id: str,
        client_id: str,
    ) -> AsyncIterator[GenerationProgress]:
        """Yield normalized events until ComfyUI reports prompt completion."""

        url = _with_client_id(self._url, client_id)
        try:
            async with self._connect(url, open_timeout=self._timeout) as socket:
                while True:
                    try:
                        raw_message = await asyncio.wait_for(socket.recv(), self._timeout)
                    except TimeoutError as exc:
                        raise ComfyUIWebSocketTimeoutError("ComfyUI WebSocket timed out") from exc
                    if isinstance(raw_message, bytes):
                        continue
                    message = _decode_message(raw_message)
                    if message is None:
                        continue
                    progress = parse_websocket_message(message, prompt_id)
                    if progress is None:
                        continue
                    yield progress
                    if progress.state in {
                        GenerationStatus.COMPLETED,
                        GenerationStatus.FAILED,
                    }:
                        return
        except ComfyUIWebSocketError:
            raise
        except TimeoutError as exc:
            raise ComfyUIWebSocketTimeoutError("ComfyUI WebSocket timed out") from exc
        except ConnectionClosed as exc:
            raise ComfyUIWebSocketDisconnectedError("ComfyUI WebSocket disconnected") from exc
        except (OSError, WebSocketException) as exc:
            raise ComfyUIWebSocketError("ComfyUI WebSocket connection failed") from exc


def parse_websocket_message(
    message: Mapping[str, object],
    prompt_id: str,
) -> GenerationProgress | None:
    """Convert a ComfyUI event into a safe domain progress value."""

    event_type = message.get("type")
    data = message.get("data")
    if not isinstance(event_type, str) or not isinstance(data, Mapping):
        return None
    event_prompt_id = data.get("prompt_id")
    if event_prompt_id is not None and event_prompt_id != prompt_id:
        return None

    if event_type == "status":
        return GenerationProgress(
            state=GenerationStatus.QUEUED,
            prompt_id=prompt_id,
            message="ComfyUIキューで待機中",
        )
    if event_type in {"execution_start", "executing"}:
        node = data.get("node")
        if event_type == "executing" and node is None:
            return GenerationProgress(
                state=GenerationStatus.COMPLETED,
                prompt_id=prompt_id,
                message="ComfyUI処理が完了しました",
            )
        return GenerationProgress(
            state=GenerationStatus.RUNNING,
            prompt_id=prompt_id,
            current_node=node if isinstance(node, str) else None,
            message="画像を生成中です",
        )
    if event_type == "progress":
        value = _safe_number(data.get("value"))
        maximum = _safe_number(data.get("max"))
        percentage = None
        if value is not None and maximum is not None and maximum > 0:
            percentage = min(100.0, max(0.0, value / maximum * 100.0))
        return GenerationProgress(
            state=GenerationStatus.RUNNING,
            prompt_id=prompt_id,
            current_node=data.get("node") if isinstance(data.get("node"), str) else None,
            value=value,
            maximum=maximum,
            percentage=percentage,
            message="画像を生成中です",
        )
    if event_type == "executed":
        return GenerationProgress(
            state=GenerationStatus.COMPLETED,
            prompt_id=prompt_id,
            message="ComfyUI処理が完了しました",
        )
    if event_type == "execution_error":
        return GenerationProgress(
            state=GenerationStatus.FAILED,
            prompt_id=prompt_id,
            message="ComfyUIで画像生成に失敗しました",
        )
    return None


def _decode_message(raw_message: str) -> Mapping[str, object] | None:
    try:
        payload = json.loads(raw_message)
    except (TypeError, ValueError):
        logger.debug("Ignoring invalid ComfyUI WebSocket JSON message")
        return None
    return payload if isinstance(payload, Mapping) else None


def _safe_number(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _with_client_id(url: str, client_id: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["clientId"] = client_id
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
