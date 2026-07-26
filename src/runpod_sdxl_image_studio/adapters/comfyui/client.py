"""Async HTTP client for the supported ComfyUI read-only endpoints."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import TracebackType
from typing import Self

import httpx

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIConnectionError,
    ComfyUIResponseError,
    ComfyUITimeoutError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUIConnectionResult,
    ComfyUIObjectInfo,
    ComfyUISystemStats,
)
from runpod_sdxl_image_studio.adapters.comfyui.parsers import (
    parse_object_info,
    parse_system_stats,
)
from runpod_sdxl_image_studio.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Small, injectable, read-only HTTP boundary for ComfyUI."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        app_settings = settings or get_settings()
        configured_base_url = base_url or app_settings.comfyui_base_url
        self._base_url = configured_base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else app_settings.comfyui_timeout_seconds
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._closed = False

    @property
    def base_url(self) -> str:
        """Return the normalized ComfyUI base URL."""

        return self._base_url

    @property
    def timeout(self) -> float:
        """Return the request timeout in seconds."""

        return self._timeout

    async def get_system_stats(self) -> ComfyUISystemStats:
        """Fetch and parse ``/system_stats``."""

        payload = await self._get_json("/system_stats")
        return parse_system_stats(payload)

    async def get_object_info(self) -> ComfyUIObjectInfo:
        """Fetch and parse ``/object_info``."""

        payload = await self._get_json("/object_info")
        logger.info("ComfyUI /object_info 取得成功")
        return parse_object_info(payload)

    async def check_connection(self) -> ComfyUIConnectionResult:
        """Perform an explicit health check and propagate adapter errors."""

        checked_at = datetime.now(UTC)
        logger.info("ComfyUI 接続確認開始")
        system_stats = await self.get_system_stats()

        logger.info("ComfyUI 接続確認成功")
        return ComfyUIConnectionResult(
            is_connected=True,
            message="ComfyUIに接続できます",
            checked_at=checked_at,
            system_stats=system_stats,
        )

    async def close(self) -> None:
        """Close an internally created HTTP client; injected clients remain owned by callers."""

        if self._closed:
            return
        if self._http_client is not None and self._owns_http_client:
            await self._http_client.aclose()
        self._closed = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def _get_json(self, path: str) -> dict[str, object]:
        if self._closed:
            raise ComfyUIConnectionError("ComfyUI client is closed")
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient()
            self._http_client = client

        url = self._join_url(path)
        try:
            response = await client.get(url, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise ComfyUITimeoutError("ComfyUI request timed out") from exc
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError("ComfyUI endpoint could not be reached") from exc
        except httpx.RequestError as exc:
            raise ComfyUIConnectionError("ComfyUI request failed") from exc

        if not 200 <= response.status_code < 300:
            raise ComfyUIResponseError(f"ComfyUI returned HTTP status {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ComfyUIResponseError("ComfyUI returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ComfyUIResponseError("ComfyUI returned a non-object JSON payload")
        return payload

    def _join_url(self, path: str) -> str:
        normalized_path = "/" + path.lstrip("/")
        return f"{self._base_url}{normalized_path}"
