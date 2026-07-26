"""Application service for ComfyUI status and capability retrieval."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUIParseError,
    ComfyUIResponseError,
    ComfyUITimeoutError,
)
from runpod_sdxl_image_studio.adapters.comfyui.parsers import parse_capabilities
from runpod_sdxl_image_studio.domain.system_status import (
    CapabilityRefreshResult,
    ComfyUIStatus,
)

logger = logging.getLogger(__name__)


class ComfyUIService:
    """Coordinate ComfyUI calls without exposing adapter details to the UI."""

    def __init__(self, client: ComfyUIClient) -> None:
        self._client = client

    async def get_status(self) -> ComfyUIStatus:
        """Check ComfyUI and aggregate system stats with available capabilities."""

        try:
            connection = await self._client.check_connection()
        except ComfyUIError as exc:
            safe_error = _safe_error_message(exc)
            logger.warning("ComfyUI connection check failed: %s", type(exc).__name__)
            return ComfyUIStatus(
                is_connected=False,
                message=safe_error,
                checked_at=datetime.now(UTC),
                system_stats=None,
                capabilities=None,
                warnings=(),
                error_summary=safe_error,
            )
        if not connection.is_connected:
            return ComfyUIStatus(
                is_connected=False,
                message=connection.message,
                checked_at=connection.checked_at,
                system_stats=None,
                capabilities=None,
                warnings=(),
                error_summary=connection.message,
            )

        refresh_result = await self.refresh_capabilities()
        if not refresh_result.is_success or refresh_result.capabilities is None:
            return ComfyUIStatus(
                is_connected=True,
                message="ComfyUIに接続できますが、能力情報を取得できませんでした",
                checked_at=connection.checked_at,
                system_stats=connection.system_stats,
                capabilities=None,
                warnings=refresh_result.warnings,
                error_summary=refresh_result.message,
            )
        capabilities = refresh_result.capabilities

        return ComfyUIStatus(
            is_connected=True,
            message="ComfyUIに接続でき、能力情報を取得しました",
            checked_at=connection.checked_at,
            system_stats=connection.system_stats,
            capabilities=capabilities,
            warnings=capabilities.warnings,
            error_summary=None,
        )

    async def refresh_capabilities(self) -> CapabilityRefreshResult:
        """Refresh model and sampler choices from ``/object_info``."""

        try:
            object_info = await self._client.get_object_info()
            capabilities = parse_capabilities(object_info)
        except ComfyUIError as exc:
            safe_error = _safe_error_message(exc)
            logger.warning("ComfyUI capability retrieval failed: %s", type(exc).__name__)
            return CapabilityRefreshResult(
                is_success=False,
                message=safe_error,
                capabilities=None,
            )

        logger.info("ComfyUI capabilities parsed with %d warning(s)", len(capabilities.warnings))
        return CapabilityRefreshResult(
            is_success=True,
            message="モデル一覧を更新しました",
            capabilities=capabilities,
            warnings=capabilities.warnings,
        )


def initial_status() -> ComfyUIStatus:
    """Return the local-only state used before a user requests a connection check."""

    return ComfyUIStatus(
        is_connected=False,
        message="未確認",
        checked_at=None,
        system_stats=None,
        capabilities=None,
        warnings=(),
        error_summary=None,
    )


def _safe_error_message(error: ComfyUIError) -> str:
    if isinstance(error, ComfyUITimeoutError):
        return "ComfyUIへの接続がタイムアウトしました"
    if isinstance(error, ComfyUIConnectionError):
        return "ComfyUIへ接続できませんでした"
    if isinstance(error, ComfyUIResponseError):
        return "ComfyUIから正常な応答を取得できませんでした"
    if isinstance(error, ComfyUIParseError):
        return "ComfyUIの応答を解析できませんでした"
    return "ComfyUIの状態取得に失敗しました"


def status_checked_at_or_now(status: ComfyUIStatus) -> datetime:
    """Return a timestamp for UI actions that need one after a refresh."""

    return status.checked_at or datetime.now(UTC)
