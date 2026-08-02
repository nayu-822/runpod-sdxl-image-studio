"""Async HTTP client for the supported ComfyUI read-only endpoints."""

from __future__ import annotations

import json
import logging
import ntpath
import posixpath
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from io import BytesIO
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote
from uuid import UUID

import httpx
from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.adapters.comfyui.exceptions import (
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUIResponseError,
    ComfyUITimeoutError,
)
from runpod_sdxl_image_studio.adapters.comfyui.models import (
    ComfyUIConnectionResult,
    ComfyUIObjectInfo,
    ComfyUIOutputImage,
    ComfyUIQueueStatus,
    ComfyUISystemStats,
    PromptHistory,
    PromptHistoryStatus,
    QueuedPrompt,
    RemotePromptState,
    RemotePromptStatus,
)
from runpod_sdxl_image_studio.adapters.comfyui.parsers import (
    parse_object_info,
    parse_prompt_history,
    parse_queued_prompt,
    parse_remote_prompt_status,
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
        self._max_output_image_bytes = app_settings.max_output_image_bytes
        self._max_upscale_input_image_bytes = app_settings.max_upscale_input_image_bytes
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

    async def queue_prompt(
        self,
        workflow: Mapping[str, object],
        client_id: str,
    ) -> QueuedPrompt:
        """Queue one fixed workflow and return its prompt identifier."""

        try:
            UUID(client_id)
        except (ValueError, AttributeError) as exc:
            raise ComfyUIResponseError("ComfyUI client id must be a UUID") from exc
        try:
            request_payload = {"prompt": dict(workflow), "client_id": client_id}
            json.dumps(request_payload)
        except (TypeError, ValueError) as exc:
            raise ComfyUIResponseError("ComfyUI workflow is not JSON serializable") from exc
        response_payload = await self._post_json("/prompt", request_payload)
        return parse_queued_prompt(response_payload)

    async def interrupt_prompt(self, prompt_id: str) -> None:
        """Request interruption of the single active ComfyUI prompt."""

        safe_prompt_id = _validate_identifier(prompt_id, "prompt id")
        await self._post_empty("/interrupt", {"prompt_id": safe_prompt_id})

    async def cancel_job(self, prompt_id: str) -> bool:
        """Cancel one prompt through ComfyUI's current job API."""

        safe_prompt_id = _validate_identifier(prompt_id, "prompt id")
        payload = await self._post_json(f"/api/jobs/{quote(safe_prompt_id, safe='')}/cancel", {})
        cancelled = payload.get("cancelled")
        if not isinstance(cancelled, bool):
            raise ComfyUIResponseError(
                "ComfyUI cancel response did not contain a boolean cancelled"
            )
        return cancelled

    async def get_queue_status(self) -> ComfyUIQueueStatus:
        """Fetch the prompt IDs in ComfyUI's pending and running queues."""

        payload = await self._get_json("/queue")
        return ComfyUIQueueStatus(
            pending_prompt_ids=_queue_prompt_ids(payload.get("queue_pending")),
            running_prompt_ids=_queue_prompt_ids(payload.get("queue_running")),
        )

    async def delete_queued_prompt(self, prompt_id: str) -> None:
        """Remove exactly one prompt from ComfyUI's pending queue."""

        safe_prompt_id = _validate_identifier(prompt_id, "prompt id")
        await self._post_empty("/queue", {"delete": [safe_prompt_id]})

    async def get_prompt_history(self, prompt_id: str) -> PromptHistory:
        """Fetch and parse one prompt's history entry."""

        safe_prompt_id = _validate_identifier(prompt_id, "prompt id")
        try:
            payload = await self._get_json(f"/history/{quote(safe_prompt_id, safe='')}")
        except ComfyUIResponseError as exc:
            if exc.status_code == 404:
                return PromptHistory(
                    safe_prompt_id,
                    False,
                    False,
                    (),
                    None,
                    False,
                    PromptHistoryStatus.NOT_FOUND,
                )
            raise
        return parse_prompt_history(payload, safe_prompt_id)

    async def get_remote_prompt_status(self, prompt_id: str) -> RemotePromptState:
        """Read a prompt state, using legacy queue/history only when modern is unsupported."""

        safe_prompt_id = _validate_identifier(prompt_id, "prompt id")
        path = f"/api/jobs/{quote(safe_prompt_id, safe='')}"
        try:
            payload = await self._get_json(path)
        except ComfyUIResponseError as exc:
            if exc.status_code not in {404, 405}:
                return RemotePromptState(safe_prompt_id, RemotePromptStatus.UNAVAILABLE)
            return await self._get_legacy_remote_prompt_status(safe_prompt_id)
        except (ComfyUIConnectionError, ComfyUITimeoutError):
            return RemotePromptState(safe_prompt_id, RemotePromptStatus.UNAVAILABLE)
        return parse_remote_prompt_status(payload, safe_prompt_id)

    async def _get_legacy_remote_prompt_status(self, prompt_id: str) -> RemotePromptState:
        """Combine queue and history reads without issuing a destructive fallback."""

        try:
            queue = await self.get_queue_status()
            if prompt_id in queue.pending_prompt_ids:
                return RemotePromptState(prompt_id, RemotePromptStatus.PENDING)
            if prompt_id in queue.running_prompt_ids:
                return RemotePromptState(prompt_id, RemotePromptStatus.IN_PROGRESS)
            history = await self.get_prompt_history(prompt_id)
        except ComfyUIError:
            return RemotePromptState(prompt_id, RemotePromptStatus.UNAVAILABLE)
        history_mapping = {
            PromptHistoryStatus.COMPLETED: RemotePromptStatus.COMPLETED,
            PromptHistoryStatus.FAILED: RemotePromptStatus.FAILED,
            PromptHistoryStatus.INTERRUPTED: RemotePromptStatus.CANCELLED,
            PromptHistoryStatus.IN_PROGRESS: RemotePromptStatus.IN_PROGRESS,
            PromptHistoryStatus.NOT_FOUND: RemotePromptStatus.NOT_FOUND,
            PromptHistoryStatus.UNKNOWN: RemotePromptStatus.UNAVAILABLE,
        }
        return RemotePromptState(prompt_id, history_mapping[history.status])

    async def get_output_image(self, image: ComfyUIOutputImage) -> bytes:
        """Fetch one validated image reference from ComfyUI's ``/view`` endpoint."""

        filename = _validate_filename(image.filename)
        subfolder = _validate_subfolder(image.subfolder)
        if image.output_type not in {"output", "temp", "input"}:
            raise ComfyUIResponseError("ComfyUI output type is not allowed")
        response = await self._request(
            "GET",
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": image.output_type},
        )
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type not in {"image/png", "image/webp"}:
            raise ComfyUIResponseError("ComfyUI returned a non-image content type")
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise ComfyUIResponseError("ComfyUI returned an invalid image size") from exc
            if declared_length < 0 or declared_length > self._max_output_image_bytes:
                raise ComfyUIResponseError("ComfyUI output image is too large")
        if not response.content or len(response.content) > self._max_output_image_bytes:
            raise ComfyUIResponseError("ComfyUI returned an empty or oversized image")
        _validate_image_bytes(response.content)
        return response.content

    async def upload_input_image(
        self,
        image_bytes: bytes,
        generation_id: UUID,
        source_sha256: str,
    ) -> ComfyUIOutputImage:
        """Stage a verified source artifact in ComfyUI's input area.

        The generated name is application-owned and therefore never trusts a source
        file name or an external URL.
        """

        if not image_bytes or len(image_bytes) > self._max_upscale_input_image_bytes:
            raise ComfyUIResponseError("upscale input image is empty or oversized")
        if len(source_sha256) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in source_sha256
        ):
            raise ComfyUIResponseError("upscale source hash is invalid")
        _validate_image_bytes(image_bytes)
        filename = f"image-studio-{generation_id.hex}-{source_sha256[:12]}.png"
        response = await self._request(
            "POST",
            "/upload/image",
            data={"type": "input", "overwrite": "false"},
            files={"image": (filename, image_bytes, "image/png")},
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ComfyUIResponseError("ComfyUI upload response is invalid") from exc
        if not isinstance(payload, Mapping):
            raise ComfyUIResponseError("ComfyUI upload response is invalid")
        returned_name = payload.get("name")
        returned_subfolder = payload.get("subfolder", "")
        returned_type = payload.get("type", "input")
        if (
            not isinstance(returned_name, str)
            or not isinstance(returned_subfolder, str)
            or not isinstance(returned_type, str)
        ):
            raise ComfyUIResponseError("ComfyUI upload response is incomplete")
        return ComfyUIOutputImage(
            filename=_validate_filename(returned_name),
            subfolder=_validate_subfolder(returned_subfolder),
            output_type=returned_type if returned_type == "input" else "input",
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
        response = await self._request("GET", path)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ComfyUIResponseError("ComfyUI returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ComfyUIResponseError("ComfyUI returned a non-object JSON payload")
        return payload

    async def _post_json(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        response = await self._request("POST", path, json=dict(payload))
        try:
            response_payload = response.json()
        except ValueError as exc:
            raise ComfyUIResponseError("ComfyUI returned invalid JSON") from exc
        if not isinstance(response_payload, dict):
            raise ComfyUIResponseError("ComfyUI returned a non-object JSON payload")
        return response_payload

    async def _post_empty(self, path: str, payload: Mapping[str, object]) -> None:
        """POST an operation whose successful response may have no body."""

        await self._request("POST", path, json=dict(payload))

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if self._closed:
            raise ComfyUIConnectionError("ComfyUI client is closed")
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient()
            self._http_client = client
        try:
            response = await client.request(
                method,
                self._join_url(path),
                timeout=self._timeout,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise ComfyUITimeoutError("ComfyUI request timed out") from exc
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError("ComfyUI endpoint could not be reached") from exc
        except httpx.RequestError as exc:
            raise ComfyUIConnectionError("ComfyUI request failed") from exc
        if not 200 <= response.status_code < 300:
            raise ComfyUIResponseError(
                f"ComfyUI returned HTTP status {response.status_code}",
                status_code=response.status_code,
            )
        return response

    def _join_url(self, path: str) -> str:
        normalized_path = "/" + path.lstrip("/")
        return f"{self._base_url}{normalized_path}"


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "/" in value or "\\" in value:
        raise ComfyUIResponseError(f"ComfyUI {label} is invalid")
    if value in {".", ".."} or ".." in value:
        raise ComfyUIResponseError(f"ComfyUI {label} is invalid")
    return value


def _queue_prompt_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    prompt_ids: list[str] = []
    for entry in value:
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
            candidates = entry[1:2] or entry
        else:
            candidates = (entry,)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                prompt_ids.append(candidate.strip())
                break
    return tuple(prompt_ids)


def _validate_filename(value: str) -> str:
    safe_value = _validate_identifier(value, "filename")
    if posixpath.basename(safe_value) != safe_value or ntpath.basename(safe_value) != safe_value:
        raise ComfyUIResponseError("ComfyUI filename must be a file name")
    return safe_value


def _validate_subfolder(value: str) -> str:
    if not isinstance(value, str) or value in {".", ".."}:
        raise ComfyUIResponseError("ComfyUI subfolder is invalid")
    if value == "":
        return ""
    normalized = value.replace("\\", "/")
    if posixpath.isabs(normalized) or ntpath.isabs(value):
        raise ComfyUIResponseError("ComfyUI subfolder is invalid")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ComfyUIResponseError("ComfyUI subfolder is invalid")
    return normalized


def _validate_image_bytes(image_bytes: bytes) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"PNG", "WEBP"}:
                    raise ComfyUIResponseError("ComfyUI returned an unsupported image format")
                image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ComfyUIResponseError("ComfyUI returned invalid image data") from exc
