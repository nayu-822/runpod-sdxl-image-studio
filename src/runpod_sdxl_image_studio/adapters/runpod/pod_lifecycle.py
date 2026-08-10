"""Fixed-scope RunPod self-termination HTTP adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from os import environ
from typing import Any
from urllib.parse import quote

import httpx

RUNPOD_API_BASE_URL = "https://rest.runpod.io/v1"


class RunPodTerminateStatus(StrEnum):
    TERMINATED = "terminated"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class RunPodIdentity:
    pod_id: str | None
    credential_available: bool

    @property
    def is_ready(self) -> bool:
        return bool(self.pod_id and self.credential_available)


@dataclass(frozen=True)
class RunPodTerminateResult:
    status: RunPodTerminateStatus
    code: str = "runpod_terminate_succeeded"


class RunPodTerminateError(RuntimeError):
    """Typed, secret-free RunPod termination failure."""

    def __init__(self, code: str, message: str, *, ambiguous: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.ambiguous = ambiguous


class RunPodLifecycleAdapter:
    """Only DELETE the pod identified by the current process environment."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float = 20.0,
        base_url: str = RUNPOD_API_BASE_URL,
    ) -> None:
        if base_url.rstrip("/") != RUNPOD_API_BASE_URL:
            raise ValueError("RunPod lifecycle base URL is fixed")
        if client is not None and str(client.base_url).rstrip("/") != RUNPOD_API_BASE_URL:
            raise ValueError("RunPod lifecycle client base URL is fixed")
        self._client = client
        self._env = env if env is not None else environ
        self._timeout_seconds = timeout_seconds

    def identity(self) -> RunPodIdentity:
        pod_id = _clean(self._env.get("RUNPOD_POD_ID"))
        api_key = _clean(self._env.get("RUNPOD_API_KEY"))
        return RunPodIdentity(pod_id, bool(api_key))

    @property
    def identity_ready(self) -> bool:
        return self.identity().is_ready

    async def terminate_self(self) -> RunPodTerminateResult:
        identity = self.identity()
        if not identity.pod_id or not identity.credential_available:
            raise RunPodTerminateError(
                "runpod_identity_missing",
                "RunPod self-termination identity is unavailable",
            )
        api_key = _clean(self._env.get("RUNPOD_API_KEY"))
        assert api_key is not None
        path = f"/pods/{quote(identity.pod_id, safe='')}"
        headers = {"Authorization": f"Bearer {api_key}"}
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                base_url=RUNPOD_API_BASE_URL,
                timeout=httpx.Timeout(self._timeout_seconds),
            )
        try:
            try:
                response = await client.delete(path, headers=headers)
            except httpx.TimeoutException as exc:
                confirmed = await self._confirm_after_timeout(client, path, headers)
                if confirmed is RunPodTerminateStatus.TERMINATED:
                    return RunPodTerminateResult(RunPodTerminateStatus.TERMINATED)
                raise RunPodTerminateError(
                    "runpod_terminate_ambiguous",
                    "RunPod termination response was ambiguous",
                    ambiguous=True,
                ) from exc
            except httpx.RequestError as exc:
                raise RunPodTerminateError(
                    "runpod_terminate_unavailable",
                    "RunPod termination service was unavailable",
                ) from exc
            if response.status_code == 204:
                return RunPodTerminateResult(RunPodTerminateStatus.TERMINATED)
            if response.status_code in {401, 403}:
                raise RunPodTerminateError(
                    "runpod_terminate_unauthorized",
                    "RunPod termination was not authorized",
                )
            if response.status_code == 404:
                raise RunPodTerminateError(
                    "runpod_terminate_not_found",
                    "RunPod self pod was not found",
                )
            if 500 <= response.status_code <= 599:
                raise RunPodTerminateError(
                    "runpod_terminate_unavailable",
                    "RunPod termination service returned an unavailable response",
                )
            raise RunPodTerminateError(
                "runpod_terminate_malformed_response",
                "RunPod termination returned an unexpected response",
            )
        finally:
            if owns_client:
                await client.aclose()

    async def _confirm_after_timeout(
        self,
        client: httpx.AsyncClient,
        path: str,
        headers: dict[str, str],
    ) -> RunPodTerminateStatus | None:
        try:
            response = await client.get(path, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError):
            return None
        if response.status_code == 404:
            return RunPodTerminateStatus.TERMINATED
        if response.status_code != 200:
            return None
        try:
            payload: Any = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        status = payload.get("status")
        if isinstance(status, str) and status.casefold() in {"terminated", "deleted"}:
            return RunPodTerminateStatus.TERMINATED
        return None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        return None
    return normalized or None


__all__ = [
    "RUNPOD_API_BASE_URL",
    "RunPodIdentity",
    "RunPodLifecycleAdapter",
    "RunPodTerminateError",
    "RunPodTerminateResult",
    "RunPodTerminateStatus",
    "RunPodPodLifecycleAdapter",
]

# Keep the adapter discoverable under the terminology used by the Phase 12 spec.
RunPodPodLifecycleAdapter = RunPodLifecycleAdapter
