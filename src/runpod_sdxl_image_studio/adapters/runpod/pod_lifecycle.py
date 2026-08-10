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
                return await self._resolve_delete_failure(
                    client,
                    path,
                    headers,
                    code="runpod_terminate_ambiguous",
                    message="RunPod termination response was ambiguous",
                    cause=exc,
                )
            except httpx.ConnectError as exc:
                # A connect failure is known to occur before a request can be
                # delivered.  It is a confirmed failure, not a delivery
                # ambiguity, and therefore may be explicitly retried later.
                raise RunPodTerminateError(
                    "runpod_terminate_unavailable",
                    "RunPod termination service was unavailable",
                ) from exc
            except httpx.RequestError as exc:
                return await self._resolve_delete_failure(
                    client,
                    path,
                    headers,
                    code="runpod_terminate_unavailable",
                    message="RunPod termination service was unavailable",
                    cause=exc,
                )
            if response.status_code == 204:
                return RunPodTerminateResult(RunPodTerminateStatus.TERMINATED)
            if response.status_code in {401, 403}:
                raise RunPodTerminateError(
                    "runpod_terminate_unauthorized",
                    "RunPod termination was not authorized",
                )
            if response.status_code == 404:
                return await self._resolve_delete_failure(
                    client,
                    path,
                    headers,
                    code="runpod_terminate_not_found",
                    message="RunPod self pod was not found",
                )
            if 500 <= response.status_code <= 599:
                return await self._resolve_delete_failure(
                    client,
                    path,
                    headers,
                    code="runpod_terminate_unavailable",
                    message="RunPod termination service returned an unavailable response",
                )
            return await self._resolve_delete_failure(
                client,
                path,
                headers,
                code="runpod_terminate_malformed_response",
                message="RunPod termination returned an unexpected response",
            )
        finally:
            if owns_client:
                await client.aclose()

    async def _resolve_delete_failure(
        self,
        client: httpx.AsyncClient,
        path: str,
        headers: dict[str, str],
        *,
        code: str,
        message: str,
        cause: BaseException | None = None,
    ) -> RunPodTerminateResult:
        confirmed, pod_exists = await self._confirm_after_delete_failure(client, path, headers)
        if confirmed is RunPodTerminateStatus.TERMINATED:
            return RunPodTerminateResult(RunPodTerminateStatus.TERMINATED)
        if pod_exists:
            error = RunPodTerminateError(
                "runpod_terminate_confirmed_failure"
                if code == "runpod_terminate_ambiguous"
                else code,
                "RunPod termination failed while the self pod still exists"
                if code == "runpod_terminate_ambiguous"
                else message,
                ambiguous=False,
            )
        else:
            error = RunPodTerminateError(
                "runpod_terminate_ambiguous",
                "RunPod termination response was ambiguous",
                ambiguous=True,
            )
        if cause is not None:
            raise error from cause
        raise error

    async def _confirm_after_delete_failure(
        self,
        client: httpx.AsyncClient,
        path: str,
        headers: dict[str, str],
    ) -> tuple[RunPodTerminateStatus | None, bool]:
        try:
            response = await client.get(path, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError):
            return None, False
        if response.status_code == 404:
            return RunPodTerminateStatus.TERMINATED, False
        if response.status_code != 200:
            return None, False
        try:
            payload: Any = response.json()
        except ValueError:
            return None, True
        if isinstance(payload, dict):
            status = payload.get("status")
            if isinstance(status, str) and status.casefold() in {"terminated", "deleted"}:
                return RunPodTerminateStatus.TERMINATED, False
        # A successful GET proves that the pod still exists, even if the
        # response body is not useful enough to classify its lifecycle state.
        return None, True


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
