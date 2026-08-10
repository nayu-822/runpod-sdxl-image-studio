from __future__ import annotations

import httpx
import pytest

from runpod_sdxl_image_studio.adapters.runpod.pod_lifecycle import (
    RUNPOD_API_BASE_URL,
    RunPodLifecycleAdapter,
    RunPodTerminateError,
    RunPodTerminateStatus,
)


def _client(handler):
    return httpx.AsyncClient(
        base_url=RUNPOD_API_BASE_URL,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_terminate_self_uses_fixed_current_pod_and_204() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        result = await adapter.terminate_self()
    assert result.status is RunPodTerminateStatus.TERMINATED
    assert len(requests) == 1
    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/v1/pods/pod-123"
    assert requests[0].headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_terminate_self_hides_unauthorized_response(status: int) -> None:
    async with _client(lambda request: httpx.Response(status, text="secret response")) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()
    assert raised.value.code == "runpod_terminate_unauthorized"
    assert "secret response" not in str(raised.value)
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (404, "runpod_terminate_not_found"),
        (503, "runpod_terminate_unavailable"),
        (200, "runpod_terminate_malformed_response"),
    ],
)
async def test_terminate_self_maps_non_success_responses(status: int, code: str) -> None:
    async with _client(lambda request: httpx.Response(status, json={"raw": "hidden"})) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()
    assert raised.value.code == code
    assert "hidden" not in str(raised.value)


@pytest.mark.asyncio
async def test_delete_timeout_is_ambiguous_without_a_second_delete() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "DELETE":
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json={"status": "running"})

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()
    assert raised.value.code == "runpod_terminate_ambiguous"
    assert raised.value.ambiguous
    assert calls == ["DELETE", "GET"]


@pytest.mark.asyncio
async def test_missing_identity_fails_closed_without_http() -> None:
    async with _client(lambda request: httpx.Response(204)) as client:
        adapter = RunPodLifecycleAdapter(client=client, env={})
        assert not adapter.identity_ready
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()
    assert raised.value.code == "runpod_identity_missing"
