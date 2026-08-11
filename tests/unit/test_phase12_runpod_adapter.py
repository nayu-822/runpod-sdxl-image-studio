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
    assert raised.value.ambiguous is False
    assert "secret response" not in str(raised.value)
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("get_response", "expected"),
    [
        (httpx.Response(200, json={"desiredStatus": "TERMINATED"}), "terminated"),
        (httpx.Response(200, json={"desiredStatus": "RUNNING"}), "confirmed"),
        (httpx.Response(200, json={"desiredStatus": "EXITED"}), "ambiguous"),
    ],
)
async def test_terminate_self_confirms_non_success_responses(
    get_response: httpx.Response,
    expected: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(503, json={"raw": "hidden"})
        return get_response

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        if expected == "terminated":
            result = await adapter.terminate_self()
            assert result.status is RunPodTerminateStatus.TERMINATED
        else:
            with pytest.raises(RunPodTerminateError) as raised:
                await adapter.terminate_self()
            assert raised.value.ambiguous is (expected == "ambiguous")
            assert raised.value.code == (
                "runpod_terminate_ambiguous"
                if expected == "ambiguous"
                else "runpod_terminate_unavailable"
            )
            assert "hidden" not in str(raised.value)


@pytest.mark.asyncio
async def test_get_404_confirms_termination_after_delete_failure() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(503)
        return httpx.Response(404)

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        result = await adapter.terminate_self()

    assert result.status is RunPodTerminateStatus.TERMINATED
    assert requests == ["DELETE", "GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_response",
    [
        httpx.Response(200, json={"desiredStatus": "UNKNOWN"}),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"status": "deleted"}),
        httpx.Response(200, json=[]),
        httpx.Response(500),
        httpx.Response(204),
        httpx.Response(200, content=b"{"),
    ],
)
async def test_unknown_or_malformed_get_confirmation_is_ambiguous(
    get_response: httpx.Response,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(503)
        return get_response

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()

    assert raised.value.code == "runpod_terminate_ambiguous"
    assert raised.value.ambiguous


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "request"])
async def test_get_transport_failure_is_ambiguous(failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(503)
        if failure == "timeout":
            raise httpx.ReadTimeout("get timeout", request=request)
        raise httpx.ReadError("get connection lost", request=request)

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()

    assert raised.value.code == "runpod_terminate_ambiguous"
    assert raised.value.ambiguous


@pytest.mark.asyncio
async def test_delete_timeout_with_running_confirmation_remains_ambiguous() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "DELETE":
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json={"desiredStatus": "RUNNING"})

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
async def test_delete_timeout_with_terminated_confirmation_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json={"desiredStatus": "TERMINATED"})

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        result = await adapter.terminate_self()

    assert result.status is RunPodTerminateStatus.TERMINATED


@pytest.mark.asyncio
async def test_delete_read_error_with_inconclusive_confirmation_is_ambiguous() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "DELETE":
            raise httpx.ReadError("connection lost", request=request)
        return httpx.Response(200, json={"desiredStatus": "RUNNING"})

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
async def test_delete_protocol_error_with_running_confirmation_is_ambiguous() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "DELETE":
            raise httpx.RemoteProtocolError("connection lost", request=request)
        return httpx.Response(200, json={"desiredStatus": "RUNNING"})

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
async def test_delete_protocol_error_with_terminated_confirmation_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            raise httpx.RemoteProtocolError("connection lost", request=request)
        return httpx.Response(200, json={"desiredStatus": "TERMINATED"})

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        result = await adapter.terminate_self()

    assert result.status is RunPodTerminateStatus.TERMINATED


@pytest.mark.asyncio
async def test_connection_failure_is_unavailable_without_raw_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret connection detail", request=request)

    async with _client(handler) as client:
        adapter = RunPodLifecycleAdapter(
            client=client,
            env={"RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "secret"},
        )
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()
    assert raised.value.code == "runpod_terminate_unavailable"
    assert "secret connection detail" not in str(raised.value)
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_missing_identity_fails_closed_without_http() -> None:
    async with _client(lambda request: httpx.Response(204)) as client:
        adapter = RunPodLifecycleAdapter(client=client, env={})
        assert not adapter.identity_ready
        with pytest.raises(RunPodTerminateError) as raised:
            await adapter.terminate_self()
    assert raised.value.code == "runpod_identity_missing"
