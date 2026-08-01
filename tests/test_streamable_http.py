"""Regression tests for the Streamable HTTP MCP transport."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(dependency) is None for dependency in ("mcp", "starlette", "httpx")),
    reason="Streamable HTTP tests require MCP, Starlette, and HTTPX",
)


def _mcp_headers() -> dict[str, str]:
    return {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }


def _initialize_request() -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1"},
        },
    }


@asynccontextmanager
async def _streamable_http_client(app):
    import httpx

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
        ) as client:
            yield client


def _get_scope(session_id: str) -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1:8000"),
            (b"accept", b"text/event-stream"),
            (b"mcp-session-id", session_id.encode("latin-1")),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
        "root_path": "",
    }


async def _capture_response_start(app, scope: dict[str, object]) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    response_started = asyncio.Event()
    request_sent = False
    disconnected = asyncio.Event()

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)
        if message["type"] == "http.response.start":
            response_started.set()

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(response_started.wait(), timeout=1)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    return messages


def test_streamable_http_initializes_lists_tools_calls_stats_and_terminates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A Streamable HTTP session handles the standard MCP lifecycle at /mcp."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = _build_streamable_http_app(host="127.0.0.1")

    async def exercise() -> None:
        async with _streamable_http_client(app) as client:
            response = await client.post("/mcp", headers=_mcp_headers(), json=_initialize_request())
            assert response.status_code == 200
            session_id = response.headers["mcp-session-id"]
            assert response.headers["content-type"].startswith("text/event-stream")

            session_headers = {**_mcp_headers(), "Mcp-Session-Id": session_id}
            initialized = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
            response = await client.post("/mcp", headers=session_headers, json=initialized)
            assert response.status_code == 202

            tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            response = await client.post("/mcp", headers=session_headers, json=tools)
            assert response.status_code == 200
            assert "mnemosyne_stats" in response.text

            stats = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "mnemosyne_stats", "arguments": {}},
            }
            response = await client.post("/mcp", headers=session_headers, json=stats)
            assert response.status_code == 200
            assert "mnemosyne" in response.text

            assert (await client.delete("/mcp", headers=session_headers)).status_code == 200
            assert (await client.post("/mcp", headers=session_headers, json=tools)).status_code == 404

    asyncio.run(exercise())


def test_streamable_http_non_loopback_requires_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network-exposed Streamable HTTP follows the existing SSE token policy."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
        _build_streamable_http_app(host="0.0.0.0")


def test_streamable_http_get_accepts_initialized_session() -> None:
    """An initialized session accepts a server-initiated GET SSE stream."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    app = _build_streamable_http_app(host="127.0.0.1")

    async def exercise() -> list[dict[str, object]]:
        async with _streamable_http_client(app) as client:
            response = await client.post("/mcp", headers=_mcp_headers(), json=_initialize_request())
            assert response.status_code == 200
            return await _capture_response_start(app, _get_scope(response.headers["mcp-session-id"]))

    messages = asyncio.run(exercise())
    response_start = next(message for message in messages if message["type"] == "http.response.start")
    assert response_start["status"] == 200
    headers = dict(response_start["headers"])
    assert headers[b"content-type"].startswith(b"text/event-stream")


def test_streamable_http_delete_removes_session_from_manager() -> None:
    """Explicit DELETE removes the terminated session from SDK registries."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    app = _build_streamable_http_app(host="127.0.0.1")

    async def exercise() -> None:
        async with _streamable_http_client(app) as client:
            response = await client.post("/mcp", headers=_mcp_headers(), json=_initialize_request())
            assert response.status_code == 200
            session_id = response.headers["mcp-session-id"]
            manager = app.state.streamable_http_manager
            assert session_id in manager._server_instances
            assert manager.session_idle_timeout == 1800

            response = await client.delete(
                "/mcp", headers={**_mcp_headers(), "Mcp-Session-Id": session_id}
            )
            assert response.status_code == 200

            assert session_id not in manager._server_instances
            assert session_id not in manager._session_owners

    asyncio.run(exercise())


def test_streamable_http_idle_session_is_reaped() -> None:
    """An idle Streamable HTTP session expires and then returns 404."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    app = _build_streamable_http_app(host="127.0.0.1", session_idle_timeout=0.05)

    async def exercise() -> object:
        async with _streamable_http_client(app) as client:
            response = await client.post("/mcp", headers=_mcp_headers(), json=_initialize_request())
            assert response.status_code == 200
            session_id = response.headers["mcp-session-id"]
            session_headers = {**_mcp_headers(), "Mcp-Session-Id": session_id}
            manager = app.state.streamable_http_manager
            assert session_id in manager._server_instances
            # Loopback requests are anonymous, so no session owner is stored.
            tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            deadline = asyncio.get_running_loop().time() + 2
            while session_id in manager._server_instances:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("idle session was not reaped")
                await asyncio.sleep(0.01)

            assert session_id not in manager._server_instances
            assert session_id not in manager._session_owners
            return await client.post("/mcp", headers=session_headers, json=tools)

    response = asyncio.run(exercise())
    assert response is not None
    assert response.status_code == 404


def test_streamable_http_loopback_rejects_dns_rebinding_headers() -> None:
    """Loopback Streamable HTTP rejects hostile Host and Origin headers."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    app = _build_streamable_http_app(host="127.0.0.1")

    async def exercise() -> tuple[object, object]:
        async with _streamable_http_client(app) as client:
            invalid_host = await client.post(
                "/mcp",
                headers={
                    **_mcp_headers(),
                    "Host": "evil.example",
                    "Origin": "http://127.0.0.1:8000",
                },
                json=_initialize_request(),
            )
            invalid_origin = await client.post(
                "/mcp",
                headers={
                    **_mcp_headers(),
                    "Host": "127.0.0.1:8000",
                    "Origin": "http://evil.example",
                },
                json=_initialize_request(),
            )
            return invalid_host, invalid_origin

    invalid_host, invalid_origin = asyncio.run(exercise())
    assert invalid_host.status_code == 421
    assert invalid_origin.status_code == 403


def test_streamable_http_non_loopback_rejects_missing_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared pure-ASGI middleware protects the /mcp route."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "test-token")
    app = _build_streamable_http_app(host="0.0.0.0")

    async def exercise():
        async with _streamable_http_client(app) as client:
            return await client.post("/mcp", json={})

    response = asyncio.run(exercise())
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"error": "missing bearer token"}


def test_streamable_http_non_loopback_rejects_invalid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty but incorrect bearer token is rejected."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "test-token")
    app = _build_streamable_http_app(host="0.0.0.0")

    async def exercise():
        async with _streamable_http_client(app) as client:
            return await client.post(
                "/mcp",
                headers={"Authorization": "Bearer wrong-token"},
                json={},
            )

    response = asyncio.run(exercise())
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"error": "invalid bearer token"}


def test_streamable_http_non_loopback_accepts_valid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid bearer token initializes and operates an MCP session."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "test-token")
    app = _build_streamable_http_app(host="0.0.0.0")

    async def exercise() -> tuple[object, object]:
        async with _streamable_http_client(app) as client:
            headers = {**_mcp_headers(), "Authorization": "bearer test-token"}
            initialized = await client.post(
                "/mcp", headers=headers, json=_initialize_request()
            )
            session_id = initialized.headers["mcp-session-id"]
            tools = await client.post(
                "/mcp",
                headers={**headers, "Mcp-Session-Id": session_id},
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            return initialized, tools

    initialized, tools = asyncio.run(exercise())
    assert initialized.status_code == 200
    assert initialized.headers["mcp-session-id"]
    assert tools.status_code == 200
    assert "mnemosyne_stats" in tools.text


def test_streamable_http_rejects_unknown_session_id() -> None:
    """Requests carrying a nonexistent MCP session ID return 404."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    app = _build_streamable_http_app(host="127.0.0.1")

    async def exercise():
        async with _streamable_http_client(app) as client:
            return await client.post(
                "/mcp",
                headers={**_mcp_headers(), "Mcp-Session-Id": "does-not-exist"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )

    response = asyncio.run(exercise())
    assert response.status_code == 404


def test_streamable_http_initializations_receive_independent_session_ids() -> None:
    """The SDK manager creates an independent transport for every MCP session."""
    from mnemosyne.mcp_server import _build_streamable_http_app

    app = _build_streamable_http_app(host="127.0.0.1")

    async def exercise():
        async with _streamable_http_client(app) as client:
            return (
                await client.post("/mcp", headers=_mcp_headers(), json=_initialize_request()),
                await client.post("/mcp", headers=_mcp_headers(), json=_initialize_request()),
            )

    first, second = asyncio.run(exercise())
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["mcp-session-id"] != second.headers["mcp-session-id"]
