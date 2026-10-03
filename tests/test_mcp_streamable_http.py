"""
Tests for the Mnemosyne MCP Streamable HTTP transport.

Covers the native MCP ``http`` transport: ``_resolve_http_auth`` (the shared
loopback-only-by-default auth gate), ``_build_streamable_http_app`` (route +
bearer-middleware wiring on top of the ``mcp`` SDK 2.x
``Server.streamable_http_app``), bearer-token rejection over HTTP, the
lifespan requirement for session handling, and CLI plumbing for
``--transport streamable-http`` / ``http``.

Run with: pytest tests/test_mcp_streamable_http.py -v
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from unittest.mock import AsyncMock, patch

import pytest


def _starlette_available() -> bool:
    try:
        import starlette  # noqa: F401
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_mcp_response(response):
    """Decode a streamable-HTTP response by its Content-Type.

    The SDK answers with either ``application/json`` or
    ``text/event-stream``; the JSON-RPC payload must be asserted on either
    way, never on raw body substrings. SSE frames carry the payload on a
    ``data:`` line.
    """
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return response.json()
    if content_type.startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:"):].strip())
        raise AssertionError(f"SSE response had no data: payload: {response.text!r}")
    raise AssertionError(f"unexpected response content-type: {content_type!r}")


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------


class TestResolveHttpAuth:
    """`_resolve_http_auth` is the shared auth gate for HTTP transports."""

    def test_loopback_skips_auth(self, monkeypatch):
        """Default 127.0.0.1 needs no token, no env var."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _resolve_http_auth
        require_auth, token = _resolve_http_auth("127.0.0.1")
        assert require_auth is False
        assert token is None

    def test_loopback_ignores_token_even_if_set(self, monkeypatch):
        """Loopback bind never requires auth regardless of env state."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "some-token")
        from mnemosyne.mcp_server import _resolve_http_auth
        require_auth, token = _resolve_http_auth("localhost")
        assert require_auth is False
        assert token is None

    def test_ipv6_loopback_skips_auth(self, monkeypatch):
        """::1 is loopback too: no token, no auth required."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _resolve_http_auth
        assert _resolve_http_auth("::1") == (False, None)

    def test_ip6_localhost_requires_token(self, monkeypatch):
        """ip6-localhost is fail-closed: the SDK's DNS-rebinding protection
        does not cover the alias, so a bearer token is mandatory."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _resolve_http_auth
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _resolve_http_auth("ip6-localhost")

    def test_ip6_localhost_with_token_requires_auth(self, monkeypatch):
        """With a token set, ip6-localhost is treated like any non-loopback."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "real-secret-123")
        from mnemosyne.mcp_server import _resolve_http_auth
        assert _resolve_http_auth("ip6-localhost") == (True, "real-secret-123")

    def test_uppercase_localhost_requires_token(self, monkeypatch):
        """LOCALHOST is not loopback: case variants are not in the SDK's
        exact tokenless allowlist, so a bearer token is mandatory."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _resolve_http_auth
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _resolve_http_auth("LOCALHOST")

    def test_uppercase_localhost_with_token_requires_auth(self, monkeypatch):
        """With a token set, LOCALHOST is treated like any non-loopback."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "real-secret-123")
        from mnemosyne.mcp_server import _resolve_http_auth
        assert _resolve_http_auth("LOCALHOST") == (True, "real-secret-123")

    def test_non_loopback_without_token_raises(self, monkeypatch):
        """0.0.0.0 with no token must refuse to start. The error message
        names the env var so operators can fix it without grepping."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _resolve_http_auth
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _resolve_http_auth("0.0.0.0")

    def test_non_loopback_empty_token_raises(self, monkeypatch):
        """Empty/whitespace token is treated as unset."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "   ")
        from mnemosyne.mcp_server import _resolve_http_auth
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _resolve_http_auth("0.0.0.0")

    def test_non_loopback_with_token_returns_pair(self, monkeypatch):
        """Properly configured non-loopback returns (True, token)."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "real-secret-123")
        from mnemosyne.mcp_server import _resolve_http_auth
        require_auth, token = _resolve_http_auth("0.0.0.0")
        assert require_auth is True
        assert token == "real-secret-123"

    def test_token_is_stripped(self, monkeypatch):
        """Trailing whitespace in the env var doesn't break auth."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "  with-spaces  ")
        from mnemosyne.mcp_server import _resolve_http_auth
        require_auth, token = _resolve_http_auth("0.0.0.0")
        assert require_auth is True
        assert token == "with-spaces"

    def test_sse_alias_preserved(self, monkeypatch):
        """The pre-streamable-http name still resolves to the same helper."""
        from mnemosyne.mcp_server import _resolve_http_auth, _resolve_sse_auth
        assert _resolve_sse_auth is _resolve_http_auth


# ---------------------------------------------------------------------------
# Host/Origin policy (DNS-rebinding protection)
# ---------------------------------------------------------------------------


class TestResolveTransportSecurity:
    """`_resolve_transport_security` implements the Host/Origin contract."""

    @pytest.fixture(autouse=True)
    def _clean_policy_env(self, monkeypatch):
        """Both policy vars start unset so assertions are environment-independent."""
        monkeypatch.delenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", raising=False)
        monkeypatch.delenv("MNEMOSYNE_MCP_ALLOWED_ORIGINS", raising=False)

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_forms_return_none(self, host, monkeypatch):
        """Loopback forms keep the SDK's built-in DNS-rebinding defaults."""
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "ignored.example.com")
        from mnemosyne.mcp_server import _resolve_transport_security
        assert _resolve_transport_security(host) is None

    def test_non_loopback_without_hosts_raises(self, monkeypatch):
        """Non-loopback is fail-closed: a Host policy is mandatory."""
        from mnemosyne.mcp_server import _resolve_transport_security
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_ALLOWED_HOSTS"):
            _resolve_transport_security("0.0.0.0")

    def test_ip6_localhost_requires_host_policy(self, monkeypatch):
        """ip6-localhost needs an explicit Host policy: the SDK's built-in
        DNS-rebinding protection does not cover the alias."""
        from mnemosyne.mcp_server import _resolve_transport_security
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_ALLOWED_HOSTS"):
            _resolve_transport_security("ip6-localhost")

    def test_uppercase_localhost_requires_host_policy(self, monkeypatch):
        """LOCALHOST needs an explicit Host policy: only the exact lowercase
        value arms the SDK's built-in DNS-rebinding protection."""
        from mnemosyne.mcp_server import _resolve_transport_security
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_ALLOWED_HOSTS"):
            _resolve_transport_security("LOCALHOST")

    def test_origins_alone_do_not_satisfy_host_policy(self, monkeypatch):
        """Origins without a Host allowlist still refuse to start."""
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_ORIGINS", "https://app.example.com")
        from mnemosyne.mcp_server import _resolve_transport_security
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_ALLOWED_HOSTS"):
            _resolve_transport_security("0.0.0.0")

    def test_non_loopback_single_host(self, monkeypatch):
        """A single Host value becomes a one-entry allowlist."""
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "mnemosyne.k.example.com")
        from mnemosyne.mcp_server import _resolve_transport_security
        security = _resolve_transport_security("0.0.0.0")
        assert security.enable_dns_rebinding_protection is True
        assert security.allowed_hosts == ["mnemosyne.k.example.com"]
        assert security.allowed_origins == []

    def test_non_loopback_host_list_trimmed(self, monkeypatch):
        """Comma-separated Hosts are split, trimmed, and empties dropped."""
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_ALLOWED_HOSTS",
            " a.example.com, b.example.com:* ,, c.example.com ",
        )
        from mnemosyne.mcp_server import _resolve_transport_security
        security = _resolve_transport_security("0.0.0.0")
        assert security.allowed_hosts == [
            "a.example.com",
            "b.example.com:*",
            "c.example.com",
        ]

    def test_origins_optional_and_parsed(self, monkeypatch):
        """Origins stay empty by default; when set they are parsed the same way."""
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "mnemosyne.k.example.com:*")
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_ALLOWED_ORIGINS",
            "https://inspector.example.com, https://app.example.com",
        )
        from mnemosyne.mcp_server import _resolve_transport_security
        security = _resolve_transport_security("0.0.0.0")
        assert security.allowed_origins == [
            "https://inspector.example.com",
            "https://app.example.com",
        ]

    def test_parse_csv_env(self, monkeypatch):
        """Single value, list, whitespace, and empties are handled."""
        from mnemosyne.mcp_server import _parse_csv_env
        monkeypatch.delenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", raising=False)
        assert _parse_csv_env("MNEMOSYNE_MCP_ALLOWED_HOSTS") == []
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "only.example.com")
        assert _parse_csv_env("MNEMOSYNE_MCP_ALLOWED_HOSTS") == ["only.example.com"]
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", " a ,, b ,")
        assert _parse_csv_env("MNEMOSYNE_MCP_ALLOWED_HOSTS") == ["a", "b"]


# ---------------------------------------------------------------------------
# App building (Starlette + middleware)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _starlette_available(),
    reason="starlette/mcp not installed -- build_streamable_http_app skipped",
)
class TestBuildStreamableHttpApp:
    """`_build_streamable_http_app` wires route + bearer middleware."""

    def test_loopback_app_has_default_mcp_route(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        app = _build_streamable_http_app(host="127.0.0.1")
        paths = [getattr(route, "path", None) for route in app.routes]
        assert "/mcp" in paths

    def test_custom_path_route(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        app = _build_streamable_http_app(host="127.0.0.1", path="/custom")
        paths = [getattr(route, "path", None) for route in app.routes]
        assert "/custom" in paths
        assert "/mcp" not in paths

    def test_loopback_app_has_no_auth_middleware(self, monkeypatch):
        """Loopback bind: app should not carry the bearer middleware."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _BearerTokenMiddleware, _build_streamable_http_app
        app = _build_streamable_http_app(host="127.0.0.1")
        middleware_classes = [m.cls for m in app.user_middleware]
        assert not any(c is _BearerTokenMiddleware for c in middleware_classes), (
            "loopback app should not install bearer middleware"
        )

    def test_non_loopback_without_token_raises(self, monkeypatch):
        """0.0.0.0 with no token: build refuses (mirrors _resolve_http_auth)."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _build_streamable_http_app(host="0.0.0.0")

    def test_non_loopback_with_token_installs_middleware(self, monkeypatch):
        """0.0.0.0 with token + host policy: app carries the bearer middleware."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "localhost:*")
        from mnemosyne.mcp_server import _build_streamable_http_app
        app = _build_streamable_http_app(host="0.0.0.0")
        names = [m.cls.__name__ for m in app.user_middleware]
        assert any("Bearer" in n for n in names), (
            f"non-loopback app should install bearer middleware, got: {names}"
        )

    def test_builder_forwards_sdk_kwargs(self, monkeypatch):
        """The SDK app builder receives path/json_response/host untouched."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from starlette.applications import Starlette
        from mnemosyne import mcp_server

        captured = {}

        class _FakeServer:
            def streamable_http_app(self, **kwargs):
                captured.update(kwargs)
                return Starlette(routes=[])

        with patch.object(mcp_server, "_build_mcp_server", return_value=_FakeServer()):
            mcp_server._build_streamable_http_app(
                host="127.0.0.1", path="/custom", json_response=True
            )

        assert captured == {
            "streamable_http_path": "/custom",
            "json_response": True,
            "host": "127.0.0.1",
            "transport_security": None,
        }

    def test_builder_uses_module_level_bearer_middleware(self, monkeypatch):
        """Auth wiring reuses the shared _BearerTokenMiddleware class."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "localhost:*")
        from mnemosyne.mcp_server import _BearerTokenMiddleware, _build_streamable_http_app
        app = _build_streamable_http_app(host="0.0.0.0")
        middleware_classes = [m.cls for m in app.user_middleware]
        assert any(c is _BearerTokenMiddleware for c in middleware_classes)

    def test_non_loopback_without_hosts_raises(self, monkeypatch):
        """0.0.0.0 with a token but no Host policy refuses to start."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.delenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_ALLOWED_HOSTS"):
            _build_streamable_http_app(host="0.0.0.0")

    def test_ip6_localhost_build_fails_closed(self, monkeypatch):
        """ip6-localhost cannot be brought up tokenlessly.

        The build refuses without a token, and once configured with a token
        and Host policy it installs the bearer middleware -- so the alias can
        never serve an arbitrary Host/Origin request unauthenticated.
        """
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        monkeypatch.delenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _build_streamable_http_app(host="ip6-localhost")

        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "localhost:*")
        app = _build_streamable_http_app(host="ip6-localhost")
        names = [m.cls.__name__ for m in app.user_middleware]
        assert any("Bearer" in n for n in names), (
            f"ip6-localhost app should install bearer middleware, got: {names}"
        )

    def test_uppercase_localhost_build_fails_closed(self, monkeypatch):
        """LOCALHOST cannot be brought up tokenlessly, mirroring ip6-localhost.

        The build refuses without a token, and once configured with a token
        and Host policy it installs the bearer middleware -- so the case
        variant can never serve an arbitrary Host/Origin request
        unauthenticated.
        """
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        monkeypatch.delenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _build_streamable_http_app(host="LOCALHOST")

        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "localhost:*")
        app = _build_streamable_http_app(host="LOCALHOST")
        names = [m.cls.__name__ for m in app.user_middleware]
        assert any("Bearer" in n for n in names), (
            f"LOCALHOST app should install bearer middleware, got: {names}"
        )

    def test_builder_forwards_transport_security(self, monkeypatch):
        """Non-loopback forwards the operator's Host/Origin policy to the SDK."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_ALLOWED_HOSTS", " a.example.com , b.example.com:* "
        )
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_ORIGINS", "https://inspector.example.com")
        from starlette.applications import Starlette
        from mnemosyne import mcp_server

        captured = {}

        class _FakeServer:
            def streamable_http_app(self, **kwargs):
                captured.update(kwargs)
                return Starlette(routes=[])

        with patch.object(mcp_server, "_build_mcp_server", return_value=_FakeServer()):
            mcp_server._build_streamable_http_app(host="0.0.0.0")

        security = captured["transport_security"]
        assert security is not None
        assert security.enable_dns_rebinding_protection is True
        assert security.allowed_hosts == ["a.example.com", "b.example.com:*"]
        assert security.allowed_origins == ["https://inspector.example.com"]


# ---------------------------------------------------------------------------
# Bearer-token rejection over HTTP
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _starlette_available(),
    reason="starlette/mcp not installed -- bearer rejection skipped",
)
class TestStreamableHttpBearerRejection:
    """A token-authed streamable HTTP app rejects unauthorized POSTs with 401."""

    @pytest.fixture
    def authed_app(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "localhost:*")
        from mnemosyne.mcp_server import _build_streamable_http_app
        return _build_streamable_http_app(host="0.0.0.0")

    def test_missing_token_rejected(self, authed_app):
        from starlette.testclient import TestClient
        resp = TestClient(authed_app).post("/mcp", json={"ping": "pong"})
        assert resp.status_code == 401
        assert "missing bearer token" in resp.json().get("error", "").lower()

    def test_wrong_token_rejected(self, authed_app):
        from starlette.testclient import TestClient
        resp = TestClient(authed_app).post(
            "/mcp",
            json={"ping": "pong"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        assert "invalid bearer token" in resp.json().get("error", "").lower()

    def test_malformed_header_rejected(self, authed_app):
        """Token without 'Bearer ' prefix is rejected as missing."""
        from starlette.testclient import TestClient
        resp = TestClient(authed_app).post(
            "/mcp",
            json={"ping": "pong"},
            headers={"Authorization": "Basic c3VwZXJzZWNyZXQ="},  # not Bearer
        )
        assert resp.status_code == 401

    def test_401_response_has_www_authenticate_header(self, authed_app):
        """Per RFC 7235, 401 should advertise the auth scheme."""
        from starlette.testclient import TestClient
        resp = TestClient(authed_app).post("/mcp", json={})
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_valid_bearer_completes_initialize_and_tools_list(self, authed_app):
        """A valid bearer completes MCP operations behind the middleware.

        Drives the full token-gated non-loopback app through an initialize
        handshake and a tools/list call, decoding each response by its
        Content-Type and asserting the JSON-RPC id, successful result shape,
        serverInfo, and a non-empty tools list. Then exercises the
        authenticated GET session stream and proves DELETE terminates the
        session: a later request on it is rejected with 404. The rejection
        tests above are the isolation control proving the middleware is what
        blocks unauthorized requests.
        """
        from starlette.testclient import TestClient

        headers = {
            "Authorization": "Bearer supersecret",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26",
        }
        # One lifespan per app instance: the session manager's run() can only
        # be entered once, so the handshake and the follow-ups share the block.
        with TestClient(authed_app, base_url="http://localhost:8080") as client:
            init = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "e2e", "version": "0"},
                    },
                },
                headers=headers,
            )
            assert init.status_code == 200
            init_body = _parse_mcp_response(init)
            assert init_body["jsonrpc"] == "2.0"
            assert init_body["id"] == 1
            assert "result" in init_body and "error" not in init_body
            assert init_body["result"]["protocolVersion"] == "2025-03-26"
            assert init_body["result"]["serverInfo"]["name"] == "mnemosyne"

            session_id = init.headers.get("mcp-session-id")
            assert session_id, "initialize must mint a session id"

            listed = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 2},
                headers={**headers, "Mcp-Session-Id": session_id},
            )
            assert listed.status_code == 200
            listed_body = _parse_mcp_response(listed)
            assert listed_body["jsonrpc"] == "2.0"
            assert listed_body["id"] == 2
            assert "result" in listed_body and "error" not in listed_body
            tools = listed_body["result"]["tools"]
            assert isinstance(tools, list) and tools, (
                "tools/list must return a non-empty tools list"
            )

            # The authenticated GET opens the session's SSE stream. The
            # stream is long-lived, so run it as a portal task and wait for the
            # response start before terminating the session with DELETE.
            portal = client.portal
            get_started = threading.Event()
            get_result = {}

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/mcp",
                "raw_path": b"/mcp",
                "query_string": b"",
                "root_path": "",
                "headers": [
                    (b"host", b"localhost:8080"),
                    (b"authorization", b"Bearer supersecret"),
                    (b"accept", b"text/event-stream"),
                    (b"mcp-session-id", session_id.encode()),
                ],
                "client": ("testclient", 50000),
                "server": ("localhost", 8080),
            }

            # ASGI requires the server to deliver http.request before the
            # app can respond, even for a bodyless GET. mcp 2.0.0 answered
            # without waiting, so an earlier version of this driver skipped
            # the message and still passed. mcp 2.1.0 reads the body to
            # enforce max_request_body_size (SDK #3336), so skipping it wedges
            # the handler before http.response.start and hangs the run. Real
            # servers always send it; only a hand-driven scope can omit it.
            request_delivered = threading.Event()

            async def receive():
                if not request_delivered.is_set():
                    request_delivered.set()
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": False,
                    }
                # The stream is long-lived: after the request is delivered
                # there is nothing more to feed it, so block until cancelled.
                while True:
                    await asyncio.sleep(3600)

            async def send(message):
                if message["type"] == "http.response.start":
                    get_result["status"] = message["status"]
                    get_result["content_type"] = dict(
                        (k.decode(), v.decode())
                        for k, v in message.get("headers", [])
                    ).get("content-type")
                    get_started.set()

            # start_task_soon runs the ASGI call inside a cancel scope owned
            # by the portal, so the future can cancel it deterministically.
            get_future = portal.start_task_soon(authed_app, scope, receive, send)
            # A task that ends without ever sending http.response.start (an
            # error, say) must still release the wait below.
            get_future.add_done_callback(lambda _f: get_started.set())

            try:
                assert get_started.wait(10), (
                    "authenticated GET stream never started"
                )
                assert get_result["status"] == 200
                assert (get_result.get("content_type") or "").startswith(
                    "text/event-stream"
                )

                # DELETE terminates the session (SDK 2.0.0 answers 200 with a
                # JSON body)...
                deleted = client.delete(
                    "/mcp", headers={**headers, "Mcp-Session-Id": session_id}
                )
                assert deleted.status_code == 200
                assert deleted.headers["content-type"].startswith(
                    "application/json"
                )

                concurrent.futures.wait([get_future], timeout=10)
                assert get_future.done(), "GET stream did not close after DELETE"

                # ...and any later request on the terminated session is
                # rejected.
                stale = client.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "method": "ping", "id": 3},
                    headers={**headers, "Mcp-Session-Id": session_id},
                )
                assert stale.status_code == 404
            finally:
                # The ASGI task holds a slot in the portal's task group, and
                # TestClient.__exit__ waits for that group to drain. If the
                # stream is still open here (DELETE raced, or any assertion
                # above failed) the exit blocks forever and the whole run
                # hangs instead of reporting. Cancelling is a no-op once the
                # task has finished, so this is safe on the happy path too.
                get_future.cancel()

    def _drive_middleware(self, header_value: bytes):
        """Run the bearer middleware directly against a stub downstream app.

        Driving the ASGI callable directly keeps positive/negative credential
        coverage deterministic (no HTTP client header-encoding in the way).
        """
        from mnemosyne.mcp_server import _BearerTokenMiddleware

        async def _run():
            reached = {"ok": False}

            async def downstream(scope, receive, send):
                reached["ok"] = True

            async def receive():
                return {}

            responses = []

            async def send(message):
                responses.append(message)

            middleware = _BearerTokenMiddleware(downstream, token="supersecret")
            scope = {"type": "http", "headers": [(b"authorization", header_value)]}
            await middleware(scope, receive, send)
            return reached, responses

        return asyncio.run(_run())

    def test_valid_credential_is_forwarded(self):
        """The correct credential is not rejected."""
        reached, responses = self._drive_middleware(b"Bearer supersecret")
        assert reached["ok"] is True
        assert responses == []

    def test_lowercase_bearer_scheme_accepted(self):
        """RFC 6750 auth-scheme tokens are case-insensitive."""
        reached, responses = self._drive_middleware(b"bearer supersecret")
        assert reached["ok"] is True
        assert responses == []

    def test_non_ascii_credential_returns_401(self):
        """A non-ASCII credential yields 401, not a 500 from compare_digest."""
        reached, responses = self._drive_middleware(b"Bearer \xff")
        assert reached["ok"] is False
        start = next(m for m in responses if m["type"] == "http.response.start")
        assert start["status"] == 401


# ---------------------------------------------------------------------------
# Lifecycle: the session manager needs the Starlette lifespan to run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _starlette_available(),
    reason="starlette/mcp not installed -- lifecycle smoke skipped",
)
class TestStreamableHttpLifecycle:
    """A streamable HTTP app only serves requests inside its lifespan."""

    def test_initialize_returns_200_inside_lifespan(self, monkeypatch):
        """POST /mcp initialize works once the lifespan/task group is up."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        from starlette.testclient import TestClient

        app = _build_streamable_http_app(host="127.0.0.1")
        # base_url keeps the Host header inside the SDK's auto-enabled
        # DNS-rebinding allow-list for loopback binds.
        with TestClient(app, base_url="http://localhost:8080") as client:
            resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "smoke", "version": "0"},
                    },
                },
                headers={
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-03-26",
                },
            )
        assert resp.status_code == 200
        assert "serverInfo" in resp.text

    def test_request_outside_lifespan_raises_task_group_error(self, monkeypatch):
        """Without the lifespan, the SDK raises its task-group guard.

        Documents why end-to-end tests must enter the TestClient context
        manager: the session manager only services requests after run().
        """
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        from starlette.testclient import TestClient

        app = _build_streamable_http_app(host="127.0.0.1")
        client = TestClient(app, base_url="http://localhost:8080")  # no context manager
        with pytest.raises(RuntimeError, match="Task group is not initialized"):
            client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                headers={"Accept": "application/json, text/event-stream"},
            )

    def test_non_allowed_host_rejected_421(self, monkeypatch):
        """Loopback builds enable DNS-rebinding protection: bad Host -> 421."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_streamable_http_app
        from starlette.testclient import TestClient

        app = _build_streamable_http_app(host="127.0.0.1")
        # Default TestClient base_url uses host "testserver", which is not in
        # the loopback allow-list, so the SDK rejects it before session handling.
        with TestClient(app) as client:
            resp = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                headers={"Accept": "application/json, text/event-stream"},
            )
        assert resp.status_code == 421

    def test_non_loopback_disallowed_host_rejected_421(self, monkeypatch):
        """A valid bearer still gets 421 when the Host is not allowlisted."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "mnemosyne.k.example.com:*")
        from mnemosyne.mcp_server import _build_streamable_http_app
        from starlette.testclient import TestClient

        app = _build_streamable_http_app(host="0.0.0.0")
        with TestClient(app, base_url="http://evil.example.com:8080") as client:
            resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "e2e", "version": "0"},
                    },
                },
                headers={
                    "Authorization": "Bearer supersecret",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-03-26",
                },
            )
        assert resp.status_code == 421

    def test_non_loopback_disallowed_origin_rejected_403(self, monkeypatch):
        """A browser Origin outside the allowlist is rejected with 403."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "mnemosyne.k.example.com:*")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_ORIGINS", "https://inspector.example.com")
        from mnemosyne.mcp_server import _build_streamable_http_app
        from starlette.testclient import TestClient

        app = _build_streamable_http_app(host="0.0.0.0")
        with TestClient(app, base_url="http://mnemosyne.k.example.com:8080") as client:
            resp = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                headers={
                    "Authorization": "Bearer supersecret",
                    "Accept": "application/json, text/event-stream",
                    "Origin": "https://evil.example.com",
                },
            )
        assert resp.status_code == 403

    def test_non_loopback_allowed_host_and_origin_succeed(self, monkeypatch):
        """An allowlisted browser Origin completes an MCP operation."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "mnemosyne.k.example.com:*")
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_ORIGINS", "https://inspector.example.com")
        from mnemosyne.mcp_server import _build_streamable_http_app
        from starlette.testclient import TestClient

        app = _build_streamable_http_app(host="0.0.0.0")
        with TestClient(app, base_url="http://mnemosyne.k.example.com:8080") as client:
            resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "e2e", "version": "0"},
                    },
                },
                headers={
                    "Authorization": "Bearer supersecret",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-03-26",
                    "Origin": "https://inspector.example.com",
                },
            )
        assert resp.status_code == 200
        assert "serverInfo" in resp.text


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


class TestStreamableHttpCli:
    """`mnemosyne mcp --transport streamable-http ...` forwards args."""

    def test_main_forwards_streamable_http_args(self):
        from mnemosyne.mcp_server import main
        with patch("mnemosyne.mcp_server.run_mcp_server") as runner:
            main(["--transport", "streamable-http", "--port", "19090", "--bank", "work"])
        runner.assert_called_once_with(
            transport="streamable-http",
            port=19090,
            bank="work",
            host="127.0.0.1",
            path="/mcp",
            json_response=False,
        )

    def test_main_forwards_path_and_json_response(self):
        from mnemosyne.mcp_server import main
        with patch("mnemosyne.mcp_server.run_mcp_server") as runner:
            main([
                "--transport", "streamable-http",
                "--path", "/custom",
                "--json-response",
                "--port", "19090",
            ])
        runner.assert_called_once_with(
            transport="streamable-http",
            port=19090,
            bank=None,
            host="127.0.0.1",
            path="/custom",
            json_response=True,
        )

    def test_main_accepts_http_alias(self):
        from mnemosyne.mcp_server import main
        with patch("mnemosyne.mcp_server.run_mcp_server") as runner:
            main(["--transport", "http", "--port", "9000"])
        runner.assert_called_once_with(
            transport="http",
            port=9000,
            bank=None,
            host="127.0.0.1",
            path="/mcp",
            json_response=False,
        )

    def test_run_mcp_server_http_alias_dispatches(self, monkeypatch):
        """transport='http' maps to the streamable HTTP runner."""
        from mnemosyne import mcp_server

        run_streamable = AsyncMock()
        monkeypatch.setattr(mcp_server, "_run_streamable_http", run_streamable)
        mcp_server.run_mcp_server(
            transport="http",
            port=9000,
            host="0.0.0.0",
            path="/x",
            json_response=True,
        )
        run_streamable.assert_awaited_once_with(
            port=9000, host="0.0.0.0", path="/x", json_response=True
        )

    def test_run_mcp_server_unknown_transport_raises(self):
        from mnemosyne.mcp_server import run_mcp_server
        with pytest.raises(ValueError, match="streamable-http"):
            run_mcp_server(transport="bogus")

    def test_run_streamable_http_default_host_is_loopback(self):
        """_run_streamable_http() default kwarg pins 127.0.0.1."""
        import inspect
        from mnemosyne.mcp_server import _run_streamable_http
        sig = inspect.signature(_run_streamable_http)
        assert sig.parameters["host"].default == "127.0.0.1"
        assert sig.parameters["path"].default == "/mcp"
