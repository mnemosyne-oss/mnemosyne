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
from unittest.mock import AsyncMock, patch

import pytest


def _starlette_available() -> bool:
    try:
        import starlette  # noqa: F401
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


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

    def test_loopback_returns_none(self, monkeypatch):
        """Loopback keeps the SDK's built-in DNS-rebinding defaults."""
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "ignored.example.com")
        from mnemosyne.mcp_server import _resolve_transport_security
        assert _resolve_transport_security("127.0.0.1") is None

    def test_non_loopback_without_hosts_raises(self, monkeypatch):
        """Non-loopback is fail-closed: a Host policy is mandatory."""
        monkeypatch.delenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", raising=False)
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
        handshake and a tools/list call, asserting JSON-RPC success plus the
        operation result. The rejection tests above are the isolation control
        proving the middleware is what blocks unauthorized requests.
        """
        from starlette.testclient import TestClient

        headers = {
            "Authorization": "Bearer supersecret",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26",
        }
        # One lifespan per app instance: the session manager's run() can only
        # be entered once, so the handshake and the follow-up share the block.
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
            session_id = init.headers.get("mcp-session-id")
            assert session_id, "initialize must mint a session id"
            assert "serverInfo" in init.text

            listed = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 2},
                headers={**headers, "Mcp-Session-Id": session_id},
            )
        assert listed.status_code == 200
        assert "tools" in listed.text

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
