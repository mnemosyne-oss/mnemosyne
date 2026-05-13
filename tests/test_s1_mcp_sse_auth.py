"""
Regression tests for S1 (security audit, 2026-05-12):

    MCP SSE transport binds 127.0.0.1 by default; binding to a non-loopback
    host requires MNEMOSYNE_MCP_TOKEN and installs a bearer-token middleware.

Pre-fix: `mnemosyne mcp --transport sse` bound `0.0.0.0` with no auth, so
anyone on the same LAN could call /sse and /messages and read/write/delete
the user's memory store. This file locks the hardened defaults in.

Run with: pytest tests/test_s1_mcp_sse_auth.py -v
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers under direct test
# ---------------------------------------------------------------------------


class TestIsLoopback:
    """`_is_loopback` decides whether a host bind needs auth."""

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "localhost", "::1", "ip6-localhost",
         "LOCALHOST", "  127.0.0.1  ", "LocalHost"],
    )
    def test_loopback_aliases(self, host):
        from mnemosyne.mcp_server import _is_loopback
        assert _is_loopback(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "192.168.1.10", "10.0.0.5", "::",
         "example.com", "fd00::1"],
    )
    def test_non_loopback(self, host):
        from mnemosyne.mcp_server import _is_loopback
        assert _is_loopback(host) is False


class TestResolveSseAuth:
    """`_resolve_sse_auth` is the gate that enforces the hardened policy."""

    def test_loopback_skips_auth(self, monkeypatch):
        """Default 127.0.0.1 needs no token, no env var."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("127.0.0.1")
        assert require_auth is False
        assert token is None

    def test_loopback_ignores_token_even_if_set(self, monkeypatch):
        """Loopback bind never requires auth regardless of env state."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "some-token")
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("localhost")
        assert require_auth is False
        assert token is None

    def test_non_loopback_without_token_raises(self, monkeypatch):
        """0.0.0.0 with no token must refuse to start. The error message
        names the env var so operators can fix it without grepping."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _resolve_sse_auth("0.0.0.0")

    def test_non_loopback_empty_token_raises(self, monkeypatch):
        """Empty/whitespace token is treated as unset."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "   ")
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _resolve_sse_auth("0.0.0.0")

    def test_non_loopback_with_token_returns_pair(self, monkeypatch):
        """Properly configured non-loopback returns (True, token)."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "real-secret-123")
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("0.0.0.0")
        assert require_auth is True
        assert token == "real-secret-123"

    def test_token_is_stripped(self, monkeypatch):
        """Trailing whitespace in the env var doesn't break auth."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "  with-spaces  ")
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("0.0.0.0")
        assert token == "with-spaces"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestMainHostArg:
    """`mnemosyne mcp` CLI: --host flag plumbs through, default is loopback."""

    def test_default_host_is_loopback(self):
        """Calling main() without --host should pass host='127.0.0.1'."""
        from mnemosyne.mcp_server import main
        with patch("mnemosyne.mcp_server.run_mcp_server") as runner:
            main(["--transport", "sse", "--port", "9000"])
        runner.assert_called_once_with(
            transport="sse", port=9000, bank=None, host="127.0.0.1"
        )

    def test_explicit_host_arg(self):
        """--host 0.0.0.0 must be threaded through."""
        from mnemosyne.mcp_server import main
        with patch("mnemosyne.mcp_server.run_mcp_server") as runner:
            main(["--transport", "sse", "--host", "0.0.0.0", "--port", "9001"])
        runner.assert_called_once_with(
            transport="sse", port=9001, bank=None, host="0.0.0.0"
        )

    def test_run_mcp_server_default_host_is_loopback(self):
        """run_mcp_server() default kwarg pins 127.0.0.1."""
        import inspect
        from mnemosyne.mcp_server import run_mcp_server
        sig = inspect.signature(run_mcp_server)
        assert sig.parameters["host"].default == "127.0.0.1"

    def test_run_sse_default_host_is_loopback(self):
        """_run_sse() default kwarg pins 127.0.0.1 as a second line of defense."""
        import inspect
        from mnemosyne.mcp_server import _run_sse
        sig = inspect.signature(_run_sse)
        assert sig.parameters["host"].default == "127.0.0.1"


# ---------------------------------------------------------------------------
# App building (Starlette + middleware)
# ---------------------------------------------------------------------------


def _starlette_available() -> bool:
    try:
        import starlette  # noqa: F401
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _starlette_available(),
    reason="starlette/mcp not installed -- build_sse_app skipped",
)
class TestBuildSseApp:
    """`_build_sse_app` is the integration point: auth gate + middleware install."""

    def test_loopback_app_has_no_auth_middleware(self, monkeypatch):
        """Loopback bind: app should not carry the bearer middleware."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_sse_app
        app = _build_sse_app(host="127.0.0.1")
        # Starlette stores user-supplied middleware on user_middleware.
        # We just check that none of them is our bearer-token class.
        names = [type(m.cls).__name__ if hasattr(m, "cls") else str(m)
                 for m in app.user_middleware]
        assert not any("Bearer" in n for n in names), (
            f"loopback app should not have bearer middleware, got: {names}"
        )

    def test_non_loopback_without_token_raises(self, monkeypatch):
        """0.0.0.0 with no token: build refuses (mirrors _resolve_sse_auth)."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_sse_app
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKEN"):
            _build_sse_app(host="0.0.0.0")

    def test_non_loopback_with_token_installs_middleware(self, monkeypatch):
        """0.0.0.0 with token: app carries the bearer middleware."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        from mnemosyne.mcp_server import _build_sse_app
        app = _build_sse_app(host="0.0.0.0")
        # At least one middleware entry should be the bearer wrapper.
        middleware_classes = [m.cls for m in app.user_middleware]
        # The inner class is defined locally inside _build_sse_app so we
        # match by class name rather than identity.
        names = [c.__name__ for c in middleware_classes]
        assert any("Bearer" in n for n in names), (
            f"non-loopback app should install bearer middleware, got: {names}"
        )

    def test_bearer_middleware_rejects_missing_token(self, monkeypatch):
        """End-to-end: TestClient hitting /sse without Authorization gets 401."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")
        client = TestClient(app)
        # POST to /messages without auth header
        resp = client.post("/messages", json={"ping": "pong"})
        assert resp.status_code == 401
        body = resp.json()
        assert "missing bearer token" in body.get("error", "").lower()

    def test_bearer_middleware_rejects_wrong_token(self, monkeypatch):
        """Wrong token: 401 (compare via hmac.compare_digest in production)."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")
        client = TestClient(app)
        resp = client.post(
            "/messages",
            json={"ping": "pong"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert "invalid bearer token" in body.get("error", "").lower()

    def test_bearer_middleware_rejects_malformed_header(self, monkeypatch):
        """Token without 'Bearer ' prefix is rejected as missing."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")
        client = TestClient(app)
        resp = client.post(
            "/messages",
            json={"ping": "pong"},
            headers={"Authorization": "Basic c3VwZXJzZWNyZXQ="},  # not Bearer
        )
        assert resp.status_code == 401

    def test_401_response_has_www_authenticate_header(self, monkeypatch):
        """Per RFC 7235, 401 should advertise the auth scheme."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")
        client = TestClient(app)
        resp = client.post("/messages", json={})
        assert resp.headers.get("www-authenticate") == "Bearer"
