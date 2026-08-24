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

import asyncio
import time
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
        ["127.0.0.1", "localhost", "::1"],
    )
    def test_loopback_aliases(self, host):
        from mnemosyne.mcp_server import _is_loopback
        assert _is_loopback(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "192.168.1.10", "10.0.0.5", "::",
         "example.com", "fd00::1", "ip6-localhost",
         "LOCALHOST", "  127.0.0.1  ", "LocalHost"],
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
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKENS", raising=False)
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "real-secret-123")
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("0.0.0.0")
        assert require_auth is True
        assert token == "real-secret-123"

    def test_token_is_stripped(self, monkeypatch):
        """Trailing whitespace in the env var doesn't break auth."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKENS", raising=False)
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
            transport="sse", port=9000, bank=None, host="127.0.0.1",
            path="/mcp", json_response=False,
        )

    def test_explicit_host_arg(self):
        """--host 0.0.0.0 must be threaded through."""
        from mnemosyne.mcp_server import main
        with patch("mnemosyne.mcp_server.run_mcp_server") as runner:
            main(["--transport", "sse", "--host", "0.0.0.0", "--port", "9001"])
        runner.assert_called_once_with(
            transport="sse", port=9001, bank=None, host="0.0.0.0",
            path="/mcp", json_response=False,
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


def _call_asgi_app(app, *, path: str, method: str, authorization: bytes):
    """Call an ASGI app with raw header bytes and return emitted messages."""
    messages = []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", authorization)],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    return messages


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

    @pytest.mark.parametrize(
        ("path", "method"),
        [("/sse", "GET"), ("/messages/", "POST")],
    )
    def test_bearer_middleware_rejects_non_ascii_token_without_500(
        self, monkeypatch, path, method
    ):
        """Non-ASCII bearer bytes are an auth failure, not a server error."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        from mnemosyne.mcp_server import _build_sse_app

        app = _build_sse_app(host="0.0.0.0")
        messages = _call_asgi_app(
            app,
            path=path,
            method=method,
            authorization=b"Bearer caf\xe9",
        )

        response_start = next(
            message for message in messages if message["type"] == "http.response.start"
        )
        assert response_start["status"] == 401

    def test_bearer_middleware_accepts_matching_ascii_token(self, monkeypatch):
        """A matching ASCII token passes through the auth middleware."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "supersecret")
        from mnemosyne.mcp_server import _build_sse_app

        app = _build_sse_app(host="0.0.0.0")
        messages = _call_asgi_app(
            app,
            path="/not-found",
            method="GET",
            authorization=b"Bearer supersecret",
        )

        response_start = next(
            message for message in messages if message["type"] == "http.response.start"
        )
        assert response_start["status"] == 404

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


# ---------------------------------------------------------------------------
# Multi-token mode (issue #761): MNEMOSYNE_MCP_TOKENS
# ---------------------------------------------------------------------------

import json as _json


class TestMultiTokenResolve:
    """`_resolve_sse_auth` with MNEMOSYNE_MCP_TOKENS (JSON object)."""

    def test_multi_tokens_parse_and_win_over_single(self, monkeypatch):
        """TOKENS takes precedence when both env vars are set."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "legacy-secret")
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_TOKENS",
            _json.dumps({"hermes-family": "tok1", "hermes-admin": "tok2"}),
        )
        from mnemosyne.mcp_server import _resolve_http_auth, _resolve_multi_tokens
        require_auth, token = _resolve_http_auth("0.0.0.0")
        assert require_auth is True
        assert token is None  # multi-token mode satisfies auth on its own
        assert _resolve_multi_tokens() == {"hermes-family": "tok1", "hermes-admin": "tok2"}

    def test_loopback_multi_token_still_enforces_auth(self, monkeypatch):
        """Review round 9: named tokens opt into multi-agent mode on every
        host -- loopback included (bearer auth + per-agent identity); the
        loopback bypass never silently swallows the configuration."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", '{"a": "t"}')
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("127.0.0.1")
        assert require_auth is True
        assert token is None

    def test_malformed_json_raises_with_env_name(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", "not-json{")
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="MNEMOSYNE_MCP_TOKENS"):
            _resolve_sse_auth("0.0.0.0")

    def test_non_object_json_raises(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", '["tok1", "tok2"]')
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="JSON object"):
            _resolve_sse_auth("0.0.0.0")

    def test_empty_name_or_token_raises(self, monkeypatch):
        """Empty/whitespace-only names AND secrets are all refused."""
        from mnemosyne.mcp_server import _resolve_sse_auth
        for mapping in ('{"": "tok1"}', '{"a": ""}', '{"  ": "tok1"}', '{"a": "   "}'):
            monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", mapping)
            with pytest.raises(RuntimeError, match="empty name or secret"):
                _resolve_sse_auth("0.0.0.0")

    def test_whitespace_only_secret_cannot_match_empty_bearer(self, monkeypatch):
        """A whitespace secret must not be accepted as an empty bearer value
        (guard against 'Authorization: Bearer ' matching a blank secret)."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", '{"a": "   "}')
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="empty name or secret"):
            _resolve_sse_auth("0.0.0.0")

    def test_error_message_mentions_both_env_vars(self, monkeypatch):
        """Non-loopback with neither var set names both options."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKENS", raising=False)
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError) as e:
            _resolve_sse_auth("0.0.0.0")
        assert "MNEMOSYNE_MCP_TOKENS" in str(e.value)
        assert "MNEMOSYNE_MCP_TOKEN" in str(e.value)


class TestMultiTokenMiddleware:
    """_BearerTokenMiddleware: per-name matching + identity propagation.

    UWAGA: nie wolno wołać GET /sse z TestClient -- SSE streamuje w
    nieskonczonosc i klient czeka na zamkniecie odpowiedzi (hang). Testy
    uzywaja wlasnej trasy /whoami (przechodzi przez ten sam middleware,
    ale odpowiada od razu).
    """

    def _build(self, monkeypatch, tokens):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", _json.dumps(tokens))
        from mnemosyne.mcp_server import _build_sse_app
        from mnemosyne.runtime_context import get_request_token_name
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        app = _build_sse_app(host="0.0.0.0")

        async def _whoami(request):
            return PlainTextResponse(get_request_token_name() or "anon")

        app.router.routes.append(Route("/whoami", _whoami))
        return app

    def test_each_named_token_accepted(self, monkeypatch):
        app = self._build(monkeypatch, {"hermes-family": "tok1", "hermes-admin": "tok2"})
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            r1 = client.get("/whoami", headers={"Authorization": "Bearer tok1"})
            r2 = client.get("/whoami", headers={"Authorization": "Bearer tok2"})
        assert r1.status_code == 200 and r1.text == "hermes-family"
        assert r2.status_code == 200 and r2.text == "hermes-admin"

    def test_wrong_token_rejected_401(self, monkeypatch):
        app = self._build(monkeypatch, {"hermes-family": "tok1"})
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            r = client.get("/whoami", headers={"Authorization": "Bearer nope"})
            assert r.status_code == 401
            assert r.json() == {"error": "invalid bearer token"}

    def test_identity_propagates_via_contextvar(self, monkeypatch):
        """Inside an authenticated request, get_request_token_name() returns
        the matched token name (this is what tool handlers consume)."""
        app = self._build(monkeypatch, {"hermes-family": "tok1", "hermes-admin": "tok2"})
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            r1 = client.get("/whoami", headers={"Authorization": "Bearer tok2"})
            r2 = client.get("/whoami", headers={"Authorization": "Bearer tok1"})
            r3 = client.get("/whoami")
        assert r1.text == "hermes-admin"
        assert r2.text == "hermes-family"
        assert r3.status_code == 401  # brak tokenu odrzucony przed handlerem

    def test_scope_state_carries_token_name(self, monkeypatch):
        """ASGI scope carries mnemosyne_token_name for downstream handlers."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", _json.dumps({"hermes-family": "tok1"}))
        from mnemosyne.mcp_server import _build_sse_app
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        app = _build_sse_app(host="0.0.0.0")

        async def _probe(request):
            # request.scope["state"] jest wypelniane przez middleware auth
            # (setdefault) PRZED wywolaniem handlera - odczyt tutaj jest
            # po porzadku wykonywania.
            name = request.scope.get("state", {}).get("mnemosyne_token_name")
            return PlainTextResponse(name or "missing")

        app.router.routes.append(Route("/probe", _probe))
        with TestClient(app) as client:
            r = client.get("/probe", headers={"Authorization": "Bearer tok1"})
        assert r.text == "hermes-family"

    def test_bearer_scheme_case_insensitive(self, monkeypatch):
        """RFC 9110: the auth-scheme token is case-insensitive -- 'bearer'
        and 'BEARER' are valid HTTP syntax and must authenticate the same
        way as 'Bearer'. Any other scheme is still a 401."""
        app = self._build(monkeypatch, {"hermes-family": "tok1"})
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            r1 = client.get("/whoami", headers={"Authorization": "bearer tok1"})
            r2 = client.get("/whoami", headers={"Authorization": "BEARER tok1"})
            r3 = client.get("/whoami", headers={"Authorization": "Basic tok1"})
        assert r1.status_code == 200 and r1.text == "hermes-family"
        assert r2.status_code == 200 and r2.text == "hermes-family"
        assert r3.status_code == 401


class TestMultiTokenParserEdgeCases:
    """Review #830: parser must reject empty mappings and duplicate secrets."""

    def test_empty_object_refused(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", "{}")
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="empty"):
            _resolve_sse_auth("0.0.0.0")

    def test_duplicate_secret_refused(self, monkeypatch):
        """Two names sharing one secret make attribution ambiguous."""
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_TOKENS",
            _json.dumps({"hermes-family": "same-secret", "ci": "same-secret"}),
        )
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="unique secret"):
            _resolve_sse_auth("0.0.0.0")

    def test_error_names_both_aliases(self, monkeypatch):
        """The duplicate-secret error names both offending entries."""
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_TOKENS",
            _json.dumps({"a": "s1", "b": "s1"}),
        )
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError) as e:
            _resolve_sse_auth("0.0.0.0")
        assert "'a'" in str(e.value) and "'b'" in str(e.value)

    def test_exact_duplicate_name_refused(self, monkeypatch):
        """Review round 7 (P1): json.loads collapses exact duplicate JSON
        members to the last value, so '{"a": "s1", "a": "s2"}' would
        silently authenticate 'a' with "s2". Must refuse at startup."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", '{"a": "s1", "a": "s2"}')
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="duplicate token name"):
            _resolve_sse_auth("0.0.0.0")

    def test_whitespace_colliding_names_refused(self, monkeypatch):
        """Review round 7 (P1): names are stripped before use, so
        '{"agent": "s1", " agent ": "s2"}' would authenticate one
        principal ('agent') with two different secrets depending on
        lookup order. Must refuse the post-normalization collision."""
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_TOKENS",
            '{"agent": "s1", " agent ": "s2"}',
        )
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="same token name"):
            _resolve_sse_auth("0.0.0.0")

    def test_duplicate_name_error_names_both_spellings(self, monkeypatch):
        """The collision error quotes both raw spellings, not the
        normalized one, so the operator can find the offending entries."""
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_TOKENS",
            '{"agent": "s1", " agent ": "s2"}',
        )
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError) as e:
            _resolve_sse_auth("0.0.0.0")
        assert "'agent'" in str(e.value) and "' agent '" in str(e.value)


@pytest.fixture(autouse=True)
def _recover_stdlib_logging():
    """Full-suite runs can leave sys.modules["logging"] shadowed by a plain
    object (upstream suites monkeypatch it), which breaks uvicorn startup
    with "No module named 'logging.StreamHandler'". Recover the real stdlib
    module before/after each SSE test — same workaround mnemosyne itself
    ships in core/sync_server.py."""
    import importlib
    import logging as _logging
    import sys as _sys
    if not hasattr(_logging, "getLogger") or not hasattr(_logging, "StreamHandler"):
        _sys.modules.pop("logging", None)
        importlib.reload(importlib.import_module("logging"))
    yield
    if not hasattr(_sys.modules.get("logging", None), "getLogger"):
        _sys.modules.pop("logging", None)
        importlib.import_module("logging")


class TestSessionIdentityBinding:
    """Maintainer review on #830: bind the token identity at the transport's
    session-creation boundary, enforce it only for active sessions, drop it
    when the session closes.

    Implementation: the middleware sets scope["user"] to an AuthenticatedUser
    whose client_id is the token name; SseServerTransport's native
    _session_owners registry then (a) binds the principal inside
    connect_sse() — the GET /sse request that created the session,
    (b) rejects POST /messages/ presented with a different credential
    ("respond exactly as if the session did not exist" -> 404),
    (c) pops the binding in connect_sse()'s finally block on close.

    The regression below drives a REAL SSE session over uvicorn+httpx:
    token A opens the stream (a background thread keeps it alive), the
    emitted endpoint event yields the transport-generated session UUID,
    then a real MCP message is POSTed with token B and later token A.

    Note: the SSE stream must keep being read while POSTs are made —
    aborting the line iterator closes the httpx response and tears the
    session down (that is the close-path, exercised separately below).
    """

    def _server(self, app):
        import threading

        uvicorn = pytest.importorskip("uvicorn")

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(600):  # 30 s — wolne runnery CI
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "uvicorn did not start within 30s"
        port = server.servers[0].sockets[0].getsockname()[1]
        return server, thread, f"http://127.0.0.1:{port}"

    def _build(self, monkeypatch, tokens):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", _json.dumps(tokens))
        from mnemosyne.mcp_server import _build_sse_app
        return _build_sse_app(host="0.0.0.0")

    def test_cross_token_message_rejected_at_transport_level(self, monkeypatch):
        """SSE opened with A; MCP message with B rejected (404, as-if-missing);
        the same message with A is accepted (202). No tool attribution can
        occur for B because the transport never reaches the session writer."""
        httpx = pytest.importorskip("httpx")
        import threading

        app = self._build(monkeypatch, {"hermes-family": "tokA", "ci": "tokB"})
        server, thread, base = self._server(app)
        client = httpx.Client(base_url=base, timeout=30)
        sse = client.send(
            client.build_request("GET", "/sse", headers={"Authorization": "Bearer tokA"}),
            stream=True,
        )
        sid_holder = {}

        def _keep_reading():
            try:
                for line in sse.iter_lines():
                    if line.startswith("data:") and "session_id=" in line:
                        sid_holder["sid"] = line.strip().split("session_id=")[1]
            except Exception:
                # stream closed by the test (close-path) — reader dies
                pass

        try:
            reader = threading.Thread(target=_keep_reading, daemon=True)
            reader.start()
            for _ in range(200):  # 10 s
                if "sid" in sid_holder:
                    break
                time.sleep(0.05)
            assert "sid" in sid_holder, "endpoint event with session_id not received within 10s"
            sid = sid_holder["sid"]
            r_b = client.post(
                f"/messages/?session_id={sid}",
                headers={"Authorization": "Bearer tokB", "Content-Type": "application/json"},
                content=_json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
            )
            assert r_b.status_code == 404, (
                "message with token B for a session created by token A must be "
                f"rejected as if the session did not exist (got {r_b.status_code})"
            )
            r_a = client.post(
                f"/messages/?session_id={sid}",
                headers={"Authorization": "Bearer tokA", "Content-Type": "application/json"},
                content=_json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            )
            assert r_a.status_code == 202
        finally:
            sse.close()
            client.close()
            server.should_exit = True
            thread.join(timeout=5)

    def test_session_owner_binding_cleared_on_close(self, monkeypatch):
        """After the SSE stream closes, the transport drops the session:
        POSTs with the ORIGINAL token are 404 too (no stale bindings)."""
        httpx = pytest.importorskip("httpx")
        import threading

        app = self._build(monkeypatch, {"hermes-family": "tokA"})
        server, thread, base = self._server(app)
        client = httpx.Client(base_url=base, timeout=30)
        sse = client.send(
            client.build_request("GET", "/sse", headers={"Authorization": "Bearer tokA"}),
            stream=True,
        )
        sid_holder = {}

        def _keep_reading():
            try:
                for line in sse.iter_lines():
                    if line.startswith("data:") and "session_id=" in line:
                        sid_holder["sid"] = line.strip().split("session_id=")[1]
            except Exception:
                # stream closed by the test (close-path) — reader dies
                pass

        try:
            reader = threading.Thread(target=_keep_reading, daemon=True)
            reader.start()
            for _ in range(200):  # 10 s
                if "sid" in sid_holder:
                    break
                time.sleep(0.05)
            assert "sid" in sid_holder, "endpoint event with session_id not received within 10s"
            sid = sid_holder["sid"]
            # close the stream -> transport ends the session (finally: pop owner/writer)
            sse.close()
            # Generous budget: teardown runs in a finally block on the SSE task,
            # which heavily loaded CI runners can starve for tens of seconds
            # (seen: >15 s on a shared runner while passing everywhere else).
            deadline = time.time() + 60
            cleared = False
            while time.time() < deadline:
                r = client.post(
                    f"/messages/?session_id={sid}",
                    headers={"Authorization": "Bearer tokA", "Content-Type": "application/json"},
                    content=_json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
                )
                if r.status_code == 404:
                    cleared = True
                    break
                time.sleep(0.2)
            assert cleared, "session binding should be dropped after SSE close"
        finally:
            client.close()
            server.should_exit = True
            thread.join(timeout=5)

    def test_author_attribution_end_to_end(self, monkeypatch, tmp_path):
        """The feature's core claim, proven end to end: a memory stored via
        a REAL tools/call mnemosyne_remember over an authenticated SSE
        session is persisted with the matched token name as its author_id
        (review round 7: assert the persisted row, not just the resolved
        identity).

        Flow: token A opens SSE -> initialize -> tools/call mnemosyne_remember
        (no author_id) -> the success result is read back over the same SSE
        stream, and the row in the isolated database carries
        author_id == 'hermes-family'.
        """
        httpx = pytest.importorskip("httpx")
        import sqlite3
        import threading

        # Isolate the write: _default_db_path() honors MNEMOSYNE_DATA_DIR at
        # runtime, so the remembered row lands in tmp_path/mnemosyne.db.
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
        marker = "attribution-e2e-pr830"

        app = self._build(monkeypatch, {"hermes-family": "tokA"})
        server, thread, base = self._server(app)
        client = httpx.Client(base_url=base, timeout=30)
        sse = client.send(
            client.build_request("GET", "/sse", headers={"Authorization": "Bearer tokA"}),
            stream=True,
        )
        events = []

        def _keep_reading():
            try:
                for line in sse.iter_lines():
                    if line.startswith("data:"):
                        events.append(line[len("data:"):].strip())
            except Exception:
                pass

        try:
            reader = threading.Thread(target=_keep_reading, daemon=True)
            reader.start()
            for _ in range(200):  # 10 s na endpoint event
                if events:
                    break
                time.sleep(0.05)
            # pierwszy data = endpoint URI
            assert events, "no endpoint event received within 10s"
            endpoint = events[0]
            sid = endpoint.split("session_id=")[1]

            def post(payload):
                r = client.post(
                    f"/messages/?session_id={sid}",
                    headers={"Authorization": "Bearer tokA", "Content-Type": "application/json"},
                    content=_json.dumps(payload),
                )
                assert r.status_code == 202

            def wait_result(want_id, deadline_s=30):
                """Read responses arriving over the SSE stream until the
                JSON-RPC reply with the given id shows up."""
                seen = 0
                deadline = time.time() + deadline_s
                while time.time() < deadline:
                    for e in events[1:seen + 1]:
                        try:
                            d = _json.loads(e)
                        except Exception:
                            continue
                        if d.get("id") == want_id:
                            return d
                    seen = len(events) - 1
                    time.sleep(0.1)
                return None

            post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "t", "version": "0"}}})
            assert wait_result(1) is not None, "no initialize result over SSE within 30s"
            post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            # THE core claim: an authenticated write, not a contextvar probe.
            post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": "mnemosyne_remember",
                             "arguments": {"content": f"remember {marker}"}}})
            result = wait_result(2)
            assert result is not None, "no tools/call result over SSE within 30s"
            assert result.get("result", {}).get("isError") is not True, (
                "authenticated mnemosyne_remember must succeed, got: "
                + repr(result.get("result"))[:300]
            )

            # The persisted row is attributed to the token name.
            db = tmp_path / "mnemosyne.db"
            assert db.exists(), f"expected isolated DB at {db}"
            conn = sqlite3.connect(db)
            try:
                rows = conn.execute(
                    "SELECT author_id FROM working_memory WHERE content LIKE ?",
                    (f"%{marker}%",),
                ).fetchall()
            finally:
                conn.close()
            assert rows, f"no working_memory row containing {marker!r} was persisted"
            authors = {r[0] for r in rows}
            assert authors == {"hermes-family"}, (
                f"persisted rows attributed to {authors!r}, expected "
                "{'hermes-family'} (the matched token name)"
            )
        finally:
            sse.close()
            client.close()
            server.should_exit = True
            thread.join(timeout=5)

    def test_spoofed_author_id_rejected_end_to_end(self, monkeypatch, tmp_path):
        """The merge blocker's regression, end to end over a REAL SSE
        session: a client authenticated as token A calls mnemosyne_remember
        with author_id="token-B". The tool call must return an error
        result (is_error=True) -- and, per review round 8, the
        "rejected before any write" guarantee is proven against the
        database: no row containing the spoofed payload may exist.
        """
        httpx = pytest.importorskip("httpx")
        import sqlite3
        import threading

        # Isolated store so the no-write assertion checks THIS run's rows.
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
        app = self._build(monkeypatch, {"hermes-family": "tokA", "token-B": "tokB"})
        server, thread, base = self._server(app)
        client = httpx.Client(base_url=base, timeout=30)
        sse = client.send(
            client.build_request("GET", "/sse", headers={"Authorization": "Bearer tokA"}),
            stream=True,
        )
        events = []

        def _keep_reading():
            try:
                for line in sse.iter_lines():
                    if line.startswith("data:"):
                        events.append(line[len("data:"):].strip())
            except Exception:
                pass

        try:
            reader = threading.Thread(target=_keep_reading, daemon=True)
            reader.start()
            for _ in range(200):  # 10 s
                if events:
                    break
                time.sleep(0.05)
            assert events, "no endpoint event received within 10s"
            sid = events[0].split("session_id=")[1]

            def post(payload):
                r = client.post(
                    f"/messages/?session_id={sid}",
                    headers={"Authorization": "Bearer tokA", "Content-Type": "application/json"},
                    content=_json.dumps(payload),
                )
                assert r.status_code == 202

            post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "t", "version": "0"}}})
            post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            # spoofing attempt: authenticated as hermes-family (tokA),
            # attributing the write to token-B
            post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": "mnemosyne_remember",
                             "arguments": {"content": "spoof probe", "author_id": "token-B"}}})

            spoof_result = None
            deadline = time.time() + 30
            while time.time() < deadline and spoof_result is None:
                for e in events[1:]:
                    try:
                        d = _json.loads(e)
                    except Exception:
                        continue
                    if d.get("id") == 2:
                        spoof_result = d
                        break
                if spoof_result is None:
                    time.sleep(0.1)
            assert spoof_result is not None, "no tools/call result over SSE within 30s"
            assert spoof_result.get("result", {}).get("isError") is True, (
                "spoofed author_id must produce an error result, got: "
                + repr(spoof_result.get("result"))[:300]
            )
            error_text = _json.dumps(spoof_result.get("result", {}))
            assert "conflicts with the authenticated identity" in error_text

            # "Rejected before any write" proven against the store: the
            # spoofed payload must not be persisted under ANY author.
            db = tmp_path / "mnemosyne.db"
            if db.exists():  # absent DB is an even stronger no-write proof
                conn = sqlite3.connect(db)
                try:
                    rows = conn.execute(
                        "SELECT author_id FROM working_memory WHERE content LIKE ?",
                        ("%spoof probe%",),
                    ).fetchall()
                finally:
                    conn.close()
                assert rows == [], (
                    f"spoofed write was persisted ({rows!r}); the rejection "
                    "must happen before any row is written"
                )
        finally:
            sse.close()
            client.close()
            server.should_exit = True
            thread.join(timeout=5)


class TestAuthoritativeTokenIdentity:
    """Review round 5 (merge blocker): with authenticated multi-token SSE,
    the transport-bound token name is the authoritative author identity.
    A client authenticated as A must not be able to persist a row
    attributed to B by passing a client-supplied author_id.
    """

    def _with_token(self, name):
        """Context-manager-free helper: run a coroutine with the request
        token bound, always restoring the previous value afterwards."""
        from mnemosyne.runtime_context import request_token_name
        return request_token_name, request_token_name.set(name)

    def test_conflicting_author_id_rejected(self):
        """Authenticated as 'hermes-family' but author_id='token-B':
        _create_instance must raise, not silently attribute to B."""
        from mnemosyne import mcp_tools as mt
        from mnemosyne.runtime_context import request_token_name

        async def _probe():
            token = request_token_name.set("hermes-family")
            try:
                mt._create_instance(author_id="token-B", bank="default")
            finally:
                request_token_name.reset(token)

        with pytest.raises(ValueError, match="conflicts with the authenticated identity"):
            asyncio.run(_probe())

    def test_matching_author_id_allowed(self):
        """author_id equal to the authenticated token name is accepted
        (no spoofing -- identity is consistent)."""
        from mnemosyne import mcp_tools as mt
        from mnemosyne.runtime_context import request_token_name

        async def _probe():
            token = request_token_name.set("hermes-family")
            try:
                return mt._create_instance(author_id="hermes-family", bank="default")
            finally:
                request_token_name.reset(token)

        inst = asyncio.run(_probe())
        author = getattr(inst, "author_id", None) or getattr(
            getattr(inst, "_author_id", None), "id", None
        )
        assert author == "hermes-family"

    def test_omitted_author_id_resolves_to_token_name(self):
        """No author_id supplied: identity falls back to the authenticated
        token name (the #761 attribution path)."""
        from mnemosyne import mcp_tools as mt
        from mnemosyne.runtime_context import request_token_name

        async def _probe():
            token = request_token_name.set("hermes-admin")
            try:
                return mt._create_instance(bank="default")
            finally:
                request_token_name.reset(token)

        inst = asyncio.run(_probe())
        author = getattr(inst, "author_id", None) or getattr(
            getattr(inst, "_author_id", None), "id", None
        )
        assert author == "hermes-admin"

    def test_unauthenticated_path_keeps_explicit_author_id(self):
        """stdio/local (no token bound): explicit author_id remains
        authoritative -- backward compatibility for unauthenticated paths."""
        from mnemosyne import mcp_tools as mt
        from mnemosyne.runtime_context import request_token_name

        async def _probe():
            # ensure no token is bound (default contextvar value is None)
            assert request_token_name.get() is None
            return mt._create_instance(author_id="legacy-agent", bank="default")

        inst = asyncio.run(_probe())
        author = getattr(inst, "author_id", None) or getattr(
            getattr(inst, "_author_id", None), "id", None
        )
        assert author == "legacy-agent"


class TestTokensEnvFailClosed:
    """Review round 6 (#830): non-string JSON names/secrets must fail at
    startup -- str() coercion would mint predictable credentials ("1",
    "None", "True") instead of surfacing the operator's mistake.

    Review round 8 (#830): a present-but-blank variable must fail closed
    too, not silently fall back to the legacy single-token contract.

    Review round 9 (#830): the same fail-closed semantics apply on
    LOOPBACK -- MNEMOSYNE_MCP_TOKENS is evaluated before the loopback
    bypass, never silently behind it."""

    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_blank_tokens_env_refused(self, monkeypatch, value):
        """MNEMOSYNE_MCP_TOKENS set but empty/whitespace: refuse startup
        instead of treating it as unset (the operator opted into
        multi-agent mode; a silent legacy-token fallback would hide the
        mistake behind an auth mode nobody intended)."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", value)
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "legacy-secret")
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="is set but empty"):
            _resolve_sse_auth("0.0.0.0")

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_tokens_env_refused_on_loopback(self, monkeypatch, value):
        """Review round 9: blank value is a startup error even on the
        default loopback host -- it must not silently start there."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", value)
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="is set but empty"):
            _resolve_sse_auth("127.0.0.1")

    def test_valid_tokens_enable_auth_on_loopback(self, monkeypatch):
        """Review round 9: valid named tokens opt into multi-agent mode on
        EVERY host -- loopback included (bearer auth + per-agent
        identity), not silently ignored behind the loopback bypass."""
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_TOKENS",
            _json.dumps({"hermes-family": "tok1"}),
        )
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("127.0.0.1")
        assert require_auth is True
        assert token is None  # multi-token middleware, not a single secret

    def test_unset_tokens_env_still_single_token_fallback(self, monkeypatch):
        """Unset falls back to the legacy contract, exactly as before."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKENS", raising=False)
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "legacy-secret")
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("0.0.0.0")
        assert require_auth is True and token == "legacy-secret"

    def test_unset_tokens_env_loopback_still_open(self, monkeypatch):
        """Unset + loopback: legacy unauthenticated loopback, unchanged."""
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKENS", raising=False)
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("127.0.0.1")
        assert require_auth is False and token is None

    @pytest.mark.parametrize(
        "payload",
        [
            '{"a": 1}',            # numeric secret
            '{"a": null}',         # null secret
            '{"a": true}',         # boolean secret
            '{"a": ["tok"]}',      # list secret
            '{"a": {"b": "tok"}}', # object secret
        ],
    )
    def test_non_string_secret_refused(self, monkeypatch, payload):
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", payload)
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="must be JSON strings"):
            _resolve_sse_auth("0.0.0.0")

    def test_number_value_refused(self, monkeypatch):
        # {"1": 2} -- numeric-looking name is a legal JSON string key;
        # the numeric VALUE must fail (it is already covered by the
        # parametrize above, this pins the numeric-name+numeric-value
        # combination too).
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", '{"1": 2}')
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="must be JSON strings"):
            _resolve_sse_auth("0.0.0.0")

    def test_valid_strings_still_accepted(self, monkeypatch):
        monkeypatch.setenv(
            "MNEMOSYNE_MCP_TOKENS",
            _json.dumps({"hermes-family": "tok1", "ci": "tok2"}),
        )
        from mnemosyne.mcp_server import _resolve_http_auth, _resolve_multi_tokens
        require_auth, token = _resolve_http_auth("0.0.0.0")
        assert require_auth is True
        assert token is None
        assert _resolve_multi_tokens() == {"hermes-family": "tok1", "ci": "tok2"}


class TestSingleTokenNoIdentityBinding:
    """Review round 6 (#830): the legacy single MNEMOSYNE_MCP_TOKEN must
    keep its prior contract -- authenticate and own sessions, but do NOT
    bind an author identity (no enforced 'default' principal, explicit
    author_id / MNEMOSYNE_AUTHOR_ID keep prior precedence)."""

    def _build_single(self, monkeypatch, token):
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKENS", raising=False)
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", token)
        from mnemosyne.mcp_server import _build_sse_app
        from mnemosyne.runtime_context import get_request_token_name
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        app = _build_sse_app(host="0.0.0.0")

        async def _whoami(request):
            # get_request_token_name() zwraca None -> brak wymuszonej tozsamosci
            return PlainTextResponse(get_request_token_name() or "anon")

        app.router.routes.append(Route("/whoami", _whoami))
        return app

    def test_single_token_authenticates_without_identity(self, monkeypatch):
        """Single-token: poprawny bearer przechodzi (200), ale contextvar
        tozsamosci NIE jest ustawiony (anon) -- kontrakt sprzed multi-token."""
        from starlette.testclient import TestClient

        app = self._build_single(monkeypatch, "legacy-secret")
        with TestClient(app) as client:
            r = client.get("/whoami", headers={"Authorization": "Bearer legacy-secret"})
        assert r.status_code == 200
        assert r.text == "anon", (
            "single-token mode must not bind an author identity "
            f"(got {r.text!r})"
        )

    def test_single_token_rejects_wrong_bearer(self, monkeypatch):
        from starlette.testclient import TestClient

        app = self._build_single(monkeypatch, "legacy-secret")
        with TestClient(app) as client:
            r = client.get("/whoami", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_single_token_explicit_author_id_preserved(self, monkeypatch):
        """Explicit author_id wins on the legacy single-token path (no
        token-bound identity to conflict with) -- prior behavior."""
        from mnemosyne import mcp_tools as mt
        from mnemosyne.runtime_context import request_token_name

        monkeypatch.delenv("MNEMOSYNE_AUTHOR_ID", raising=False)
        assert request_token_name.get() is None  # no bound identity
        inst = mt._create_instance(author_id="legacy-agent", bank="default")
        author = getattr(inst, "author_id", None) or getattr(
            getattr(inst, "_author_id", None), "id", None
        )
        assert author == "legacy-agent"


class TestStreamableHttpMultiToken:
    """Review round 7 (P2): the PR wires the multi-token middleware into
    `_build_streamable_http_app` too, but only SSE had end-to-end coverage.
    Drive a REAL Streamable HTTP server over uvicorn+httpx: authenticated
    initialize, cross-token session rejection (the SDK's stateful manager
    binds each session to the principal from scope["user"] and answers
    404 "Session not found" for a mismatched credential), and token-bound
    write attribution persisted by an actual mnemosyne_remember call.
    """

    def _server(self, app):
        import threading

        uvicorn = pytest.importorskip("uvicorn")

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(600):  # 30 s — wolne runnery CI
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "uvicorn did not start within 30s"
        port = server.servers[0].sockets[0].getsockname()[1]
        return server, thread, f"http://127.0.0.1:{port}"

    def _build(self, monkeypatch, tmp_path, tokens):
        # Isolate writes AND satisfy the non-loopback Host policy the
        # builder enforces via _resolve_transport_security.
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", _json.dumps(tokens))
        monkeypatch.setenv("MNEMOSYNE_MCP_ALLOWED_HOSTS", "testserver")
        from mnemosyne.mcp_server import _build_streamable_http_app
        return _build_streamable_http_app(
            host="0.0.0.0", path="/mcp", json_response=True,
        )

    # Headers every request carries: the transport-security check matches
    # the Host header against MNEMOSYNE_MCP_ALLOWED_HOSTS, and initialize
    # demands both content types per the MCP spec.
    _BASE_HEADERS = {
        "Host": "testserver",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    def _initialize(self):
        return {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "t", "version": "0"}}}

    def test_authenticated_init_and_cross_token_session_rejected(self, monkeypatch, tmp_path):
        """Initialize with token A succeeds; the session A created rejects
        token B with 404 "Session not found" (as-if-missing) while token A
        keeps using it (202)."""
        httpx = pytest.importorskip("httpx")

        app = self._build(monkeypatch, tmp_path, {"hermes-family": "tokA", "ci": "tokB"})
        server, thread, base = self._server(app)
        client = httpx.Client(base_url=base, timeout=30)
        try:
            r = client.post("/mcp", headers={**self._BASE_HEADERS, "Authorization": "Bearer tokA"},
                            content=_json.dumps(self._initialize()))
            assert r.status_code == 200, f"authenticated initialize failed: {r.status_code}"
            sid = r.headers.get("mcp-session-id")
            assert sid, "initialize response must carry mcp-session-id"

            sess = {**self._BASE_HEADERS, "Mcp-Session-Id": sid}
            r_init = client.post("/mcp", headers={**sess, "Authorization": "Bearer tokA"},
                                 content=_json.dumps({"jsonrpc": "2.0",
                                                      "method": "notifications/initialized"}))
            assert r_init.status_code == 202

            r_b = client.post("/mcp", headers={**sess, "Authorization": "Bearer tokB"},
                              content=_json.dumps({"jsonrpc": "2.0", "id": 2,
                                                   "method": "tools/list"}))
            assert r_b.status_code == 404, (
                "session created with token A must reject token B as if the "
                f"session did not exist (got {r_b.status_code})"
            )
            assert "Session not found" in r_b.text

            r_a = client.post("/mcp", headers={**sess, "Authorization": "Bearer tokA"},
                              content=_json.dumps({"jsonrpc": "2.0", "id": 3,
                                                   "method": "tools/list"}))
            assert r_a.status_code == 200, (
                "the owning token must keep full access to its session "
                f"(got {r_a.status_code})"
            )

            # A second principal can open its own session independently.
            r_b2 = client.post("/mcp", headers={**self._BASE_HEADERS, "Authorization": "Bearer tokB"},
                               content=_json.dumps(self._initialize()))
            assert r_b2.status_code == 200 and r_b2.headers.get("mcp-session-id")
        finally:
            client.close()
            server.should_exit = True
            thread.join(timeout=5)

    def test_token_bound_write_attribution(self, monkeypatch, tmp_path):
        """A successful mnemosyne_remember over authenticated Streamable
        HTTP persists the matched token name as the row's author_id."""
        httpx = pytest.importorskip("httpx")
        import sqlite3

        marker = "http-attribution-pr830"
        app = self._build(monkeypatch, tmp_path, {"hermes-family": "tokA"})
        server, thread, base = self._server(app)
        client = httpx.Client(base_url=base, timeout=30)
        try:
            r = client.post("/mcp", headers={**self._BASE_HEADERS, "Authorization": "Bearer tokA"},
                            content=_json.dumps(self._initialize()))
            assert r.status_code == 200
            sid = r.headers["mcp-session-id"]
            sess = {**self._BASE_HEADERS, "Mcp-Session-Id": sid}
            client.post("/mcp", headers={**sess, "Authorization": "Bearer tokA"},
                        content=_json.dumps({"jsonrpc": "2.0",
                                             "method": "notifications/initialized"}))

            r2 = client.post("/mcp", headers={**sess, "Authorization": "Bearer tokA"},
                             content=_json.dumps({
                                 "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                 "params": {"name": "mnemosyne_remember",
                                            "arguments": {"content": f"remember {marker}"}}}))
            assert r2.status_code == 200, f"tools/call failed: {r2.status_code} {r2.text[:200]}"
            assert '"isError":true' not in r2.text.replace(" ", ""), (
                "authenticated mnemosyne_remember must succeed: " + r2.text[:300]
            )

            db = tmp_path / "mnemosyne.db"
            assert db.exists(), f"expected isolated DB at {db}"
            conn = sqlite3.connect(db)
            try:
                rows = conn.execute(
                    "SELECT author_id FROM working_memory WHERE content LIKE ?",
                    (f"%{marker}%",),
                ).fetchall()
            finally:
                conn.close()
            assert rows, f"no working_memory row containing {marker!r} was persisted"
            assert {row[0] for row in rows} == {"hermes-family"}, (
                f"persisted rows attributed to {rows!r}, expected the "
                "matched token name 'hermes-family'"
            )
        finally:
            client.close()
            server.should_exit = True
            thread.join(timeout=5)
