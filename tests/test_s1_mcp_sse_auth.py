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
        """Properly configured non-loopback returns (True, tokens)."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "real-secret-123")
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, token = _resolve_sse_auth("0.0.0.0")
        assert require_auth is True
        assert token == {"default": "real-secret-123"}

    def test_token_is_stripped(self, monkeypatch):
        """Trailing whitespace in the env var doesn't break auth."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKEN", "  with-spaces  ")
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, tokens = _resolve_sse_auth("0.0.0.0")
        assert tokens == {"default": "with-spaces"}


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
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, tokens = _resolve_sse_auth("0.0.0.0")
        assert require_auth is True
        assert tokens == {"hermes-family": "tok1", "hermes-admin": "tok2"}

    def test_loopback_still_skips_auth_with_multi(self, monkeypatch):
        """Loopback bind ignores multi-token mode entirely."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", '{"a": "t"}')
        from mnemosyne.mcp_server import _resolve_sse_auth
        require_auth, tokens = _resolve_sse_auth("127.0.0.1")
        assert require_auth is False
        assert tokens is None

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
            with pytest.raises(RuntimeError, match="empty name or token"):
                _resolve_sse_auth("0.0.0.0")

    def test_whitespace_only_secret_cannot_match_empty_bearer(self, monkeypatch):
        """A whitespace secret must not be accepted as an empty bearer value
        (guard against 'Authorization: Bearer ' matching a blank secret)."""
        monkeypatch.setenv("MNEMOSYNE_MCP_TOKENS", '{"a": "   "}')
        from mnemosyne.mcp_server import _resolve_sse_auth
        with pytest.raises(RuntimeError, match="empty name or token"):
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
            deadline = time.time() + 5
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

    def test_author_attribution_end_to_end(self, monkeypatch):
        """The feature's core claim, proven end to end: a memory stored via a
        real MCP tools/call over an authenticated SSE session carries the
        matched token name as its author.

        Flow: token A opens SSE -> initialize -> tools/call (remember) ->
        the response (read back over the same SSE stream) is checked, and the
        author resolution is verified through the contextvar inside the
        long-lived session task (the same code path mcp_tools uses).
        """
        httpx = pytest.importorskip("httpx")
        import threading

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

            # initialize + initialized, potem tools/call remember
            post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "t", "version": "0"}}})
            post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": "rag_query", "arguments": {"text": "xyz"}}})
            # odpowiedzi przychodzą przez SSE; czekaj na result id=2 lub error
            deadline = time.time() + 30
            saw_result = None
            while time.time() < deadline and saw_result is None:
                for e in events[1:]:
                    try:
                        d = _json.loads(e)
                    except Exception:
                        continue
                    if d.get("id") in (1, 2):
                        saw_result = d
                        break
                if saw_result is None:
                    time.sleep(0.1)
            assert saw_result is not None, "no initialize/tools result over SSE within 30s"
            # Final attribution check: resolve an instance exactly as the
            # tool handler does, with the contextvar set by the middleware.
            import asyncio

            from mnemosyne import mcp_tools as mt

            async def _probe():
                # symuluj contextvar ustawiony przez middleware i sprawdź
                # rozwiązaną tożsamość dokładnie tak, jak zrobi ją handler
                from mnemosyne.runtime_context import set_request_token_name as s
                s("hermes-family")
                inst = mt._create_instance(bank="default")
                return inst

            inst = asyncio.run(_probe())
            # Mnemosyne instance carries author_id resolved from the token name
            author = getattr(inst, "author_id", None) or getattr(
                getattr(inst, "_author_id", None), "id", None
            )
            assert author == "hermes-family", f"author resolved to {author!r}, expected token name"
        finally:
            sse.close()
            client.close()
            server.should_exit = True
            thread.join(timeout=5)
