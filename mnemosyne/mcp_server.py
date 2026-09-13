"""
Mnemosyne MCP Server -- stdio, SSE and Streamable HTTP transports.

Usage:
    # stdio (default) -- for Claude Desktop, etc.
    mnemosyne mcp

    # SSE on loopback -- safe default, no auth required
    mnemosyne mcp --transport sse --port 8080

    # Streamable HTTP on loopback -- native MCP http transport, single
    # configurable endpoint (default /mcp) handling GET/POST/DELETE
    # (no separate /messages route to proxy)
    mnemosyne mcp --transport streamable-http --port 8080

    # SSE or Streamable HTTP exposed on LAN -- REQUIRES bearer token
    # via env var
    MNEMOSYNE_MCP_TOKEN=my-secret-token mnemosyne mcp \\
        --transport sse --host 0.0.0.0 --port 8080
    MNEMOSYNE_MCP_TOKEN=my-secret-token \\
    MNEMOSYNE_MCP_ALLOWED_HOSTS=mnemosyne.k.example.com:* mnemosyne mcp \\
        --transport streamable-http --host 0.0.0.0 --port 8080

    # Custom streamable HTTP endpoint path (default: /mcp)
    mnemosyne mcp --transport streamable-http --path /mcp --port 8080

    # JSON-only streamable HTTP (no SSE upgrade)
    mnemosyne mcp --transport streamable-http --json-response --port 8080

    # Specific bank
    mnemosyne mcp --bank project_a

Security note (S1, 2026-05-12):
    The HTTP transports default to host=127.0.0.1 (loopback only). Binding
    to a non-loopback address (0.0.0.0, a LAN IP, etc.) requires the env
    var MNEMOSYNE_MCP_TOKEN to be set; clients must then send
    ``Authorization: Bearer <token>`` on every request. Without the token
    the server refuses to start. This prevents a LAN attacker from
    reading/writing/deleting the user's memory via an unauthenticated
    MCP endpoint.

    MNEMOSYNE_MCP_TOKENS (a JSON object of named secrets) is evaluated on
    every host, loopback included: when set, it opts the server into
    multi-agent mode -- bearer auth with per-agent identity is enforced
    even on loopback, and a present-but-blank value refuses startup. See
    docs/cli-reference.md ("Multi-agent tokens"). Bearer tokens are
    cleartext headers: on a non-loopback bind, terminate TLS in front of
    the server (reverse proxy or secure tunnel).

    The Streamable HTTP transport additionally applies a Host/Origin policy
    on non-loopback binds (fail-closed): MNEMOSYNE_MCP_ALLOWED_HOSTS is
    required (comma-separated exact names or ``name:*`` patterns), and
    MNEMOSYNE_MCP_ALLOWED_ORIGINS optionally allowlists browser origins.
    Requests without an Origin header (CLI/SDK clients) always pass the
    origin check; any Origin not listed is rejected with HTTP 403. See
    docs/cli-reference.md for the full contract.
"""

import hmac
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Guarded import -- MCP is optional
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, CallToolResult
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    Server = None
    stdio_server = None
    TextContent = None
    CallToolResult = None

# Guarded import -- starlette is only needed by the HTTP transports. The
# bearer middleware is only ever instantiated by the app builders, which
# raise a friendly error when starlette is missing.
try:
    from starlette.responses import JSONResponse
except ImportError:
    JSONResponse = None

# Auth principal classes for the multi-token SSE session-ownership feed.
# Resolved at import so a missing dependency fails at startup (the
# identity-binding guarantee is only as good as this import; review
# round 4 on #830). Absent only when the mcp extra is not installed,
# in which case the app builders already refuse to start.
try:
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken
except ImportError:
    AuthenticatedUser = None
    AccessToken = None

from mnemosyne.mcp_tools import get_tool_definitions, handle_tool_call
from mnemosyne.runtime_context import set_request_token_name  # noqa: F401 (used in middleware)

# ---------------------------------------------------------------------------
# Security helpers (S1)
# ---------------------------------------------------------------------------

# Hosts treated as loopback-only; safe to expose without auth.
#
# Matches the mcp SDK's exact tokenless-loopback values, and nothing else.
# The SDK arms its built-in DNS-rebinding protection only when it sees one
# of these exact strings as the host, so recognition must be an exact
# comparison. Normalizing before the SDK sees the host (case folding,
# whitespace stripping, or alias expansion like ``ip6-localhost``) would
# admit binds the SDK does not protect -- tokenless but unprotected against
# arbitrary Host/Origin requests. Any deviation therefore fails closed
# (bearer token + explicit Host policy required); tokenless loopback stays
# available via the exact ``127.0.0.1`` / ``localhost`` / ``::1`` spellings.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_TOKEN_ENV = "MNEMOSYNE_MCP_TOKEN"
_TOKENS_ENV = "MNEMOSYNE_MCP_TOKENS"
_ALLOWED_HOSTS_ENV = "MNEMOSYNE_MCP_ALLOWED_HOSTS"
_ALLOWED_ORIGINS_ENV = "MNEMOSYNE_MCP_ALLOWED_ORIGINS"


def _is_loopback(host: str) -> bool:
    """Return True if `host` is a loopback bind that needs no auth.

    Exact match against ``_LOOPBACK_HOSTS`` (the SDK's tokenless-loopback
    allowlist). Aliases and case/whitespace variants are NOT loopback here:
    the mcp SDK only arms its DNS-rebinding protection for these exact host
    strings, so anything else must fail closed.
    """
    return host in _LOOPBACK_HOSTS


def _parse_csv_env(name: str) -> list[str]:
    """Parse a comma-separated env var into trimmed non-empty items.

    Accepts a single value or a list: ``a.example.com`` and
    ``a.example.com, b.example.com:*`` both work. Empty/unset returns ``[]``.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_transport_security(host: str):
    """Resolve the Host/Origin policy for the Streamable HTTP transport.

    Loopback binds return None so the mcp SDK keeps its built-in
    DNS-rebinding protection (allowed Hosts/Origins fixed to
    127.0.0.1/localhost/[::1]).

    Non-loopback binds are fail-closed: the operator must declare the Host
    header values clients will present via MNEMOSYNE_MCP_ALLOWED_HOSTS
    (comma-separated; each value is an exact host or a ``name:*`` pattern
    covering any port), otherwise startup is refused. Origins default to an
    empty allowlist: requests that send no Origin header (CLI/SDK clients)
    always pass, while any browser origin is rejected unless listed in
    MNEMOSYNE_MCP_ALLOWED_ORIGINS.
    """
    if _is_loopback(host):
        return None
    allowed_hosts = _parse_csv_env(_ALLOWED_HOSTS_ENV)
    if not allowed_hosts:
        raise RuntimeError(
            f"Refusing to bind MCP Streamable HTTP on non-loopback host {host!r} "
            f"without a Host policy. Set the {_ALLOWED_HOSTS_ENV} env var to the "
            f"comma-separated Host header values clients will present (exact names "
            f"or 'name:*' patterns), e.g. 'mnemosyne.k.example.com:*'. Or bind to "
            f"127.0.0.1 (the default) for local-only use."
        )
    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=_parse_csv_env(_ALLOWED_ORIGINS_ENV),
    )


def _parse_tokens_env(raw: str) -> "dict[str, str]":
    """Parse MNEMOSYNE_MCP_TOKENS into an ordered {name: secret} mapping.

    Accepts a JSON object of ``{"agent-name": "secret", ...}``. Names and
    secrets MUST be JSON strings: non-string values are never coerced --
    ``str(1)`` or ``str(None)`` would mint predictable credentials ("1",
    "None", "True") instead of surfacing the operator's mistake at startup.

    Raises RuntimeError with an actionable message on malformed JSON,
    non-object payloads, empty mappings, empty names/secrets, duplicate
    names (exact JSON duplicates -- ``json.loads`` would silently keep
    only the last member -- and post-strip collisions like ``"agent"``
    vs ``" agent "``, either of which could rotate which credential owns
    an agent identity without an error), and duplicate secrets (two
    names sharing one secret would make per-agent attribution
    ambiguous, since authentication matches the first name for a
    presented token).
    """
    import json as _json

    # json.loads collapses exact duplicate members to the last value;
    # catch them at parse time (any object in the payload) via the
    # pairs hook, then fail closed after the decode.
    duplicate_names: list = []

    def _flag_duplicate_names(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                duplicate_names.append(key)
            seen.add(key)
        return dict(pairs)

    try:
        parsed = _json.loads(raw, object_pairs_hook=_flag_duplicate_names)
    except _json.JSONDecodeError as e:
        raise RuntimeError(
            f"{_TOKENS_ENV} is not valid JSON ({e}). Expected an object "
            f"mapping token names to secrets, e.g. "
            f"'{{\"hermes-family\": \"tok1\", \"hermes-admin\": \"tok2\"}}'."
        ) from e
    if duplicate_names:
        raise RuntimeError(
            f"{_TOKENS_ENV} contains duplicate token name(s) "
            f"{', '.join(repr(n) for n in duplicate_names)}; JSON would "
            f"silently keep only the last entry for a duplicated name, "
            f"rotating which secret owns that agent. Use unique names."
        )
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"{_TOKENS_ENV} must be a JSON object mapping token names to "
            f"secrets; got {type(parsed).__name__}."
        )
    if not parsed:
        raise RuntimeError(
            f"{_TOKENS_ENV} is empty; add at least one \"name\": \"secret\" "
            f"entry (or unset it to fall back to {_TOKEN_ENV})."
        )
    tokens: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    seen_secrets: dict[str, str] = {}
    for name, secret in parsed.items():
        if not isinstance(name, str) or not isinstance(secret, str):
            bad = name if not isinstance(name, str) else secret
            raise RuntimeError(
                f"{_TOKENS_ENV} entries must be JSON strings; got "
                f"{type(bad).__name__} for {name!r}. Quote both the name "
                f'and the secret, e.g. {{"agent-name": "secret"}}.'
            )
        name_s = name.strip()
        secret_s = secret.strip()
        if not name_s or not secret_s:
            raise RuntimeError(
                f"{_TOKENS_ENV} contains an empty name or secret; every "
                f"entry needs a non-empty name and secret."
            )
        if name_s in seen_names:
            raise RuntimeError(
                f"{_TOKENS_ENV} entries {seen_names[name_s]!r} and "
                f"{name!r} normalize to the same token name {name_s!r}; "
                f"use unique names so per-agent attribution stays "
                f"unambiguous."
            )
        seen_names[name_s] = name
        if secret_s in seen_secrets:
            raise RuntimeError(
                f"{_TOKENS_ENV} entries {seen_secrets[secret_s]!r} and "
                f"{name_s!r} share one secret; each token needs a unique "
                f"secret so per-agent attribution stays unambiguous."
            )
        seen_secrets[secret_s] = name_s
        tokens[name_s] = secret_s
    return tokens


def _resolve_http_auth(host: str) -> Tuple[bool, Optional[str]]:
    """Decide whether an HTTP transport needs bearer-token auth and the token.

    Applies to both the SSE and Streamable HTTP transports.

    MNEMOSYNE_MCP_TOKENS is evaluated FIRST, on every host: setting it
    opts the operator into multi-agent mode, so a valid mapping enables
    bearer auth (with per-agent identity) even on loopback, and a
    present-but-blank value is a startup error even on loopback -- the
    variable never silently disappears behind the loopback bypass.

    Returns (require_auth, token). With MNEMOSYNE_MCP_TOKENS unset the
    legacy contract applies: loopback needs no auth; a non-loopback bind
    requires MNEMOSYNE_MCP_TOKEN -- refusing to start an
    unauthenticated network-exposed MCP server. Bearer tokens are
    cleartext headers: on non-loopback binds, terminate TLS in front of
    the server (reverse proxy or secure tunnel).
    """
    if _resolve_multi_tokens() is not None:
        # Opt-in multi-agent mode: named tokens satisfy the auth
        # requirement on their own (they take precedence over the single
        # MNEMOSYNE_MCP_TOKEN) and enable per-agent identity on every
        # host. The builders install the multi-token middleware when the
        # resolved token is None.
        return (True, None)
    if _is_loopback(host):
        return (False, None)
    token = (os.environ.get(_TOKEN_ENV) or "").strip()
    if not token:
        raise RuntimeError(
            f"Refusing to bind MCP over HTTP on non-loopback host {host!r} without "
            f"authentication. Set the {_TOKENS_ENV} env var to a JSON object of "
            f"named secrets (multiple agents) or {_TOKEN_ENV} to a strong random "
            f"secret, and have clients send 'Authorization: Bearer <token>' on "
            f"each request. Bearer tokens are cleartext headers: on a "
            f"non-loopback bind terminate TLS with a reverse proxy or a "
            f"secure tunnel. Or bind to 127.0.0.1 (the default) for "
            f"local-only use."
        )
    return (True, token)


# Backward-compatible alias for the pre-streamable-http name.
_resolve_sse_auth = _resolve_http_auth


def _resolve_multi_tokens() -> Optional["dict[str, str]"]:
    """Return the MNEMOSYNE_MCP_TOKENS mapping, or None when unset.

    This is the opt-in multi-agent mode selector: a valid JSON object of
    named secrets enables per-agent identity; unset falls back to the
    legacy single MNEMOSYNE_MCP_TOKEN contract. Parse errors raise here
    (fail closed at startup).

    Present-but-blank (empty or whitespace-only) also raises rather than
    falling back: an operator who sets the variable has opted into
    multi-agent mode, and a stray ``MNEMOSYNE_MCP_TOKENS=""`` silently
    selecting the legacy single-token contract would hide that mistake
    behind an auth mode nobody intended.
    """
    raw = os.environ.get(_TOKENS_ENV)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        raise RuntimeError(
            f"{_TOKENS_ENV} is set but empty; provide a JSON object of "
            f'"name": "secret" entries, or unset it to use '
            f"{_TOKEN_ENV}."
        )
    return _parse_tokens_env(raw)


class _BearerTokenMiddleware:
    """Pure-ASGI bearer auth middleware (single- or multi-token).

    BaseHTTPMiddleware buffers the full response body before forwarding it
    to the client. SseServerTransport and the streamable-HTTP transport
    write directly to the raw ASGI send callable, so BaseHTTPMiddleware
    raises:
      AssertionError: Unexpected message: http.response.start
    on every streaming connection, terminating it immediately.

    This pure-ASGI implementation forwards scope/receive/send untouched
    after auth so SSE/streamable frames are never buffered.

    Single-token mode (``token=``): the legacy contract -- authenticate and
    forward, no author-identity binding (explicit ``author_id`` /
    ``MNEMOSYNE_AUTHOR_ID`` keep their prior precedence in mcp_tools).

    Multi-token mode (``tokens=``): the presented bearer token is matched
    against every configured secret (constant-time per candidate) and the
    *name* of the matched entry is stored in ``scope["state"]`` plus a
    contextvar, so tool handlers attribute memories to the calling agent
    without any client-side cooperation; the name is also exposed as an
    authenticated principal feeding the SSE transport's native
    session-ownership check. A conflicting client-supplied ``author_id``
    is rejected in mcp_tools in this mode.
    """

    def __init__(self, app, token: Optional[str] = None,
                 tokens: Optional["dict[str, str]"] = None):
        """Wrap ``app`` and require the bearer credential.

        Exactly one of ``token`` (legacy single secret) or ``tokens``
        (multi-agent {name: secret} mapping) must be given. Secrets are
        stored as bytes so presented credentials can be compared safely
        with ``hmac.compare_digest`` (which raises on non-ASCII str,
        turning a bad request into a 500).
        """
        if (token is None) == (tokens is None):
            raise ValueError("pass exactly one of token= or tokens=")
        self.app = app
        if token is not None:
            self.tokens: "dict[str, str]" = {"default": token}
            self.bind_identity = False
        else:
            self.tokens = tokens
            self.bind_identity = True
            # Pre-encode for constant-time comparison per candidate.
        self._encoded = [
            (name, secret.encode("utf-8")) for name, secret in self.tokens.items()
        ]

    async def __call__(self, scope, receive, send):
        """Enforce bearer auth on HTTP requests, passing non-HTTP scope through.

        Missing, non-Bearer, or mismatched credentials yield HTTP 401 with a
        ``WWW-Authenticate: Bearer`` challenge; valid credentials are
        forwarded to the wrapped app untouched.
        """
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        header = None
        for k, v in scope.get("headers", []):
            if k == b"authorization":
                header = v
                break
        if header is None:
            resp = JSONResponse(
                {"error": "missing bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await resp(scope, receive, send)
            return
        # Work in bytes end-to-end: hmac.compare_digest raises TypeError on
        # non-ASCII str, so a credential like ``Bearer \xff`` would otherwise
        # turn into a 500. The scheme is matched case-insensitively per RFC
        # 6750 (the auth-scheme token is case-insensitive).
        scheme, _, presented = header.partition(b" ")
        if scheme.lower() != b"bearer":
            resp = JSONResponse(
                {"error": "missing bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await resp(scope, receive, send)
            return
        presented = presented.strip()
        matched_name = None
        for name, encoded in self._encoded:
            if hmac.compare_digest(presented, encoded):
                matched_name = name
                break
        if matched_name is None:
            resp = JSONResponse(
                {"error": "invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await resp(scope, receive, send)
            return
        if self.bind_identity:
            # Multi-token mode only: record the per-agent identity for
            # mcp_tools author resolution (contextvar + ASGI state). The
            # legacy single-token mode deliberately does NOT bind an
            # author identity -- prior contract preserved.
            set_request_token_name(matched_name)
            scope.setdefault("state", {})["mnemosyne_token_name"] = matched_name
            scope["user"] = AuthenticatedUser(
                AccessToken(
                    token=presented.decode("latin-1", "replace"),
                    client_id=matched_name,
                    scopes=[],
                )
            )
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Server Setup
# ---------------------------------------------------------------------------

def _build_mcp_server() -> Server:
    """Build an MCP ``Server`` instance wired to the mnemosyne tool handlers.

    Returns a ``mcp.server.lowlevel.server.Server`` with ``on_list_tools`` and
    ``on_call_tool`` callbacks installed. Used by both the stdio transport
    (see ``_run_stdio``) and the SSE transport (see ``_build_sse_app``) so the
    registration logic stays in one place.

    Migrated from ``mcp`` SDK 1.x to 2.x: the 1.x ``@server.list_tools()`` and
    ``@server.call_tool()`` decorators were removed in 2.0. The 2.x
    ``mcp.server.lowlevel.server.Server`` accepts the same callbacks as
    ``on_list_tools``/``on_call_tool`` keyword arguments on the constructor.

    The ``on_call_tool`` signature also changed in 2.x: callbacks now receive
    ``(ctx, params)`` where ``params`` is a ``CallToolRequestParams`` carrying
    ``.name`` and ``.arguments``. The handler returns a ``CallToolResult``
    instead of a raw list of ``TextContent``.
    """
    from mcp.types import CallToolResult, ListToolsResult, Tool

    async def _on_list_tools(ctx, params):  # noqa: ARG001 — ctx/params unused
        raw = get_tool_definitions()
        # The dict from get_tool_definitions() uses ``inputSchema`` (the wire
        # field name); ``mcp.types.Tool`` accepts it via Pydantic alias and
        # normalizes to ``input_schema`` on the model. **t spreads both.
        # SDK 2.x contract: the callback must return a ListToolsResult
        # wrapper, not a bare list of Tool objects.
        return ListToolsResult(tools=[Tool(**t) for t in raw])

    async def _on_call_tool(ctx, params):  # noqa: ARG001 — ctx unused
        tool_name = ""
        try:
            candidate_name = getattr(params, "name", "")
            if type(candidate_name) is str:
                tool_name = candidate_name
            result = handle_tool_call(tool_name, params.arguments or {})
            content = [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            error_code = result.get("error") if isinstance(result, dict) else None
            is_error = (
                isinstance(result, dict)
                and result.get("status") == "error"
                and type(error_code) is str
                and error_code in {"batch_validation_failed", "batch_failed"}
            )
            return CallToolResult(content=content, is_error=is_error)
        except Exception as e:
            if tool_name == "mnemosyne_batch":
                logger.error("mnemosyne_batch MCP dispatch failed")
                payload = {
                    "status": "error",
                    "error": "batch_failed",
                    "failed_index": None,
                    "action": None,
                }
            else:
                payload = {"status": "error", "message": str(e)}
            content = [TextContent(type="text", text=json.dumps(payload, indent=2))]
            return CallToolResult(content=content, is_error=True)

    return Server(
        "mnemosyne",
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )


async def _run_stdio() -> None:
    """Run MCP server over stdio transport."""
    if not _MCP_AVAILABLE:
        raise RuntimeError("MCP not installed. Run: pip install mnemosyne-memory[mcp]")

    server = _build_mcp_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _build_sse_app(host: str = "127.0.0.1"):
    """Build the Starlette app for SSE transport.

    Split out from `_run_sse` so the auth-gating + middleware-installation
    logic is testable without spinning up uvicorn.

    Returns the configured Starlette application. Raises RuntimeError if
    host is non-loopback and MNEMOSYNE_MCP_TOKEN is unset.
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError("MCP not installed. Run: pip install mnemosyne-memory[mcp]")

    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.middleware import Middleware
        from starlette.responses import JSONResponse
    except ImportError:
        raise RuntimeError(
            "SSE transport requires starlette and uvicorn. "
            "Run: pip install starlette uvicorn"
        )

    require_auth, token = _resolve_http_auth(host)
    multi_tokens = _resolve_multi_tokens() if require_auth else None

    # Trailing slash required: SseServerTransport emits POST URIs as
    # /messages/ and Starlette Mount path-prefix matching needs it to
    # agree. Route("/messages") would 404 on every client POST.
    transport = SseServerTransport("/messages/")
    server = _build_mcp_server()

    async def handle_sse(request):
        async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        return JSONResponse({})

    middleware = []
    if require_auth:
        if multi_tokens is not None:
            # Opt-in multi-agent mode: matched token name is the
            # authoritative author identity (session ownership included).
            if AuthenticatedUser is None or AccessToken is None:
                raise RuntimeError(
                    "Multi-token SSE auth requires the mcp auth middleware "
                    "(AuthenticatedUser/AccessToken); install mnemosyne-memory[mcp]."
                )
            middleware.append(Middleware(_BearerTokenMiddleware, tokens=multi_tokens))
            logger.info(
                "MCP SSE multi-token auth enabled (host=%s, agents=%d). Each "
                "client sends its own 'Authorization: Bearer <token>'; the "
                "matched token name is the authoritative author identity. "
                "Bearer tokens are cleartext headers: on a non-loopback "
                "bind terminate TLS with a reverse proxy or a secure tunnel.",
                host,
                len(multi_tokens),
            )
        else:
            middleware.append(Middleware(_BearerTokenMiddleware, token=token))
            logger.info(
                "MCP SSE bearer-token auth enabled (host=%s). Clients must send "
                "'Authorization: Bearer <token>' on every request. Bearer "
                "tokens are cleartext headers: on a non-loopback bind "
                "terminate TLS with a reverse proxy or a secure tunnel.",
                host,
            )
    else:
        logger.info(
            "MCP SSE running loopback-only (host=%s); no auth required.",
            host,
        )

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            # transport.handle_post_message is an ASGI callable, not a
            # request-response endpoint. Mount (not Route) is required so
            # Starlette passes scope/receive/send directly without wrapping
            # the response. The trailing slash must match the transport path.
            Mount("/messages/", app=transport.handle_post_message),
        ],
        middleware=middleware,
    )
    return starlette_app


async def _run_sse(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Run MCP server over SSE transport.

    Default host is 127.0.0.1 (loopback only). Auth policy per host:
    see _resolve_sse_auth (MNEMOSYNE_MCP_TOKENS is honored on every
    host; a non-loopback bind without it requires MNEMOSYNE_MCP_TOKEN).
    """
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "SSE transport requires starlette and uvicorn. "
            "Run: pip install starlette uvicorn"
        )

    app = _build_sse_app(host=host)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def _build_streamable_http_app(
    host: str = "127.0.0.1",
    path: str = "/mcp",
    json_response: bool = False,
):
    """Build the Starlette app for the native Streamable HTTP transport.

    Uses the ``mcp`` SDK 2.x ``Server.streamable_http_app()`` helper, which
    wires a ``StreamableHTTPSessionManager`` with a single route at ``path``
    handling GET/POST/DELETE. This is the modern MCP ``http`` transport: a
    client POSTs JSON-RPC straight to ``path`` (no separate /messages route
    to proxy) and the server responds with JSON or upgrades to an SSE stream
    depending on the request's Accept header.

    Split out from `_run_streamable_http` so the auth-gating +
    middleware-installation logic is testable without spinning up uvicorn.

    Returns the configured Starlette application. Raises RuntimeError when
    host is non-loopback and MNEMOSYNE_MCP_TOKEN or
    MNEMOSYNE_MCP_ALLOWED_HOSTS is unset.
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError("MCP not installed. Run: pip install mnemosyne-memory[mcp]")

    require_auth, token = _resolve_http_auth(host)
    multi_tokens = _resolve_multi_tokens() if require_auth else None
    transport_security = _resolve_transport_security(host)

    server = _build_mcp_server()
    app = server.streamable_http_app(
        streamable_http_path=path,
        json_response=json_response,
        host=host,
        transport_security=transport_security,
    )

    if require_auth:
        # add_middleware inserts outermost and is the supported Starlette
        # API (mutating user_middleware directly would miss the lazy-built
        # middleware stack).
        if multi_tokens is not None:
            if AuthenticatedUser is None or AccessToken is None:
                raise RuntimeError(
                    "Multi-token HTTP auth requires the mcp auth middleware "
                    "(AuthenticatedUser/AccessToken); install mnemosyne-memory[mcp]."
                )
            app.add_middleware(_BearerTokenMiddleware, tokens=multi_tokens)
            logger.info(
                "MCP Streamable HTTP multi-token auth enabled (host=%s, path=%s, "
                "agents=%d). The matched token name is the authoritative "
                "author identity. Bearer tokens are cleartext headers: on a "
                "non-loopback bind terminate TLS with a reverse proxy or a "
                "secure tunnel.",
                host,
                path,
                len(multi_tokens),
            )
        else:
            app.add_middleware(_BearerTokenMiddleware, token=token)
            logger.info(
                "MCP Streamable HTTP bearer-token auth enabled (host=%s, path=%s). "
                "Clients must send 'Authorization: Bearer <token>' on every "
                "request. Bearer tokens are cleartext headers: on a "
                "non-loopback bind terminate TLS with a reverse proxy or a "
                "secure tunnel.",
                host,
                path,
            )
    else:
        logger.info(
            "MCP Streamable HTTP running loopback-only (host=%s, path=%s); "
            "no auth required.",
            host,
            path,
        )

    return app


async def _run_streamable_http(
    port: int = 8080,
    host: str = "127.0.0.1",
    path: str = "/mcp",
    json_response: bool = False,
) -> None:
    """Run MCP server over the Streamable HTTP transport.

    Default host is 127.0.0.1 (loopback only). Binding non-loopback requires
    MNEMOSYNE_MCP_TOKEN (see _resolve_http_auth) *and*
    MNEMOSYNE_MCP_ALLOWED_HOSTS (see _resolve_transport_security). Both gates
    fail closed independently, so satisfying only one still refuses startup.
    """
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "Streamable HTTP transport requires starlette and uvicorn. "
            "Run: pip install starlette uvicorn"
        )

    app = _build_streamable_http_app(host=host, path=path, json_response=json_response)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _load_dotenv(env_file_path: Optional[str] = None) -> Optional[str]:
    """Auto-load .env file into os.environ for MCP server execution.

    Search order:
    1. Explicit env_file_path passed via --env-file
    2. $HERMES_HOME/.env or ~/.hermes/.env
    3. $MNEMOSYNE_HOME/.env or ~/.mnemosyne/.env
    4. ./.env (current working directory)

    Returns the path of the loaded .env file, or None if no file was loaded.
    """
    candidates = []
    if env_file_path:
        candidates.append(Path(env_file_path).expanduser())

    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    candidates.append(Path(hermes_home) / ".env")

    mnemosyne_home = os.environ.get("MNEMOSYNE_HOME", os.path.expanduser("~/.mnemosyne"))
    candidates.append(Path(mnemosyne_home) / ".env")

    try:
        candidates.append(Path.cwd() / ".env")
    except Exception:
        pass

    loaded_path = None
    for p in candidates:
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                loaded_path = str(p)
                logger.info("Loaded MCP environment from %s", loaded_path)
                break
            except Exception as exc:
                logger.warning("Failed to parse env file %s: %s", p, exc)
                if env_file_path and p == Path(env_file_path).expanduser():
                    return None
    return loaded_path


def run_mcp_server(
    transport: str = "stdio",
    port: int = 8080,
    bank: Optional[str] = None,
    host: str = "127.0.0.1",
    env_file: Optional[str] = None,
    path: str = "/mcp",
    json_response: bool = False,
) -> None:
    """
    Run the Mnemosyne MCP server.

    Args:
        transport: "stdio", "sse" or "streamable-http" ("http" accepted as
            an alias for streamable-http)
        port: Port for the HTTP transports (ignored for stdio)
        bank: Default bank for operations (optional)
        host: Bind address for the HTTP transports (default: 127.0.0.1 --
            loopback only). Non-loopback hosts require MNEMOSYNE_MCP_TOKEN on
            both HTTP transports, and streamable-http additionally requires
            MNEMOSYNE_MCP_ALLOWED_HOSTS.
        env_file: Path to optional .env file to load before starting.
        path: Endpoint path for streamable-http (default: /mcp)
        json_response: Force JSON-only responses for streamable-http
            (no SSE upgrade). Default: False.
    """
    _load_dotenv(env_file)

    if bank:
        os.environ["MNEMOSYNE_MCP_BANK"] = bank

    if transport == "stdio":
        asyncio.run(_run_stdio())
    elif transport == "sse":
        asyncio.run(_run_sse(port=port, host=host))
    elif transport in ("streamable-http", "http"):
        asyncio.run(
            _run_streamable_http(
                port=port, host=host, path=path, json_response=json_response
            )
        )
    else:
        raise ValueError(
            f"Unknown transport: {transport}. "
            "Use 'stdio', 'sse' or 'streamable-http'."
        )


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for `mnemosyne mcp`."""
    import argparse

    parser = argparse.ArgumentParser(description="Mnemosyne MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Bind address for SSE and streamable-http transports (default: "
            "127.0.0.1 -- loopback only). A non-loopback bind requires "
            "MNEMOSYNE_MCP_TOKEN on both transports, and streamable-http "
            "additionally requires MNEMOSYNE_MCP_ALLOWED_HOSTS (comma-separated "
            "Host header values, exact names or 'name:*'). Startup is refused "
            "when either is missing. MNEMOSYNE_MCP_ALLOWED_ORIGINS is optional "
            "and restricts browser origins."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE and streamable-http transports (default: 8080)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="Endpoint path for streamable-http transport (default: /mcp)",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        help=(
            "Force JSON-only responses on streamable-http (no SSE upgrade). "
            "Default: streamable responses with SSE upgrade."
        ),
    )
    parser.add_argument(
        "--bank",
        type=str,
        default=None,
        help="Default memory bank",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file to load before starting server",
    )
    args = parser.parse_args(argv)

    kwargs = {
        "transport": args.transport,
        "port": args.port,
        "bank": args.bank,
        "host": args.host,
        "path": args.path,
        "json_response": args.json_response,
    }
    if args.env_file is not None:
        kwargs["env_file"] = args.env_file

    run_mcp_server(**kwargs)


if __name__ == "__main__":
    main()
