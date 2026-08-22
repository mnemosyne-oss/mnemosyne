"""
Mnemosyne MCP Server -- stdio and SSE transports.

Usage:
    # stdio (default) -- for Claude Desktop, etc.
    mnemosyne mcp

    # SSE on loopback -- safe default, no auth required
    mnemosyne mcp --transport sse --port 8080

    # SSE exposed on LAN -- REQUIRES bearer token via env var
    MNEMOSYNE_MCP_TOKEN=my-secret-token mnemosyne mcp \\
        --transport sse --host 0.0.0.0 --port 8080

    # Specific bank
    mnemosyne mcp --bank project_a

Security note (S1, 2026-05-12):
    The SSE transport defaults to host=127.0.0.1 (loopback only). Binding
    to a non-loopback address (0.0.0.0, a LAN IP, etc.) requires the env
    var MNEMOSYNE_MCP_TOKEN to be set; clients must then send
    ``Authorization: Bearer <token>`` on every request. Without the token
    the server refuses to start. This prevents a LAN attacker from
    reading/writing/deleting the user's memory via an unauthenticated
    MCP endpoint.
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

from mnemosyne.mcp_tools import get_tool_definitions, handle_tool_call
from mnemosyne.runtime_context import set_request_token_name  # noqa: F401 (re-export)

# ---------------------------------------------------------------------------
# Security helpers (S1)
# ---------------------------------------------------------------------------

# Hosts treated as loopback-only; safe to expose without auth.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "ip6-localhost"})

_TOKEN_ENV = "MNEMOSYNE_MCP_TOKEN"
_TOKENS_ENV = "MNEMOSYNE_MCP_TOKENS"


def _is_loopback(host: str) -> bool:
    """Return True if `host` is a loopback bind that needs no auth."""
    return host.strip().lower() in _LOOPBACK_HOSTS


def _parse_tokens_env(raw: str) -> "dict[str, str]":
    """Parse MNEMOSYNE_MCP_TOKENS into an ordered {name: token} mapping.

    Accepts a JSON object of ``{"agent-name": "secret", ...}``. Raises
    RuntimeError with an actionable message on malformed JSON, empty
    names/tokens, or duplicate names -- refusing to silently degrade to
    fewer credentials than the operator configured.
    """
    import json as _json

    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError as e:
        raise RuntimeError(
            f"{_TOKENS_ENV} is not valid JSON ({e}). Expected an object "
            f'mapping token names to secrets, e.g. '
            f"'{{\"hermes-family\": \"tok1\", \"hermes-admin\": \"tok2\"}}'."
        ) from e
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"{_TOKENS_ENV} must be a JSON object mapping token names to "
            f"secrets; got {type(parsed).__name__}."
        )
    tokens: dict[str, str] = {}
    for name, token in parsed.items():
        name_s = str(name).strip()
        token_s = str(token).strip()
        if not name_s or not token_s:
            raise RuntimeError(
                f"{_TOKENS_ENV} contains an empty name or token; every "
                f"entry needs a non-empty name and secret."
            )
        if name_s in tokens:
            raise RuntimeError(
                f"{_TOKENS_ENV} contains duplicate name {name_s!r}."
            )
        tokens[name_s] = token_s
    return tokens


def _resolve_sse_auth(host: str) -> Tuple[bool, Optional[dict[str, str]]]:
    """Decide whether SSE needs bearer-token auth and which tokens apply.

    Returns (require_auth, tokens) where ``tokens`` maps token names to
    secrets. When the multi-token env var ``MNEMOSYNE_MCP_TOKENS`` is set
    (a JSON object), each named token is accepted and the *name* of the
    presented token is recorded as the author identity for tool calls
    (via ``MNEMOSYNE_AUTHOR_ID``-style resolution in mcp_tools), enabling
    per-agent audit trails from a single instance. The legacy single-token
    ``MNEMOSYNE_MCP_TOKEN`` remains fully supported.

    Raises RuntimeError when host is non-loopback and neither env var is
    set/parseable -- refusing to start an unauthenticated network-exposed
    MCP server.
    """
    if _is_loopback(host):
        return (False, None)
    multi_raw = (os.environ.get(_TOKENS_ENV) or "").strip()
    if multi_raw:
        return (True, _parse_tokens_env(multi_raw))
    token = (os.environ.get(_TOKEN_ENV) or "").strip()
    if not token:
        raise RuntimeError(
            f"Refusing to bind MCP SSE on non-loopback host {host!r} without "
            f"authentication. Set {_TOKENS_ENV} to a JSON object of named "
            f"secrets (multiple agents) or {_TOKEN_ENV} to a single secret, "
            f"and have clients send 'Authorization: Bearer <token>' on "
            f"each request. Or bind to 127.0.0.1 (the default) for local-only "
            f"use."
        )
    return (True, {"default": token})


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
        try:
            result = handle_tool_call(params.name, params.arguments or {})
            content = [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            return CallToolResult(content=content)
        except Exception as e:
            # SDK 2.x contract: return a CallToolResult with is_error=True so
            # clients can distinguish implementation failures from successful
            # calls. Preserves the existing error payload shape for backward
            # compatibility with any caller already parsing the error content.
            content = [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}, indent=2))]
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

    require_auth, tokens = _resolve_sse_auth(host)

    # Trailing slash required: SseServerTransport emits POST URUs as
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
        assert tokens is not None and len(tokens) > 0

        class _BearerTokenMiddleware:
            """Pure-ASGI bearer auth middleware (single- or multi-token).

            BaseHTTPMiddleware buffers the full response body before
            forwarding it to the client. SseServerTransport writes
            directly to the raw ASGI send callable, so
            BaseHTTPMiddleware raises:
              AssertionError: Unexpected message: http.response.start
            on every SSE connect, terminating the stream immediately.

            This pure-ASGI implementation forwards scope/receive/send
            untouched after auth so SSE frames are never buffered.

            Multi-token mode (MNEMOSYNE_MCP_TOKENS): the presented
            bearer token is matched against every configured secret
            (constant-time per candidate) and the *name* of the matched
            entry is stored in ``scope["state"]`` plus a contextvar,
            so tool handlers can attribute memories to the calling
            agent without any client-side cooperation.
            """

            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope.get("type") != "http":
                    await self.app(scope, receive, send)
                    return
                header = b""
                for k, v in scope.get("headers", []):
                    if k == b"authorization":
                        header = v
                        break
                if not header.startswith(b"Bearer "):
                    resp = JSONResponse(
                        {"error": "missing bearer token"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    await resp(scope, receive, send)
                    return
                presented = header[len(b"Bearer "):].strip()
                matched_name = None
                for name, secret in tokens.items():
                    if hmac.compare_digest(presented, secret.encode("utf-8")):
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
                # Per-agent identity: contextvar (consumed by mcp_tools'
                # author resolution) + ASGI scope state for inspection.
                set_request_token_name(matched_name)
                scope.setdefault("state", {})["mnemosyne_token_name"] = matched_name
                await self.app(scope, receive, send)

        middleware.append(Middleware(_BearerTokenMiddleware))
        logger.info(
            "MCP SSE bearer-token auth enabled (host=%s, tokens=%d). Clients "
            "must send 'Authorization: Bearer <token>' on every request.",
            host,
            len(tokens),
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

    Default host is 127.0.0.1 (loopback only). Binding non-loopback
    requires MNEMOSYNE_MCP_TOKEN -- see _resolve_sse_auth.
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
) -> None:
    """
    Run the Mnemosyne MCP server.

    Args:
        transport: "stdio" or "sse"
        port: Port for SSE transport (ignored for stdio)
        bank: Default bank for operations (optional)
        host: Bind address for SSE transport (default: 127.0.0.1 -- loopback
            only). Non-loopback hosts require MNEMOSYNE_MCP_TOKEN.
        env_file: Path to optional .env file to load before starting.
    """
    _load_dotenv(env_file)

    if bank:
        os.environ["MNEMOSYNE_MCP_BANK"] = bank

    if transport == "stdio":
        asyncio.run(_run_stdio())
    elif transport == "sse":
        asyncio.run(_run_sse(port=port, host=host))
    else:
        raise ValueError(f"Unknown transport: {transport}. Use 'stdio' or 'sse'.")


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for `mnemosyne mcp`."""
    import argparse

    parser = argparse.ArgumentParser(description="Mnemosyne MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Bind address for SSE transport (default: 127.0.0.1 -- loopback "
            "only). Use 0.0.0.0 to expose on LAN; this requires the "
            "MNEMOSYNE_MCP_TOKEN env var to be set."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)"
    )
    parser.add_argument(
        "--bank",
        type=str,
        default=None,
        help="Default memory bank"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file to load before starting server"
    )
    args = parser.parse_args(argv)

    kwargs = {
        "transport": args.transport,
        "port": args.port,
        "bank": args.bank,
        "host": args.host,
    }
    if args.env_file is not None:
        kwargs["env_file"] = args.env_file

    run_mcp_server(**kwargs)


if __name__ == "__main__":
    main()
