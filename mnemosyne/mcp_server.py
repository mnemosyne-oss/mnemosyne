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
import sys
import json
import asyncio
import logging
from typing import Optional, Tuple
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Security helpers (S1)
# ---------------------------------------------------------------------------

# Hosts treated as loopback-only; safe to expose without auth.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "ip6-localhost"})

_TOKEN_ENV = "MNEMOSYNE_MCP_TOKEN"


def _is_loopback(host: str) -> bool:
    """Return True if `host` is a loopback bind that needs no auth."""
    return host.strip().lower() in _LOOPBACK_HOSTS


def _resolve_sse_auth(host: str) -> Tuple[bool, Optional[str]]:
    """Decide whether SSE needs bearer-token auth and what the token is.

    Returns (require_auth, token). Raises RuntimeError when host is
    non-loopback and the MNEMOSYNE_MCP_TOKEN env var is unset/empty --
    refusing to start an unauthenticated network-exposed MCP server.
    """
    if _is_loopback(host):
        return (False, None)
    token = (os.environ.get(_TOKEN_ENV) or "").strip()
    if not token:
        raise RuntimeError(
            f"Refusing to bind MCP SSE on non-loopback host {host!r} without "
            f"authentication. Set the {_TOKEN_ENV} env var to a strong random "
            f"secret and have clients send 'Authorization: Bearer <token>' on "
            f"each request. Or bind to 127.0.0.1 (the default) for local-only "
            f"use."
        )
    return (True, token)


# ---------------------------------------------------------------------------
# Server Setup
# ---------------------------------------------------------------------------

async def _run_stdio() -> None:
    """Run MCP server over stdio transport."""
    if not _MCP_AVAILABLE:
        raise RuntimeError("MCP not installed. Run: pip install mnemosyne-memory[mcp]")

    server = Server("mnemosyne")

    @server.list_tools()
    async def list_tools():
        return get_tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        try:
            result = handle_tool_call(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}, indent=2))]

    async with stdio_server(server) as (read_stream, write_stream):
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
        from starlette.routing import Route
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
    except ImportError:
        raise RuntimeError(
            "SSE transport requires starlette and uvicorn. "
            "Run: pip install starlette uvicorn"
        )

    require_auth, token = _resolve_sse_auth(host)

    transport = SseServerTransport("/messages")
    server = Server("mnemosyne")

    @server.list_tools()
    async def list_tools():
        return get_tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        try:
            result = handle_tool_call(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}, indent=2))]

    async def handle_sse(request):
        async with transport.connect_sse(request.scope, request.receive, request.send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    async def handle_messages(request):
        await transport.handle_post_message(request.scope, request.receive, request.send)

    middleware = []
    if require_auth:
        expected = token

        class _BearerTokenMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                header = request.headers.get("authorization", "")
                if not header.startswith("Bearer "):
                    return JSONResponse(
                        {"error": "missing bearer token"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                presented = header[len("Bearer "):].strip()
                if not hmac.compare_digest(presented, expected):
                    return JSONResponse(
                        {"error": "invalid bearer token"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return await call_next(request)

        middleware.append(Middleware(_BearerTokenMiddleware))
        logger.info(
            "MCP SSE bearer-token auth enabled (host=%s). Clients must send "
            "'Authorization: Bearer <token>' on every request.",
            host,
        )
    else:
        logger.info(
            "MCP SSE running loopback-only (host=%s); no auth required.",
            host,
        )

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
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

def run_mcp_server(
    transport: str = "stdio",
    port: int = 8080,
    bank: Optional[str] = None,
    host: str = "127.0.0.1",
) -> None:
    """
    Run the Mnemosyne MCP server.

    Args:
        transport: "stdio" or "sse"
        port: Port for SSE transport (ignored for stdio)
        bank: Default bank for operations (optional)
        host: Bind address for SSE transport (default: 127.0.0.1 -- loopback
            only). Non-loopback hosts require MNEMOSYNE_MCP_TOKEN.
    """
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
    args = parser.parse_args(argv)

    run_mcp_server(transport=args.transport, port=args.port, bank=args.bank, host=args.host)


if __name__ == "__main__":
    main()
