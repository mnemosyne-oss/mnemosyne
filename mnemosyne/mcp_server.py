"""
Mnemosyne MCP Server -- stdio, legacy SSE, and Streamable HTTP transports.

Usage:
    # stdio (default) -- for Claude Desktop, etc.
    mnemosyne mcp

    # SSE on loopback -- safe default, no auth required
    mnemosyne mcp --transport sse --port 8080

    # SSE exposed on LAN -- REQUIRES bearer token via env var
    MNEMOSYNE_MCP_TOKEN=my-secret-token mnemosyne mcp \\
        --transport sse --host 0.0.0.0 --port 8080

    # Streamable HTTP on loopback -- DNS-rebinding protection enabled
    mnemosyne mcp --transport streamable-http --port 8080

    # Specific bank
    mnemosyne mcp --bank project_a

Security note (S1, 2026-05-12):
    Network MCP transports default to host=127.0.0.1 (loopback only). Binding
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

from mnemosyne.mcp_tools import (
    get_tool_definitions,
    handle_tool_call,
    validate_mcp_extraction_policies,
)

# ---------------------------------------------------------------------------
# Security helpers (S1)
# ---------------------------------------------------------------------------

# Hosts treated as loopback-only; safe to expose without auth.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "ip6-localhost"})

_TOKEN_ENV = "MNEMOSYNE_MCP_TOKEN"


def _is_loopback(host: str) -> bool:
    """Return True if `host` is a loopback bind that needs no auth."""
    return host.strip().lower() in _LOOPBACK_HOSTS


def _resolve_mcp_auth(host: str) -> Tuple[bool, Optional[str]]:
    """Decide whether a network-exposed MCP transport needs bearer auth.

    Returns (require_auth, token). Raises RuntimeError when host is
    non-loopback and the MNEMOSYNE_MCP_TOKEN env var is unset/empty --
    refusing to start an unauthenticated network-exposed MCP server.
    """
    if _is_loopback(host):
        return (False, None)
    token = (os.environ.get(_TOKEN_ENV) or "").strip()
    if not token:
        raise RuntimeError(
            f"Refusing to bind MCP transport on non-loopback host {host!r} without "
            f"authentication. Set the {_TOKEN_ENV} env var to a strong random "
            f"secret and have clients send 'Authorization: Bearer <token>' on "
            f"each request. Or bind to 127.0.0.1 (the default) for local-only "
            f"use."
        )
    return (True, token)


def _resolve_sse_auth(host: str) -> Tuple[bool, Optional[str]]:
    """Backward-compatible name for the shared MCP transport auth policy."""
    return _resolve_mcp_auth(host)


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
    validate_mcp_extraction_policies()
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
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse
    except ImportError:
        raise RuntimeError(
            "SSE transport requires starlette and uvicorn. "
            "Run: pip install starlette uvicorn"
        )

    # Trailing slash required: SseServerTransport emits POST URIs as
    # /messages/ and Starlette Mount path-prefix matching needs it to
    # agree. Route("/messages") would 404 on every client POST.
    transport = SseServerTransport("/messages/")
    server = _build_mcp_server()

    async def handle_sse(request):
        async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        return JSONResponse({})

    routes = [
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        # transport.handle_post_message is an ASGI callable, not a
        # request-response endpoint. Mount (not Route) is required so
        # Starlette passes scope/receive/send directly without wrapping
        # the response. The trailing slash must match the transport path.
        Mount("/messages/", app=transport.handle_post_message),
    ]
    return _build_authenticated_mcp_app(routes, host=host, transport_name="SSE")


def _build_authenticated_mcp_app(routes, host: str, transport_name: str, lifespan=None):
    """Wrap MCP ASGI routes with the shared non-buffering bearer middleware."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse

    require_auth, token = _resolve_mcp_auth(host)
    middleware = []
    if require_auth:
        assert token is not None
        expected = token

        class _BearerTokenMiddleware:
            """Pure-ASGI bearer authentication safe for streamed responses."""

            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope.get("type") != "http":
                    await self.app(scope, receive, send)
                    return
                header = next(
                    (v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"authorization"),
                    "",
                )
                scheme, _, credentials = header.partition(" ")
                if scheme.casefold() != "bearer":
                    response = JSONResponse(
                        {"error": "missing bearer token"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    await response(scope, receive, send)
                    return
                if not hmac.compare_digest(credentials.strip(), expected):
                    response = JSONResponse(
                        {"error": "invalid bearer token"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    await response(scope, receive, send)
                    return
                await self.app(scope, receive, send)

        middleware.append(Middleware(_BearerTokenMiddleware))
        logger.info("MCP %s bearer-token auth enabled (host=%s).", transport_name, host)
    else:
        logger.info("MCP %s running loopback-only (host=%s); no auth required.", transport_name, host)
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


def _build_streamable_http_app(
    host: str = "127.0.0.1",
    *,
    session_idle_timeout: float = 1800,
):
    """Build a stateful Streamable HTTP app mounted at ``/mcp``."""
    if not _MCP_AVAILABLE:
        raise RuntimeError("MCP not installed. Run: pip install mnemosyne-memory[mcp]")

    from contextlib import asynccontextmanager

    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.server.transport_security import TransportSecuritySettings
        from starlette.routing import Route
    except ImportError as exc:
        raise RuntimeError(
            "Streamable HTTP transport requires starlette and a recent MCP SDK. "
            "Run: pip install starlette uvicorn 'mcp>=2.0.0,<3'"
        ) from exc

    security_settings = None
    if _is_loopback(host):
        security_settings = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "127.0.0.1:*",
                "127.0.0.1",
                "localhost:*",
                "localhost",
                "[::1]:*",
                "[::1]",
                "ip6-localhost:*",
                "ip6-localhost",
            ],
            allowed_origins=[
                "http://127.0.0.1:*",
                "http://127.0.0.1",
                "http://localhost:*",
                "http://localhost",
                "http://[::1]:*",
                "http://[::1]",
                "http://ip6-localhost:*",
                "http://ip6-localhost",
            ],
        )

    manager = StreamableHTTPSessionManager(
        _build_mcp_server(),
        security_settings=security_settings,
        session_idle_timeout=session_idle_timeout,
    )

    @asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    async def handle_streamable_http_request(scope, receive, send):
        """Delegate requests and remove transports terminated by DELETE."""
        session_id = None
        if scope.get("method") == "DELETE":
            session_id = next(
                (
                    value.decode("latin-1")
                    for name, value in scope.get("headers", [])
                    if name == b"mcp-session-id"
                ),
                None,
            )

        try:
            await manager.handle_request(scope, receive, send)
        finally:
            if session_id is not None:
                # MCP SDK 2.x does not consistently unregister a transport
                # after DELETE. These private registries are guarded because
                # cleanup is best effort across supported SDK releases.
                server_instances = getattr(manager, "_server_instances", None)
                session_owners = getattr(manager, "_session_owners", None)
                if isinstance(server_instances, dict) and isinstance(session_owners, dict):
                    transport = server_instances.get(session_id)
                    if transport is not None and getattr(transport, "is_terminated", False):
                        server_instances.pop(session_id, None)
                        session_owners.pop(session_id, None)

    class _StreamableHTTPRoute:
        """Keep the ASGI handler on the exact Streamable HTTP endpoint."""

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)

    app = _build_authenticated_mcp_app(
        [
            Route(
                "/mcp",
                endpoint=_StreamableHTTPRoute(handle_streamable_http_request),
                methods=["GET", "POST", "DELETE"],
            )
        ],
        host=host,
        transport_name="Streamable HTTP",
        lifespan=lifespan,
    )
    app.state.streamable_http_manager = manager
    return app


async def _run_sse(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Run MCP server over legacy SSE transport."""
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


async def _run_streamable_http(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Run MCP server over stateful Streamable HTTP at ``/mcp``."""
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "Streamable HTTP transport requires starlette and uvicorn. "
            "Run: pip install starlette uvicorn"
        )

    app = _build_streamable_http_app(host=host)
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
        transport: "stdio", "sse", or "streamable-http"
        port: Port for network transports (ignored for stdio)
        bank: Default bank for operations (optional)
        host: Bind address for network transports (default: 127.0.0.1 -- loopback
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
    elif transport == "streamable-http":
        asyncio.run(_run_streamable_http(port=port, host=host))
    else:
        raise ValueError(
            f"Unknown transport: {transport}. Use 'stdio', 'sse', or 'streamable-http'."
        )


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for `mnemosyne mcp`."""
    import argparse

    parser = argparse.ArgumentParser(description="Mnemosyne MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Bind address for network MCP transports (default: 127.0.0.1 -- loopback "
            "only). Use 0.0.0.0 to expose on LAN; this requires the "
            "MNEMOSYNE_MCP_TOKEN env var to be set."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for network MCP transports (default: 8080)"
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
