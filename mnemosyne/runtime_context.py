"""Cross-transport request context (identity plumbing).

Holds per-request values that transports (e.g. the SSE bearer-token
middleware) record and tool handlers consume. Kept in a dedicated module
so ``mcp_tools`` can read it without importing the (MCP-dependent)
``mcp_server`` module -- avoiding a circular import.

Current entry:
- ``request_token_name``: name of the authenticated multi-token entry
  (``MNEMOSYNE_MCP_TOKENS``). ``mcp_tools._create_instance`` uses it as
  the author identity fallback so memories are attributed to the calling
  agent. Defaults to None everywhere (stdio transport, single-token
  mode) -- fully backward compatible.
"""
from __future__ import annotations

import contextvars

request_token_name: contextvars.ContextVar = contextvars.ContextVar(
    "mnemosyne_request_token_name", default=None
)


def set_request_token_name(name: str) -> None:
    """Record the authenticated token name for the current task."""
    request_token_name.set(name)


def get_request_token_name():
    """Return the authenticated token name (or None outside SSE/multi-token)."""
    return request_token_name.get()
