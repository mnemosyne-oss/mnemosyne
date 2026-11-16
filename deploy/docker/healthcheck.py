"""Bounded, authenticated liveness check; never reads or writes memories."""
import asyncio
import json
import logging
import os

# httpx2 is the HTTP client the mcp SDK's Streamable HTTP transport expects.
# It arrives transitively with mcp and is pinned in requirements.lock; revisit
# this import when upgrading mcp.
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def probe() -> None:
    tokens = json.loads(os.environ["MNEMOSYNE_MCP_TOKENS"])
    token = next(iter(tokens.values()))
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Missing token")
    port = int(os.environ.get("MNEMOSYNE_MCP_HEALTH_PORT", "8080"))
    host = os.environ["MNEMOSYNE_MCP_ALLOWED_HOSTS"].split(",")[0].strip().replace(":*", f":{port}")
    headers = {"Authorization": f"Bearer {token}", "Host": host}
    async with httpx2.AsyncClient(headers=headers, timeout=5, trust_env=False) as client:
        async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if not (await session.list_tools()).tools:
                    raise RuntimeError("No tools")


def main() -> int:
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(asyncio.wait_for(probe(), timeout=10))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
