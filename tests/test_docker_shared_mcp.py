"""Real network contract for the shared-memory Docker deployment.

Catches loss of shared visibility, author confusion and missing HTTP gates.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import pytest


@asynccontextmanager
async def agent(url, token):
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=15) as client:
        async with streamable_http_client(url, http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def payload(result):
    assert not result.is_error, result
    return json.loads(result.content[0].text)


@pytest.fixture
def shared_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MNEMOSYNE_", "OPENROUTER_", "OPENAI_"))}
    env.update(MNEMOSYNE_DATA_DIR=str(tmp_path), MNEMOSYNE_DISABLE_EMBEDDINGS="1",
               MNEMOSYNE_LLM_BACKEND="none", MNEMOSYNE_DEFAULT_SCOPE="global",
               MNEMOSYNE_MCP_TOKENS=json.dumps({"hermes": "test-hermes", "codex": "test-codex", "claude-code": "test-claude"}),
               MNEMOSYNE_MCP_ALLOWED_HOSTS=f"127.0.0.1:{port}")
    with (tmp_path / "server.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "mnemosyne.mcp_server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", str(port)], env=env, stdout=log, stderr=log)
        url = f"http://127.0.0.1:{port}/mcp"
        try:
            for _ in range(100):
                if proc.poll() is not None:
                    pytest.fail((tmp_path / "server.log").read_text())
                try:
                    if httpx.get(url, timeout=.2).status_code == 401:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(.05)
            else:
                pytest.fail("MCP listener did not become ready")
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_agents_share_memory_and_keep_authors(shared_server):
    async def exercise():
        async with agent(shared_server, "test-hermes") as a, agent(shared_server, "test-codex") as b:
            assert (await a.list_tools()).tools
            for session, marker in [(a, "Copper otter prefers vanilla tea"), (b, "Silver badger prefers cinnamon coffee")]:
                assert payload(await session.call_tool("mnemosyne_remember", {"content": marker}))["status"] == "stored"
            for session, query, author in [(b, "Copper otter", "hermes"), (a, "Silver badger", "codex")]:
                rows = payload(await session.call_tool("mnemosyne_recall", {"query": query}))["results"]
                assert any(query in r["content"] and r["author_id"] == author for r in rows)
        async with agent(shared_server, "test-claude") as c:
            rows = payload(await c.call_tool("mnemosyne_recall", {"query": "Copper otter"}))["results"]
            assert any("Copper otter" in r["content"] for r in rows)
            result = await c.call_tool("mnemosyne_remember", {"content": "Spoof must not be written", "author_id": "hermes"})
            assert json.loads(result.content[0].text)["status"] == "error"
    asyncio.run(exercise())


def test_http_auth_and_host_gates(shared_server):
    assert httpx.get(shared_server).status_code == 401
    assert httpx.get(shared_server, headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = httpx.get(shared_server, headers={"Authorization": "Bearer test-hermes", "Host": "untrusted.invalid"})
    assert response.status_code == 421


def test_healthcheck_probes_running_server(shared_server):
    from pathlib import Path
    from urllib.parse import urlsplit
    probe = Path(__file__).resolve().parents[1] / 'deploy/docker/healthcheck.py'
    env = dict(os.environ, MNEMOSYNE_MCP_HEALTH_PORT=str(urlsplit(shared_server).port),
               MNEMOSYNE_MCP_ALLOWED_HOSTS=f'127.0.0.1:{urlsplit(shared_server).port}',
               MNEMOSYNE_MCP_TOKENS='{"hermes":"test-hermes"}')
    result = subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    env['MNEMOSYNE_MCP_TOKENS'] = '{"hermes":"wrong"}'
    assert subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, timeout=15).returncode == 1
    env['MNEMOSYNE_MCP_HEALTH_PORT'] = '1'
    assert subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, timeout=15).returncode == 1
