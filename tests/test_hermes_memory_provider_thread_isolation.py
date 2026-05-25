"""Test Telegram thread/topic isolation for Mnemosyne via gateway_session_key.

Verifies that:
1. Different gateway_session_keys isolate session-scoped memories per thread
2. scope='global' memories still surface across all threads
3. The fallback (no gateway_session_key) preserves legacy session behavior
4. Prefetch respects thread boundaries
"""
from __future__ import annotations

import json
from pathlib import Path

from hermes_memory_provider import MnemosyneMemoryProvider


def _make_provider(
    tmp_path: Path,
    monkeypatch,
    session_id: str = "test-session",
    gateway_session_key: str | None = None,
    agent_identity: str = "test",
) -> MnemosyneMemoryProvider:
    """Create a MnemosyneMemoryProvider with optional gateway_session_key."""
    data_dir = tmp_path / "mnemosyne-data"
    hermes_home = tmp_path / "profiles" / agent_identity
    hermes_home.mkdir(parents=True)
    (data_dir / "private").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir / "private"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    provider = MnemosyneMemoryProvider()
    kwargs = dict(
        session_id=session_id,
        hermes_home=str(hermes_home),
        agent_identity=agent_identity,
    )
    if gateway_session_key is not None:
        kwargs["gateway_session_key"] = gateway_session_key
    provider.initialize(**kwargs)
    assert provider._beam is not None
    return provider


def _remember(provider, content: str, scope: str = "session", **extra):
    """Store a memory via the provider's mnemosyne_remember tool handler."""
    args = {"content": content, "scope": scope, **extra}
    return json.loads(provider.handle_tool_call("mnemosyne_remember", args))


def _recall(provider, query: str = "", **extra):
    """Recall memories via the provider's mnemosyne_recall tool handler."""
    args = {"query": query, **extra}
    return json.loads(provider.handle_tool_call("mnemosyne_recall", args))


# ---- Tests ----


def test_gateway_session_key_isolates_session_memories(tmp_path, monkeypatch):
    """Two threads with different gateway_session_keys should NOT see each
    other's session-scoped memories (but scope='global' should cross)."""
    # Thread A
    prov_a = _make_provider(
        tmp_path, monkeypatch,
        session_id="sess-a",
        gateway_session_key="agent:main:telegram:dm:12345:11111",
        agent_identity="thread_a",
    )
    # Thread B
    prov_b = _make_provider(
        tmp_path, monkeypatch,
        session_id="sess-b",
        gateway_session_key="agent:main:telegram:dm:12345:22222",
        agent_identity="thread_b",
    )

    # Store session-scoped memories in each thread
    r1 = _remember(prov_a, "Secret A: the sky is green", scope="session")
    assert r1.get("status") == "stored", f"remember A failed: {r1}"

    r2 = _remember(prov_b, "Secret B: the ocean is purple", scope="session")
    assert r2.get("status") == "stored", f"remember B failed: {r2}"

    # Store a global memory (should be visible everywhere)
    r3 = _remember(prov_a, "Global: water is wet", scope="global")
    assert r3.get("status") == "stored", f"global remember failed: {r3}"

    # Thread A recalls: should see its own session memory + global
    recall_a = _recall(prov_a, query="secret")
    a_contents = {r["content"] for r in recall_a.get("results", [])}
    assert "Secret A: the sky is green" in a_contents, \
        f"Thread A should see its own secret, got: {a_contents}"
    assert "Secret B: the ocean is purple" not in a_contents, \
        f"Thread A should NOT see thread B's secret, got: {a_contents}"

    # Both threads should see global memory
    recall_a_all = _recall(prov_a, query="global")
    a_global_contents = {r["content"] for r in recall_a_all.get("results", [])}
    assert "Global: water is wet" in a_global_contents, \
        f"Thread A should see global memory, got: {a_global_contents}"

    recall_b_all = _recall(prov_b, query="global")
    b_global_contents = {r["content"] for r in recall_b_all.get("results", [])}
    assert "Global: water is wet" in b_global_contents, \
        f"Thread B should see global memory, got: {b_global_contents}"


def test_no_gateway_session_key_falls_back_to_session_id(tmp_path, monkeypatch):
    """When gateway_session_key is not provided, session_id is used as before
    (backward compatibility with CLI / non-gateway sessions)."""
    prov = _make_provider(
        tmp_path, monkeypatch,
        session_id="legacy-cli-session",
        gateway_session_key=None,
    )
    assert prov._session_id == "hermes_legacy-cli-session", \
        f"Expected hermes_legacy-cli-session, got {prov._session_id}"

    # Storing and recalling should work as before
    r = _remember(prov, "CLI memory: hello world", scope="session")
    assert r.get("status") == "stored", f"CLI remember failed: {r}"

    recall = _recall(prov, query="hello")
    contents = {r["content"] for r in recall.get("results", [])}
    assert "CLI memory: hello world" in contents, \
        f"CLI recall should work, got: {contents}"


def test_gateway_session_key_sanitized_for_session_id(tmp_path, monkeypatch):
    """gateway_session_key with colons/special chars is valid as a session_id
    (SQLite TEXT column handles any string)."""
    prov = _make_provider(
        tmp_path, monkeypatch,
        session_id="dummy",
        gateway_session_key="agent:main:telegram:dm:2032169058:42964",
    )
    expected = "hermes_agent:main:telegram:dm:2032169058:42964"
    assert prov._session_id == expected, \
        f"Expected {expected}, got {prov._session_id}"

    # Make sure storing and recalling works with this session_id
    r = _remember(prov, "Memory in topic 42964", scope="session")
    assert r.get("status") == "stored", f"Remember failed: {r}"

    recall = _recall(prov, query="topic")
    contents = {r["content"] for r in recall.get("results", [])}
    assert "Memory in topic 42964" in contents, \
        f"Recall should work with colon session_id, got: {contents}"


def test_empty_gateway_session_key_falls_back(tmp_path, monkeypatch):
    """An empty gateway_session_key should be treated the same as None
    (falls back to session_id)."""
    prov = _make_provider(
        tmp_path, monkeypatch,
        session_id="fallback-test",
        gateway_session_key="",
    )
    assert prov._session_id == "hermes_fallback-test", \
        f"Expected hermes_fallback-test, got {prov._session_id}"


def test_prefetch_scopes_to_thread(tmp_path, monkeypatch):
    """prefetch() should only return memories from the current thread's session,
    plus scope='global' memories."""
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "")  # default in user's env

    prov_a = _make_provider(
        tmp_path, monkeypatch,
        session_id="prefetch-a",
        gateway_session_key="agent:main:telegram:dm:12345:aaaaa",
        agent_identity="prefetch_a",
    )
    prov_b = _make_provider(
        tmp_path, monkeypatch,
        session_id="prefetch-b",
        gateway_session_key="agent:main:telegram:dm:12345:bbbbb",
        agent_identity="prefetch_b",
    )

    # Store via the regular remember tool
    _remember(prov_a, "Secret A: the sky is green", scope="session")
    _remember(prov_b, "Secret B: the ocean is purple", scope="session")
    _remember(prov_a, "Global: all animals eat", scope="global")

    # Prefetch with a query broad enough to match all memories.
    # prefetch() returns formatted string, not JSON.
    prefetch_a_output = prov_a.prefetch(query="animals nature sky ocean")
    prefetch_b_output = prov_b.prefetch(query="animals nature sky ocean")

    # Thread A should see its session memory + global
    assert "Secret A: the sky is green" in prefetch_a_output, \
        f"Thread A prefetch missing its memory, got:\n{prefetch_a_output}"
    assert "Secret B: the ocean is purple" not in prefetch_a_output, \
        f"Thread A prefetch should NOT have thread B's memory, got:\n{prefetch_a_output}"
    assert "Global: all animals eat" in prefetch_a_output, \
        f"Thread A prefetch should have global memory, got:\n{prefetch_a_output}"

    # Thread B should see its session memory + global
    assert "Secret B: the ocean is purple" in prefetch_b_output, \
        f"Thread B prefetch missing its memory, got:\n{prefetch_b_output}"
    assert "Secret A: the sky is green" not in prefetch_b_output, \
        f"Thread B prefetch should NOT have thread A's memory, got:\n{prefetch_b_output}"
    assert "Global: all animals eat" in prefetch_b_output, \
        f"Thread B prefetch should have global memory, got:\n{prefetch_b_output}"
