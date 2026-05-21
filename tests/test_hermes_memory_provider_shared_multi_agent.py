from __future__ import annotations

import json
from pathlib import Path

from hermes_memory_provider import MnemosyneMemoryProvider


AGENTS = ["Mob", "Rook", "Vale", "Nia", "Kite"]


def _call(provider: MnemosyneMemoryProvider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


def _agent_provider(tmp_path: Path, monkeypatch, agent: str, shared_db: Path) -> MnemosyneMemoryProvider:
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "mnemosyne-data"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    hermes_home = tmp_path / "profiles" / agent
    hermes_home.mkdir(parents=True, exist_ok=True)
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id=f"{agent.lower()}-session",
        hermes_home=str(hermes_home),
        agent_identity=agent,
        profile_isolation=True,
        shared_surface_path=str(shared_db),
    )
    assert provider._beam is not None
    assert provider._resolve_profile_bank() == agent.lower()
    return provider


def test_five_profile_agents_share_surface_but_keep_private_memories_isolated(tmp_path, monkeypatch):
    shared_db = tmp_path / "mnemosyne-data" / "shared" / "mnemosyne.db"
    providers = {
        agent: _agent_provider(tmp_path, monkeypatch, agent, shared_db)
        for agent in AGENTS
    }

    # Each agent writes one private memory and one shared surface memory.
    for agent, provider in providers.items():
        private = _call(provider, "mnemosyne_remember", {
            "content": f"Private {agent} scratch note should stay in {agent} bank only",
            "source": "fact",
            "importance": 0.8,
        })
        shared = _call(provider, "mnemosyne_shared_remember", {
            "content": f"Surface meta: {agent} project root lives at /tmp/{agent.lower()}-project",
            "kind": "meta",
            "importance": 0.8,
            "veracity": "stated",
        })
        assert private["status"] == "stored"
        assert shared["status"] == "stored_shared"

    # Every agent can recall every shared surface memory from the shared DB.
    for reader, provider in providers.items():
        result = _call(provider, "mnemosyne_shared_recall", {"query": "project root", "limit": 10})
        contents = [row.get("content", "") for row in result["results"]]
        assert result["count"] >= len(AGENTS), reader
        for agent in AGENTS:
            assert any(f"{agent} project root lives" in content for content in contents), (reader, agent, contents)
        assert all(row.get("shared_surface") is True for row in result["results"])
        assert all(row.get("bank") == "surface" for row in result["results"])

    # Shared DB must not contain private scratch notes.
    any_provider = providers[AGENTS[0]]
    _call(any_provider, "mnemosyne_shared_stats", {})
    leaked = any_provider._surface_beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content LIKE ?",
        ("%scratch note should stay%",),
    ).fetchone()[0]
    assert leaked == 0

    # Private profile DBs are separate paths.
    private_paths = {agent: provider._beam.db_path for agent, provider in providers.items()}
    assert len(set(private_paths.values())) == len(AGENTS)
    assert all(path != shared_db for path in private_paths.values())


def test_five_profile_agents_reject_raw_chat_shared_dump(tmp_path, monkeypatch):
    shared_db = tmp_path / "mnemosyne-data" / "shared" / "mnemosyne.db"
    providers = {
        agent: _agent_provider(tmp_path, monkeypatch, agent, shared_db)
        for agent in AGENTS
    }

    for agent, provider in providers.items():
        result = _call(provider, "mnemosyne_shared_remember", {
            "content": f"[USER] raw chat dump from {agent} must not be shared",
            "kind": "meta",
        })
        assert result == {"error": "raw conversation content is not allowed in shared memory"}

    stats_provider = providers[AGENTS[0]]
    _call(stats_provider, "mnemosyne_shared_stats", {})
    count = stats_provider._surface_beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content LIKE ?",
        ("%raw chat dump%",),
    ).fetchone()[0]
    assert count == 0
