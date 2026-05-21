from __future__ import annotations

import json
from pathlib import Path

from hermes_memory_provider import MnemosyneMemoryProvider


def _provider(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "mnemosyne-data"
    hermes_home = tmp_path / "profiles" / "Mob"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir / "private"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id="mob-session",
        hermes_home=str(hermes_home),
        agent_identity="Mob",
        shared_surface_path=str(data_dir / "shared" / "mnemosyne.db"),
    )
    assert provider._beam is not None
    return provider, data_dir


def _call(provider: MnemosyneMemoryProvider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


def test_shared_surface_db_uses_configured_path(tmp_path, monkeypatch):
    provider, data_dir = _provider(tmp_path, monkeypatch)

    stats = _call(provider, "mnemosyne_shared_stats", {})

    assert stats["shared_db"] == str(data_dir / "shared" / "mnemosyne.db")
    assert provider._shared_surface_path.exists()
    assert provider._beam.db_path != provider._surface_beam.db_path


def test_shared_remember_stores_global_surface_memory(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)

    result = _call(provider, "mnemosyne_shared_remember", {
        "content": "Project root lives at /tmp/project",
        "kind": "meta",
        "importance": 0.8,
        "veracity": "stated",
    })

    assert result["status"] == "stored_shared"
    assert result["memory_id"].startswith("sf_")
    row = provider._surface_beam.conn.execute(
        "SELECT content, source, scope FROM working_memory WHERE id = ?",
        (result["memory_id"],),
    ).fetchone()
    assert row is not None
    assert row[0] == "Surface meta: Project root lives at /tmp/project"
    assert row[1] == "surface_manual"
    assert row[2] == "global"


def test_shared_remember_is_idempotent_for_same_content(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)
    args = {"content": "Surface meta: Mob project lives at /tmp/mob", "kind": "meta"}

    first = _call(provider, "mnemosyne_shared_remember", args)
    second = _call(provider, "mnemosyne_shared_remember", args)

    assert first["status"] == "stored_shared"
    assert second["status"] == "existing_shared"
    assert first["memory_id"] == second["memory_id"]
    count = provider._surface_beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?",
        (first["memory_id"],),
    ).fetchone()[0]
    assert count == 1


def test_shared_recall_tags_rows_as_surface_bank(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)
    _call(provider, "mnemosyne_shared_remember", {
        "content": "Surface meta: Mob wiki lives at /tmp/mob-wiki",
        "kind": "meta",
    })

    result = _call(provider, "mnemosyne_shared_recall", {"query": "mob wiki", "limit": 5})

    assert result["count"] >= 1
    match = next(r for r in result["results"] if "Mob wiki" in r.get("content", ""))
    assert match["shared_surface"] is True
    assert match["bank"] == "surface"


def test_shared_forget_deletes_then_reports_not_found(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)
    stored = _call(provider, "mnemosyne_shared_remember", {
        "content": "Surface meta: temporary shared fact",
        "kind": "meta",
    })

    deleted = _call(provider, "mnemosyne_shared_forget", {"memory_id": stored["memory_id"]})
    missing = _call(provider, "mnemosyne_shared_forget", {"memory_id": stored["memory_id"]})
    recalled = _call(provider, "mnemosyne_shared_recall", {"query": "temporary shared fact", "limit": 5})

    assert deleted["status"] == "deleted"
    assert missing["status"] == "not_found"
    assert all("temporary shared fact" not in r.get("content", "") for r in recalled["results"])


def test_shared_stats_returns_counts_and_path(tmp_path, monkeypatch):
    provider, data_dir = _provider(tmp_path, monkeypatch)

    stats = _call(provider, "mnemosyne_shared_stats", {})

    assert stats["provider"] == "mnemosyne_shared"
    assert stats["shared_db"] == str(data_dir / "shared" / "mnemosyne.db")
    assert "working" in stats
    assert "episodic" in stats


def test_private_remember_does_not_write_shared_db(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)

    private = _call(provider, "mnemosyne_remember", {
        "content": "Private Mob-only fact",
        "source": "fact",
        "importance": 0.8,
    })
    _call(provider, "mnemosyne_shared_stats", {})
    shared_count = provider._surface_beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content LIKE ?",
        ("%Private Mob-only fact%",),
    ).fetchone()[0]

    assert private["status"] == "stored"
    assert shared_count == 0

