from __future__ import annotations

import json
from datetime import datetime, timedelta
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


def test_invalidate_missing_replacement_reports_its_own_status(tmp_path, monkeypatch):
    """A visible target with an invisible replacement must not be reported
    as if the target itself were missing."""
    provider, _ = _provider(tmp_path, monkeypatch)
    stored = _call(provider, "mnemosyne_remember", {
        "content": "target with an unreadable replacement", "source": "fact",
    })

    result = _call(provider, "mnemosyne_invalidate", {
        "memory_id": stored["memory_id"],
        "replacement_id": "0000000000000000",
    })

    assert result == {
        "status": "replacement_not_found",
        "memory_id": stored["memory_id"],
        "replacement_id": "0000000000000000",
        "bank": "private",
    }
    row = provider._beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
        (stored["memory_id"],),
    ).fetchone()
    assert tuple(row) == (None, None)


def test_invalidate_missing_target_stays_memory_not_found(tmp_path, monkeypatch):
    """The existing target-not-found status is preserved even when the
    replacement itself is valid."""
    provider, _ = _provider(tmp_path, monkeypatch)
    replacement = _call(provider, "mnemosyne_remember", {
        "content": "healthy replacement row", "source": "fact",
    })

    result = _call(provider, "mnemosyne_invalidate", {
        "memory_id": "ffffffffffffffff",
        "replacement_id": replacement["memory_id"],
    })

    assert result["status"] == "memory_not_found"
    assert result["bank"] == "private"


def test_invalidate_inactive_target_stays_memory_not_found(tmp_path, monkeypatch):
    """The replacement is healthy; the target is not. Re-invalidating an
    already-superseded row must report the target, not blame the live
    replacement (review on #1113) — invalidate() only touches ACTIVE rows."""
    provider, _ = _provider(tmp_path, monkeypatch)
    first = _call(provider, "mnemosyne_remember", {
        "content": "row that gets superseded first", "source": "fact",
    })
    successor = _call(provider, "mnemosyne_remember", {
        "content": "healthy successor row", "source": "fact",
    })

    retired = _call(provider, "mnemosyne_invalidate", {
        "memory_id": first["memory_id"],
        "replacement_id": successor["memory_id"],
    })
    assert retired["status"] == "invalidated"

    result = _call(provider, "mnemosyne_invalidate", {
        "memory_id": first["memory_id"],
        "replacement_id": successor["memory_id"],
    })
    assert result["status"] == "memory_not_found"


def test_invalidate_expired_target_stays_memory_not_found(tmp_path, monkeypatch):
    """The predicate under test also handles `valid_until`, not just
    superseded_by (review on #1113, second round). The expired stamp is
    written in the same naive-local ISO family that
    BeamMemory.invalidate(replacement_id=...) compares against, one day in
    the past — so the row is expired on every host timezone, and the case
    isolates the valid_until half of the predicate (superseded_by stays
    NULL). A bare get() still returns the row, which is exactly what used
    to make the failed invalidation wrongly blame the healthy replacement."""
    provider, _ = _provider(tmp_path, monkeypatch)
    target = _call(provider, "mnemosyne_remember", {
        "content": "row expired through valid_until", "source": "fact",
    })
    replacement = _call(provider, "mnemosyne_remember", {
        "content": "healthy replacement row", "source": "fact",
    })

    past = (datetime.now() - timedelta(days=1)).isoformat()
    provider._beam.conn.execute(
        "UPDATE working_memory SET valid_until = ? WHERE id = ?",
        (past, target["memory_id"]),
    )
    provider._beam.conn.commit()

    result = _call(provider, "mnemosyne_invalidate", {
        "memory_id": target["memory_id"],
        "replacement_id": replacement["memory_id"],
    })
    assert result["status"] == "memory_not_found"


def test_invalidate_rejects_self_replacement(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)
    stored = _call(provider, "mnemosyne_remember", {
        "content": "row guarded against self-supersede", "source": "fact",
    })

    result = _call(provider, "mnemosyne_invalidate", {
        "memory_id": stored["memory_id"],
        "replacement_id": stored["memory_id"],
    })

    assert "error" in result
    row = provider._beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
        (stored["memory_id"],),
    ).fetchone()
    assert tuple(row) == (None, None)


def test_invalidate_replacement_must_be_visible_in_the_routed_bank(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch)
    surface = _call(provider, "mnemosyne_shared_remember", {
        "content": "surface target for cross-bank check", "kind": "meta",
    })
    private = _call(provider, "mnemosyne_remember", {
        "content": "private replacement lives in the other bank", "source": "fact",
    })

    result = _call(provider, "mnemosyne_invalidate", {
        "memory_id": surface["memory_id"],
        "replacement_id": private["memory_id"],
    })
    assert result["status"] == "replacement_not_found"
    assert result["bank"] == "surface"

    surface_repl = _call(provider, "mnemosyne_shared_remember", {
        "content": "surface replacement for chaining", "kind": "meta",
    })
    ok = _call(provider, "mnemosyne_invalidate", {
        "memory_id": surface["memory_id"],
        "replacement_id": surface_repl["memory_id"],
    })
    assert ok["status"] == "invalidated"
    assert ok["bank"] == "surface"
    row = provider._surface_beam.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?",
        (surface["memory_id"],),
    ).fetchone()
    assert row[0] == surface_repl["memory_id"]


def test_invalidate_explicit_surface_selector_beats_prefix_inference(tmp_path, monkeypatch):
    """A bare (no sf_ prefix) id must invalidate through the surface beam when
    the caller names the bank, and through the private bank when it does not —
    so prefix inference can never mask a broken selector."""
    provider, _ = _provider(tmp_path, monkeypatch)
    provider._ensure_surface_beam()
    target = provider._surface_beam.remember(
        "explicit selector target", source="surface_manual", scope="global",
    )
    replacement = provider._surface_beam.remember(
        "explicit selector replacement", source="surface_manual", scope="global",
    )
    assert not target.startswith("sf_")

    # Without the selector the bare id resolves to the private bank, where it
    # does not exist.
    inferred = _call(provider, "mnemosyne_invalidate", {"memory_id": target})
    assert inferred == {
        "status": "memory_not_found", "memory_id": target, "bank": "private",
    }

    # Selector plus no replacement: the surface beam answers.
    out = _call(provider, "mnemosyne_invalidate", {"memory_id": target, "bank": "surface"})
    assert out == {"status": "invalidated", "memory_id": target, "bank": "surface"}
    row = provider._surface_beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
        (target,),
    ).fetchone()
    assert row[0] is not None and row[1] is None

    # Selector plus replacement: both ids resolve on the surface and chain.
    target2 = provider._surface_beam.remember(
        "chained selector target", source="surface_manual", scope="global",
    )
    out = _call(provider, "mnemosyne_invalidate", {
        "memory_id": target2, "replacement_id": replacement, "bank": "surface",
    })
    assert out["status"] == "invalidated"
    assert out["bank"] == "surface"
    row = provider._surface_beam.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?",
        (target2,),
    ).fetchone()
    assert row[0] == replacement

