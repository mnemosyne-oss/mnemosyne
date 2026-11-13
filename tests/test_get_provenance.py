"""Identity and scope provenance for direct BEAM reads."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.core.beam import BeamMemory


def _use_empty_config(tmp_path, monkeypatch):
    """Isolate provenance checks from the user's configured write defaults."""
    data_dir = tmp_path / "config-data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))


def test_get_returns_working_memory_author_and_scope(tmp_path, monkeypatch):
    """Direct working-memory reads preserve the writer's identity and scope."""
    _use_empty_config(tmp_path, monkeypatch)
    memory = BeamMemory(
        session_id="fleet",
        db_path=tmp_path / "mnemosyne.db",
        author_id="repo-admin",
        author_type="agent",
    )
    memory_id = memory.remember(
        "get provenance working sentinel",
        source="test",
        scope="global",
    )

    result = memory.get(memory_id)

    assert result["author_id"] == "repo-admin"
    assert result["author_type"] == "agent"
    assert result["scope"] == "global"


def test_get_returns_episodic_memory_author_and_scope(tmp_path, monkeypatch):
    """Direct episodic reads preserve writer provenance and event dates."""
    _use_empty_config(tmp_path, monkeypatch)
    memory = BeamMemory(
        session_id="fleet",
        db_path=tmp_path / "mnemosyne.db",
        author_id="repo-admin",
        author_type="agent",
    )
    memory_id = memory.consolidate_to_episodic(
        "get provenance episodic sentinel",
        source_wm_ids=[],
        source="test",
        scope="global",
        event_date="2026-09-26",
        event_date_precision="day",
    )

    result = memory.get(memory_id)

    assert result["author_id"] == "repo-admin"
    assert result["author_type"] == "agent"
    assert result["scope"] == "global"
    assert result["event_date"] == "2026-09-26"
    assert result["event_date_precision"] == "day"


@pytest.mark.parametrize("memory_store", ["working", "episodic"])
@pytest.mark.parametrize(
    ("channel_id", "trust_tier"),
    [("cursor", "EXTERNAL_WRITE"), ("import-bank", "IMPORTED"), (None, None)],
)
def test_get_returns_stored_channel_and_trust_tier(
    tmp_path, monkeypatch, memory_store, channel_id, trust_tier
):
    """Read stored provenance, including SQL NULL, without writing either tier."""
    _use_empty_config(tmp_path, monkeypatch)
    memory = BeamMemory(
        session_id="fleet",
        db_path=tmp_path / "mnemosyne.db",
        channel_id="current-channel",
    )
    if memory_store == "working":
        memory_id = memory.remember("stored working provenance", source="test")
    else:
        memory_id = memory.consolidate_to_episodic(
            "stored episodic provenance", source_wm_ids=[], source="test"
        )

    # Existing/imported rows can differ from the current writer's identity,
    # including nullable provenance in imported data.
    memory.conn.execute(
        f"UPDATE {memory_store}_memory SET channel_id = ?, trust_tier = ? WHERE id = ?",
        (channel_id, trust_tier, memory_id),
    )
    memory.conn.commit()
    changes_before = memory.conn.total_changes

    result = memory.get(memory_id)

    assert result["memory_store"] == memory_store
    assert result["channel_id"] == channel_id
    assert result["trust_tier"] == trust_tier
    assert memory.conn.total_changes == changes_before


@pytest.fixture
def legacy_provenance_db(tmp_path, monkeypatch):
    """Create both pre-identity memory tables, with rows but no provenance columns."""
    _use_empty_config(tmp_path, monkeypatch)
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE working_memory (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT,
                timestamp TEXT, session_id TEXT DEFAULT 'default',
                importance REAL DEFAULT 0.5, metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE episodic_memory (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL, source TEXT, timestamp TEXT,
                session_id TEXT DEFAULT 'default', importance REAL DEFAULT 0.5,
                metadata_json TEXT, summary_of TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO working_memory (id, content, session_id)
            VALUES ('legacy-working', 'legacy working provenance', 'legacy');
            INSERT INTO episodic_memory (id, content, session_id)
            VALUES ('legacy-episodic', 'legacy episodic provenance', 'legacy');
        """)
        for pragma in ("PRAGMA table_info(working_memory)", "PRAGMA table_info(episodic_memory)"):
            columns = {row[1] for row in conn.execute(pragma)}
            assert "channel_id" not in columns
            assert "trust_tier" not in columns
    return db_path


@pytest.mark.parametrize("memory_store", ["working", "episodic"])
def test_get_returns_migrated_trust_default(legacy_provenance_db, memory_store):
    """Read the STATED trust default installed by legacy-schema migration."""
    memory = BeamMemory(session_id="legacy", db_path=legacy_provenance_db)

    result = memory.get(f"legacy-{memory_store}")

    assert result["memory_store"] == memory_store
    assert result["channel_id"] is None
    assert result["trust_tier"] == "STATED"


@pytest.mark.parametrize("memory_store", ["working", "episodic"])
def test_get_provenance_keeps_session_visibility(tmp_path, monkeypatch, memory_store):
    """Return the writer's provenance while retaining private-row isolation."""
    _use_empty_config(tmp_path, monkeypatch)
    writer = BeamMemory(
        session_id="writer", db_path=tmp_path / "mnemosyne.db", channel_id="cursor"
    )
    if memory_store == "working":
        global_id = writer.remember("shared provenance", source="test", scope="global")
        private_id = writer.remember("private provenance", source="test", scope="session")
    else:
        global_id = writer.consolidate_to_episodic(
            "shared provenance", source_wm_ids=[], source="test", scope="global"
        )
        private_id = writer.consolidate_to_episodic(
            "private provenance", source_wm_ids=[], source="test", scope="session"
        )

    reader = BeamMemory(
        session_id="reader", db_path=writer.db_path, channel_id="reader-channel"
    )
    assert writer.get(private_id)["channel_id"] == "cursor"
    result = reader.get(global_id)
    assert result["memory_store"] == memory_store
    assert result["channel_id"] == "cursor"
    assert result["trust_tier"] == "STATED"
    assert reader.get(private_id) is None
    assert reader.get("missing-memory-id") is None


@pytest.mark.parametrize("memory_store", ["working", "episodic"])
@pytest.mark.parametrize(
    ("channel_id", "trust_tier"),
    [("cursor", "EXTERNAL_WRITE"), (None, "IMPORTED"), ("archive", None), (None, None)],
)
def test_json_roundtrip_preserves_stored_provenance(
    tmp_path, monkeypatch, memory_store, channel_id, trust_tier
):
    """Backup, restore and forced restore retain values and explicit NULLs."""
    _use_empty_config(tmp_path, monkeypatch)
    source = BeamMemory(session_id="fleet", db_path=tmp_path / "source.db")
    target = BeamMemory(
        session_id="fleet", db_path=tmp_path / "target.db", channel_id="reader"
    )
    try:
        if memory_store == "working":
            memory_id = source.remember("backup provenance sentinel", source="mcp")
        else:
            memory_id = source.consolidate_to_episodic(
                "backup provenance sentinel", [], source="mcp"
            )
        source.conn.execute(
            f"UPDATE {memory_store}_memory SET channel_id = ?, trust_tier = ? WHERE id = ?",
            (channel_id, trust_tier, memory_id),
        )
        source.conn.commit()
        payload = json.loads(json.dumps(source.export_to_dict()))
        exported = next(row for row in payload[f"{memory_store}_memory"] if row["id"] == memory_id)
        assert exported["channel_id"] == channel_id
        assert exported["trust_tier"] == trust_tier

        stats = target.import_from_dict(payload)
        assert stats[f"{memory_store}_memory"]["inserted"] == 1
        result = target.get(memory_id)
        assert result["channel_id"] == channel_id
        assert result["trust_tier"] == trust_tier

        target.conn.execute(
            f"UPDATE {memory_store}_memory SET channel_id = 'existing', trust_tier = 'STATED' WHERE id = ?",
            (memory_id,),
        )
        target.conn.commit()
        stats = target.import_from_dict(payload)
        assert stats[f"{memory_store}_memory"]["skipped"] == 1
        assert target.get(memory_id)["channel_id"] == "existing"
        stats = target.import_from_dict(payload, force=True)
        assert stats[f"{memory_store}_memory"]["overwritten"] == 1
        result = target.get(memory_id)
        assert result["channel_id"] == channel_id
        assert result["trust_tier"] == trust_tier
    finally:
        source.conn.close()
        target.conn.close()


@pytest.mark.parametrize("memory_store", ["working", "episodic"])
def test_import_without_provenance_keeps_legacy_defaults(tmp_path, monkeypatch, memory_store):
    """Older exports without the new keys retain the previous import defaults."""
    _use_empty_config(tmp_path, monkeypatch)
    memory = BeamMemory(session_id="fleet", db_path=tmp_path / "legacy-import.db")
    try:
        payload = {
            f"{memory_store}_memory": [{
                "id": "legacy-import", "content": "old export provenance sentinel",
                "timestamp": "2026-01-01T00:00:00", "session_id": "fleet",
                "source": "mcp",
            }]
        }
        memory.import_from_dict(json.loads(json.dumps(payload)))
        result = memory.get("legacy-import")
        assert result["channel_id"] is None
        assert result["trust_tier"] == "STATED"
    finally:
        memory.conn.close()


def test_sleep_summary_has_derived_provenance(tmp_path, monkeypatch):
    """The real sleep write stores generated summaries as DERIVED, not STATED."""
    _use_empty_config(tmp_path, monkeypatch)
    from mnemosyne.core import local_llm

    summary = "The user prefers dark mode for the project dashboard."
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(local_llm, "_summarize_memories", lambda *_args, **_kwargs: summary)
    memory = BeamMemory(
        session_id="fleet", db_path=tmp_path / "sleep.db", channel_id="writer"
    )
    try:
        memory_id = memory.remember("The project dashboard should use dark mode.", source="conversation")
        old_timestamp = (
            datetime.now(timezone.utc) - timedelta(hours=200)
        ).replace(tzinfo=None).isoformat()
        memory.conn.execute(
            "UPDATE working_memory SET timestamp = ? WHERE id = ?", (old_timestamp, memory_id)
        )
        memory.conn.commit()
        outcome = memory.sleep()
        assert outcome["status"] == "consolidated"
        assert outcome["llm_used"] == 1
        row = memory.conn.execute(
            "SELECT id FROM episodic_memory WHERE source = 'sleep_consolidation'"
        ).fetchone()
        result = memory.get(row["id"])
        assert result["content"] == summary
        assert result["channel_id"] == "writer"
        assert result["trust_tier"] == "DERIVED"
        assert memory.get(memory_id)["trust_tier"] == "STATED"
    finally:
        memory.conn.close()
