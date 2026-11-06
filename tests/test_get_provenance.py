"""Identity and scope provenance for direct BEAM reads."""

import pytest

from mnemosyne.core.beam import BeamMemory


def _use_empty_config(tmp_path, monkeypatch):
    data_dir = tmp_path / "config-data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))


def test_get_returns_working_memory_author_and_scope(tmp_path, monkeypatch):
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
    # including nullable provenance in older data.
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


@pytest.mark.parametrize("memory_store", ["working", "episodic"])
def test_get_provenance_keeps_session_visibility(tmp_path, monkeypatch, memory_store):
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
