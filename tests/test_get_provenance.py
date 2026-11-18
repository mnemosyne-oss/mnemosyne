"""Identity and scope provenance for direct BEAM reads."""

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
