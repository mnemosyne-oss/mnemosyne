"""Real conflict counters survive public MCP and both Hermes sleep adapters."""
import importlib
import json
from types import SimpleNamespace

import pytest

from mnemosyne import mcp_tools
from mnemosyne.core import llm_conflict_detector as lcd
from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def beam(tmp_path, monkeypatch):
    """Seed two sessions without embedding or network dependencies."""
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "CONFLICT_LLM_BASE_URL", "https://validator.test/v1")
    monkeypatch.setattr(lcd, "CONFLICT_LLM_API_KEY", "fake-key")
    monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
    monkeypatch.setattr("mnemosyne.core.model_refresh.infer_model_update_proposals", lambda items: [])
    memory = BeamMemory(session_id="first-session", db_path=tmp_path / "memory.db")
    for session in ("first-session", "second-session"):
        for i in range(3):
            memory.conn.execute(
                "INSERT INTO working_memory(id,content,source,timestamp,session_id) VALUES (?,?,?,?,?)",
                (f"{session}-{i}", f"Project meeting changed to day {i}", "conversation",
                 f"2026-01-01T{10+i}:00:00", session),
            )
    memory.conn.commit()
    yield memory
    memory.conn.close()



@pytest.mark.parametrize("surface", ["mcp", "hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize("all_sessions", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_public_sleep_surfaces_preserve_both_conflict_counters(
    beam, monkeypatch, surface, all_sessions, dry_run,
):
    """Exercise dispatch, real SQLite consolidation, and the serialized response."""
    monkeypatch.setattr(
        BeamMemory, "_detect_conflicts",
        lambda self, rows: [(rows[0]["id"], rows[1]["id"]), (rows[1]["id"], rows[2]["id"])],
    )
    monkeypatch.setattr(
        lcd, "validate_conflict_pair",
        lambda older, newer, **kwargs: (older.endswith("day 0"), 0.97, "corrected"),
    )
    arguments = {"force": True, "dry_run": dry_run, "all_sessions": all_sessions}
    before = sorted(beam.conn.iterdump())
    if surface == "mcp":
        wrapper = SimpleNamespace(beam=beam, sleep=beam.sleep, sleep_all_sessions=beam.sleep_all_sessions)
        monkeypatch.setattr(mcp_tools, "_create_instance", lambda **kwargs: wrapper)
        response = mcp_tools.handle_tool_call("mnemosyne_sleep", arguments)
    else:
        module = importlib.import_module(surface)
        provider = module.MnemosyneMemoryProvider()
        provider._beam = beam
        provider._session_id = beam.session_id
        provider._reflect_max_calls_per_session = None
        response = json.loads(provider.handle_tool_call("mnemosyne_sleep", arguments))
    assert "error" not in response
    result = response["result"]
    sessions = 2 if all_sessions else 1
    assert result["conflicts_resolved"] == (0 if dry_run else sessions)
    assert result["conflicts_detected_only"] == (2 * sessions if dry_run else sessions)
    if all_sessions:
        assert result["sessions_scanned"] == sessions
        assert len(result["session_results"]) == sessions
        for key in ("conflicts_resolved", "conflicts_detected_only"):
            assert result[key] == sum(row[key] for row in result["session_results"])
    if dry_run:
        assert sorted(beam.conn.iterdump()) == before
    else:
        assert beam.conn.execute("SELECT COUNT(*) FROM memory_validations").fetchone()[0] == sessions
        assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == sessions
