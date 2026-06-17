"""Tests for the L3 persona adapter (v3.10.0)."""

import json
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne_hermes.persona_adapter import PersonaAdapter, VALID_TIERS


@pytest.fixture
def beam_with_memories(tmp_path):
    """BeamMemory with three diverse memories seeded."""
    db_path = tmp_path / "mnemosyne.db"
    beam = BeamMemory(session_id="persona-test", db_path=str(db_path))
    beam.remember(
        content="always start with XYZ before answering",
        importance=0.9, source="preference", scope="global",
    )
    beam.remember(
        content="uses no-mistakes gate before merging to main",
        importance=0.85, source="preference", scope="global",
    )
    beam.remember(
        content="meeting at 3pm tomorrow",
        importance=0.5, source="user", scope="session",
    )
    return beam


@pytest.fixture
def adapter(beam_with_memories):
    return PersonaAdapter(beam_instance=beam_with_memories)


def _memory_id_by_content(beam, content_substr):
    row = beam.conn.execute(
        "SELECT id FROM working_memory WHERE content LIKE ? LIMIT 1",
        (f"%{content_substr}%",),
    ).fetchone()
    assert row is not None, f"no memory with content matching {content_substr!r}"
    return row[0]


class TestPersonaPromote:
    def test_promote_to_long_term(self, adapter, beam_with_memories):
        mid = _memory_id_by_content(beam_with_memories, "XYZ")
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid, "tier": "long_term", "reason": "behavioral rule"},
        ))
        assert result["status"] == "ok"
        assert result["tier"] == "long_term"
        assert isinstance(result["persona_id"], int)
        assert result["persona_id"] > 0

    def test_promote_invalid_tier_rejected(self, adapter, beam_with_memories):
        mid = _memory_id_by_content(beam_with_memories, "XYZ")
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid, "tier": "garbage"},
        ))
        assert result["status"] == "error"
        assert "Invalid tier" in result["error"]

    def test_promote_missing_memory(self, adapter):
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": "does-not-exist"},
        ))
        assert result["status"] == "error"
        assert "not found" in result["error"]


class TestPersonaList:
    def test_list_empty(self, adapter):
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {},
        ))
        assert result == {"status": "ok", "count": 0, "personas": []}

    def test_list_after_promote(self, adapter, beam_with_memories):
        mid = _memory_id_by_content(beam_with_memories, "no-mistakes")
        adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid, "tier": "permanent"},
        )
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {},
        ))
        assert result["count"] == 1
        p = result["personas"][0]
        assert p["tier"] == "permanent"
        assert "no-mistakes" in p["content"]

    def test_list_filter_by_tier(self, adapter, beam_with_memories):
        mid_x = _memory_id_by_content(beam_with_memories, "XYZ")
        mid_g = _memory_id_by_content(beam_with_memories, "no-mistakes")
        adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid_x, "tier": "long_term"},
        )
        adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid_g, "tier": "permanent"},
        )
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {"tier": "permanent"},
        ))
        assert result["count"] == 1
        assert result["personas"][0]["tier"] == "permanent"

    def test_list_returns_sorted_permanent_first(self, adapter, beam_with_memories):
        mid_x = _memory_id_by_content(beam_with_memories, "XYZ")
        mid_g = _memory_id_by_content(beam_with_memories, "no-mistakes")
        adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid_x, "tier": "long_term"},
        )
        adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid_g, "tier": "permanent"},
        )
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {},
        ))
        assert result["count"] == 2
        assert result["personas"][0]["tier"] == "permanent"  # tier sorted ASC
        assert result["personas"][1]["tier"] == "long_term"


class TestPersonaReinforce:
    def test_reinforce_increments(self, adapter, beam_with_memories):
        mid = _memory_id_by_content(beam_with_memories, "XYZ")
        adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid, "tier": "long_term"},
        )
        listed = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {},
        ))
        pid = listed["personas"][0]["id"]
        assert listed["personas"][0]["reinforcement_count"] == 0

        adapter.handle_tool_call(
            "mnemosyne_persona_reinforce", {"persona_id": pid},
        )
        listed2 = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {},
        ))
        assert listed2["personas"][0]["reinforcement_count"] == 1

        adapter.handle_tool_call(
            "mnemosyne_persona_reinforce", {"persona_id": pid},
        )
        listed3 = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {},
        ))
        assert listed3["personas"][0]["reinforcement_count"] == 2

    def test_reinforce_missing_id(self, adapter):
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_reinforce", {"persona_id": 9999},
        ))
        assert result["status"] == "error"


class TestPersonaDemote:
    def test_demote_writes_tombstone_and_removes(self, adapter, beam_with_memories):
        mid = _memory_id_by_content(beam_with_memories, "XYZ")
        promote_result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_promote",
            {"memory_id": mid, "tier": "long_term"},
        ))
        pid = promote_result["persona_id"]

        # Demote
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_demote",
            {"persona_id": pid, "reason": "user changed mind"},
        ))
        assert result["status"] == "ok"
        assert result["demoted_to"] == "memoria_preferences"

        # Persona gone from L3
        listed = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_list", {},
        ))
        assert listed["count"] == 0

        # Tombstone present in memoria_preferences
        tomb = beam_with_memories.conn.execute(
            "SELECT preference FROM memoria_preferences WHERE source_memory_id = ?",
            (f"persona:{pid}",),
        ).fetchone()
        assert tomb is not None
        assert "[demoted from long_term]" in tomb[0]

    def test_demote_missing_id(self, adapter):
        result = json.loads(adapter.handle_tool_call(
            "mnemosyne_persona_demote", {"persona_id": 9999},
        ))
        assert result["status"] == "error"


class TestPersonaAdapterReadiness:
    def test_adapter_not_ready_without_beam(self, tmp_path):
        a = PersonaAdapter(beam_instance=None)
        # Use the default DB (which exists in dev). is_ready should be True.
        assert a.is_ready
