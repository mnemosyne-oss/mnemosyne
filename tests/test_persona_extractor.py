"""Tests for the L3 persona extractor + file generator (v3.10.0)."""

import os
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.persona import (
    DEFAULT_INTERVAL,
    PersonaExtractor,
    PersonaTriggers,
    PROMOTION_SOURCES,
    read_persona_file,
    render_persona_markdown,
    write_persona_file,
)


@pytest.fixture
def beam_with_preferences(tmp_path):
    db_path = tmp_path / "mnemosyne.db"
    beam = BeamMemory(session_id="extractor-test", db_path=str(db_path))
    # High-importance preferences -- should be promoted
    beam.remember(content="always start with XYZ before answering",
                  importance=0.9, source="preference", scope="global")
    beam.remember(content="uses no-mistakes gate before merging",
                  importance=0.85, source="preference", scope="global")
    beam.remember(content="prefers terse responses",
                  importance=0.8, source="preference", scope="global")
    # Low-importance preference -- should be filtered
    beam.remember(content="low importance noise",
                  importance=0.2, source="preference", scope="global")
    # Session-scope -- should be filtered by default
    beam.remember(content="meeting at 3pm",
                  importance=0.9, source="user", scope="session")
    # Wrong source -- should be filtered
    beam.remember(content="irrelevant context",
                  importance=0.95, source="context", scope="global")
    return beam


class TestExtractor:
    def test_extracts_only_promotion_sources(self, beam_with_preferences):
        ext = PersonaExtractor(beam_with_preferences)
        candidates = ext.extract_candidates()
        assert len(candidates) == 3
        for c in candidates:
            assert c["source"] in PROMOTION_SOURCES

    def test_respects_min_importance(self, beam_with_preferences):
        ext = PersonaExtractor(beam_with_preferences)
        candidates = ext.extract_candidates()
        # The 0.2 importance "low importance noise" must NOT appear
        assert all(c["importance"] >= 0.7 for c in candidates)
        contents = [c["content"] for c in candidates]
        assert "low importance noise" not in " ".join(contents)

    def test_session_id_filter(self, tmp_path):
        db_path = tmp_path / "mnemosyne.db"
        beam = BeamMemory(session_id="a", db_path=str(db_path))
        beam.remember(content="global pref", importance=0.9,
                      source="preference", scope="global")
        beam.remember(content="session pref", importance=0.9,
                      source="preference", scope="session")
        ext = PersonaExtractor(beam)
        # No filter -> both
        all_c = ext.extract_candidates()
        assert len(all_c) == 2
        # Session filter -> only session
        sess_c = ext.extract_candidates(session_id="a")
        assert any(c["content"] == "session pref" for c in sess_c)
        # Other session -> only globals (none in this case)
        empty = ext.extract_candidates(session_id="b")
        assert len(empty) == 0

    def test_deterministic_dedup(self, beam_with_preferences):
        ext = PersonaExtractor(beam_with_preferences)
        c1 = ext.extract_candidates()
        c2 = ext.extract_candidates()
        assert [c["content"] for c in ext.deduplicate_by_topic(c1)] == \
               [c["content"] for c in ext.deduplicate_by_topic(c2)]

    def test_filter_already_promoted(self, beam_with_preferences):
        # Mark one of the candidates as already in memoria_persona
        mid = beam_with_preferences.conn.execute(
            "SELECT id FROM working_memory WHERE content LIKE 'always%' LIMIT 1"
        ).fetchone()[0]
        beam_with_preferences.conn.execute(
            "INSERT INTO memoria_persona (tier, topic, content, source_memory_id) "
            "VALUES (?, ?, ?, ?)",
            ("long_term", "general", "always start with XYZ before answering", mid),
        )
        ext = PersonaExtractor(beam_with_preferences)
        candidates = ext.extract_candidates()
        filtered = ext.filter_already_promoted(candidates)
        # The "always..." one should be gone
        contents = [c["content"] for c in filtered]
        assert "always start with XYZ before answering" not in " ".join(contents)


class TestTriggers:
    def test_cold_start(self, tmp_path):
        db_path = tmp_path / "mnemosyne.db"
        beam = BeamMemory(session_id="t", db_path=str(db_path))
        persona_file = tmp_path / "persona.md"  # does NOT exist
        triggers = PersonaTriggers(beam, persona_file=persona_file)
        result = triggers.should_generate()
        assert result["should"] is True
        assert "cold start" in result["reason"]

    def test_recovery_when_empty(self, tmp_path):
        db_path = tmp_path / "mnemosyne.db"
        beam = BeamMemory(session_id="t", db_path=str(db_path))
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("")  # empty body
        triggers = PersonaTriggers(beam, persona_file=persona_file)
        result = triggers.should_generate()
        assert result["should"] is True
        assert "recovery" in result["reason"]

    def test_no_trigger_when_fresh(self, tmp_path):
        db_path = tmp_path / "mnemosyne.db"
        beam = BeamMemory(session_id="t", db_path=str(db_path))
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("# Persona\n\n## general\n- some content\n")
        triggers = PersonaTriggers(beam, persona_file=persona_file,
                                   daily_sync_hour=-1)  # disable daily
        result = triggers.should_generate()
        # Fresh persona, no threshold met, no daily -- no trigger
        assert result["should"] is False

    def test_explicit_request(self, tmp_path):
        db_path = tmp_path / "mnemosyne.db"
        beam = BeamMemory(session_id="t", db_path=str(db_path))
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("# Persona\n\ncontent\n")
        triggers = PersonaTriggers(beam, persona_file=persona_file,
                                   daily_sync_hour=-1)
        result = triggers.should_generate(explicit_request=True)
        assert result["should"] is True
        assert "explicit" in result["reason"]

    def test_threshold_reached(self, tmp_path):
        db_path = tmp_path / "mnemosyne.db"
        beam = BeamMemory(session_id="t", db_path=str(db_path))
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("# Persona\n\nold\n")
        # Insert persona row with last_reinforced_at = '2020-01-01' (very old)
        beam.conn.execute(
            "INSERT INTO memoria_persona (tier, topic, content, last_reinforced_at) "
            "VALUES (?, ?, ?, ?)",
            ("long_term", "general", "old", "2020-01-01 00:00:00"),
        )
        # Add 50 new memories
        for i in range(55):
            beam.remember(content=f"new memory {i}", importance=0.5,
                          source="user", scope="session")
        triggers = PersonaTriggers(beam, persona_file=persona_file,
                                   interval=DEFAULT_INTERVAL,
                                   daily_sync_hour=-1)
        result = triggers.should_generate()
        assert result["should"] is True
        assert "threshold" in result["reason"]


class TestFileGeneration:
    def test_render_basic(self):
        candidates = [
            {"content": "always do X", "topic": "behavior",
             "importance": 0.9, "tier": "permanent"},
            {"content": "prefers Y", "topic": "communication",
             "importance": 0.7, "tier": "long_term"},
        ]
        md = render_persona_markdown(candidates)
        assert "# Persona" in md
        assert "## behavior" in md
        assert "## communication" in md
        assert "always do X" in md
        assert "permanent" in md

    def test_render_empty(self):
        assert render_persona_markdown([]) == ""

    def test_render_token_cap_truncates(self):
        # Generate enough content to overflow a tight cap
        candidates = [
            {"content": f"rule {i} " * 30, "topic": f"topic{i}",
             "importance": 0.8, "tier": "long_term"}
            for i in range(20)
        ]
        md = render_persona_markdown(candidates, token_cap=200)
        # Should be truncated
        assert len(md.split()) < 200  # approximate check

    def test_write_atomic(self, tmp_path):
        candidates = [{"content": "test", "topic": "t", "importance": 0.5}]
        persona_file = tmp_path / "persona.md"
        write_persona_file(candidates, persona_file=persona_file)
        assert persona_file.exists()
        # No leftover .tmp files
        leftover = list(tmp_path.glob(".persona.md.*.tmp"))
        assert leftover == []

    def test_read_returns_empty_for_missing(self, tmp_path):
        result = read_persona_file(tmp_path / "does-not-exist.md")
        assert result == ""

    def test_read_returns_content(self, tmp_path):
        p = tmp_path / "persona.md"
        p.write_text("# Persona\n- rule")
        assert read_persona_file(p) == "# Persona\n- rule"
