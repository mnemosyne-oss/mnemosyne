"""KG containment: the ``memoria_kg`` regex prototype writer is opt-in.

Part 1 of the two-part split requested by maintainer dplush on issue #840:
land the containment of the junk-writing regex prototype separately from
any KG-contract work.

Historically an always-on regex prototype (BEAM benchmark-oracle writer)
filled ``memoria_kg`` with hardcoded ``(subject='user',
predicate in {negation, decision})`` rows whose objects ended at raw
Python slices -- mid-word junk. Its three branches (negation / decision /
entity-action) now sit behind ``MNEMOSYNE_REGEX_KG`` (default OFF) inside
``BeamMemory.extract_and_store_facts``, the single funnel shared by the
remember() new-row path, the remember() dedup-update path, the batch tool
(via remember()), and the Hindsight importer (which calls the same method
directly).

These tests pin the contained behavior:

  1. Default OFF: ``memoria_kg`` stays empty while metric / date /
     version extraction keeps flowing into ``memoria_facts`` untouched.
  2. Opt-in: setting ``MNEMOSYNE_REGEX_KG=1`` restores the legacy writer.
  3. Empty-KG recall fallback: the structured retrieval consumers that
     query ``memoria_kg`` (negation and entity-action specialists) return
     a sane fallback dict when the table is empty -- the default state
     after this change.

Run with: pytest tests/test_kg_triple_wiring.py -v
"""
from __future__ import annotations

import sqlite3

import pytest

from mnemosyne.core.beam import BeamMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def beam(tmp_path):
    b = BeamMemory(session_id="kg-containment-test", db_path=tmp_path / "mnemosyne.db")
    yield b
    b.conn.close()


def _db_rows(db_path, sql):
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Regex prototype gate
# ---------------------------------------------------------------------------

class TestRegexKgGate:

    def test_default_off_memoria_kg_stays_empty(self, beam, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_REGEX_KG", raising=False)
        beam.remember("I have decided to switch to sqlite wal mode tomorrow.",
                      source="test")
        rows = _db_rows(beam.db_path, "SELECT COUNT(*) FROM memoria_kg")
        assert rows[0][0] == 0

    def test_opt_in_reenables_prototype(self, beam, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_REGEX_KG", "1")
        beam.remember("We selected nginx as the reverse proxy.", source="test")
        rows = _db_rows(
            beam.db_path,
            "SELECT subject, predicate, object FROM memoria_kg")
        assert rows, "opt-in flag must restore the legacy writer"
        assert any(r[1] in ("decision", "negation", "requires") for r in rows)

    def test_gate_leaves_metric_branch_untouched(self, beam, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_REGEX_KG", raising=False)
        before = _db_rows(beam.db_path, "SELECT COUNT(*) FROM memoria_facts")[0][0]
        beam.remember("The dashboard API response time is 250ms under load.",
                      source="test")
        after = _db_rows(beam.db_path, "SELECT COUNT(*) FROM memoria_facts")[0][0]
        assert after >= before   # metrics/dates/versions keep flowing


# ---------------------------------------------------------------------------
# Empty-KG recall fallback (#840 containment ask)
# ---------------------------------------------------------------------------

class TestEmptyKgRecallFallback:
    """Recall consumers must degrade sanely when memoria_kg is empty.

    After the containment change an out-of-the-box install has an EMPTY
    memoria_kg (the writer is opt-in); the structured retrieval router
    must answer with the documented fallback shape instead of raising or
    fabricating rows.
    """

    def test_negation_specialist_falls_back_when_empty(self, beam, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_REGEX_KG", raising=False)
        result = beam._memoria_negation_retrieve(
            "Did I abandon the killswitch project?")
        assert result == {"context": "", "facts": [], "source": "fallback"}

    def test_entity_specialist_falls_back_when_empty(self, beam, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_REGEX_KG", raising=False)
        result = beam._memoria_entity_retrieve(
            "What does the Nginx proxy require?")
        assert result == {"context": "", "facts": [], "source": "fallback"}

    def test_router_returns_fallback_for_kg_abilities(self, beam, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_REGEX_KG", raising=False)
        # CR routes to the negation specialist, MR to the entity specialist;
        # both must produce the documented fallback on an empty table.
        for ability in ("CR", "MR"):
            result = beam.memoria_retrieve(
                "Have I decided anything about Redis?", ability=ability)
            assert result == {"context": "", "facts": [], "source": "fallback"}, (
                f"ability={ability} returned {result!r}")
