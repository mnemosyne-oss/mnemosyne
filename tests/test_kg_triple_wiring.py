"""KG triple wiring: LLM-extracted SPO triples reach the temporal TripleStore.

Historically the extraction prompt asked the model for a ``kg`` category of
subject-predicate-object triples, but the parser iterated only
('facts', 'instructions', 'preferences', 'timelines') and silently dropped
every triple. Meanwhile an always-on regex prototype (BEAM benchmark-oracle
writer) filled ``memoria_kg`` with hardcoded
``(subject='user', predicate in {negation, decision})`` rows whose objects
ended at raw Python slices — mid-word junk.

These tests pin the repaired behavior:

  1. ``_parse_kg_triples`` recovers the ``kg`` category (string, 3-list, and
     dict item shapes) from the same JSON payload, and never guesses an SPO
     split out of prose.
  2. ``validate_kg_triples`` rejects empty fields and conversational filler,
     truncates over-long objects at word boundaries (never mid-word), folds
     predicates to lowercase snake_case, and dedupes within a batch.
  3. ``remember(..., extract=True)`` writes validated triples into the
     bank's ``triples`` table with ``valid_from = now`` and
     ``source='llm_extraction'`` provenance, while ``memoria_kg`` stays
     EMPTY because the regex writer is opt-in via ``MNEMOSYNE_REGEX_KG``.
  4. TripleStore single-current-truth semantics still hold for repeated
     ``(subject, predicate)`` writes (supersede).
  5. Empty-KG recall fallback: with the regex writer contained (opt-in),
     a fresh install has an empty ``memoria_kg``; the structured retrieval
     consumers answer with the documented fallback shape instead of raising.
  6. End-to-end: mocked LLM JSON -> remember(extract=True) -> readable back
     through the public query surfaces (TripleStore.query and the
     mnemosyne_triple_query MCP tool handler).

Run with: pytest tests/test_kg_triple_wiring.py -v
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from unittest.mock import MagicMock

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.extraction import (
    KG_MAX_OBJECT_CHARS,
    _parse_kg_triples,
    validate_kg_triples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kg_payload(kg_items):
    """Build an extraction-style JSON payload with only the kg array filled."""
    return json.dumps({
        "facts": [], "instructions": [], "preferences": [],
        "timelines": [], "kg": kg_items,
    })


@pytest.fixture()
def beam(tmp_path):
    b = BeamMemory(session_id="kg-wiring-test", db_path=tmp_path / "mnemosyne.db")
    yield b
    b.conn.close()


@pytest.fixture()
def fake_host_llm(monkeypatch):
    """Enable the host tier with a deterministic canned-JSON backend.

    Patches mnemosyne.core.local_llm module globals (they are read at call
    time) instead of env vars, which are consumed at import time.
    """
    import mnemosyne.core.local_llm as local_llm
    from mnemosyne.core.llm_backends import set_host_llm_backend

    backend = MagicMock()
    backend.name = "fake-host"
    backend.complete.return_value = _kg_payload([
        "user prefers postgresql",
        ["bank-a", "validates", "triple store write path"],
        "user decision whether to abandon ship",        # filler -> rejected
        {"subject": "ops", "predicate": "Monitors",
         "object": "watch latency dashboards " * 40},   # >300 chars -> truncated
    ])
    set_host_llm_backend(backend)
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    yield backend
    set_host_llm_backend(None)


def _db_rows(db_path, sql):
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1+2. Parser / validator pure functions
# ---------------------------------------------------------------------------

class TestParseKgTriples:

    def test_all_item_shapes(self):
        triples = validate_kg_triples(_parse_kg_triples(_kg_payload([
            "user prefers vim",
            ["agent-7", "deployed_on", "fly.io"],
            {"subject": "mnemosyne", "predicate": "Stores",
             "object": "memory banks"},
            "only-two-tokens",   # no SPO split possible -> dropped
            "",                  # empty -> dropped
        ])))
        assert ("user", "prefers", "vim") in triples
        assert ("agent-7", "deployed_on", "fly.io") in triples
        # predicate normalized to lowercase; multi-word object preserved
        assert ("mnemosyne", "stores", "memory banks") in triples
        assert len(triples) == 3

    def test_non_json_prose_yields_nothing(self):
        # We must never fabricate an SPO split out of free prose.
        assert _parse_kg_triples("Here are the facts: user likes X") == []

    def test_no_facts_sentinel(self):
        assert _parse_kg_triples("NO_FACTS") == []

    def test_markdown_fence(self):
        fenced = "```json\n" + _kg_payload(["a b c"]) + "\n```"
        assert validate_kg_triples(_parse_kg_triples(fenced)) == [("a", "b", "c")]

    def test_null_fields_rejected_before_str_conversion(self):
        # JSON null must never become the literal string "None".
        assert _parse_kg_triples(_kg_payload([[None, "uses", "sqlite"]])) == []
        assert _parse_kg_triples(_kg_payload([["user", None, "sqlite"]])) == []
        assert _parse_kg_triples(_kg_payload([["user", "uses", None]])) == []


class TestValidateKgTriples:

    def test_rejects_empty_fields_and_filler(self):
        assert validate_kg_triples([
            ("user", "", "something"),
            ("", "uses", "x"),
            ("user", "decision", "whether to abandon ship"),
            ("what to do next", "decision", "ship"),
        ]) == []

    def test_predicate_normalized_to_snake_case(self):
        # Hyphens/space runs/leading/trailing separators all collapse to
        # a single snake_case predicate so "works-at" and "works at"
        # cannot persist as separate current truths (supersede keys on
        # subject+predicate).
        out = validate_kg_triples([
            ("user", "works-at", "acme"),
            ("user", "works at", "acme"),
            ("user", "--works--at--", "acme"),
        ])
        assert [t[1] for t in out] == ["works_at"]

    def test_separator_only_predicate_rejected(self):
        # Normalization of "-" / "___" yields "" — must not persist a
        # triple with an empty predicate (CodeRabbit e1c067d follow-up).
        assert validate_kg_triples([
            ("user", "-", "x"),
            ("user", "___", "x"),
            ("user", "  - -  ", "x"),
        ]) == []

    def test_object_truncated_at_word_boundary(self):
        long_obj = "deploy the service across every region " * 12
        out = validate_kg_triples([("platform", "requires", long_obj)])
        assert len(out) == 1
        subject, predicate, obj = out[0]
        assert len(obj) <= KG_MAX_OBJECT_CHARS == 300
        assert long_obj.startswith(obj)
        assert long_obj[len(obj)] == " "          # cut exactly at a space
        assert obj == obj.rstrip()                # no dangling punctuation/space

    def test_uncuttable_single_token_rejected(self):
        assert validate_kg_triples([
            ("platform", "requires",
             "x" * 400)          # one unbreakable token -> rejected outright
        ]) == []


# ---------------------------------------------------------------------------
# 3+5. Regex prototype gate + empty-KG recall fallback (#840 containment)
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


class TestEmptyKgRecallFallback:
    """Recall consumers must degrade sanely when memoria_kg is empty.

    After containment (regex writer opt-in) an out-of-the-box install has
    an EMPTY memoria_kg; the structured retrieval router must answer with
    the documented fallback dict instead of raising or fabricating rows.
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


# ---------------------------------------------------------------------------
# 4+6. LLM wiring end-to-end (mocked LLM) through store AND query surfaces
# ---------------------------------------------------------------------------

class TestLlmTripleWiring:

    def test_extract_writes_validated_triples(self, beam, fake_host_llm):
        beam.remember("Pinning our stack: we standardize on postgresql "
                      "and this bank exercises the triple write path.",
                      source="test", extract=True)

        rows = _db_rows(
            beam.db_path,
            "SELECT subject, predicate, object, valid_from, valid_until, source "
            "FROM triples ORDER BY id")

        clean = [r for r in rows if r[5] == "llm_extraction"]
        assert len(clean) >= 2                       # junk row was rejected
        assert all(r[4] is None for r in clean)      # all currently open
        today = date.today().isoformat()
        assert all(r[3] == today for r in clean)     # valid_from = now

        # Filler never reaches the store; long objects are word-truncated.
        assert not any("whether to abandon ship" in r[2] for r in rows)
        long_objs = [r for r in rows if r[2].startswith("watch latency dashboards")]
        assert long_objs and len(long_objs[0][2]) <= KG_MAX_OBJECT_CHARS

        # The regex table stays empty even when the LLM path runs.
        kg = _db_rows(beam.db_path, "SELECT COUNT(*) FROM memoria_kg")
        assert kg[0][0] == 0

    def test_end_to_end_json_to_query_surfaces(self, beam, tmp_path, monkeypatch):
        """LLM JSON -> remember(extract=True) -> public read surfaces.

        dplush's second-PR ask: prove the whole contract end to end --
        mocked LLM returns extraction JSON carrying kg triples, remember()
        stores them via the validated write path, and the triples are then
        readable back through BOTH public consumers of the contract:
        TripleStore.query() and the mnemosyne_triple_query tool handler
        (the same routing mnemosyne_triple_add/query expose over MCP).

        The tool handler resolves its bank under MNEMOSYNE_DATA_DIR, so the
        test pins that env var to tmp_path and drives the real handler --
        no production data dir is ever touched.
        """
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        import mnemosyne.core.local_llm as local_llm
        from mnemosyne.core.llm_backends import set_host_llm_backend
        from mnemosyne.core.triples import TripleStore

        payload = _kg_payload([
            "user prefers sqlite",
            ["bank-b", "mirrors", "postgres primary"],
        ])

        class _B:
            name = "fake-e2e"

            def complete(self, prompt, *, max_tokens, temperature, timeout,
                         provider=None, model=None):
                return payload

        set_host_llm_backend(_B())
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        try:
            res = beam.remember("Stack note: we standardize on sqlite here.",
                                source="t", extract=True)
        finally:
            set_host_llm_backend(None)

        assert res is not None  # remember completed; wiring is best-effort

        # Surface 1: TripleStore.query reads back what the LLM JSON implied.
        kg = TripleStore(db_path=beam.db_path)
        got = kg.query(subject="user", predicate="prefers")
        assert [r["object"] for r in got] == ["sqlite"]

        mirror = kg.query(subject="bank-b", predicate="mirrors")
        assert [r["object"] for r in mirror] == ["postgres primary"]

        # Surface 2: the mnemosyne_triple_query tool handler (real code path,
        # not a mock) answers against the same bank DB.
        from mnemosyne.mcp_tools import handle_tool_call

        result = handle_tool_call("mnemosyne_triple_query", {
            "subject": "user",
            "predicate": "prefers",
            "bank": "default",
        })
        assert result["store"] == "triples"
        assert result["results_count"] == 1
        assert result["results"][0]["object"] == "sqlite"

        # And the regex junk table stayed empty through the whole flow.
        kg_rows = _db_rows(beam.db_path, "SELECT COUNT(*) FROM memoria_kg")
        assert kg_rows[0][0] == 0

    def test_supersede_single_current_truth(self, beam, monkeypatch):
        import json as _json

        def respond(prompt, *, max_tokens, temperature):
            return _json.dumps({
                "facts": [], "instructions": [], "preferences": [],
                "timelines": [],
                "kg": [f"user prefers {PREF['current']}"],
            })

        PREF = {"current": "postgresql"}
        import mnemosyne.core.local_llm as local_llm
        from mnemosyne.core.llm_backends import set_host_llm_backend

        class _B:
            name = "fake"

            def complete(self, prompt, *, max_tokens, temperature, timeout,
                         provider=None, model=None):
                return respond(prompt, max_tokens=max_tokens,
                               temperature=temperature)

        set_host_llm_backend(_B())
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        try:
            beam.remember("We standardize on postgresql.", source="t", extract=True)
            PREF["current"] = "mysql"
            beam.remember("After evaluation we standardize on mysql instead.",
                          source="t", extract=True)
        finally:
            set_host_llm_backend(None)

        closed = _db_rows(
            beam.db_path,
            "SELECT COUNT(*) FROM triples WHERE subject='user' "
            "AND predicate='prefers' AND object='postgresql' "
            "AND valid_until IS NOT NULL")
        open_mysql = _db_rows(
            beam.db_path,
            "SELECT COUNT(*) FROM triples WHERE object='mysql' "
            "AND valid_until IS NULL")
        assert closed[0][0] == 1
        assert open_mysql[0][0] == 1
