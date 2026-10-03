"""
Regression tests for fact-extraction object quality (mnemosyne-oss/mnemosyne#837).

`EpisodicGraph.extract_facts` runs four regexes ("X is Y", "X has Y",
"X uses Y", "X works at Y") on every remembered row. Two defects let junk
triples reach `facts`, `graph_edges` and `consolidated_facts`, from where
`fact_recall` surfaced them:

1. The optional article was not anchored as a whole word, so
   "Alice is already ready" captured article "a" + object "lready". ("Alice
   is there" captured "the" + "re" the same way, but never persisted: the
   existing `len(obj) > 2` check already discarded a two-character object.)
2. Nothing guarded the object side, so "Bob is different" persisted
   (Bob, is, different), and every such state/stance/filler token shared
   (subject, predicate) with the real facts about Bob, which the veracity
   consolidator then read as contradictions.

The fix is a whole-word article group in the three affected patterns plus
`_is_low_quality_object`, a closed word list (no suffix or shape heuristic,
so a name can never be rejected). The patterns capture a single object
token and this fix does not change that, so the guard is a rule about one
word: an adjective phrase reaches it as its leading modifier ("an extremely
reliable editor" -> "extremely"), which is why the modifiers are listed.
`test_the_object_is_a_single_token` pins the capture width as it stands.
These tests pin the unit behaviour and
then walk each ingest path that reaches `extract_facts`: `remember`, the
dedup-update branch of `remember`, `remember_batch`,
`consolidate_to_episodic`, and the `fact_recall` mix that reads the result.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.episodic_graph import (
    EpisodicGraph,
    _is_low_quality_object,
    _is_low_quality_subject,
)

JUNK = ("Bob is different. Bob is already ready. Bob is there. Bob has it. "
        "The frame has legs. An engineer uses Python.")
GOOD = ("Alice is a developer. Alice uses Rust. Alice works at Anthropic. "
        "The Matrix is a film. The Matrix works at Warner.")


@pytest.fixture
def graph() -> EpisodicGraph:
    return EpisodicGraph(conn=sqlite3.connect(":memory:"))


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "object_guard.db"


def _triples(graph: EpisodicGraph, content: str):
    return [(f.subject, f.predicate, f.object)
            for f in graph.extract_facts(content, "m1")]


def _fact_rows(conn: sqlite3.Connection):
    return {tuple(r) for r in conn.execute(
        "SELECT subject, predicate, object FROM facts").fetchall()}


def _consolidated_rows(conn: sqlite3.Connection):
    return {tuple(r) for r in conn.execute(
        "SELECT subject, predicate, object FROM consolidated_facts").fetchall()}


def _assert_clean(rows, label: str):
    """No row's object is a truncated article victim or a state/filler word,
    and no row's subject is an article-led common-noun phrase."""
    objects = {obj for _, _, obj in rows}
    for junk in ("lready", "re", "different", "it", "nother", "ory"):
        assert junk not in objects, f"{label}: junk object {junk!r} in {rows}"
    for subject, _, _ in rows:
        tokens = subject.split()
        assert not (len(tokens) > 1
                    and tokens[0].lower() in ("the", "a", "an")
                    and tokens[1][:1].islower()), (
            f"{label}: article-led common-noun subject {subject!r} in {rows}")


# ---------------------------------------------------------------------------
# _is_low_quality_object: closed list, never a shape rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("obj", [
    "", "   ", "different", "already", "there", "it", "nothing",
    "definitely", "apparently", "same", "done", "been",
])
def test_object_guard_rejects_value_free_tokens(obj):
    assert _is_low_quality_object(obj)


@pytest.mark.parametrize("obj", [
    "developer",            # lone common noun
    "Rust", "ComfyUI",      # lone proper noun / product
    "Sally", "Italy",       # names ending in -ly must never be rejected
    "family", "assembly",   # common nouns ending in -ly must never be rejected
    "fast",                 # an adjective the list does not name stays a fact
])
def test_object_guard_preserves_real_objects(obj):
    assert not _is_low_quality_object(obj)


def test_object_guard_has_no_suffix_heuristic():
    """The historical `-ly` rule had false positives; the guard is a list."""
    assert not _is_low_quality_object("Kelly")
    assert not _is_low_quality_object("jelly")
    assert _is_low_quality_object("definitely")


def test_subject_guard_rejects_article_led_common_nouns():
    assert _is_low_quality_subject("The silence")
    assert _is_low_quality_subject("A frame")
    assert _is_low_quality_subject("An engineer")
    assert not _is_low_quality_subject("Alice")
    assert not _is_low_quality_subject("Alice Smith")


def test_subject_guard_keeps_article_led_names():
    """An article can open a name, so the word after it decides.
    Raised in review of #876."""
    assert not _is_low_quality_subject("The Matrix")
    assert not _is_low_quality_subject("The Netherlands")
    assert not _is_low_quality_subject("A New Hope")
    assert _is_low_quality_subject("The")
    assert _is_low_quality_subject("A")


# ---------------------------------------------------------------------------
# extract_facts: whole-word articles and the object guard, all four patterns
# ---------------------------------------------------------------------------


def test_article_is_matched_as_a_whole_word(graph):
    assert _triples(graph, "Alice is already ready") == []
    assert _triples(graph, "Alice is there") == []
    assert _triples(graph, "Carol is another person") == []
    # "the" must not be consumed from inside the object word, on every
    # pattern that takes an optional article
    assert _triples(graph, "Alice is theory") == [("Alice", "is", "theory")]
    assert _triples(graph, "Bob has theory") == [("Bob", "has", "theory")]
    assert _triples(graph, "Bob uses theory") == [("Bob", "uses", "theory")]


def test_article_is_still_consumed_when_present(graph):
    assert _triples(graph, "Alice is a developer") == [("Alice", "is", "developer")]
    assert _triples(graph, "Alice is an engineer") == [("Alice", "is", "engineer")]
    assert _triples(graph, "Bob is the lead") == [("Bob", "is", "lead")]
    assert _triples(graph, "Bob has a dog") == [("Bob", "has", "dog")]
    assert _triples(graph, "Bob uses the terminal") == [("Bob", "uses", "terminal")]


def test_a_leading_modifier_is_not_a_fact(graph):
    """An adjective phrase arrives as its modifier alone, which names no
    value. Reported in review of #876 against the public entry point."""
    assert _triples(graph, "Alice is really talented.") == []
    assert _triples(graph, "Bob has a very useful tool.") == []
    assert _triples(graph, "Carol uses an extremely reliable editor.") == []
    assert _triples(graph, "Dave is originally from Texas") == []


def test_the_object_is_a_single_token(graph):
    """Pinned as it is, not endorsed: the patterns stop at the first word
    boundary, so a multi-word value keeps only its first token. Widening the
    capture would change every object row and is not part of this fix."""
    assert _triples(graph, "Alice is a senior developer") == [("Alice", "is", "senior")]
    assert _triples(graph, "Bob works at Acme Corp") == [("Bob", "works_at", "Acme")]


def test_low_value_objects_are_rejected_on_every_pattern(graph):
    assert _triples(graph, "Bob is different") == []
    assert _triples(graph, "Bob has nothing") == []
    assert _triples(graph, "Bob uses it") == []
    assert _triples(graph, "Bob works at Nothing") == [("Bob", "works_at", "Nothing")], (
        "a capitalised token is treated as a name and kept")


def test_real_facts_survive(graph):
    assert _triples(graph, "Alice uses Rust") == [("Alice", "uses", "Rust")]
    assert _triples(graph, "Alice uses ComfyUI daily") == [("Alice", "uses", "ComfyUI")]
    assert _triples(graph, "Alice is Sally") == [("Alice", "is", "Sally")]
    assert _triples(graph, "Bob works at Anthropic") == [("Bob", "works_at", "Anthropic")]
    assert _triples(graph, "Rust is fast") == [("Rust", "is", "fast")]


def test_article_led_subjects_are_rejected(graph):
    assert _triples(graph, "The silence is different") == []
    assert _triples(graph, "The build is a success") == []
    assert _triples(graph, "A frame has legs") == []
    assert _triples(graph, "An engineer uses Python") == []
    assert _triples(graph, "The service works at Acme") == []


def test_article_led_names_are_kept(graph):
    """The article rule must not cost a fact about a named entity, on any of
    the four patterns."""
    assert _triples(graph, "The Matrix is a film") == [("The Matrix", "is", "film")]
    assert _triples(graph, "A New Hope has a sequel") == [("A New Hope", "has", "sequel")]
    assert _triples(graph, "The Matrix uses Python") == [("The Matrix", "uses", "Python")]
    assert _triples(graph, "The Matrix works at Warner") == [("The Matrix", "works_at", "Warner")]


def test_mixed_content_keeps_only_the_real_facts(graph):
    triples = _triples(graph, f"{JUNK} {GOOD}")
    assert ("Alice", "is", "developer") in triples
    assert ("Alice", "uses", "Rust") in triples
    assert ("Alice", "works_at", "Anthropic") in triples
    assert all(s != "Bob" for s, _, _ in triples), triples


# ---------------------------------------------------------------------------
# End to end: every ingest path that reaches extract_facts
# ---------------------------------------------------------------------------


def test_remember_does_not_persist_junk(temp_db):
    beam = BeamMemory(session_id="guard-remember", db_path=temp_db)
    beam.remember(JUNK, source="conversation")
    beam.remember(GOOD, source="conversation")
    facts = _fact_rows(beam.conn)
    _assert_clean(facts, "facts")
    assert ("Alice", "is", "developer") in facts
    assert ("The Matrix", "is", "film") in facts, (
        "an article-led name is a fact and must persist")
    consolidated = _consolidated_rows(beam.conn)
    _assert_clean(consolidated, "consolidated_facts")
    assert any(s == "Alice" for s, _, _ in consolidated)
    assert not any(s == "Bob" for s, _, _ in consolidated)


def test_dedup_update_does_not_persist_junk(temp_db):
    """The duplicate-content branch of remember() re-runs graph ingest on
    the existing row; it must apply the same guard."""
    beam = BeamMemory(session_id="guard-dedup", db_path=temp_db)
    first = beam.remember(JUNK, source="conversation")
    again = beam.remember(JUNK, source="conversation")
    assert again == first
    assert not any(s == "Bob" for s, _, _ in _fact_rows(beam.conn))
    assert not any(s == "Bob" for s, _, _ in _consolidated_rows(beam.conn))
    junk_edges = beam.conn.execute(
        "SELECT COUNT(*) FROM graph_edges WHERE target LIKE ?",
        (f"fact_{first}_%",),
    ).fetchone()[0]
    assert junk_edges == 0


def test_remember_batch_does_not_persist_junk(temp_db):
    beam = BeamMemory(session_id="guard-batch", db_path=temp_db)
    ids = beam.remember_batch([
        {"content": JUNK, "source": "conversation"},
        {"content": GOOD, "source": "conversation"},
    ])
    assert len(ids) == 2
    facts = _fact_rows(beam.conn)
    _assert_clean(facts, "facts")
    assert ("Alice", "uses", "Rust") in facts
    assert ("The Matrix", "works_at", "Warner") in facts
    assert not any(s == "Bob" for s, _, _ in facts)
    _assert_clean(_consolidated_rows(beam.conn), "consolidated_facts")


def test_consolidate_to_episodic_does_not_persist_junk(temp_db):
    beam = BeamMemory(session_id="guard-consolidate", db_path=temp_db)
    wm_id = beam.remember("Alice met Bob for coffee", source="conversation")
    beam.consolidate_to_episodic(f"{JUNK} {GOOD}", [wm_id])
    facts = _fact_rows(beam.conn)
    _assert_clean(facts, "facts")
    assert ("Alice", "works_at", "Anthropic") in facts
    assert not any(s == "Bob" for s, _, _ in facts)
    _assert_clean(_consolidated_rows(beam.conn), "consolidated_facts")


def test_fact_recall_mix_carries_no_junk(temp_db):
    """fact_recall reads both `facts` (FTS/LIKE) and `consolidated_facts`
    by capitalised query word; neither source may hand back a junk triple."""
    beam = BeamMemory(session_id="guard-recall", db_path=temp_db)
    beam.remember(JUNK, source="conversation")
    beam.remember(GOOD, source="conversation")
    for query in ("Bob", "Alice", "different", "Bob different"):
        for hit in beam.fact_recall(query):
            content = hit["content"]
            assert "different" not in content, (query, content)
            assert "lready" not in content, (query, content)
            assert not content.endswith(" is re"), (query, content)
            assert not content.endswith(" has it"), (query, content)
    alice = {h["content"] for h in beam.fact_recall("Alice")}
    assert any("Alice is developer" in c or "Alice uses Rust" in c for c in alice), alice
    matrix = {h["content"] for h in beam.fact_recall("Matrix")}
    assert any("The Matrix" in c for c in matrix), matrix
