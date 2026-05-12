"""
Regression tests for `consolidate_fact` PRIMARY KEY collision fix.

Pre-fix: `consolidated_facts.id` was generated as
``f"cf_{subject}_{predicate}_{object}".replace(" ", "_")[:100]``.
Two distinct facts with the same first ~95 chars after replace
produced identical PKs. The resulting IntegrityError raised by
`consolidate_fact`'s INSERT was swallowed by the broad
``except: pass`` at `beam.py:_ingest_graph_and_veracity`
(Phase 4 wrapper), producing silent data loss.

Post-fix: `compute_fact_id` returns a SHA-256 hash of NFC-normalized
SPO with length-prefix framing. Always-uniform 27 chars
(`cf_` + 24 hex). Collision-safe across arbitrary content lengths
and smuggle-safe against in-field separator characters.

These tests pin:
  - Long distinct SPOs that would have collided pre-fix produce
    different IDs.
  - Same SPO → same ID (idempotency).
  - Distinct facts with overlapping truncations produce distinct
    IDs.
  - `consolidate_fact` dedup-by-SPO still works (old rows + new
    rows coexist in mixed-format DBs).
  - The polyphonic engine's `_fact_voice` produces matching IDs
    (RRF fusion keys align with stored IDs).
  - resolve_conflict still works when the caller passes an ID
    obtained via compute_fact_id.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.veracity_consolidation import (
    VeracityConsolidator,
    compute_fact_id,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "consolidated_facts.db"


# ---------------------------------------------------------------------------
# compute_fact_id core contract
# ---------------------------------------------------------------------------


def test_compute_fact_id_is_deterministic_for_same_spo():
    """Same (subject, predicate, object) → same ID across calls.
    Idempotency is the whole reason consolidate_fact can dedup."""
    a = compute_fact_id("Alice", "is", "developer")
    b = compute_fact_id("Alice", "is", "developer")
    assert a == b


def test_compute_fact_id_distinguishes_distinct_spos():
    """Two distinct SPOs produce distinct IDs."""
    a = compute_fact_id("Alice", "is", "developer")
    b = compute_fact_id("Bob", "is", "developer")
    c = compute_fact_id("Alice", "owns", "developer")
    d = compute_fact_id("Alice", "is", "manager")
    assert len({a, b, c, d}) == 4


def test_compute_fact_id_format_is_stable():
    """Format is `cf_` + 24 hex chars = 27 total. Stable for any input."""
    fid = compute_fact_id("X", "is", "Y")
    assert fid.startswith("cf_"), fid
    assert len(fid) == 27, f"unexpected length {len(fid)}: {fid}"
    # All chars after `cf_` should be hex.
    hex_part = fid[3:]
    assert all(c in "0123456789abcdef" for c in hex_part), (
        f"non-hex chars in {fid}"
    )


def test_compute_fact_id_long_content_does_not_collide():
    """Pre-fix bug: long SPOs that shared their first ~95 chars after
    replace produced identical truncated IDs. This test pins that
    distinct long SPOs now produce distinct hashes regardless of
    length."""
    long_subject_a = "Alice Anderson the Senior Staff Engineer responsible for the authentication subsystem"
    long_subject_b = "Alice Anderson the Senior Staff Engineer responsible for the authorization subsystem"
    # The two subjects differ only at the last word. Pre-fix truncation
    # at 100 chars would have included both in full so they wouldn't
    # have collided in THIS specific case — but other pathological
    # cases (very long objects with common prefixes) would. We test a
    # case where the pre-fix output WOULD have collided:
    long_predicate = "is_described_in_the_internal_documentation_at_section_4_paragraph_3_as"
    obj = "a competent and reliable engineer"
    a = compute_fact_id(long_subject_a, long_predicate, obj)
    b = compute_fact_id(long_subject_b, long_predicate, obj)
    assert a != b, (
        "long-content collision: subjects differ but IDs match — "
        "compute_fact_id not actually hashing the full input"
    )


def test_compute_fact_id_separator_prevents_smuggling():
    """The unit-separator (\\x1f) join prevents an attacker (or
    coincidence) from constructing different SPOs that look the
    same after concatenation. E.g.,
        ('a_b', 'c', 'd') vs ('a', 'b_c', 'd')
    must hash to distinct IDs even though their underscore-joined
    forms would be equal."""
    a = compute_fact_id("a_b", "c", "d")
    b = compute_fact_id("a", "b_c", "d")
    assert a != b, (
        "boundary smuggling: differently-bucketed SPOs hash to same id"
    )


# ---------------------------------------------------------------------------
# consolidate_fact end-to-end
# ---------------------------------------------------------------------------


def test_consolidate_fact_stores_hash_based_id(temp_db):
    """The stored row's id should be the hash, not the legacy
    truncated f-string."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Carol", "leads", "the platform team", "stated")
    rows = cons.conn.execute(
        "SELECT id FROM consolidated_facts WHERE subject = 'Carol'"
    ).fetchall()
    assert len(rows) == 1
    stored_id = rows[0]["id"]
    assert stored_id == compute_fact_id("Carol", "leads", "the platform team")
    assert stored_id.startswith("cf_") and len(stored_id) == 27


def test_consolidate_fact_dedup_by_spo_still_works(temp_db):
    """The pre-fix dedup query (`WHERE subject=? AND predicate=?
    AND object=?`) doesn't care what the ID format is, so
    consolidate_fact's idempotency is preserved across the format
    change. Two calls with the same SPO produce one row whose
    mention_count is 2."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Dan", "is", "an engineer", "stated", "mem_a")
    cons.consolidate_fact("Dan", "is", "an engineer", "stated", "mem_b")
    rows = cons.conn.execute(
        "SELECT id, mention_count FROM consolidated_facts "
        "WHERE subject = 'Dan'"
    ).fetchall()
    assert len(rows) == 1, "dedup failed — same SPO produced two rows"
    assert rows[0]["mention_count"] == 2


def test_consolidate_fact_distinct_long_content_both_stored(temp_db):
    """Two facts that would have collided pre-fix (same truncated
    prefix) both get stored under distinct IDs."""
    cons = VeracityConsolidator(db_path=temp_db)
    long_pred = "is_described_in_the_internal_documentation_at_section_4_paragraph_3_as"
    cons.consolidate_fact(
        "EngineerLeadAlice", long_pred,
        "a competent and reliable engineer who delivers on time",
        "stated", "mem_x",
    )
    cons.consolidate_fact(
        "EngineerLeadAlice", long_pred,
        "a competent and reliable engineer who escalates blockers",
        "stated", "mem_y",
    )
    rows = cons.conn.execute(
        "SELECT id FROM consolidated_facts WHERE subject = 'EngineerLeadAlice'"
    ).fetchall()
    assert len(rows) == 2, (
        "long-content collision: pre-fix both rows would have had "
        "the same truncated ID and one would have been lost"
    )
    ids = {r["id"] for r in rows}
    assert len(ids) == 2, f"distinct facts share an ID: {ids}"


# ---------------------------------------------------------------------------
# Backward compat: mixed-format DBs work
# ---------------------------------------------------------------------------


def test_mixed_format_db_dedup_still_finds_old_rows(temp_db):
    """Existing DBs may have rows with the pre-fix `cf_X_y_Z`
    format. The dedup query in `consolidate_fact` matches on SPO,
    not on ID, so old rows are still found on UPDATE — their
    mention_count increments correctly. Only NEW rows get the new
    format ID. Pre- and post-format rows coexist."""
    cons = VeracityConsolidator(db_path=temp_db)
    # Simulate a legacy row by inserting directly with the old format.
    legacy_id = "cf_Eve_is_a_lawyer"
    cons.conn.execute(
        """INSERT INTO consolidated_facts
           (id, subject, predicate, object, confidence, mention_count,
            first_seen, last_seen, sources_json, veracity)
           VALUES (?, 'Eve', 'is', 'a lawyer', 0.5, 1,
                   '2026-01-01T00:00:00', '2026-01-01T00:00:00',
                   '[]', 'stated')""",
        (legacy_id,),
    )
    cons.conn.commit()

    # New consolidate_fact call on the same SPO should UPDATE the
    # legacy row (matched via SPO query), not INSERT a new one.
    result = cons.consolidate_fact("Eve", "is", "a lawyer", "stated", "mem_new")

    rows = cons.conn.execute(
        "SELECT id, mention_count FROM consolidated_facts "
        "WHERE subject = 'Eve'"
    ).fetchall()
    assert len(rows) == 1, (
        "legacy row not matched — consolidate_fact created a duplicate "
        "instead of updating"
    )
    # The legacy row's ID is preserved (we updated by row.id).
    assert rows[0]["id"] == legacy_id
    assert rows[0]["mention_count"] == 2


# ---------------------------------------------------------------------------
# Cross-module: polyphonic engine produces matching IDs
# ---------------------------------------------------------------------------


def test_polyphonic_fact_voice_id_matches_stored(temp_db):
    """`polyphonic_recall._fact_voice` must use compute_fact_id so
    its RRF fusion keys align with the stored consolidated_facts.id.
    Pre-fix the engine reconstructed the legacy truncated f-string;
    post-fix it shares the same generator as consolidate_fact."""
    from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Frank", "is", "a researcher", "stated")
    stored_id = cons.conn.execute(
        "SELECT id FROM consolidated_facts WHERE subject = 'Frank'"
    ).fetchone()[0]

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=cons.conn)
    # `_fact_voice` looks up facts by capitalized words from the query;
    # exercise it with a query that includes 'Frank' so the word loop
    # hits our seeded fact.
    fact_results = engine._fact_voice("Frank")
    assert fact_results, "fact voice returned [] — seeded fact not found"
    engine_id = fact_results[0].memory_id
    assert engine_id == stored_id, (
        f"engine fact-voice ID {engine_id!r} != stored fact ID "
        f"{stored_id!r} — RRF fusion keys diverge from consolidate_fact"
    )


# ---------------------------------------------------------------------------
# resolve_conflict path
# ---------------------------------------------------------------------------


def test_resolve_conflict_with_compute_fact_id(temp_db):
    """Callers that need to reference a fact by ID can use
    compute_fact_id to derive the correct stored ID. Verifies the
    new generator is the single source of truth."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Grace", "is", "the CTO", "stated")
    cons.consolidate_fact("Grace", "is", "the VP", "inferred")
    conflicts = cons.get_conflicts()
    assert conflicts, "no conflicts surfaced for Grace's competing roles"

    # Resolve in favor of "is the CTO" using the hash-derived ID.
    winning_id = compute_fact_id("Grace", "is", "the CTO")
    cons.resolve_conflict(conflicts[0]["id"], winning_id)

    # Conflict should be resolved.
    remaining = cons.get_conflicts()
    assert len(remaining) == 0


# ---------------------------------------------------------------------------
# /review hardening (commit 2)
# ---------------------------------------------------------------------------


class TestReviewHardening:
    """Closes the 5 must-fix findings from the /review army on
    commit 1:
      1. \\x1f separator smuggling (4-source)
      2. SHA-256 vs SHA-1 (codebase consistency)
      3. Unicode NFC/NFD normalization
      4. Input validation
      5. Legacy ID preservation in _fact_voice + resolve_conflict
    """

    def test_separator_smuggling_does_not_collide(self):
        """Pre-fix /review caught: two SPOs with embedded \\x1f could
        produce identical joined byte strings under naive `\\x1f.join`.
        Post-fix length-prefix framing makes the encoding injective."""
        # ("a\x1f", "b", "c") vs ("a", "\x1fb", "c") would naively
        # both produce b"a\x1f\x1fb\x1fc". Length-prefix prevents that.
        a = compute_fact_id("a\x1f", "b", "c")
        b = compute_fact_id("a", "\x1fb", "c")
        assert a != b, (
            "separator smuggling: distinct SPOs hashed to same ID"
        )

    def test_unicode_nfc_and_nfd_hash_identically(self):
        """Same logical text in NFC and NFD form (e.g., 'café' as
        precomposed `é` vs decomposed `e` + combining acute) should
        produce the same ID. /review caught the missing
        normalization as a silent-dedup-miss bug."""
        nfc = "café"  # é as a single codepoint
        nfd = "café"  # e + combining acute
        assert nfc != nfd, "test setup: NFC/NFD strings are equal"
        a = compute_fact_id(nfc, "is", "open")
        b = compute_fact_id(nfd, "is", "open")
        assert a == b, (
            "Unicode normalization not applied: NFC/NFD diverge"
        )

    def test_input_validation_rejects_empty_strings(self):
        """compute_fact_id must reject empty SPO components with
        ValueError rather than silently hashing ('', '', '')."""
        with pytest.raises(ValueError, match="must be non-empty"):
            compute_fact_id("", "is", "developer")
        with pytest.raises(ValueError, match="must be non-empty"):
            compute_fact_id("Alice", "", "developer")
        with pytest.raises(ValueError, match="must be non-empty"):
            compute_fact_id("Alice", "is", "")

    def test_input_validation_rejects_non_string(self):
        """compute_fact_id must reject non-str inputs with TypeError."""
        with pytest.raises(TypeError, match="must be a str"):
            compute_fact_id(None, "is", "developer")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be a str"):
            compute_fact_id("Alice", 42, "developer")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be a str"):
            compute_fact_id("Alice", "is", b"developer")  # type: ignore[arg-type]

    def test_hash_uses_sha256_codebase_consistency(self):
        """Pin the literal hash for a known SPO so a future "let's
        switch hashes" PR can't silently change stored IDs across
        upgrades. This locks SHA-256 of NFC-normalized
        length-prefix-encoded SPO."""
        import hashlib
        import unicodedata

        # Expected: SHA-256 of "5:Alice2:is9:developer" (where 5, 2, 9
        # are the UTF-8 byte lengths of each NFC-normalized component).
        s = unicodedata.normalize("NFC", "Alice").encode("utf-8")
        p = unicodedata.normalize("NFC", "is").encode("utf-8")
        o = unicodedata.normalize("NFC", "developer").encode("utf-8")
        framed = (
            f"{len(s)}:".encode() + s
            + f"{len(p)}:".encode() + p
            + f"{len(o)}:".encode() + o
        )
        expected = "cf_" + hashlib.sha256(framed).hexdigest()[:24]
        assert compute_fact_id("Alice", "is", "developer") == expected

    def test_fact_voice_uses_stored_id_for_legacy_rows(self, temp_db):
        """For mixed-format DBs (legacy rows with pre-fix
        f-string IDs, new rows with hash IDs), `_fact_voice` must
        return the row's stored ID, not the recomputed hash.
        /review caught the prior unconditional `compute_fact_id` as
        a 2-source GATE FAIL."""
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        cons = VeracityConsolidator(db_path=temp_db)
        # Seed a legacy-format row directly (mimic an existing DB).
        legacy_id = "cf_Henry_is_a_lead"
        cons.conn.execute(
            """INSERT INTO consolidated_facts
               (id, subject, predicate, object, confidence,
                mention_count, first_seen, last_seen, sources_json,
                veracity)
               VALUES (?, 'Henry', 'is', 'a lead', 0.8, 1,
                       '2026-01-01', '2026-01-01', '[]', 'stated')""",
            (legacy_id,),
        )
        cons.conn.commit()

        engine = PolyphonicRecallEngine(db_path=temp_db, conn=cons.conn)
        results = engine._fact_voice("Henry")
        assert results, "fact voice returned [] for seeded legacy row"
        # Engine should return the legacy id, not the new hash.
        assert results[0].memory_id == legacy_id, (
            f"engine recomputed instead of using stored legacy id: "
            f"got {results[0].memory_id!r}, expected {legacy_id!r}"
        )

    def test_resolve_conflict_rejects_ambiguous_winning_id(self, temp_db):
        """If winning_fact_id matches neither fact_a_id nor
        fact_b_id, resolve_conflict must NOT silently mark the
        wrong row as superseded. /review caught the prior
        `losing_id = fact_b_id if winning == fact_a_id else fact_a_id`
        default as a silent winner/loser swap on mixed-format DBs."""
        cons = VeracityConsolidator(db_path=temp_db)
        cons.consolidate_fact("Iris", "is", "the lead", "stated")
        cons.consolidate_fact("Iris", "is", "the manager", "inferred")
        conflicts = cons.get_conflicts()
        assert conflicts

        # Pass an ID that matches NEITHER fact_a nor fact_b.
        bogus_id = "cf_definitely_not_in_db_0000000000"
        # Should log + return without writing — verify no row is
        # superseded.
        cons.resolve_conflict(conflicts[0]["id"], bogus_id)

        rows = cons.conn.execute(
            "SELECT id, superseded_by FROM consolidated_facts "
            "WHERE subject = 'Iris'"
        ).fetchall()
        for row in rows:
            assert row["superseded_by"] is None, (
                f"bogus winning_id caused {row['id']} to be marked "
                f"superseded_by={row['superseded_by']!r} — silent "
                "winner/loser swap regressed"
            )
        # The conflict itself should remain unresolved.
        remaining = cons.get_conflicts()
        assert len(remaining) == len(conflicts), (
            "bogus winning_id marked the conflict resolved without "
            "actually superseding anything"
        )

    def test_consolidated_fact_dataclass_carries_id(self, temp_db):
        """ConsolidatedFact dataclass must expose the stored id so
        polyphonic_recall._fact_voice can preserve legacy formats.
        Pre-fix the dataclass lacked `id` and the engine had to
        recompute."""
        cons = VeracityConsolidator(db_path=temp_db)
        cons.consolidate_fact("Jack", "owns", "the auth service", "stated")
        facts = cons.get_consolidated_facts(subject="Jack")
        assert facts
        # Dataclass must carry id.
        assert facts[0].id is not None
        # And it must equal the stored id.
        stored_id = cons.conn.execute(
            "SELECT id FROM consolidated_facts WHERE subject = 'Jack'"
        ).fetchone()[0]
        assert facts[0].id == stored_id
