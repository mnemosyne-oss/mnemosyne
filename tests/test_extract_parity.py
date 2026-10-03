"""Regression tests for [C12.a]: Mnemosyne.remember(extract=True) writes
fact triples but skips the `facts` table. The canonical helper
_extract_and_store_facts in beam.py writes BOTH tables; the wrapper's
inline extract block only wrote triples. As a result, wrapper-extracted
facts were visible through recall() (which scores fact triples) but
invisible through fact_recall() (which queries the facts table directly).

Bug: mnemosyne/core/memory.py — wrapper's `if extract:` block only called
triples.add_facts(), never _store_facts_in_table().

These tests assert wrapper / direct parity across all four observable
effects of extract=True:
  1. triples table populated
  2. facts table populated
  3. recall() can find the memory
  4. fact_recall() can find the fact

Plus a parity check for extract_entities=True.
"""

import pytest

from mnemosyne.core.memory import Mnemosyne


@pytest.fixture
def fake_extract_facts(monkeypatch):
    """Patch extract_facts_safe to return deterministic facts.
    Both Mnemosyne.remember and BeamMemory's _extract_and_store_facts
    import via `from mnemosyne.core.extraction import extract_facts_safe`,
    so a module-level patch covers both paths.
    """
    facts = [
        "alice was born in boston",
        "alice studied mathematics at MIT",
    ]
    monkeypatch.setattr(
        "mnemosyne.core.extraction.extract_facts_safe",
        lambda content, **kwargs: list(facts),
    )
    return facts


def _facts_table_count(db_path) -> int:
    """Count rows in the facts table directly. Returns 0 if table missing
    (the bug surface — table never gets written when wrapper fails to
    populate it for the first time)."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT COUNT(*) FROM facts")
        return cur.fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def _triples_table_count(db_path) -> int:
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT COUNT(*) FROM triples")
        return cur.fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


class TestWrapperExtractFactsTableParity:
    """C12.a: Mnemosyne.remember(extract=True) must populate the facts
    table the same way BeamMemory.remember(extract=True) does."""

    def test_wrapper_extract_writes_facts_table(self, tmp_path, fake_extract_facts):
        db_path = tmp_path / "c12a.db"
        mem = Mnemosyne(session_id="c12a", db_path=db_path)
        mem.remember(
            "Alice was born in Boston in 1990 and studied math at MIT.",
            source="user",
            extract=True,
        )
        assert _facts_table_count(db_path) >= 1, (
            "facts table empty after Mnemosyne.remember(extract=True); "
            "wrapper path should populate it like BeamMemory does"
        )

    def test_wrapper_extract_writes_annotations_table(self, tmp_path, fake_extract_facts):
        """Regression guard: the facts-table fix must produce a populated
        annotation store. Post-E6 (PR #70), extraction routes to the new
        AnnotationStore instead of the legacy `triples` table — the C12.a
        contract is preserved by verifying `kind='fact'` annotations exist.
        """
        from mnemosyne.core.annotations import AnnotationStore
        db_path = tmp_path / "c12a.db"
        mem = Mnemosyne(session_id="c12a", db_path=db_path)
        mem.remember(
            "Alice was born in Boston in 1990 and studied math at MIT.",
            source="user",
            extract=True,
        )
        ann_store = AnnotationStore(db_path=db_path)
        fact_rows = ann_store.query_by_kind("fact")
        assert len(fact_rows) >= 1, (
            "annotations table empty for kind='fact' after extract=True; "
            "the wrapper's extraction behavior must not regress"
        )

    def test_wrapper_extracted_fact_is_visible_via_fact_recall(self, tmp_path, fake_extract_facts):
        """The contract: extract=True through the wrapper must produce
        facts retrievable through the public fact_recall surface."""
        db_path = tmp_path / "c12a.db"
        mem = Mnemosyne(session_id="c12a", db_path=db_path)
        mem.remember(
            "Alice was born in Boston in 1990 and studied math at MIT.",
            source="user",
            extract=True,
        )
        results = mem.beam.fact_recall("alice")
        assert results, "fact_recall returned no results for wrapper-extracted facts"
        contents = " ".join(str(r.get("content", "")).lower() for r in results)
        assert "alice" in contents

    def test_extract_runs_on_dedup_for_backfill(self, tmp_path, fake_extract_facts):
        """Backfill contract: a user with pre-existing working_memory rows
        (written before extract=True was supported) calls
        `mem.remember(same_content, extract=True)` to populate the facts
        table after-the-fact. Even though the dedup path fires (content
        already exists), extraction must still run.

        Pre-fix this regression scenario was silently broken: my initial
        delegation moved extraction inside BeamMemory.remember, which has
        an early-return on dedup that skipped both extract blocks. Locks
        in the fix that makes the dedup branch also call
        _extract_and_store_facts / _extract_and_store_entities.
        """
        from mnemosyne.core.beam import BeamMemory
        db_path = tmp_path / "c12a.db"
        # Pre-existing row, no extraction (simulating an old DB)
        beam = BeamMemory(session_id="c12a", db_path=db_path)
        first_id = beam.remember(
            "Alice was born in Boston in 1990 and studied math at MIT.",
            source="user",
            extract=False,
        )
        # Backfill: same content, now with extract=True
        mem = Mnemosyne(session_id="c12a", db_path=db_path)
        second_id = mem.remember(
            "Alice was born in Boston in 1990 and studied math at MIT.",
            source="user",
            extract=True,
        )
        assert first_id == second_id, (
            "Dedup did not fire: backfill expectation requires the "
            "second call to recognize the existing row"
        )
        assert _facts_table_count(db_path) >= 1, (
            "Backfill failed: facts table empty after extract=True on "
            "duplicate content. Dedup branch in BeamMemory.remember must "
            "run extraction so the C12.a contract holds for backfill scenarios."
        )

    def test_wrapper_and_direct_paths_produce_same_table_state(self, tmp_path, fake_extract_facts):
        """Wrapper path and direct-Beam path must produce equivalent
        fact-table state for the same input. Eliminates the asymmetry
        that v2 plan §C12.a calls out."""
        wrapper_db = tmp_path / "wrapper.db"
        direct_db = tmp_path / "direct.db"
        content = "Alice was born in Boston in 1990 and studied math at MIT."

        wrapper_mem = Mnemosyne(session_id="parity", db_path=wrapper_db)
        wrapper_mem.remember(content, source="user", extract=True)

        # Direct path: BeamMemory.remember(extract=True) is the canonical
        # one that already populates both tables.
        from mnemosyne.core.beam import BeamMemory
        direct_beam = BeamMemory(session_id="parity", db_path=direct_db)
        direct_beam.remember(content, source="user", extract=True)

        wrapper_facts = _facts_table_count(wrapper_db)
        direct_facts = _facts_table_count(direct_db)
        assert wrapper_facts == direct_facts, (
            f"Wrapper wrote {wrapper_facts} facts rows; direct wrote {direct_facts}. "
            f"Paths should produce identical fact-table state for identical input."
        )


class TestWrapperExtractEntitiesParity:
    """Adjacent parity check: extract_entities=True path. Option A delegates
    this to BeamMemory's _extract_and_store_entities helper; this test
    locks in equivalent observable behavior."""

    def test_wrapper_extract_entities_writes_mention_annotations(self, tmp_path):
        """Post-E6: mentions land in the annotations table, not triples.
        Method renamed to match the new storage target."""
        from mnemosyne.core.annotations import AnnotationStore
        db_path = tmp_path / "c12a.db"
        mem = Mnemosyne(session_id="c12a", db_path=db_path)
        # A content string that the regex extractor will pick entities from.
        # extract_entities_regex matches things like CapitalizedWords and
        # @mentions depending on the regex. Use a simple sentence with
        # capitalized proper nouns.
        mem.remember(
            "Alice met Bob in Paris last Tuesday.",
            source="user",
            extract_entities=True,
        )
        # At least one annotation with kind='mentions' should exist.
        ann_store = AnnotationStore(db_path=db_path)
        rows = ann_store.query_by_kind("mentions")
        assert len(rows) >= 1, (
            "extract_entities=True did not produce 'mentions' annotations"
        )

    def test_dialogue_does_not_write_quoted_fragments(self, tmp_path):
        """Ingest-path regression: a quoted span never reaches the mentions
        vocabulary, and the names in the same turn still do."""
        from mnemosyne.core.annotations import AnnotationStore
        db_path = tmp_path / "c12b.db"
        mem = Mnemosyne(session_id="c12b", db_path=db_path)
        mem.remember(
            "'Okay,' Alice said. 'The room is quiet now.' Bob nodded.",
            source="user",
            extract_entities=True,
        )
        ann_store = AnnotationStore(db_path=db_path)
        # filter_noise=False: assert the fragments were never written, not
        # that a read-time filter hides them.
        values = {
            row["value"]
            for row in ann_store.query_by_kind("mentions", filter_noise=False)
        }
        assert "Alice" in values
        assert "Bob" in values
        assert not [v for v in values if any(c in v for c in ",.'\"")], (
            f"quoted dialogue reached the mentions vocabulary: {sorted(values)}"
        )
