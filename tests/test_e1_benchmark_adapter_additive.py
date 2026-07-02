"""Regression tests for E1 — additive BEAM benchmark adapter.

Pre-E1: `_benchmarks/evaluate_beam_end_to_end.py::ingest_conversation` ran a
destructive batch-summary pattern. Per BATCH_SIZE=500 messages it:
  1. SELECTed all working_memory rows for the session
  2. Built a synthetic summary: "Batch N: first_3_msg_contents[:100chars]"
     truncated to 500 chars
  3. Called consolidate_to_episodic with that paltry summary
  4. DELETEd all source working_memory rows

Net effect: ~99% of message content was discarded before recall could
see it. At 500K messages (~250K conversation turns) the BEAM benchmark
ran on ~500 episodic rows of mostly-empty template strings instead of
the actual corpus. Every experiment arm produced a noise-dominated
signal.

Post-E1 (option b — use the real pipeline now that E3 is additive):
  - `remember_batch` writes originals (unchanged)
  - Timestamps are backdated past sleep's cutoff (benchmark-internal
    detail, real users don't need this)
  - `beam.sleep()` produces real LLM-generated (or AAAK-fallback)
    summaries on top of preserved originals
  - working_memory row count is preserved
  - episodic_memory grows with substantive summaries

These tests pin the new contract.
"""

import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add _benchmarks/ to path so we can import the benchmark adapter directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def disable_llm(monkeypatch):
    """Force the AAAK fallback path in beam.sleep() so the test is
    deterministic without depending on a local LLM model. AAAK is
    phrase-substitution + compaction; uncommon literal tokens survive
    intact."""
    monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)


def _make_messages(n: int) -> list:
    """Build n synthetic conversation messages with distinct unique
    tokens. The tokens are deliberately rare so post-ingest recall can
    locate them precisely."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        # Unique token per message — survives aaak summarization.
        msgs.append({
            "role": role,
            "content": (
                f"e1uniq{i:04d} message content for index {i} "
                f"talking about fooproject{i}"
            ),
        })
    return msgs


def _import_benchmark_adapter():
    """Lazy import so module-load failures don't kill the whole test
    suite (the tool has optional deps like requests)."""
    import importlib.util
    tool_path = _REPO_ROOT / "_benchmarks" / "evaluate_beam_end_to_end.py"
    spec = importlib.util.spec_from_file_location(
        "_e1_benchmark_adapter", tool_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestE1AdditiveBenchmarkIngest:

    def test_originals_preserved_after_ingest(self, temp_db, disable_llm):
        """[E1 invariant] Post-ingest working_memory row count equals
        the input message count. Pre-E1 the destructive consolidation
        DELETEd source rows, leaving ~0 rows after each batch."""
        adapter = _import_benchmark_adapter()
        beam = BeamMemory(session_id="e1-test", db_path=temp_db)

        messages = _make_messages(10)
        stats = adapter.ingest_conversation(beam, messages)

        wm_count = sqlite3.connect(str(temp_db)).execute(
            "SELECT COUNT(*) FROM working_memory WHERE session_id = ?",
            ("e1-test",),
        ).fetchone()[0]

        assert wm_count == len(messages), (
            f"E1 contract violated: working_memory has {wm_count} rows, "
            f"input had {len(messages)} messages. Pre-E1 the destructive "
            f"consolidation DELETEd source rows; post-E1 originals stay."
        )
        # Stats should reflect actual stored content, not deleted count.
        assert stats["wm_count"] == len(messages), (
            f"stats[wm_count] should reflect retained rows post-E1; "
            f"got {stats['wm_count']}, expected {len(messages)}"
        )

    def test_originals_recallable_after_ingest(self, temp_db, disable_llm):
        """[E1 invariant] Recall returns the actual message content,
        not a template summary. Pre-E1 unique tokens after the first
        ~3 messages in each batch were silently discarded; only the
        first 100 chars of the first 3 messages survived into the
        synthetic summary."""
        adapter = _import_benchmark_adapter()
        beam = BeamMemory(session_id="e1-test", db_path=temp_db)

        messages = _make_messages(10)
        adapter.ingest_conversation(beam, messages)

        # Probe a token from the 5th message — pre-E1 this would have
        # been discarded because only the first 3 message-contents made
        # it into the synthetic summary.
        results = beam.recall("e1uniq0005", top_k=20)
        assert results, (
            "recall(e1uniq0005) returned 0 results — the 5th message's "
            "content was discarded by the ingest path. E1 should "
            "preserve it via the additive sleep pipeline."
        )
        # The hit should contain the actual message content, not a
        # template prefix.
        contents = [r.get("content", "") for r in results]
        assert any("e1uniq0005" in c for c in contents), (
            f"recall returned hits but none carry the unique token; "
            f"the row was likely returned as a synthetic summary "
            f"rather than the real content. Got: {contents}"
        )

    def test_episodic_summaries_are_substantive(self, temp_db, disable_llm):
        """[E1 invariant] Episodic summaries produced by sleep() are
        real consolidations (LLM-generated or aaak-compressed), not
        the pre-E1 template `Batch N: first_3_msg_contents[:100]...`."""
        adapter = _import_benchmark_adapter()
        beam = BeamMemory(session_id="e1-test", db_path=temp_db)

        messages = _make_messages(10)
        adapter.ingest_conversation(beam, messages)

        rows = sqlite3.connect(str(temp_db)).execute(
            "SELECT content, source FROM episodic_memory "
            "WHERE session_id = ?",
            ("e1-test",),
        ).fetchall()

        assert rows, (
            "No episodic summaries produced — sleep didn't run, or it "
            "produced zero summaries because the cutoff/backdate isn't "
            "working as expected."
        )
        # Pre-E1 template form: "Batch N: ..." — must NOT appear.
        for content, source in rows:
            assert not content.startswith("Batch "), (
                f"episodic row uses pre-E1 template form: "
                f"{content[:80]!r} (source={source!r})"
            )
            # Real summaries should mention some of the ingested tokens
            # or be a non-trivial length, not a fixed-pattern stub.
            assert len(content) > 0, f"empty episodic content"
        # At least one summary should reference content from the batch
        # (via aaak the unique tokens get phrase-substituted but the
        # length signal should still scale with input).
        assert any(len(c) > 50 for c, _ in rows), (
            "all episodic summaries are short stubs — sleep may have "
            "fallen back to a trivial path"
        )

    def test_originals_marked_consolidated(self, temp_db, disable_llm):
        """[E3 composition] Sleep marks the originals consolidated_at;
        the benchmark adapter should drive the additive sleep path
        that produces marked-but-preserved rows."""
        adapter = _import_benchmark_adapter()
        beam = BeamMemory(session_id="e1-test", db_path=temp_db)

        messages = _make_messages(8)
        adapter.ingest_conversation(beam, messages)

        marked = sqlite3.connect(str(temp_db)).execute(
            "SELECT COUNT(*) FROM working_memory "
            "WHERE session_id = ? AND consolidated_at IS NOT NULL",
            ("e1-test",),
        ).fetchone()[0]

        assert marked == len(messages), (
            f"only {marked}/{len(messages)} originals were marked "
            f"consolidated_at; sleep didn't fire on every row, or the "
            f"backdate step is off"
        )

    def test_sleep_drains_full_batch_under_small_sleep_batch_size(
        self, temp_db, disable_llm, monkeypatch
    ):
        """[E1 Codex review P2] If MNEMOSYNE_SLEEP_BATCH is configured
        below the benchmark's BATCH_SIZE, a single beam.sleep() call
        only claims part of the backdated batch. The remaining rows
        carry a TTL-old timestamp AND consolidated_at IS NULL —
        _trim_working_memory's predicate. On the next remember_batch
        the trim would DELETE them as 'old working memory,' violating
        the preserved-corpus contract. The adapter must drain the
        full batch (loop sleep until no_op) regardless of env config."""
        # Override SLEEP_BATCH_SIZE to 3 so a 10-message batch needs
        # multiple sleep calls to drain. The constant is module-level
        # and read at import time; monkeypatching the module attr is
        # sufficient because sleep() reads via the bound name.
        import mnemosyne.core.beam as beam_module
        monkeypatch.setattr(beam_module, "SLEEP_BATCH_SIZE", 3)

        adapter = _import_benchmark_adapter()
        beam = BeamMemory(session_id="e1-drain", db_path=temp_db)

        # 10 messages → at least 4 sleep iterations needed (3 + 3 + 3 + 1).
        messages = _make_messages(10)
        adapter.ingest_conversation(beam, messages)

        # All originals must survive AND be marked consolidated.
        conn = sqlite3.connect(str(temp_db))
        wm_count = conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE session_id = ?",
            ("e1-drain",),
        ).fetchone()[0]
        unconsolidated = conn.execute(
            "SELECT COUNT(*) FROM working_memory "
            "WHERE session_id = ? AND consolidated_at IS NULL",
            ("e1-drain",),
        ).fetchone()[0]
        conn.close()

        assert wm_count == len(messages), (
            f"only {wm_count}/{len(messages)} rows survived under "
            f"SLEEP_BATCH_SIZE=3; _trim deleted un-drained rows."
        )
        assert unconsolidated == 0, (
            f"{unconsolidated} rows still un-consolidated after ingest; "
            f"sleep didn't drain the full batch when SLEEP_BATCH_SIZE "
            f"was smaller than the benchmark batch."
        )

    def test_backdate_does_not_clobber_other_batches(self, temp_db, disable_llm):
        """[E1 adversarial F1/F3] The per-batch backdate UPDATE must
        scope to the current batch's ids. If it walked every
        consolidated_at-IS-NULL row in the session it would clobber
        timestamps from prior batches whose sleep partially failed,
        AND would silently mutate any pre-existing user data sharing
        the session_id. Test: seed a row outside the benchmark's
        ingestion path, run the benchmark with messages in the same
        session, verify the seeded row's timestamp is untouched."""
        adapter = _import_benchmark_adapter()
        beam = BeamMemory(session_id="e1-scope", db_path=temp_db)

        # Pre-seed a row in the SAME session, with a current timestamp.
        # This represents a row that arrived through some other path
        # (real-user write, prior partial-failed sleep, hand edit).
        pre_seed_ts = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, session_id, importance, consolidated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            ("pre-seed", "pre-existing content", "user", pre_seed_ts, "e1-scope", 0.5),
        )
        beam.conn.commit()

        # Now run the benchmark over different messages in the same session.
        messages = _make_messages(5)
        adapter.ingest_conversation(beam, messages)

        # Pre-seeded row's timestamp must be untouched.
        seeded_ts = sqlite3.connect(str(temp_db)).execute(
            "SELECT timestamp FROM working_memory WHERE id = 'pre-seed'"
        ).fetchone()[0]
        assert seeded_ts == pre_seed_ts, (
            f"pre-seeded row's timestamp was clobbered by the batch "
            f"backdate UPDATE: was {pre_seed_ts!r}, now {seeded_ts!r}. "
            f"E1 backdate must scope to batch_ids, not session_id."
        )

    def test_multibatch_ingest_preserves_corpus(self, temp_db, disable_llm):
        """At BATCH_SIZE boundary the loop iterates multiple times.
        Each batch must contribute its messages to working_memory and
        its summary to episodic — no batch's content should be lost.

        Assumptions pinned (so a future BATCH_SIZE / source-grouping
        refactor surfaces this test as part of its blast radius):
          - BATCH_SIZE=500 (module-level constant in the adapter)
          - Messages alternate role between 'user' and 'assistant',
            producing 2 source groups per batch and at least 2
            summaries per batch via sleep's group-by-source logic
          - 600 messages → 2 batches → ep_count >= 2 minimum
            (typically 4 because 2 sources × 2 batches)"""
        adapter = _import_benchmark_adapter()
        # Patch BATCH_SIZE down to 5 so we exercise the multi-batch path
        # without ingesting hundreds of messages in a unit test.
        import importlib.util
        tool_path = _REPO_ROOT / "_benchmarks" / "evaluate_beam_end_to_end.py"
        spec = importlib.util.spec_from_file_location("_e1_multibatch", tool_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # ingest_conversation reads BATCH_SIZE from its own closure
        # (defined as a local in the function). Monkeypatching it
        # requires either making it a module-level constant first
        # OR seeding enough messages to cross the default boundary.
        # Easier path: use the default 500 and seed enough messages
        # to span 2 batches.
        beam = BeamMemory(session_id="e1-multibatch", db_path=temp_db)
        # Two batches' worth — at default BATCH_SIZE=500 this is
        # 1000 messages. Skip the test if that's too slow on CI.
        msg_count = 600  # > BATCH_SIZE, < what would crawl on CI
        messages = _make_messages(msg_count)
        mod.ingest_conversation(beam, messages)

        wm_count = sqlite3.connect(str(temp_db)).execute(
            "SELECT COUNT(*) FROM working_memory WHERE session_id = ?",
            ("e1-multibatch",),
        ).fetchone()[0]
        ep_count = sqlite3.connect(str(temp_db)).execute(
            "SELECT COUNT(*) FROM episodic_memory WHERE session_id = ?",
            ("e1-multibatch",),
        ).fetchone()[0]

        assert wm_count == msg_count, (
            f"multi-batch ingest lost rows: got {wm_count}/{msg_count}"
        )
        assert ep_count >= 2, (
            f"expected at least one summary per batch, got "
            f"{ep_count} for {msg_count} messages across "
            f"~{(msg_count // 500) + 1} batches"
        )
