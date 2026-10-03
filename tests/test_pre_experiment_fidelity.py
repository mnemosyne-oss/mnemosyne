"""Pre-experiment fidelity fixes — regression tests for E4.a.1, E2.a.10, C29.

This file pins three pre-BEAM-recovery-experiment fixes surfaced by the
end-to-end audit on 2026-05-11:

- **E4.a.1 (experiment-relevant):** `consolidate_to_episodic` destroys source-row
  veracity at consolidation. Pre-fix the INSERT omitted the veracity column;
  post-sleep rows took schema default 'unknown' (0.8 multiplier) regardless
  of how confident the sources were. Post-E4 `remember_batch` populates
  veracity per-row, so the destruction is asymmetric and contaminates
  the experiment's ability to measure consolidated-memory recall quality.

- **E2.a.10 (defensive):** `remember_batch` embedding loop silently
  swallowed partial-failure (IndexError mid-loop on short vectors array,
  exception during embed). At 250K-row scale a transient failure would
  invisibly bias the vector voice toward earlier-ingested rows with zero
  operator signal.

- **C29 (cleanup):** veracity weight constants were duplicated across
  `veracity_consolidation.py` (Bayesian compounding) and `beam.py`
  (recall multiplier). Drift risk under env-var overrides.

Why bundle: all three are pre-experiment fidelity work; all three are
small; all three share the veracity / consolidation / embedding ingest
surface. One PR + one /review pass minimizes maintainer review overhead.
"""
from __future__ import annotations

import logging
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from mnemosyne.core.beam import (
    BeamMemory,
    STATED_WEIGHT,
    INFERRED_WEIGHT,
    TOOL_WEIGHT,
    IMPORTED_WEIGHT,
    UNKNOWN_WEIGHT,
)
from mnemosyne.core.veracity_consolidation import (
    VERACITY_WEIGHTS,
    VERACITY_ALLOWED,
    aggregate_veracity,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


# ─────────────────────────────────────────────────────────────────
# C29 — VERACITY_WEIGHTS centralization
# ─────────────────────────────────────────────────────────────────


class TestC29WeightCentralization:
    """beam.py reads default values from veracity_consolidation.VERACITY_WEIGHTS
    so a single change in one place is reflected everywhere — eliminating
    silent drift between Bayesian compounding (consolidation) and the
    veracity multiplier (recall)."""

    def test_default_weights_match_canonical_dict(self):
        """When no env vars set, all beam.py constants equal the canonical
        VERACITY_WEIGHTS dict values."""
        assert STATED_WEIGHT == VERACITY_WEIGHTS["stated"]
        assert INFERRED_WEIGHT == VERACITY_WEIGHTS["inferred"]
        assert TOOL_WEIGHT == VERACITY_WEIGHTS["tool"]
        assert IMPORTED_WEIGHT == VERACITY_WEIGHTS["imported"]
        assert UNKNOWN_WEIGHT == VERACITY_WEIGHTS["unknown"]

    def test_canonical_dict_labels_match_allowlist(self):
        """The keys of VERACITY_WEIGHTS must equal VERACITY_ALLOWED —
        clamp_veracity uses VERACITY_ALLOWED as the gate; if a weight
        exists for a label outside the allowlist, callers can never
        reach that branch (dead weight)."""
        assert set(VERACITY_WEIGHTS.keys()) == VERACITY_ALLOWED


# ─────────────────────────────────────────────────────────────────
# E4.a.1 — aggregate_veracity helper + consolidate_to_episodic wiring
# ─────────────────────────────────────────────────────────────────


class TestAggregateVeracityHelper:
    """Direct unit tests on `aggregate_veracity` — no DB, bypasses sleep
    complexity so the aggregation logic is testable in isolation."""

    def test_empty_input_returns_unknown(self):
        assert aggregate_veracity([]) == "unknown"

    def test_none_input_returns_unknown(self):
        # Defensive: caller might pass None when source rows had no
        # veracity column populated.
        assert aggregate_veracity(None) == "unknown"

    def test_all_invalid_input_returns_unknown(self):
        """Non-canonical labels don't vote; if all sources are invalid,
        the aggregate falls back to 'unknown'."""
        assert aggregate_veracity(["bogus", "made-up", None]) == "unknown"

    def test_single_label_returns_that_label(self):
        assert aggregate_veracity(["stated"]) == "stated"
        assert aggregate_veracity(["inferred"]) == "inferred"

    def test_all_same_label_returns_that_label(self):
        assert aggregate_veracity(["stated"] * 5) == "stated"
        assert aggregate_veracity(["inferred"] * 10) == "inferred"

    def test_clear_majority_wins(self):
        assert aggregate_veracity(["stated", "stated", "stated", "inferred"]) == "stated"
        assert aggregate_veracity(["tool", "tool", "tool", "stated", "inferred"]) == "tool"

    def test_two_way_tie_among_non_unknown_breaks_to_lowest_weight(self):
        """Tied counts among non-'unknown' labels → pick lowest weight.
        stated=1.0, inferred=0.7 → 'inferred' (lower weight) wins."""
        assert aggregate_veracity(["stated", "inferred"]) == "inferred"
        assert aggregate_veracity(["stated", "tool"]) == "tool"  # tool=0.5

    def test_unknown_is_low_priority_not_counted_against_canonical(self):
        """H1 review fix: 'unknown' is the schema default — operator
        intent vs. never-set can't be distinguished. So 'unknown' is
        filtered out of the candidate set whenever any non-'unknown'
        label is present, preventing legacy rows from diluting
        confident signals."""
        # Single 'stated' beats five 'unknown' (pre-H1 would have been 'unknown').
        assert aggregate_veracity(["stated", "unknown", "unknown",
                                    "unknown", "unknown", "unknown"]) == "stated"
        # 'inferred' + 'unknown' → 'inferred' wins (unknown filtered).
        assert aggregate_veracity(["inferred", "unknown"]) == "inferred"
        # All-'unknown' falls back to 'unknown' (no other candidates).
        assert aggregate_veracity(["unknown", "unknown"]) == "unknown"
        # 'tool' + 'unknown' → 'tool' wins (unknown filtered, single
        # candidate beats no competition).
        assert aggregate_veracity(["tool", "unknown", "unknown"]) == "tool"

    def test_three_way_tie_breaks_to_lowest_weight(self):
        # tool=0.5, inferred=0.7, imported=0.6 → tool wins (lowest)
        assert aggregate_veracity(["tool", "inferred", "imported"]) == "tool"

    def test_balanced_three_way_majority_tie_at_count(self):
        """M1 review fix: 3 + 3 + 3 across stated/tool/inferred ties at
        max_count=3 → tie-break to lowest weight = 'tool'. The realistic
        BEAM-scale case where a session has balanced contributions from
        all three sources is more common than singleton-each."""
        assert aggregate_veracity(
            ["stated"] * 3 + ["tool"] * 3 + ["inferred"] * 3
        ) == "tool"

    def test_invalid_values_dropped_then_aggregate(self):
        """Non-canonical labels filtered out; canonical labels still vote."""
        assert aggregate_veracity(["stated", "bogus", "stated", None]) == "stated"
        # Junk-only with one valid: that one wins
        assert aggregate_veracity(["junk", "more junk", "inferred"]) == "inferred"


class TestE4a1ConsolidateToEpisodicVeracity:
    """`consolidate_to_episodic` now takes a `veracity` kwarg; the INSERT
    populates the column. Pre-fix the column wasn't included in the INSERT
    so post-sleep rows defaulted to 'unknown'."""

    def test_consolidate_with_explicit_veracity_stored(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="The user said they prefer dark mode",
            source_wm_ids=["wm-1", "wm-2"],
            veracity="stated",
        )
        row = beam.conn.execute(
            "SELECT veracity FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row["veracity"] == "stated"

    def test_consolidate_with_no_veracity_defaults_unknown(self, temp_db):
        """Back-compat: legacy callers that don't pass veracity get
        the schema default 'unknown', matching pre-fix behavior."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="Legacy caller without veracity",
            source_wm_ids=["wm-1"],
        )
        row = beam.conn.execute(
            "SELECT veracity FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row["veracity"] == "unknown"

    def test_consolidate_clamps_invalid_veracity(self, temp_db):
        """Trust-boundary clamp at the kwarg: bogus values fall back
        to 'unknown' with a WARNING log."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="Caller passed garbage veracity",
            source_wm_ids=["wm-1"],
            veracity="some-random-junk",
        )
        row = beam.conn.execute(
            "SELECT veracity FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row["veracity"] == "unknown"


class TestE4a1SleepEndToEndVeracity:
    """Full sleep() flow with E4-style per-row veracity should preserve
    the aggregated signal in the episodic summary."""

    def _seed_wm_with_veracity(self, db_path, session_id, ts, items):
        """Insert N working_memory rows with explicit veracity values.
        Returns the list of inserted ids."""
        conn = sqlite3.connect(db_path)
        ids = []
        for i, (content, veracity) in enumerate(items):
            rid = f"wm-{session_id}-{i}"
            ids.append(rid)
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, "
                "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rid, content, "conversation", ts, session_id, 0.5, veracity),
            )
        conn.commit()
        conn.close()
        return ids

    def test_all_stated_sources_produce_stated_summary(self, temp_db, monkeypatch):
        """Homogeneous-stated sources → stated summary (1.0 multiplier,
        not the legacy 0.8 unknown default)."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        self._seed_wm_with_veracity(temp_db, "s1", old_ts, [
            ("user wants feature A", "stated"),
            ("user wants feature B", "stated"),
            ("user wants feature C", "stated"),
        ])

        beam.sleep(dry_run=False)
        ep_rows = beam.conn.execute(
            "SELECT veracity FROM episodic_memory"
        ).fetchall()
        assert len(ep_rows) == 1
        assert ep_rows[0]["veracity"] == "stated", (
            "Homogeneous stated sources must produce a stated summary; "
            "pre-fix this would have been 'unknown'."
        )

    def test_mixed_sources_aggregate_correctly(self, temp_db, monkeypatch):
        """Majority stated, minority inferred → stated wins by count."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        self._seed_wm_with_veracity(temp_db, "s1", old_ts, [
            ("explicit user fact 1", "stated"),
            ("explicit user fact 2", "stated"),
            ("explicit user fact 3", "stated"),
            ("derived note", "inferred"),
        ])

        beam.sleep(dry_run=False)
        ep_rows = beam.conn.execute(
            "SELECT veracity FROM episodic_memory"
        ).fetchall()
        assert len(ep_rows) == 1
        assert ep_rows[0]["veracity"] == "stated"

    def test_tied_sources_conservative_resolution(self, temp_db, monkeypatch):
        """2-stated + 2-inferred → tie → inferred wins (lower weight)."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        self._seed_wm_with_veracity(temp_db, "s1", old_ts, [
            ("fact 1", "stated"),
            ("fact 2", "stated"),
            ("note 1", "inferred"),
            ("note 2", "inferred"),
        ])

        beam.sleep(dry_run=False)
        ep_rows = beam.conn.execute(
            "SELECT veracity FROM episodic_memory"
        ).fetchall()
        assert len(ep_rows) == 1
        assert ep_rows[0]["veracity"] == "inferred"

    def test_legacy_null_veracity_sources_default_unknown(self, temp_db, monkeypatch):
        """Pre-E4 source rows had no veracity set (column NULL or 'unknown');
        the aggregator falls back to 'unknown' for them — back-compat."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        # Insert via raw SQL with NULL veracity to simulate pre-E4 rows.
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, "
            "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, NULL)",
            ("wm-legacy-1", "legacy row", "conversation", old_ts, "s1", 0.5),
        )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)
        ep_rows = beam.conn.execute(
            "SELECT veracity FROM episodic_memory"
        ).fetchall()
        assert len(ep_rows) == 1
        # No valid labels in sources → 'unknown' fallback.
        assert ep_rows[0]["veracity"] == "unknown"


# ─────────────────────────────────────────────────────────────────
# E2.a.10 — embedding loop bounds check + logging
# ─────────────────────────────────────────────────────────────────


class TestE2a10EmbeddingLoopDefense:
    """`remember_batch` embedding block: length mismatch must skip + log
    rather than partially-store; exception must log + skip rather than
    silently swallow."""

    def test_length_mismatch_skips_storage_with_warning(self, temp_db, caplog):
        """If `_embeddings.embed()` returns fewer vectors than inputs,
        skip vector storage entirely and log a WARNING — pre-fix the
        IndexError mid-loop would have silently dropped the whole batch."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        items = [{"content": f"row {i}"} for i in range(5)]

        # Patch _embeddings.embed to return short vectors.
        with patch("mnemosyne.core.beam._embeddings") as mock_emb:
            mock_emb.available.return_value = True
            mock_emb.embed.return_value = np.zeros((3, 384), dtype=np.float32)
            mock_emb._DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
            mock_emb.serialize.side_effect = lambda v: "[serialized]"
            with caplog.at_level(logging.WARNING):
                beam.remember_batch(items)

        # No embeddings should have been stored (skip-on-mismatch).
        rows = beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings"
        ).fetchone()
        assert rows[0] == 0
        # WARNING log captured.
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "mismatch" in r.message]
        assert warnings, (
            "Expected a WARNING log for the length mismatch; got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_embed_returns_none_logs_warning(self, temp_db, caplog):
        """If embed() returns None, log + skip cleanly."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        items = [{"content": "row"}]

        with patch("mnemosyne.core.beam._embeddings") as mock_emb:
            mock_emb.available.return_value = True
            mock_emb.embed.return_value = None
            with caplog.at_level(logging.WARNING):
                beam.remember_batch(items)

        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "returned None" in r.message]
        assert warnings

    def test_embed_exception_logs_with_diagnostic(self, temp_db, caplog):
        """If embed() raises, the WARNING log carries the exception
        repr so operators can diagnose."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        items = [{"content": "row"}]

        with patch("mnemosyne.core.beam._embeddings") as mock_emb:
            mock_emb.available.return_value = True
            mock_emb.embed.side_effect = RuntimeError("disk full sim")
            with caplog.at_level(logging.WARNING):
                beam.remember_batch(items)

        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "embedding storage failed" in r.message]
        assert warnings, (
            "Expected a WARNING log on embed() exception; got: "
            f"{[r.message for r in caplog.records]}"
        )
        assert any("disk full sim" in r.message for r in warnings), (
            "Exception repr should appear in the log for operator diagnosis."
        )

    def test_happy_path_still_stores_embeddings(self, temp_db):
        """Sanity: normal flow with matched-length vectors still stores."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        items = [{"content": f"row {i}"} for i in range(3)]

        with patch("mnemosyne.core.beam._embeddings") as mock_emb:
            mock_emb.available.return_value = True
            mock_emb.embed.return_value = np.zeros((3, 384), dtype=np.float32)
            mock_emb._DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
            mock_emb.serialize.side_effect = lambda v: "[serialized]"
            beam.remember_batch(items)

        rows = beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings"
        ).fetchone()
        assert rows[0] == 3

    def test_wm_rows_persisted_even_on_length_mismatch(self, temp_db):
        """L4 review fix: the embedding skip must not roll back the
        working_memory rows themselves — those committed before the
        embedding block runs. Only vectors are dropped."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        items = [{"content": f"row {i}"} for i in range(5)]

        with patch("mnemosyne.core.beam._embeddings") as mock_emb:
            mock_emb.available.return_value = True
            mock_emb.embed.return_value = np.zeros((3, 384), dtype=np.float32)
            mock_emb._DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
            mock_emb.serialize.side_effect = lambda v: "[serialized]"
            beam.remember_batch(items)

        # WM rows survive; only embeddings are skipped.
        wm_count = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory"
        ).fetchone()[0]
        emb_count = beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings"
        ).fetchone()[0]
        assert wm_count == 5
        assert emb_count == 0

    def test_exception_log_includes_exception_type(self, temp_db, caplog):
        """M3 review fix: log message includes `type(exc).__name__` so
        operators can distinguish `sqlite3.OperationalError` from
        `RuntimeError` etc. without parsing the message."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        items = [{"content": "row"}]

        class CustomKaboom(Exception):
            pass

        with patch("mnemosyne.core.beam._embeddings") as mock_emb:
            mock_emb.available.return_value = True
            mock_emb.embed.side_effect = CustomKaboom("boom")
            with caplog.at_level(logging.WARNING):
                beam.remember_batch(items)

        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "embedding storage failed" in r.message]
        assert warnings
        assert any("CustomKaboom" in r.message for r in warnings), (
            "Expected exception type name to appear in the log; got: "
            f"{[r.message for r in warnings]}"
        )


# ─────────────────────────────────────────────────────────────────
# Review-hardening tests
# ─────────────────────────────────────────────────────────────────


class TestReviewHardening:
    """Tests pinning the cross-source-convergent review findings on
    commit 4 of the bundle: dedup-veracity refresh (P1), graph veracity
    threading (H2), polyphonic-arm veracity reach (L2)."""

    def test_dedup_remember_refreshes_veracity_to_non_unknown(self, temp_db):
        """P1 review fix: re-remembering the same content with a stronger
        veracity must update the row (don't strand the stale label).
        Without this, E4.a.1's sleep-time aggregator inherits the old
        label and re-deflates the summary."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = "same content reasserted with stronger veracity"
        # First remember with default (which clamps via remember(), but
        # let's pass an explicit 'unknown' to simulate a system-default ingest).
        mid = beam.remember(content, source="conversation", veracity="unknown")
        row = beam.conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row["veracity"] == "unknown"

        # Re-remember the same content as 'stated' — should upgrade.
        beam.remember(content, source="conversation", veracity="stated")
        row = beam.conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row["veracity"] == "stated", (
            "dedup-update should have refreshed veracity from 'unknown' "
            "to 'stated'; pre-fix the stale label persisted."
        )

    def test_dedup_remember_with_unknown_preserves_existing_stronger_label(self, temp_db):
        """P1 fix policy: only upgrade when new veracity is non-'unknown'.
        A backfill call that doesn't carry trust signal (defaults to
        'unknown' via the clamp) must NOT downgrade an existing 'stated'."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = "stated content that gets backfilled later"
        # First remember as 'stated'.
        mid = beam.remember(content, source="conversation", veracity="stated")
        # Backfill with no explicit veracity (defaults to 'unknown').
        beam.remember(content, source="conversation", veracity="unknown")
        row = beam.conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row["veracity"] == "stated", (
            "'unknown' backfill should not have downgraded existing 'stated'; "
            "the CASE-WHEN guard protects against this."
        )

    def test_consolidate_threads_aggregated_veracity_into_graph_ingest(
        self, temp_db, monkeypatch
    ):
        """H2 review fix: `consolidate_to_episodic` now passes the
        aggregated veracity (not hardcoded 'inferred') into
        `_ingest_graph_and_veracity`, so downstream Bayesian compounding
        on consolidated facts uses the source-aggregated signal."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        captured = {}

        def spy(memory_id, content, source, veracity="unknown"):
            captured["veracity"] = veracity

        monkeypatch.setattr(beam, "_ingest_graph_and_veracity", spy)
        beam.consolidate_to_episodic(
            summary="some stated summary",
            source_wm_ids=["wm-1"],
            veracity="stated",
        )
        assert captured["veracity"] == "stated", (
            "graph/fact extraction should have received the aggregated "
            "'stated' veracity; pre-fix it received hardcoded 'inferred'."
        )

    def test_polyphonic_arm_consumes_aggregated_veracity(self, temp_db, monkeypatch):
        """L2 review fix: end-to-end test that Arm B's polyphonic recall
        path applies the new aggregated veracity multiplier. Without this
        test, a future refactor could silently strip the multiplier from
        the engine path and the experiment would lose the trust-signal
        ranking entirely on Arm B."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Two episodic rows: one stated, one unknown. Same content matches
        # the query equally — only veracity multiplier differentiates.
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-stated", "user wants dark mode", "consolidation",
             datetime.now().isoformat(), "s1", 0.5, "stated"),
        )
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, veracity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-unknown", "user wants dark mode", "consolidation",
             datetime.now().isoformat(), "s1", 0.5, "unknown"),
        )
        beam.conn.commit()

        # Mock the engine to return both rows with equal RRF score so
        # the veracity multiplier is the only differentiator.
        class _FakeEngine:
            def __init__(self, results):
                self._results = results
            def recall(self, *, query, query_embedding, top_k,
                       default_dense_source_filter=True, source=None, topic=None,
                       episodic_where=None, episodic_params=()):
                return self._results

        engine = _FakeEngine([
            PolyphonicResult(memory_id="ep-stated", combined_score=0.5,
                             voice_scores={"vector": 0.5}, metadata={}),
            PolyphonicResult(memory_id="ep-unknown", combined_score=0.5,
                             voice_scores={"vector": 0.5}, metadata={}),
        ])
        monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: engine)

        results = beam.recall("dark mode", top_k=10)
        # Both surface; stated ranks higher because veracity multiplier
        # 1.0 > 0.8 for unknown.
        ids = [r["id"] for r in results if r["id"] in {"ep-stated", "ep-unknown"}]
        assert len(ids) == 2
        assert ids[0] == "ep-stated", (
            f"stated row should rank above unknown row; got order: {ids}"
        )
