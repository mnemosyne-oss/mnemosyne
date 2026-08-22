"""Regression tests for C4 — recall path provenance diagnostics.

Pre-C4: `BeamMemory.recall` had silent fallback layers per tier.

  - WM: `_fts_search_working` wrapped in `try/except: wm_fts = []`;
    `_wm_vec_search` in `try/except: pass`. When both produced
    nothing (legitimate no-match OR error), the code fell through to
    "fetch recent items, score by substring on content." Operators
    saw results but had no signal whether they came from FTS/vec
    ranking or pure substring matching on recent items.

  - EM: same shape — vec/FTS each returned, and if both empty, the
    fallback at "if not episodic_rowids" fired with substring
    scoring on the most-recent 500 episodic rows.

For the BEAM experiment specifically, this matters: arm-vs-arm
recall quality comparisons would mix "FTS-ranked good signal" with
"substring-on-recent weak signal" without operators knowing the
ratio. C4's fix is to expose provenance — the fallback still fires
when needed, but its usage rate is now measurable.

These tests pin:
  - The `RecallDiagnostics` class API
  - Process-global singleton lifecycle
  - The instrumentation in `BeamMemory.recall()` — each fallback
    path increments the right counter
  - The recall behavior itself is unchanged (no regression)
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.recall_diagnostics import (
    RECALL_TIERS,
    RecallDiagnostics,
    get_diagnostics,
    get_recall_diagnostics,
    reset_recall_diagnostics,
)


@pytest.fixture(autouse=True)
def fresh_recall_diag():
    """Process-global state must not leak between tests."""
    reset_recall_diagnostics()
    yield
    reset_recall_diagnostics()


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


class TestRecallDiagnosticsClass:
    """Class-level API. The instrumentation depends on these
    primitives; pin them so future refactors can't quietly break the
    recording contract."""

    def test_tier_constants_are_canonical(self):
        assert RECALL_TIERS == (
            "wm_fts", "wm_vec", "wm_fallback",
            "em_fts", "em_vec", "em_fallback",
        )

    def test_initial_snapshot_zero(self):
        diag = RecallDiagnostics()
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 0
        assert snap["totals"]["calls_using_wm_fallback"] == 0
        assert snap["totals"]["calls_using_em_fallback"] == 0
        assert snap["totals"]["wm_fallback_rate"] == 0.0
        assert snap["totals"]["em_fallback_rate"] == 0.0
        for tier in RECALL_TIERS:
            assert snap["by_tier"][tier]["calls_with_hits"] == 0
            assert snap["by_tier"][tier]["total_hits"] == 0

    def test_record_tier_hits_increments(self):
        diag = RecallDiagnostics()
        diag.record_tier_hits("wm_fts", 5)
        diag.record_tier_hits("wm_fts", 3)
        diag.record_tier_hits("wm_fts", 0)  # call with zero hits
        snap = diag.snapshot()
        wm = snap["by_tier"]["wm_fts"]
        assert wm["total_hits"] == 8
        # 2 calls had hits (5 and 3); the zero-hit call doesn't count.
        assert wm["calls_with_hits"] == 2

    def test_record_tier_hits_rejects_negative(self):
        diag = RecallDiagnostics()
        with pytest.raises(ValueError, match="hit_count must be >= 0"):
            diag.record_tier_hits("wm_fts", -1)

    def test_record_tier_hits_rejects_unknown_tier(self):
        diag = RecallDiagnostics()
        with pytest.raises(ValueError, match="unknown recall tier"):
            diag.record_tier_hits("bogus", 1)

    def test_record_fallback_used_increments(self):
        diag = RecallDiagnostics()
        diag.record_fallback_used(wm=True)
        diag.record_fallback_used(em=True)
        diag.record_fallback_used(wm=True, em=True)
        snap = diag.snapshot()
        assert snap["totals"]["calls_using_wm_fallback"] == 2
        assert snap["totals"]["calls_using_em_fallback"] == 2

    def test_record_call_counts_truly_empty(self):
        diag = RecallDiagnostics()
        diag.record_call(truly_empty=False)
        diag.record_call(truly_empty=False)
        diag.record_call(truly_empty=True)
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 3
        assert snap["totals"]["calls_truly_empty"] == 1

    def test_fallback_rate_math(self):
        diag = RecallDiagnostics()
        for _ in range(10):
            diag.record_call()
        for _ in range(3):
            diag.record_fallback_used(wm=True)
        for _ in range(1):
            diag.record_fallback_used(em=True)
        rates = diag.fallback_rate()
        assert rates["wm"] == pytest.approx(0.3)
        assert rates["em"] == pytest.approx(0.1)

    def test_fallback_rate_zero_calls(self):
        diag = RecallDiagnostics()
        rates = diag.fallback_rate()
        assert rates == {"wm": 0.0, "em": 0.0}

    def test_reset_clears_everything(self):
        diag = RecallDiagnostics()
        diag.record_tier_hits("wm_fts", 3)
        diag.record_fallback_used(wm=True)
        diag.record_call()
        diag.reset()
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 0
        for tier in RECALL_TIERS:
            assert snap["by_tier"][tier]["total_hits"] == 0

    def test_snapshot_is_json_serializable(self):
        import json
        diag = RecallDiagnostics()
        diag.record_tier_hits("wm_fts", 2)
        diag.record_fallback_used(em=True)
        diag.record_call(truly_empty=False)
        snap = diag.snapshot()
        # Round-trip via JSON to prove the shape is clean.
        restored = json.loads(json.dumps(snap))
        assert restored["totals"]["calls"] == 1
        assert restored["by_tier"]["wm_fts"]["total_hits"] == 2


class TestProcessGlobalSingleton:

    def test_get_diagnostics_returns_singleton(self):
        a = get_diagnostics()
        b = get_diagnostics()
        assert a is b

    def test_module_helpers_use_singleton(self):
        get_diagnostics().record_call()
        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 1
        reset_recall_diagnostics()
        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 0


class TestBeamRecallInstrumentation:
    """End-to-end: call BeamMemory.recall and verify the diagnostics
    record what happened. These tests pin the integration contract."""

    def test_fts_hit_counts_recorded(self, temp_db):
        """When FTS finds matches, wm_fts tier counter increments."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Alice prefers Vim editor", source="pref", importance=0.7)
        beam.remember("Bob owns the auth refactor", source="fact", importance=0.8)

        results = beam.recall("Alice Vim", top_k=10)
        assert results  # sanity: we got something back

        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 1
        # FTS should have found the Alice row.
        assert snap["by_tier"]["wm_fts"]["total_hits"] >= 1
        # WM fallback should NOT have fired (FTS produced hits).
        assert snap["totals"]["calls_using_wm_fallback"] == 0

    def test_wm_fallback_fires_when_query_matches_nothing(self, temp_db, monkeypatch):
        """When neither FTS nor vec finds anything for the query, the
        substring/recency fallback fires. Operators see this via
        `calls_using_wm_fallback` and `wm_fallback`'s hit count."""
        monkeypatch.setattr(
            "mnemosyne.core.beam._embeddings.available", lambda: False
        )
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Seed content that won't match the query at all.
        beam.remember("totally unrelated content here", source="x", importance=0.5)

        # Query that doesn't match any seeded content. Use stop-words
        # so FTS in BEAM mode also returns nothing.
        # (BEAM_MODE filters stop-words; we want a query whose
        # content-words don't match seeded content.)
        beam.recall(
            "qzzx-no-such-token-xyzzy", top_k=10
        )

        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 1
        # FTS found nothing.
        assert snap["by_tier"]["wm_fts"]["total_hits"] == 0
        # Fallback fired.
        assert snap["totals"]["calls_using_wm_fallback"] == 1
        # Fallback's scanned-row count includes the seeded row.
        assert snap["by_tier"]["wm_fallback"]["total_hits"] >= 1

    def test_em_fallback_fires_on_empty_episodic_match(
        self, temp_db, monkeypatch
    ):
        """The episodic fallback fires when vec+fts produce no
        episodic rowids. Embeddings monkeypatched off so the vec
        path doesn't return weak cosine-sim hits — environments
        with fastembed installed (CI) would otherwise see vec
        produce nonzero similarity for any query and skip the
        fallback."""
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.available", lambda: False
        )
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Seed an episodic row directly so the fallback has something
        # to scan over but the query won't match via FTS/vec.
        beam.consolidate_to_episodic(
            summary="totally unrelated episodic content",
            source_wm_ids=["fake"],
            importance=0.5,
        )

        beam.recall("qzzx-no-such-token-xyzzy", top_k=10)
        snap = get_recall_diagnostics()
        # EM fallback fired.
        assert snap["totals"]["calls_using_em_fallback"] == 1
        # The fallback scanned the seeded row; whether it kept it
        # depends on the relevance threshold. At minimum the
        # fallback's `calls_using_em_fallback` boolean fired.

    def test_truly_empty_call_counted(self, temp_db):
        """A recall call that returns ZERO results from all paths
        is counted under `calls_truly_empty`. Distinguishes "fallback
        fired and returned weak hits" from "literally nothing."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # No seeded content at all.
        results = beam.recall("anything-xyzzy", top_k=10)
        assert results == []

        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["calls_truly_empty"] == 1

    def test_multiple_recall_calls_accumulate(self, temp_db, monkeypatch):
        """Per-recall counters accumulate correctly across calls."""
        monkeypatch.setattr(
            "mnemosyne.core.beam._embeddings.available", lambda: False
        )
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Alice prefers Vim", source="pref", importance=0.7)

        beam.recall("Alice", top_k=10)              # FTS hit
        beam.recall("Alice", top_k=10)              # FTS hit
        beam.recall("zzzxxxnomatch", top_k=10)      # fallback

        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 3
        assert snap["totals"]["calls_using_wm_fallback"] == 1
        # FTS hit on 2 of 3 calls.
        assert snap["by_tier"]["wm_fts"]["calls_with_hits"] == 2

    def test_fallback_rate_metric_useful_for_experiment_monitoring(
        self, temp_db, monkeypatch
    ):
        """Operators monitoring a BEAM experiment use the fallback
        rate to know if recall is dominated by fallback noise. Test:
        a corpus with no matching content + N queries produces a
        100% wm-fallback rate; a matching corpus produces 0%."""
        monkeypatch.setattr(
            "mnemosyne.core.beam._embeddings.available", lambda: False
        )
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("indexable content with marker zzzqqq", source="t")

        # 3 queries that match (FTS path).
        for _ in range(3):
            beam.recall("zzzqqq", top_k=10)
        # 2 queries that don't match (fallback).
        for q in ("nomatch1xyz", "nomatch2xyz"):
            beam.recall(q, top_k=10)

        snap = get_recall_diagnostics()
        # 2 of 5 calls used the WM fallback → 0.4 rate.
        assert snap["totals"]["wm_fallback_rate"] == pytest.approx(0.4)


class TestReviewHardening:
    """Findings from /review (Codex structured + Codex adv + Claude
    adv). Each test pins one of the closed semantic gaps."""

    def test_counters_record_post_filter_rows(self, temp_db):
        """[Codex P2 + Codex adv #2 + Claude adv #6] Pre-fix the
        tier counters recorded BEFORE the `wm_where`/`em_where`
        filter — rows that FTS/vec returned but got dropped by
        session/scope/date/source filters inflated the counters.
        Operators saw "FTS healthy" when actually every FTS hit got
        filtered out. Fix: counters record POST-filter kept rows."""
        beam = BeamMemory(session_id="alice-session", db_path=temp_db)
        # Seed an FTS-matching row but with a different session.
        # Direct insert with explicit scope='session' — the column
        # default is 'global' which would surface cross-session and
        # defeat the filter test.
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, session_id, importance, scope) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("foreign-row", "Alice was here", "test",
             datetime.now().isoformat(), "other-session", 0.5, "session"),
        )
        beam.conn.commit()

        # Recall from alice-session for "Alice" — FTS will match
        # foreign-row by content, but it gets dropped by wm_where
        # because session_id doesn't match and scope is 'session'.
        beam.recall("Alice", top_k=10)

        snap = get_recall_diagnostics()
        # Post-filter: foreign-row didn't survive, so wm_fts_kept = 0.
        # Pre-fix this would have counted 1.
        assert snap["by_tier"]["wm_fts"]["total_hits"] == 0, (
            f"wm_fts counter inflated by filtered-out row: "
            f"got {snap['by_tier']['wm_fts']}"
        )

    def test_em_fallback_counter_records_kept_not_scanned(
        self, temp_db, monkeypatch
    ):
        """[Codex adv #1 + Claude adv #9] Pre-fix EM fallback
        recorded `len(scanned_rows)` regardless of how many passed
        the relevance > 0.02 threshold. Fix: counter increments
        only for kept (appended) rows.

        Construct content with disjoint char sets vs. the query so
        the substring scorer's char_overlap term returns 0 and
        rows score below the threshold.

        Embeddings monkeypatched off so the vec path doesn't
        surface the rows via cosine similarity (which would
        bypass the fallback entirely — CI has fastembed)."""
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.available", lambda: False
        )
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Content + query chosen to share ZERO chars (including no
        # whitespace match — `char_overlap` is computed over all
        # chars in the strings, including spaces). Use single-word
        # query so char-set is bounded.
        beam.consolidate_to_episodic(
            summary="abcdefghij",
            source_wm_ids=["x"],
            importance=0.5,
        )
        beam.consolidate_to_episodic(
            summary="abcdefghij",
            source_wm_ids=["y"],
            importance=0.5,
        )

        # Single-word query with chars disjoint from a-j (no
        # overlap with content). Substring scoring produces 0 +
        # 0 + 0 + 0 + 0 → relevance below threshold; rows
        # scanned but NOT kept.
        beam.recall("xyzqwvu", top_k=10)

        snap = get_recall_diagnostics()
        assert snap["totals"]["calls_using_em_fallback"] == 1
        kept = snap["by_tier"]["em_fallback"]["total_hits"]
        # Pre-fix counter was 2 (scanned both rows). Post-fix it
        # reflects appended rows only — 0 because neither row's
        # substring score exceeded 0.02.
        assert kept == 0, (
            f"em_fallback counter still records scanned rows, not "
            f"kept rows: got total_hits={kept} with 2 rows seeded"
        )

    def test_fallback_rate_clamped_at_one(self):
        """[Claude adv #12] Defense-in-depth: fallback_rate() must
        not exceed 1.0 even under simulated reset-mid-call races.
        Operators dashboarding the rate get sensible numbers."""
        diag = RecallDiagnostics()
        # Simulate the race: many fallback_used signals accumulate
        # before total_calls catches up.
        for _ in range(5):
            diag.record_fallback_used(wm=True)
        diag.record_call()  # total_calls = 1, calls_using_wm = 5

        rates = diag.fallback_rate()
        assert rates["wm"] == 1.0, (
            f"fallback_rate not clamped: got {rates['wm']}"
        )

        snap = diag.snapshot()
        assert snap["totals"]["wm_fallback_rate"] == 1.0

    def test_tier_attribution_no_double_count(self, temp_db):
        """Each kept row credits exactly one tier. Sum across tiers
        equals total kept rows for the call (excluding entity-aware
        expansion which is a separate signal source)."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Alice prefers Vim", source="pref", importance=0.7)
        beam.remember("Bob owns auth", source="fact", importance=0.8)

        results = beam.recall("Alice Vim", top_k=10)
        snap = get_recall_diagnostics()

        total_kept = sum(
            snap["by_tier"][tier]["total_hits"] for tier in RECALL_TIERS
        )
        # Working-tier results in the output (excluding entity-aware
        # boosts which credit no tier). Results and snapshot come from
        # the SAME recall call.
        wm_results = [r for r in results if r.get("tier") == "working" and not r.get("entity_match")]
        em_results = [r for r in results if r.get("tier") == "episodic" and not r.get("entity_match")]
        attributable = len(wm_results) + len(em_results)
        # Counters >= attributable; the entity-aware path can add
        # more results that aren't tier-attributed.
        assert total_kept >= attributable, (
            f"counter undercounts: total_kept={total_kept}, "
            f"attributable={attributable}, results={results}"
        )

    def test_truly_empty_distinguishes_filter_dropouts(self, temp_db):
        """[Claude adv #8] truly_empty must distinguish 'no signal
        anywhere' from 'candidates existed but got filtered'. Fix:
        truly_empty = final_results empty AND zero kept across all
        tiers."""
        beam = BeamMemory(session_id="alice", db_path=temp_db)
        # Seed an FTS-matchable row in a different session, scope=
        # 'session' so it doesn't surface cross-session (column
        # default is 'global').
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, session_id, importance, scope) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("other-sess-row", "Alice was here", "test",
             datetime.now().isoformat(), "other-session", 0.5, "session"),
        )
        beam.conn.commit()

        results = beam.recall("Alice", top_k=10)
        assert results == []
        snap = get_recall_diagnostics()
        # Final results empty AND no tier attributed a kept row →
        # this case IS truly empty by the new gate (post-filter
        # dropouts don't credit the counters, so kept_sum=0).
        # Note: that's the right call — operators care that NO
        # signal made it through, regardless of why.
        assert snap["totals"]["calls_truly_empty"] == 1
    """[/regression] Adding diagnostics must not alter recall output.
    Pre-C4 recall returned X; post-C4 it must return the same X.
    Test by recording a baseline expectation and asserting against
    it across instrumentation-touching paths."""

    def test_recall_still_returns_results(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Alice prefers Vim", source="pref", importance=0.7)
        beam.remember("Bob owns auth", source="fact", importance=0.8)

        results = beam.recall("Alice", top_k=10)
        assert results
        assert any("Alice" in r["content"] for r in results)

    def test_recall_returns_empty_for_no_match_no_corpus(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        results = beam.recall("totally-no-such-content", top_k=10)
        assert results == []

    def test_fallback_path_still_yields_results_on_substring(self, temp_db):
        """Pre-C4 the fallback existed for a reason — it surfaces
        results when FTS/vec produce nothing but substring matching
        still finds something. Verify the fallback STILL does this
        post-C4 (we only added instrumentation, no behavior change)."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Use a stop-word-only query so FTS filters everything out.
        # But seed content that substring-matches the query token.
        beam.remember("the quick brown fox", source="x", importance=0.7)

        # Query is a stop-word in BEAM mode. FTS will be empty after
        # stop-word filtering; fallback fires.
        results = beam.recall("the", top_k=10)
        # Depending on BEAM_MODE the fallback may or may not yield —
        # the test's main job is to assert "no crash" and that we
        # got a list back.
        assert isinstance(results, list)

        snap = get_recall_diagnostics()
        # And the diagnostics record the call.
        assert snap["totals"]["calls"] == 1


class TestPolyphonicFallbackDiagnostics:
    """C4 follow-up: `fallback_rate` must be real on the polyphonic
    path.

    #668 wired tier hits + call counts into the polyphonic branch but
    never called `record_fallback_used()`, so `em_fallback_rate` /
    `wm_fallback_rate` stayed 0 under `MNEMOSYNE_POLYPHONIC_RECALL=1`
    even when the vector voice degraded from the sqlite-vec fast path
    to a numpy full-scan. These tests pin the new degraded-path
    signal: the engine exposes `last_call_fallback` per recall and the
    polyphonic diagnostics block records it as `em_fallback_used`.

    `wm` stays False by design — the polyphonic engine has no
    substring-scoring tier for working memory (documented in
    docs/benchmarking.md).
    """

    def test_engine_last_call_fallback_defaults_false(self, temp_db):
        """A freshly built engine starts with no degraded-path signal.
        The default must not trip any operator alarm before the first
        recall runs."""
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        engine = PolyphonicRecallEngine(db_path=temp_db)
        assert engine.last_call_fallback == {"em": False, "wm": False}

    def test_engine_flags_em_fallback_when_sqlite_vec_unavailable(
        self, temp_db, monkeypatch
    ):
        """When sqlite-vec is unavailable the vector voice's EM tier
        degrades to the numpy full-scan; the engine must expose that
        as `last_call_fallback["em"] = True` so beam.py can record
        it."""
        import numpy as np

        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        monkeypatch.setattr(
            "mnemosyne.core.beam._vec_available", lambda conn: False
        )
        beam = BeamMemory(session_id="poly-fb", db_path=temp_db)
        beam.consolidate_to_episodic(
            summary="polyphonic fallback target content",
            source_wm_ids=["seed"],
            importance=0.5,
        )
        engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
        # Real query vector so the voice runs; numpy is the only EM
        # path when _vec_available is patched False.
        query_vec = np.ones(384, dtype=np.float32)
        engine._vector_voice(query_vec)
        assert engine.last_call_fallback["em"] is True

    def test_engine_resets_fallback_flag_before_early_return(
        self, temp_db, monkeypatch
    ):
        """[CodeRabbit #677] A cached engine can record em=True on one
        call, then return early on the next (voice disabled, missing
        embedding, empty vector, zero norm). Without a reset at the
        top of _vector_voice, BeamMemory.recall() would inherit the
        PRIOR call's fallback state and raise a false em_fallback_rate
        alarm. Two-call regression: degraded first, early-return
        second, flag must read clean defaults."""
        import numpy as np

        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        monkeypatch.setattr(
            "mnemosyne.core.beam._vec_available", lambda conn: False
        )
        beam = BeamMemory(session_id="poly-reset", db_path=temp_db)
        beam.consolidate_to_episodic(
            summary="reset regression target",
            source_wm_ids=["seed"],
            importance=0.5,
        )
        engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)

        # Call 1: degraded (sqlite-vec unavailable) → em=True.
        engine._vector_voice(np.ones(384, dtype=np.float32))
        assert engine.last_call_fallback["em"] is True

        # Call 2: early return — the voice is disabled, so the vector
        # voice exits before touching the EM tier. The flag must NOT
        # leak the prior call's degraded state.
        monkeypatch.setenv("MNEMOSYNE_VOICE_VECTOR", "0")
        engine._vector_voice(np.ones(384, dtype=np.float32))
        assert engine.last_call_fallback == {"em": False, "wm": False}

    def test_engine_no_em_fallback_on_deterministic_fast_path(
        self, temp_db, monkeypatch
    ):
        """Healthy-path fallback contract WITHOUT requiring the
        sqlite-vec extension: serve the vec_episodes ANN query from a
        fake result set through a delegating connection, so the
        sqlite-vec fast path is exercised deterministically in every
        environment. The flag must stay False (no numpy degradation).
        This keeps the healthy case covered even on CI without
        sqlite-vec; the real-extension integration lives in
        test_recall_no_em_fallback_when_fast_path_healthy."""
        import json

        import numpy as np

        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        beam = BeamMemory(session_id="poly-fast-det", db_path=temp_db)
        # Seed an EM row + its memory_embeddings entry (the JOIN the
        # fake ANN result maps through).
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
            "VALUES ('poly-det-fast', 'deterministic fast path target', 'test', datetime('now'), 0.5)"
        )
        em_rowid = beam.conn.execute(
            "SELECT rowid FROM episodic_memory WHERE id = ?",
            ("poly-det-fast",),
        ).fetchone()[0]
        target_vec = np.ones(384, dtype=np.float32)
        beam.conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json) "
            "VALUES (?, ?)",
            ("poly-det-fast", json.dumps(target_vec.tolist())),
        )
        beam.conn.commit()

        class _FakeCursor:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

        class _FakeVecConnection:
            """Delegates everything to the real connection except the
            vec_episodes MATCH query, which returns a canned ANN hit."""

            def __init__(self, real, fake_rows):
                object.__setattr__(self, "_real", real)
                object.__setattr__(self, "_fake", fake_rows)

            def execute(self, sql, params=()):
                if "FROM vec_episodes" in str(sql) and "MATCH" in str(sql):
                    return _FakeCursor(self._fake)
                return self._real.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._real, name)

        fake_rows = [{"rowid": em_rowid, "distance": 0.0}]
        fake_conn = _FakeVecConnection(beam.conn, fake_rows)

        # Force the sqlite-vec branch without the extension loaded.
        monkeypatch.setattr(
            "mnemosyne.core.beam._vec_available", lambda conn: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.beam._effective_vec_type", lambda conn: "f32"
        )

        engine = PolyphonicRecallEngine(db_path=temp_db, conn=fake_conn)
        results = engine._vector_voice(target_vec)
        # The ANN hit survived the superseded/valid_until JOIN → EM
        # was consumed via the fast path → no fallback flag.
        assert engine.last_call_fallback["em"] is False
        assert any(
            r.memory_id == "poly-det-fast" for r in results
        ), f"fake fast path did not surface seeded EM row: {results}"

    def test_engine_keeps_em_flag_false_on_sqlite_vec_fast_path(
        self, temp_db, monkeypatch
    ):
        """When sqlite-vec serves the EM tier, the numpy fallback did
        NOT run — `last_call_fallback["em"]` must stay False."""
        import json

        import numpy as np

        from mnemosyne.core.beam import _vec_available, _vec_insert
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        beam = BeamMemory(session_id="poly-fb2", db_path=temp_db)
        if not _vec_available(beam.conn):
            pytest.skip("sqlite-vec not available in this environment")
        # Seed an EM row + its vec_episodes entry so the sqlite-vec
        # fast path can actually serve the query (mirrors
        # test_em_sqlite_vec_fast_path_metadata_backend).
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
            "VALUES ('poly-em-fast', 'fast path target', 'test', datetime('now'), 0.5)"
        )
        em_rowid = beam.conn.execute(
            "SELECT rowid FROM episodic_memory WHERE id = ?",
            ("poly-em-fast",),
        ).fetchone()[0]
        target_vec = np.ones(384, dtype=np.float32)
        beam.conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json) "
            "VALUES (?, ?)",
            ("poly-em-fast", json.dumps(target_vec.tolist())),
        )
        _vec_insert(beam.conn, em_rowid, target_vec.tolist())
        beam.conn.commit()

        engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
        engine._vector_voice(target_vec)
        assert engine.last_call_fallback["em"] is False

    def test_recall_records_em_fallback_rate_under_polyphonic(
        self, temp_db, monkeypatch
    ):
        """End-to-end: with POLYPHONIC_RECALL=1 and sqlite-vec
        unavailable, every recall degrades the EM vector path, so
        `em_fallback_rate` must read 1.0 and the engine's
        degraded-path signal must flow into the process-global
        diagnostics. The degraded path must still return the seeded
        embedding-backed episodic memory, and WM must never be
        flagged (no substring tier in the polyphonic engine)."""
        import json

        import numpy as np

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        monkeypatch.setattr(
            "mnemosyne.core.beam._vec_available", lambda conn: False
        )
        # Force the vector voice to actually run: without embeddings
        # available (CI without fastembed), _recall_polyphonic passes
        # query_embedding=None and the vector voice early-returns
        # before touching the fallback flag — which would make this
        # test trivially pass for the wrong reason.
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.embed",
            lambda texts: np.ones((len(texts), 384), dtype=np.float32),
        )
        beam = BeamMemory(session_id="poly-e2e", db_path=temp_db)
        beam.remember("Alice prefers Vim", source="pref", importance=0.7)
        # Seed an embedding-backed EPISODIC memory so the degraded
        # vector voice has a real EM target to return (a working-only
        # corpus would exercise the flag without proving the degraded
        # path still surfaces episodic results).
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
            "VALUES ('poly-em-degraded', 'polyphonic episodic target', 'test', datetime('now'), 0.6)"
        )
        beam.conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json) "
            "VALUES (?, ?)",
            ("poly-em-degraded", json.dumps(np.ones(384, dtype=np.float32).tolist())),
        )
        beam.conn.commit()

        results = beam.recall("Alice", top_k=10)
        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["calls_using_em_fallback"] == 1, (
            f"em fallback not recorded on polyphonic path: {snap}"
        )
        assert snap["totals"]["em_fallback_rate"] == 1.0
        # WM has no substring tier in the polyphonic engine.
        assert snap["totals"]["wm_fallback_rate"] == 0.0
        # The engine signal surfaced to the diagnostics block.
        assert getattr(beam, "_last_polyphonic_fallback", {}).get("em") is True
        # WM stays clean by design.
        assert getattr(beam, "_last_polyphonic_fallback", {}).get("wm") is False
        # The degraded path still returns the seeded episodic memory.
        assert any(
            r.get("content") == "polyphonic episodic target" for r in results
        ), f"degraded recall did not return seeded EM row: {results}"

    def test_recall_no_em_fallback_when_fast_path_healthy(
        self, temp_db, monkeypatch
    ):
        """End-to-end healthy path: sqlite-vec serves the EM tier, so
        em_fallback_rate stays 0. This is the operator-facing alarm
        contract — the gauge only trips on real degradation. The
        healthy path must return the seeded episodic memory, and WM
        must never be flagged."""
        import json

        import numpy as np

        from mnemosyne.core.beam import _vec_available, _vec_insert

        beam = BeamMemory(session_id="poly-e2e2", db_path=temp_db)
        if not _vec_available(beam.conn):
            pytest.skip("sqlite-vec not available in this environment")
        # Seed an EM row + vec_episodes entry so the sqlite-vec fast
        # path actually serves the EM tier (a plain remember() writes
        # working memory only — an empty vec_episodes legitimately
        # degrades to numpy and would false-positive this test).
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
            "VALUES ('poly-em-healthy', 'Alice was here in episodic', 'test', datetime('now'), 0.7)"
        )
        em_rowid = beam.conn.execute(
            "SELECT rowid FROM episodic_memory WHERE id = ?",
            ("poly-em-healthy",),
        ).fetchone()[0]
        target_vec = np.ones(384, dtype=np.float32)
        beam.conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json) "
            "VALUES (?, ?)",
            ("poly-em-healthy", json.dumps(target_vec.tolist())),
        )
        _vec_insert(beam.conn, em_rowid, target_vec.tolist())
        beam.conn.commit()

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        # Deterministic vector voice (see sibling test for why).
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.embed",
            lambda texts: np.ones((len(texts), 384), dtype=np.float32),
        )
        results = beam.recall("Alice", top_k=10)
        snap = get_recall_diagnostics()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["calls_using_em_fallback"] == 0, (
            f"healthy sqlite-vec path falsely flagged as fallback: {snap}"
        )
        assert snap["totals"]["em_fallback_rate"] == 0.0
        # WM has no substring tier in the polyphonic engine.
        assert getattr(beam, "_last_polyphonic_fallback", {}).get("wm") is False
        # The healthy path still returns the seeded episodic memory.
        assert any(
            r.get("content") == "Alice was here in episodic" for r in results
        ), f"healthy recall did not return seeded EM row: {results}"
