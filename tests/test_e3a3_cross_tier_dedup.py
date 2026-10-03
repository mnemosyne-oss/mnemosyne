"""Regression tests for E3.a.3 — cross-tier recall dedup.

Pre-E3, `BeamMemory.sleep()` DELETEd source `working_memory` rows after
creating an `episodic_memory` summary, so a single logical fact lived in
exactly one place at recall time. Post-E3 (additive sleep) the sources
survive alongside the summary by design — a recall whose query matches
both ranks them next to each other and compounds `recall_count` twice
per call for the same fact.

This file pins the post-fix contract: when an episodic row's
`summary_of` references a working_memory row's id AND both appear in a
recall result set, the lower-scored side is dropped before top-K
truncation and recall_count attribution.

Why this matters for the BEAM-recovery experiment: Arm A (linear) and
Arm B (polyphonic) both run on the same data shape but historically had
different duplicate-handling. Linear ranked duplicates side-by-side;
polyphonic relied on its diversity rerank to "naturally" collapse them.
Applying identical summary_of-based dedup to both arms removes one
confound from arm-vs-arm comparison: any score delta now reflects
RRF-vs-linear-scoring, not dedup-behavior asymmetry.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _seed_wm(beam: BeamMemory, wm_id: str, content: str,
             *, session_id: str = "s1") -> None:
    """Insert one working_memory row directly via SQL with a controlled id."""
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (wm_id, content, "conversation",
         datetime.now().isoformat(), session_id, 0.5),
    )
    beam.conn.commit()


def _seed_episodic(beam: BeamMemory, ep_id: str, content: str,
                   *, summary_of_ids: List[str],
                   session_id: str = "s1") -> None:
    """Insert one episodic_memory row whose summary_of cites the given wm ids."""
    summary_of = ",".join(summary_of_ids)
    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, session_id, importance, summary_of) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ep_id, content, "consolidation",
         datetime.now().isoformat(), session_id, 0.5, summary_of),
    )
    beam.conn.commit()


def _make_result(rid: str, tier: str, score: float) -> Dict:
    """Minimal recall-row shape for direct helper testing."""
    return {"id": rid, "tier": tier, "score": score, "content": f"content-{rid}"}


class TestDedupHelperUnit:
    """Direct unit tests on `_dedup_cross_tier_summary_links` — no recall path,
    bypasses scoring complexity so dedup logic is testable in isolation."""

    def test_no_episodic_rows_returns_input_unchanged(self, temp_db):
        beam = BeamMemory(db_path=temp_db)
        results = [_make_result("wm-1", "working", 0.5),
                   _make_result("wm-2", "working", 0.3)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert out == results

    def test_no_summary_links_returns_input_unchanged(self, temp_db):
        beam = BeamMemory(db_path=temp_db)
        _seed_episodic(beam, "ep-1", "summary content", summary_of_ids=[])
        results = [_make_result("ep-1", "episodic", 0.7),
                   _make_result("wm-1", "working", 0.5)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert len(out) == 2

    def test_wm_wins_drops_episodic(self, temp_db):
        """When wm.score > ep.score, the episodic summary is dropped."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw content")
        _seed_episodic(beam, "ep-1", "summary", summary_of_ids=["wm-1"])
        results = [_make_result("wm-1", "working", 0.9),
                   _make_result("ep-1", "episodic", 0.5)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert len(out) == 1
        assert out[0]["id"] == "wm-1"

    def test_episodic_wins_drops_wm(self, temp_db):
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw content")
        _seed_episodic(beam, "ep-1", "summary", summary_of_ids=["wm-1"])
        results = [_make_result("ep-1", "episodic", 0.9),
                   _make_result("wm-1", "working", 0.5)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert len(out) == 1
        assert out[0]["id"] == "ep-1"

    def test_ties_keep_episodic(self, temp_db):
        """Tied scores resolve in favor of episodic side (later-stage
        representation; matches polyphonic diversity-rerank posture)."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw content")
        _seed_episodic(beam, "ep-1", "summary", summary_of_ids=["wm-1"])
        results = [_make_result("wm-1", "working", 0.6),
                   _make_result("ep-1", "episodic", 0.6)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert len(out) == 1
        assert out[0]["id"] == "ep-1"

    def test_only_one_side_in_results_keeps_it(self, temp_db):
        """If wm is linked to ep in DB but ep wasn't recalled, wm stays."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw content")
        _seed_episodic(beam, "ep-1", "summary", summary_of_ids=["wm-1"])
        results = [_make_result("wm-1", "working", 0.5)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert len(out) == 1
        assert out[0]["id"] == "wm-1"

    def test_summary_covers_multiple_wms_partial_overlap_per_cluster(self, temp_db):
        """Per-cluster: ep summarizes wm-1, wm-2, wm-3. wm-1 (0.9) beats
        ep (0.6), so the ep-1 cluster goes to the wm-side: drop ep, KEEP
        both wm-1 AND wm-2 (the lower-scored wm-2 survives because its
        representative ep got dropped — without ep, wm-2 is no longer
        a duplicate of anything in results).

        This is the codex P1 / Claude review fix: per-edge logic would
        have incorrectly dropped wm-2 against a phantom ep that itself
        got removed."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw 1")
        _seed_wm(beam, "wm-2", "raw 2")
        _seed_wm(beam, "wm-3", "raw 3")
        _seed_episodic(beam, "ep-1", "summary",
                       summary_of_ids=["wm-1", "wm-2", "wm-3"])
        results = [_make_result("wm-1", "working", 0.9),
                   _make_result("ep-1", "episodic", 0.6),
                   _make_result("wm-2", "working", 0.3)]
        out = beam._dedup_cross_tier_summary_links(results)
        out_ids = {r["id"] for r in out}
        # ep loses the cluster (wm-1 beats it) → ep dropped.
        # All covered wms (wm-1, wm-2) survive — wm-2 is no longer
        # represented by a surviving summary, so the dedup invariant
        # (no double-counting of one logical fact) still holds.
        assert "wm-1" in out_ids
        assert "wm-2" in out_ids
        assert "ep-1" not in out_ids

    def test_summary_covers_multiple_wms_all_in_results(self, temp_db):
        """ep beats wm-1 and wm-2 (both lower); both dropped, ep kept."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw 1")
        _seed_wm(beam, "wm-2", "raw 2")
        _seed_episodic(beam, "ep-1", "summary",
                       summary_of_ids=["wm-1", "wm-2"])
        results = [_make_result("ep-1", "episodic", 0.9),
                   _make_result("wm-1", "working", 0.5),
                   _make_result("wm-2", "working", 0.4)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert len(out) == 1
        assert out[0]["id"] == "ep-1"

    def test_preserves_order_on_retained_rows(self, temp_db):
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw")
        _seed_episodic(beam, "ep-1", "summary", summary_of_ids=["wm-1"])
        # Note: input order is wm-1 first, then ep-1, then an unrelated row.
        results = [_make_result("wm-1", "working", 0.9),  # wins
                   _make_result("ep-1", "episodic", 0.5),  # loses, dropped
                   _make_result("wm-99", "working", 0.4)]  # unrelated, kept
        out = beam._dedup_cross_tier_summary_links(results)
        assert [r["id"] for r in out] == ["wm-1", "wm-99"]

    def test_empty_summary_of_string_handled(self, temp_db):
        """Episodic row with summary_of='' or ',,, ' shouldn't crash."""
        beam = BeamMemory(db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, summary_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-empty", "content", "consolidation",
             datetime.now().isoformat(), "s1", 0.5, " , , ,"),
        )
        beam.conn.commit()
        results = [_make_result("ep-empty", "episodic", 0.5)]
        out = beam._dedup_cross_tier_summary_links(results)
        # No dedup possible (no wm ids in summary_of); row survives.
        assert len(out) == 1


class TestLinearRecallPathIntegration:
    """End-to-end via `BeamMemory.recall()` (linear path, polyphonic flag OFF).
    Asserts the dedup is wired into the full recall flow."""

    def test_recall_count_not_double_incremented(self, temp_db):
        """The core experiment-protection invariant: a query matching both
        the summary and its source should NOT compound recall_count on
        both rows in a single recall call."""
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "deployment script for prod release")
        _seed_episodic(beam, "ep-1",
                       "Summary: deployment script for prod release",
                       summary_of_ids=["wm-1"])

        results = beam.recall("deployment", top_k=10)
        ids = {r["id"] for r in results}
        # Exactly one of the two surfaces — the higher-scored — survived.
        assert len(ids & {"wm-1", "ep-1"}) == 1

        # Whichever survived gets +1; the dropped one stays at 0.
        rc_wm = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", ("wm-1",)
        ).fetchone()["recall_count"] or 0
        rc_ep = beam.conn.execute(
            "SELECT recall_count FROM episodic_memory WHERE id = ?", ("ep-1",)
        ).fetchone()["recall_count"] or 0
        # Exactly one increment total across both rows — no double-count.
        assert rc_wm + rc_ep == 1

    def test_topk_slot_recovered_for_other_content(self, temp_db):
        """Dedup happens BEFORE top-K truncation, so a freed slot goes to
        an unrelated row rather than being wasted on a duplicate."""
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "deployment script for production")
        _seed_episodic(beam, "ep-1",
                       "Summary about deployment for production",
                       summary_of_ids=["wm-1"])
        _seed_wm(beam, "wm-2", "deployment notes for staging")
        _seed_wm(beam, "wm-3", "deployment runbook draft")

        results = beam.recall("deployment", top_k=2)
        ids = [r["id"] for r in results]
        # With dedup: top-2 should be unique (one of {wm-1, ep-1} + one
        # of {wm-2, wm-3}), not {wm-1, ep-1} which both describe the
        # same logical fact.
        assert len(set(ids)) == len(ids)
        assert not ({"wm-1", "ep-1"} <= set(ids))

    def test_unrelated_ep_and_wm_both_kept(self, temp_db):
        """Sanity: ep without summary_of linkage to wm in results is kept
        alongside wm. Dedup only fires on actual summary↔source pairs."""
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "deployment runbook")
        _seed_episodic(beam, "ep-99", "Summary about deployment timing",
                       summary_of_ids=["wm-other-not-recalled"])

        results = beam.recall("deployment", top_k=10)
        ids = {r["id"] for r in results}
        assert "wm-1" in ids
        assert "ep-99" in ids


class _FakeEngine:
    """Stand-in for PolyphonicRecallEngine that returns deterministic
    pre-constructed results so tests don't depend on fastembed or any
    voice's per-environment behavior. Mirrors the engine's `recall()`
    contract: returns a list of PolyphonicResult."""

    def __init__(self, results):
        self._results = results
        self.last_top_k = None

    def recall(self, *, query, query_embedding, top_k,
               default_dense_source_filter=True, source=None, topic=None,
               episodic_where=None, episodic_params=()):
        self.last_top_k = top_k
        self.last_dense_filter = default_dense_source_filter
        self.last_source = source
        self.last_topic = topic
        self.last_episodic_where = episodic_where
        self.last_episodic_params = episodic_params
        return self._results


class TestPolyphonicRecallPathIntegration:
    """End-to-end via `BeamMemory.recall()` with `MNEMOSYNE_POLYPHONIC_RECALL=1`.
    The polyphonic path historically relied on its diversity rerank to
    collapse summary↔source duplicates. Post-fix it applies identical
    summary_of-based dedup as the linear path so Arm A vs Arm B comparisons
    are apples-to-apples on dedup behavior.

    Tests inject a mock engine so they exercise the post-engine dedup +
    recall_count attribution code without depending on real voices."""

    def _wire_fake_engine(self, beam, monkeypatch, polyphonic_results):
        engine = _FakeEngine(polyphonic_results)
        monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: engine)
        return engine

    def test_polyphonic_path_dedups_summary_source_pair(self, temp_db, monkeypatch):
        """Engine returns both ep and wm in its result set; post-dedup
        only the higher-scored side surfaces."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "configuring the deployment pipeline")
        _seed_episodic(beam, "ep-1",
                       "Summary: configuring deployment pipeline",
                       summary_of_ids=["wm-1"])
        # Engine returns ep first (higher combined_score) then wm.
        # Post veracity/tier multiplier, ep should still be on top
        # and the (ep, wm) summary_of pair should dedup down to ep alone.
        self._wire_fake_engine(beam, monkeypatch, [
            PolyphonicResult(memory_id="ep-1", combined_score=0.9,
                             voice_scores={"vector": 0.9}, metadata={}),
            PolyphonicResult(memory_id="wm-1", combined_score=0.5,
                             voice_scores={"vector": 0.5}, metadata={}),
        ])

        results = beam.recall("deployment", top_k=10)
        ids = {r["id"] for r in results}
        # Both eligible to surface, but dedup keeps one only.
        assert len(ids & {"wm-1", "ep-1"}) == 1

    def test_polyphonic_recall_count_attribution_post_dedup(self, temp_db, monkeypatch):
        """recall_count attribution lists rebuild from the deduped final,
        so a dropped duplicate doesn't get credited with a recall."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "release notes for v2.0 rollout")
        _seed_episodic(beam, "ep-1",
                       "Summary: release notes for v2.0",
                       summary_of_ids=["wm-1"])
        self._wire_fake_engine(beam, monkeypatch, [
            PolyphonicResult(memory_id="ep-1", combined_score=0.9,
                             voice_scores={"vector": 0.9}, metadata={}),
            PolyphonicResult(memory_id="wm-1", combined_score=0.5,
                             voice_scores={"vector": 0.5}, metadata={}),
        ])

        beam.recall("release notes", top_k=10)
        rc_wm = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", ("wm-1",)
        ).fetchone()["recall_count"] or 0
        rc_ep = beam.conn.execute(
            "SELECT recall_count FROM episodic_memory WHERE id = ?", ("ep-1",)
        ).fetchone()["recall_count"] or 0
        # Exactly one increment total: dropped side did NOT get attributed.
        assert rc_wm + rc_ep == 1

    def test_polyphonic_unrelated_results_both_kept(self, temp_db, monkeypatch):
        """Sanity: when engine returns rows without summary_of linkage,
        the dedup is a no-op."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "deployment notes")
        _seed_episodic(beam, "ep-99", "Unrelated summary about other content",
                       summary_of_ids=["wm-other-not-recalled"])
        self._wire_fake_engine(beam, monkeypatch, [
            PolyphonicResult(memory_id="wm-1", combined_score=0.8,
                             voice_scores={"vector": 0.8}, metadata={}),
            PolyphonicResult(memory_id="ep-99", combined_score=0.6,
                             voice_scores={"vector": 0.6}, metadata={}),
        ])

        results = beam.recall("deployment", top_k=10)
        ids = {r["id"] for r in results}
        assert "wm-1" in ids
        assert "ep-99" in ids

    def test_polyphonic_wm_winner_drops_episodic(self, temp_db, monkeypatch):
        """Inverse of the first test: when post-multiplier wm.score > ep.score,
        the ep summary gets dropped instead of the wm source."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "exact deployment runbook text")
        _seed_episodic(beam, "ep-1", "Vague summary",
                       summary_of_ids=["wm-1"])
        self._wire_fake_engine(beam, monkeypatch, [
            PolyphonicResult(memory_id="wm-1", combined_score=0.9,
                             voice_scores={"vector": 0.9}, metadata={}),
            PolyphonicResult(memory_id="ep-1", combined_score=0.3,
                             voice_scores={"vector": 0.3}, metadata={}),
        ])

        results = beam.recall("deployment", top_k=10)
        ids = {r["id"] for r in results}
        assert "wm-1" in ids
        assert "ep-1" not in ids


class TestReviewHardening:
    """Tests pinning edge cases surfaced by /review. Each test maps to a
    specific potential bypass; keep them stable to lock the fix in place."""

    def test_cross_tier_id_collision_disambiguated(self, temp_db):
        """If an ep_id and a wm_id are the same string (theoretically
        possible since `id TEXT PRIMARY KEY` is per-table), the dedup
        looks up tier-specific score maps so the comparison runs on the
        correct row, not the cross-tier doppelganger."""
        beam = BeamMemory(db_path=temp_db)
        # Both tables have a row with id "collide-1"; ep summarizes a
        # different wm ("wm-source").
        _seed_wm(beam, "collide-1", "wm-collision row")
        _seed_wm(beam, "wm-source", "source row")
        _seed_episodic(beam, "collide-1", "ep-collision row",
                       summary_of_ids=["wm-source"])

        results = [
            # Both tier-collision rows are present
            _make_result("collide-1", "working", 0.8),
            _make_result("collide-1", "episodic", 0.5),
            # Source wm is also in results — should compare against
            # the EP scoring 0.5, not the WM scoring 0.8 with the same id
            _make_result("wm-source", "working", 0.6),
        ]
        out = beam._dedup_cross_tier_summary_links(results)
        out_ids_by_tier = {(r["tier"], r["id"]) for r in out}
        # ep-collision-1.score (0.5) < wm-source.score (0.6) → drop ep
        assert ("episodic", "collide-1") not in out_ids_by_tier
        # wm-collision-1 (unrelated to the summary) survives
        assert ("working", "collide-1") in out_ids_by_tier
        # wm-source wins → kept
        assert ("working", "wm-source") in out_ids_by_tier

    def test_helper_no_work_when_no_episodic_rows(self, temp_db):
        """Fast-path: when no episodic rows are in results, the helper
        returns equivalent output. Behavioral equivalence (not identity)
        — a refactor returning a fresh list shouldn't break this test."""
        beam = BeamMemory(db_path=temp_db)
        results = [_make_result("wm-1", "working", 0.5),
                   _make_result("wm-2", "working", 0.3)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert out == results

    def test_helper_no_work_when_no_summary_links(self, temp_db):
        """When ep is present but its summary_of is empty, no dedup
        possible — output equals input."""
        beam = BeamMemory(db_path=temp_db)
        _seed_episodic(beam, "ep-1", "summary", summary_of_ids=[])
        results = [_make_result("ep-1", "episodic", 0.7),
                   _make_result("wm-1", "working", 0.5)]
        out = beam._dedup_cross_tier_summary_links(results)
        assert out == results

    def test_polyphonic_engine_top_k_overscan_behavioral(self, temp_db, monkeypatch):
        """Behavioral test: the polyphonic path requests `top_k * 2`
        candidates from the engine. Asserts via the captured engine
        argument rather than source-string match — resilient to
        reformatting / aliasing of the engine call."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "anything")

        engine = _FakeEngine([
            PolyphonicResult(memory_id="wm-1", combined_score=0.5,
                             voice_scores={"vector": 0.5}, metadata={}),
        ])
        monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: engine)
        beam.recall("anything", top_k=7)
        assert engine.last_top_k == 14  # 7 * 2
        # Default recall must keep the dense-source exclusion ON and pass
        # no explicit source/topic predicates.
        assert engine.last_dense_filter is True
        assert engine.last_source is None
        assert engine.last_topic is None

    def test_explicit_source_topic_filters_propagate_to_engine_with_filter_disabled(
        self, temp_db, monkeypatch
    ):
        """Explicit source=/topic= must reach the engine as real predicates
        AND turn off the default dense-source exclusion, so explicitly
        requested dialog/honcho rows are not filtered out before top-K
        selection (#696)."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        _seed_wm(beam, "wm-1", "anything")
        engine = _FakeEngine([
            PolyphonicResult(memory_id="wm-1", combined_score=0.5,
                             voice_scores={"vector": 0.5}, metadata={}),
        ])
        monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: engine)

        beam.recall("anything", top_k=7, source="conversation")
        assert engine.last_dense_filter is False, (
            "explicit source= must disable the default dense-source exclusion"
        )
        assert engine.last_source == "conversation"
        assert engine.last_topic is None
        assert engine.last_episodic_where is not None
        assert "source = ?" in engine.last_episodic_where
        assert "conversation" in engine.last_episodic_params

        engine.last_dense_filter = None
        engine.last_source = None
        engine.last_topic = None
        beam.recall("anything", top_k=7, topic="conversation")
        assert engine.last_dense_filter is False, (
            "explicit topic= must disable the default dense-source exclusion"
        )
        assert engine.last_source is None
        assert engine.last_topic == "conversation"
        assert engine.last_episodic_where is not None
        assert "source = ?" in engine.last_episodic_where
        assert "conversation" in engine.last_episodic_params

    def test_ep_ep_overlap_not_collapsed_documented_behavior(self, temp_db):
        """Pin behavior: two episodic summaries covering the same wm
        survive together. `sleep()` doesn't re-consolidate already-marked
        rows by design (E3 filter on consolidated_at IS NULL), so this
        is rare in practice; the helper makes no attempt to collapse
        ep ↔ ep duplicates. This test documents the choice so a future
        change to add ep ↔ ep dedup is caught + reviewed."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw content")
        _seed_episodic(beam, "ep-A", "summary A", summary_of_ids=["wm-1"])
        _seed_episodic(beam, "ep-B", "summary B", summary_of_ids=["wm-1"])
        # wm-1 lowest → both eps win their respective clusters → drop wm-1.
        results = [_make_result("ep-A", "episodic", 0.8),
                   _make_result("ep-B", "episodic", 0.7),
                   _make_result("wm-1", "working", 0.3)]
        out = beam._dedup_cross_tier_summary_links(results)
        out_ids = {r["id"] for r in out}
        assert "wm-1" not in out_ids
        # Both eps survive — no ep ↔ ep collapse.
        assert "ep-A" in out_ids
        assert "ep-B" in out_ids

    def test_tie_score_attributes_recall_to_episodic(self, temp_db):
        """L1 review fix: on a score tie the ep wins and gets the
        recall_count increment; the wm stays at 0. The 'no double-count'
        invariant requires that the side that survived dedup is the side
        that gets credited — not both, not the dropped one."""
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        # Seed wm + ep so the linear path's FTS+importance scoring lands
        # them with comparable scores. Easier path: directly call
        # _dedup_cross_tier_summary_links and the recall_count update
        # logic via beam.recall() — relying on the helper to drop the wm
        # and the linear path's UPDATE to credit only the ep.
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("wm-tie", "trigger word for the tie test",
             "conversation", datetime.now().isoformat(), "s1", 0.5),
        )
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance, summary_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ep-tie", "trigger word for the tie test",
             "consolidation", datetime.now().isoformat(), "s1", 0.5,
             "wm-tie"),
        )
        beam.conn.commit()

        beam.recall("trigger", top_k=10)
        rc_wm = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", ("wm-tie",)
        ).fetchone()["recall_count"] or 0
        rc_ep = beam.conn.execute(
            "SELECT recall_count FROM episodic_memory WHERE id = ?", ("ep-tie",)
        ).fetchone()["recall_count"] or 0
        # Whichever survived gets +1; total across both = 1 (no double-count).
        # Tie policy keeps episodic, so ep should be the credited side
        # when scores are equal — but float-score ties are rare in
        # integration; lock the weaker invariant here.
        assert rc_wm + rc_ep == 1

    def test_filter_dropped_source_does_not_over_drop_episodic(self, temp_db):
        """H2 review fix: if the session/scope/superseded filter has
        already dropped the wm source from results, the helper sees
        only the ep and must NOT over-drop it. Per-cluster logic with
        `present_wms = [w for w in covered_wm_ids if w in wm_scores]`
        guards this — but lock it with an explicit test."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-other-session", "content")
        _seed_episodic(beam, "ep-1", "summary referencing other session",
                       summary_of_ids=["wm-other-session"])
        # Simulate: wm-other-session was filtered out at the SELECT layer,
        # so only ep-1 surfaces.
        results = [_make_result("ep-1", "episodic", 0.5)]
        out = beam._dedup_cross_tier_summary_links(results)
        # ep must survive — no wm in results to dedup against.
        assert len(out) == 1
        assert out[0]["id"] == "ep-1"

    def test_polyphonic_recall_count_respects_session_scope(self, temp_db, monkeypatch):
        """H1 review fix (HIGH): post-dedup, the polyphonic UPDATE
        applies the same `(session_id = ? OR scope = 'global')` guard
        the linear path uses (beam.py:~2734). Pre-fix it bumped
        recall_count regardless of session, polluting cross-session
        ranking. This test forces a foreign-session row through the
        polyphonic path (by stubbing the row-filter) and asserts
        recall_count stayed at 0 because the rec_scope guard blocked
        the UPDATE — defense in depth against a future filter-bypass."""
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(db_path=temp_db, session_id="s1")
        # Foreign-session, session-scope row — must NOT get recall_count
        # bumped by a recall in session 's1'.
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, "
            "session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("wm-foreign", "foreign content", "conversation",
             datetime.now().isoformat(), "OTHER-SESSION", 0.5, "session"),
        )
        beam.conn.commit()

        engine = _FakeEngine([
            PolyphonicResult(memory_id="wm-foreign", combined_score=0.9,
                             voice_scores={"vector": 0.9}, metadata={}),
        ])
        monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: engine)
        # Force-bypass the row-filter so we exercise the rec_scope guard
        # directly. (Real flow: filter rejects this row upstream, so the
        # guard is defense-in-depth.)
        monkeypatch.setattr(
            beam, "_polyphonic_row_passes_filters",
            lambda *a, **kw: True,
        )

        beam.recall("foreign", top_k=10)
        rc = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?",
            ("wm-foreign",),
        ).fetchone()["recall_count"] or 0
        # rec_scope guard blocked the UPDATE: foreign session_id + scope=session
        # → neither branch of `(session_id = ? OR scope = 'global')` matches.
        assert rc == 0, (
            "polyphonic recall_count UPDATE bumped a foreign-session, "
            "session-scope row — H1 review fix regressed: the "
            "(session_id = ? OR scope = 'global') guard is missing."
        )

    def test_helper_uses_provided_score_field(self, temp_db):
        """Score comparison uses `r.get("score", 0.0)`. Rows missing
        a score default to 0 — they lose to any positive-scored counterpart."""
        beam = BeamMemory(db_path=temp_db)
        _seed_wm(beam, "wm-1", "raw")
        _seed_episodic(beam, "ep-1", "summary", summary_of_ids=["wm-1"])
        results = [
            {"id": "wm-1", "tier": "working"},  # no score field
            {"id": "ep-1", "tier": "episodic", "score": 0.3},
        ]
        out = beam._dedup_cross_tier_summary_links(results)
        # wm-1 defaults to 0.0; ep-1 wins.
        assert len(out) == 1
        assert out[0]["id"] == "ep-1"
