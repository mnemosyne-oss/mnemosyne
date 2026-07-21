"""Tests for derived-memory ranking defaults (upstream issue #506).

Two defects made sleep-derived rows outrank the source memories they
paraphrase:

1. ``consolidate_to_episodic()`` omitted the ``tier`` column from its INSERT,
   so consolidation summaries entered at the schema default tier 1 (full 1.0x
   recall weight) for TIER2_DAYS.
2. Model-refresh proposal rows used the LLM's self-reported confidence
   (routinely 0.85-0.95) directly as ranking importance, outranking curated
   content and defeating the injection gate's importance<0.65 drop condition.

Fixes under test: ``tier`` kwarg + ``MNEMOSYNE_CONSOLIDATION_TIER`` env
default (3) on consolidate_to_episodic, and
``MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP`` env cap (0.5) on proposal importance.

``TestRecallRankingEffect`` is the recall-side half: it drives the real
``recall()`` path so the tier default is pinned to observable ranking
behavior (which row survives ``_dedup_cross_tier_summary_links``), not just
to the stored column value.

Both env vars are read inside the call, not into module-level constants at
import — the tests set them with ``monkeypatch`` after import and after the
``BeamMemory`` instance exists, so they would fail if the values were
snapshotted at import time (the bug class in issue #482).
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core.beam import (
    BeamMemory,
    cap_proposal_importance,
    resolve_consolidation_tier,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _episodic_row(db_path, memory_id):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT tier, importance, source FROM episodic_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


class TestConsolidationTier:
    def test_resolver_reads_env_at_call_time(self, monkeypatch):
        """Direct unit coverage of the production resolver. Setting the env var
        AFTER import must take effect — if the default were snapshotted into a
        module-level constant (the #482 bug class) this would fail."""
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)
        assert resolve_consolidation_tier() == 3
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "1")
        assert resolve_consolidation_tier() == 1
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "not-a-number")
        assert resolve_consolidation_tier() == 3

    @pytest.mark.parametrize("given,expected", [(0, 1), (1, 1), (2, 2), (3, 3), (9, 3), (-5, 1)])
    def test_resolver_clamps_explicit_tier(self, monkeypatch, given, expected):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "2")  # must be ignored
        assert resolve_consolidation_tier(given) == expected

    def test_default_tier_is_3(self, temp_db, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="derived summary", source_wm_ids=["a", "b"],
            source="sleep_consolidation",
        )
        row = _episodic_row(temp_db, mid)
        assert row is not None
        assert row["tier"] == 3

    def test_explicit_tier_kwarg_wins(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "3")
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="explicit tier", source_wm_ids=["a"], tier=2,
        )
        assert _episodic_row(temp_db, mid)["tier"] == 2

    def test_env_var_overrides_default(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "1")
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="legacy behavior", source_wm_ids=["a"],
        )
        assert _episodic_row(temp_db, mid)["tier"] == 1

    def test_tier_clamped_to_valid_range(self, temp_db, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        low = beam.consolidate_to_episodic(
            summary="clamp low", source_wm_ids=["a"], tier=0,
        )
        high = beam.consolidate_to_episodic(
            summary="clamp high", source_wm_ids=["a"], tier=9,
        )
        assert _episodic_row(temp_db, low)["tier"] == 1
        assert _episodic_row(temp_db, high)["tier"] == 3

    def test_invalid_env_value_falls_back_to_3(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "not-a-number")
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="bad env", source_wm_ids=["a"],
        )
        assert _episodic_row(temp_db, mid)["tier"] == 3

    def test_default_tier_does_not_truncate_stored_content(self, temp_db, monkeypatch):
        """The tier default sets ranking weight, NOT stored-content compression.

        `degrade_episodic()` couples tier with content rewriting (LLM summary at
        1->2, key-signal extraction at 2->3), but its SELECTs only match rows
        currently at tier 1 or 2, so a row inserted straight at tier 3 keeps its
        full text. That is deliberate — the sleep() summary is already the
        compressed artifact — and this test pins it so the contract can't drift
        into silently truncating consolidation output.
        """
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)
        beam = BeamMemory(db_path=str(temp_db), session_id="t506c")

        long_summary = (
            "Consolidated summary of the competitive ladder sessions. " + "detail " * 200
        ).strip()
        assert len(long_summary) > 300  # exceeds TIER3_MAX_CHARS

        mid = beam.consolidate_to_episodic(
            summary=long_summary, source_wm_ids=["a"], source="sleep_consolidation",
        )

        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT tier, content FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()
        conn.close()

        assert row["tier"] == 3
        assert row["content"] == long_summary, (
            "tier-3-by-default must not truncate the stored summary"
        )

        # And degrade_episodic() leaves it alone: it only rewrites tier 1/2 rows.
        before = row["content"]
        beam.degrade_episodic(dry_run=False)
        conn = sqlite3.connect(str(temp_db))
        after = conn.execute(
            "SELECT content FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()[0]
        conn.close()
        assert after == before


class TestProposalImportanceCap:
    """Exercise the production helper directly, not a copy of its expression."""

    def test_high_confidence_is_capped(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", raising=False)
        assert cap_proposal_importance(0.95) == 0.5

    def test_low_confidence_passes_through(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", raising=False)
        assert cap_proposal_importance(0.3) == 0.3

    def test_cap_configurable(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", "0.8")
        assert cap_proposal_importance(0.95) == 0.8

    def test_explicit_cap_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", "0.8")
        assert cap_proposal_importance(0.95, cap=0.2) == 0.2

    def test_invalid_env_value_falls_back_to_default_cap(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", "not-a-number")
        assert cap_proposal_importance(0.95) == 0.5

    def test_missing_confidence_falls_back_to_default(self, monkeypatch):
        """The production `or 0.5` fallback exists for proposals with no
        confidence field; a None confidence must not become 0.0."""
        monkeypatch.delenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", raising=False)
        assert cap_proposal_importance(None) == 0.5
        assert cap_proposal_importance({}.get("confidence")) == 0.5

    def test_zero_confidence_is_preserved_not_defaulted(self, monkeypatch):
        """0.0 is a real (lowest) confidence, distinct from absent. The old
        `float(confidence or 0.5)` idiom conflated the two."""
        monkeypatch.delenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", raising=False)
        assert cap_proposal_importance(0.0) == 0.0

    @pytest.mark.parametrize("bad_cap", ["-1", "-0.5"])
    def test_negative_cap_is_clamped_to_zero(self, monkeypatch, bad_cap):
        """A misconfigured negative cap must not produce negative importance:
        recall scoring and the injection gate both assume [0, 1]."""
        monkeypatch.setenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", bad_cap)
        assert cap_proposal_importance(0.95) == 0.0

    def test_cap_above_one_is_clamped(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", "5.0")
        assert cap_proposal_importance(0.95) == 0.95

    def test_sleep_stores_capped_proposal_importance(self, temp_db, monkeypatch):
        """End-to-end: a sleep pass with a stubbed proposal generator stores
        proposal rows at capped importance while metadata keeps raw confidence."""
        monkeypatch.delenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", raising=False)
        monkeypatch.setenv("MNEMOSYNE_MODEL_REFRESH_AUTO_APPLY", "0")

        from mnemosyne.core import model_refresh

        def fake_proposals(items):
            return [{
                "category": "project",
                "name": "test_slot",
                "body": "test proposal body",
                "confidence": 0.95,
                "evidence_ids": [],
                "action": "update",
                "reason": "test",
            }]

        monkeypatch.setattr(model_refresh, "infer_model_update_proposals", fake_proposals)

        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        # Seed old rows so sleep() has something to consolidate.
        import sqlite3 as _sq
        from datetime import datetime, timedelta
        conn = _sq.connect(str(temp_db))
        ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [(f"t506-{i}", f"content {i}", "conversation", ts, "t506") for i in range(6)],
        )
        conn.commit()
        conn.close()

        beam.sleep()

        conn = _sq.connect(str(temp_db))
        conn.row_factory = _sq.Row
        rows = conn.execute(
            "SELECT importance, metadata_json FROM working_memory WHERE source = 'sleep_model_refresh_proposal'"
        ).fetchall()
        conn.close()
        assert rows, "sleep() stored no proposal rows"
        import json
        for r in rows:
            assert r["importance"] <= 0.5, f"proposal importance {r['importance']} exceeds cap"
            meta = json.loads(r["metadata_json"])
            assert float(meta.get("confidence", 0)) == 0.95, "raw confidence must survive in metadata"


# --- Recall-side assertions -------------------------------------------------
#
# The scenario below reproduces the production shape that motivated #506: a
# handful of long source rows (wiki chunks / session digests) each covering
# PART of the query's vocabulary, plus one LLM summary that paraphrases all of
# them and therefore mentions EVERY query term. The summary's per-row lexical
# relevance (1.0) beats each source's (0.375-0.625), so at tier 1 it wins its
# whole cluster in `_dedup_cross_tier_summary_links` and the sources are
# dropped from the result set entirely -- the "expected page pushed out of
# top-5" symptom. No source row reaches the 0.95 `exact_source_hit` escape
# hatch, which is exactly why that escape hatch did not save the corpus.

_RECALL_QUERY = "ladder promotion threshold masters rating decay"

_RECALL_SOURCES = [
    ("Competitive ladder configuration. The ladder promotion boundary for the "
     "masters bracket is 1800 points, applied uniformly across regions since the "
     "June rebalance. Players below that value remain in the diamond bracket and "
     "continue to accrue points from ranked matches only."),
    ("Masters bracket promotion requires 1800 rating and at least ten wins in the "
     "current season. The win requirement was added to stop rating farming through "
     "queue dodging. Season resets soft-reset rating toward the population mean."),
    ("Inactivity decay begins after fourteen days without a ranked match and "
     "removes twenty five points per week. Decay stops at the floor of the current "
     "bracket, and support can waive decay once per account per season."),
]

_RECALL_SUMMARY = (
    "Consolidated summary of the competitive ladder sessions: the group "
    "discussed the ladder promotion threshold for masters, the rating "
    "requirement, and inactivity decay behaviour."
)


def _seed_sources_and_summary(db_path, *, age_hours=1):
    """Seed aged source rows plus one derived summary; return (beam, summary_id).

    The tier is left to `consolidate_to_episodic`'s own resolution (env var /
    default) rather than passed as a kwarg, so these tests run unchanged against
    an unpatched build: on unpatched code the tier-1 test still passes (that IS
    the pre-fix behavior) while the default-tier test fails on the ranking
    assertion instead of on a missing keyword argument.
    """
    BeamMemory(db_path=str(db_path), session_id="t506r")  # create schema
    ts = (datetime.now() - timedelta(hours=age_hours)).isoformat()
    conn = sqlite3.connect(str(db_path))
    ids = []
    for i, content in enumerate(_RECALL_SOURCES):
        ids.append(f"src-{i}")
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id,"
            " importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"src-{i}", content, "wiki", ts, "t506r", 0.9, "global"),
        )
    conn.commit()
    conn.close()
    beam = BeamMemory(db_path=str(db_path), session_id="t506r")
    summary_id = beam.consolidate_to_episodic(
        summary=_RECALL_SUMMARY, source_wm_ids=ids, source="sleep_consolidation",
        importance=0.6, scope="global",
    )
    return beam, summary_id


class TestRecallRankingEffect:
    """Pin the tier default to observable recall() behavior, not just the column."""

    def test_tier1_summary_evicts_its_sources_from_recall(self, temp_db, monkeypatch):
        """Pre-fix behavior (tier 1): the summary wins its cluster and the source
        rows are dropped from the result set."""
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "1")
        beam, summary_id = _seed_sources_and_summary(temp_db)

        results = beam.recall(_RECALL_QUERY, top_k=10)
        ids = [r["id"] for r in results]

        assert summary_id in ids, "derived summary should be recalled at tier 1"
        assert results[0]["id"] == summary_id, (
            f"tier-1 summary should outrank its sources; got order {ids}"
        )
        assert not [r for r in results if r["tier"] == "working"], (
            f"tier-1 summary should have evicted all source rows; got {ids}"
        )

    def test_default_tier_keeps_sources_ahead_of_the_summary(self, temp_db, monkeypatch):
        """Post-fix default (tier 3): every source row survives and outranks the
        summary that paraphrases it."""
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)
        beam, summary_id = _seed_sources_and_summary(temp_db)

        results = beam.recall(_RECALL_QUERY, top_k=10)
        ids = [r["id"] for r in results]

        source_ids = [r["id"] for r in results if r["tier"] == "working"]
        assert len(source_ids) == len(_RECALL_SOURCES), (
            f"all source rows should survive at the default tier; got {ids}"
        )
        assert results[0]["id"] != summary_id, (
            f"derived summary must not top the ranking by default; got order {ids}"
        )
        if summary_id in ids:
            assert ids.index(summary_id) > max(ids.index(s) for s in source_ids), (
                f"summary should rank below every source it summarizes; got {ids}"
            )

    def test_tier_multiplier_is_the_mechanism(self, temp_db, monkeypatch):
        """With cross-tier dedup disabled, the same summary's score drops by the
        tier-3 weight ratio -- isolating tier as the lever, not embedding quality
        or result-set composition."""
        monkeypatch.setenv("MNEMOSYNE_CROSS_TIER_DEDUP", "0")

        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "1")
        beam1, sid1 = _seed_sources_and_summary(temp_db)
        hot = next(r for r in beam1.recall(_RECALL_QUERY, top_k=10) if r["id"] == sid1)

        with tempfile.TemporaryDirectory() as tmpdir2:
            monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "3")
            beam3, sid3 = _seed_sources_and_summary(Path(tmpdir2) / "t.db")
            cold = next(r for r in beam3.recall(_RECALL_QUERY, top_k=10) if r["id"] == sid3)

        assert hot["degradation_tier"] == 1
        assert cold["degradation_tier"] == 3
        assert cold["score"] < hot["score"], (
            f"tier-3 summary scored {cold['score']}, tier-1 scored {hot['score']}"
        )

    def test_recency_discount_can_resurface_a_demoted_summary(self, temp_db, monkeypatch):
        """The tier fix is necessary but not sufficient: `_recency_decay` on the
        aged source rows keeps discounting them while the fresh summary does not
        decay, so a demoted summary still overtakes very old sources.

        Documents the second mechanism raised on #506 rather than asserting a
        behavior this PR changes -- the crossover in this fixture sits between
        200h and 400h of source age.
        """
        monkeypatch.setenv("MNEMOSYNE_CROSS_TIER_DEDUP", "0")
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)

        def best_source_vs_summary(age_hours, path):
            beam, sid = _seed_sources_and_summary(path, age_hours=age_hours)
            results = beam.recall(_RECALL_QUERY, top_k=10)
            summary = next((r for r in results if r["id"] == sid), None)
            sources = [r for r in results if r["tier"] == "working"]
            return (max((r["score"] for r in sources), default=0.0),
                    summary["score"] if summary else 0.0)

        fresh_source, fresh_summary = best_source_vs_summary(200, temp_db)
        assert fresh_source > fresh_summary, (
            "at 200h the sources should still beat a tier-3 summary "
            f"({fresh_source} vs {fresh_summary})"
        )

        with tempfile.TemporaryDirectory() as tmpdir2:
            old_source, old_summary = best_source_vs_summary(
                800, Path(tmpdir2) / "t.db"
            )
        assert old_summary > old_source, (
            "documented residual: at 800h the recency discount lets even a "
            f"tier-3 summary win ({old_summary} vs {old_source})"
        )

