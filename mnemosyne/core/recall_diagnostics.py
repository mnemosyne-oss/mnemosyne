"""Recall path provenance diagnostics (C4).

Pre-C4: `BeamMemory.recall` had three silent fallback layers per
tier (working and episodic):

  1. `_fts_search_working` / `_fts_search` — wrapped in `try/except`
     for WM. Returns empty list on exception, indistinguishable from
     legitimate "no match."
  2. `_wm_vec_search` / `_vec_search` / `_in_memory_vec_search` —
     wrapped in `try/except: pass` for WM. Same shape.
  3. When both FTS and vec return nothing, the code falls through to
     "fetch recent items by timestamp DESC, score by substring."

Operators (and the BEAM experiment) saw recall results without any
provenance. Could be from FTS-ranked hits (good signal), vector
similarity hits (good signal), or pure substring scoring on recent
items (weak signal). When the experiment compares arm A vs arm B
recall quality, hits from the fallback layer add noise to the
comparison.

This module exposes a process-global counter set that records per-
recall-call signals: how many FTS hits, how many vec hits, whether
the fallback fired. Operators query via `get_recall_diagnostics()`
to see the fallback-usage rate over a measurement window.

Diagnostics are read-only signal — they never alter recall behavior.
The fallback still fires when needed (legitimate no-match case);
diagnostics expose WHEN and HOW OFTEN.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# Canonical recall tiers. Each tier represents a result-source path
# that recall() can return from. Operators look at per-tier counters
# to know what fraction of their recall traffic actually comes from
# the FTS/vec primary path vs. the substring/recency fallback.
RECALL_TIERS = ("wm_fts", "wm_vec", "wm_fallback", "em_fts", "em_vec", "em_fallback")


class RecallExplainTrace:
    """Per-query recall explanation recorder.

    This is intentionally opt-in and local to one recall() call. It is not
    connected to the process-global counters above, so explain=False keeps the
    existing hot path free of per-candidate allocations.
    """

    def __init__(self, *, query: str, top_k: int, engine: str, filters: Dict[str, Any], weights: Dict[str, Any]) -> None:
        self.query = query
        self.top_k = top_k
        self.engine = engine
        self.filters = {k: v for k, v in filters.items() if v is not None}
        self.weights = weights
        self.embedding = {"available": False, "computed": False}
        self.stages: List[Dict[str, Any]] = []
        self.candidates: List[Dict[str, Any]] = []
        self.truncated_count = 0

    def set_embedding(self, *, available: bool, computed: bool) -> None:
        self.embedding = {"available": bool(available), "computed": bool(computed)}

    def add_stage(self, name: str, *, raw_count: int, after_filter_count: int, kept_count: int, fallback_used: bool = False) -> None:
        self.stages.append({
            "name": name,
            "raw_count": int(raw_count),
            "after_filter_count": int(after_filter_count),
            "kept_count": int(kept_count),
            "fallback_used": bool(fallback_used),
        })

    @staticmethod
    def _source_path(result: Dict[str, Any]) -> str:
        tier = result.get("tier") or "unknown"
        if tier == "working":
            if result.get("fts_score", 0.0) > 0:
                return "wm_fts"
            if result.get("dense_score", 0.0) > 0:
                return "wm_vec"
            return "wm_fallback"
        if tier == "episodic":
            if result.get("fts_score", 0.0) > 0:
                return "em_fts"
            if result.get("dense_score", 0.0) > 0:
                return "em_vec"
            return "em_fallback"
        return str(tier)

    def add_ranked_candidates(self, ranked_results: List[Dict[str, Any]], final_results: List[Dict[str, Any]], *, preview_chars: int = 120, max_candidates: int = 50) -> None:
        final_ids = {r.get("id") for r in final_results}
        self.truncated_count = max(0, len(ranked_results) - len(final_results))
        for rank, result in enumerate(ranked_results[:max_candidates], start=1):
            kept = result.get("id") in final_ids
            content = str(result.get("content") or "")
            self.candidates.append({
                "id": result.get("id") if kept else None,
                "tier": result.get("tier"),
                "source_path": self._source_path(result),
                "rank": rank,
                "kept": kept,
                "drop_reason": None if kept else "top_k_truncated",
                "scores": {
                    "keyword": result.get("keyword_score", 0.0),
                    "fts": result.get("fts_score", 0.0),
                    "dense": result.get("dense_score", 0.0),
                    "importance": result.get("importance", 0.0),
                    "recency_decay": result.get("recency_decay", 0.0),
                    "final": result.get("score", 0.0),
                },
                "content_preview": content[:preview_chars] if kept else None,
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "weights": self.weights,
            "embedding": self.embedding,
            "stages": self.stages,
            "candidates": self.candidates,
            "truncation": {"top_k": self.top_k, "dropped_count": self.truncated_count},
        }


@dataclass
class _TierStats:
    """Per-tier counters.

    `calls_with_hits`: how many recall() invocations got at least
        one kept-and-attributed result from this path.
    `total_hits`: total kept result rows attributed to this path
        across all calls. Post-filter and post-relevance-threshold;
        a row counts here only if it ended up in `results` AND was
        sourced from this tier.

    Tier attribution semantics:
      - `wm_fts` / `em_fts`: rows that came from the FTS5 index hit
        set (overlap with vec credited here).
      - `wm_vec` / `em_vec`: rows uniquely contributed by vector
        search (i.e., in the vec result set but NOT in the FTS
        result set).
      - `wm_fallback` / `em_fallback`: rows from the substring-
        scoring fallback path (fires when FTS+vec both produced
        zero candidates).

    Sum over tiers per call = total kept rows that recall() emitted
    (excluding the entity-aware expansion path which is a separate
    signal source).
    """
    calls_with_hits: int = 0
    total_hits: int = 0


class RecallDiagnostics:
    """Process-global recall path counters.

    Designed for low-overhead instrumentation: each method takes
    sub-microsecond on the happy path. The single `_lock` gates all
    mutations.

    Diagnostics are read-only signal — they never alter recall
    behavior. The fallback still fires when needed; this class
    exists so operators can see how often.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tier_stats: Dict[str, _TierStats] = {
            tier: _TierStats() for tier in RECALL_TIERS
        }
        # Outer-call counters.
        self._total_calls = 0
        # Per-call breakdown of fallback usage. `total_calls` counts
        # every recall() invocation; `calls_using_*_fallback` counts
        # the subset where the fallback layer actually fired
        # (regardless of whether it returned hits).
        self._calls_using_wm_fallback = 0
        self._calls_using_em_fallback = 0
        # Calls where BOTH wm and em paths produced zero hits AND
        # fallback fired but still found nothing — the truly-empty case.
        self._calls_truly_empty = 0
        self._created_at = datetime.now().isoformat()

    @staticmethod
    def _validate_tier(tier: str) -> None:
        if tier not in RECALL_TIERS:
            raise ValueError(
                f"unknown recall tier {tier!r}; "
                f"valid tiers: {RECALL_TIERS}"
            )

    def record_tier_hits(self, tier: str, hit_count: int) -> None:
        """Record that `tier` returned `hit_count` results on this
        recall call. Operators can compute (per-tier hits) / total
        calls to see how often each path contributes."""
        self._validate_tier(tier)
        if hit_count < 0:
            raise ValueError(f"hit_count must be >= 0, got {hit_count}")
        with self._lock:
            stats = self._tier_stats[tier]
            if hit_count > 0:
                stats.calls_with_hits += 1
            stats.total_hits += hit_count

    def record_fallback_used(self, *, wm: bool = False, em: bool = False) -> None:
        """Record that the WM and/or EM fallback layer fired during
        a recall call. Each call should record once with the correct
        booleans; both can be True if both tiers fell back."""
        with self._lock:
            if wm:
                self._calls_using_wm_fallback += 1
            if em:
                self._calls_using_em_fallback += 1

    def record_call(self, *, truly_empty: bool = False) -> None:
        """Record an outer recall() invocation. `truly_empty=True`
        means the call returned zero results from every path
        including the fallback — the legitimate "nothing matched"
        outcome."""
        with self._lock:
            self._total_calls += 1
            if truly_empty:
                self._calls_truly_empty += 1

    def fallback_rate(self) -> Dict[str, float]:
        """Per-tier fraction of calls where the fallback layer was
        used. Operators alarm if these rise above an expected
        baseline (which depends on the corpus + query distribution).

        Returns:
            {"wm": 0.0-1.0, "em": 0.0-1.0}

        Both are 0.0 when no calls have run yet.
        """
        with self._lock:
            if self._total_calls == 0:
                return {"wm": 0.0, "em": 0.0}
            # Clamp at 1.0 to defend against a reset-mid-call race
            # where pre-reset record_fallback_used calls accumulate
            # against a post-reset _total_calls counter. The clamp
            # ensures dashboards never see impossible >1.0 rates.
            return {
                "wm": min(1.0, self._calls_using_wm_fallback / self._total_calls),
                "em": min(1.0, self._calls_using_em_fallback / self._total_calls),
            }

    def snapshot(self) -> Dict:
        """Return a JSON-serializable view of current state.

        Shape:
            {
              "created_at": iso-timestamp,
              "snapshot_at": iso-timestamp,
              "totals": {
                "calls": N,
                "calls_using_wm_fallback": N,
                "calls_using_em_fallback": N,
                "calls_truly_empty": N,
                "wm_fallback_rate": 0.0-1.0,
                "em_fallback_rate": 0.0-1.0,
              },
              "by_tier": {
                "wm_fts": {"calls_with_hits": N, "total_hits": N},
                "wm_vec": ...
                "wm_fallback": ...
                "em_fts": ...
                "em_vec": ...
                "em_fallback": ...
              },
            }

        Operators wanting "is the experiment running on real signal or
        fallback noise?" check `totals.wm_fallback_rate` and
        `totals.em_fallback_rate`. Above ~0.2 means most recall
        traffic is hitting the weak-signal fallback path — that
        would dominate arm-vs-arm comparisons in the experiment.
        """
        with self._lock:
            by_tier = {}
            for tier in RECALL_TIERS:
                stats = self._tier_stats[tier]
                by_tier[tier] = {
                    "calls_with_hits": stats.calls_with_hits,
                    "total_hits": stats.total_hits,
                }
            if self._total_calls == 0:
                wm_rate = 0.0
                em_rate = 0.0
            else:
                # Clamp matches fallback_rate() — see comment there.
                wm_rate = min(1.0, self._calls_using_wm_fallback / self._total_calls)
                em_rate = min(1.0, self._calls_using_em_fallback / self._total_calls)
            return {
                "created_at": self._created_at,
                "snapshot_at": datetime.now().isoformat(),
                "totals": {
                    "calls": self._total_calls,
                    "calls_using_wm_fallback": self._calls_using_wm_fallback,
                    "calls_using_em_fallback": self._calls_using_em_fallback,
                    "calls_truly_empty": self._calls_truly_empty,
                    "wm_fallback_rate": wm_rate,
                    "em_fallback_rate": em_rate,
                },
                "by_tier": by_tier,
            }

    def reset(self) -> None:
        """Reset all counters. Useful for tests and for operators
        starting a fresh measurement window."""
        with self._lock:
            self._tier_stats = {tier: _TierStats() for tier in RECALL_TIERS}
            self._total_calls = 0
            self._calls_using_wm_fallback = 0
            self._calls_using_em_fallback = 0
            self._calls_truly_empty = 0
            self._created_at = datetime.now().isoformat()


# Process-global singleton with thread-safe lazy init.
_singleton_lock = threading.Lock()
_singleton: Optional[RecallDiagnostics] = None


def get_diagnostics() -> RecallDiagnostics:
    """Return the process-global RecallDiagnostics instance."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = RecallDiagnostics()
    return _singleton


def get_recall_diagnostics() -> Dict:
    """Convenience: snapshot of current diagnostics. Use this when
    monitoring a BEAM experiment to verify that recall signal is
    coming from FTS/vec rather than the substring fallback."""
    return get_diagnostics().snapshot()


def reset_recall_diagnostics() -> None:
    """Convenience: reset the process-global counters. Useful for
    tests + operators starting a fresh measurement window before a
    benchmark run."""
    get_diagnostics().reset()
