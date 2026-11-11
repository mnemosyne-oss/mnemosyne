"""Tests for cross-session global contradiction resolution.

The pass resolves factual contradictions among `scope='global'` memories across
different sessions so recall/prefetch stop co-presenting the stale and current
versions. Conflict *detection* needs embeddings, which are unavailable in CI
(`MNEMOSYNE_NO_EMBEDDINGS=1`), so the tests stub `BeamMemory._detect_conflicts`
to return a deterministic cross-session pair and assert the resolution (cross-
session `invalidate` + recall filtering) is real and non-vacuous.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def disable_llm(monkeypatch):
    monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
    # pin the deterministic linear recall path (recall needs embeddings for the
    # polyphonic path; the linear path works under MNEMOSYNE_NO_EMBEDDINGS).
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")


def _stub_detect_pair(*ids):
    return lambda items, similarity_threshold=0.88, min_gap_hours=1.0: [
        (ids[i], ids[i + 1]) for i in range(len(ids) - 1)
    ]


def test_global_contradiction_resolved_across_sessions(temp_db, monkeypatch, disable_llm):
    """Two contrary `global` facts in different sessions both surface from a
    third session; after resolution the older is superseded and recall returns
    only the newer."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    B = BeamMemory(session_id="tB", db_path=temp_db)
    C = BeamMemory(session_id="tC", db_path=temp_db)

    blue = A.remember("[USER] favorite color is blue", source="conversation",
                      importance=0.7, scope="global")
    green = B.remember("[USER] favorite color is green now", source="conversation",
                       importance=0.7, scope="global")

    before = {r["id"] for r in C.recall("favorite color", top_k=10)}
    assert blue in before and green in before, "sanity: both global rows surface cross-session"

    C._detect_conflicts = _stub_detect_pair(blue, green)
    res = C.resolve_cross_session_conflicts()
    assert res["status"] == "resolved"
    assert res["invalidated"] == 1 and res["conflicts_resolved"] == 1

    after = {r["id"] for r in C.recall("favorite color", top_k=10)}
    assert blue not in after and green in after, "loser hidden, winner remains"

    row = C.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (blue,)
    ).fetchone()
    assert row is not None and row[0] == green, "older row marked superseded by newer"


def test_disabled_by_default_noop(temp_db, disable_llm):
    """Without the feature flag the pass is a no-op (opt-in behavior)."""
    A = BeamMemory(session_id="tA", db_path=temp_db)
    b = A.remember("[USER] favorite color is blue", source="conversation",
                   importance=0.7, scope="global")
    A.remember("[USER] favorite color is green now", source="conversation",
               importance=0.7, scope="global")
    A._detect_conflicts = _stub_detect_pair(b, "replacement")
    res = A.resolve_cross_session_conflicts()
    assert res["status"] == "disabled"
    row = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (b,)
    ).fetchone()
    assert row[0] is None


def test_session_private_rows_not_scanned(temp_db, monkeypatch, disable_llm):
    """Session-private (non-global) rows are never considered or mutated."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    private = A.remember("[USER] a private thought", source="conversation",
                         importance=0.5, scope="session")
    A._detect_conflicts = lambda items, similarity_threshold=0.88, min_gap_hours=1.0: []
    res = A.resolve_cross_session_conflicts()
    assert res["rows_scanned"] == 0, "only global rows should be selected"
    row = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (private,)
    ).fetchone()
    assert row[0] is None


def test_llm_branch_confirms_and_declines(temp_db, monkeypatch, disable_llm):
    """With LLM conflict detection on, a confirmed pair is resolved and a
    declined pair is left untouched (the LLM gate is the context/ambiguity
    backstop)."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)

    A = BeamMemory(session_id="tA", db_path=temp_db)
    B = BeamMemory(session_id="tB", db_path=temp_db)
    C = BeamMemory(session_id="tC", db_path=temp_db)

    # confirmed
    blue = A.remember("[USER] favorite color is blue", source="conversation",
                      importance=0.7, scope="global")
    green = B.remember("[USER] favorite color is green now", source="conversation",
                       importance=0.7, scope="global")
    C._detect_conflicts = _stub_detect_pair(blue, green)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (True, 0.9, "green", "valid"))
    assert C.resolve_cross_session_conflicts()["invalidated"] == 1

    # declined (different context / complement): nothing superseded
    b2 = A.remember("[USER] favorite color is mauve", source="conversation",
                    importance=0.7, scope="global")
    g2 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")
    C2 = BeamMemory(session_id="tC2", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(b2, g2)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None, "valid"))
    res = C2.resolve_cross_session_conflicts()
    assert res["invalidated"] == 0 and res["conflicts_resolved"] == 0


def test_min_gap_hours_parametrizes_detection(temp_db, monkeypatch, disable_llm):
    """The cross-session pass forwards a relaxed min_gap_hours to detection."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    seen = {}
    A._detect_conflicts = lambda items, similarity_threshold=0.88, min_gap_hours=1.0: (
        seen.update(min_gap_hours=min_gap_hours, items=len(items)) or []
    )
    A.remember("[USER] favorite color is blue", source="conversation",
               importance=0.7, scope="global")
    A.remember("[USER] favorite color is green now", source="conversation",
               importance=0.7, scope="global")
    A.resolve_cross_session_conflicts()
    assert seen.get("min_gap_hours") == 0.0, "cross-session default gap is 0"


def test_sleep_retirement_guard_excludes_superseded(temp_db, monkeypatch, disable_llm):
    """A superseded-but-unconsolidated working row is not folded into a fresh
    episodic summary (retirement guard)."""
    A = BeamMemory(session_id="tA", db_path=temp_db)
    normal = A.remember("[USER] alpha normal memory", source="conversation", importance=0.6)
    stale = A.remember("[USER] beta stale memory", source="conversation", importance=0.6)
    replacement = A.remember("[USER] gamma correction memory", source="conversation", importance=0.6)
    A.invalidate(stale, replacement_id=replacement)  # sets superseded_by

    A.sleep(force=True)

    c_normal = A.conn.execute(
        "SELECT consolidated_at FROM working_memory WHERE id = ?", (normal,)
    ).fetchone()
    c_stale = A.conn.execute(
        "SELECT consolidated_at FROM working_memory WHERE id = ?", (stale,)
    ).fetchone()
    assert c_normal[0] is not None, "normal row should be consolidated"
    assert c_stale[0] is None, "superseded row must not be consolidated"


def test_dry_run_reports_and_does_not_mutate(temp_db, monkeypatch, disable_llm):
    """Dry run — the default reporting mode of the tool — lists candidates with
    zero mutation (and, being deterministic, no LLM calls)."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    B = BeamMemory(session_id="tB", db_path=temp_db)
    C = BeamMemory(session_id="tC", db_path=temp_db)
    blue = A.remember("[USER] favorite color is blue", source="conversation",
                      importance=0.7, scope="global")
    green = B.remember("[USER] favorite color is green now", source="conversation",
                       importance=0.7, scope="global")
    C._detect_conflicts = _stub_detect_pair(blue, green)
    res = C.resolve_cross_session_conflicts(dry_run=True)
    assert res["status"] == "dry_run"
    assert res["pairs_flagged"] >= 1
    assert res["invalidated"] == 0
    row = C.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (blue,)
    ).fetchone()
    assert row[0] is None, "dry run must not supersede"


def test_dry_run_llm_eval_switches_llm_without_mutating(temp_db, monkeypatch, disable_llm):
    """`llm_eval=True` makes a dry run run the LLM gate (previewing exactly what
    an apply would confirm/decline) while still not mutating. The default dry
    run stays deterministic (no LLM calls). This closes the dry-run/apply
    parity gap under LLM detection without changing the deterministic default."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)

    A = BeamMemory(session_id="tA", db_path=temp_db)
    B = BeamMemory(session_id="tB", db_path=temp_db)
    blue = A.remember("[USER] favorite color is blue", source="conversation",
                      importance=0.7, scope="global")
    green = B.remember("[USER] favorite color is green now", source="conversation",
                       importance=0.7, scope="global")
    C = BeamMemory(session_id="tC", db_path=temp_db)
    C._detect_conflicts = _stub_detect_pair(blue, green)

    # default dry run: deterministic, LLM never consulted even when the flag is on
    calls = []
    def _confirm(older, newer, session_id, db_path):
        calls.append((older, newer))
        return True, 0.9, "now", "valid"
    monkeypatch.setattr(lcd, "validate_conflict_pair", _confirm)
    res = C.resolve_cross_session_conflicts(dry_run=True)
    assert res["llm_validations"] == 0 and len(calls) == 0
    assert res["conflicts_resolved"] == 1

    # dry run + llm_eval: LLM consulted (confirms) but nothing mutates
    confirmed = C.resolve_cross_session_conflicts(dry_run=True, llm_eval=True)
    assert confirmed["llm_validations"] >= 1
    assert confirmed["conflicts_resolved"] == 1
    assert confirmed["invalidated"] == 0
    row = C.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (blue,)).fetchone()
    assert row[0] is None, "llm_eval dry run must not supersede"

    # dry run + llm_eval where the LLM declines: leaks the apply-time veto
    C2 = BeamMemory(session_id="tC2", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(blue, green)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None, "valid"))
    declined = C2.resolve_cross_session_conflicts(dry_run=True, llm_eval=True)
    assert declined["llm_validations"] >= 1
    assert declined["conflicts_resolved"] == 0
    assert declined["invalidated"] == 0


def test_source_grouping_excludes_different_sources(temp_db, monkeypatch, disable_llm):
    """Contradictory global rows carrying different `source` values land in
    separate groups, so no pair is flagged (the grouping contract the resolver
    relies on)."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    a1 = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    a2 = A.remember("[USER] favorite color is green now", source="notes",
                    importance=0.7, scope="global")
    A._detect_conflicts = _stub_detect_pair(a1, a2)
    res = A.resolve_cross_session_conflicts()
    assert res["pairs_flagged"] == 0
    assert res["invalidated"] == 0
    assert res["status"] == "no_op"


def test_episodic_global_rows_are_scanned(temp_db, monkeypatch, disable_llm):
    """`scope='global'` episodic rows are included in the candidate scan (the
    `_embedding_map` episodic branch has real coverage here)."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    wm = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    ep = A.consolidate_to_episodic(
        "[USER] favorite color is green now", source_wm_ids=[wm],
        source="conversation", importance=0.7, scope="global",
    )
    A._detect_conflicts = _stub_detect_pair(wm, ep)
    assert A.resolve_cross_session_conflicts(dry_run=True)["rows_scanned"] == 2
    assert A.resolve_cross_session_conflicts(dry_run=True)["invalidated"] == 0


def test_detect_conflicts_min_gap_threshold(temp_db, disable_llm):
    """Direct (non-stubbed) test of `_detect_conflicts` gap logic: a pair closer
    than `min_gap_hours` is rejected while the same content separated by more
    than `min_gap_hours` is flagged."""
    A = BeamMemory(session_id="tA", db_path=temp_db)

    def rows_with_gap(gap_hours):
        older_ts = (datetime.now() - timedelta(hours=gap_hours)).isoformat()
        return [
            {"id": "older", "content": "favorite color is blue for the car",
             "timestamp": older_ts, "superseded_by": None},
            {"id": "newer", "content": "my favorite color choice is green grass everywhere",
             "timestamp": datetime.now().isoformat(), "superseded_by": None},
        ]

    # Inject embeddings directly (CI has no embedder).
    A._embedding_map = lambda ids: {
        "older": np.array([0.0, 1.0], dtype=np.float32),
        "newer": np.array([0.0, 1.0], dtype=np.float32),
    }
    close = rows_with_gap(0.4)   # < 1h apart
    far = rows_with_gap(3.0)     # > 1h apart
    assert A._detect_conflicts(close, min_gap_hours=1.0) == []
    conflicts = A._detect_conflicts(far, min_gap_hours=1.0)
    assert len(conflicts) == 1 and conflicts[0] == ("older", "newer")


def test_dry_run_and_apply_chained_counts_match(temp_db, monkeypatch, disable_llm):
    """A dry run and an explicit apply report identical conflict counts for
    chained candidate pairs ((A,B),(B,C)) — the chain guard must apply in both
    modes, so a dry run truly reports what an apply will do."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    ids = [A.remember(f"[USER] favorite color is variant number {i}",
                      source="conversation", importance=0.7, scope="global")
           for i in range(3)]
    A._detect_conflicts = _stub_detect_pair(*ids)  # -> [(id0,id1),(id1,id2)]
    dry = A.resolve_cross_session_conflicts(dry_run=True)
    app = A.resolve_cross_session_conflicts(dry_run=False)
    assert dry["conflicts_resolved"] == app["conflicts_resolved"] == 1
    assert app["invalidated"] == 1
    # id1 became a replacement and must not itself be superseded (no orphans).
    r1 = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (ids[1],)
    ).fetchone()
    assert r1[0] is None, "replacement row must stay live"
    r0 = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (ids[0],)
    ).fetchone()
    assert r0[0] == ids[1]


def test_llm_validation_cap_bounds_calls(temp_db, monkeypatch, disable_llm):
    """With LLM confirmation on, an apply runs at most `max_llm_validations`
    LLM calls per invocation — independent of candidate count — so a provider
    tool never issues unbounded LLM round-trips while its shared lock is held."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)

    A = BeamMemory(session_id="tA", db_path=temp_db)
    ids = [A.remember(f"[USER] favorite color is variant number {i}",
                      source="conversation", importance=0.7, scope="global")
           for i in range(4)]
    # Two non-chained pairs: (id0,id1) and (id2,id3).
    A._detect_conflicts = (
        lambda items, similarity_threshold=0.88, min_gap_hours=1.0: [(ids[0], ids[1]), (ids[2], ids[3])]
    )
    calls = {"n": 0}
    monkeypatch.setattr(
        lcd, "validate_conflict_pair",
        lambda older, newer, session_id=None, db_path=None: (
            calls.__setitem__("n", calls["n"] + 1) or (True, 0.9, "new", "valid")
        ),
    )
    res = A.resolve_cross_session_conflicts(max_llm_validations=1)
    assert calls["n"] == 1, "LLM validation calls must be capped"
    assert res["llm_validations"] == 1
    assert res["invalidated"] == 1
    assert res["llm_cap_reached"] is True


def test_bank_scan_failure_returns_failed(temp_db, monkeypatch, disable_llm):
    """If a bank query fails, resolution is aborted with an explicit status and
    nothing is mutated (no silent single-bank resolution)."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    A.remember("[USER] favorite color is blue", source="conversation",
               importance=0.7, scope="global")
    A.conn.execute("DROP TABLE episodic_memory")
    A.conn.commit()
    A._detect_conflicts = _stub_detect_pair("x", "y")
    res = A.resolve_cross_session_conflicts()
    assert res["status"] == "failed"
    assert res["invalidated"] == 0 and res["conflicts_resolved"] == 0
    row = A.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()
    assert row[0] == 1, "nothing may be mutated on a failed scan"


def test_max_candidates_negative_normalized(temp_db, monkeypatch, disable_llm):
    """A negative `max_candidates` override is normalized to a positive bound
    (never an unbounded LIMIT) and truncation is reported."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    for i in range(3):
        A.remember(f"[USER] favorite color variant {i}", source="conversation",
                   importance=0.7, scope="global")
    A._detect_conflicts = lambda items, similarity_threshold=0.88, min_gap_hours=1.0: []
    res = A.resolve_cross_session_conflicts(max_candidates=-5)
    assert res["rows_scanned"] == 1, "negative cap must clamp to a positive bound"
    assert res["candidates_truncated"] is True


def test_max_candidates_exact_not_truncated(temp_db, monkeypatch, disable_llm):
    """An exactly-full bank (rows == max_candidates) is not flagged as truncated."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    A.remember("[USER] favorite color is green", source="conversation",
               importance=0.7, scope="global")
    A._detect_conflicts = lambda items, similarity_threshold=0.88, min_gap_hours=1.0: []
    res = A.resolve_cross_session_conflicts(max_candidates=1)
    assert res["rows_scanned"] == 1
    assert res["candidates_truncated"] is False


def _insert_episodic_dup(A, mem_id, content, source="conversation"):
    """Insert an episodic row that reuses a working-memory id (id collision
    across banks: `id` is only unique per bank)."""
    now = datetime.now().isoformat()
    A.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, session_id, importance, metadata_json,"
        " summary_of, valid_until, superseded_by, scope, recall_count,"
        " last_recalled, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mem_id, content, source, now, "tA", 0.7, "{}", None,
         None, None, "global", 0, None, now),
    )
    A.conn.commit()


def test_max_llm_validations_negative_normalized(temp_db, monkeypatch, disable_llm):
    """A negative `max_llm_validations` override is clamped to >= 1 so it does
    not trigger the cap immediately and still permits LLM-confirmed resolution."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)

    A = BeamMemory(session_id="tA", db_path=temp_db)
    ids = [A.remember(f"[USER] favorite color variant {i}", source="conversation",
                      importance=0.7, scope="global") for i in range(2)]
    A._detect_conflicts = (
        lambda items, similarity_threshold=0.88, min_gap_hours=1.0: [(ids[0], ids[1])]
    )
    calls = {"n": 0}
    monkeypatch.setattr(
        lcd, "validate_conflict_pair",
        lambda older, newer, session_id=None, db_path=None: (
            calls.__setitem__("n", calls["n"] + 1) or (True, 0.9, "new", "valid")
        ),
    )
    res = A.resolve_cross_session_conflicts(max_llm_validations=-5)
    assert calls["n"] == 1, "negative cap must clamp to at least 1 validation"
    assert res["invalidated"] == 1
    assert res["llm_cap_reached"] is False


def test_invalidate_respects_bank(temp_db, disable_llm):
    """`invalidate(..., bank=...)` supersedes only the named bank's row even when
    an identical id exists in the other bank."""
    A = BeamMemory(session_id="tA", db_path=temp_db)
    stale = A.remember("[USER] favorite color is blue", source="conversation",
                       importance=0.7, scope="global")
    repl = A.remember("[USER] favorite color is green now", source="conversation",
                      importance=0.7, scope="global")
    _insert_episodic_dup(A, stale, "[USER] favorite color is blue")

    assert A.invalidate(stale, replacement_id=repl, bank="episodic_memory") is True
    w = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (stale,)
    ).fetchone()
    e = A.conn.execute(
        "SELECT superseded_by FROM episodic_memory WHERE id = ?", (stale,)
    ).fetchone()
    assert w[0] is None, "bank binding must not supersede the working copy"
    assert e[0] == repl, "the episodic copy must be superseded"


def test_duplicate_id_across_banks_not_cross_resolved(temp_db, monkeypatch, disable_llm):
    """A duplicate id across banks makes the detector's pair ambiguous: the
    resolver must not cross-supersede the other bank's row."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    blue = A.remember("[USER] favorite color is blue", source="conversation",
                      importance=0.7, scope="global")
    green = A.remember("[USER] favorite color is green now", source="conversation",
                       importance=0.7, scope="global")
    _insert_episodic_dup(A, blue, "[USER] favorite color is blue")
    A._detect_conflicts = _stub_detect_pair(blue, green)

    res = A.resolve_cross_session_conflicts()
    assert res["status"] == "no_op"
    assert res["invalidated"] == 0
    w = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (blue,)
    ).fetchone()
    e = A.conn.execute(
        "SELECT superseded_by FROM episodic_memory WHERE id = ?", (blue,)
    ).fetchone()
    assert w[0] is None and e[0] is None, "ambiguous duplicate id must not cross-supersede"


def test_duplicate_id_across_sources_skipped(temp_db, monkeypatch, disable_llm):
    """Ambiguity is computed across ALL scanned rows (before grouping): a
    duplicate id in another bank with a *different* source still makes every
    pair touching that id unactionable."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tA", db_path=temp_db)
    blue = A.remember("[USER] favorite color is blue", source="conversation",
                      importance=0.7, scope="global")
    green = A.remember("[USER] favorite color is green now", source="conversation",
                       importance=0.7, scope="global")
    # Episodic duplicate of blue with a different source (separate group, but
    # 'blue' must still be treated as ambiguous globally).
    _insert_episodic_dup(A, blue, "[USER] favorite color is blue", source="import")
    A._detect_conflicts = _stub_detect_pair(blue, green)

    res = A.resolve_cross_session_conflicts()
    assert res["invalidated"] == 0 and res["conflicts_resolved"] == 0
    w = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (blue,)
    ).fetchone()
    e = A.conn.execute(
        "SELECT superseded_by FROM episodic_memory WHERE id = ?", (blue,)
    ).fetchone()
    assert w[0] is None and e[0] is None, "cross-source duplicate id must not resolve"


def test_invalidate_preserves_existing_successor(temp_db, disable_llm):
    """Supersession is atomic on an active target: once a row already has a
    successor (e.g. a concurrent resolver won the race), a later invalidate()
    must not overwrite it, and returns False."""
    A = BeamMemory(session_id="tA", db_path=temp_db)
    old = A.remember("[USER] favorite color is blue", source="conversation",
                     importance=0.7, scope="global")
    new = A.remember("[USER] favorite color is green now", source="conversation",
                     importance=0.7, scope="global")
    other = A.remember("[USER] favorite color is magenta", source="conversation",
                       importance=0.7, scope="global")
    assert A.invalidate(old, replacement_id=other) is True
    # A later attempt must not overwrite the existing successor.
    assert A.invalidate(old, replacement_id=new) is False
    row = A.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (old,)
    ).fetchone()
    assert row[0] == other, "existing successor must be preserved"


def test_resolver_scan_judges_offset_expiry_chronologically(temp_db, monkeypatch, disable_llm):
    """Review on #1113 (fifth round): the resolver's candidate scan must use
    the same julianday eligibility as every other surface. A legacy row
    storing an offset-bearing expiry that is already past (18:30+07:00 =
    11:30Z at a 12:00Z-class now) sorts lexically AFTER a UTC now string,
    so the old text comparison scanned the dead row as a live candidate and
    could supersede it; a still-future offset value that sorts lexically
    BEFORE now must stay scannable.
    """
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    A = BeamMemory(session_id="tR", db_path=temp_db)
    dead = A.remember("[USER] stale global fact", source="conversation",
                      importance=0.7, scope="global")
    alive = A.remember("[USER] live global fact", source="conversation",
                       importance=0.7, scope="global")
    # Control row: keeps the source group at >=2 members once the dead row is
    # filtered, so _detect_conflicts actually runs and the capture below is
    # non-vacuous (a group of one is skipped before the detector is called).
    control = A.remember("[USER] control global fact", source="conversation",
                         importance=0.7, scope="global")
    now = datetime.now(timezone.utc)
    dead_stored = (now - timedelta(minutes=30)).astimezone(
        timezone(timedelta(hours=7))).isoformat()
    alive_stored = (now + timedelta(hours=1)).astimezone(
        timezone(timedelta(hours=-5))).isoformat()
    # Fixture sanity: both values mislead a lexical comparison in opposite
    # directions, so a text-compare scan gets BOTH verdicts wrong.
    assert dead_stored > now.isoformat()
    assert alive_stored < now.isoformat()
    A.conn.execute("UPDATE working_memory SET valid_until = ? WHERE id = ?",
                   (dead_stored, dead))
    A.conn.execute("UPDATE working_memory SET valid_until = ? WHERE id = ?",
                   (alive_stored, alive))
    A.conn.commit()

    scanned_ids = set()

    def _capture(items, similarity_threshold=0.88, min_gap_hours=1.0):
        scanned_ids.update(i["id"] for i in items)
        return []

    A._detect_conflicts = _capture
    res = A.resolve_cross_session_conflicts(dry_run=True)
    assert res["status"] == "dry_run"
    assert res["rows_scanned"] == 2, (
        "only the two live rows may enter the candidate scan")
    assert scanned_ids, "capture must have run (control keeps the group non-vacuous)"
    assert dead not in scanned_ids, (
        "an expired row must never enter the candidate scan")
    assert alive in scanned_ids, (
        "a chronologically future expiry must stay scannable")
    assert control in scanned_ids


def test_resolver_scan_uses_utc_now_under_non_utc_host(temp_db, monkeypatch, disable_llm):
    """The scan's ``now`` operand must be aware UTC on any host. Under a
    far-west host clock a row that expired 30 minutes ago in UTC could sit
    inside the host's naive-local offset window and scan as active; the
    julianday comparison against aware-UTC now excludes it everywhere.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() unavailable on this platform")
    original = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    try:
        time.tzset()
        # The fixture is only meaningful if the host clock really moved west
        # of UTC — on a UTC-parked environment (e.g. missing tzdata, where
        # tzset() silently keeps UTC) the old naive-local comparison would
        # pass untested, so skip rather than claim a green (review on #1142).
        host_offset = datetime.now().astimezone().utcoffset()
        if host_offset is None or host_offset >= timedelta(0):
            pytest.skip(
                f"TZ=America/Los_Angeles did not take effect (offset {host_offset}); "
                "this test needs a west-of-UTC host clock to be non-vacuous")
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
        A = BeamMemory(session_id="tW", db_path=temp_db)
        expired = A.remember("[USER] recently expired global", source="conversation",
                             importance=0.7, scope="global")
        # Two live control rows: the source group stays at >=2 members even
        # after correct filtering, so the detector runs and the capture
        # below observes real candidate sets either way.
        ctl1 = A.remember("[USER] still-live global one", source="conversation",
                          importance=0.7, scope="global")
        ctl2 = A.remember("[USER] still-live global two", source="conversation",
                          importance=0.7, scope="global")
        past_naive = (datetime.now(timezone.utc) - timedelta(minutes=30)
                      ).replace(tzinfo=None).isoformat()
        A.conn.execute("UPDATE working_memory SET valid_until = ? WHERE id = ?",
                       (past_naive, expired))
        A.conn.commit()

        scanned_ids = set()

        def _capture(items, similarity_threshold=0.88, min_gap_hours=1.0):
            scanned_ids.update(i["id"] for i in items)
            return []

        A._detect_conflicts = _capture
        res = A.resolve_cross_session_conflicts(dry_run=True)
        assert res["status"] == "dry_run"
        # Pre-fix, the host-local naive now (~7h behind UTC under
        # America/Los_Angeles) read the UTC-expired row as live: rows_scanned
        # was 3 and the expired id reached the detector's items. The UTC now
        # must scan only the two live controls.
        assert res["rows_scanned"] == 2, (
            "only the live control rows may enter the candidate scan")
        assert scanned_ids == {ctl1, ctl2}, (
            "a UTC-expired row must not scan as a live candidate on a "
            "west-of-UTC host")
    finally:
        if original is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original)
        time.tzset()


def test_reject_cached_then_skipped(temp_db, monkeypatch):
    """#714 follow-up: an LLM-rejected pair is cached with a TTL and later
    runs skip re-validating it (the perpetual LLM re-validation churn)."""
    A = BeamMemory(session_id="tCRA", db_path=temp_db)
    B = BeamMemory(session_id="tCRB", db_path=temp_db)
    a1 = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    b1 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")

    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None, "valid"))

    # run 1: LLM rejects -> nothing invalidated, pair persisted to the cache.
    C1 = BeamMemory(session_id="tCRA", db_path=temp_db)
    C1._detect_conflicts = _stub_detect_pair(a1, b1)
    r1 = C1.resolve_cross_session_conflicts()
    assert r1["llm_validations"] == 1
    assert r1["skipped_cached"] == 0
    assert r1["invalidated"] == 0

    # run 2: the same rejected pair is now served from the cache (no LLM call).
    C2 = BeamMemory(session_id="tCRA", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(a1, b1)
    r2 = C2.resolve_cross_session_conflicts()
    assert r2["llm_validations"] == 0
    assert r2["skipped_cached"] == 1
    assert r2["invalidated"] == 0

    # the reject was persisted with a TTL.
    rows = _cache_rows(temp_db)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "reject"
    assert rows[0]["expires_at"]


def test_dry_run_reports_cache_without_mutating(temp_db, monkeypatch):
    """#714 follow-up: a dry run reads the cache (reports the skip) but never
    writes it — dry runs stay read-only."""
    A = BeamMemory(session_id="tCRD", db_path=temp_db)
    B = BeamMemory(session_id="tCRD2", db_path=temp_db)
    a1 = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    b1 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None, "valid"))

    # seed the cache with an apply.
    C1 = BeamMemory(session_id="tCRD", db_path=temp_db)
    C1._detect_conflicts = _stub_detect_pair(a1, b1)
    C1.resolve_cross_session_conflicts()
    assert len(_cache_rows(temp_db)) == 1

    # a dry run reports the skip but adds no row.
    C2 = BeamMemory(session_id="tCRD", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(a1, b1)
    r = C2.resolve_cross_session_conflicts(dry_run=True)
    assert r["skipped_cached"] == 1
    assert len(_cache_rows(temp_db)) == 1


def test_expired_verdict_is_readjudicated(temp_db, monkeypatch):
    """#714 follow-up: once the TTL elapses the pair is re-validated."""
    A = BeamMemory(session_id="tCRE", db_path=temp_db)
    B = BeamMemory(session_id="tCRE2", db_path=temp_db)
    a1 = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    b1 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None, "valid"))

    C1 = BeamMemory(session_id="tCRE", db_path=temp_db)
    C1._detect_conflicts = _stub_detect_pair(a1, b1)
    C1.resolve_cross_session_conflicts()
    assert len(_cache_rows(temp_db)) == 1

    # expire the verdict in place.
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE cross_session_verdict_cache SET expires_at = '2000-01-01'")
    conn.commit()
    conn.close()

    C2 = BeamMemory(session_id="tCRE", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(a1, b1)
    r2 = C2.resolve_cross_session_conflicts()
    assert r2["skipped_cached"] == 0
    assert r2["llm_validations"] == 1  # re-adjudicated after expiry


def _cache_rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM cross_session_verdict_cache")]
    finally:
        conn.close()
    return rows


def test_failed_validation_not_cached(temp_db, monkeypatch):
    """A FAILED validation (provider unavailable / malformed response) is a
    distinct status from a valid negative verdict and must NOT be cached:
    the next apply retries the validation instead of trusting a 30-day
    reject for an attempt that never reached a verdict."""
    A = BeamMemory(session_id="tCRF", db_path=temp_db)
    B = BeamMemory(session_id="tCRF2", db_path=temp_db)
    a1 = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    b1 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)

    states = iter(["failed", "valid"])

    def _stateful(older, newer, session_id=None, db_path=None):
        return (False, 0.2, None, next(states))

    monkeypatch.setattr(lcd, "validate_conflict_pair", _stateful)

    # run 1: the validation FAILS -> nothing is cached.
    C1 = BeamMemory(session_id="tCRF", db_path=temp_db)
    C1._detect_conflicts = _stub_detect_pair(a1, b1)
    r1 = C1.resolve_cross_session_conflicts()
    assert r1["llm_validations"] == 1
    assert r1["invalidated"] == 0

    # the table must not even exist: a failed validation writes nothing.
    conn = sqlite3.connect(str(temp_db))
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='cross_session_verdict_cache'").fetchone()
    finally:
        conn.close()
    assert exists is None, "a failed validation must not create the cache"

    # run 2: the pair is re-validated (not skipped from a cached reject);
    # the now-valid verdict IS persisted for later runs.
    C2 = BeamMemory(session_id="tCRF", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(a1, b1)
    r2 = C2.resolve_cross_session_conflicts()
    assert r2["llm_validations"] == 1, "the failed pair must be retried"
    assert r2["skipped_cached"] == 0
    assert len(_cache_rows(temp_db)) == 1


def test_content_edit_invalidates_cached_verdict(temp_db, monkeypatch):
    """A cached verdict is bound to the content snapshots of both memories:
    updating either memory's content (update_working keeps the id) makes the
    verdict stale — the pair is re-adjudicated and the stale row replaced,
    even though the pair ids and TTL are unchanged."""
    A = BeamMemory(session_id="tCRG", db_path=temp_db)
    B = BeamMemory(session_id="tCRG2", db_path=temp_db)
    a1 = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    b1 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None, "valid"))

    C1 = BeamMemory(session_id="tCRG", db_path=temp_db)
    C1._detect_conflicts = _stub_detect_pair(a1, b1)
    r1 = C1.resolve_cross_session_conflicts()
    assert r1["llm_validations"] == 1 and r1["skipped_cached"] == 0
    assert len(_cache_rows(temp_db)) == 1

    # edit one side of the pair (id retained, content changed).
    assert A.update_working(a1, content="[USER] favorite color is blue AND red")

    # the stale verdict is re-adjudicated (not reused) and its row replaced.
    C2 = BeamMemory(session_id="tCRG", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(a1, b1)
    r2 = C2.resolve_cross_session_conflicts()
    assert r2["llm_validations"] == 1, "a stale verdict must be re-adjudicated"
    assert r2["skipped_cached"] == 0
    rows = _cache_rows(temp_db)
    assert len(rows) == 1, "the stale row must be replaced, not duplicated"
    assert rows[0]["content_sig"] is not None
    assert rows[0]["verdict"] == "reject"

    # and the NEW content is now what the cache trusts: an unchanged rerun
    # skips without an LLM call.
    C3 = BeamMemory(session_id="tCRG", db_path=temp_db)
    C3._detect_conflicts = _stub_detect_pair(a1, b1)
    r3 = C3.resolve_cross_session_conflicts()
    assert r3["llm_validations"] == 0 and r3["skipped_cached"] == 1


def test_cache_write_does_not_block_second_connection(temp_db, monkeypatch):
    """Cache writes must not hold the writer lock across a later LLM
    validation call: with a caller-owned deferred transaction and two
    rejected pairs (default WAL journal), a separate connection must be
    able to become a WRITER during the second validation. (timeout=0 so a
    regression fails fast instead of waiting out the default 5s busy
    timeout.)"""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)

    A = BeamMemory(session_id="tCRH", db_path=temp_db)
    B = BeamMemory(session_id="tCRH2", db_path=temp_db)
    a1 = A.remember("[USER] fact one", source="conversation",
                    importance=0.7, scope="global")
    a2 = A.remember("[USER] fact two", source="conversation",
                    importance=0.7, scope="global")
    b1 = B.remember("[USER] fact one changed", source="conversation",
                    importance=0.7, scope="global")
    b2 = B.remember("[USER] fact two changed", source="conversation",
                    importance=0.7, scope="global")

    calls = {"n": 0}
    second_writes = []

    def _validate(older, newer, session_id=None, db_path=None):
        calls["n"] += 1
        if calls["n"] == 2:
            conn2 = sqlite3.connect(str(temp_db), timeout=0)
            try:
                conn2.execute(
                    "INSERT INTO working_memory "
                    "(id, content, source, timestamp, session_id, scope) "
                    "VALUES ('tcrh-writer', '[USER] concurrent write', "
                    "'conversation', '2026-11-11T00:00:00', 'tCRH', 'global')")
                conn2.commit()
                second_writes.append(True)
            finally:
                conn2.close()
        return (False, 0.2, None, "valid")

    monkeypatch.setattr(lcd, "validate_conflict_pair", _validate)

    C = BeamMemory(session_id="tCRH", db_path=temp_db)
    C._detect_conflicts = (
        lambda items, similarity_threshold=0.88, min_gap_hours=1.0:
        [(a1, b1), (a2, b2)])
    # caller-owned deferred transaction (as the Hermes provider does).
    C.conn.execute("BEGIN DEFERRED")
    try:
        r = C.resolve_cross_session_conflicts()
        assert r["llm_validations"] == 2
        assert r["invalidated"] == 0
        assert second_writes == [True], (
            "a second connection must be able to write while the apply is "
            "validating: the cache must not hold the writer lock across a "
            "validate_conflict_pair call")
    finally:
        # the caller commits its transaction (which now also carries the
        # queued verdict writes).
        C.conn.commit()

    # the deferred writes flushed after the validations, inside the caller
    # transaction.
    assert len(_cache_rows(temp_db)) == 2


def test_dry_run_cold_and_evaluated_preview(temp_db, monkeypatch):
    """Dry-run cache contract: a cold plain preview creates no table; an
    evaluated (llm_eval) uncached rejection persists no row; a cached
    preview preserves the complete row, including content_sig, confidence,
    decided_at and expires_at (row counts alone could not catch an
    overwrite of any of those)."""
    A = BeamMemory(session_id="tCRI", db_path=temp_db)
    B = BeamMemory(session_id="tCRI2", db_path=temp_db)
    a1 = A.remember("[USER] favorite color is blue", source="conversation",
                    importance=0.7, scope="global")
    b1 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    import mnemosyne.core.llm_conflict_detector as lcd
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None, "valid"))

    def _table_exists():
        conn = sqlite3.connect(str(temp_db))
        try:
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='cross_session_verdict_cache'").fetchone()
        finally:
            conn.close()

    # 1. cold plain preview: no table is created (reads never create
    #    schema) and nothing is skipped.
    C0 = BeamMemory(session_id="tCRI", db_path=temp_db)
    C0._detect_conflicts = _stub_detect_pair(a1, b1)
    r0 = C0.resolve_cross_session_conflicts(dry_run=True)
    assert r0["skipped_cached"] == 0
    assert _table_exists() is None, (
        "a cold plain preview must not create the cache table")

    # 2. seed with an apply.
    C1 = BeamMemory(session_id="tCRI", db_path=temp_db)
    C1._detect_conflicts = _stub_detect_pair(a1, b1)
    C1.resolve_cross_session_conflicts()
    assert len(_cache_rows(temp_db)) == 1
    before = _cache_rows(temp_db)[0]

    # 3. evaluated uncached rejection (fresh pair): persists no row and
    #    leaves the seeded row untouched.
    a2 = A.remember("[USER] favorite food is pasta", source="conversation",
                    importance=0.7, scope="global")
    b2 = B.remember("[USER] favorite food is ramen now", source="conversation",
                    importance=0.7, scope="global")
    C2 = BeamMemory(session_id="tCRI", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(a2, b2)
    r2 = C2.resolve_cross_session_conflicts(dry_run=True, llm_eval=True)
    assert r2["llm_validations"] == 1
    rows = _cache_rows(temp_db)
    assert len(rows) == 1, "an evaluated dry-run rejection must persist no verdict"
    assert rows[0] == before

    # 4. cached preview: reports the skip and preserves the complete row.
    C3 = BeamMemory(session_id="tCRI", db_path=temp_db)
    C3._detect_conflicts = _stub_detect_pair(a1, b1)
    r3 = C3.resolve_cross_session_conflicts(dry_run=True)
    assert r3["skipped_cached"] == 1
    after = _cache_rows(temp_db)
    assert len(after) == 1
    assert after[0] == before, "a cached preview must preserve the full row"
