"""Tests for cross-session global contradiction resolution.

The pass resolves factual contradictions among `scope='global'` memories across
different sessions so recall/prefetch stop co-presenting the stale and current
versions. Conflict *detection* needs embeddings, which are unavailable in CI
(`MNEMOSYNE_NO_EMBEDDINGS=1`), so the tests stub `BeamMemory._detect_conflicts`
to return a deterministic cross-session pair and assert the resolution (cross-
session `invalidate` + recall filtering) is real and non-vacuous.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
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
                        lambda older, newer, session_id, db_path: (True, 0.9, "green"))
    assert C.resolve_cross_session_conflicts()["invalidated"] == 1

    # declined (different context / complement): nothing superseded
    b2 = A.remember("[USER] favorite color is mauve", source="conversation",
                    importance=0.7, scope="global")
    g2 = B.remember("[USER] favorite color is green now", source="conversation",
                    importance=0.7, scope="global")
    C2 = BeamMemory(session_id="tC2", db_path=temp_db)
    C2._detect_conflicts = _stub_detect_pair(b2, g2)
    monkeypatch.setattr(lcd, "validate_conflict_pair",
                        lambda older, newer, session_id, db_path: (False, 0.2, None))
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
        return True, 0.9, "now"
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
                        lambda older, newer, session_id, db_path: (False, 0.2, None))
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
            calls.__setitem__("n", calls["n"] + 1) or (True, 0.9, "new")
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
            calls.__setitem__("n", calls["n"] + 1) or (True, 0.9, "new")
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
