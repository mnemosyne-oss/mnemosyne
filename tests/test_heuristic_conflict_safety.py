"""Regression tests: heuristic conflict pairs must never auto-invalidate.

BeamMemory.sleep() Phase-1 detects similar row pairs with a pure-vector
heuristic (_detect_conflicts, cosine > 0.88 + token overlap). When LLM
conflict validation is disabled (MNEMOSYNE_LLM_CONFLICT_DETECTION unset —
the default), the heuristic branch used to call invalidate() on every
detected pair, silently setting valid_until on the older row.

Cosine similarity alone cannot distinguish a true contradiction from a
benign restatement of the same fact: in production this auto-expired 59%
of the memory pool (142/243 items) across consolidation passes, with no
log and no recovery path (valid_until permanently excludes the row from
recall).

Contract after the fix:
  - heuristic pairs are only counted and logged (conflicts_detected_only)
  - no row's valid_until / superseded_by changes when LLM validation is off
"""

import sqlite3

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db(tmp_path):
    return tmp_path / "test.db"


def _row_state(db_path, row_id):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
            (row_id,),
        ).fetchone()
    finally:
        conn.close()


def _make_similar_pair(beam):
    """Two rows >1h apart, cosine > 0.88, >=2 shared tokens, edit-dist > 0.3.

    Same register, restating the same underlying fact differently — the
    exact class the heuristic mislabels as a contradiction. Embeddings are
    inserted directly so the test does not depend on an embedding backend.
    """
    import json
    import numpy as np

    base = np.zeros(8, dtype=np.float32)
    base[0] = 1.0
    vec_a = base + np.array([0.05, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    vec_b = base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # Rows must be >1h apart for the heuristic pair check. remember() stamps
    # now(); backdate the first row directly.
    id_a = beam.remember(
        content="User plans to visit the central library downtown on Friday morning",
        source="conversation",
        dedupe=False,
    )
    id_b = beam.remember(
        content="Scranton branch: user intends to go to the central library downtown Friday early instead",
        source="conversation",
        dedupe=False,
    )
    cur = beam.conn.cursor()
    cur.execute(
        "UPDATE working_memory SET timestamp = ? WHERE id = ?",
        ("2026-01-01T10:00:00", id_a),
    )
    for mid, vec in ((id_a, vec_a), (id_b, vec_b)):
        cur.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
            (mid, json.dumps([float(x) for x in vec])),
        )
    beam.conn.commit()
    return id_a, id_b


def _heuristic_pair_flags(beam, id_a, id_b):
    rows = [
        dict(beam.conn.execute(
            "SELECT id, content, timestamp, superseded_by FROM working_memory WHERE id = ?",
            (rid,),
        ).fetchone())
        for rid in (id_a, id_b)
    ]
    return beam._detect_conflicts(rows)


def test_heuristic_pairs_never_invalidate_without_llm(temp_db, monkeypatch):
    # The gate is baked at detector import time, so delenv alone cannot
    # force the heuristic branch under a flag-set CI env: patch the module
    # constant directly so the branch is deterministic regardless of
    # collection order or deployment env.
    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED",
        False,
    )
    beam = BeamMemory(session_id="s-conflict", db_path=temp_db)
    id_a, id_b = _make_similar_pair(beam)

    conflicts = _heuristic_pair_flags(beam, id_a, id_b)
    assert conflicts, "fixture must produce a heuristic pair"

    result = beam.sleep(force=True)

    assert result["conflicts_detected_only"] == 1, (
        "fixture produces exactly one heuristic pair: the counter must "
        "match it exactly (an over-count hides double detection)"
    )
    assert result["conflicts_resolved"] == 0

    for rid in (id_a, id_b):
        state = _row_state(temp_db, rid)
        assert state["valid_until"] is None, (
            "heuristic conflict must not set valid_until"
        )
        assert state["superseded_by"] is None, (
            "heuristic conflict must not set superseded_by"
        )


def test_dry_run_never_invalidates(temp_db, monkeypatch):
    # Same import-time-gate discipline as above: patch the constant.
    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED",
        False,
    )
    beam = BeamMemory(session_id="s-conflict", db_path=temp_db)
    id_a, id_b = _make_similar_pair(beam)

    result = beam.sleep(force=True, dry_run=True)
    assert result["conflicts_resolved"] == 0
    assert result["conflicts_detected_only"] == 1

    for rid in (id_a, id_b):
        state = _row_state(temp_db, rid)
        assert state["valid_until"] is None
        assert state["superseded_by"] is None


def test_detected_pairs_are_logged(caplog, temp_db, monkeypatch):
    """The audit trail is the INFO log line: assert it fires with the pair
    count (auditor round-2: log-deletion mutation kept the tests green)."""
    import logging

    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED",
        False,
    )
    beam = BeamMemory(session_id="s-conflict", db_path=temp_db)
    id_a, id_b = _make_similar_pair(beam)

    with caplog.at_level(logging.INFO, logger="mnemosyne.core.beam"):
        result = beam.sleep(force=True)

    records = [
        r for r in caplog.records
        if "conflict heuristics" in r.getMessage()
    ]
    assert records, "detected-only pairs must emit the audit log line"
    assert str(result["conflicts_detected_only"]) in records[0].getMessage(), (
        "log line must carry the detected-pair count"
    )


def test_gate_enabled_without_endpoint_counts_detected_only(temp_db, monkeypatch):
    """Gate=true with NO endpoint: validate_conflict_pair returns
    not-conflict for every pair, which used to drop them from BOTH
    counters. They must land in conflicts_detected_only instead."""
    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED",
        True,
    )
    from mnemosyne.core import llm_conflict_detector as lcd

    monkeypatch.setattr(lcd, "CONFLICT_LLM_BASE_URL", None)

    beam = BeamMemory(session_id="s-conflict", db_path=temp_db)
    id_a, id_b = _make_similar_pair(beam)

    result = beam.sleep(force=True)

    assert result["conflicts_resolved"] == 0
    assert result["conflicts_detected_only"] == 1, (
        "unvalidatable pairs must be counted as detected-only, not dropped"
    )
    for rid in (id_a, id_b):
        state = _row_state(temp_db, rid)
        assert state["valid_until"] is None
        assert state["superseded_by"] is None


def test_dry_run_does_not_call_conflict_llm(temp_db, monkeypatch):
    """dry_run must stay side-effect-free: no LLM call, no cost row
    (round-4: validate_conflict_pair sat outside the dry_run guard and
    log_cost persisted real spend during a read-only preview)."""
    from unittest.mock import patch

    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.CONFLICT_LLM_BASE_URL",
        "https://conflict-llm.test",
    )
    beam = BeamMemory(session_id="s-conflict", db_path=temp_db)
    id_a, id_b = _make_similar_pair(beam)

    with patch(
        "mnemosyne.core.llm_conflict_detector.validate_conflict_pair",
        side_effect=AssertionError("LLM must not be called in dry_run"),
    ):
        result = beam.sleep(force=True, dry_run=True)

    # Planned validations surface as detected-only, never as resolutions.
    assert result["conflicts_resolved"] == 0
    assert result["conflicts_detected_only"] == 1
    for rid in (id_a, id_b):
        state = _row_state(temp_db, rid)
        assert state["valid_until"] is None
        assert state["superseded_by"] is None


def test_invalidate_failure_counts_as_detected(temp_db, monkeypatch):
    """invalidate() returning False (self-pair / replacement vanished during
    the LLM window) must NOT increment conflicts_resolved — round-4: the
    counter used to claim resolutions that never happened."""
    from unittest.mock import patch

    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "mnemosyne.core.llm_conflict_detector.CONFLICT_LLM_BASE_URL",
        "https://conflict-llm.test",
    )
    beam = BeamMemory(session_id="s-conflict", db_path=temp_db)
    id_a, id_b = _make_similar_pair(beam)

    with patch.object(beam, "_detect_conflicts", return_value=[(id_a, id_b)]), \
         patch(
             "mnemosyne.core.llm_conflict_detector.validate_conflict_pair",
             return_value=(True, 0.97, "the corrected fact"),
         ), \
         patch.object(beam, "invalidate", return_value=False):
        result = beam.sleep(force=True)

    assert result["conflicts_resolved"] == 0, (
        "a refused invalidation must not be reported as resolved"
    )
    assert result["conflicts_detected_only"] == 1
