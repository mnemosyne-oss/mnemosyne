"""Pipeline-level coverage for the recency-decay age clamp.

Three other files touch this behaviour and none of them makes this statement:

``test_recency_decay_upper_bound.py`` pins ``_recency_decay`` at the function
boundary (exact frozen-clock values in the near-1.0 band, exact saturation for
future stamps, a 20-year overflow case). ``test_enhanced_recall_cache.py`` pins
*invalidation*, with ``recall`` stubbed, because that is the property it tests.
``test_beam.py``'s future-dated cases are the ``#525`` ``valid_until`` tests:
they exercise read filtering, not row-``timestamp`` scoring.

What is left uncovered is what the defect actually looked like from outside:
decay enters the score as a **multiplier**, so a row misdated into the future
used to outrank a row that was genuinely present. These tests write row
timestamps explicitly, drive the real ``BeamMemory.recall()`` on both the
working and episodic paths, and assert the reported ``recency_decay``, the
resulting ``score``, and the order the rows come back in.

Two properties are pinned, in opposite directions, so neither can be satisfied
by flattening the curve:

1. a future-dated row earns **no** bonus — it scores no higher than a row
   stamped at the present instant, and gains no margin over it;
2. a genuinely stale row keeps its **full** penalty — clamping the age must not
   turn into ignoring it.

Deterministic by construction: embeddings are disabled so scoring runs on the
lexical path only, every stamp is derived from ``datetime.now(timezone.utc)``
so the assertions hold at any wall clock, and no test writes outside ``tmp_path``.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory

QUERY = "deploy window"
CONTENT = "the deploy window opens at dusk, operator standing by"

FUTURE = timedelta(hours=9)  # a UTC+9 local-naive stamp read as UTC
STALE = timedelta(days=30)

# One halflife expressed in hours, so the stale expectation tracks the constant
# instead of hard-coding 168.
HALFLIFE_HOURS = beam_module.RECENCY_HALFLIFE_HOURS
STALE_DECAY = math.exp(-(STALE.total_seconds() / 3600.0) / HALFLIFE_HOURS)

# ``recall()`` reports ``score`` as ``round(score, 4)`` while ordering uses the
# unrounded float. A tolerance therefore has to clear the 4-dp quantisation
# (1e-4 absolute, ~1.9e-4 relative around 0.53) and still sit well below the
# defect's margin, which was ~3.7% on identical content. 1% is 50x above the
# quantisation and 3.7x below the bug.
MAX_UNEARNED_MARGIN = 0.01


@pytest.fixture
def lexical_only(monkeypatch):
    """Score on the lexical path alone: no model load, no vector drift."""
    monkeypatch.setattr("mnemosyne.core.embeddings.available", lambda: False)


def _row_id(table: str, suffix: str) -> str:
    return f"{table}-{suffix}"


def _insert(beam: BeamMemory, table: str, rows) -> None:
    """Insert rows with explicit timestamps, bypassing the write path.

    ``remember()`` stamps the current time, so the only way to construct a
    legacy future-dated row is to write the column directly.
    """
    assert table in ("working_memory", "episodic_memory"), table  # f-string below
    for rid, stamp, session in rows:
        beam.conn.execute(
            f"INSERT INTO {table} "
            "(id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, CONTENT, "test", stamp, session, 0.5),
        )
    beam.conn.commit()


def _recall(beam: BeamMemory):
    out = beam.recall(QUERY, top_k=10, explain=True, _cross_session=True)
    if isinstance(out, dict):
        hits = out.get("hits") or out.get("results") or []
    else:  # older signature returned a bare list
        hits = out
    return hits, [h["id"] for h in hits]


def _score_single_row(tmp_path: Path, table: str, offset: timedelta):
    """Seed one row at ``now + offset`` and return its explain hit."""
    stamp = (datetime.now(timezone.utc) + offset).isoformat()
    beam = BeamMemory(session_id="s-present", db_path=tmp_path / f"{table}.db")
    _insert(beam, table, [(_row_id(table, "only"), stamp, "s-present")])
    hits, ids = _recall(beam)
    assert len(hits) == 1, f"expected a single recalled row, got {ids}"
    return hits[0]


def _three_row_ranking(tmp_path: Path, table: str):
    """Present, future-dated and stale rows with identical content.

    Identical content is deliberate: base score, lexical score and importance
    are then equal by construction, so the surviving ordering difference can
    only come from the recency factor.
    """
    now = datetime.now(timezone.utc)
    rows = [
        (_row_id(table, "present"), now.isoformat(), "s-present"),
        (_row_id(table, "future"), (now + FUTURE).isoformat(), "s-future"),
        (_row_id(table, "stale"), (now - STALE).isoformat(), "s-stale"),
    ]
    beam = BeamMemory(session_id="s-present", db_path=tmp_path / f"{table}-rank.db")
    _insert(beam, table, rows)
    hits, ids = _recall(beam)
    assert len(hits) == 3, f"expected all three rows back, got {ids}"
    return {h["id"]: h for h in hits}, ids


# --------------------------------------------------------------------------
# 1. No bonus for a future-dated row (score)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["working_memory", "episodic_memory"])
def test_future_dated_row_scores_no_higher_than_a_present_row(
    tmp_path: Path, lexical_only, table: str
):
    """The multiplier must not reward a row for being misdated.

    Present and future rows carry identical content in identical sessions, so
    their base scores are equal. Unclamped, the future row's decay is above 1.0
    and multiplies its score ~3.7% above the present row's; clamped, the two are
    equal to the last bit.
    """
    present = _score_single_row(tmp_path / "present", table, timedelta(0))
    future = _score_single_row(tmp_path / "future", table, FUTURE)

    # Reported values are ``round(x, 4)``: the present row's true decay is
    # 0.99999845 (it is microseconds old by recall time), so compare at
    # rounding precision rather than claiming exactness.
    assert present["recency_decay"] == pytest.approx(1.0, abs=1e-4), present["recency_decay"]
    assert future["recency_decay"] == 1.0, (
        f"a row {FUTURE} in the future must saturate at 1.0, got "
        f"{future['recency_decay']}"
    )
    assert future["score"] <= present["score"] * (1 + MAX_UNEARNED_MARGIN), (
        f"future-dated row scored {future['score']} against a present row's "
        f"{present['score']} ({100 * (future['score'] / present['score'] - 1):.2f}% "
        f"more): the clamp is not reaching {table}"
    )
    # Two-sided: the clamp must not boost the misdated row, and it must not
    # penalise it either. A one-sided bound would accept a regression that
    # multiplies the future row down while every other assertion still holds.
    assert future["score"] >= present["score"] * (1 - MAX_UNEARNED_MARGIN), (
        f"future-dated row scored {future['score']} against a present row's "
        f"{present['score']} ({100 * (future['score'] / present['score'] - 1):.2f}%): "
        f"a future stamp must be treated as new, not as stale, in {table}"
    )
    assert present["score"] > 0.0, "sanity: the row must actually score"


# --------------------------------------------------------------------------
# 2. No unearned margin over a present row
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["working_memory", "episodic_memory"])
def test_future_dated_row_gains_no_unearned_margin(tmp_path: Path, lexical_only, table: str):
    """On identical content the misdated row may tie the newest row, never lead it by a margin.

    Which of the two ranks first is deliberately *not* pinned: the clamp
    saturates the future row at exactly 1.0, while the present control is a
    fraction of a second old by the time ``recall`` runs (measured decay
    0.99999845 against 1.0). A sub-second tie is not a defect and would be a
    flaky assertion. The defect was a *margin*: unclamped, the future row's
    factor reached 1.055, adding ~3.7% to its score on identical content and
    taking rank 0 outright.
    """
    hits, ids = _three_row_ranking(tmp_path, table)
    present = hits[_row_id(table, "present")]
    future = hits[_row_id(table, "future")]

    assert present["recency_decay"] == 1.0, present["recency_decay"]
    assert future["recency_decay"] == 1.0, (
        f"a row {FUTURE} in the future must saturate at 1.0, got "
        f"{future['recency_decay']}"
    )
    # Reported scores are 4-dp rounded, so the tolerance clears quantisation
    # rather than the sub-second tie; see MAX_UNEARNED_MARGIN.
    margin = future["score"] - present["score"]
    assert margin <= present["score"] * MAX_UNEARNED_MARGIN, (
        f"the future-dated row gained {margin:.6f} over the present row "
        f"({future['score']} vs {present['score']}), i.e. "
        f"{100 * margin / present['score']:.2f}% on identical content"
    )
    assert ids[-1] == _row_id(table, "stale"), f"the stale row must rank last; got {ids}"


# --------------------------------------------------------------------------
# 3. Staleness still costs: the clamp is not a floor on the curve
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["working_memory", "episodic_memory"])
def test_stale_row_keeps_its_full_penalty(tmp_path: Path, lexical_only, table: str):
    """A mutation that returns 1.0 for everything must fail here.

    The clamp exists to stop ages below zero. It must not lift any other age,
    so a 30-day-old row keeps ``exp(-720/168)`` of its score and ranks last.
    """
    hits, ids = _three_row_ranking(tmp_path, table)
    stale = hits[_row_id(table, "stale")]

    # The expectation is derived from the shipped half-life rather than
    # hard-coded, because this file guards the clamp and not the curve's
    # calibration -- a deliberate default change should move the expectation
    # with it. What must not happen silently is a half-life so large that 30
    # days stops costing anything: then "ranks last" would assert nothing. The
    # constant is env-overridable (MNEMOSYNE_RECENCY_HALFLIFE), so express the
    # requirement as a property of the derived penalty, not of the constant.
    assert STALE_DECAY < 0.05, (
        f"at {STALE} the expected decay is {STALE_DECAY:.4f} under a "
        f"{HALFLIFE_HOURS:g}h half-life -- too close to 1.0 for a stale-last "
        f"ranking assertion to mean anything; re-derive this test"
    )

    assert stale["recency_decay"] == pytest.approx(STALE_DECAY, abs=1e-4), (
        f"expected the ordinary curve at {STALE}: {STALE_DECAY:.4f}, got "
        f"{stale['recency_decay']}"
    )
    assert ids[-1] == _row_id(table, "stale"), (
        f"the stale row must rank last; got {ids}"
    )
    assert stale["score"] < hits[_row_id(table, "present")]["score"], (
        f"stale scored {stale['score']}, present scored "
        f"{hits[_row_id(table, 'present')]['score']}: recency is not being applied"
    )
