"""Polyphonic recall hydrates the first tier the caller may see.

Invariant: an engine hit comes back as a row from a tier it may come from
that passes the caller's filters, and is dropped only when no such row exists.

Grid: tier hint x episodic row x working row x cross-session.

- Tier hint: none (graph and fact hits), "episodic" or "working" (typed hits).
  An untyped hit tries episodic first; a typed hit reads only its own tier.
- Each tier's row for the shared id: absent, the caller's own, or another
  session's (visible only with cross-session on).
"""
import itertools

import pytest

import mnemosyne.core.beam as bm
from mnemosyne.core.polyphonic_recall import PolyphonicResult

CALLER = "caller-session"
OTHER = "other-session"
MID = "shared-id"
TIERS = ("episodic", "working")


class _FakeEngine:
    def __init__(self, results):
        self._results = results

    def recall(self, **kwargs):
        return list(self._results)


def _insert(beam, tier, owner):
    table = "episodic_memory" if tier == "episodic" else "working_memory"
    beam.conn.execute(
        f"INSERT INTO {table} (id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, ?, 'user', datetime('now'), ?, 0.5, 'session')",
        (MID, f"{tier} quokka", CALLER if owner == "mine" else OTHER),
    )
    beam.conn.commit()


def _expected(hint, rows, cross_session):
    visible = {"mine", "foreign"} if cross_session else {"mine"}
    for tier in (hint,) if hint else TIERS:
        if rows[tier] in visible:
            return [(MID, tier)]
    return []


CELLS = [
    (hint, ep, wm, cross)
    for hint, ep, wm, cross in itertools.product(
        (None, "episodic", "working"), ("absent", "mine", "foreign"),
        ("absent", "mine", "foreign"), (False, True))
]


@pytest.mark.parametrize("hint,episodic,working,cross_session", CELLS,
                         ids=[f"{h or 'untyped'}-ep_{e}-wm_{w}-{'cross' if c else 'scoped'}"
                              for h, e, w, c in CELLS])
def test_hit_hydrates_first_eligible_tier(tmp_path, monkeypatch, hint, episodic, working,
                                          cross_session):
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    if cross_session:
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "1")
    else:
        monkeypatch.delenv("MNEMOSYNE_CROSS_SESSION", raising=False)
    beam = bm.BeamMemory(session_id=CALLER, db_path=tmp_path / "memory.db")
    rows = {"episodic": episodic, "working": working}
    for tier, owner in rows.items():
        if owner != "absent":
            _insert(beam, tier, owner)
    metadata = {"_self_echo_tier": hint} if hint else {}
    engine = _FakeEngine([PolyphonicResult(memory_id=MID, combined_score=0.9,
                                           voice_scores={"graph": 0.9}, metadata=metadata)])
    monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: engine)

    results = beam.recall("quokka", top_k=5)

    assert [(r["id"], r["tier"]) for r in results] == _expected(hint, rows, cross_session)


@pytest.mark.parametrize("hint,expected", [
    (None, [(MID, "working", "working quokka", CALLER)]),
    ("episodic", []),
], ids=["untyped-falls-through", "typed-episodic-stays"])
def test_caller_filter_rejecting_episodic_row(tmp_path, monkeypatch, hint, expected):
    # Both rows are the caller's, so only the source filter tells them apart.
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.delenv("MNEMOSYNE_CROSS_SESSION", raising=False)
    beam = bm.BeamMemory(session_id=CALLER, db_path=tmp_path / "memory.db")
    _insert(beam, "episodic", "mine")
    _insert(beam, "working", "mine")
    beam.conn.execute("UPDATE working_memory SET source = 'conversation' WHERE id = ?", (MID,))
    beam.conn.commit()
    metadata = {"_self_echo_tier": hint} if hint else {}
    engine = _FakeEngine([PolyphonicResult(memory_id=MID, combined_score=0.9,
                                           voice_scores={"graph": 0.9}, metadata=metadata)])
    monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: engine)

    results = beam.recall("quokka", top_k=5, source="conversation")

    assert [(r["id"], r["tier"], r["content"], r["session_id"]) for r in results] == expected
