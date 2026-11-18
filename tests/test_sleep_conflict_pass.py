"""Sleep-path wiring for the veracity conflict-resolution pass.

Two defects fixed together:

- Defect 1: ``run_consolidation_pass`` was documented as the sleep
  path's retry hook (PR #1082 review) but no sleep code called it.
  Post-fix, ``BeamMemory.sleep()`` invokes it once per pass, bounded by
  the existing ``MNEMOSYNE_CONFLICT_PAIR_BUDGET`` /
  ``MNEMOSYNE_CONFLICT_TIME_BUDGET_S`` constants, best-effort, and
  records the outcome in the returned dict.
- Defect 2: ``resolve_conflict_by_facts`` only wrote
  ``consolidated_facts.superseded_by`` and never stamped
  ``conflicts.resolution``, so a wired pass could not drain the ledger.
  Post-fix it stamps the matching open row(s) by the order-normalized
  pair (the E8 expression-index key), in the same write scope as the
  supersession, using ``resolve_conflict``'s resolution format.
"""
import pytest

import mnemosyne.core.beam as beam_module
from mnemosyne.core import llm_conflict_detector as lcd
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.veracity_consolidation import VeracityConsolidator


@pytest.fixture
def beam(tmp_path, monkeypatch):
    """Hermetic beam: no handshake, no census walk, no LLM conflict
    validation, no model-refresh inference."""
    monkeypatch.setenv("MNEMOSYNE_FEDERATION_HANDSHAKE", "0")
    monkeypatch.setenv("MNEMOSYNE_FLEET_CENSUS", "0")
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", False)
    monkeypatch.setattr(
        "mnemosyne.core.model_refresh.infer_model_update_proposals",
        lambda items: [],
    )
    memory = BeamMemory(
        session_id="conflict-pass", db_path=tmp_path / "memory.db"
    )
    # Two eligible working rows so sleep() takes the consolidation path
    # instead of the no-op early return.
    for i in range(2):
        memory.conn.execute(
            "INSERT INTO working_memory(id, content, source, timestamp, "
            "session_id) VALUES (?, ?, ?, ?, ?)",
            (f"wm-{i}", f"note {i}", "conversation",
             f"2026-01-0{i + 1}T10:00:00", "conflict-pass"),
        )
    memory.conn.commit()
    yield memory
    memory.conn.close()


def _seed_pair(vc, subject, winner_obj, loser_obj):
    """Seed a high-confidence fact plus a lower-confidence contradicting
    fact. Returns (winning_id, losing_id). The conflicts row for the
    pair is written by consolidate_fact's own queue drain (handshake off
    in the fixture -> decision None -> row inserted)."""
    for _ in range(4):
        vc.consolidate_fact(subject, "is", winner_obj, "stated", "src_high")
    vc.consolidate_fact(subject, "is", loser_obj, "inferred", "src_low")
    rows = vc.conn.execute(
        "SELECT id, object FROM consolidated_facts WHERE subject = ?",
        (subject,),
    ).fetchall()
    ids = {r["object"]: r["id"] for r in rows}
    return ids[winner_obj], ids[loser_obj]


# ---------------------------------------------------------------------------
# (a) A wired sleep pass resolves an obvious pair
# ---------------------------------------------------------------------------

def test_wired_sleep_resolves_obvious_pair(beam):
    vc = beam.veracity_consolidator
    winner, loser = _seed_pair(vc, "Eve", "CEO", "VP")

    result = beam.sleep(force=True)

    assert result["status"] == "consolidated"
    assert result["veracity_conflict_pass"] == {"resolutions": 1}
    loser_row = beam.conn.execute(
        "SELECT superseded_by FROM consolidated_facts WHERE id = ?", (loser,)
    ).fetchone()
    assert loser_row["superseded_by"] == winner


def test_sleep_invokes_pass_exactly_once_with_beam_budgets(beam, monkeypatch):
    vc = beam.veracity_consolidator
    calls = []
    original = vc.run_consolidation_pass

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(vc, "run_consolidation_pass", counting)

    beam.sleep(force=True)

    assert len(calls) == 1
    assert calls[0]["pair_budget"] == beam_module._CONFLICT_PAIR_BUDGET
    assert calls[0]["time_budget_s"] == beam_module._CONFLICT_TIME_BUDGET_S


# ---------------------------------------------------------------------------
# (b) The matching conflicts row gets non-NULL resolution, and the
#     losing fact gets superseded_by
# ---------------------------------------------------------------------------

def test_wired_sleep_stamps_conflict_row(beam):
    vc = beam.veracity_consolidator
    winner, loser = _seed_pair(vc, "Eve", "CEO", "VP")

    beam.sleep(force=True)

    stored = beam.conn.execute(
        "SELECT fact_a_id, fact_b_id, resolution, resolved_at FROM conflicts"
    ).fetchone()
    # The detector does not canonicalize orientation: the queued row
    # stores (loser, winner). The stamp must have matched it anyway.
    assert (stored["fact_a_id"], stored["fact_b_id"]) == (loser, winner)
    assert stored["resolution"] == f"superseded_by_{winner}"
    assert stored["resolved_at"] is not None
    loser_row = beam.conn.execute(
        "SELECT superseded_by FROM consolidated_facts WHERE id = ?", (loser,)
    ).fetchone()
    assert loser_row["superseded_by"] == winner


def test_resolve_conflict_by_facts_stamps_open_row_swapped_orientation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MNEMOSYNE_FEDERATION_HANDSHAKE", "0")
    cons = VeracityConsolidator(db_path=tmp_path / "cons.db")
    for _ in range(4):
        cons.consolidate_fact("Eve", "is", "CEO", "stated", "src")
    cons.consolidate_fact("Eve", "is", "VP", "inferred", "src")
    ids = {
        r["object"]: r["id"]
        for r in cons.conn.execute(
            "SELECT id, object FROM consolidated_facts WHERE subject = 'Eve'"
        ).fetchall()
    }
    row = cons.conn.execute(
        "SELECT fact_a_id, fact_b_id, resolution FROM conflicts"
    ).fetchone()
    assert (row["fact_a_id"], row["fact_b_id"]) == (ids["VP"], ids["CEO"])
    assert row["resolution"] is None

    cons.resolve_conflict_by_facts(ids["CEO"], ids["VP"])

    stamped = cons.conn.execute(
        "SELECT resolution, resolved_at FROM conflicts"
    ).fetchone()
    assert stamped["resolution"] == f"superseded_by_{ids['CEO']}"
    assert stamped["resolved_at"] is not None
    vp = cons.conn.execute(
        "SELECT superseded_by FROM consolidated_facts WHERE id = ?",
        (ids["VP"],),
    ).fetchone()
    assert vp["superseded_by"] == ids["CEO"]


def test_resolve_conflict_by_facts_first_writer_wins(tmp_path, monkeypatch):
    """A second resolution attempt on the same pair must not overwrite
    the first stamp (still-open rows only)."""
    monkeypatch.setenv("MNEMOSYNE_FEDERATION_HANDSHAKE", "0")
    cons = VeracityConsolidator(db_path=tmp_path / "cons.db")
    for _ in range(4):
        cons.consolidate_fact("Eve", "is", "CEO", "stated", "src")
    cons.consolidate_fact("Eve", "is", "VP", "inferred", "src")
    ids = {
        r["object"]: r["id"]
        for r in cons.conn.execute(
            "SELECT id, object FROM consolidated_facts WHERE subject = 'Eve'"
        ).fetchall()
    }

    cons.resolve_conflict_by_facts(ids["CEO"], ids["VP"])
    first = cons.conn.execute(
        "SELECT resolution FROM conflicts"
    ).fetchone()["resolution"]

    cons.resolve_conflict_by_facts(ids["VP"], ids["CEO"])
    second = cons.conn.execute(
        "SELECT resolution FROM conflicts"
    ).fetchone()["resolution"]

    assert first == f"superseded_by_{ids['CEO']}"
    assert second == first


# ---------------------------------------------------------------------------
# (c) The pass respects the pair/time budget
# ---------------------------------------------------------------------------

def test_pass_respects_pair_budget(beam, monkeypatch):
    vc = beam.veracity_consolidator
    _seed_pair(vc, "Eve", "CEO", "VP")
    _seed_pair(vc, "Grace", "CTO", "CFO")
    monkeypatch.setattr(beam_module, "_CONFLICT_PAIR_BUDGET", 1)

    result = beam.sleep(force=True)

    assert result["veracity_conflict_pass"] == {"resolutions": 1}
    assert beam.conn.execute(
        "SELECT COUNT(*) AS n FROM conflicts WHERE resolution IS NOT NULL"
    ).fetchone()["n"] == 1
    assert beam.conn.execute(
        "SELECT COUNT(*) AS n FROM consolidated_facts "
        "WHERE superseded_by IS NOT NULL"
    ).fetchone()["n"] == 1


def test_pass_respects_time_budget(beam, monkeypatch):
    """An exhausted wall-clock budget stops the loop cleanly: nothing
    resolves, nothing crashes."""
    vc = beam.veracity_consolidator
    _seed_pair(vc, "Eve", "CEO", "VP")
    _seed_pair(vc, "Grace", "CTO", "CFO")
    monkeypatch.setattr(beam_module, "_CONFLICT_TIME_BUDGET_S", 0.0)

    result = beam.sleep(force=True)

    assert result["status"] == "consolidated"
    assert result["veracity_conflict_pass"] == {"resolutions": 0}
    assert beam.conn.execute(
        "SELECT COUNT(*) AS n FROM conflicts WHERE resolution IS NOT NULL"
    ).fetchone()["n"] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) AS n FROM consolidated_facts "
        "WHERE superseded_by IS NOT NULL"
    ).fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# (d) A dry run writes nothing
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing(beam):
    vc = beam.veracity_consolidator
    _seed_pair(vc, "Eve", "CEO", "VP")
    before = sorted(beam.conn.iterdump())

    result = beam.sleep(force=True, dry_run=True)

    assert sorted(beam.conn.iterdump()) == before
    assert result["veracity_conflict_pass"] is None
    assert beam.conn.execute(
        "SELECT COUNT(*) AS n FROM conflicts WHERE resolution IS NOT NULL"
    ).fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# (e) An exception inside the pass does not fail sleep()
# ---------------------------------------------------------------------------

def test_pass_exception_does_not_fail_sleep(beam, monkeypatch):
    vc = beam.veracity_consolidator
    _seed_pair(vc, "Eve", "CEO", "VP")

    def boom(*args, **kwargs):
        raise RuntimeError("pass exploded (message may contain secrets)")

    monkeypatch.setattr(vc, "run_consolidation_pass", boom)

    result = beam.sleep(force=True)

    assert result["status"] == "consolidated"
    assert result["veracity_conflict_pass"] == {"error": "RuntimeError"}
    assert result["summaries_created"] >= 1
    assert beam.conn.execute(
        "SELECT COUNT(*) AS n FROM conflicts WHERE resolution IS NOT NULL"
    ).fetchone()["n"] == 0
