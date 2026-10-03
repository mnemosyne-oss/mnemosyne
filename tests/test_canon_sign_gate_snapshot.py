"""Fixture regression for the canon-sign gate's cross-table read snapshot.

The gate (canon_sign_head_test.py) reads working_memory (its A0-A6 arms) and
conflicts (A7) through ONE pinned read transaction, so a concurrent writer
committing between the two reads cannot be scored as a store state that never
existed at a single moment (the v6 law, from CodeRabbit review 5330458141).
This is the fixture test that review's follow-up asked for (comment
4119017434): a committed conflicts update is interleaved between the gate's
two reads, driven through the gate's own helpers, so a refactor that drops
the read txn fails here instead of reopening the mixed-state window.

Run: pytest tests/test_canon_sign_gate_snapshot.py
"""
import importlib.util
import sqlite3
from pathlib import Path
from unittest import mock

GATE_PATH = Path(__file__).with_name("canon_sign_head_test.py")


def _load_gate():
    # A bare import must be inert: no argv-derived DB path, no sqlite open.
    spec = importlib.util.spec_from_file_location("canon_sign_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def _fixture_db(tmp_path):
    """Minimal WAL store: only the tables/columns the gate's two reads touch.

    WAL mirrors the production banks and is what lets the interleaved commit
    reach fresh readers while the gate's open read txn keeps its snapshot.
    """
    path = tmp_path / "gate-snapshot.db"
    db = sqlite3.connect(path)
    db.execute("pragma journal_mode=WAL")
    db.execute(
        "create table working_memory (id text primary key, content text, "
        "valid_until text, superseded_by text, metadata_json text, created_at text)"
    )
    db.execute(
        "insert into working_memory values ('wm_1', 'Surface meta: Astraea "
        "canon-sign row probe', null, null, '{}', '2026-09-28 00:00:00')"
    )
    db.execute("create table conflicts (id text primary key, resolution text)")
    db.execute("insert into conflicts values ('cf_before', null)")
    db.commit()
    db.close()
    return path


def _commit_conflict(path, conflict_id):
    """A live writer lands a new unresolved conflict row, committed."""
    writer = sqlite3.connect(path)
    writer.execute("insert into conflicts values (?, null)", (conflict_id,))
    writer.commit()
    writer.close()


def test_bare_import_of_the_gate_is_inert():
    # Direct import-time proof, not just absence of leftover attributes: patch
    # sqlite3.connect across the whole module exec (the gate references it as
    # a module attribute, so the patch is on its real call path) and require
    # zero calls. The attribute checks below stay as the second witness.
    with mock.patch.object(sqlite3, "connect") as connect:
        gate = _load_gate()
    assert not connect.called, "gate called sqlite3.connect at import time"
    assert not hasattr(gate, "db"), "gate opened a database at import time"
    assert not hasattr(gate, "rows"), "gate ran its reads at import time"


def test_interleaved_conflict_commit_stays_out_of_the_gate_snapshot(tmp_path):
    gate = _load_gate()
    path = _fixture_db(tmp_path)
    conn = gate.open_read_connection(str(path))

    rows = gate.read_working_rows(conn)          # A0-A6 read pins the snapshot
    assert len(rows) == 1

    _commit_conflict(path, "cf_interloper")      # commit lands between the reads

    # A7 on the same pinned connection must still report the pre-write count:
    # the pair (working_memory@T0, conflicts@T1) never existed at one moment.
    assert gate.read_unresolved_conflict_count(conn) == 1, (
        "A7 read left the pinned snapshot — the gate could again report a "
        "mixed cross-table state"
    )

    # The write really landed, or the assertion above would be vacuous.
    # (The gate's reads expect sqlite3.Row; the fresh connection opts in too.)
    fresh = sqlite3.connect(str(path))
    fresh.row_factory = sqlite3.Row
    try:
        assert gate.read_unresolved_conflict_count(fresh) == 2
    finally:
        fresh.close()
    conn.close()


def test_control_without_the_pinned_txn_the_commit_is_visible(tmp_path):
    gate = _load_gate()
    path = _fixture_db(tmp_path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row               # deliberately no read txn

    assert len(gate.read_working_rows(conn)) == 1
    _commit_conflict(path, "cf_interloper")

    # Control: without BEGIN the same interleave DOES change A7. The fixture
    # can observe the race, so the snapshot assertion above has teeth — and
    # a regression that drops the gate's read txn shows up as this mixed
    # A0-A6/A7 pair rather than a silent PASS.
    assert gate.read_unresolved_conflict_count(conn) == 2, (
        "control broken: the fixture must be able to surface the interleaved write"
    )
    conn.close()
