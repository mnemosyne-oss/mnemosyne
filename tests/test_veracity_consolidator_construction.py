"""Regression tests for VeracityConsolidator construction in BeamMemory.

Pre-fix (issue #229 / PR #230): both construction sites built the
consolidator positionally -- ``VeracityConsolidator(self.conn)``. Because
the signature is ``__init__(self, db_path=None, conn=None)``, the live
sqlite3 Connection was bound to ``db_path``, the ``conn is None`` branch
ran, and ``sqlite3.connect(str(self.db_path))`` stringified the Connection
into a phantom file (e.g. ``<sqlite3.Connection object at 0x...>``). The
consolidator then operated on an empty throwaway DB, so the
consolidated_facts recall path silently returned nothing -- a nasty silent
bug with no error.

Post-fix: both sites pass ``conn=self.conn, db_path=self.db_path`` by
keyword, sharing BeamMemory's connection.

These tests pin both sites the maintainer called out:
  - site A, beam.py __init__  -> the cached ``beam.veracity_consolidator``
  - site B, beam.py fact_recall -> the per-call local consolidator
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "mnemosyne_vc.db"


# --- Site A: BeamMemory.__init__ (cached consolidator) ---------------------

def test_init_consolidator_shares_beam_connection(temp_db):
    """The consolidator built in __init__ must reuse BeamMemory's own
    connection, not open its own DB. The positional regression flips every
    one of these invariants."""
    beam = BeamMemory(session_id="vc-init", db_path=temp_db)
    vc = beam.veracity_consolidator
    assert vc is not None, "veracity_consolidator was not constructed"

    # Shares the live connection -> reads/writes hit the same DB.
    assert vc.conn is beam.conn
    # Did not open (and does not own) a private connection.
    assert vc._owns_connection is False
    # db_path is a real path, not a stringified Connection.
    assert not isinstance(vc.db_path, sqlite3.Connection)
    assert Path(vc.db_path) == Path(temp_db)


def test_init_does_not_create_phantom_connection_file(temp_db, tmp_path, monkeypatch):
    """The positional bug ran ``sqlite3.connect(str(connection))``, creating
    a stray file named after the Connection repr *in the cwd*. chdir into an
    isolated dir so that file (if any) lands here and is detectable."""
    monkeypatch.chdir(tmp_path)
    before = set(p.name for p in tmp_path.iterdir())
    BeamMemory(session_id="vc-phantom", db_path=temp_db)
    after = set(p.name for p in tmp_path.iterdir())
    stray = [n for n in (after - before) if "object at" in n]
    assert not stray, f"phantom connection-repr file(s) created: {stray}"


# --- Site B: BeamMemory.fact_recall (per-call consolidator) ----------------

def test_fact_recall_sees_consolidated_facts_via_shared_connection(temp_db):
    """fact_recall builds its own VeracityConsolidator. If it shares the
    beam connection, a fact consolidated on that connection is recalled; the
    positional regression would query an empty phantom DB and find nothing."""
    from mnemosyne.core.veracity_consolidation import VeracityConsolidator

    beam = BeamMemory(session_id="vc-recall", db_path=temp_db)

    # Write the fact through a consolidator explicitly bound to the beam
    # connection, so the fact lands in the *real* DB independent of site A's
    # state. (Using beam.veracity_consolidator would mask the bug: if both
    # sites regress they share the same phantom path and still rendezvous.)
    # Multiple mentions push confidence (1 - 0.7^n) above fact_recall's
    # hardcoded min_confidence=0.3 regardless of the comparison operator.
    writer = VeracityConsolidator(conn=beam.conn, db_path=beam.db_path)
    for i in range(3):
        writer.consolidate_fact(
            subject="Python",
            predicate="is",
            object="a programming language",
            veracity="confirmed",
            source=f"m{i}",
        )

    results = beam.fact_recall("python")

    hit = next(
        (r for r in results if r.get("subject") == "Python"
         and "programming language" in r.get("content", "")),
        None,
    )
    assert hit is not None, (
        "consolidated fact not surfaced by fact_recall -- the local "
        "consolidator is not sharing the beam connection"
    )
