"""Tests for query-side vec0 dimension-mismatch degradation in _vec_search().

The write path already degrades: ``_wm_vec_upsert`` failures are logged and
the vector is dropped (``vec_working upsert failed ...``). The working-memory
query path degrades too: ``_wm_vec_search_sqlite`` wraps its KNN in
try/except and returns ``[]``. ``_vec_search`` (the episodic KNN over
``vec_episodes``) was the one unprotected path: a query embedding whose
dimension disagreed with the table's raised ``sqlite3.OperationalError``
straight out of ``recall()``, crashing the agent process. It must degrade the
same way, logging an actionable pointer to the existing mismatch guidance.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import mnemosyne.core.beam as beam


def test_vec_search_degrades_on_dimension_mismatch(tmp_path, monkeypatch, caplog):
    """A query vector at the wrong dimension must return [] (vector voice
    disabled for that call), not raise out of recall()."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "store.db"
    # Create the store at 768: vec_episodes is dimensioned at EMBEDDING_DIM.
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    beam.init_beam(db)

    # A 384-dim query against the 768-dim table previously raised
    # "Dimension mismatch for query vector" straight through recall().
    with caplog.at_level("WARNING", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(beam._get_connection(db), [0.01] * 384, k=5)
    assert rows == []
    assert any(
        "dimension mismatch" in r.message.lower() and "vec_episodes" in r.message.lower()
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_vec_search_unaffected_when_dimension_matches(tmp_path, monkeypatch):
    """The guard must not weaken the healthy path: a matching-dim query still
    returns rows (and an empty vec_episodes still returns [])."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "match.db"
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    beam.init_beam(db)

    # No vectors stored: healthy query returns [] without any warning.
    rows = beam._vec_search(beam._get_connection(db), [0.01] * 768, k=5)
    assert rows == []
