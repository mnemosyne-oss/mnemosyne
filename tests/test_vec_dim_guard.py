"""Tests for the sqlite-vec dimension-consistency guard in init_beam().

The dimension of a sqlite-vec ``vec0`` table is fixed at creation time. If a
database already stores vectors at one dimension and the process is later
configured (EMBEDDING_DIM) for a different one, creating a new ``vec0`` table at
the configured dimension is silently wrong: every insert of a real vector then
fails and recall reads an empty/incompatible index. init_beam() must detect the
established dimension from the database and refuse to create mismatched tables,
leaving existing tables untouched, rather than corrupting the store.
"""
from __future__ import annotations

from pathlib import Path

import mnemosyne.core.beam as beam


def test_existing_vec_dim_none_on_fresh_db(tmp_path):
    """A database with no vec0 tables has no established dimension."""
    conn = beam._get_connection(Path(tmp_path) / "fresh.db")
    assert beam._existing_vec_dim(conn) is None


def test_existing_vec_dim_reads_declared_dimension(tmp_path):
    """The helper reads the dimension declared in a vec0 table's DDL."""
    if not beam._SQLITE_VEC_AVAILABLE:
        import pytest  # type: ignore

        pytest.skip("sqlite-vec unavailable")
    conn = beam._get_connection(Path(tmp_path) / "declared.db")
    conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[768])")
    conn.commit()
    assert beam._existing_vec_dim(conn) == 768


def test_init_beam_skips_vec_creation_on_dim_mismatch(tmp_path, monkeypatch):
    """init_beam() must not create vec0 tables at a dimension that disagrees with
    vectors already stored in the database, and must leave existing tables alone."""
    if not beam._SQLITE_VEC_AVAILABLE:
        import pytest  # type: ignore

        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "store.db"

    # Initialize the store at dimension 768.
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    beam.init_beam(db)
    conn = beam._get_connection(db)

    # Simulate a database written before vec_working existed: only vec_episodes
    # (at the real dimension, 768) remains.
    conn.execute("DROP TABLE IF EXISTS vec_working")
    conn.commit()
    assert beam._existing_vec_dim(conn) == 768

    # Reopen configured for a DIFFERENT dimension (the misconfiguration).
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)
    beam.init_beam(db)

    # The guard must NOT have created vec_working at the wrong dimension.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_working'"
    ).fetchone()
    assert row is None or "[384]" not in row[0]

    # The existing data table is left untouched at its real dimension.
    ep = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_episodes'"
    ).fetchone()
    assert ep is not None and "[768]" in ep[0]


def test_init_beam_creates_vec_tables_when_dim_matches(tmp_path, monkeypatch):
    """When the configured dimension matches (or the DB is fresh), the vec0 tables
    are created normally -- the guard must not be a false positive."""
    if not beam._SQLITE_VEC_AVAILABLE:
        import pytest  # type: ignore

        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "match.db"
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    beam.init_beam(db)
    conn = beam._get_connection(db)

    working = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_working'"
    ).fetchone()
    assert working is not None and "[768]" in working[0]
