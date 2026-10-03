"""Tests for the sqlite-vec dimension-consistency guard in init_beam().

The dimension of a sqlite-vec ``vec0`` table is fixed at creation time. If a
database already stores vectors at one dimension and the process is later
configured (EMBEDDING_DIM) for a different one, creating a new ``vec0`` table at
that configured dimension is silently wrong: every insert of a real vector then
fails and recall reads an empty/incompatible index. init_beam() must detect the
established dimension from the database and refuse to create mismatched tables,
leaving existing tables untouched, rather than corrupting the store.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sqlite3

import pytest

from mnemosyne.core.memory import Mnemosyne
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


def test_init_beam_returns_frozen_fresh_status(tmp_path, monkeypatch):
    """Fresh databases report their configured dimension without a mismatch."""
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)

    result = beam.init_beam(Path(tmp_path) / "fresh-status.db")

    assert result == beam.BeamInitResult(False, None, 384, ())
    with pytest.raises(FrozenInstanceError):
        setattr(result, "configured_dim", 768)
    assert result.stored_dims == ()
    with pytest.raises(FrozenInstanceError):
        setattr(result, "stored_dims", (("vec_episodes", 768),))


def test_init_beam_skips_vec_creation_on_dim_mismatch(tmp_path, monkeypatch):
    """A mismatch preserves tables and skips later vector-table backfill/creation."""
    if not beam._SQLITE_VEC_AVAILABLE:
        import pytest  # type: ignore

        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "store.db"

    # Initialize the store at dimension 768.
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    initial = beam.init_beam(db)
    assert initial == beam.BeamInitResult(False, None, 768, ())
    conn = beam._get_connection(db)

    # Simulate a database written before vec_working and vec_facts existed.
    conn.execute("DROP TABLE IF EXISTS vec_working")
    conn.execute("DROP TABLE IF EXISTS vec_facts")
    conn.commit()
    assert beam._existing_vec_dim(conn) == 768

    # Reopen configured for a DIFFERENT dimension (the misconfiguration).
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)
    backfill_calls = []
    monkeypatch.setattr(
        beam,
        "_backfill_vec_working_from_memory_embeddings",
        lambda backfill_conn: backfill_calls.append(backfill_conn),
    )
    result = beam.init_beam(db)

    assert result == beam.BeamInitResult(True, 768, 384, (("vec_episodes", 768),))
    assert backfill_calls == []

    # The guard must NOT have created either missing table at the wrong dimension.
    assert conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_working'"
    ).fetchone() is None
    assert conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_facts'"
    ).fetchone() is None

    # The existing data table is left untouched at its real dimension.
    ep = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_episodes'"
    ).fetchone()
    assert ep is not None and "[768]" in ep[0]


def test_init_beam_reports_mismatch_when_sqlite_vec_is_unavailable(tmp_path, monkeypatch):
    """Status reads an existing vec0 dimension even without sqlite-vec available."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "unavailable-status.db"
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    beam.init_beam(db)
    conn = beam._get_connection(db)
    conn.execute("DROP TABLE IF EXISTS vec_working")
    conn.execute("DROP TABLE IF EXISTS vec_facts")
    conn.commit()

    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)
    monkeypatch.setattr(beam, "_SQLITE_VEC_AVAILABLE", False)
    backfill_calls = []
    monkeypatch.setattr(
        beam,
        "_backfill_vec_working_from_memory_embeddings",
        lambda backfill_conn: backfill_calls.append(backfill_conn),
    )

    result = beam.init_beam(db)

    assert result == beam.BeamInitResult(True, 768, 384, (("vec_episodes", 768),))
    assert backfill_calls == []
    assert conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_working'"
    ).fetchone() is None
    assert conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_facts'"
    ).fetchone() is None


def test_init_beam_creates_vec_tables_when_dim_matches(tmp_path, monkeypatch):
    """A fresh configured database creates vec tables normally."""
    if not beam._SQLITE_VEC_AVAILABLE:
        import pytest  # type: ignore

        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "match.db"
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    result = beam.init_beam(db)
    conn = beam._get_connection(db)

    working = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_working'"
    ).fetchone()
    assert working is not None and "[768]" in working[0]
    assert result == beam.BeamInitResult(False, None, 768, ())


def test_init_beam_reports_match_and_constructor_preserves_status(tmp_path, monkeypatch):
    """Existing matching vec tables and normal construction expose status."""
    if not beam._SQLITE_VEC_AVAILABLE:
        import pytest  # type: ignore

        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "constructor-status.db"
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    beam.init_beam(db)  # Ignoring the new result remains compatible.

    result = beam.init_beam(db)
    memory = beam.BeamMemory(session_id="status", db_path=db)

    assert result == beam.BeamInitResult(
        False,
        768,
        768,
        (("vec_episodes", 768), ("vec_working", 768), ("vec_facts", 768)),
    )
    assert memory.init_result == result


def test_mnemosyne_constructor_exposes_beam_init_status(tmp_path, monkeypatch):
    """The regular Mnemosyne API exposes the status produced by its BeamMemory."""
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)

    memory = Mnemosyne(session_id="status", db_path=tmp_path / "mnemosyne.db")

    assert memory.init_result == memory.beam.init_result
    expected_existing_dim = 384 if beam._SQLITE_VEC_AVAILABLE else None
    expected_stored_dims = (
        (("vec_episodes", 384), ("vec_working", 384), ("vec_facts", 384))
        if beam._SQLITE_VEC_AVAILABLE
        else ()
    )
    assert memory.init_result == beam.BeamInitResult(
        False, expected_existing_dim, 384, expected_stored_dims
    )


def test_ignored_mismatch_result_recovers_after_matching_reinit(tmp_path, monkeypatch):
    """Ignoring a mismatch result does not block recovery after configuration repair."""
    if not beam._SQLITE_VEC_AVAILABLE:
        import pytest  # type: ignore

        pytest.skip("sqlite-vec unavailable")

    db = Path(tmp_path) / "recovery.db"
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    beam.init_beam(db)

    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)
    beam.init_beam(db)  # Historical ignored-return call pattern.

    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    recovered = beam.init_beam(db)
    assert recovered == beam.BeamInitResult(
        False,
        768,
        768,
        (("vec_episodes", 768), ("vec_working", 768), ("vec_facts", 768)),
    )


def test_dim_mismatch_message_is_self_healing():
    """The mismatch message says it is not corruption and gives recovery commands."""
    msg = beam._dim_mismatch_message((("vec_episodes", 768),), configured_dim=384)
    low = msg.lower()
    assert "not database corruption" in low, msg
    assert "mnemosyne reindex" in low, msg
    assert "mnemosyne_embedding_dim=" in low, msg
    assert "768" in msg and "384" in msg


@pytest.mark.parametrize(
    "creation_order",
    [
        ("vec_episodes", "vec_working", "vec_facts"),
        ("vec_facts", "vec_working", "vec_episodes"),
    ],
)
def test_init_beam_reports_mixed_stored_dims_without_sqlite_vec(
    tmp_path, monkeypatch, caplog, creation_order
):
    """Mixed legacy indexes are order-independent and never gain new vec tables."""
    db = Path(tmp_path) / "mixed-no-extension.db"
    conn = beam._get_connection(db)
    dimensions = {"vec_episodes": 768, "vec_working": 384, "vec_facts": 384}
    for table in creation_order:
        conn.execute(f"CREATE TABLE {table} (embedding int8[{dimensions[table]}])")
    conn.commit()
    before = dict(
        conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE name IN "
            "('vec_episodes', 'vec_working', 'vec_facts')"
        )
    )

    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)
    monkeypatch.setattr(beam, "_SQLITE_VEC_AVAILABLE", False)
    backfill_calls = []
    monkeypatch.setattr(
        beam,
        "_backfill_vec_working_from_memory_embeddings",
        lambda backfill_conn: backfill_calls.append(backfill_conn),
    )

    result = beam.init_beam(db)

    assert result == beam.BeamInitResult(
        True,
        None,
        384,
        (("vec_episodes", 768), ("vec_working", 384), ("vec_facts", 384)),
    )
    assert backfill_calls == []
    assert dict(
        conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE name IN "
            "('vec_episodes', 'vec_working', 'vec_facts')"
        )
    ) == before
    assert "mixed vector-index dimensions" in caplog.text
    assert "vec_episodes=768, vec_working=384, vec_facts=384" in caplog.text


def test_legacy_vec_episodes_rejects_configured_dimension_query(tmp_path):
    """The mixed-state status protects the real sqlite-vec failure boundary."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    conn = beam._get_connection(Path(tmp_path) / "legacy-query.db")
    conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[768])")

    with pytest.raises(sqlite3.OperationalError, match="[Dd]imension mismatch"):
        conn.execute(
            "SELECT rowid FROM vec_episodes "
            "WHERE embedding MATCH vec_quantize_int8(?, 'unit') AND k=1",
            (json.dumps([0.0] * 384),),
        ).fetchall()
