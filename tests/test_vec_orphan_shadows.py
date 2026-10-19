"""Regression coverage for orphaned sqlite-vec shadow tables (#1120).

A ``.recover`` pass or a partial restore can leave a vec0 table's shadow tables
without the virtual table itself. ``CREATE VIRTUAL TABLE ... USING vec0`` then
fails with a shadow-table collision, which aborted startup from 4.0.0b4 while
3.15.1 continued. Initialization must leave the orphaned payload untouched,
continue without that dense tier, and still propagate unrelated DDL errors.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

import mnemosyne.core.beam as beam


class _RecordingCursor(sqlite3.Cursor):
    """Record vec0 creations and pretend they succeeded (no extension needed)."""

    def execute(self, sql, parameters=()):
        normalized = " ".join(str(sql).split())
        if normalized.startswith("CREATE VIRTUAL TABLE") and "USING vec0" in normalized:
            self.connection.vec_creates.append(normalized)
            return self
        return super().execute(sql, parameters)


class _RecordingConnection(sqlite3.Connection):
    def cursor(self, *args, **kwargs):
        kwargs["factory"] = _RecordingCursor
        return super().cursor(*args, **kwargs)


class _FaultCursor(_RecordingCursor):
    """Fail the episodes DDL the way a disk or lock error would."""

    def execute(self, sql, parameters=()):
        normalized = " ".join(str(sql).split())
        if normalized.startswith("CREATE VIRTUAL TABLE") and "vec_episodes" in normalized:
            raise sqlite3.OperationalError("disk I/O error")
        return super().execute(sql, parameters)


class _FaultConnection(sqlite3.Connection):
    def cursor(self, *args, **kwargs):
        kwargs["factory"] = _FaultCursor
        return super().cursor(*args, **kwargs)


def _connect(path: Path, factory=_RecordingConnection):
    conn = sqlite3.connect(path, factory=factory)
    conn.vec_creates = []
    return conn


def _add_shadow_tables(conn: sqlite3.Connection, tiers, populate: bool = False) -> None:
    for tier in tiers:
        for suffix in ("info", "chunks"):
            conn.execute(f"CREATE TABLE {tier}_{suffix} (payload BLOB)")
            if populate:
                conn.execute(f"INSERT INTO {tier}_{suffix} (payload) VALUES (?)", (b"kept",))
    conn.commit()


def _shadow_ddl(conn: sqlite3.Connection) -> dict:
    return {
        name: sql
        for name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name LIKE 'vec_%'"
        )
    }


def _prepare(monkeypatch, conn) -> None:
    monkeypatch.setattr(beam, "_SQLITE_VEC_AVAILABLE", True)
    monkeypatch.setattr(beam, "_detect_vec_type", lambda _conn: "float32")
    monkeypatch.setattr(beam, "_get_connection", lambda _path: conn)


def test_all_orphaned_tiers_degrade_without_mutation(tmp_path, monkeypatch, caplog):
    path = tmp_path / "orphaned.db"
    conn = _connect(path)
    _add_shadow_tables(conn, beam._VEC_TABLE_NAMES, populate=True)
    before = _shadow_ddl(conn)
    _prepare(monkeypatch, conn)

    with caplog.at_level(logging.ERROR):
        result = beam._init_beam_locked(path)

    assert result.orphaned_vec_tiers == ("vec_episodes", "vec_working", "vec_facts")
    assert conn.vec_creates == []
    assert _shadow_ddl(conn) == before
    for tier in beam._VEC_TABLE_NAMES:
        assert conn.execute(f"SELECT COUNT(*) FROM {tier}_info").fetchone()[0] == 1
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "fts_episodes" in names, "initialization must continue past the orphaned tiers"
    assert any(
        "Orphaned sqlite-vec shadow tables" in record.getMessage()
        for record in caplog.records
    )
    conn.close()


def test_only_the_orphaned_tier_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "facts-only.db"
    conn = _connect(path)
    _add_shadow_tables(conn, ("vec_facts",))
    _prepare(monkeypatch, conn)

    result = beam._init_beam_locked(path)

    assert result.orphaned_vec_tiers == ("vec_facts",)
    assert any("vec_episodes" in sql for sql in conn.vec_creates)
    assert any("vec_working" in sql for sql in conn.vec_creates)
    assert not any("vec_facts" in sql for sql in conn.vec_creates)
    conn.close()


def test_healthy_store_still_creates_every_tier(tmp_path, monkeypatch):
    path = tmp_path / "healthy.db"
    conn = _connect(path)
    _prepare(monkeypatch, conn)

    result = beam._init_beam_locked(path)

    assert result.orphaned_vec_tiers == ()
    assert len(conn.vec_creates) == 3
    conn.close()


def test_unrelated_ddl_error_still_propagates_with_an_orphan(tmp_path, monkeypatch):
    path = tmp_path / "faulty.db"
    conn = _connect(path, factory=_FaultConnection)
    _add_shadow_tables(conn, ("vec_facts",))
    _prepare(monkeypatch, conn)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        beam._init_beam_locked(path)
    conn.close()


def test_detection_requires_a_missing_parent_and_a_surviving_shadow(tmp_path):
    conn = sqlite3.connect(tmp_path / "detect.db")

    assert beam._orphaned_vec_shadow_tiers(conn) == ()
    conn.execute("CREATE TABLE vec_episodes_info (payload BLOB)")
    assert beam._orphaned_vec_shadow_tiers(conn) == ("vec_episodes",)
    conn.execute("CREATE TABLE vec_episodes (embedding BLOB)")
    assert beam._orphaned_vec_shadow_tiers(conn) == ()
    conn.close()
