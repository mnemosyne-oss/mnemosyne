"""Regression coverage for the trim-before-embedding parent race."""

import json
import logging
import sqlite3

from mnemosyne.core import beam


def _minimal_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE working_memory (
            id TEXT PRIMARY KEY
        );
        CREATE TABLE memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    return conn


def test_embedding_store_skips_parent_removed_by_trim(monkeypatch):
    conn = _minimal_connection()
    vec_calls = []
    monkeypatch.setattr(beam, "np", None)
    monkeypatch.setattr(beam._embeddings, "serialize", json.dumps)
    monkeypatch.setattr(
        beam, "_wm_vec_upsert", lambda *args, **_kwargs: vec_calls.append(args)
    )

    beam._store_working_embedding(conn, "trimmed", [0.1, 0.2])

    assert conn.execute("SELECT count(*) FROM memory_embeddings").fetchone()[0] == 0
    assert vec_calls == []
    assert conn.execute("PRAGMA foreign_key_list(memory_embeddings)").fetchall() == []


def test_embedding_store_preserves_present_parent_and_updates(monkeypatch):
    conn = _minimal_connection()
    conn.execute("INSERT INTO working_memory(id) VALUES ('live')")
    vec_calls = []
    monkeypatch.setattr(beam, "np", None)
    monkeypatch.setattr(beam._embeddings, "serialize", json.dumps)
    monkeypatch.setattr(
        beam, "_wm_vec_upsert", lambda *args, **_kwargs: vec_calls.append(args)
    )

    beam._store_working_embedding(conn, "live", [0.1, 0.2])
    beam._store_working_embedding(conn, "live", [0.3, 0.4])

    row = conn.execute(
        "SELECT embedding_json, model FROM memory_embeddings WHERE memory_id='live'"
    ).fetchone()
    assert json.loads(row[0]) == [0.3, 0.4]
    assert row[1] == beam._embeddings._DEFAULT_MODEL
    assert [call[1:] for call in vec_calls] == [
        ("live", [0.1, 0.2]),
        ("live", [0.3, 0.4]),
    ]


def test_remember_trim_before_embedding_is_quiet(tmp_path, monkeypatch, caplog):
    memory = beam.BeamMemory(session_id="trim-race", db_path=tmp_path / "mnemosyne.db")
    vec_calls = []
    monkeypatch.setattr(beam, "np", None)
    monkeypatch.setattr(beam._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam._embeddings, "embed", lambda _values: [[0.1, 0.2]])
    monkeypatch.setattr(beam._embeddings, "serialize", json.dumps)
    monkeypatch.setattr(
        beam, "_wm_vec_upsert", lambda *args, **_kwargs: vec_calls.append(args)
    )

    def delete_new_parent() -> None:
        memory.conn.execute(
            "DELETE FROM working_memory WHERE session_id = ?", (memory.session_id,)
        )
        memory.conn.commit()

    monkeypatch.setattr(memory, "_trim_working_memory", delete_new_parent)
    with caplog.at_level(logging.WARNING):
        memory_id = memory.remember("deterministic trim-before-embedding regression")

    assert memory.conn.execute(
        "SELECT count(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert memory.conn.execute(
        "SELECT count(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert memory.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert vec_calls == []
    assert "embedding storage failed" not in caplog.text