"""Regression tests for get_episodic_stats vector-count fallback.

Pre-fix: BeamMemory.get_episodic_stats only counted the sqlite-vec ANN
table (``vec_episodes``). When the running Python's ``sqlite3`` module is
built without loadable-extension support, sqlite-vec can never load and
``vec_episodes`` never exists, so the reported ``vectors`` count is always
0 -- even though semantic recall is fully functional via the binary-vector
voice (``episodic_memory.binary_vector``) or the float32 JSON embeddings
(``memory_embeddings``). The bogus 0 also tripped a false
"episodic vectors=0, run sleep" hint in diagnose.

Post-fix: when vec_episodes is unavailable/empty, the stat falls back to
counting whichever representation is actually populated, and reports a
truthful ``vec_type`` ("binary" or "json") instead of "none".

These tests deliberately seed the fallback tables directly rather than
relying on fastembed/sqlite-vec being installed, so they pin the fallback
logic deterministically in any environment (mirroring the CI/macOS
no-extension case).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "mnemosyne_epstats.db"


def _add_episodic(beam: BeamMemory, mem_id: str) -> None:
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES (?, ?, 'test', datetime('now'), 0.5)",
        (mem_id, f"content for {mem_id}"),
    )
    beam.conn.commit()


def test_no_vectors_reports_none(temp_db):
    """Episodic rows with no vector representation report 0 / 'none'."""
    beam = BeamMemory(session_id="epstats", db_path=temp_db)
    _add_episodic(beam, "em-a")

    stats = beam.get_episodic_stats()
    assert stats["total"] == 1
    assert stats["vectors"] == 0
    assert stats["vec_type"] == "none"


def test_binary_vector_counted_when_ann_absent(temp_db):
    """When vec_episodes is absent, populated binary_vector columns are
    counted and reported as the 'binary' backend."""
    beam = BeamMemory(session_id="epstats", db_path=temp_db)
    _add_episodic(beam, "em-a")
    _add_episodic(beam, "em-b")
    # Only em-a carries a binary vector.
    beam.conn.execute(
        "UPDATE episodic_memory SET binary_vector = ? WHERE id = 'em-a'",
        (b"\x00\x01\x02\x03",),
    )
    beam.conn.commit()

    stats = beam.get_episodic_stats()
    assert stats["total"] == 2
    assert stats["vectors"] == 1
    assert stats["vec_type"] == "binary"


def test_json_embeddings_counted_when_no_binary(temp_db):
    """With no binary vectors, JSON embeddings in memory_embeddings are
    counted and reported as the 'json' backend."""
    beam = BeamMemory(session_id="epstats", db_path=temp_db)
    _add_episodic(beam, "em-a")
    vec = np.ones(8, dtype=np.float32)
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
        "VALUES (?, ?, ?)",
        ("em-a", json.dumps(vec.tolist()), "test-model"),
    )
    beam.conn.commit()

    stats = beam.get_episodic_stats()
    assert stats["total"] == 1
    assert stats["vectors"] == 1
    assert stats["vec_type"] == "json"


def test_binary_preferred_over_json(temp_db):
    """Binary vectors take precedence over JSON embeddings in the fallback
    chain (binary is the cheaper recall voice)."""
    beam = BeamMemory(session_id="epstats", db_path=temp_db)
    _add_episodic(beam, "em-a")
    beam.conn.execute(
        "UPDATE episodic_memory SET binary_vector = ? WHERE id = 'em-a'",
        (b"\x00\x01",),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
        "VALUES (?, ?, ?)",
        ("em-a", json.dumps([1.0, 2.0]), "test-model"),
    )
    beam.conn.commit()

    stats = beam.get_episodic_stats()
    assert stats["vec_type"] == "binary"


def test_fallback_respects_author_filter(temp_db):
    """The fallback count honors the same author/channel filter as the
    total count, so a filtered view doesn't over-report vectors."""
    beam = BeamMemory(session_id="epstats", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance, author_id) "
        "VALUES ('em-a', 'a', 'test', datetime('now'), 0.5, 'alice')"
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance, author_id) "
        "VALUES ('em-b', 'b', 'test', datetime('now'), 0.5, 'bob')"
    )
    for mid in ("em-a", "em-b"):
        beam.conn.execute(
            "UPDATE episodic_memory SET binary_vector = ? WHERE id = ?",
            (b"\x00\x01", mid),
        )
    beam.conn.commit()

    stats = beam.get_episodic_stats(author_id="alice")
    assert stats["total"] == 1
    assert stats["vectors"] == 1
    assert stats["vec_type"] == "binary"
