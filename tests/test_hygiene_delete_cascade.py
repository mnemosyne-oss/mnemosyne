"""
Tests for the hygiene delete cascade (issue #960).

Pre-fix, ``clean_noise(delete)`` executed a bare
``DELETE FROM {table} WHERE id=?``, stranding annotation/embedding and
sqlite-vec rows (FTS was already trigger-maintained via em_ad/wm_ad, so
it needs no handling here). Post-fix, the delete cascades to
annotations/memory_embeddings/gists (by memory_id) and to the vec
mirror (by rowid, best-effort).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory, _vec_available, _vec_insert
from mnemosyne.core.hygiene import NoiseCandidate, clean_noise


@pytest.fixture
def temp_db(tmp_path: Path):
    """Beam-backed temp DB with a gists table for cascade coverage."""
    db_path = tmp_path / "cascade_test.db"
    beam = BeamMemory(session_id="cascade-test", db_path=db_path)
    beam.conn.execute(
        "CREATE TABLE IF NOT EXISTS gists (id TEXT PRIMARY KEY, text TEXT, memory_id TEXT)"
    )
    beam.conn.commit()
    yield db_path, beam
    beam.conn.close()


def _insert_row(beam, table, memory_id, content="noise content"):
    """Insert one base row; return its rowid for vec seeding."""
    beam.conn.execute(
        f"INSERT INTO {table} (id, content, source, timestamp, session_id, importance, metadata_json) "
        f"VALUES (?, ?, 'test', '2025-01-01T00:00:00', 'cascade-test', 0.5, '{{}}')",
        (memory_id, content),
    )
    beam.conn.commit()
    return beam.conn.execute(
        f"SELECT rowid FROM {table} WHERE id = ?", (memory_id,)
    ).fetchone()[0]


def _seed_side_rows(beam, memory_id, rowid=None, vec_table=None):
    """Attach annotation/embedding/gist rows (plus a vec row when asked)."""
    beam.conn.execute(
        "INSERT INTO annotations (memory_id, kind, value) VALUES (?, 'mentions', 'x')",
        (memory_id,),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, '[]')",
        (memory_id,),
    )
    beam.conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, 'gist text', ?)",
        (f"gist-{memory_id}", memory_id),
    )
    if rowid is not None and vec_table is not None:
        if vec_table == "vec_episodes":
            _vec_insert(beam.conn, rowid, [0.1] * 384)
        else:
            from mnemosyne.core.beam import _vec_table_insert
            _vec_table_insert(beam.conn, vec_table, rowid, [0.1] * 384)
    beam.conn.commit()


def _candidate(memory_id, table):
    """Build a delete-suggested candidate for a seeded row."""
    return NoiseCandidate(
        memory_id=memory_id, table_name=table,
        content_preview="noise", noise_score=0.9,
        noise_reasons=["test"], suggested_action="delete",
    )


def _counts(beam, memory_id):
    """Count the base row and its side rows (-1 for missing vec tables)."""
    conn = beam.conn
    base_ep = conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", (memory_id,)).fetchone()[0]
    base_wm = conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)).fetchone()[0]
    ann = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (memory_id,)).fetchone()[0]
    emb = conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)).fetchone()[0]
    try:
        vec_ep = conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
    except Exception:
        vec_ep = -1
    try:
        vec_wm = conn.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0]
    except Exception:
        vec_wm = -1
    return base_ep, base_wm, ann, emb, vec_ep, vec_wm


def _gist_count(beam, memory_id):
    """Count gists rows for a memory."""
    return beam.conn.execute(
        "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]


def test_episodic_delete_cascades_side_rows(temp_db):
    """Episodic delete removes base, annotations, embeddings, gists, vec."""
    db_path, beam = temp_db
    vec_ok = _vec_available(beam.conn)
    if not vec_ok:
        pytest.skip("sqlite-vec not available in this environment")
    rowid = _insert_row(beam, "episodic_memory", "em-1")
    _seed_side_rows(beam, "em-1", rowid, "vec_episodes")

    result = clean_noise(db_path, [_candidate("em-1", "episodic_memory")],
                         action="delete", confirm=True, dry_run=False)

    assert result.deleted == 1
    assert result.errors == []
    base_ep, _, ann, emb, vec_ep, _ = _counts(beam, "em-1")
    assert (base_ep, ann, emb, vec_ep) == (0, 0, 0, 0)
    assert _gist_count(beam, "em-1") == 0


def test_episodic_delete_cascades_without_vec(temp_db):
    """Annotations/embeddings must go even where no vec row exists."""
    db_path, beam = temp_db
    _insert_row(beam, "episodic_memory", "em-2")
    _seed_side_rows(beam, "em-2")

    result = clean_noise(db_path, [_candidate("em-2", "episodic_memory")],
                         action="delete", confirm=True, dry_run=False)

    assert result.deleted == 1
    assert result.errors == []
    base_ep, _, ann, emb, _, _ = _counts(beam, "em-2")
    assert (base_ep, ann, emb) == (0, 0, 0)


def test_working_delete_cascades_vec_working(temp_db):
    """Working delete removes base, annotations, embeddings, gists, vec."""
    db_path, beam = temp_db
    vec_ok = _vec_available(beam.conn)
    if not vec_ok:
        pytest.skip("sqlite-vec not available in this environment")
    rowid = _insert_row(beam, "working_memory", "wm-1")
    _seed_side_rows(beam, "wm-1", rowid, "vec_working")

    result = clean_noise(db_path, [_candidate("wm-1", "working_memory")],
                         action="delete", confirm=True, dry_run=False)

    assert result.deleted == 1
    assert result.errors == []
    _, base_wm, ann, emb, _, vec_wm = _counts(beam, "wm-1")
    assert (base_wm, ann, emb, vec_wm) == (0, 0, 0, 0)
    assert _gist_count(beam, "wm-1") == 0


def test_legacy_memories_delete_has_no_vec_mirror(temp_db):
    """The legacy table has no vec mirror: delete must simply succeed."""
    db_path, beam = temp_db
    beam.conn.execute(
        "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, content TEXT, "
        "source TEXT, timestamp TEXT, session_id TEXT, importance REAL, metadata_json TEXT)"
    )
    _insert_row(beam, "memories", "leg-1")
    _seed_side_rows(beam, "leg-1")

    result = clean_noise(db_path, [_candidate("leg-1", "memories")],
                         action="delete", confirm=True, dry_run=False)

    assert result.deleted == 1
    assert result.errors == []
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = 'leg-1'").fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = 'leg-1'").fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = 'leg-1'").fetchone()[0] == 0
    assert _gist_count(beam, "leg-1") == 0


def test_missing_vec_table_does_not_abort_delete(temp_db):
    """Vec cascade is best-effort: without the vec table the base delete stands."""
    db_path, beam = temp_db
    _insert_row(beam, "episodic_memory", "em-3")
    _seed_side_rows(beam, "em-3")
    beam.conn.execute("DROP TABLE IF EXISTS vec_episodes")
    beam.conn.commit()

    result = clean_noise(db_path, [_candidate("em-3", "episodic_memory")],
                         action="delete", confirm=True, dry_run=False)

    assert result.deleted == 1
    assert result.errors == []
    base_ep, _, ann, emb, _, _ = _counts(beam, "em-3")
    assert (base_ep, ann, emb) == (0, 0, 0)
    assert _gist_count(beam, "em-3") == 0


def test_missing_vec_table_probe_is_best_effort(temp_db, monkeypatch):
    """The vec-table probe is exercised even without sqlite-vec installed.

    clean_noise consults vec support once per run; forcing that answer to
    True while the table is absent drives the missing-table probe
    (SELECT 1 ... LIMIT 0 raising OperationalError, caught inside the
    cascade) in environments where sqlite-vec is unavailable — the one
    path the unmocked test above cannot reach there. Side-row cleanup
    assertions are retained unchanged.
    """
    import mnemosyne.core.hygiene as hygiene

    db_path, beam = temp_db
    _insert_row(beam, "episodic_memory", "em-3b")
    _seed_side_rows(beam, "em-3b")
    beam.conn.execute("DROP TABLE IF EXISTS vec_episodes")
    beam.conn.commit()

    real_ensure = hygiene._hygiene_ensure_vec
    consulted = []

    def _spy_ensure(conn):
        consulted.append(True)
        real_ensure(conn)
        return True

    monkeypatch.setattr(hygiene, "_hygiene_ensure_vec", _spy_ensure)

    result = clean_noise(db_path, [_candidate("em-3b", "episodic_memory")],
                         action="delete", confirm=True, dry_run=False)

    assert consulted == [True]
    assert result.deleted == 1
    assert result.errors == []
    base_ep, _, ann, emb, _, _ = _counts(beam, "em-3b")
    assert (base_ep, ann, emb) == (0, 0, 0)
    assert _gist_count(beam, "em-3b") == 0


def test_required_cascade_failure_rolls_back_base_delete(temp_db):
    """A failing side-row delete rolls back the base row and records an error."""
    db_path, beam = temp_db
    _insert_row(beam, "episodic_memory", "em-4")
    _seed_side_rows(beam, "em-4")
    beam.conn.execute(
        "CREATE TRIGGER fail_ann_delete BEFORE DELETE ON annotations "
        "BEGIN SELECT RAISE(ABORT, 'forced annotations failure'); END"
    )
    beam.conn.commit()

    result = clean_noise(db_path, [_candidate("em-4", "episodic_memory")],
                         action="delete", confirm=True, dry_run=False)

    assert result.deleted == 0
    assert len(result.errors) == 1
    base_ep, _, ann, emb, _, _ = _counts(beam, "em-4")
    assert (base_ep, ann, emb) == (1, 1, 1)
    assert _gist_count(beam, "em-4") == 1
