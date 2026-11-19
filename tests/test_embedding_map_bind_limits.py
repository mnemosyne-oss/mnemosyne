"""Stored-vector hydration must not lose evidence at SQLite binding boundaries."""

import json
import sqlite3

import numpy as np
import pytest

from mnemosyne.core import beam as core
from mnemosyne.core.beam import BeamMemory


class _BudgetCursor:
    """Execute real SQLite statements, rejecting only an excessive bind count."""

    def __init__(self, cursor, connection):
        self._cursor = cursor
        self._connection = connection

    def execute(self, sql, params=()):
        self._connection.check(params)
        self._cursor.execute(sql, params)
        return self

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _LegacyBudgetConnection:
    """Model the pre-3.11 API without inventing SQL results or transactions."""

    def __init__(self, connection):
        self._connection = connection
        self.bind_counts = []

    def check(self, params):
        self.bind_counts.append(len(params))
        if len(params) > 999:
            raise sqlite3.OperationalError("test SQLite budget exceeded: more than 999 binds")

    def cursor(self, *args, **kwargs):
        return _BudgetCursor(self._connection.cursor(*args, **kwargs), self)

    def execute(self, sql, params=()):
        return self.cursor().execute(sql, params)

    def __getattr__(self, name):
        if name in {"getlimit", "setlimit"}:
            raise AttributeError(name)
        return getattr(self._connection, name)


def _make_beam(tmp_path, source, *, vec_type="float"):
    """Create the actual vector table for fallback; model inference is unnecessary."""
    beam = BeamMemory(session_id="vectors", db_path=tmp_path / "vectors.db")
    if source == "episodic":
        sqlite_vec = pytest.importorskip("sqlite_vec")
        beam.conn.enable_load_extension(True)
        try:
            sqlite_vec.load(beam.conn)
        finally:
            beam.conn.enable_load_extension(False)
        beam.conn.execute("DROP TABLE IF EXISTS vec_episodes")
        beam.conn.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding {vec_type}[8])")
        assert core._vec_available(beam.conn)
    return beam


def _seed(beam, ids, source, *, vec_type="float", vectors=None):
    """Store known vectors in real parent rows, with no embedding generation."""
    default_vector = [1.0] * 8
    vectors = vectors or {}
    if source == "json":
        beam.conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, 'stored vector original', 'conversation', '2020-01-01', 'vectors')",
            [(mid,) for mid in ids],
        )
        beam.conn.executemany(
            "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
            [(mid, json.dumps(vectors.get(mid, default_vector))) for mid in ids],
        )
    else:
        expression = {"float": "?", "int8": "vec_int8(?)", "bit": "vec_quantize_binary(?)"}[vec_type]
        for mid in ids:
            cursor = beam.conn.execute(
                "INSERT INTO episodic_memory (id, content, source, timestamp, session_id) "
                "VALUES (?, 'stored episodic vector', 'sleep_consolidation', '2020-01-01', 'vectors')",
                (mid,),
            )
            beam.conn.execute(
                f"INSERT INTO vec_episodes (rowid, embedding) VALUES (?, {expression})",
                (cursor.lastrowid, bytes([127] * 8) if vec_type == "int8" else json.dumps(vectors.get(mid, default_vector))),
            )
    beam.conn.commit()


def _apply_budget(beam, api, limit=999):
    """Native and portable budgets run independently; only native tests can skip."""
    if api == "native":
        if not hasattr(beam.conn, "setlimit"):
            pytest.skip("native SQLite limit API requires Python 3.11+")
        beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    else:
        assert limit == 999
        beam.conn = _LegacyBudgetConnection(beam.conn)
        assert not hasattr(beam.conn, "getlimit") and not hasattr(beam.conn, "setlimit")


@pytest.mark.parametrize("source", ["json", "episodic"])
@pytest.mark.parametrize("api", ["native", "legacy"])
@pytest.mark.parametrize("count", [999, 1000, 1801])
def test_embedding_map_keeps_all_vectors_across_bind_boundary(tmp_path, source, api, count):
    """Unbounded primary, episodic-ID or vector-rowid SQL loses these stored vectors."""
    beam = _make_beam(tmp_path, source)
    ids = [f"vector-{i:04}" for i in range(count)]
    expected = {mid: [i + 1, -i, 0.5, -0.5, 1, 2, 3, 4] for i, mid in enumerate(ids)}
    _seed(beam, ids, source, vectors=expected)
    _apply_budget(beam, api)
    changes = beam.conn.total_changes
    result = beam._embedding_map(ids)
    assert set(result) == set(ids)
    for mid, vector in result.items():
        np.testing.assert_array_equal(vector, np.array(expected[mid], dtype=np.float32))
        assert vector.dtype == np.float32
    assert beam.conn.total_changes == changes
    if api == "legacy":
        assert beam.conn.bind_counts and max(beam.conn.bind_counts) <= 999


@pytest.mark.parametrize("source", ["json", "episodic"])
def test_embedding_map_respects_small_live_limit(tmp_path, source):
    """A hard-coded999 batch must not override a connection's live31-variable budget."""
    beam = _make_beam(tmp_path, source)
    ids = [f"live-{i}" for i in range(32)]
    _seed(beam, ids, source)
    _apply_budget(beam, "native", 31)
    result = beam._embedding_map(ids)
    assert set(result) == set(ids)
    assert all(np.array_equal(v, np.ones(8, dtype=np.float32)) for v in result.values())


@pytest.mark.parametrize("source", ["json", "episodic"])
@pytest.mark.parametrize("api", ["native", "legacy"])
def test_conflict_pair_survives_padding_beyond_bind_limit(tmp_path, source, api):
    """Missing vector evidence must not turn a real conflict into a false no-op."""
    beam = _make_beam(tmp_path, source)
    _seed(beam, ["older", "newer"], source)
    rows = [
        {"id": "older", "timestamp": "2020-01-01T00:00:00", "content": "favorite color is blue for the car", "superseded_by": None},
        {"id": "newer", "timestamp": "2020-01-02T00:00:00", "content": "my favorite color choice is green grass everywhere", "superseded_by": None},
    ]
    padding = [
        {"id": f"missing-{i}", "timestamp": "2020-01-03T00:00:00", "content": "unrelated stored text", "superseded_by": None}
        for i in range(1000)
    ]
    _apply_budget(beam, api)
    changes = beam.conn.total_changes
    assert beam._detect_conflicts(rows) == [("older", "newer")]
    assert beam._detect_conflicts(rows + padding) == [("older", "newer")]
    assert beam.conn.total_changes == changes


@pytest.mark.parametrize("vec_type,expected", [("int8", 127.0), ("bit", 1.0)])
@pytest.mark.parametrize("api", ["native", "legacy"])
def test_quantized_fallback_keeps_decode_across_batches(tmp_path, vec_type, expected, api):
    """Batching must preserve the actual table's int8/bit decoding contract."""
    beam = _make_beam(tmp_path, "episodic", vec_type=vec_type)
    ids = [f"quantized-{i}" for i in range(1000)]
    _seed(beam, ids, "episodic", vec_type=vec_type)
    _apply_budget(beam, api)
    result = beam._embedding_map(ids)
    assert set(result) == set(ids)
    for vector in result.values():
        np.testing.assert_array_equal(vector, np.full(8, expected, dtype=np.float32))
        assert vector.dtype == np.float32


def test_mixed_sources_preserve_precedence_decode_and_missing_ids(tmp_path):
    """Malformed JSON falls back per ID; duplicates/missing IDs do not replace primary evidence."""
    beam = _make_beam(tmp_path, "episodic")
    episodic_ids = [f"episode-{i:04}" for i in range(1001)]
    _seed(beam, episodic_ids, "episodic")
    primary = [0.5, -0.25, 0.125, -0.0625, 0.03125, 0.0, -0.5, 0.25]
    beam.conn.execute("INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)", (episodic_ids[0], json.dumps(primary)))
    beam.conn.execute("INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)", (episodic_ids[-1], "not-json"))
    beam.conn.commit()
    _apply_budget(beam, "legacy")
    changes = beam.conn.total_changes
    result = beam._embedding_map(episodic_ids + [episodic_ids[0], episodic_ids[-1], "absent"])
    assert set(result) == set(episodic_ids)
    np.testing.assert_array_equal(result[episodic_ids[0]], np.array(primary, dtype=np.float32))
    for mid in episodic_ids[1:]:
        np.testing.assert_array_equal(result[mid], np.ones(8, dtype=np.float32))
    assert beam.conn.total_changes == changes
    assert max(beam.conn.bind_counts) <= 999


def test_empty_input_never_queries_connection(tmp_path):
    """The empty-input fast path remains a read-free empty mapping."""
    beam = _make_beam(tmp_path, "json")
    _apply_budget(beam, "legacy")
    assert beam._embedding_map([]) == {}
    assert beam.conn.bind_counts == []


def test_sql_failure_keeps_existing_optional_primary_fallback(tmp_path):
    """An actual SQLite read denial stays best-effort and permits episodic evidence."""
    beam = _make_beam(tmp_path, "episodic")
    _seed(beam, ["fallback"], "episodic")
    denied = []
    def deny_primary_read(action, table, _column, _database, _trigger):
        if action == sqlite3.SQLITE_READ and table == "memory_embeddings":
            denied.append(table)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    beam.conn.set_authorizer(deny_primary_read)
    try:
        result = beam._embedding_map(["fallback"])
    finally:
        beam.conn.set_authorizer(None)
    assert denied, "the primary lookup must encounter the real SQLite denial"
    assert set(result) == {"fallback"}
    np.testing.assert_array_equal(result["fallback"], np.ones(8, dtype=np.float32))


@pytest.mark.parametrize("primary", ["older", "newer"])
def test_mixed_source_conflict_crosses_batch_boundary(tmp_path, primary):
    """A mixed JSON/fallback pair must survive on opposite sides of a900-ID batch."""
    beam = _make_beam(tmp_path, "episodic")
    fallback = "newer" if primary == "older" else "older"
    _seed(beam, [primary], "json")
    _seed(beam, [fallback], "episodic")
    older = {"id": "older", "timestamp": "2020-01-01T00:00:00", "content": "favorite color is blue for the car", "superseded_by": None}
    newer = {"id": "newer", "timestamp": "2020-01-02T00:00:00", "content": "my favorite color choice is green grass everywhere", "superseded_by": None}
    padding = [
        {"id": f"gap-{i}", "timestamp": "2020-01-01T12:00:00", "content": "unrelated text", "superseded_by": None}
        for i in range(1000)
    ]
    rows = [older, *padding, newer]
    _apply_budget(beam, "legacy")
    changes = beam.conn.total_changes
    assert beam._detect_conflicts([older, newer]) == [("older", "newer")]
    assert beam._detect_conflicts(rows) == [("older", "newer")]
    result = beam._embedding_map([r["id"] for r in rows])
    assert set(result) == {"older", "newer"}
    np.testing.assert_array_equal(result["older"], np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(result["newer"], np.ones(8, dtype=np.float32))
    assert beam.conn.total_changes == changes
    assert max(beam.conn.bind_counts) <= 999
