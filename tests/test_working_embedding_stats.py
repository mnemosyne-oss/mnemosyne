"""Working stats are filtered storage presence, not embedding health (#1103)."""
import hashlib
import itertools
import json
import sqlite3

import pytest

from mnemosyne.core import beam as bm, embeddings as em
from mnemosyne.core.config import MnemosyneConfig, get_config


NEW_KEYS = {"embedding_rows", "ann_indexed_rows", "ann_index_available"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(em, "available", lambda: False)
    MnemosyneConfig.reset_instance()
    return bm.BeamMemory(db_path=tmp_path / "stats.db", session_id="synthetic")


def add_row(store, mid, *, author="alpha", kind="human", channel="red", embedded=True,
            indexed=True, consolidated=False, pinned=False):
    store.conn.execute(
        "INSERT INTO working_memory(id, content, source, timestamp, session_id, "
        "author_id, author_type, channel_id, consolidated_at, pinned) "
        "VALUES (?, ?, 'test', '2020-01-01T00:00:00', 'synthetic', ?, ?, ?, ?, ?)",
        (mid, "Synthetic fixture " + mid, author, kind, channel,
         "2020-01-02T00:00:00" if consolidated else None, int(pinned)),
    )
    if embedded:
        store.conn.execute(
            "INSERT INTO memory_embeddings(memory_id, embedding_json, model) VALUES (?, ?, ?)",
            (mid, "not-json", "synthetic-stale-model"),
        )
    if indexed and bm._SQLITE_VEC_AVAILABLE:
        rowid = store.conn.execute("SELECT rowid FROM working_memory WHERE id=?", (mid,)).fetchone()[0]
        bm._vec_table_insert(store.conn, "vec_working", rowid, [1.0] + [0.0] * (bm.EMBEDDING_DIM - 1))
    store.conn.commit()


def old_stats(conn, filters):
    clauses = [f"{key} = ?" for key, value in filters.items() if value]
    params = [value for value in filters.values() if value]
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = conn.execute("SELECT consolidated_at, pinned, timestamp FROM working_memory" + where,
                        params).fetchall()
    total = len(rows)
    consolidated = sum(r[0] is not None for r in rows)
    return {"total": total, "consolidated": consolidated, "unconsolidated": total - consolidated,
            "pinned_unconsolidated": sum(r[0] is None and r[1] == 1 for r in rows),
            "last": max((r[2] for r in rows), default=None)}


def assert_counts(stats, total, embedded, indexed, available):
    assert stats["total"] == total
    assert stats["embedding_rows"] == embedded
    assert stats["ann_indexed_rows"] == indexed
    assert stats["ann_index_available"] is available
    assert type(stats["embedding_rows"]) is int
    assert type(stats["ann_indexed_rows"]) is int


def test_empty_full_partial_and_presence_not_readiness(store):
    available = bool(bm._SQLITE_VEC_AVAILABLE)
    assert_counts(store.get_working_stats(), 0, 0, 0, available)
    add_row(store, "full", consolidated=True, pinned=True)
    assert_counts(store.get_working_stats(), 1, 1, int(available), available)
    add_row(store, "no-embedding", embedded=False, indexed=False, pinned=True)
    add_row(store, "missing-mirror", indexed=False)
    # Malformed JSON/stale model still count; no parsing/model check or repair.
    assert_counts(store.get_working_stats(), 3, 2, int(available), available)
    assert store.conn.execute("SELECT embedding_json FROM memory_embeddings WHERE memory_id='full'").fetchone()[0] == "not-json"
    assert {k: v for k, v in store.get_working_stats().items() if k not in NEW_KEYS} == old_stats(store.conn, {})


FILTER_CASES = list(itertools.product((None, "alpha", "beta", "absent"),
                                      (None, "human", "agent", "absent"),
                                      (None, "red", "blue", "absent")))
FILTER_CASES += [("", "", ""), ("alpha' OR 1=1 --", None, None)]


@pytest.mark.parametrize("author,kind,channel", FILTER_CASES)
def test_each_filter_and_combination(store, author, kind, channel):
    fixtures = [
        ("a", "alpha", "human", "red", True, True),
        ("b", "alpha", "agent", "blue", True, False),
        ("c", "beta", "human", "blue", False, False),
        ("d", "beta", "agent", "red", True, True),
    ]
    for mid, a, t, c, embedded, indexed in fixtures:
        add_row(store, mid, author=a, kind=t, channel=c, embedded=embedded, indexed=indexed)
    filters = dict(author_id=author, author_type=kind, channel_id=channel)
    expected = [r for r in fixtures if (not author or r[1] == author)
                and (not kind or r[2] == kind) and (not channel or r[3] == channel)]
    stats = store.get_working_stats(**filters)
    available = bool(bm._SQLITE_VEC_AVAILABLE)
    assert_counts(stats, len(expected), sum(r[4] for r in expected),
                  sum(r[5] for r in expected) if available else 0, available)
    assert {k: v for k, v in stats.items() if k not in NEW_KEYS} == old_stats(store.conn, filters)


def test_orphan_nonworking_and_ann_only_rows(store):
    add_row(store, "embedded", indexed=False)
    add_row(store, "ann-only", embedded=False)
    store.conn.execute(
        "INSERT INTO episodic_memory(id,content,source,timestamp,session_id) "
        "VALUES ('episode','Synthetic episode','test','2020-01-01T00:00:00','synthetic')"
    )
    store.conn.executemany(
        "INSERT INTO memory_embeddings(memory_id,embedding_json) VALUES (?, '[]')",
        [("orphan",), ("episode",)],
    )
    if bm._SQLITE_VEC_AVAILABLE:
        bm._vec_table_insert(store.conn, "vec_working", 99999, [1.0] + [0.0] * (bm.EMBEDDING_DIM - 1))
    store.conn.commit()
    available = bool(bm._SQLITE_VEC_AVAILABLE)
    assert_counts(store.get_working_stats(), 2, 1, int(available), available)


def test_presence_includes_expired_superseded_and_conversation(store):
    for mid in ("expired", "superseded", "conversation"):
        add_row(store, mid)
    store.conn.execute("UPDATE working_memory SET valid_until='2001-01-01T00:00:00' WHERE id='expired'")
    store.conn.execute("UPDATE working_memory SET superseded_by='expired' WHERE id='superseded'")
    store.conn.execute("UPDATE working_memory SET source='conversation' WHERE id='conversation'")
    store.conn.commit()
    available = bool(bm._SQLITE_VEC_AVAILABLE)
    assert_counts(store.get_working_stats(), 3, 3, 3 if available else 0, available)


def test_duplicate_representations_do_not_multiply_parents(store):
    # Defensive presence semantics also hold for a synthetic legacy/corrupt
    # representation table without its normal primary key. Fixture-only DDL.
    add_row(store, "parent", indexed=False)
    store.conn.execute("DROP TABLE memory_embeddings")
    store.conn.execute("CREATE TABLE memory_embeddings(memory_id TEXT, embedding_json TEXT)")
    store.conn.executemany("INSERT INTO memory_embeddings VALUES ('parent', ?)", [("[]",), ("not-json",)])
    store.conn.commit()
    assert store.get_working_stats()["embedding_rows"] == 1


def test_missing_ann_table_is_unavailable(store):
    add_row(store, "stored", indexed=False)
    if bm._SQLITE_VEC_AVAILABLE:
        store.conn.execute("DROP TABLE vec_working")
        store.conn.commit()
    assert_counts(store.get_working_stats(), 1, 1, 0, False)


def test_existing_ann_without_loaded_module(store):
    if not bm._SQLITE_VEC_AVAILABLE:
        pytest.skip("Requires sqlite-vec to create a real persisted virtual table")
    add_row(store, "stored")
    # Reopen the persisted DB WITHOUT loading sqlite-vec. No Beam init (which
    # would load it/create schema); call the real stats method on this handle.
    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.OperationalError, match="no such module: vec0"):
            conn.execute("SELECT 1 FROM vec_working LIMIT 0")
        read_handle = object.__new__(bm.BeamMemory)
        read_handle.conn = conn
        conn.execute("PRAGMA query_only=ON")
        assert_counts(read_handle.get_working_stats(), 1, 1, 0, False)


def test_missing_embedding_table_is_not_empty(store):
    store.conn.execute("DROP TABLE memory_embeddings")
    store.conn.commit()
    with pytest.raises(sqlite3.OperationalError, match="no such table: memory_embeddings"):
        store.get_working_stats()


class FaultCursor:
    """Inject exactly one SQL operation; every predecessor query stays real."""
    def __init__(self, conn, target, error):
        self.cursor = conn.cursor()
        self.target = target
        self.error = error
        self.statements = []

    def execute(self, sql, *args):
        self.statements.append(sql)
        if self.target in sql:
            raise self.error
        self.cursor.execute(sql, *args)
        return self

    def fetchone(self):
        return self.cursor.fetchone()


class FaultConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


@pytest.mark.parametrize("target", ["SELECT COUNT(*) FROM working_memory", "memory_embeddings",
                                   "SELECT 1 FROM vec_working LIMIT 0", "FROM vec_working WHERE"])
@pytest.mark.parametrize("message", ["database is locked", "disk I/O error", "database disk image is malformed",
                                    "no such table: unrelated", "no such module: unrelated"])
def test_unrelated_operational_faults_propagate_identity(store, target, message):
    if target == "FROM vec_working WHERE" and not bm._SQLITE_VEC_AVAILABLE:
        pytest.skip("ANN count is only reached with an available index")
    add_row(store, "stored")
    error = sqlite3.OperationalError(message)
    cursor = FaultCursor(store.conn, target, error)
    read_handle = object.__new__(bm.BeamMemory)
    read_handle.conn = FaultConnection(cursor)
    with pytest.raises(sqlite3.OperationalError) as raised:
        read_handle.get_working_stats()
    assert raised.value is error
    assert str(raised.value) == message
    if target != "SELECT COUNT(*) FROM working_memory":
        assert cursor.statements[0] == "SELECT COUNT(*) FROM working_memory"


@pytest.mark.parametrize("message", ["no such table: vec_working", "no such module: vec0"])
def test_capability_errors_after_successful_probe_are_not_suppressed(store, message):
    if not bm._SQLITE_VEC_AVAILABLE:
        pytest.skip("Requires the ANN count to be reached")
    error = sqlite3.OperationalError(message)
    cursor = FaultCursor(store.conn, "FROM vec_working WHERE", error)
    read_handle = object.__new__(bm.BeamMemory)
    read_handle.conn = FaultConnection(cursor)
    with pytest.raises(sqlite3.OperationalError) as raised:
        read_handle.get_working_stats()
    assert raised.value is error
    assert "SELECT 1 FROM vec_working LIMIT 0" in cursor.statements


def persistent_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not p.name.endswith("-shm")}


def test_stats_read_only_no_models_network_or_repair(store, tmp_path, monkeypatch):
    add_row(store, "stored", indexed=False)
    assert get_config().config_path.is_relative_to(tmp_path)
    assert get_config().config_path.is_file()
    def forbidden(*args, **kwargs):
        pytest.fail("Stats attempted model/network/repair activity")
    for name in ("available", "embed", "embed_query"):
        monkeypatch.setattr(em, name, forbidden)
    monkeypatch.setattr(bm, "repair_vec_working", forbidden)
    monkeypatch.setattr(bm, "_backfill_vec_working_from_memory_embeddings", forbidden)
    import socket
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    before = persistent_hashes(tmp_path)
    changes = store.conn.total_changes
    statements = []
    store.conn.execute("PRAGMA query_only=ON")
    store.conn.set_trace_callback(statements.append)
    try:
        stats = store.get_working_stats(author_id="alpha", author_type="human", channel_id="red")
    finally:
        store.conn.set_trace_callback(None)
        store.conn.execute("PRAGMA query_only=OFF")
    assert stats["embedding_rows"] == 1
    assert store.conn.total_changes == changes
    assert persistent_hashes(tmp_path) == before
    assert not store.conn.in_transaction
    assert all(s.lstrip().upper().startswith(("SELECT", "--")) for s in statements), statements


@pytest.mark.parametrize("embedding_dim_drift", [False, True])
def test_standalone_advertised_dispatch_recall_and_bank_isolation(tmp_path, monkeypatch, embedding_dim_drift):
    np = pytest.importorskip("numpy")
    from mnemosyne_hermes import MnemosyneMemoryProvider
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    MnemosyneConfig.reset_instance()
    if embedding_dim_drift:
        monkeypatch.setattr(em, "EMBEDDING_DIM", bm.EMBEDDING_DIM * 2)
    def vector(text):
        # Match the store's dimension; earlier embedding-module reloads can
        # change em.EMBEDDING_DIM without changing BEAM's imported dimension.
        out = np.zeros(bm.EMBEDDING_DIM, dtype=np.float32)
        out[1 if "cobalt" in text.lower() else 0] = 1
        return out
    # ONLY the model boundary is deterministic. Provider/store/ANN/JSON
    # fallback/candidate selection/ranking/public recall remain real.
    monkeypatch.setattr(em, "available", lambda: True)
    monkeypatch.setattr(em, "embed", lambda texts: np.array([vector(t) for t in texts]))
    monkeypatch.setattr(em, "embed_query", vector)
    def provider(identity):
        p = MnemosyneMemoryProvider()
        p.initialize("synthetic-session", agent_context="primary", hermes_home=str(tmp_path / "hermes"),
                     agent_identity=identity, profile_isolation=True, auto_sleep=False,
                     shared_surface_read=False)
        assert p._beam is not None, str(p._init_error)
        assert p._beam.db_path.is_relative_to(tmp_path)
        assert not p._auto_sleep_enabled
        assert "mnemosyne_stats" in {s["name"] for s in p.get_tool_schemas()}
        assert p.has_tool("mnemosyne_stats")
        return p
    def call(p, name, args=None):
        result = json.loads(p.handle_tool_call(name, args or {}))
        assert "error" not in result, result
        return result
    p = provider("alpha")
    available = bool(bm._SQLITE_VEC_AVAILABLE)
    assert_counts(call(p, "mnemosyne_stats")["working"], 0, 0, 0, available)
    first = call(p, "mnemosyne_remember", {"content": "Amber observatory telescope calibration uses prism fixture.", "scope": "session"})["memory_id"]
    second = call(p, "mnemosyne_remember", {"content": "Cobalt submarine ballast maintenance uses valve fixture.", "scope": "session"})["memory_id"]
    full = call(p, "mnemosyne_stats")
    assert_counts(full["working"], 2, 2, 2 if available else 0, available)
    monkeypatch.setattr(em, "available", lambda: False)
    call(p, "mnemosyne_remember", {"content": "Silver greenhouse irrigation fixture.", "scope": "session"})
    monkeypatch.setattr(em, "available", lambda: True)
    query = {"query": "Amber observatory telescope calibration", "limit": 5, "explain": True}
    recall = call(p, "mnemosyne_recall", query)
    assert any(r["id"] == first and r["tier"] == "working" and r["dense_score"] > 0 for r in recall["results"])
    assert not any(r["id"] == second for r in recall["results"])
    q = provider("beta")
    foreign = call(q, "mnemosyne_remember", {"content": "Amber observatory telescope foreign bank sentinel.", "scope": "session"})["memory_id"]
    assert p._beam.db_path != q._beam.db_path
    assert_counts(call(q, "mnemosyne_stats")["working"], 1, 1, int(available), available)
    assert not any(r["id"] == foreign for r in call(p, "mnemosyne_recall", query)["results"])
    # Public stats are no-write too, including provider dispatch and episodic.
    assert get_config().config_path.is_relative_to(tmp_path)
    assert get_config().config_path.is_file()
    def forbidden(*args, **kwargs):
        pytest.fail("Public stats attempted model/repair activity")
    for name in ("available", "embed", "embed_query"):
        monkeypatch.setattr(em, name, forbidden)
    monkeypatch.setattr(bm, "repair_vec_working", forbidden)
    monkeypatch.setattr(bm, "_backfill_vec_working_from_memory_embeddings", forbidden)
    before = persistent_hashes(tmp_path)
    changes = p._beam.conn.total_changes
    statements = []
    p._beam.conn.execute("PRAGMA query_only=ON")
    p._beam.conn.set_trace_callback(statements.append)
    try:
        stats = call(p, "mnemosyne_stats")
    finally:
        p._beam.conn.set_trace_callback(None)
        p._beam.conn.execute("PRAGMA query_only=OFF")
    assert persistent_hashes(tmp_path) == before
    assert p._beam.conn.total_changes == changes
    assert_counts(stats["working"], 3, 2, 2 if available else 0, available)
    assert stats["episodic"]["total"] == stats["episodic"]["vectors"] == 0
    assert stats["episodic"]["vec_type"] == "none"
    assert all(s.lstrip().upper().startswith(("SELECT", "--")) for s in statements), statements
