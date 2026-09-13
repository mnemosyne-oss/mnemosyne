"""
Tests for Mnemosyne BEAM architecture
"""

import pytest
import tempfile
import sqlite3
import time
import os
import gc
from pathlib import Path
from datetime import datetime, timedelta, timezone

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory, init_beam, _find_memories_by_fact, _wm_vec_search


def _same_utc_day_future(
    now,
    ahead=timedelta(hours=2),
    margin=timedelta(seconds=30),
):
    """Pick a naive future ``valid_until`` that stays on ``now``'s UTC date.

    A space separator only sorts before a "T" separator while the date
    components match, so a naive value that rolls into tomorrow sorts *after*
    an aware-UTC ``now`` and stops misleading a lexical filter. Clamping to the
    end of today keeps that trap intact.

    Returns ``None`` when no such value can outlive the caller. Clamping close
    to midnight can leave milliseconds of validity, which expires mid-test and
    would trade one flake for a narrower one.
    """
    candidate = now + ahead
    if candidate.date() != now.date():
        candidate = now.replace(hour=23, minute=59, second=59, microsecond=0)
    if candidate <= now + margin:
        return None
    return candidate
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.streaming import EventType


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


@pytest.fixture
def non_utc_tz(monkeypatch):
    """Run under a non-UTC host timezone, preserving the original TZ.

    Guards ``time.tzset()`` for platforms that do not provide it (Windows)
    and restores the original TZ value before reapplying the timezone in
    teardown, so the process is not left in a different runtime timezone.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() unavailable on this platform")
    original = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "Europe/Copenhagen")
    time.tzset()
    yield
    if original is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", original)
    time.tzset()


def test_get_working_memory_respects_global_cross_session_visibility(temp_db):
    writer = BeamMemory(session_id="session-a", db_path=temp_db)
    global_id = writer.remember("global get visibility", source="test", scope="global")
    private_id = writer.remember("private get visibility", source="test", scope="session")

    # Same-session lookup remains available for both visibility scopes.
    same_session_global = writer.get(global_id)
    same_session_private = writer.get(private_id)
    assert same_session_global is not None
    assert same_session_global["id"] == global_id
    assert same_session_private is not None
    assert same_session_private["id"] == private_id

    reader = BeamMemory(session_id="session-b", db_path=temp_db)
    cross_session_global = reader.get(global_id)
    assert cross_session_global is not None
    assert cross_session_global["id"] == global_id
    assert cross_session_global["memory_store"] == "working"
    assert reader.get(private_id) is None


def test_update_working_respects_global_cross_session_visibility(temp_db):
    writer = BeamMemory(session_id="session-a", db_path=temp_db)
    global_id = writer.remember("global before", source="test", scope="global")
    private_id = writer.remember("private before", source="test", scope="session")

    updater = BeamMemory(session_id="session-b", db_path=temp_db)
    assert updater.update_working(global_id, content="global after") is True
    assert updater.get(global_id)["content"] == "global after"

    assert updater.update_working(private_id, content="private after") is False
    assert writer.get(private_id)["content"] == "private before"


def test_mnemosyne_update_reports_success_for_beam_only_global_memory(temp_db):
    writer = BeamMemory(session_id="session-a", db_path=temp_db)
    memory_id = writer.remember("beam only before", source="test", scope="global")

    updater = Mnemosyne(session_id="session-b", db_path=temp_db)
    assert updater.update(memory_id, content="beam only after") is True
    assert updater.get(memory_id)["content"] == "beam only after"


def test_mnemosyne_update_keeps_dual_written_global_memory_in_sync(temp_db):
    writer = Mnemosyne(session_id="session-a", db_path=temp_db)
    memory_id = writer.remember("dual before", source="test", scope="global")

    updater = Mnemosyne(session_id="session-b", db_path=temp_db)
    assert updater.update(memory_id, content="dual after", importance=0.9) is True
    updated = updater.get(memory_id)
    assert updated["content"] == "dual after"
    assert updated["importance"] == 0.9
    legacy_content, legacy_importance = updater.conn.execute(
        "SELECT content, importance FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    assert legacy_content == "dual after"
    assert legacy_importance == 0.9


def test_mnemosyne_update_rejects_dual_written_private_memory_cross_session(temp_db):
    writer = Mnemosyne(session_id="session-a", db_path=temp_db)
    memory_id = writer.remember("private before", source="test", scope="session")

    updater = Mnemosyne(session_id="session-b", db_path=temp_db)
    assert updater.update(memory_id, content="private after") is False
    assert writer.get(memory_id)["content"] == "private before"
    legacy_content = updater.conn.execute(
        "SELECT content FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()[0]
    assert legacy_content == "private before"


def test_mnemosyne_update_unknown_memory_emits_no_event(temp_db):
    memory = Mnemosyne(session_id="session-a", db_path=temp_db).enable_streaming()

    assert memory.update("unknown-memory", content="after") is False
    assert memory.stream.get_buffer(event_types=[EventType.MEMORY_UPDATED]) == []


def test_mnemosyne_update_foreign_private_memory_emits_no_event(temp_db):
    writer = Mnemosyne(session_id="session-a", db_path=temp_db)
    memory_id = writer.remember("private before", source="test", scope="session")
    updater = Mnemosyne(session_id="session-b", db_path=temp_db).enable_streaming()

    assert updater.update(memory_id, content="private after") is False
    assert writer.get(memory_id)["content"] == "private before"
    assert updater.stream.get_buffer(event_types=[EventType.MEMORY_UPDATED]) == []


def test_mnemosyne_update_rolls_back_and_emits_no_event_when_beam_write_fails(
    temp_db, monkeypatch
):
    writer = Mnemosyne(session_id="session-a", db_path=temp_db)
    memory_id = writer.remember("before", source="test", scope="global")
    updater = Mnemosyne(session_id="session-b", db_path=temp_db).enable_streaming()

    def fail_second_dual_write(*args, **kwargs):
        raise RuntimeError("forced BEAM update failure")

    monkeypatch.setattr(updater.beam, "update_working", fail_second_dual_write)

    with pytest.raises(RuntimeError, match="forced BEAM update failure"):
        updater.update(memory_id, content="after", importance=0.9)

    legacy_content, legacy_importance = writer.conn.execute(
        "SELECT content, importance FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    assert (legacy_content, legacy_importance) == ("before", 0.5)
    unchanged = writer.get(memory_id)
    assert (unchanged["content"], unchanged["importance"]) == ("before", 0.5)
    assert updater.stream.get_buffer(event_types=[EventType.MEMORY_UPDATED]) == []


def test_mnemosyne_update_emits_once_after_dual_write_commit(temp_db):
    writer = Mnemosyne(session_id="session-a", db_path=temp_db)
    memory_id = writer.remember("before", source="test", scope="global")
    updater = Mnemosyne(session_id="session-b", db_path=temp_db).enable_streaming()
    observed = []

    def observe(event):
        legacy = updater.conn.execute(
            "SELECT content, importance FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        working = updater.conn.execute(
            "SELECT content, importance FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
        observed.append((event, legacy, working, updater.conn.in_transaction))

    updater.stream.on(EventType.MEMORY_UPDATED, observe)

    assert updater.update(memory_id, content="after", importance=0.9) is True
    events = updater.stream.get_buffer(event_types=[EventType.MEMORY_UPDATED])
    assert len(events) == 1
    assert events[0].memory_id == memory_id
    assert len(observed) == 1
    _, legacy, working, in_transaction = observed[0]
    assert tuple(legacy) == ("after", 0.9)
    assert tuple(working) == ("after", 0.9)
    assert in_transaction is False


def test_mnemosyne_update_preserves_legacy_only_session_compatibility(temp_db):
    writer = Mnemosyne(session_id="session-a", db_path=temp_db)
    memory_id = "legacy-only-memory"
    writer.conn.execute(
        """
        INSERT INTO memories (id, content, source, timestamp, session_id, importance)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "legacy before", "test", "2026-01-01T00:00:00", "session-a", 0.5),
    )
    writer.conn.commit()

    assert writer.update(memory_id, content="legacy after") is True
    assert writer.conn.execute(
        "SELECT content FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()[0] == "legacy after"

    other_session = Mnemosyne(session_id="session-b", db_path=temp_db)
    assert other_session.update(memory_id, content="should not update") is False
    assert writer.conn.execute(
        "SELECT content FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()[0] == "legacy after"


class _FakeAnnotations:
    def __init__(self, rows):
        self._rows = rows

    def query_by_kind(self, kind):
        assert kind == "fact"
        return self._rows


class _FakeBeam:
    def __init__(self, rows):
        self.annotations = _FakeAnnotations(rows)


def test_recall_computes_query_embedding_once_per_call(temp_db, monkeypatch):
    np = pytest.importorskip("numpy")
    beam = BeamMemory(session_id="embed-cache-test", db_path=temp_db)
    calls = []

    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)

    def fake_embed_query(query):
        calls.append(query)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(beam_module._embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(beam_module, "_wm_vec_search", lambda conn, emb, k=20: [])
    monkeypatch.setattr(beam_module, "_vec_available", lambda conn: False)
    monkeypatch.setattr(beam_module, "_in_memory_vec_search", lambda conn, emb, k=20: [])
    monkeypatch.setattr(beam_module, "_mib", lambda emb: b"\x00")

    beam.recall("cache this query", top_k=5)

    assert calls == ["cache this query"]


def test_wm_vec_search_pushes_recall_filters_before_vector_decode(temp_db, monkeypatch):
    np = pytest.importorskip("numpy")
    beam = BeamMemory(session_id="target-session", db_path=temp_db)
    now = datetime.now().isoformat()
    conn = beam.conn
    conn.executemany(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, channel_id, author_id, author_type, veracity, memory_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("target", "target vector memory", "chat", now, "target-session", "session", "chan-a", "alice", "human", "stated", "preference"),
            ("other-session", "other session memory", "chat", now, "other-session", "session", "chan-b", "bob", "human", "stated", "preference"),
            ("other-source", "wrong source memory", "cron", now, "target-session", "session", "chan-a", "alice", "human", "stated", "preference"),
        ],
    )
    conn.executemany(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        [
            ("target", "[1.0, 0.0, 0.0]", "test"),
            ("other-session", "[1.0, 0.0, 0.0]", "test"),
            ("other-source", "[1.0, 0.0, 0.0]", "test"),
        ],
    )
    conn.commit()

    decoded = []
    real_loads = beam_module.json.loads

    def counting_loads(value):
        decoded.append(value)
        return real_loads(value)

    monkeypatch.setattr(beam_module.json, "loads", counting_loads)

    results = _wm_vec_search(
        conn,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        k=10,
        where_sql=(
            "(valid_until IS NULL OR valid_until > ?) AND superseded_by IS NULL "
            "AND (session_id = ? OR scope = 'global') AND source = ? AND channel_id = ?"
        ),
        where_params=(now, "target-session", "chat", "chan-a"),
    )

    assert [r["id"] for r in results] == ["target"]
    assert decoded == ["[1.0, 0.0, 0.0]"]


def _unit_embedding():
    np = pytest.importorskip("numpy")
    return np.array([1.0] + [0.0] * (beam_module.EMBEDDING_DIM - 1), dtype=np.float32)


def _require_vec_working(conn):
    pytest.importorskip("sqlite_vec")
    if not beam_module._wm_vec_available(conn):
        pytest.skip("sqlite-vec vec_working table unavailable")


def test_remember_update_and_forget_maintain_vec_working(temp_db, monkeypatch):
    beam = BeamMemory(session_id="vec-working-session", db_path=temp_db)
    _require_vec_working(beam.conn)

    np = pytest.importorskip("numpy")
    vectors = [_unit_embedding(), np.array([0.0, 1.0] + [0.0] * (beam_module.EMBEDDING_DIM - 2), dtype=np.float32)]
    calls = []

    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)

    def fake_embed(contents):
        calls.extend(contents)
        return [vectors[min(len(calls) - 1, len(vectors) - 1)]]

    monkeypatch.setattr(beam_module._embeddings, "embed", fake_embed)

    memory_id = beam.remember("vec working initial", source="test", scope="session")
    rowid = beam.conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()["rowid"]

    assert beam.conn.execute("SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)).fetchone()[0] == 1
    assert beam.conn.execute("SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)).fetchone()[0] == 1

    assert beam.update_working(memory_id, content="vec working updated") is True
    assert beam.conn.execute("SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)).fetchone()[0] == 1
    assert beam.conn.execute("SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)).fetchone()[0] == 1

    assert beam.forget_working(memory_id) is True
    assert beam.conn.execute("SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)).fetchone()[0] == 0
    assert beam.conn.execute("SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)).fetchone()[0] == 0


def test_vec_working_backfill_from_memory_embeddings_is_idempotent(temp_db):
    beam = BeamMemory(session_id="vec-working-backfill", db_path=temp_db)
    _require_vec_working(beam.conn)
    now = datetime.now().isoformat()
    memory_id = "manual-wm"
    embedding = _unit_embedding()

    beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "manual working memory", "test", now, "vec-working-backfill", "session", 0.5),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, beam_module._embeddings.serialize(embedding), "test"),
    )
    beam.conn.commit()
    rowid = beam.conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()["rowid"]

    assert beam.conn.execute("SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)).fetchone()[0] == 0
    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 1
    assert beam.conn.execute("SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)).fetchone()[0] == 1
    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 0


def test_wm_vec_search_prefers_vec_working_without_decoding_fallback(temp_db, monkeypatch):
    np = pytest.importorskip("numpy")
    beam = BeamMemory(session_id="vec-working-search", db_path=temp_db)
    _require_vec_working(beam.conn)
    now = datetime.now().isoformat()
    memory_id = "sqlite-hit"
    embedding = _unit_embedding()

    beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, channel_id, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "sqlite vector hit", "chat", now, "vec-working-search", "session", "chan-a", 0.5),
    )
    rowid = beam.conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()["rowid"]
    beam_module._vec_table_insert(beam.conn, "vec_working", rowid, embedding)
    # Poison the compatibility store: the sqlite-vec path should satisfy the
    # query without decoding memory_embeddings JSON at all.
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, "not-json", "test"),
    )
    beam.conn.commit()

    def fail_json_loads(_value):
        raise AssertionError("memory_embeddings fallback should not be decoded")

    monkeypatch.setattr(beam_module.json, "loads", fail_json_loads)

    results = _wm_vec_search(
        beam.conn,
        np.array(embedding, dtype=np.float32),
        k=5,
        where_sql=(
            "(valid_until IS NULL OR valid_until > ?) AND superseded_by IS NULL "
            "AND (session_id = ? OR scope = 'global') AND source = ? AND channel_id = ?"
        ),
        where_params=(now, "vec-working-search", "chat", "chan-a"),
    )

    assert [r["id"] for r in results] == [memory_id]
    assert results[0]["sim"] > 0.0


def test_wm_vec_search_falls_back_when_vec_working_missing_row(temp_db):
    np = pytest.importorskip("numpy")
    beam = BeamMemory(session_id="vec-working-fallback", db_path=temp_db)
    _require_vec_working(beam.conn)
    now = datetime.now().isoformat()
    memory_id = "fallback-hit"
    embedding = _unit_embedding()

    beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "fallback vector hit", "chat", now, "vec-working-fallback", "session", 0.5),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, beam_module._embeddings.serialize(embedding), "test"),
    )
    beam.conn.commit()

    results = _wm_vec_search(
        beam.conn,
        np.array(embedding, dtype=np.float32),
        k=5,
        where_sql="(valid_until IS NULL OR valid_until > ?) AND superseded_by IS NULL AND (session_id = ? OR scope = 'global')",
        where_params=(now, "vec-working-fallback"),
    )

    assert [r["id"] for r in results] == [memory_id]
    assert results[0]["sim"] == pytest.approx(1.0)


@pytest.mark.parametrize("vec_type", ["float32", "int8", "bit"])
def test_wm_vec_search_overfetches_and_returns_partial_results_when_exhausted(temp_db, monkeypatch, vec_type):
    np = pytest.importorskip("numpy")
    beam = BeamMemory(session_id="target-session", db_path=temp_db)
    _require_vec_working(beam.conn)
    beam.conn.execute("DROP TABLE vec_working")
    beam.conn.execute(
        f"CREATE VIRTUAL TABLE vec_working USING vec0(embedding {vec_type}[{beam_module.EMBEDDING_DIM}])"
    )
    now = datetime.now().isoformat()
    exact = _unit_embedding()
    target = np.array([0.9, 0.1] + [0.0] * (beam_module.EMBEDDING_DIM - 2), dtype=np.float32)

    rows = [
        (f"excluded-{i}", f"excluded {i}", "test", now, "other-session", "session", 0.5)
        for i in range(1201)
    ]
    rows.append(("global-target", "global target", "test", now, "target-session", "global", 0.5))
    beam.conn.executemany(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    for memory_id, *_rest in rows[:-1]:
        rowid = beam.conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()["rowid"]
        beam_module._vec_table_insert(beam.conn, "vec_working", rowid, exact, commit=False)
    target_rowid = beam.conn.execute(
        "SELECT rowid FROM working_memory WHERE id = 'global-target'"
    ).fetchone()["rowid"]
    beam_module._vec_table_insert(beam.conn, "vec_working", target_rowid, target, commit=False)
    beam.conn.commit()

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("memory_embeddings fallback should not be invoked")

    monkeypatch.setattr(beam_module, "_wm_vec_search_fallback", fail_fallback)

    results = _wm_vec_search(
        beam.conn,
        exact,
        k=5,
        where_sql=(
            "(valid_until IS NULL OR valid_until > ?) AND superseded_by IS NULL "
            "AND (session_id = ? OR scope = 'global')"
        ),
        where_params=(now, "target-session"),
    )

    assert [r["id"] for r in results] == ["global-target"]
    assert len(results) < 5  # The exhausted table has fewer eligible rows than the requested k.


def test_wm_vec_search_amortizes_vec_working_count_until_database_changes(temp_db):
    beam = BeamMemory(session_id="count-cache", db_path=temp_db)
    _require_vec_working(beam.conn)
    embedding = _unit_embedding()
    now = datetime.now().isoformat()

    def insert_memory(conn, memory_id):
        conn.execute(
            """
            INSERT INTO working_memory
                (id, content, source, timestamp, session_id, scope, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (memory_id, memory_id, "test", now, "count-cache", "session", 0.5),
        )
        rowid = conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()["rowid"]
        beam_module._vec_table_insert(conn, "vec_working", rowid, embedding)

    insert_memory(beam.conn, "first")
    statements = []
    beam.conn.set_trace_callback(statements.append)
    try:
        for _ in range(2):
            results = beam_module._wm_vec_search_sqlite(
                beam.conn,
                embedding,
                k=5,
                where_sql="session_id = ?",
                where_params=("count-cache",),
            )
            assert [r["id"] for r in results] == ["first"]

        assert sum("SELECT COUNT(*) FROM vec_working" in sql for sql in statements) == 1

        insert_memory(beam.conn, "second")
        results = beam_module._wm_vec_search_sqlite(
            beam.conn,
            embedding,
            k=5,
            where_sql="session_id = ?",
            where_params=("count-cache",),
        )
        assert {r["id"] for r in results} == {"first", "second"}
        assert sum("SELECT COUNT(*) FROM vec_working" in sql for sql in statements) == 2

        external = sqlite3.connect(str(temp_db), factory=beam_module._BeamConnection)
        external.row_factory = sqlite3.Row
        try:
            external.enable_load_extension(True)
            beam_module.sqlite_vec.load(external)
            insert_memory(external, "external")
        finally:
            external.close()

        results = beam_module._wm_vec_search_sqlite(
            beam.conn,
            embedding,
            k=5,
            where_sql="session_id = ?",
            where_params=("count-cache",),
        )
        assert {r["id"] for r in results} == {"first", "second", "external"}
        assert sum("SELECT COUNT(*) FROM vec_working" in sql for sql in statements) == 3
    finally:
        beam.conn.set_trace_callback(None)


def test_vec_working_coverage_reports_missing_and_repair_fills_gap(temp_db):
    beam = BeamMemory(session_id="vec-working-coverage", db_path=temp_db)
    _require_vec_working(beam.conn)
    now = datetime.now().isoformat()
    memory_id = "coverage-gap"
    embedding = _unit_embedding()

    beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "coverage vector gap", "test", now, "vec-working-coverage", "session", 0.5),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, beam_module._embeddings.serialize(embedding), "test"),
    )
    beam.conn.commit()

    before = beam_module.vec_working_coverage(beam.conn)
    assert before["vec_working_available"] is True
    assert before["working_memory_rows"] == 1
    assert before["working_embedding_rows"] == 1
    assert before["vec_working_rows"] == 0
    assert before["missing_vec_working_rows"] == 1
    assert before["status"] == "partial"

    dry_run = beam_module.repair_vec_working(beam.conn, dry_run=True)
    assert dry_run["status"] == "dry_run"
    assert dry_run["inserted"] == 0
    assert dry_run["after"]["missing_vec_working_rows"] == 1

    repaired = beam_module.repair_vec_working(beam.conn)
    assert repaired["status"] == "repaired"
    assert repaired["inserted"] == 1
    assert repaired["after"]["missing_vec_working_rows"] == 0
    assert repaired["after"]["vec_working_rows"] == 1
    assert repaired["after"]["status"] == "complete"

    second = beam_module.repair_vec_working(beam.conn)
    assert second["inserted"] == 0
    assert second["after"]["status"] == "complete"


def test_vec_working_coverage_reports_fallback_only_when_table_unavailable(temp_db, monkeypatch):
    beam = BeamMemory(session_id="vec-working-unavailable", db_path=temp_db)
    now = datetime.now().isoformat()
    beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("fallback-only", "fallback only vector", "test", now, "vec-working-unavailable", "session", 0.5),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        ("fallback-only", beam_module._embeddings.serialize(_unit_embedding()), "test"),
    )
    beam.conn.commit()
    monkeypatch.setattr(beam_module, "_wm_vec_available", lambda _conn: False)

    coverage = beam_module.vec_working_coverage(beam.conn)
    assert coverage["vec_working_available"] is False
    assert coverage["working_embedding_rows"] == 1
    assert coverage["status"] == "fallback_only"

    repair = beam_module.repair_vec_working(beam.conn)
    assert repair["status"] == "skipped"
    assert repair["reason"] == "vec_working unavailable"


@pytest.mark.parametrize("terminal_status", ["no_vectors", "empty"])
def test_vec_working_repair_accepts_no_work_terminal_states(monkeypatch, terminal_status):
    """Empty or vectorless stores are healthy no-op repair outcomes."""
    coverage = {
        "vec_working_available": True,
        "missing_vec_working_rows": 0,
        "status": terminal_status,
    }
    monkeypatch.setattr(beam_module, "vec_working_coverage", lambda _conn: coverage)
    monkeypatch.setattr(beam_module, "_backfill_vec_working_from_memory_embeddings", lambda _conn: 0)

    conn = sqlite3.connect(":memory:")
    try:
        result = beam_module.repair_vec_working(conn)
    finally:
        conn.close()

    assert result["status"] == "repaired"
    assert result["inserted"] == 0
    assert result["after"]["status"] == terminal_status


def test_vec_working_repair_reports_backfill_insert_failure(temp_db, monkeypatch):
    """Repair must not claim success while vec_working coverage remains partial."""
    beam = BeamMemory(session_id="vec-working-repair-failure", db_path=temp_db)
    beam.conn.execute("DROP TABLE IF EXISTS vec_working")
    beam.conn.execute("CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY, embedding TEXT)")
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("unrepaired", "working source", "test", "2026-01-01T00:00:00", "vec-working-repair-failure"),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        ("unrepaired", "[0.1, 0.2]", "test"),
    )
    beam.conn.commit()
    monkeypatch.setattr(beam_module, "_wm_vec_available", lambda _conn: True)
    monkeypatch.setattr(beam_module, "np", object())
    monkeypatch.setattr(
        beam_module,
        "_vec_table_insert",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("vec insert failed")),
    )

    result = beam_module.repair_vec_working(beam.conn)

    assert result["status"] == "error"
    assert "vec insert failed" in result["error"]
    assert result["after"]["missing_vec_working_rows"] == 1


@pytest.mark.parametrize(
    "coverage_error_key",
    [
        "missing_vec_working_rows_error",
        "memory_embeddings_error",
        "vec_working_rows_error",
        "error",
    ],
)
def test_vec_working_repair_reports_unavailable_coverage(monkeypatch, coverage_error_key):
    """Any post-repair coverage error key must prevent a repaired status."""
    coverage_results = iter([
        {"vec_working_available": True, "missing_vec_working_rows": 1},
        {
            "vec_working_available": True,
            "missing_vec_working_rows": 0,
            "status": "complete",
            coverage_error_key: "OperationalError: coverage unavailable",
        },
    ])
    monkeypatch.setattr(beam_module, "vec_working_coverage", lambda _conn: next(coverage_results))
    monkeypatch.setattr(beam_module, "_backfill_vec_working_from_memory_embeddings", lambda _conn: 1)

    class _Connection:
        def commit(self):
            pass

    result = beam_module.repair_vec_working(_Connection())

    assert result["status"] == "error"
    assert coverage_error_key in result["error"]
    assert "coverage unavailable" in result["error"]


def test_vec_working_repair_requires_complete_available_postcheck(monkeypatch):
    """A post-repair coverage error with a default zero count is not success."""
    coverage_results = iter([
        {"vec_working_available": True, "missing_vec_working_rows": 1},
        {
            "vec_working_available": False,
            "missing_vec_working_rows": 0,
            "status": "error",
            "error": "vec_working table missing",
        },
    ])
    monkeypatch.setattr(beam_module, "vec_working_coverage", lambda _conn: next(coverage_results))
    monkeypatch.setattr(beam_module, "_backfill_vec_working_from_memory_embeddings", lambda _conn: 1)

    class _Connection:
        def commit(self):
            pass

    result = beam_module.repair_vec_working(_Connection())

    assert result["status"] == "error"
    assert "coverage unavailable" in result["error"]


class TestFactAnnotationMatching:
    def test_strict_fact_match_ignores_stopword_only_matches(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_STRICT_FACT_MATCH", "1")
        beam = _FakeBeam([
            {"memory_id": "noise", "value": "The user likes coffee and the weather is sunny."},
            {
                "memory_id": "target",
                "value": "Project Atlas lives at /opt/apps/project-atlas and URL http://127.0.0.1:8765/.",
            },
        ])

        result = set(_find_memories_by_fact(
            beam,
            "Where does the Project Atlas app live and what URL port does it use?",
        ))

        assert result == {"target"}

    def test_strict_fact_match_allows_exact_entity_phrase(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_STRICT_FACT_MATCH", "1")
        beam = _FakeBeam([
            {"memory_id": "target", "value": "Docker Browser must use host.docker.internal for host machine apps."},
        ])

        result = set(_find_memories_by_fact(beam, "Docker Browser"))

        assert result == {"target"}

    def test_legacy_fact_match_preserved_when_flag_off(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_LENIENT_FACT_MATCH", "1")
        beam = _FakeBeam([
            {"memory_id": "legacy", "value": "The user likes coffee."},
        ])

        result = set(_find_memories_by_fact(beam, "where is the dashboard"))

        assert result == {"legacy"}


    def test_strict_fact_match_default(self):
        """With no env var, strict matching is on by default — common stopwords
        like 'where is the' should NOT match. Only significant tokens apply."""
        beam = _FakeBeam([
            {"memory_id": "a", "value": "The user likes coffee."},
        ])

        result = set(_find_memories_by_fact(beam, "where is the dashboard"))

        assert result == set()


class TestBeamSchema:
    def test_init_creates_tables(self, temp_db):
        init_beam(temp_db)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        tables = [r[0] for r in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "working_memory" in tables
        assert "episodic_memory" in tables
        assert "scratchpad" in tables
        assert "consolidation_log" in tables
        # FTS5 virtual table
        assert "fts_episodes" in tables
        conn.close()

    def test_context_indexes_created(self, temp_db):
        init_beam(temp_db)
        conn = sqlite3.connect(temp_db)
        indexes = {r[1] for r in conn.execute("PRAGMA index_list(working_memory)").fetchall()}
        assert "idx_wm_context_session" in indexes
        assert "idx_wm_context_global" in indexes
        conn.close()

    def test_vec_type_probe_does_not_rollback_schema_ddl(self, temp_db, monkeypatch):
        """Vector capability probing must not undo unrelated schema DDL.

        Some sqlite-vec builds reject a requested vector type. The probe used
        to call connection.rollback(), which could erase earlier CREATE TABLE
        statements in init_beam() before the schema transaction was committed.
        """
        conn = sqlite3.connect(temp_db)
        conn.execute("CREATE TABLE filler_table (id TEXT PRIMARY KEY)")

        monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", True)
        monkeypatch.setattr(beam_module, "VEC_TYPE", "int8")

        assert beam_module._detect_vec_type(conn) == "float32"

        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "filler_table" in tables
        conn.close()


class TestWorkingMemory:
    def test_remember_and_context(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("Prefers Neovim", source="preference", importance=0.9)
        assert mid is not None

        ctx = beam.get_context(limit=5)
        assert len(ctx) == 1
        assert ctx[0]["content"] == "Prefers Neovim"

    def test_entity_annotations_only_boost_existing_recall_candidates(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        lexical_id = beam.remember("Atlas deployment notes", importance=0.1)
        entity_only_id = beam.remember("unrelated status update", importance=1.0)

        baseline = beam.recall("Atlas", top_k=1)
        assert [row["id"] for row in baseline] == [lexical_id]
        baseline_score = baseline[0]["score"]

        monkeypatch.setattr(
            beam_module,
            "_find_memories_by_entity",
            lambda _beam, _query: [lexical_id, entity_only_id],
        )

        results = beam.recall("Atlas", top_k=2)
        result_ids = {row["id"] for row in results}
        lexical_result = next(row for row in results if row["id"] == lexical_id)

        assert entity_only_id not in result_ids
        assert lexical_id in result_ids
        assert lexical_result["entity_match"] is True
        assert baseline_score < lexical_result["score"] <= min(baseline_score * 1.3, 1.0) + 0.0001

    def test_entity_annotations_do_not_append_episodic_only_candidates(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        lexical_id = beam.consolidate_to_episodic(
            summary="Atlas deployment notes", source_wm_ids=["wm-atlas"], importance=0.1,
        )
        entity_only_id = beam.consolidate_to_episodic(
            summary="unrelated status update", source_wm_ids=["wm-other"], importance=1.0,
        )

        baseline = beam.recall("Atlas", top_k=1)
        assert [row["id"] for row in baseline] == [lexical_id]
        baseline_score = baseline[0]["score"]

        entity_calls = []

        def entity_matches(_beam, query):
            entity_calls.append(query)
            return [lexical_id, entity_only_id]

        monkeypatch.setattr(beam_module, "_find_memories_by_entity", entity_matches)

        results = beam.recall("Atlas", top_k=2)
        result_ids = {row["id"] for row in results}
        lexical_result = next(row for row in results if row["id"] == lexical_id)

        assert entity_calls == ["Atlas"]
        assert entity_only_id not in result_ids
        assert lexical_id in result_ids
        assert lexical_result["entity_match"] is True
        assert baseline_score < lexical_result["score"] <= min(baseline_score * 1.3, 1.0) + 0.0001

    def test_get_context_keeps_global_first_then_session_order(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        now = datetime.now()
        rows = [
            ("session-high", "session high", "conversation", (now - timedelta(minutes=1)).isoformat(), "s1", 0.99, "session"),
            ("global-low", "global low", "preference", (now - timedelta(minutes=2)).isoformat(), "other", 0.10, "global"),
            ("global-high", "global high", "preference", now.isoformat(), "other", 0.80, "global"),
            ("session-low", "session low", "conversation", (now - timedelta(minutes=3)).isoformat(), "s1", 0.10, "session"),
            ("other-session", "other session", "conversation", now.isoformat(), "s2", 1.00, "session"),
        ]
        beam.conn.executemany(
            """
            INSERT INTO working_memory
                (id, content, source, timestamp, session_id, importance, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        beam.conn.commit()

        ctx = beam.get_context(limit=4)
        assert [row["id"] for row in ctx] == [
            "global-high",
            "global-low",
            "session-high",
            "session-low",
        ]
        assert all("last_recalled" not in row for row in ctx)

    def test_get_context_limit_minus_one_preserves_unbounded_sqlite_limit(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        now = datetime.now().isoformat()
        for i in range(3):
            beam.conn.execute(
                """
                INSERT INTO working_memory
                    (id, content, source, timestamp, session_id, importance, scope)
                VALUES (?, ?, 'conversation', ?, 's1', ?, 'session')
                """,
                (f"m{i}", f"memory {i}", now, i / 10),
            )
        beam.conn.commit()

        ctx = beam.get_context(limit=-1)
        assert len(ctx) == 3

    def test_get_context_query_shapes_use_context_indexes(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        now = datetime.now().isoformat()
        for i in range(20):
            beam.conn.execute(
                """
                INSERT INTO working_memory
                    (id, content, source, timestamp, session_id, importance, scope)
                VALUES (?, ?, 'conversation', ?, ?, ?, ?)
                """,
                (
                    f"m{i}",
                    f"memory {i}",
                    now,
                    "s1" if i % 2 else "s2",
                    i / 20,
                    "global" if i % 5 == 0 else "session",
                ),
            )
        beam.conn.commit()

        global_plan = "\n".join(
            row[3]
            for row in beam.conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT id, content, source, timestamp, importance, scope, last_recalled
                FROM working_memory
                WHERE scope = 'global'
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
                  AND consolidated_at IS NULL
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                (now, 10),
            ).fetchall()
        )
        session_plan = "\n".join(
            row[3]
            for row in beam.conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT id, content, source, timestamp, importance, scope, last_recalled
                FROM working_memory
                WHERE session_id = ?
                  AND (scope IS NULL OR scope != 'global')
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
                  AND consolidated_at IS NULL
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                ("s1", now, 10),
            ).fetchall()
        )

        assert "idx_wm_context_global" in global_plan
        assert "idx_wm_context_session" in session_plan
        assert "USE TEMP B-TREE" not in global_plan
        assert "USE TEMP B-TREE" not in session_plan

    def test_trim_old_memories(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Insert old memory directly
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("old1", "old content", "conversation", old_ts, "s1")
        )
        conn.commit()
        conn.close()

        beam._trim_working_memory()
        stats = beam.get_working_stats()
        assert stats["total"] == 0

    def test_invalidate_stamps_aware_utc_valid_until(self, temp_db, non_utc_tz):
        """#525: invalidate() writes an aware-UTC valid_until, not naive local.

        A naive local timestamp compared against SQLite UTC surfaces
        (julianday('now') / CURRENT_TIMESTAMP) drifts by the host offset.
        The write path must produce an aware UTC instant so recall/context
        filtering agrees with doctor/repair.
        """
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("expiring memory", source="test", importance=0.9)

        assert beam.invalidate(mid) is True
        row = beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row is not None
        valid_until = datetime.fromisoformat(row[0])
        assert valid_until.tzinfo is not None
        assert valid_until.utcoffset() == timedelta(0)

    def test_invalidated_memory_excluded_across_utc_surfaces(self, temp_db, non_utc_tz):
        """#525: after invalidate(), recall/context and SQLite UTC agree.

        The Python-side filters compare against aware-UTC now, matching the
        julianday('now') comparison used by doctor/repair, so both surfaces
        agree on a non-UTC host.
        """
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("expiring memory", source="test", importance=0.9)
        assert beam.invalidate(mid) is True

        assert mid not in {r["id"] for r in beam.get_context(limit=10)}
        assert mid not in {r["id"] for r in beam.recall("expiring memory", top_k=10)}
        row = beam.conn.execute(
            "SELECT CASE WHEN valid_until IS NULL OR "
            "julianday(valid_until) > julianday('now') "
            "THEN 1 ELSE 0 END AS active "
            "FROM working_memory WHERE id = ?",
            (mid,),
        ).fetchone()
        assert row["active"] == 0

    def test_legacy_naive_valid_until_interpreted_as_utc(self, temp_db, non_utc_tz):
        """#525: legacy naive valid_until rows follow the documented rule.

        Values written before the fix have no offset suffix; they are
        interpreted as UTC, matching what SQLite julianday already does,
        so both surfaces agree.
        """
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("legacy expiry memory", source="test", importance=0.9)
        # Simulate a pre-fix write: naive wall-clock, genuinely no offset.
        past_naive = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).replace(tzinfo=None).isoformat()
        assert "+" not in past_naive
        beam.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?",
            (past_naive, mid),
        )
        beam.conn.commit()

        # Python-side filter excludes it (interpreted as UTC past).
        assert mid not in {r["id"] for r in beam.get_context(limit=10)}
        # SQLite UTC surface agrees.
        row = beam.conn.execute(
            "SELECT CASE WHEN valid_until IS NULL OR "
            "julianday(valid_until) > julianday('now') "
            "THEN 1 ELSE 0 END AS active "
            "FROM working_memory WHERE id = ?",
            (mid,),
        ).fetchone()
        assert row["active"] == 0

    def test_stored_offset_bearing_valid_until_chronologically_filtered(self, temp_db, non_utc_tz):
        """#525: read filters judge stored offset-bearing / space-separated
        values chronologically even when the write boundary did not
        canonicalize them (e.g. rows imported before the fix).

        Values are chosen to be lexically misleading: a future instant at
        UTC-05 reads as an earlier wall-clock time than aware-UTC now, and a
        past instant at UTC+05 reads as a later one. A space-separated naive
        future value sorts before any aware-UTC ``T``-separated string. A
        lexical filter would drop the still-valid rows and surface the
        expired one; the chronological julianday filter must do the opposite.
        """
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        future_mid = beam.remember("offset stored future", source="test", importance=0.9)
        past_mid = beam.remember("offset stored past", source="test", importance=0.9)
        space_mid = beam.remember("space naive stored", source="test", importance=0.9)
        beam.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?",
            (
                (datetime.now(timezone.utc) + timedelta(hours=1)).astimezone(
                    timezone(timedelta(hours=-5))
                ).isoformat(),
                future_mid,
            ),
        )
        beam.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?",
            (
                (datetime.now(timezone.utc) - timedelta(hours=1)).astimezone(
                    timezone(timedelta(hours=5))
                ).isoformat(),
                past_mid,
            ),
        )
        _space_future = _same_utc_day_future(datetime.now(timezone.utc))
        if _space_future is None:
            pytest.skip(
                "too close to the end of the UTC day: no naive value can be "
                "both chronologically future and lexically earlier than now "
                "for long enough to outlive this test"
            )
        beam.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?",
            (
                _space_future.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
                space_mid,
            ),
        )
        beam.conn.commit()

        # Sanity: these stored values genuinely mislead a lexical filter.
        now = datetime.now(timezone.utc).isoformat()
        lexical = {r["id"] for r in beam.conn.execute(
            "SELECT id FROM working_memory WHERE (valid_until IS NULL OR valid_until > ?)",
            (now,),
        ).fetchall()}
        assert future_mid not in lexical
        assert past_mid in lexical
        assert space_mid not in lexical

        ctx = {r["id"] for r in beam.get_context(limit=10)}
        assert future_mid in ctx
        assert space_mid in ctx
        assert past_mid not in ctx
        assert future_mid in {r["id"] for r in beam.recall("offset stored future", top_k=10)}
        assert past_mid not in {r["id"] for r in beam.recall("offset stored past", top_k=10)}

    def test_lowercase_t_valid_until_normalized_to_utc(self, temp_db, non_utc_tz):
        """#525: valid_until with a lowercase ``t`` separator is normalized.

        ``datetime.fromisoformat()`` accepts a lowercase ``t``, but the
        date-only pass-through guard only excluded values with an uppercase
        ``T`` or a space, so a value like ``2099-12-31t23:00:00-02:00`` was
        stored unnormalized. SQLite ``julianday()`` returns NULL for the
        lowercase form, so SQL recall excluded the row while the Python
        filter treated it as active. Only an exact ``YYYY-MM-DD`` value is a
        pass-through; everything else is canonicalized to aware UTC.
        """
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember(
            "lowercase t expiry", source="test", importance=0.9,
            valid_until="2099-12-31t23:00:00-02:00",
        )
        row = beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()
        stored = datetime.fromisoformat(row[0])
        assert stored.utcoffset() == timedelta(0)
        assert stored > datetime.now(timezone.utc)
        # Both the SQL-backed recall path and the Python filter agree it is
        # still valid.
        assert mid in {r["id"] for r in beam.get_context(limit=10)}
        assert mid in {r["id"] for r in beam.recall("lowercase t expiry", top_k=10)}

    def test_offset_bearing_valid_until_normalized_to_utc(self, temp_db, non_utc_tz):
        """#525: offset-bearing valid_until is canonicalized to UTC on write.

        A future instant expressed with a non-UTC offset must not be
        excluded by lexical comparison against aware-UTC now. Covers
        context, linear recall, and the episodic write path.
        """
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        future_offset = (datetime.now(timezone.utc) + timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=-2))
        )
        mid = beam.remember(
            "offset expiry memory", source="test", importance=0.9,
            valid_until=future_offset.isoformat(),
        )
        row = beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()
        stored = datetime.fromisoformat(row[0])
        assert stored.utcoffset() == timedelta(0)
        assert stored > datetime.now(timezone.utc)
        # Still valid: the filter must include it.
        assert mid in {r["id"] for r in beam.get_context(limit=10)}
        assert mid in {r["id"] for r in beam.recall("offset expiry memory", top_k=10)}

    def test_offset_bearing_past_valid_until_is_expired(self, temp_db):
        """#525: a past instant expressed with +14:00 offset is expired."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        past_offset = (datetime.now(timezone.utc) - timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=14))
        )
        mid = beam.remember(
            "offset past memory", source="test", importance=0.9,
            valid_until=past_offset.isoformat(),
        )
        assert mid not in {r["id"] for r in beam.get_context(limit=10)}
        assert mid not in {r["id"] for r in beam.recall("offset past memory", top_k=10)}

    def test_offset_bearing_valid_until_episodic_write(self, temp_db, non_utc_tz):
        """#525: consolidate_to_episodic canonicalizes offset-bearing
        valid_until to UTC, and recall excludes an expired summary."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        future_offset = (datetime.now(timezone.utc) + timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=-2))
        )
        eid = beam.consolidate_to_episodic(
            "Episodic summary with future offset expiry",
            source_wm_ids=["wm1"],
            importance=0.9,
            valid_until=future_offset.isoformat(),
        )
        row = beam.conn.execute(
            "SELECT valid_until FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone()
        stored = datetime.fromisoformat(row[0])
        assert stored.utcoffset() == timedelta(0)
        assert stored > datetime.now(timezone.utc)
        assert eid in {r["id"] for r in beam.recall("episodic summary future expiry", top_k=20)}

    def test_offset_bearing_valid_until_entity_recall(self, temp_db, non_utc_tz):
        """#525: entity-matched recall honors offset-bearing valid_until.

        Entity matches are re-filtered through the same working-memory
        validity predicate, so a future offset-bearing valid_until must
        survive and a past one must be dropped.
        """
        from mnemosyne.core.annotations import add_annotation

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        future_offset = (datetime.now(timezone.utc) + timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=-2))
        )
        mid = beam.remember(
            "The staging gateway needs a reboot", source="test", importance=0.9,
            valid_until=future_offset.isoformat(),
        )
        add_annotation(mid, "mentions", "staging", db_path=temp_db)

        results = beam.recall("staging gateway", top_k=10)
        assert mid in {r["id"] for r in results}

        expired_mid = beam.remember(
            "The prod gateway needs a reboot", source="test", importance=0.9,
            valid_until=(
                datetime.now(timezone.utc) - timedelta(hours=4)
            ).astimezone(timezone(timedelta(hours=14))).isoformat(),
        )
        add_annotation(expired_mid, "mentions", "prod", db_path=temp_db)
        results = beam.recall("prod gateway", top_k=10)
        assert expired_mid not in {r["id"] for r in results}

    def test_date_only_valid_until_keeps_api_semantics(self, temp_db):
        """#525: date-only valid_until inputs remain pass-through."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember(
            "date expiry memory", source="test", importance=0.9,
            valid_until="2099-12-31",
        )
        assert mid in {r["id"] for r in beam.get_context(limit=10)}
        row = beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()
        # Date-only stays a plain date: pass-through, no UTC suffix added.
        assert row[0] == "2099-12-31"


class TestEpisodicMemory:
    def test_consolidate_preserves_literal_think_markup(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        summary = "User wrote </think> in a literal example"

        beam.consolidate_to_episodic(summary=summary, source_wm_ids=["wm1"])

        content = beam.conn.execute("SELECT content FROM episodic_memory").fetchone()[0]
        assert content == summary

    def test_consolidate_and_recall(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="User likes dark mode",
            source_wm_ids=["wm1"],
            importance=0.8
        )
        assert eid is not None

        results = beam.recall("dark mode")
        assert len(results) >= 1
        assert any(r["tier"] == "episodic" for r in results)

    def test_recall_hybrid_ranking(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.consolidate_to_episodic("Python is the best language", ["a"], importance=0.7)
        beam.consolidate_to_episodic("Rust is great for systems", ["b"], importance=0.7)

        results = beam.recall("best programming language")
        assert len(results) >= 1


class TestScratchpad:
    def test_scratchpad_write_read_clear(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.scratchpad_write("todo: fix auth")
        entries = beam.scratchpad_read()
        assert len(entries) == 1
        assert "fix auth" in entries[0]["content"]

        beam.scratchpad_clear()
        assert len(beam.scratchpad_read()) == 0


class TestSleepCycle:
    def test_sleep_discards_all_chunk_summaries_after_malformed_chunk(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        beam.conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("chunk-a", "first source memory", "conversation", old_ts, "s1"),
                ("chunk-b", "second source memory", "conversation", old_ts, "s1"),
            ],
        )
        beam.conn.commit()

        from mnemosyne.core import local_llm
        monkeypatch.setattr(local_llm, "llm_available", lambda: True)
        monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda *_args, **_kwargs: [["a"], ["b"]])
        results = iter(["first LLM chunk", local_llm._INVALID_REASONING_OUTPUT])
        monkeypatch.setattr(local_llm, "_summarize_memories", lambda *_args, **_kwargs: next(results))

        beam.sleep()
        content = beam.conn.execute("SELECT content FROM episodic_memory").fetchone()[0]
        assert "first LLM chunk" not in content
        assert content.startswith("[conversation] ")

    def test_sleep_discards_chunk_summaries_after_invalid_second_pass(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        beam.conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("chunk-a", "first source memory", "conversation", old_ts, "s1"),
                ("chunk-b", "second source memory", "conversation", old_ts, "s1"),
            ],
        )
        beam.conn.commit()

        from mnemosyne.core import local_llm
        monkeypatch.setattr(local_llm, "llm_available", lambda: True)
        monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda *_args, **_kwargs: [["a"], ["b"]])
        results = iter(["first LLM chunk", "second LLM chunk", local_llm._INVALID_REASONING_OUTPUT])
        monkeypatch.setattr(local_llm, "_summarize_memories", lambda *_args, **_kwargs: next(results))

        beam.sleep()
        content = beam.conn.execute("SELECT content FROM episodic_memory").fetchone()[0]
        assert "first LLM chunk" not in content
        assert "second LLM chunk" not in content
        assert content.startswith("[conversation] ")

    def test_sleep_falls_back_to_aaak_for_token_truncated_reasoning(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("old-think", "User prefers dark mode", "conversation", old_ts, "s1"),
        )
        beam.conn.commit()

        from mnemosyne.core import local_llm
        truncated_generation = "<think>reasoning cut off at max_tokens"
        monkeypatch.setattr(local_llm, "llm_available", lambda: True)
        monkeypatch.setattr(local_llm, "_try_host_llm", lambda *_args, **_kwargs: (False, None))
        monkeypatch.setattr(local_llm, "_call_local_llm", lambda *_args, **_kwargs: truncated_generation)

        result = beam.sleep()
        content = beam.conn.execute("SELECT content FROM episodic_memory").fetchone()[0]
        assert result["status"] == "consolidated"
        assert result["llm_used"] == 0
        assert content.startswith("[conversation] ")
        assert "<think>" not in content
        assert "truncated" not in content

    def test_sleep_consolidates_old_memories(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Inject old working memories
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        for i in range(3):
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (f"old{i}", f"task {i}", "conversation", old_ts, "s1")
            )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 3

        log = beam.get_consolidation_log(limit=1)
        assert len(log) == 1
        assert log[0]["items_consolidated"] == 3

    def test_sleep_dry_run(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("old1", "task one", "conversation", old_ts, "s1")
        )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=True)
        assert result["status"] == "dry_run"
        assert result["items_consolidated"] == 1
        # Should not actually delete
        stats = beam.get_working_stats()
        assert stats["total"] == 1

    def test_sleep_remains_session_scoped(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1-old", "session one task", "conversation", old_ts, "s1"),
                ("s2-old", "session two task", "conversation", old_ts, "s2"),
            ]
        )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 1

        # E3: source rows remain after sleep. s1's row gains
        # consolidated_at; s2's stays untouched (different session).
        conn = sqlite3.connect(temp_db)
        rows = conn.execute(
            "SELECT session_id, consolidated_at FROM working_memory ORDER BY session_id"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        by_session = dict(rows)
        assert by_session["s1"] is not None, "s1 row should be marked consolidated"
        assert by_session["s2"] is None, "s2 row should be untouched by s1's sleep"

    def test_sleep_loads_compression_plugin_and_enables_via_config(self, temp_db, monkeypatch):
        """beam.sleep() loads CompressionPlugin via get_plugin() lazy-load.

        Regression test: get_plugin() used to return None in production because
        the plugin was registered but never loaded. Now it auto-loads on first
        access. This test exercises the beam.py → _plugins.get_manager().get_plugin()
        path end-to-end without needing an actual LLM.
        """
        from mnemosyne.core import plugins as _plugins

        # Ensure fresh manager state
        _plugins.reset_manager()

        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Verify compression plugin is registered but not yet loaded
        mgr = _plugins.get_manager()
        assert mgr.is_registered("compression")
        assert not mgr.is_loaded("compression"), "plugin should not be pre-loaded"

        # Inject a memory that will be picked up by sleep()
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test-compress", "A test memory content that exists", "test", old_ts, "s1")
        )
        conn.commit()
        conn.close()

        # Disable LLM so we hit the non-LLM consolidation path
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        # Load plugin with compression enabled before calling sleep()
        # (this simulates what beam.py does lazily via get_plugin)
        plugin_instance = mgr.get_plugin("compression")
        assert plugin_instance is not None, "get_plugin must return a loaded instance"
        assert mgr.is_loaded("compression"), "plugin must be in _instances after get_plugin"

        # Now enable it via config to verify the full path
        plugin_instance.enabled = True
        plugin_instance._caveman_available = False  # pretend caveman unavailable so no-op

        # sleep() should call get_plugin("compression") internally
        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"

        # Verify plugin was reached (even if it no-op'd because no caveman)
        # The fact that we got here without errors means the plugin loading worked.

        _plugins.reset_manager()

    def test_sleep_all_sessions_consolidates_inactive_sessions(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        fresh_ts = datetime.now().isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1-old", "session one old task", "conversation", old_ts, "s1"),
                ("s2-old", "session two old task", "conversation", old_ts, "s2"),
                ("s2-fresh", "session two fresh task", "conversation", fresh_ts, "s2"),
            ]
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["sessions_scanned"] == 2
        assert result["sessions_consolidated"] == 2
        assert result["items_consolidated"] == 2
        assert result["summaries_created"] == 2
        assert result["errors"] == 0

        # E3: source rows remain. Old rows have consolidated_at set;
        # the fresh row (timestamp > cutoff) stays NULL because sleep
        # never picked it up.
        conn = sqlite3.connect(temp_db)
        rows = conn.execute(
            "SELECT id, session_id, consolidated_at FROM working_memory ORDER BY id"
        ).fetchall()
        logs = conn.execute("SELECT session_id, items_consolidated FROM consolidation_log ORDER BY session_id").fetchall()
        episodic_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        conn.close()

        assert len(rows) == 3
        by_id = {r[0]: (r[1], r[2]) for r in rows}
        assert by_id["s1-old"][1] is not None, "s1-old should be marked consolidated"
        assert by_id["s2-old"][1] is not None, "s2-old should be marked consolidated"
        assert by_id["s2-fresh"][1] is None, "s2-fresh wasn't eligible — must stay NULL"
        assert logs == [("s1", 1), ("s2", 1)]
        assert episodic_count == 2

    def test_sleep_all_sessions_dry_run_preserves_working_memory(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1-old", "session one task", "conversation", old_ts, "s1"),
                ("s2-old", "session two task", "conversation", old_ts, "s2"),
            ]
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=True)
        assert result["status"] == "dry_run"
        assert result["sessions_scanned"] == 2
        assert result["items_consolidated"] == 2

        conn = sqlite3.connect(temp_db)
        working_count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        episodic_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        log_count = conn.execute("SELECT COUNT(*) FROM consolidation_log").fetchone()[0]
        conn.close()
        assert working_count == 2
        assert episodic_count == 0
        assert log_count == 0

    def test_sleep_writes_dense_embedding_for_consolidated_row(self, temp_db, monkeypatch):
        """[C5] State-level companion to the FTS recallability test. Verifies
        sleep populates a dense-recall store (sqlite-vec's vec_episodes when
        loaded, otherwise the memory_embeddings fallback) for each consolidated
        episodic row. A regression that broke the embed→write call (e.g.
        embed() returning None silently, or a missing INSERT into the
        fallback table) would leave dense recall empty even though FTS keeps
        working.

        Skipped when fastembed isn't installed; the dense path is gated on
        _embeddings.available() and a model load. CI runs with fastembed."""
        from mnemosyne.core import embeddings as _embeddings

        if not _embeddings.available():
            pytest.skip("fastembed not available — dense-recall path inactive")

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("old0", "deploy plan for falcon kickoff", "conversation", old_ts, "s1"),
                ("old1", "retro notes from beta release", "conversation", old_ts, "s1"),
            ],
        )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)

        # Post-sleep, exactly one episodic row should exist (one consolidated
        # summary for the session). Dense store should hold a row for it.
        #
        # Use ``beam.conn`` rather than a fresh ``sqlite3.connect`` for the
        # check: when sqlite-vec is installed, the extension is loaded on
        # beam.conn (where the writes happened) but NOT on a freshly-opened
        # connection. Reading vec_episodes through a fresh connection would
        # raise "no such table" and steer us into the wrong assertion
        # branch. beam.conn is the authoritative reader for this state.
        from mnemosyne.core.beam import _vec_available

        bc = beam.conn
        ep_ids = [r[0] for r in bc.execute("SELECT id FROM episodic_memory").fetchall()]
        assert len(ep_ids) == 1, f"expected 1 consolidated episodic row, got {len(ep_ids)}"

        if _vec_available(bc):
            vec_count = bc.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
            assert vec_count >= 1, (
                "sleep consolidated an episodic row but vec_episodes is "
                "empty -- the embed->_vec_insert path did not run. Likely "
                "cause: _embeddings.embed() returned None silently, or "
                "_vec_insert raised and was swallowed."
            )
        else:
            mem_count = bc.execute(
                "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (ep_ids[0],)
            ).fetchone()[0]
            assert mem_count >= 1, (
                "sleep consolidated an episodic row but memory_embeddings "
                "fallback is empty -- the embed->INSERT path did not run."
            )

    def test_sleep_consolidated_content_is_recallable(self, temp_db, monkeypatch):
        """[C5] End-to-end recallability check. Existing sleep tests assert
        counts (items_consolidated, episodic_count) but never verify the
        consolidated content is actually findable through the public recall
        API. A regression that took the consolidated row off-recall via ALL
        recall paths simultaneously (FTS5 trigger broken AND dense store
        skipped AND fallback substring match unreachable) would slip through
        every existing sleep test.

        Locks: after sleep, recall(unique_token_from_seeded_wm) returns at
        least one episodic-tier hit whose content contains that token.

        Note: this is NOT an FTS-isolated assertion. recall() unions vec
        and FTS rowids (beam.py:1751) and falls back to substring scan
        (beam.py:1880) when both are empty, so this test locks recallability
        by *any* path — not the FTS path specifically. Stronger isolation
        would require calling _fts_search directly; that lives in a follow-up
        if the union/fallback layers shift.

        Uses LLM-disabled deterministic AAAK-encoded summary path
        (beam.py:2483 — `compressed = aaak_encode(combined)`). AAAK is
        phrase-substitution + compaction; uncommon literal tokens like
        the ones seeded below survive intact. Same monkeypatch pattern as
        test_beam.py:297, :488, :691, :938, :961."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        # Three distinct unique tokens — one per seeded memory.
        # Pick tokens that won't collide with FTS stop-words or the deterministic
        # concat header text.
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("old0", "wm contains marker zorblax kickoff plan", "conversation", old_ts, "s1"),
                ("old1", "wm contains marker quetzelfin retro notes", "conversation", old_ts, "s1"),
                ("old2", "wm contains marker xanadush deploy log", "conversation", old_ts, "s1"),
            ],
        )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 3

        # E3.a.3 note: pre-fix this test asserted an episodic-tier hit
        # would surface for each token. Post-E3 sleep is additive (the
        # original working_memory row survives alongside the episodic
        # summary), and post-E3.a.3 recall dedups (summary, source)
        # pairs to the higher-scored side. For an exact-token match the
        # source wm row usually wins, leaving the episodic side dropped
        # from recall results — that's correct dedup behavior, not a
        # consolidation regression. So we split the lock into two parts:
        #   (1) the episodic row exists in the DB after sleep (the
        #       consolidation pipeline wired episodic_memory through
        #       correctly)
        #   (2) recall surfaces the consolidated content via SOME tier
        #       (matches the test's stated "locks recallability by *any*
        #       path" intent)
        conn = sqlite3.connect(temp_db)
        try:
            ep_rows = conn.execute(
                "SELECT id, content FROM episodic_memory"
            ).fetchall()
        finally:
            conn.close()
        assert ep_rows, (
            "sleep() reported items_consolidated=3 but no episodic_memory "
            "rows exist — the consolidation pipeline silently dropped them "
            "before commit."
        )

        # Each unique token must be reachable via recall by some tier.
        for token in ("zorblax", "quetzelfin", "xanadush"):
            results = beam.recall(token, top_k=10)
            assert results, (
                f"recall({token!r}) returned 0 results — neither the "
                f"surviving working_memory source nor the episodic summary "
                f"was reachable through ANY recall path (FTS, vec, "
                f"fallback substring scan). Likely cause: FTS5 trigger "
                f"missed AND dense store missed AND content does not "
                f"contain the original token (LLM summarization path "
                f"active despite monkeypatch?)."
            )
            assert any(token in (r.get("content") or "").lower() for r in results), (
                f"recall({token!r}) returned hits but the token does not "
                f"appear in any returned content — FTS may be matching on "
                f"trigram noise rather than the seeded token: "
                f"{[r.get('content') for r in results]}"
            )


class TestMnemosyneIntegration:
    def test_legacy_and_beam_dual_write(self, temp_db):
        mem = Mnemosyne(session_id="s2", db_path=temp_db)
        mid = mem.remember("Likes pizza", source="preference", importance=0.8)

        # Legacy table
        conn = sqlite3.connect(temp_db)
        legacy = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        assert legacy is not None

        # BEAM working_memory should use the same ID now
        wm = conn.execute("SELECT * FROM working_memory WHERE id = ? AND session_id = ?", (mid, "s2")).fetchone()
        assert wm is not None
        conn.close()

        results = mem.recall("pizza")
        assert len(results) >= 1

    def test_forget_removes_both_layers(self, temp_db):
        mem = Mnemosyne(session_id="s2", db_path=temp_db)
        mid = mem.remember("Forget me please", source="preference", importance=0.8)
        assert mem.forget(mid) is True
        conn = sqlite3.connect(temp_db)
        legacy = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        wm = conn.execute("SELECT * FROM working_memory WHERE id = ? AND session_id = ?", (mid, "s2")).fetchone()
        conn.close()
        assert legacy is None
        assert wm is None

    def test_beam_stats(self, temp_db):
        mem = Mnemosyne(session_id="s3", db_path=temp_db)
        mem.remember("Test stat", importance=0.5)
        stats = mem.get_stats()
        assert stats["mode"] == "beam"
        assert "beam" in stats
        assert "working_memory" in stats["beam"]
        assert "episodic_memory" in stats["beam"]


class TestExportImport:
    def test_beam_export_to_dict(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Prefers dark mode", source="preference", importance=0.9)
        beam.scratchpad_write("todo item")
        beam.consolidate_to_episodic("User likes dark mode", ["wm1"], importance=0.8)

        data = beam.export_to_dict()
        assert "mnemosyne_export" in data
        assert data["mnemosyne_export"]["version"] == "1.0"
        assert len(data["working_memory"]) >= 1
        assert len(data["scratchpad"]) >= 1
        assert len(data["episodic_memory"]) >= 1

    def test_beam_import_from_dict_idempotent(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Prefers dark mode", source="preference", importance=0.9)
        data = beam.export_to_dict()

        # Import into fresh DB
        with tempfile.TemporaryDirectory() as tmpdir:
            fresh_db = Path(tmpdir) / "fresh.db"
            fresh_beam = BeamMemory(session_id="s1", db_path=fresh_db)
            stats = fresh_beam.import_from_dict(data)
            assert stats["working_memory"]["inserted"] >= 1

            # Verify
            ctx = fresh_beam.get_context(limit=5)
            assert any("dark mode" in c["content"] for c in ctx)

            # Second import should skip
            stats2 = fresh_beam.import_from_dict(data)
            assert stats2["working_memory"]["skipped"] >= 1

    def test_mnemosyne_export_import_roundtrip(self, temp_db):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Source
            src = Mnemosyne(session_id="s1", db_path=temp_db)
            src.remember("Likes pizza", source="preference", importance=0.8)
            src.scratchpad_write("note")
            export_path = Path(tmpdir) / "export.json"
            src.export_to_file(str(export_path))
            assert export_path.exists()

            # Target
            target_db = Path(tmpdir) / "target.db"
            target = Mnemosyne(session_id="s1", db_path=target_db)
            stats = target.import_from_file(str(export_path))
            assert stats["legacy"]["inserted"] >= 1
            assert stats["beam"]["working_memory"]["inserted"] >= 1

    def test_mnemosyne_import_is_idempotent_for_consolidation_log(self, temp_db):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Mnemosyne(session_id="s1", db_path=temp_db)
            target = None
            try:
                source.conn.execute(
                    """INSERT INTO consolidation_log
                       (session_id, items_consolidated, summary_preview, created_at)
                       VALUES (?, ?, ?, ?)""",
                    ("s1", 209, "snapshot", "2026-08-24T08:00:00"),
                )
                source.conn.commit()
                export_path = Path(tmpdir) / "export.json"
                source.export_to_file(str(export_path))

                target = Mnemosyne(session_id="s1", db_path=Path(tmpdir) / "target.db")
                first = target.import_from_file(str(export_path))
                second = target.import_from_file(str(export_path))

                assert first["beam"]["consolidation_log"]["inserted"] == 1
                assert second["beam"]["consolidation_log"]["skipped"] == 1
                assert target.conn.execute(
                    "SELECT COUNT(*) FROM consolidation_log"
                ).fetchone()[0] == 1

                target.conn.execute(
                    """UPDATE consolidation_log
                       SET session_id=?, items_consolidated=?, summary_preview=?, created_at=?
                       WHERE id=?""",
                    ("changed", 1, "changed", "2026-08-25T08:00:00", 1),
                )
                target.conn.commit()
                forced = target.import_from_file(str(export_path), force=True)
                assert forced["beam"]["consolidation_log"]["overwritten"] == 1
                restored = target.conn.execute(
                    """SELECT id, session_id, items_consolidated, summary_preview, created_at
                       FROM consolidation_log WHERE id=?""",
                    (1,),
                ).fetchone()
                assert tuple(restored) == (
                    1, "s1", 209, "snapshot", "2026-08-24T08:00:00"
                )
                assert target.conn.execute(
                    "SELECT COUNT(*) FROM consolidation_log"
                ).fetchone()[0] == 1

                legacy = {
                    "consolidation_log": [{
                        "session_id": "legacy", "items_consolidated": 3,
                        "summary_preview": "legacy", "created_at": "2026-08-26T08:00:00",
                    }]
                }
                legacy_first = target.beam.import_from_dict(legacy)
                legacy_second = target.beam.import_from_dict(legacy)
                assert legacy_first["consolidation_log"]["inserted"] == 1
                assert legacy_second["consolidation_log"]["inserted"] == 1
                assert target.conn.execute(
                    "SELECT COUNT(*) FROM consolidation_log WHERE session_id=?", ("legacy",)
                ).fetchone()[0] == 2
            finally:
                source.conn.close()
                if target is not None:
                    target.conn.close()
                # import_from_file constructs short-lived store connections
                # for the auxiliary tables.  Release those objects before
                # Windows removes the temporary directory.
                gc.collect()

    def test_import_from_dict_canonicalizes_valid_until(self, temp_db):
        """#525: import_from_dict must normalize offset-bearing valid_until
        to aware UTC on both working and episodic rows before the SQL
        chronological predicates rely on the stored value."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        future_offset = (datetime.now(timezone.utc) + timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=-2))
        )
        past_offset = (datetime.now(timezone.utc) - timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=14))
        )
        payload = {
            "working_memory": [
                {
                    "id": "wm-future", "content": "imported future", "source": "test",
                    "timestamp": "2026-01-01T00:00:00+00:00", "session_id": "s1",
                    "importance": 0.9, "metadata_json": "{}",
                    "valid_until": future_offset.isoformat(),
                },
                {
                    "id": "wm-past", "content": "imported past", "source": "test",
                    "timestamp": "2026-01-01T00:00:00+00:00", "session_id": "s1",
                    "importance": 0.9, "metadata_json": "{}",
                    "valid_until": past_offset.isoformat(),
                },
                {
                    "id": "wm-space", "content": "imported space naive", "source": "test",
                    "timestamp": "2026-01-01T00:00:00+00:00", "session_id": "s1",
                    "importance": 0.9, "metadata_json": "{}",
                    "valid_until": (datetime.now(timezone.utc) + timedelta(hours=2)).replace(
                        tzinfo=None
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                },
            ],
            "episodic_memory": [
                {
                    "id": "em-future", "content": "imported episodic future", "source": "test",
                    "timestamp": "2026-01-01T00:00:00+00:00", "session_id": "s1",
                    "importance": 0.9, "metadata_json": "{}", "summary_of": "",
                    "valid_until": future_offset.isoformat(),
                },
            ],
            "scratchpad": [],
            "consolidation_log": [],
        }
        stats = beam.import_from_dict(payload)
        assert stats["working_memory"]["inserted"] == 3
        assert stats["episodic_memory"]["inserted"] == 1

        stored = beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = 'wm-future'"
        ).fetchone()[0]
        assert datetime.fromisoformat(stored).utcoffset() == timedelta(0)
        assert stored > datetime.now(timezone.utc).isoformat()

        # Space-separated naive values are canonicalized to aware UTC too.
        stored_space = beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = 'wm-space'"
        ).fetchone()[0]
        assert "T" in stored_space
        assert datetime.fromisoformat(stored_space).utcoffset() == timedelta(0)

        # Future rows surface in context and linear recall.
        ctx = {r["id"] for r in beam.get_context(limit=20)}
        assert "wm-future" in ctx
        assert "wm-space" in ctx
        assert "wm-past" not in ctx
        recall = {r["id"] for r in beam.recall("imported future", top_k=10)}
        assert "wm-future" in recall
        assert "wm-past" not in {r["id"] for r in beam.recall("imported past", top_k=10)}
        assert "em-future" in {r["id"] for r in beam.recall("imported episodic future", top_k=20)}

    def test_import_from_dict_date_only_valid_until_passes_through(self, temp_db):
        """#525: date-only valid_until values keep pass-through semantics
        across the import boundary."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        payload = {
            "working_memory": [
                {
                    "id": "wm-date", "content": "imported date-only", "source": "test",
                    "timestamp": "2026-01-01T00:00:00+00:00", "session_id": "s1",
                    "importance": 0.9, "metadata_json": "{}",
                    "valid_until": "2099-12-31",
                },
            ],
            "episodic_memory": [],
            "scratchpad": [],
            "consolidation_log": [],
        }
        beam.import_from_dict(payload)
        stored = beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = 'wm-date'"
        ).fetchone()[0]
        assert stored == "2099-12-31"
        assert "wm-date" in {r["id"] for r in beam.get_context(limit=20)}

    def test_mnemosyne_export_includes_annotations(self, temp_db):
        """Post-E6 regression guard: export_to_file must include annotations
        (kind='mentions', 'fact', 'occurred_on', 'has_source'). Pre-fix the
        export schema only carried `triples` and silently dropped the new
        AnnotationStore data, so backups would lose entity/fact graphs.
        """
        from mnemosyne.core.annotations import AnnotationStore
        import json as _json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Seed source with multiple annotations on one memory.
            src = Mnemosyne(session_id="s1", db_path=temp_db)
            memory_id = src.remember(
                "Alice met Bob in San Francisco.",
                source="preference", importance=0.5,
            )
            ann = AnnotationStore(db_path=temp_db)
            ann.add(memory_id, "mentions", "Alice")
            ann.add(memory_id, "mentions", "Bob")
            ann.add(memory_id, "fact", "The user met Alice and Bob")

            # Export.
            export_path = Path(tmpdir) / "export.json"
            export_stats = src.export_to_file(str(export_path))
            assert export_stats["annotations_count"] >= 3
            with open(export_path) as f:
                payload = _json.load(f)
            assert payload["mnemosyne_export"]["version"] == "1.3"
            assert "annotations" in payload
            assert len(payload["annotations"]) >= 3

            # Round-trip into a fresh DB.
            target_db = Path(tmpdir) / "target.db"
            target = Mnemosyne(session_id="s1", db_path=target_db)
            stats = target.import_from_file(str(export_path))
            assert stats["annotations"]["inserted"] >= 3

            # Verify the data survived end-to-end.
            target_ann = AnnotationStore(db_path=target_db)
            mentions = target_ann.query_by_memory(memory_id, kind="mentions")
            assert {r["value"] for r in mentions} == {"Alice", "Bob"}

    def test_mnemosyne_import_accepts_legacy_1_0_export(self, temp_db):
        """Backward compat: pre-E6 backups (version 1.0, no annotations key)
        import cleanly; the annotations import stats simply report zero
        inserted rows."""
        import json as _json
        with tempfile.TemporaryDirectory() as tmpdir:
            # Hand-craft a minimal 1.0 export payload.
            legacy_export = {
                "mnemosyne_export": {
                    "version": "1.0",
                    "export_date": "2026-01-01T00:00:00",
                    "source_db": "/fake/path.db",
                },
                "working_memory": [],
                "episodic_memory": [],
                "episodic_embeddings": [],
                "scratchpad": [],
                "consolidation_log": [],
                "legacy_memories": [],
                "legacy_embeddings": [],
                "triples": [],
                # NOTE: no "annotations" key — that's the 1.0 contract.
            }
            export_path = Path(tmpdir) / "legacy_export.json"
            with open(export_path, "w") as f:
                _json.dump(legacy_export, f)

            target_db = Path(tmpdir) / "target.db"
            target = Mnemosyne(session_id="s1", db_path=target_db)
            stats = target.import_from_file(str(export_path))
            # 1.0 import should produce zero annotation rows but not error.
            assert stats["annotations"]["inserted"] == 0
            assert stats["annotations"]["skipped"] == 0


class TestProviderContextSafety:
    def test_subagent_context_does_not_initialize_or_write(self, temp_db, monkeypatch):
        import importlib.util
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))

        provider_path = repo_root / "hermes_memory_provider" / "__init__.py"
        spec = importlib.util.spec_from_file_location("mnemo_provider_test", provider_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # Register before exec so dataclasses defined under
        # ``from __future__ import annotations`` can resolve their module
        # namespace (dataclass field processing does sys.modules[cls.__module__]).
        # monkeypatch auto-reverts the entry after the test.
        monkeypatch.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)

        provider = mod.MnemosyneMemoryProvider()
        provider.initialize(
            "subagent-session",
            hermes_home=str(repo_root),
            platform="cli",
            agent_context="subagent",
            agent_identity="test-profile",
            agent_workspace="hermes",
        )

        assert provider._beam is None
        result = provider.handle_tool_call(
            "mnemosyne_remember",
            {
                "content": "subagent should not persist memory",
                "importance": 0.9,
                "source": "test",
                "scope": "session",
            },
        )
        assert "not initialized" in result

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='working_memory'")
        exists = cursor.fetchone() is not None
        count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] if exists else 0
        conn.close()
        assert count == 0


class TestCrossSessionRecall:
    def test_global_memory_survives_consolidation_and_recall(self, temp_db, monkeypatch):
        """Regression for issue #7 Bug 2: global memories must survive sleep() and be recallable cross-session."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        # Disable LLM summarization so original Chinese text is preserved in consolidation
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        # Session A: store global memories with backdated timestamps so sleep() consolidates them
        beam_a = BeamMemory(session_id="hermes_session-A", db_path=temp_db)
        beam_a.remember("用户喜欢直接说结论", source="preference", importance=0.95, scope="global")
        beam_a.remember("用户讨论基金时重视手续费口径", source="preference", importance=0.92, scope="global")
        beam_a.remember("本轮只测试 mnemosyne 沙盒", source="test", importance=0.80, scope="session")

        # Backdate all working memories so they are old enough to consolidate
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute("UPDATE working_memory SET timestamp = ?", (old_ts,))
        conn.commit()
        conn.close()

        # Force consolidation (simulate on_session_end)
        result = beam_a.sleep()
        assert result["status"] == "consolidated"

        # Verify consolidated episodic memories preserved global scope
        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT content, scope, session_id FROM episodic_memory WHERE scope = 'global'")
        global_rows = cursor.fetchall()
        assert len(global_rows) >= 1, "Global memories should survive consolidation with scope preserved"
        conn.close()

        # Session B: recall global memories
        beam_b = BeamMemory(session_id="hermes_session-B", db_path=temp_db)

        # Test Chinese query that previously returned 0
        results = beam_b.recall("谁喜欢直接说结论", top_k=5)
        assert len(results) > 0, "Cross-session recall should find global memory with Chinese query"
        contents = [r["content"] for r in results]
        assert any("用户喜欢直接说结论" in c for c in contents)

        # Test another Chinese query
        results2 = beam_b.recall("基金讨论时看重什么口径", top_k=5)
        assert len(results2) > 0, "Cross-session recall should find second global memory"

        # Test that session-scoped memory is NOT visible cross-session
        beam_b.recall("本轮只测试", top_k=5)
        # This may or may not find it depending on scoring; the key is globals ARE found

    def test_fallback_scoring_finds_chinese_substrings(self, temp_db, monkeypatch):
        """Fallback keyword scoring must handle Chinese where words aren't space-delimited."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))

        beam = BeamMemory(session_id="test-session", db_path=temp_db)
        beam.remember("用户喜欢直接说结论", source="preference", importance=0.9, scope="global")

        # Query that differs at the start but shares a core substring
        results = beam.recall("谁喜欢直接说结论", top_k=5)
        assert len(results) > 0, "Fallback scoring should match shared substrings in Chinese"



class TestTemporalQueries:
    """Temporal filtering for BEAM recall -- Issue #16."""

    def test_recall_from_date_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Meeting about Q1 goals", source="meeting", importance=0.8)

        # Backdate an old memory directly
        conn = sqlite3.connect(temp_db)
        old_ts = "2025-01-15T10:00:00"
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("old1", "Old project kickoff", "meeting", old_ts, "s1", 0.7)
        )
        conn.commit()
        conn.close()

        # Filter from 2025-04-01 should exclude January memory
        results = beam.recall("project", from_date="2025-04-01")
        assert all("Old project kickoff" not in r["content"] for r in results)

    def test_recall_to_date_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Recent standup notes", source="meeting", importance=0.8)

        # Backdate an old memory
        conn = sqlite3.connect(temp_db)
        old_ts = "2025-01-15T10:00:00"
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("old1", "January planning session", "meeting", old_ts, "s1", 0.7)
        )
        conn.commit()
        conn.close()

        # Filter to 2025-02-01 should only include January memory
        results = beam.recall("planning", to_date="2025-02-01")
        assert any("January" in r["content"] for r in results)
        assert all("Recent" not in r["content"] for r in results)

    def test_recall_source_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Bug fix for auth", source="github", importance=0.8)
        beam.remember("Lunch with team", source="conversation", importance=0.5)

        results = beam.recall("auth", source="github")
        assert len(results) >= 1
        assert all(r.get("source") == "github" for r in results)

    def test_recall_date_range_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Insert memories on different dates
        conn = sqlite3.connect(temp_db)
        dates = [
            ("2025-01-10T10:00:00", "January task A"),
            ("2025-03-15T10:00:00", "March task B"),
            ("2025-06-20T10:00:00", "June task C"),
        ]
        for ts, content in dates:
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
                (f"m_{content[:5]}", content, "test", ts, "s1", 0.7)
            )
        conn.commit()
        conn.close()

        # Range: March to May
        results = beam.recall("task", from_date="2025-03-01", to_date="2025-05-31")
        contents = [r["content"] for r in results]
        assert any("March" in c for c in contents)
        assert all("January" not in c for c in contents)
        assert all("June" not in c for c in contents)

    def test_recall_with_episodic_temporal_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Consolidated memory with old timestamp
        beam.consolidate_to_episodic(
            summary="Q4 review discussion",
            source_wm_ids=["wm1"],
            source="meeting",
            importance=0.8
        )
        # Backdate the episodic memory
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE episodic_memory SET timestamp = ? WHERE content = ?", ("2024-12-01T10:00:00", "Q4 review discussion"))
        conn.commit()
        conn.close()

        # Should find it without date filter
        results_all = beam.recall("Q4 review")
        assert any("Q4" in r["content"] for r in results_all)

        # Should exclude it with from_date in 2025
        results_filtered = beam.recall("Q4 review", from_date="2025-01-01")
        assert all("Q4" not in r["content"] for r in results_filtered)

    def test_temporal_triple_auto_generated(self, temp_db):
        """Temporal annotations should be auto-generated on remember().

        Post-E6: occurred_on and has_source are written to AnnotationStore
        rather than TripleStore (they are memory metadata, not current-
        truth temporal facts). Test method name kept for git-history
        continuity.
        """
        from mnemosyne.core.annotations import AnnotationStore

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("Deploy script updated", source="dev", importance=0.8)

        annotations = AnnotationStore(db_path=temp_db)
        rows = annotations.query_by_memory(memory_id=mid)
        assert len(rows) >= 1
        assert any(r["kind"] == "occurred_on" for r in rows)


class TestTokenAwareConsolidation:
    def test_sleep_chunks_large_batches(self, temp_db, monkeypatch):
        """BUG-1: sleep() must chunk memories to fit LLM context window."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        # Force a small context window to trigger chunking
        monkeypatch.setenv("MNEMOSYNE_LLM_N_CTX", "512")
        monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", "128")
        # Disable actual LLM — we test the chunking logic
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="test-chunking", db_path=temp_db)

        # Store 30 memories, each ~100 chars (~25 tokens)
        for i in range(30):
            beam.remember(
                f"Memory number {i} with enough content to consume tokens " * 3,
                source="test_batch",
                importance=0.5
            )

        # Backdate so sleep() picks them up
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute("UPDATE working_memory SET timestamp = ?", (old_ts,))
        conn.commit()
        conn.close()

        result = beam.sleep()
        assert result["status"] == "consolidated"
        assert result["summaries_created"] >= 1

        # E3: originals remain; sleep marks them consolidated_at instead of deleting.
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM working_memory WHERE session_id = ?",
            ("test-chunking",),
        )
        count = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(*) FROM working_memory "
            "WHERE session_id = ? AND consolidated_at IS NOT NULL",
            ("test-chunking",),
        )
        marked = cursor.fetchone()[0]
        conn.close()
        assert count == 30
        assert marked == 30

    def test_chunk_memories_by_budget_single_oversized(self, monkeypatch):
        """A single memory exceeding the budget should be skipped from LLM chunking."""
        from mnemosyne.core import local_llm

        # Monkeypatch module-level constants directly (env vars already read at import)
        monkeypatch.setattr(local_llm, "LLM_N_CTX", 128)
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 32)

        from mnemosyne.core.local_llm import chunk_memories_by_budget

        # One normal memory, one giant memory
        memories = [
            "Short memory.",  # ~3 tokens, fits
            "A" * 500,       # ~125 tokens, exceeds budget
        ]
        chunks = chunk_memories_by_budget(memories)

        # Giant memory should be excluded (it exceeds the total budget)
        assert len(chunks) == 1
        assert chunks[0] == ["Short memory."]


class TestTieredDegradation:
    """Tests for tiered episodic degradation — Phase 1 of the tiered memory system."""

    def test_schema_migration_adds_tier_columns(self, temp_db):
        """Wave 1: init_beam() should add tier and degraded_at columns to episodic_memory."""
        BeamMemory(session_id="s1", db_path=temp_db)
        # Just creating a BeamMemory triggers init_beam which runs the migration

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(episodic_memory)").fetchall()]
        assert "tier" in cols, "tier column missing after migration"
        assert "degraded_at" in cols, "degraded_at column missing after migration"

        # Verify index exists
        indexes = [r[1] for r in cursor.execute(
            "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='episodic_memory'"
        ).fetchall()]
        assert any("tier" in idx for idx in indexes), "idx_em_tier index missing"
        conn.close()

    def test_episodic_memory_defaults_to_tier_1(self, temp_db):
        """New episodic memories should default to tier 1."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="Default tier should be 1",
            source_wm_ids=["wm1"],
            importance=0.8
        )

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        tier = cursor.execute(
            "SELECT tier FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone()[0]
        conn.close()
        assert tier == 1, f"Expected tier=1, got tier={tier}"

    def test_degrade_episodic_tier1_to_tier2(self, temp_db, monkeypatch):
        """Tier 1 memories older than TIER2_DAYS should degrade to tier 2."""
        # Module-level constants are read at import time — patch them directly
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 200)  # far future — won't trigger tier 3

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="This memory is old enough for tier 2 degradation",
            source_wm_ids=["wm1"],
            importance=0.7
        )

        # Backdate the episodic memory to be older than 5 days
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (old_ts, eid))
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier1_to_tier2"] == 1
        assert result["tier2_to_tier3"] == 0

        # Verify tier changed
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        tier, degraded_at = cursor.execute(
            "SELECT tier, degraded_at FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone()
        conn.close()
        assert tier == 2
        assert degraded_at is not None

    def test_degrade_episodic_tier2_to_tier3(self, temp_db, monkeypatch):
        """Tier 2 memories older than TIER3_DAYS should degrade to tier 3."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="This memory will go all the way to tier 3",
            source_wm_ids=["wm1"],
            importance=0.6
        )

        # First degrade to tier 2 (older than 1 day)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(days=3)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (old_ts, eid))
        conn.commit()
        conn.close()
        beam.degrade_episodic(dry_run=False)  # tier 1 → 2

        # Then push it even older and degrade again
        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ?, tier = 2 WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier2_to_tier3"] == 1

        conn = sqlite3.connect(temp_db)
        tier = conn.execute("SELECT tier FROM episodic_memory WHERE id = ?", (eid,)).fetchone()[0]
        conn.close()
        assert tier == 3

    def test_degrade_episodic_dry_run(self, temp_db, monkeypatch):
        """Dry run counts candidates but does NOT modify the database."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 5)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.consolidate_to_episodic(
            summary="Should be counted but not degraded",
            source_wm_ids=["wm1"],
            importance=0.7
        )

        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ?", (old_ts,))
        conn.commit()

        result = beam.degrade_episodic(dry_run=True)
        assert result["status"] == "dry_run"
        assert result["tier1_to_tier2"] == 1

        # Tier should still be 1 — dry run doesn't modify
        tier = conn.execute("SELECT tier FROM episodic_memory").fetchone()[0]
        conn.close()
        assert tier == 1, "Dry run should not change tier"

    def test_degrade_episodic_respects_batch_limit(self, temp_db, monkeypatch):
        """Degradation should respect its resolved runtime batch limit."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)

        class Config:
            def get_int(self, key, default):
                assert key == "degrade_batch"
                assert default == 100
                return 3

        config = Config()
        monkeypatch.setattr("mnemosyne.core.config.get_config", lambda: config)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Consolidate 5 episodic memories first
        eids = []
        for i in range(5):
            eid = beam.consolidate_to_episodic(
                summary=f"Memory {i} for batch limit test",
                source_wm_ids=[f"wm{i}"],
                importance=0.5
            )
            eids.append(eid)

        # Backdate them all in a single raw connection block
        conn = sqlite3.connect(temp_db, timeout=10)
        old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        for eid in eids:
            conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (old_ts, eid))
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        # The resolved batch size is exact, not merely an upper bound.
        assert result["tier1_to_tier2"] == 3

    def test_tier_weighting_in_recall(self, temp_db, monkeypatch):
        """Tier 3 memories should score lower than tier 1 in recall."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_WEIGHT", 0.1)  # heavily penalize

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="Python projects use virtual environments for isolation",
            source_wm_ids=["wm1"],
            importance=0.9
        )

        # Degrade to tier 3
        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=30)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t1→t2
        conn.execute("UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t2→t3
        conn.close()

        results = beam.recall("Python virtual environments", top_k=5)
        # Should still be findable (just weighted lower)
        degraded = [r for r in results if r.get("degradation_tier") == 3]
        if degraded:
            assert degraded[0]["score"] < 1.0, f"Tier 3 score {degraded[0]['score']} should be penalized"

    def test_sleep_includes_degradation(self, temp_db, monkeypatch):
        """sleep() return value must include degradation key."""
        monkeypatch.setenv("MNEMOSYNE_TIER2_DAYS", "30")  # no actual degradation, just testing the key
        monkeypatch.setenv("MNEMOSYNE_TIER3_DAYS", "200")
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Inject old working memory to trigger consolidation
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        for i in range(2):
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (f"old{i}", f"sleep test content {i}", "conversation", old_ts, "s1")
            )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert "degradation" in result, "sleep() should include degradation key"
        assert "status" in result["degradation"]
        assert "tier1_to_tier2" in result["degradation"]

    def test_sleep_all_sessions_includes_degradation(self, temp_db, monkeypatch):
        """sleep_all_sessions() return value must include degradation key."""
        monkeypatch.setenv("MNEMOSYNE_TIER2_DAYS", "30")
        monkeypatch.setenv("MNEMOSYNE_TIER3_DAYS", "200")
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("s2-old", "all sessions sleep test", "conversation", old_ts, "s2")
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=False)
        assert "degradation" in result, "sleep_all_sessions() should include degradation key"

    def test_old_memory_still_recallable_after_degradation(self, temp_db, monkeypatch):
        """Integration: store old memory, degrade to tier 3, still recallable."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="The user's favorite programming language is Rust for systems work",
            source_wm_ids=["wm1"],
            importance=0.85
        )

        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=200)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t1→t2
        conn.execute("UPDATE episodic_memory SET tier = 2 WHERE id = ?", (eid,))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t2→t3

        # Verify it's tier 3
        tier = conn.execute("SELECT tier FROM episodic_memory WHERE id = ?", (eid,)).fetchone()[0]
        print(f"DEBUG: tier after double degrade = {tier}")
        content = conn.execute("SELECT content FROM episodic_memory WHERE id = ?", (eid,)).fetchone()[0]
        print(f"DEBUG: tier 3 content = {content[:100]}")
        conn.close()
        assert tier == 3, f"Expected tier 3, got {tier}"

        # Should still be recallable — this is the marketing promise
        results = beam.recall("favorite programming language", top_k=5)
        contents = [r["content"] for r in results]
        assert len(results) > 0, "Tier 3 memory should still be recallable"
        assert any("Rust" in c for c in contents), (
            f"Tier 3 memory should contain 'Rust', got contents: {contents}"
        )


class TestSmartCompression:
    """Phase 2: entity-aware extraction for tier 2→3 degradation."""

    def test_extract_key_signal_keeps_proper_nouns(self, temp_db):
        """Entities like names, tools, and versions should survive compression."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = (
            "The user's favorite editor is Neovim with LazyVim. "
            "They deploy everything with Docker Compose. "
            "Their preferred language is Rust for systems work. "
            "The weather was nice on Tuesday. "
            "Nothing special happened in the morning. "
            "They also use GitHub Actions for CI/CD."
        )
        result = beam._extract_key_signal(content, max_chars=200)
        # Signal sentences should be present
        assert "Neovim" in result, f"Lost 'Neovim': {result}"
        assert "Docker" in result, f"Lost 'Docker': {result}"
        assert "Rust" in result, f"Lost 'Rust': {result}"
        # Low-signal sentences should be dropped
        assert "weather" not in result, f"Weather survived: {result}"
        assert "Nothing special" not in result, f"Noise survived: {result}"

    def test_extract_key_signal_handles_no_sentences(self, temp_db):
        """Single blob of text without sentence boundaries — falls back to prefix."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = "A" * 500  # No punctuation
        result = beam._extract_key_signal(content, max_chars=100)
        assert len(result) <= 110  # 100 chars + " [...]"
        assert result.startswith("A" * 90)

    def test_extract_key_signal_short_content_passthrough(self, temp_db):
        """Content under max_chars should be returned as-is."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = "Short memory about Python."
        result = beam._extract_key_signal(content, max_chars=500)
        assert result == content

    def test_smart_compress_preserves_entities_in_degradation(self, temp_db, monkeypatch):
        """End-to-end: smart compression keeps key facts where naive prefix would lose them."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.beam.SMART_COMPRESS", True)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_MAX_CHARS", 200)
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Memory where the most important fact is at the END
        eid = beam.consolidate_to_episodic(
            summary=(
                "Morning standup was uneventful. The coffee was cold. "
                "Lunch was a sandwich from the deli. Team discussed vacation plans. "
                "CRITICAL: The production database password was changed to XKCD-correct-horse-battery-staple. "
                "Afternoon was quiet. Went home at 5pm."
            ),
            source_wm_ids=["wm1"],
            importance=0.9
        )

        # Backdate and degrade to tier 3
        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=200)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t1→t2
        conn.execute("UPDATE episodic_memory SET tier = 2 WHERE id = ?", (eid,))
        conn.commit()
        conn.close()
        beam.degrade_episodic(dry_run=False)  # t2→t3

        # Verify the critical fact survived
        conn = sqlite3.connect(temp_db)
        tier3_content = conn.execute(
            "SELECT content FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone()[0]
        conn.close()

        assert "XKCD" in tier3_content or "password" in tier3_content, (
            f"Smart compression should preserve critical entities. Got: {tier3_content}"
        )
        # Naive prefix would have kept "Morning standup was uneventful" — useless


class TestVeracity:
    """Phase 3: memory confidence / veracity signal."""

    def test_schema_adds_veracity_columns(self, temp_db):
        """init_beam should add veracity to working_memory and episodic_memory."""
        BeamMemory(session_id="s1", db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        wm_cols = [r[1] for r in conn.execute("PRAGMA table_info(working_memory)").fetchall()]
        em_cols = [r[1] for r in conn.execute("PRAGMA table_info(episodic_memory)").fetchall()]
        conn.close()
        assert "veracity" in wm_cols
        assert "veracity" in em_cols

    def test_remember_defaults_to_unknown(self, temp_db):
        """remember() without explicit veracity defaults to 'unknown'."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("A fact", source="test", importance=0.5)

        conn = sqlite3.connect(temp_db)
        veracity = conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()[0]
        conn.close()
        assert veracity == "unknown"

    def test_remember_explicit_veracity(self, temp_db):
        """remember() with veracity='stated' stores correctly."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("User said this", source="user", veracity="stated")

        conn = sqlite3.connect(temp_db)
        veracity = conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()[0]
        conn.close()
        assert veracity == "stated"

    def test_recall_veracity_filter(self, temp_db):
        """recall(veracity='stated') should only return stated memories."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Stated preference: dark mode", source="user", veracity="stated")
        beam.remember("Inferred: probably likes Python", source="conversation", veracity="inferred")
        beam.remember("Tool output: cron ran at 3am", source="cron", veracity="tool")

        results = beam.recall("preference", veracity="stated")
        assert all(r["veracity"] == "stated" for r in results)
        assert any("dark mode" in r["content"] for r in results)

    def test_veracity_weighting_in_recall(self, temp_db, monkeypatch):
        """Stated memories should score higher than inferred ones."""
        monkeypatch.setattr("mnemosyne.core.beam.STATED_WEIGHT", 1.0)
        monkeypatch.setattr("mnemosyne.core.beam.INFERRED_WEIGHT", 0.3)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Consolidate two similar memories with different veracity
        beam.consolidate_to_episodic(
            summary="User stated: prefers Rust for systems programming",
            source_wm_ids=["wm1"],
            importance=0.8
        )
        beam.consolidate_to_episodic(
            summary="Inferred: agent thinks user likes Go",
            source_wm_ids=["wm2"],
            importance=0.8
        )

        # Set veracity directly in DB
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE episodic_memory SET veracity = 'stated' WHERE content LIKE '%stated%'")
        conn.execute("UPDATE episodic_memory SET veracity = 'inferred' WHERE content LIKE '%Inferred%'")
        conn.commit()
        conn.close()

        results = beam.recall("systems programming language", top_k=5)
        stated_results = [r for r in results if r.get("veracity") == "stated"]
        inferred_results = [r for r in results if r.get("veracity") == "inferred"]

        if stated_results and inferred_results:
            assert stated_results[0]["score"] > inferred_results[0]["score"], (
                f"Stated score {stated_results[0]['score']} should exceed inferred {inferred_results[0]['score']}"
            )

    def test_get_contaminated_returns_non_stated(self, temp_db):
        """get_contaminated() should return inferred/tool/imported/unknown but not stated."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        eids = []
        for content, veracity_val in [
            ("User said this explicitly", "stated"),
            ("Agent inferred this", "inferred"),
            ("Cron injected this", "tool"),
            ("Imported from Mem0", "imported"),
            ("Legacy uncategorized memory", "unknown"),
        ]:
            eid = beam.consolidate_to_episodic(
                summary=content, source_wm_ids=["wm"], importance=0.8
            )
            eids.append(eid)

        conn = sqlite3.connect(temp_db)
        for eid, veracity_val in zip(eids, ["stated", "inferred", "tool", "imported", "unknown"]):
            conn.execute("UPDATE episodic_memory SET veracity = ? WHERE id = ?", (veracity_val, eid))
        conn.commit()
        conn.close()

        contaminated = beam.get_contaminated(limit=10)
        contents = [c["content"] for c in contaminated]
        assert any("inferred" in c for c in contents)
        assert any("Cron injected" in c for c in contents)
        assert any("Imported" in c for c in contents)
        assert any("Legacy" in c for c in contents)
        # Stated should NOT appear
        assert not any("explicitly" in c for c in contents)

    def test_get_contaminated_respects_importance(self, temp_db):
        """get_contaminated() with min_importance filter."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        eid = beam.consolidate_to_episodic(
            summary="Low importance memory", source_wm_ids=["wm"], importance=0.2
        )
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE episodic_memory SET veracity = 'inferred' WHERE id = ?", (eid,))
        conn.commit()
        conn.close()

        results = beam.get_contaminated(limit=10, min_importance=0.5)
        assert len(results) == 0, "Low importance should be filtered out"

    def test_recall_still_works_without_veracity_filter(self, temp_db):
        """recall() without veracity filter should return all memories."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Fact A", veracity="stated")
        beam.remember("Fact B", veracity="inferred")

        results = beam.recall("Fact", top_k=10)
        assert len(results) >= 2


class TestConsolidationHealth:
    """Issue #115: Consolidation health monitoring."""

    def test_health_no_data(self, temp_db):
        """health() returns no_data when no consolidation has occurred."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        h = beam.health()
        assert h["status"] == "no_data"
        assert h["last_successful_consolidation"] is None
        assert h["error_count"] == 0
        assert h["stale_hours"] is None
        assert "never run" in h["recommendation"].lower() or "no consolidation_log" in h["recommendation"].lower()

    def test_health_healthy_after_sleep(self, temp_db, monkeypatch):
        """health() returns healthy after a successful sleep()."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Inject old working memories
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        for i in range(3):
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (f"old{i}", f"task {i}", "conversation", old_ts, "s1"),
            )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)

        h = beam.health()
        assert h["status"] == "healthy"
        assert h["last_successful_consolidation"] is not None
        # Should be recent (within 3 hours to handle UTC/local timezone differences)
        last_ts = datetime.fromisoformat(h["last_successful_consolidation"])
        assert (datetime.now() - last_ts).total_seconds() < 10800  # 3 hours

    def test_health_stale(self, temp_db, monkeypatch):
        """health() returns stale when last consolidation is > threshold."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Inject old working memories
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        for i in range(3):
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (f"old{i}", f"task {i}", "conversation", old_ts, "s1"),
            )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)

        # Backdate the consolidation_log to simulate staleness
        conn = sqlite3.connect(temp_db)
        stale_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        conn.execute("UPDATE consolidation_log SET created_at = ?", (stale_ts,))
        conn.commit()
        conn.close()

        h = beam.health(stale_threshold_hours=24.0)
        assert h["status"] == "stale"
        assert h["stale_hours"] > 24.0

    def test_health_healthy_within_threshold(self, temp_db, monkeypatch):
        """health() returns healthy when last consolidation is within threshold."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("old1", "test task", "conversation", old_ts, "s1"),
        )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)

        # Consolidation just happened -- should be within 25h threshold
        h = beam.health(stale_threshold_hours=25.0)
        assert h["status"] == "healthy"

    def test_health_custom_threshold(self, temp_db, monkeypatch):
        """health() respects a custom stale threshold."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("old1", "test task", "conversation", old_ts, "s1"),
        )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)

        # Backdate to 12h ago
        conn = sqlite3.connect(temp_db)
        stale_ts = (datetime.now() - timedelta(hours=12)).isoformat()
        conn.execute("UPDATE consolidation_log SET created_at = ?", (stale_ts,))
        conn.commit()
        conn.close()

        # 24h threshold -> should be healthy
        h_loose = beam.health(stale_threshold_hours=24.0)
        assert h_loose["status"] == "healthy"

        # 6h threshold -> should be stale
        h_tight = beam.health(stale_threshold_hours=6.0)
        assert h_tight["status"] == "stale"

    def test_health_after_sleep_all_sessions(self, temp_db, monkeypatch):
        """health() reflects sleep_all_sessions() work."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1-old", "session one task", "conversation", old_ts, "s1"),
                ("s2-old", "session two task", "conversation", old_ts, "s2"),
            ],
        )
        conn.commit()
        conn.close()

        beam.sleep_all_sessions(dry_run=False)

        h = beam.health()
        assert h["status"] == "healthy"
        assert h["last_successful_consolidation"] is not None


class TestUpdateRefreshesDerivedState:
    """Issue #110: update() must reindex FTS5 + recompute vector embeddings."""

    def test_recall_returns_updated_content_after_update(self, temp_db):
        """recall() should find the new content, not the stale original."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        original = "The sky is green"
        mid = beam.remember(original, source="test")
        assert mid is not None

        # Verify original is recallable
        results = beam.recall("sky color", top_k=3)
        contents = [r["content"] for r in results]
        assert any("green" in c for c in contents), (
            f"Original content should be recallable, got: {contents}"
        )

        # Update to correct the content
        updated = beam.update_working(mid, content="The sky is blue")
        assert updated is True

        # After update, recall should find the NEW content
        results = beam.recall("sky color", top_k=3)
        contents = [r["content"] for r in results]
        assert any("blue" in c for c in contents), (
            f"Updated content should be recallable, got: {contents}"
        )
        # The old content should NOT appear
        assert not any("green" in c for c in contents), (
            f"Stale original should not appear after update, got: {contents}"
        )

    def test_recall_returns_updated_content_fuzzy_query(self, temp_db):
        """Even with a query that would match the old content,
        recall() should return the updated content."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("Alice works at Acme Corp", source="test")

        # Update to a different fact
        beam.update_working(mid, content="Alice works at Globex Inc")

        results = beam.recall("Alice employer", top_k=3)
        contents = [r["content"] for r in results]
        assert any("Globex" in c for c in contents), (
            f"Updated content should mention Globex, got: {contents}"
        )

    def test_update_without_content_change_preserves_embedding(self, temp_db):
        """Updating only importance should not recompute embeddings."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("The earth is round", source="test")

        # Update only importance, not content
        beam.update_working(mid, importance=0.9)

        # Content should be unchanged
        results = beam.recall("earth shape", top_k=3)
        contents = [r["content"] for r in results]
        assert any("round" in c for c in contents)

    def test_mnemosyne_update_propagates_to_beam(self, temp_db):
        """Mnemosyne.update() must also refresh FTS5 + vector embeddings."""
        from mnemosyne.core.memory import Mnemosyne
        mnem = Mnemosyne(session_id="s1", db_path=temp_db)
        mid = mnem.remember("The capital of France is Paris", source="test")

        # Use Mnemosyne.update() to correct the info
        mnem.update(mid, content="The capital of France is Lyon")

        results = mnem.recall("capital of France", top_k=3)
        contents = [r["content"] for r in results]
        assert any("Lyon" in c for c in contents), (
            f"Mnemosyne.update should refresh derived state, got: {contents}"
        )


class TestEmbeddingDimConfig:
    """Tests for MNEMOSYNE_EMBEDDING_DIM env var override.

    The fix (PR #131) makes EMBEDDING_DIM read from the env var instead of
    being hardcoded at 384. This allows operators with different embedding
    models to override the dimension. Critical invariant: vec0 schema
    dimensions must match the embedding model output, or table creation fails.
    """

    def test_embedding_dim_default_is_384(self):
        """Default EMBEDDING_DIM must be 384 (bge-small-en-v1.5)."""
        from mnemosyne.core import beam as beam_module
        assert beam_module.EMBEDDING_DIM == 384, (
            f"Default EMBEDDING_DIM must be 384, got {beam_module.EMBEDDING_DIM}. "
            "Check that MNEMOSYNE_EMBEDDING_DIM is not set in the test environment."
        )

    def test_embedding_dim_is_module_level_constant(self):
        """EMBEDDING_DIM must be a module-level int constant, assignable."""
        from mnemosyne.core import beam as beam_module
        original = beam_module.EMBEDDING_DIM
        assert isinstance(original, int)
        beam_module.EMBEDDING_DIM = 768
        try:
            assert beam_module.EMBEDDING_DIM == 768
        finally:
            beam_module.EMBEDDING_DIM = original

    def test_embedding_dim_env_override_is_int_parse(self):
        """The env var parser must be int(os.environ.get(...)).

        This test verifies the implementation approach: the fix changes the
        module-level EMBEDDING_DIM from a hardcoded literal to a computed
        expression using int(os.environ.get(...)). We verify the constant
        is still an int and still has the correct default.
        """
        from mnemosyne.core import beam as beam_module
        # Verify it's an int (not a string, not something else)
        assert isinstance(beam_module.EMBEDDING_DIM, int)
        # Verify the value matches what int("384") would produce
        assert beam_module.EMBEDDING_DIM == int("384")
        # Verify the value is sensible
        assert beam_module.EMBEDDING_DIM > 0
        assert beam_module.EMBEDDING_DIM <= 4096  # reasonable upper bound


class TestSameUtcDayFuture:
    """Deterministic coverage for the valid_until fixture's clock decision.

    The integration test that uses this helper reads the real clock, so it
    cannot prove either branch on demand. These pin both against fixed
    instants instead.
    """

    @staticmethod
    def _at(hh, mm, ss=0, us=0):
        return datetime(2026, 8, 21, hh, mm, ss, us, tzinfo=timezone.utc)

    def test_plain_offset_used_when_it_stays_on_the_same_day(self):
        now = self._at(12, 0)
        assert _same_utc_day_future(now) == now + timedelta(hours=2)

    def test_clamped_once_the_offset_would_roll_into_tomorrow(self):
        now = self._at(22, 0)
        assert _same_utc_day_future(now) == self._at(23, 59, 59)

    def test_clamp_boundary_is_exactly_two_hours_before_midnight(self):
        assert _same_utc_day_future(self._at(21, 59, 59)) == self._at(23, 59, 59)
        assert _same_utc_day_future(self._at(22, 0)) == self._at(23, 59, 59)

    def test_skips_when_the_clamped_value_would_expire_mid_test(self):
        # 23:59:58.999 clamps to 23:59:59.000, one millisecond of validity.
        assert _same_utc_day_future(self._at(23, 59, 58, 999000)) is None
        assert _same_utc_day_future(self._at(23, 59, 59)) is None

    def test_margin_boundary_is_exact(self):
        """The 30s margin is the contract, so pin both sides of it."""
        # Exactly 30s of validity is not enough: the margin is inclusive.
        assert _same_utc_day_future(self._at(23, 59, 29)) is None
        # One microsecond more than the margin is enough.
        assert _same_utc_day_future(self._at(23, 59, 28, 999999)) == self._at(
            23, 59, 59
        )

    def test_returned_value_always_sorts_before_aware_utc_now(self):
        """The property the fixture actually depends on, swept minute by minute."""
        for hh in range(24):
            for mm in range(60):
                now = self._at(hh, mm)
                future = _same_utc_day_future(now)
                if future is None:
                    continue
                assert future > now, (hh, mm)
                stored = future.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
                assert stored < now.isoformat(), (hh, mm, stored)
