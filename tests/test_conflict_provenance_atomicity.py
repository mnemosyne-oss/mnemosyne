"""A conflict is resolved only when supersession and its provenance commit together."""
from contextlib import closing
import json
import sqlite3

import pytest

from mnemosyne.core import beam as bm
from mnemosyne.core import llm_conflict_detector as lcd


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "CONFLICT_LLM_BASE_URL", "https://validator.test/v1")
    monkeypatch.setattr(lcd, "CONFLICT_LLM_API_KEY", "fake-key")
    monkeypatch.setattr(lcd, "validate_conflict_pair", lambda *a, **kw: (True, 0.97, "corrected"))
    monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
    monkeypatch.setattr("mnemosyne.core.model_refresh.infer_model_update_proposals", lambda items: [])
    mem = bm.BeamMemory(session_id="atomic", db_path=tmp_path / "memory.db")
    for i in range(3):
        mem.conn.execute(
            "INSERT INTO working_memory(id,content,source,timestamp,session_id) VALUES (?,?,?,?,?)",
            (f"row-{i}", f"Project meeting date is day {i}", "conversation", f"2026-01-01T{10+i}:00:00", "atomic"),
        )
    mem.conn.commit()
    monkeypatch.setattr(bm.BeamMemory, "_detect_conflicts", lambda self, rows: [("row-0", "row-1")])
    yield mem
    mem.conn.close()


def state(memory):
    return tuple(memory.conn.execute(
        "SELECT valid_until,superseded_by FROM working_memory WHERE id='row-0'"
    ).fetchone())


@pytest.mark.parametrize("api", ["sleep", "sleep_all_sessions"])
@pytest.mark.parametrize("failure", ["ABORT", "FAIL", "ROLLBACK", "IGNORE", "deferred_commit"])
def test_provenance_failure_rolls_back_supersession_and_counts(memory, api, failure, caplog):
    if failure == "deferred_commit":
        memory.conn.execute("PRAGMA foreign_keys=ON")
        assert memory.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        memory.conn.executescript("""
            CREATE TABLE provenance_parent(id INTEGER PRIMARY KEY);
            CREATE TABLE provenance_child(parent_id INTEGER REFERENCES provenance_parent(id)
                DEFERRABLE INITIALLY DEFERRED);
            CREATE TRIGGER reject_provenance AFTER INSERT ON memory_validations
            WHEN NEW.validator='llm_conflict' BEGIN
                INSERT INTO provenance_child(parent_id) VALUES (999);
            END;
        """)
    else:
        action = "RAISE(IGNORE)" if failure == "IGNORE" else f"RAISE({failure}, 'private-provenance-failure')"
        memory.conn.execute(
            "CREATE TRIGGER reject_provenance BEFORE INSERT ON memory_validations "
            f"WHEN NEW.validator='llm_conflict' BEGIN SELECT {action}; END"
        )
    memory.conn.commit()
    assert state(memory) == (None, None)
    result = getattr(memory, api)(force=True)
    assert state(memory) == (None, None)
    assert result["conflicts_resolved"] == 0
    assert result["conflicts_detected_only"] == 1
    assert memory.conn.execute("SELECT COUNT(*) FROM memory_validations").fetchone()[0] == 0
    assert not memory.conn.in_transaction
    assert "private-provenance-failure" not in caplog.text
    # Optional conflict validation failure must not prevent actual consolidation.
    assert memory.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 1
    assert memory.conn.execute("SELECT COUNT(*) FROM working_memory WHERE consolidation_claimed_at IS NOT NULL").fetchone()[0] == 0
    if failure == "deferred_commit":
        assert memory.conn.execute("SELECT COUNT(*) FROM provenance_child").fetchone()[0] == 0


@pytest.mark.parametrize("without_json1", [False, True])
def test_failed_provenance_does_not_consume_successful_supersession(memory, monkeypatch, without_json1):
    if without_json1:
        def authorize(action, arg1, arg2, db, trigger):
            if action == sqlite3.SQLITE_FUNCTION and arg2 == "json_extract":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        memory.conn.set_authorizer(authorize)
    memory.conn.execute("""
        CREATE TRIGGER reject_first BEFORE INSERT ON memory_validations
        WHEN (SELECT superseded_by FROM working_memory WHERE id=NEW.memory_id)='row-1'
        BEGIN SELECT RAISE(ABORT, 'reject first provenance'); END
    """)
    memory.conn.commit()
    monkeypatch.setattr(bm.BeamMemory, "_detect_conflicts", lambda self, rows: [("row-0", "row-1"), ("row-0", "row-2")])
    result = memory.sleep(force=True)
    assert result["conflicts_resolved"] == result["conflicts_detected_only"] == 1
    assert state(memory)[1] == "row-2"
    records = memory.conn.execute("SELECT note FROM memory_validations").fetchall()
    assert len(records) == 1
    assert json.loads(records[0][0])["replacement_id"] == "row-2"
    assert not memory.conn.in_transaction


def test_observer_never_sees_supersession_before_provenance(memory, monkeypatch):
    real_invalidate = memory.invalidate
    observed = []

    def invalidate(mid, replacement_id, **kwargs):
        result = real_invalidate(mid, replacement_id=replacement_id, **kwargs)
        with closing(sqlite3.connect(memory.db_path)) as reader:
            observed.append(reader.execute(
                "SELECT valid_until,superseded_by FROM working_memory WHERE id=?", (mid,)
            ).fetchone())
            assert reader.execute("SELECT COUNT(*) FROM memory_validations").fetchone()[0] == 0
        return result

    monkeypatch.setattr(memory, "invalidate", invalidate)
    result = memory.sleep(force=True)
    assert observed == [(None, None)]
    assert result["conflicts_resolved"] == 1
    assert result["conflicts_detected_only"] == 0
    assert state(memory)[1] == "row-1"
    assert memory.conn.execute("SELECT COUNT(*) FROM memory_validations").fetchone()[0] == 1


@pytest.mark.parametrize("api", ["sleep", "sleep_all_sessions"])
@pytest.mark.parametrize("cache_fails", [False, True])
def test_conflict_cache_io_runs_only_after_durable_provenance(memory, monkeypatch, api, cache_fails):
    """A disposable cache must not control the conflict transaction's outcome."""
    calls = []
    committed = []
    real_after_commit = memory._invalidate_query_cache_after_commit

    def invalidate_cache():
        calls.append(memory.conn.in_transaction)
        if cache_fails:
            raise sqlite3.OperationalError("cache unavailable")

    def after_commit(operation):
        if operation == "sleep.conflict":
            # A separate reader must see both writes before cache I/O starts.
            with closing(sqlite3.connect(memory.db_path)) as reader:
                target = reader.execute(
                    "SELECT superseded_by FROM working_memory WHERE id='row-0'"
                ).fetchone()[0]
                provenance = reader.execute(
                    "SELECT COUNT(*) FROM memory_validations WHERE validator='llm_conflict'"
                ).fetchone()[0]
            committed.append((memory.conn.in_transaction, target, provenance))
        return real_after_commit(operation)

    monkeypatch.setattr(memory, "_invalidate_query_cache", invalidate_cache)
    monkeypatch.setattr(memory, "_invalidate_query_cache_after_commit", after_commit)
    result = getattr(memory, api)(force=True)
    assert result["conflicts_resolved"] == 1
    assert result["conflicts_detected_only"] == 0
    assert state(memory)[1] == "row-1"
    assert committed == [(False, "row-1", 1)]
    assert calls and not any(calls)
    assert memory.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 1
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE consolidation_claimed_at IS NOT NULL"
    ).fetchone()[0] == 0
    assert not memory.conn.in_transaction


@pytest.mark.parametrize("table", ["working_memory", "episodic_memory"])
@pytest.mark.parametrize("replacement", [None, "row-1"])
@pytest.mark.parametrize("caller_owned", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_invalidate_cache_deferral_preserves_transaction_ownership(
    memory, monkeypatch, table, replacement, caller_owned, defer,
):
    target_id = "row-0"
    if table == "episodic_memory":
        target_id = memory.consolidate_to_episodic(
            "Previous project schedule", source_wm_ids=[], source="test",
        )
    memory.conn.commit()
    calls = []
    monkeypatch.setattr(memory, "_invalidate_query_cache", lambda: calls.append(memory.conn.in_transaction))
    if caller_owned:
        memory.conn.execute("BEGIN")
    assert memory.invalidate(
        target_id, replacement_id=replacement, defer_cache_invalidation=defer,
    )
    assert calls == ([] if defer else [caller_owned])
    assert memory.conn.in_transaction == caller_owned
    row = memory.conn.execute(
        f"SELECT valid_until,superseded_by FROM {table} WHERE id=?", (target_id,),
    ).fetchone()
    assert row[0] is not None and row[1] == replacement
    if caller_owned:
        memory.conn.rollback()
        assert tuple(memory.conn.execute(
            f"SELECT valid_until,superseded_by FROM {table} WHERE id=?", (target_id,),
        ).fetchone()) == (None, None)
    else:
        with closing(sqlite3.connect(memory.db_path)) as reader:
            assert tuple(reader.execute(
                f"SELECT valid_until,superseded_by FROM {table} WHERE id=?", (target_id,),
            ).fetchone()) == tuple(row)


@pytest.mark.parametrize("api", ["sleep", "sleep_all_sessions"])
def test_conflict_postcommit_clears_cache_refilled_before_commit(memory, monkeypatch, api):
    from mnemosyne.core.query_cache import QueryCache

    cache = QueryCache(db_path=memory.db_path.parent / "query_cache.db")
    memory._query_cache = cache
    real_invalidate = memory.invalidate
    after_commit = memory._invalidate_query_cache_after_commit
    observed = []
    key = "v2:" + "a" * 64

    def invalidate(mid, replacement_id, **kwargs):
        result = real_invalidate(mid, replacement_id=replacement_id, **kwargs)
        assert memory.conn.in_transaction
        cache.put(key, [{"id": "row-0", "content": "stale"}])
        assert cache.get(key) is not None
        return result

    def check_cleared(operation):
        after_commit(operation)
        if operation == "sleep.conflict":
            # Check immediately: later sleep invalidations must not mask a gap.
            observed.append(cache.get(key))
            with closing(sqlite3.connect(memory.db_path.parent / "query_cache.db")) as reader:
                assert reader.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 0

    monkeypatch.setattr(memory, "invalidate", invalidate)
    monkeypatch.setattr(memory, "_invalidate_query_cache_after_commit", check_cleared)
    try:
        result = getattr(memory, api)(force=True)
        assert result["conflicts_resolved"] == 1
        assert observed == [None]
    finally:
        cache.close()
        memory._query_cache = None
