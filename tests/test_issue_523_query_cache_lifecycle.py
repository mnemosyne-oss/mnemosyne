"""QueryCache persistence obeys TTL and eviction across restarts (#523)."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import pytest

from mnemosyne.core import query_cache as query_cache_module
from mnemosyne.core.query_cache import QueryCache


def _opaque(label: str) -> str:
    return "v2:" + hashlib.sha256(label.encode()).hexdigest()


def _rows(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return sorted(row[0] for row in conn.execute("SELECT normalized FROM query_cache"))
    finally:
        conn.close()


def _age_rows(db_path: Path, seconds: int, keys: list[str] | None = None) -> None:
    conn = sqlite3.connect(db_path)
    try:
        modifier = f"-{seconds} seconds"
        if keys is None:
            conn.execute("UPDATE query_cache SET created_at = datetime('now', ?)", (modifier,))
        else:
            conn.executemany(
                "UPDATE query_cache SET created_at = datetime('now', ?) WHERE normalized = ?",
                [(modifier, key) for key in keys],
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def cache_db(tmp_path: Path) -> Path:
    return tmp_path / "query_cache.db"


@pytest.mark.parametrize("journal", ["wal", "delete"])
@pytest.mark.parametrize("opaque", [False, True], ids=["semantic", "opaque"])
def test_subsecond_insert_age_survives_restart(cache_db, monkeypatch, journal, opaque):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", journal)
    clock = [1704067200.9]
    monkeypatch.setattr(query_cache_module.time, "time", lambda: clock[0])
    key = _opaque("fractional") if opaque else "alpha beta"
    cache = QueryCache(db_path=cache_db, ttl_seconds=1)
    try:
        # Control SQLite's whole-second wall clock as well as Python's clock.
        cache._conn.create_function("current_timestamp", 0, lambda: "2024-01-01 00:00:00")
        put = cache.put_opaque if opaque else cache.put
        get = cache.get_opaque if opaque else cache.get
        put(key, [{"id": "fresh"}])
        clock[0] = 1704067201.01
        assert get(key) == [{"id": "fresh"}]
    finally:
        cache.close()

    reloaded = QueryCache(db_path=cache_db, ttl_seconds=1)
    try:
        get = reloaded.get_opaque if opaque else reloaded.get
        assert get(key) == [{"id": "fresh"}]
        clock[0] = 1704067201.91
        assert get(key) is None
        assert _rows(cache_db) == []
    finally:
        reloaded.close()


@pytest.mark.parametrize("journal", ["wal", "delete"])
@pytest.mark.parametrize("opaque", [False, True], ids=["semantic", "opaque"])
@pytest.mark.parametrize("removal", ["expiry", "sweep", "size"])
def test_stale_instance_removal_preserves_fresh_peer_row(
    cache_db, monkeypatch, journal, opaque, removal
):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", journal)
    clock = [1704067200.0]
    monkeypatch.setattr(query_cache_module.time, "time", lambda: clock[0])
    key = _opaque("shared") if opaque else "shared query"
    old = QueryCache(db_path=cache_db, ttl_seconds=1000 if removal == "size" else 60, max_size=1)
    peer = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        old_put = old.put_opaque if opaque else old.put
        peer_put = peer.put_opaque if opaque else peer.put
        old_get = old.get_opaque if opaque else old.get
        peer_get = peer.get_opaque if opaque else peer.get
        old_put(key, [{"id": "old"}])
        clock[0] = 1704067250.0
        peer_put(key, [{"id": "fresh"}])
        clock[0] = 1704067261.0
        if removal == "expiry":
            assert old_get(key) is None
        else:
            old.put("another request", [{"id": "other"}])
            assert old_get(key) is None
        assert peer_get(key) == [{"id": "fresh"}]
    finally:
        old.close()
        peer.close()

    reloaded = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        get = reloaded.get_opaque if opaque else reloaded.get
        assert get(key) == [{"id": "fresh"}]
    finally:
        reloaded.close()


@pytest.mark.parametrize("journal", ["wal", "delete"])
@pytest.mark.parametrize("opaque", [False, True], ids=["semantic", "opaque"])
def test_peer_replacement_at_same_timestamp_is_not_deleted(
    cache_db, monkeypatch, journal, opaque
):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", journal)
    clock = [1704067200.5]
    monkeypatch.setattr(query_cache_module.time, "time", lambda: clock[0])
    key = _opaque("same-time") if opaque else "shared query"
    old = QueryCache(db_path=cache_db, ttl_seconds=1)
    peer = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        (old.put_opaque if opaque else old.put)(key, [{"id": "old"}])
        (peer.put_opaque if opaque else peer.put)(key, [{"id": "fresh"}])
        clock[0] = 1704067201.6
        assert (old.get_opaque if opaque else old.get)(key) is None
    finally:
        old.close()
        peer.close()
    reloaded = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        assert (reloaded.get_opaque if opaque else reloaded.get)(key) == [{"id": "fresh"}]
    finally:
        reloaded.close()


def test_fresh_entries_survive_restart_with_their_insert_time(cache_db: Path):
    opaque = _opaque("fresh")
    cache = QueryCache(db_path=cache_db, ttl_seconds=600)
    cache.put("alpha beta", [{"id": "semantic"}], embedding=[1.0])
    cache.put_opaque(opaque, [{"id": "opaque"}])
    cache.close()

    reloaded = QueryCache(db_path=cache_db, ttl_seconds=600)
    try:
        assert reloaded.get("beta alpha") == [{"id": "semantic"}]
        assert reloaded.get_opaque(opaque) == [{"id": "opaque"}]
        # created_at is UTC; a local-time reading would be off by the host offset.
        for key in (reloaded._normalize("alpha beta"), opaque):
            assert abs(reloaded._insert_times[key] - time.time()) < 5
    finally:
        reloaded.close()


def test_expired_entries_miss_after_restart_and_leave_sqlite(cache_db: Path):
    opaque = _opaque("expired")
    cache = QueryCache(db_path=cache_db, ttl_seconds=60)
    cache.put("alpha beta", [{"id": "semantic"}], embedding=[1.0])
    cache.put_opaque(opaque, [{"id": "opaque"}])
    cache.put("gamma delta", [{"id": "kept"}])
    cache.close()
    _age_rows(cache_db, 7200, keys=["alpha beta", opaque])

    reloaded = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        assert reloaded.get("alpha beta") is None
        assert reloaded.get_opaque(opaque) is None
        assert reloaded.get("gamma delta") == [{"id": "kept"}]
        assert reloaded.stats()["size"] == 1
        assert _rows(cache_db) == ["delta gamma"]
    finally:
        reloaded.close()


def test_restored_entry_with_remaining_ttl_expires_on_time(cache_db: Path, monkeypatch):
    opaque = _opaque("remaining")
    cache = QueryCache(db_path=cache_db, ttl_seconds=600)
    cache.put("alpha beta", [{"id": "semantic"}])
    cache.put_opaque(opaque, [{"id": "opaque"}])
    cache.close()
    _age_rows(cache_db, 500)

    reloaded = QueryCache(db_path=cache_db, ttl_seconds=600)
    try:
        assert reloaded.get("alpha beta") == [{"id": "semantic"}]
        assert reloaded.get_opaque(opaque) == [{"id": "opaque"}]
        now = time.time()
        monkeypatch.setattr(query_cache_module.time, "time", lambda: now + 200)
        assert reloaded.get("alpha beta") is None
        assert reloaded.get_opaque(opaque) is None
        assert _rows(cache_db) == []
    finally:
        reloaded.close()


def test_ttl_expiry_in_process_deletes_persisted_rows(cache_db: Path, monkeypatch):
    opaque = _opaque("in-process")
    cache = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        cache.put("alpha beta", [{"id": "semantic"}])
        cache.put_opaque(opaque, [{"id": "opaque"}])
        cache.put("gamma delta", [{"id": "swept"}])
        now = time.time()
        monkeypatch.setattr(query_cache_module.time, "time", lambda: now + 120)

        assert cache.get("alpha beta") is None
        assert cache.get_opaque(opaque) is None
        assert _rows(cache_db) == ["delta gamma"]

        # A later put sweeps the remaining expired entry from SQLite too.
        cache.put("epsilon zeta", [{"id": "new"}])
        assert _rows(cache_db) == ["epsilon zeta"]
    finally:
        cache.close()
    monkeypatch.undo()

    reloaded = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        assert reloaded.get("gamma delta") is None
        assert reloaded.get("epsilon zeta") == [{"id": "new"}]
    finally:
        reloaded.close()


def test_size_eviction_of_opaque_key_does_not_reappear_after_reload(cache_db: Path):
    keys = [_opaque(f"request-{index}") for index in range(3)]
    cache = QueryCache(db_path=cache_db, max_size=2)
    try:
        for index, key in enumerate(keys):
            cache.put_opaque(key, [{"id": f"r{index}"}])
        assert cache.get_opaque(keys[0]) is None
        assert _rows(cache_db) == sorted(keys[1:])
    finally:
        cache.close()

    reloaded = QueryCache(db_path=cache_db, max_size=2)
    try:
        assert reloaded.get_opaque(keys[0]) is None
        assert reloaded.get_opaque(keys[1]) == [{"id": "r1"}]
        assert reloaded.get_opaque(keys[2]) == [{"id": "r2"}]
        assert reloaded.stats()["size"] == 2
    finally:
        reloaded.close()


def test_size_eviction_of_semantic_key_does_not_reappear_after_reload(cache_db: Path):
    cache = QueryCache(db_path=cache_db, max_size=2)
    try:
        cache.put("first query", [{"id": "first"}], embedding=[1.0, 0.0])
        cache.put("second query", [{"id": "second"}], embedding=[0.0, 1.0])
        cache.put("third query", [{"id": "third"}])
        assert cache.get("first query") is None
        assert _rows(cache_db) == ["query second", "query third"]
        assert cache.get("an unrelated question", embedding=[1.0, 0.0]) is None
    finally:
        cache.close()

    reloaded = QueryCache(db_path=cache_db, max_size=2)
    try:
        assert reloaded.get("first query") is None
        assert reloaded.get("second query") == [{"id": "second"}]
        assert reloaded.get("third query") == [{"id": "third"}]
    finally:
        reloaded.close()


@pytest.mark.parametrize("journal", ["wal", "delete"])
@pytest.mark.parametrize("failure", ["second_delete", "commit"])
def test_failed_expiry_cleanup_rolls_back_before_next_put(
    cache_db: Path, monkeypatch, journal: str, failure: str
):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", journal)
    cache = QueryCache(db_path=cache_db, ttl_seconds=60)
    first, second = "alpha beta", "delta gamma"
    try:
        cache.put(first, [{"id": "first"}])
        cache.put(second, [{"id": "second"}])
        if failure == "second_delete":
            # The first DELETE succeeds; a later DELETE in executemany aborts.
            cache._conn.execute(
                "CREATE TRIGGER fail_second BEFORE DELETE ON query_cache "
                "WHEN OLD.normalized = 'delta gamma' "
                "BEGIN SELECT RAISE(ABORT, 'second delete failed'); END"
            )
        else:
            # SQLite allows the DELETE, then rejects COMMIT on the deferred FK.
            cache._conn.execute("PRAGMA foreign_keys=ON")
            cache._conn.execute("CREATE TABLE cleanup_parent (id INTEGER PRIMARY KEY)")
            cache._conn.execute(
                "CREATE TABLE cleanup_child (parent_id INTEGER REFERENCES cleanup_parent(id) "
                "DEFERRABLE INITIALLY DEFERRED)"
            )
            cache._conn.execute(
                "CREATE TRIGGER fail_commit AFTER DELETE ON query_cache "
                "BEGIN INSERT INTO cleanup_child (parent_id) VALUES (999); END"
            )
        cache._conn.commit()
        # Advance only the Python TTL clock; the real SQLite rows stay intact.
        now = time.time()
        monkeypatch.setattr(query_cache_module.time, "time", lambda: now + 120)
        cache.put("epsilon zeta", [{"id": "new"}])

        assert cache._conn.in_transaction is False
        assert _rows(cache_db) == [first, second, "epsilon zeta"]
        # An independent writer must not be held behind the failed cleanup.
        with sqlite3.connect(cache_db, timeout=0.1) as other:
            other.execute(
                "INSERT INTO query_cache (normalized, results_json) VALUES (?, ?)",
                ("independent", "[]"),
            )
        cache.put("iota kappa", [{"id": "later"}])
        assert cache._conn.in_transaction is False
        assert _rows(cache_db) == [first, second, "epsilon zeta", "independent", "iota kappa"]
    finally:
        cache.close()


def test_cleanup_keeps_best_effort_behavior_when_rollback_fails(cache_db: Path, monkeypatch):
    cache = QueryCache(db_path=cache_db, ttl_seconds=60)
    try:
        cache.put("alpha beta", [{"id": "old"}])
        cache._conn.close()
        now = time.time()
        monkeypatch.setattr(query_cache_module.time, "time", lambda: now + 120)
        # A closed real connection rejects both DELETE and ROLLBACK.
        cache.put("gamma delta", [{"id": "new"}])
        assert cache.get("gamma delta") == [{"id": "new"}]
    finally:
        cache.close()


def test_eviction_after_reload_drops_oldest_restored_entry(cache_db: Path):
    keys = [_opaque(f"order-{index}") for index in range(4)]
    cache = QueryCache(db_path=cache_db, max_size=3)
    for index, key in enumerate(keys[:3]):
        cache.put_opaque(key, [{"id": f"r{index}"}])
    cache.close()
    _age_rows(cache_db, 30, keys=[keys[0]])
    _age_rows(cache_db, 20, keys=[keys[1]])
    _age_rows(cache_db, 10, keys=[keys[2]])

    reloaded = QueryCache(db_path=cache_db, max_size=3)
    try:
        reloaded.put_opaque(keys[3], [{"id": "r3"}])
        assert reloaded.get_opaque(keys[0]) is None
        for index, key in enumerate(keys[1:], start=1):
            assert reloaded.get_opaque(key) == [{"id": f"r{index}"}]
        assert _rows(cache_db) == sorted(keys[1:])
    finally:
        reloaded.close()


def test_reload_over_max_size_trims_oldest_rows(cache_db: Path):
    cache = QueryCache(db_path=cache_db, max_size=10)
    for index in range(4):
        cache.put(f"query number{index}", [{"id": index}])
    cache.close()
    for index, age in enumerate((40, 30, 20, 10)):
        _age_rows(cache_db, age, keys=[f"number{index} query"])

    reloaded = QueryCache(db_path=cache_db, max_size=2)
    try:
        assert reloaded.stats()["size"] == 2
        assert reloaded.get("query number0") is None
        assert reloaded.get("query number1") is None
        assert reloaded.get("query number3") == [{"id": 3}]
        assert _rows(cache_db) == ["number2 query", "number3 query"]
    finally:
        reloaded.close()


@pytest.mark.parametrize("journal", ["wal", "delete"])
@pytest.mark.parametrize("opaque", [False, True], ids=["semantic", "opaque"])
@pytest.mark.parametrize(
    "created_at", [None, "", "not-a-timestamp"], ids=["null", "empty", "invalid"]
)
def test_row_without_readable_insert_time_is_dropped_on_load(
    cache_db: Path, monkeypatch, journal: str, opaque: bool, created_at: str | None
):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", journal)
    key = _opaque("undated") if opaque else "alpha beta"
    cache = QueryCache(db_path=cache_db)
    try:
        if opaque:
            cache.put_opaque(key, [{"id": "undated"}])
        else:
            cache.put(key, [{"id": "undated"}])
        cache.put("dated query", [{"id": "dated"}])
    finally:
        cache.close()

    with sqlite3.connect(cache_db) as conn:
        conn.execute(
            "UPDATE query_cache SET created_at = ? WHERE normalized = ?",
            (created_at, key),
        )

    reloaded = QueryCache(db_path=cache_db)
    try:
        assert reloaded._conn is not None
        assert reloaded._conn.in_transaction is False
        if opaque:
            assert reloaded.get_opaque(key) is None
        else:
            assert reloaded.get(key) is None
        assert key not in reloaded._insert_times
        assert reloaded.get("dated query") == [{"id": "dated"}]
        assert _rows(cache_db) == ["dated query"]

        # Cleanup must commit and release the write lock for another process.
        with sqlite3.connect(cache_db, timeout=0.1) as other:
            other.execute(
                "INSERT INTO query_cache (normalized, results_json) VALUES (?, ?)",
                ("independent", "[]"),
            )
        assert reloaded._conn.in_transaction is False
        assert _rows(cache_db) == ["dated query", "independent"]
    finally:
        reloaded.close()
