"""Regression tests for the versioned, atomic CJK FTS migration."""
from contextlib import closing
import sqlite3

import pytest

from mnemosyne.core.beam import (
    _BeamConnection, _cjk_bigram, _migrate_cjk_fts, _register_cjk_bigram,
)


@pytest.fixture(params=[False, True], ids=["upstream-raw", "fork-tokenized"])
def legacy(request, tmp_path):
    """Pre-version-marker indexes, with all six original trigger definitions."""
    conn = sqlite3.connect(tmp_path / "legacy.db", factory=_BeamConnection)
    token = "cjk_bigram({}.content)" if request.param else "{}.content"
    conn.executescript("""
        CREATE TABLE working_memory(id TEXT PRIMARY KEY, content TEXT, importance REAL);
        CREATE TABLE episodic_memory(rowid INTEGER PRIMARY KEY, id TEXT, content TEXT, importance REAL);
        CREATE VIRTUAL TABLE fts_working USING fts5(id UNINDEXED, content);
        CREATE VIRTUAL TABLE fts_episodes USING fts5(content, content='episodic_memory', content_rowid='rowid');
    """)
    for prefix, table, fts in [("wm", "working_memory", "fts_working"), ("em", "episodic_memory", "fts_episodes")]:
        if prefix == "wm":
            insert = f"INSERT INTO {fts}(id, content) VALUES(new.id, {token.format('new')});"
            delete = f"DELETE FROM {fts} WHERE id=old.id;"
        else:
            insert = f"INSERT INTO {fts}(rowid, content) VALUES(new.rowid, {token.format('new')});"
            delete = f"INSERT INTO {fts}({fts}, rowid, content) VALUES('delete', old.rowid, {token.format('old')});"
        for suffix, event, body in [("ai", "INSERT", insert), ("ad", "DELETE", delete), ("au", "UPDATE", delete + insert)]:
            conn.execute(f"CREATE TRIGGER {prefix}_{suffix} AFTER {event} ON {table} BEGIN {body} END")
        conn.execute(f"INSERT INTO {table}(id, content, importance) VALUES('old', '前缀腾讯云后缀 Docker Привет café', 0.5)")
    conn.commit()
    yield conn
    conn.close()


def hits(conn, table, word):
    return conn.execute(f'SELECT rowid FROM {table} WHERE {table} MATCH ?', ('"' + word + '"',)).fetchall()


def snapshot(conn):
    return conn.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall()


def test_upgrade_historical_and_new_rows_crud(legacy):
    conn = legacy
    _migrate_cjk_fts(conn)
    for table, fts in [("working_memory", "fts_working"), ("episodic_memory", "fts_episodes")]:
        for term in ("腾讯", "Docker", "Привет", "café"):
            assert hits(conn, fts, term)
        for id_ in ("old", "new"):
            if id_ == "new":
                conn.execute(f"INSERT INTO {table}(id,content) VALUES(?,?)", (id_, "前缀腾讯云后缀"))
            conn.execute(f"UPDATE {table} SET content=? WHERE id=?", ("独立熊猫词汇", id_))
            assert hits(conn, fts, "熊猫")
            conn.execute(f"DELETE FROM {table} WHERE id=?", (id_,))
        assert not hits(conn, fts, "熊猫")
        assert not hits(conn, fts, "腾讯")
    conn.execute("INSERT INTO fts_episodes(fts_episodes) VALUES ('integrity-check')")
    conn.execute("INSERT INTO fts_working(fts_working) VALUES ('integrity-check')")


def test_metadata_and_unchanged_content_do_not_touch_fts(legacy):
    _migrate_cjk_fts(legacy)
    for table in ("working_memory", "episodic_memory"):
        before = legacy.total_changes
        legacy.execute(f"UPDATE {table} SET importance=0.8 WHERE id='old'")
        assert legacy.total_changes - before == 1
        before = legacy.total_changes
        legacy.execute(f"UPDATE {table} SET content=content WHERE id='old'")
        assert legacy.total_changes - before == 1


def test_second_migration_is_noop(legacy):
    _migrate_cjk_fts(legacy)
    before = legacy.total_changes
    triggers = snapshot(legacy)
    _migrate_cjk_fts(legacy)
    assert legacy.total_changes == before
    assert snapshot(legacy) == triggers
    assert legacy.execute("SELECT version FROM cjk_fts_schema").fetchall() == [(1,)]


def test_rebuild_failure_rolls_back_triggers_indexes_and_marker(legacy):
    triggers = snapshot(legacy)
    old_hits = hits(legacy, "fts_episodes", "腾讯")
    def fail(text):
        raise ValueError("injected tokenizer failure")
    legacy.create_function("cjk_bigram", 1, fail)
    with pytest.raises(sqlite3.OperationalError):
        _migrate_cjk_fts(legacy)
    _register_cjk_bigram(legacy)
    assert snapshot(legacy) == triggers
    assert hits(legacy, "fts_episodes", "腾讯") == old_hits
    assert not legacy.execute("SELECT name FROM sqlite_master WHERE name='cjk_fts_schema'").fetchall()
    legacy.execute("INSERT INTO episodic_memory(id, content) VALUES('after', 'rollback write')")
    assert hits(legacy, "fts_episodes", "rollback")
    _migrate_cjk_fts(legacy)
    assert hits(legacy, "fts_episodes", "腾讯")


def test_ddl_failure_rolls_back_trigger_drops(legacy):
    triggers = snapshot(legacy)
    def deny(action, arg1, arg2, db, source):
        if action == sqlite3.SQLITE_CREATE_TRIGGER and arg1 == "em_au":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    legacy.set_authorizer(deny)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            _migrate_cjk_fts(legacy)
    finally:
        # Python 3.10 treats set_authorizer(None) as calling None -> SQLITE_DENY;
        # pass an explicit allow-all callback for 3.10+ compatibility.
        legacy.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
    assert snapshot(legacy) == triggers
    _migrate_cjk_fts(legacy)
    assert hits(legacy, "fts_episodes", "腾讯")


def test_enclosing_transaction_can_rollback_migration(legacy):
    triggers = snapshot(legacy)
    legacy.execute("BEGIN")
    _migrate_cjk_fts(legacy)
    legacy.rollback()
    assert snapshot(legacy) == triggers
    assert not legacy.execute("SELECT name FROM sqlite_master WHERE name='cjk_fts_schema'").fetchall()


def test_future_version_is_rejected(legacy):
    legacy.execute("CREATE TABLE cjk_fts_schema(version INTEGER PRIMARY KEY)")
    legacy.execute("INSERT INTO cjk_fts_schema VALUES (2)")
    with pytest.raises(RuntimeError, match="Unsupported"):
        _migrate_cjk_fts(legacy)
    assert legacy.execute("SELECT version FROM cjk_fts_schema").fetchone()[0] == 2


def test_factory_udf_preserves_mixed_scripts():
    with closing(sqlite3.connect(":memory:", factory=_BeamConnection)) as conn:
        assert conn.execute("SELECT cjk_bigram(?)", ("腾讯Привет café",)).fetchone()[0] == _cjk_bigram("腾讯Привет café")
