"""Tests for fail-closed legacy SQL dump replay in restore_backup."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mnemosyne.dr import recovery


def _make_simple_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


def _rewrite_backup_payload(backup_path: Path, payload: bytes) -> None:
    import gzip as _gzip

    meta_path = backup_path.with_suffix(".gz.json")
    compressed = _gzip.compress(payload)
    backup_path.write_bytes(compressed)
    meta = json.loads(meta_path.read_text())
    meta["backup_checksum"] = hashlib.sha256(compressed).hexdigest()[:16]
    meta["dump_checksum"] = hashlib.sha256(payload).hexdigest()
    meta_path.write_text(json.dumps(meta, indent=2))


@pytest.mark.parametrize("value", [
    "a\r\nb", "a\rb", "a\u2028b", "a\u2029b", "a\x85b", "a\vb", "a\fb",
    "--not a comment; 'quoted'\r\nCREATE VIRTUAL TABLE ordinary USING fts5(v)",
])
def test_legacy_sql_restore_preserves_literal_separators(tmp_path, value):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE t(v TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))
    backup = recovery.create_backup(source, tmp_path / "backups", format="sql")
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(Path(backup["backup_path"]), target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT v FROM t").fetchall() == [(value,)]


def test_legacy_sql_restore_accepts_multiple_statements_per_line(tmp_path):
    source = tmp_path / "source.db"
    _make_simple_db(source)
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    payload = (
        "BEGIN;CREATE TABLE t(v TEXT);CREATE TABLE audit(v TEXT);"
        "CREATE TRIGGER capture AFTER INSERT ON t BEGIN "
        "INSERT INTO audit VALUES (new.v);INSERT INTO audit VALUES ('semi;colon');END;"
        "INSERT INTO t VALUES ('a;''b\r\n界');COMMIT;-- trailing comment"
    )
    _rewrite_backup_payload(backup, payload.encode())
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT v FROM t").fetchall() == [("a;'b\r\n界",)]
        assert conn.execute("SELECT v FROM audit").fetchall() == [("a;'b\r\n界",), ("semi;colon",)]


@pytest.mark.parametrize("identifier", ['"t;""name"', '`t;``name`', '[t;name]'])
def test_legacy_sql_restore_handles_quoted_identifiers_and_comments(tmp_path, identifier):
    source = tmp_path / "source.db"
    _make_simple_db(source)
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    payload = (
        "-- line comment with ; and ' quote\r\n"
        f"CREATE TABLE {identifier}(v TEXT);"
        "/* block comment with ; and unmatched ' \" [ */"
        f"INSERT INTO {identifier} VALUES ('kept; value');"
        "/* trailing comment; */"
    )
    _rewrite_backup_payload(backup, payload.encode())
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute(f"SELECT v FROM {identifier}").fetchall() == [("kept; value",)]


@pytest.mark.parametrize("options", ["tokenize='porter'", "content='items', content_rowid='id'"])
@pytest.mark.parametrize("value", ["running safely", "create virtual table"])
def test_legacy_fts5_restore_decodes_quoted_ddl_and_skips_shadow_rows(tmp_path, options, value):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute(f"CREATE VIRTUAL TABLE items_fts USING fts5(v, {options})")
        conn.execute("INSERT INTO items VALUES (1, ?)", (value,))
        conn.execute("INSERT INTO items_fts(rowid, v) VALUES (1, ?)", (value,))
    backup = recovery.create_backup(source, tmp_path / "backups", format="sql")
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(Path(backup["backup_path"]), target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT v FROM items_fts").fetchall() == [(value,)]
        assert conn.execute(
            "SELECT rowid FROM items_fts WHERE items_fts MATCH ?", (value.split()[0],)
        ).fetchall() == [(1,)]
        if "porter" in options:
            assert conn.execute(
                "SELECT rowid FROM items_fts WHERE items_fts MATCH ?",
                ("run" if value == "running safely" else "creat",),
            ).fetchall() == [(1,)]
        else:
            conn.execute("UPDATE items SET v = 'changed external value' WHERE id = 1")
            assert conn.execute("SELECT v FROM items_fts").fetchall() == [("changed external value",)]


@pytest.mark.parametrize("name", ["my fts", 'my "fts', "my`fts", "my]fts", "界 fts"])
def test_legacy_sql_restore_virtual_quoted_names(tmp_path, name):
    identifier = '"' + name.replace('"', '""') + '"'
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute(f"CREATE VIRTUAL TABLE {identifier} USING fts5(body)")
        conn.execute(f"INSERT INTO {identifier} VALUES ('orchid ledger')")
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute(f"SELECT body FROM {identifier}").fetchall() == [("orchid ledger",)]
        assert conn.execute(
            f"SELECT rowid FROM {identifier} WHERE {identifier} MATCH 'orchid'"
        ).fetchall() == [(1,)]


@pytest.mark.parametrize(("options", "ordinary"), [
    ("", "ft_stat"), ("content=''", "ft_content"),
    ("content='items'", "ft_content"), ("columnsize=0", "ft_docsize"),
])
def test_legacy_sql_restore_preserves_ordinary_shadow_like_names(tmp_path, options, ordinary):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE items(body TEXT)")
        conn.execute("INSERT INTO items VALUES ('orchid ledger')")
        conn.execute(f"CREATE VIRTUAL TABLE ft USING fts5(body{', ' + options if options else ''})")
        conn.execute("INSERT INTO ft VALUES ('orchid ledger')")
        conn.execute(f"CREATE TABLE {ordinary}(v TEXT)")
        conn.execute(f"INSERT INTO {ordinary} VALUES ('ordinary data must survive')")
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute(f"SELECT v FROM {ordinary}").fetchall() == [("ordinary data must survive",)]
        assert conn.execute("SELECT rowid FROM ft WHERE ft MATCH 'orchid'").fetchall() == [(1,)]


def test_legacy_sql_restore_keeps_unicode_case_distinct_from_shadow_name(tmp_path):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute('CREATE VIRTUAL TABLE "Ä" USING fts5(body)')
        conn.execute('INSERT INTO "Ä" VALUES (\'orchid\')')
        conn.execute('CREATE TABLE "ä_data"(v TEXT)')
        conn.execute('INSERT INTO "ä_data" VALUES (\'ordinary data\')')
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute('SELECT v FROM "ä_data"').fetchall() == [("ordinary data",)]
        assert conn.execute('SELECT body FROM "Ä"').fetchall() == [("orchid",)]


def test_legacy_sql_restore_escaped_single_quote_identifier(tmp_path):
    # Python 3.11's iterdump cannot produce names containing a single quote;
    # exercise the supported SQL replay boundary with an explicit dump instead.
    source = tmp_path / "source.db"
    _make_simple_db(source)
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    payload = "CREATE VIRTUAL TABLE 'my ''fts' USING fts5(body);INSERT INTO 'my ''fts' VALUES ('orchid');"
    _rewrite_backup_payload(backup, payload.encode())
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT body FROM 'my ''fts'").fetchall() == [("orchid",)]


def test_legacy_sql_restore_preserves_vec0_snapshot_rowids(tmp_path):
    sqlite_vec = pytest.importorskip("sqlite_vec")
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("SQLite extension loading is unavailable")
    source = tmp_path / "source.db"
    vector = sqlite_vec.serialize_float32([0.1, 0.2])
    with sqlite3.connect(source) as conn:
        recovery._load_sqlite_vec(conn)
        conn.execute("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[2])")
        conn.execute("INSERT INTO vectors(rowid, embedding) VALUES (7, ?)", (vector,))
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        recovery._load_sqlite_vec(conn)
        assert conn.execute("SELECT rowid, embedding FROM vectors").fetchall() == [(7, vector)]
        assert conn.execute(
            "SELECT rowid FROM vectors WHERE embedding MATCH ? AND k=1", (vector,)
        ).fetchall() == [(7,)]


def test_legacy_sql_restore_preserves_fts_snapshot_rowid_gaps(tmp_path):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE VIRTUAL TABLE ft USING fts5(body)")
        conn.executemany("INSERT INTO ft(rowid, body) VALUES (?, ?)", [(7, "orchid"), (19, "ledger")])
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    target = tmp_path / "restored.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT rowid, body FROM ft ORDER BY rowid").fetchall() == [
            (7, "orchid"), (19, "ledger"),
        ]
        assert conn.execute("SELECT rowid FROM ft WHERE ft MATCH 'ledger'").fetchall() == [(19,)]


@pytest.mark.parametrize(("dump_style", "missing"), [
    ("direct", None), ("bootstrap", None),
    ("direct", "one"), ("bootstrap", "one"), ("bootstrap", "all"),
])
def test_legacy_sql_restore_contentless_shadow_snapshot(tmp_path, dump_style, missing):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE VIRTUAL TABLE ft USING fts5(body, content='')")
        conn.execute("INSERT INTO ft(rowid, body) VALUES (17, 'orchid')")
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    statements = list(recovery._sql_statements(gzip.decompress(backup.read_bytes()).decode()))
    # Python 3.10 emits direct virtual DDL; newer iterdump versions emit a
    # sqlite_master bootstrap. Exercise both explicitly on every interpreter.
    headers = [s for s in statements if s.lstrip().startswith(
        ("CREATE VIRTUAL TABLE ft ", "INSERT INTO sqlite_master(")
    )]
    assert len(headers) == 1
    header = "CREATE VIRTUAL TABLE ft USING fts5(body, content='');"
    if dump_style == "bootstrap":
        header = (
            "PRAGMA writable_schema=ON;\n"
            "INSERT INTO sqlite_master(type,name,tbl_name,rootpage,sql)"
            "VALUES('table','ft','ft',0,'CREATE VIRTUAL TABLE ft USING fts5(body, content='''')');\n"
            "PRAGMA writable_schema=OFF;"
        )
    statements = [header if s == headers[0] else s for s in statements]
    removed = [s for s in statements if s.lstrip().startswith(
        ("CREATE TABLE 'ft_", 'INSERT INTO "ft_') if missing == "all" else ("CREATE TABLE 'ft_idx'",)
    )] if missing else []
    if missing:
        assert len(removed) > 1 if missing == "all" else len(removed) == 1
    # With all physical tables removed, direct DDL is indistinguishable from
    # a supported logical dump. Only bootstrap syntax unambiguously requires
    # a complete physical snapshot in that case.
    _rewrite_backup_payload(backup, "".join(s for s in statements if s not in removed).encode())
    target = tmp_path / "target.db"
    _make_simple_db(target)
    before = target.read_bytes()
    if missing:
        with pytest.raises(RuntimeError, match="Incomplete virtual-table snapshot"):
            recovery.restore_backup(backup, target)
        assert target.read_bytes() == before
    else:
        assert recovery.restore_backup(backup, target)["restored"]
        with sqlite3.connect(target) as conn:
            assert conn.execute("SELECT rowid, body FROM ft").fetchall() == [(17, None)]
            assert conn.execute("SELECT rowid FROM ft WHERE ft MATCH 'orchid'").fetchall() == [(17,)]
    assert not list(tmp_path.glob("*.restore_staged*"))


def test_legacy_sql_restore_accepts_direct_logical_fts_dump(tmp_path):
    source = tmp_path / "source.db"
    _make_simple_db(source)
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    _rewrite_backup_payload(backup, (
        "CREATE VIRTUAL TABLE ft USING fts5(body);"
        "INSERT INTO ft(rowid, body) VALUES(17, 'orchid');"
    ).encode())
    target = tmp_path / "target.db"
    assert recovery.restore_backup(backup, target)["restored"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT rowid, body FROM ft").fetchall() == [(17, "orchid")]
        assert conn.execute("SELECT rowid FROM ft WHERE ft MATCH 'orchid'").fetchall() == [(17,)]


def test_legacy_direct_ddl_without_shadows_replays_encoded_logical_rows(tmp_path):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE VIRTUAL TABLE ft USING fts5(body, content='')")
        conn.execute("INSERT INTO ft(rowid, body) VALUES(17, 'orchid')")
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    statements = list(recovery._sql_statements(gzip.decompress(backup.read_bytes()).decode()))
    logical_rows = [s for s in statements if s.lstrip().startswith('INSERT INTO "ft" VALUES')]
    assert len(logical_rows) == 1
    # Remove every physical shadow statement and normalize to Python 3.10's
    # direct DDL. Without a bootstrap marker this is also valid logical SQL.
    # Its visible NULL row encodes neither the original rowid nor index terms;
    # this compatibility case does not claim a lossless physical round trip.
    payload = "CREATE VIRTUAL TABLE ft USING fts5(body, content='');" + "".join(logical_rows)
    _rewrite_backup_payload(backup, payload.encode())
    target = tmp_path / "target.db"
    result = recovery.restore_backup(backup, target)
    assert result["restored"] and result["integrity_check"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT rowid, body FROM ft").fetchall() == [(1, None)]
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


@pytest.mark.parametrize("tail", [
    "INSERT INTO partial VALUES ('unterminated;\r\nvalue)",
    "CREATE TRIGGER incomplete AFTER INSERT ON partial BEGIN SELECT 1;",
])
def test_legacy_sql_restore_refuses_malformed_tail_without_touching_target(tmp_path, tail):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _make_simple_db(source)
    _make_simple_db(target)
    original = target.read_bytes()
    backup = Path(recovery.create_backup(source, tmp_path / "backups", format="sql")["backup_path"])
    _rewrite_backup_payload(backup, ("CREATE TABLE partial(v TEXT);" + tail).encode())
    with pytest.raises(sqlite3.Error):
        recovery.restore_backup(backup, target)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.restore_staged*"))


def test_legacy_fts5_restore_skips_shadow_ddl(tmp_path, caplog):
    db_path = tmp_path / "src.db"
    backup_dir = tmp_path / "backups"
    restore_target = tmp_path / "fts_restored.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("CREATE VIRTUAL TABLE items_fts USING fts5(v)")
    conn.execute("INSERT INTO items (id, v) VALUES (1, 'hello world')")
    conn.execute("INSERT INTO items_fts (rowid, v) VALUES (1, 'hello world')")
    conn.commit()
    conn.close()

    backup = recovery.create_backup(
        db_path=db_path, backup_dir=backup_dir, format="sql"
    )
    backup_path = Path(backup["backup_path"])

    # shadow DDL present in dump
    assert b"items_fts_config" in gzip.decompress(backup_path.read_bytes())

    import logging

    caplog.set_level(logging.WARNING)
    result = recovery.restore_backup(backup_path, restore_target)
    assert result["restored"] is True
    assert result["integrity_check"] is True

    target_conn = sqlite3.connect(str(restore_target))
    try:
        rows = target_conn.execute("SELECT id, v FROM items").fetchall()
        assert rows == [(1, "hello world")]
        match_rows = target_conn.execute(
            "SELECT rowid FROM items_fts WHERE items_fts MATCH 'hello'"
        ).fetchall()
        assert len(match_rows) == 1
        assert match_rows[0][0] == 1
    finally:
        target_conn.close()

    assert any("Skipping shadow-table DDL" in r.message for r in caplog.records)


def test_legacy_empty_payload_refused(tmp_path):
    db_path = tmp_path / "src.db"
    backup_dir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=backup_dir, format="sql")
    backup_path = Path(backup["backup_path"])

    target = tmp_path / "target.db"
    _make_simple_db(target)
    target_bytes_before = target.read_bytes()

    _rewrite_backup_payload(backup_path, b"")

    with pytest.raises(RuntimeError, match="Refusing to restore"):
        recovery.restore_backup(backup_path, target)

    assert target.read_bytes() == target_bytes_before


def test_legacy_comments_only_payload_refused(tmp_path):
    db_path = tmp_path / "src.db"
    backup_dir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=backup_dir, format="sql")
    backup_path = Path(backup["backup_path"])

    target = tmp_path / "target.db"
    _make_simple_db(target)
    target_bytes_before = target.read_bytes()

    _rewrite_backup_payload(backup_path, b"-- legacy dump\n-- comments only\n")

    with pytest.raises(RuntimeError, match="Refusing to restore"):
        recovery.restore_backup(backup_path, target)

    assert target.read_bytes() == target_bytes_before
