"""Real vec0 repair integration on a disposable SQLite backup copy."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import pytest

import mnemosyne.doctor as doctor
import mnemosyne.repair as repair

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="repair is Linux-only")


def _copy_with_vec0(path: Path) -> int:
    sqlite_vec = pytest.importorskip("sqlite_vec")
    source = sqlite3.connect(":memory:")
    source.enable_load_extension(True)
    try:
        sqlite_vec.load(source)
    finally:
        source.enable_load_extension(False)
    try:
        source.executescript(
            "CREATE TABLE working_memory (id TEXT PRIMARY KEY, valid_until TEXT, superseded_by TEXT);"
            "CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);"
            "CREATE VIRTUAL TABLE vec_working USING vec0(embedding float[2]);"
            "INSERT INTO working_memory (id) VALUES ('selected'), ('untouched');"
            "INSERT INTO memory_embeddings VALUES ('selected', '[3,4]');"
        )
        rowid = source.execute("SELECT rowid FROM working_memory WHERE id='selected'").fetchone()[0]
        destination = sqlite3.connect(path)
        try:
            source.backup(destination)
        finally:
            destination.close()
        return rowid
    finally:
        source.close()


def test_vec0_bound_apply_creates_backup_and_is_idempotent(tmp_path):
    """This must run on an FS with working linkat(AT_EMPTY_PATH), not a mocked gate."""
    database = tmp_path / "copy.sqlite"
    selected = _copy_with_vec0(database)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(doctor.doctor_report_payload(
        doctor.build_doctor_report("default", database), include_candidates=True)), encoding="utf-8")
    backup = tmp_path / "before.sqlite"
    first = repair.run_repair(
        db_path=database, bank_name="default", report_path=report,
        selections=["working_memory:selected"], apply=True, backup_path=backup,
    )
    assert first["applied"] == [{"table": "working_memory", "status": "applied"}]
    assert first["backup"] is True
    assert backup.exists()
    current = repair._open_writable_repair_db(database)
    try:
        assert [r[0] for r in current.execute("SELECT rowid FROM vec_working")] == [selected]
        assert current.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 2
    finally:
        current.close()
    snapshot = repair._open_writable_repair_db(backup)
    try:
        assert snapshot.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0] == 0
        assert snapshot.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        snapshot.close()
    second_backup = tmp_path / "not-created.sqlite"
    second = repair.run_repair(
        db_path=database, bank_name="default", report_path=report,
        selections=["working_memory:selected"], apply=True, backup_path=second_backup,
    )
    assert second["applied"] == []
    assert second["skipped"] == [
        {"table": "working_memory", "status": "skipped", "reason": "already_present"}]
    assert second["backup"] is False
    assert not second_backup.exists()


@pytest.mark.parametrize("failure", ["load", "disable"])
def test_write_connection_loader_failure_closes_connection(tmp_path, monkeypatch, failure):
    database = tmp_path / "copy.sqlite"
    _copy_with_vec0(database)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(doctor.doctor_report_payload(
        doctor.build_doctor_report("default", database), include_candidates=True)), encoding="utf-8")
    before = database.read_bytes()
    backup = tmp_path / "not-created.sqlite"
    opened = []
    loader_calls = []
    original_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(repair.sqlite3, "connect", tracked_connect)
    if failure == "load":
        def failed_load(conn):
            loader_calls.append(conn)
            return False

        monkeypatch.setattr(repair, "_load_optional_sqlite_vec", failed_load)
    else:
        def failed_disable(conn):
            loader_calls.append(conn)
            raise doctor._SQLiteVecExtensionDisableError()

        monkeypatch.setattr(repair, "_load_optional_sqlite_vec", failed_disable)
    with pytest.raises(repair.RepairError, match="fingerprint|safely opened"):
        repair.run_repair(
            db_path=database, bank_name="default", report_path=report,
            selections=["working_memory:selected"], apply=True, backup_path=backup,
        )
    assert len(loader_calls) == 1
    assert opened[-1] is loader_calls[0]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[-1].execute("SELECT 1")
    assert not backup.exists()
    assert database.read_bytes() == before
