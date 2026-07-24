"""Tests for the read-only, content-safe doctor foundations."""

import builtins
import json
import os
import sqlite3
import stat
import sys
import types
from pathlib import Path
from typing import cast

import pytest

from mnemosyne import doctor
from mnemosyne.doctor import (
    DoctorReport,
    Finding,
    ReferenceContractRegistry,
    RuntimeDiagnosticsAdapter,
    SQLiteHealthAdapter,
    VectorCoverageAdapter,
    RepairCandidate,
    SEVERITY_WARNING,
    STATUS_OK,
    STATUS_PRESENT_BUT_UNLOADABLE,
    STATUS_UNKNOWN,
    SchemaFingerprint,
    TableFingerprint,
    build_doctor_report,
    doctor_report_payload,
    inspect_schema_fingerprint,
    open_readonly_doctor_db,
    render_doctor_json,
    render_doctor_markdown,
    write_doctor_artifacts_atomically,
)


def _readonly_fixture(tmp_path, ddl_and_rows):
    db_path = tmp_path / "doctor-adapters.db"
    writable = sqlite3.connect(db_path)
    writable.executescript(ddl_and_rows)
    writable.commit()
    writable.close()
    return open_readonly_doctor_db(db_path)


def _queryable_vec0_fixture(
    tmp_path,
    ddl_and_rows,
    *,
    factory=sqlite3.Connection,
    vector_table="vec_working",
    verify_capability=True,
):
    """Simulate a vec0 DDL while retaining a queryable normal-table cache.

    sqlite-vec is intentionally not a test dependency.  Rewriting the catalog
    DDL after a normal table exists lets this connection prove both parts of
    the doctor contract: metadata says vec0 and the exact table query is
    usable.  A reopened connection correctly treats the same fixture as an
    unavailable vec0 extension, which is covered separately below.
    """

    db_path = tmp_path / "queryable-vec0.db"
    conn = sqlite3.connect(db_path, factory=factory)
    conn.executescript(ddl_and_rows)
    conn.execute("PRAGMA writable_schema = ON")
    conn.execute(
        "UPDATE sqlite_master SET sql = "
        f"'CREATE VIRTUAL TABLE {vector_table} USING vec0(embedding float[3])' "
        f"WHERE name = '{vector_table}'"
    )
    conn.execute("PRAGMA writable_schema = OFF")
    assert "USING vec0" in conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (vector_table,)
    ).fetchone()[0]
    if verify_capability:
        assert conn.execute(f'SELECT 1 FROM "{vector_table}" LIMIT 0').fetchone() is None
    conn.execute("PRAGMA query_only = ON")
    return conn


def test_open_readonly_doctor_db_enables_read_safe_connection_options(tmp_path):
    db_path = tmp_path / "doctor.db"
    writable = sqlite3.connect(db_path)
    writable.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)")
    writable.execute("INSERT INTO items (label) VALUES ('safe fixture')")
    writable.commit()
    writable.close()

    conn = open_readonly_doctor_db(db_path)
    try:
        row = conn.execute("SELECT id, label FROM items").fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["label"] == "safe fixture"
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_open_readonly_doctor_db_rejects_schema_and_data_writes(tmp_path):
    db_path = tmp_path / "doctor.db"
    writable = sqlite3.connect(db_path)
    writable.execute("CREATE TABLE existing_items (id INTEGER PRIMARY KEY)")
    writable.commit()
    writable.close()

    conn = open_readonly_doctor_db(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE forbidden_items (id INTEGER PRIMARY KEY)")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO existing_items (id) VALUES (1)")
    finally:
        conn.close()


def test_optional_sqlite_vec_load_keeps_doctor_connection_write_protected(tmp_path):
    sqlite_vec = pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "doctor.db"
    sqlite3.connect(db_path).close()

    conn = open_readonly_doctor_db(db_path)
    try:
        assert doctor._load_optional_sqlite_vec(conn) is True
        with pytest.raises(sqlite3.OperationalError, match="not authorized"):
            conn.execute("SELECT load_extension(?)", ("unused",))
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE forbidden_items (id INTEGER PRIMARY KEY)")
    finally:
        conn.close()


def test_optional_sqlite_vec_load_preserves_unloadable_fallback(tmp_path, monkeypatch):
    db_path = tmp_path / "doctor.db"
    sqlite3.connect(db_path).close()
    failing_sqlite_vec = types.SimpleNamespace(
        load=lambda _conn: (_ for _ in ()).throw(sqlite3.OperationalError("unavailable"))
    )
    monkeypatch.setitem(sys.modules, "sqlite_vec", failing_sqlite_vec)

    conn = open_readonly_doctor_db(db_path)
    try:
        assert doctor._load_optional_sqlite_vec(conn) is False
    finally:
        conn.close()


def test_build_doctor_report_preserves_unloadable_vec0_fallback(tmp_path, monkeypatch):
    db_path = tmp_path / "doctor.db"
    writable = sqlite3.connect(db_path)
    writable.execute("CREATE TABLE vec_working (id INTEGER PRIMARY KEY)")
    writable.execute("PRAGMA writable_schema=ON")
    writable.execute(
        "UPDATE sqlite_master SET sql = "
        "'CREATE VIRTUAL TABLE vec_working USING vec0(embedding float[2])' "
        "WHERE name = 'vec_working'"
    )
    writable.execute("PRAGMA writable_schema=OFF")
    writable.commit()
    writable.close()
    failing_sqlite_vec = types.SimpleNamespace(
        load=lambda _conn: (_ for _ in ()).throw(sqlite3.OperationalError("unavailable"))
    )
    monkeypatch.setitem(sys.modules, "sqlite_vec", failing_sqlite_vec)

    report = build_doctor_report("work", db_path)

    assert report.sqlite_health["vec0"]["status"] == STATUS_PRESENT_BUT_UNLOADABLE
    assert any(finding.code == "sqlite.vec0_capability" for finding in report.findings)


def test_optional_sqlite_vec_load_disables_extensions_after_success(monkeypatch):
    calls: list[object] = []
    fake_sqlite_vec = types.SimpleNamespace(load=lambda _conn: calls.append("load"))

    class TrackingConnection:
        def enable_load_extension(self, enabled) -> None:
            calls.append(enabled)

    monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

    assert doctor._load_optional_sqlite_vec(cast(sqlite3.Connection, TrackingConnection())) is True
    assert calls == [True, "load", False]


def test_optional_sqlite_vec_load_fails_closed_when_disable_fails(monkeypatch):
    calls: list[object] = []
    fake_sqlite_vec = types.SimpleNamespace(load=lambda _conn: calls.append("load"))

    class DisableFailingConnection:
        def enable_load_extension(self, enabled) -> None:
            calls.append(enabled)
            if not enabled:
                raise sqlite3.OperationalError("disable failed")

    monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

    with pytest.raises(doctor._SQLiteVecExtensionDisableError):
        doctor._load_optional_sqlite_vec(cast(sqlite3.Connection, DisableFailingConnection()))
    assert calls == [True, "load", False]


def test_optional_sqlite_vec_load_falls_back_when_enabling_extensions_is_unsupported(monkeypatch):
    calls: list[bool] = []
    fake_sqlite_vec = types.SimpleNamespace(load=lambda _conn: pytest.fail("load should not run"))

    class UnsupportedConnection:
        def enable_load_extension(self, enabled) -> None:
            calls.append(enabled)
            raise sqlite3.OperationalError("extension loading unsupported")

    monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

    assert doctor._load_optional_sqlite_vec(cast(sqlite3.Connection, UnsupportedConnection())) is False
    assert calls == [True]


def test_optional_sqlite_vec_load_fails_closed_when_load_and_disable_fail(monkeypatch):
    fake_sqlite_vec = types.SimpleNamespace(
        load=lambda _conn: (_ for _ in ()).throw(sqlite3.OperationalError("load failed"))
    )

    class LoadAndDisableFailingConnection:
        def enable_load_extension(self, enabled) -> None:
            if not enabled:
                raise sqlite3.OperationalError("disable failed")

    monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

    with pytest.raises(doctor._SQLiteVecExtensionDisableError):
        doctor._load_optional_sqlite_vec(cast(sqlite3.Connection, LoadAndDisableFailingConnection()))


def test_build_doctor_report_fails_closed_when_extensions_cannot_be_disabled(tmp_path, monkeypatch):
    db_path = tmp_path / "doctor.db"
    sqlite3.connect(db_path).close()
    monkeypatch.setattr(
        doctor,
        "_load_optional_sqlite_vec",
        lambda _conn: (_ for _ in ()).throw(doctor._SQLiteVecExtensionDisableError()),
    )

    report = build_doctor_report("work", db_path)

    assert report.sqlite_health["status"] == "unavailable"
    assert report.reference_contracts["status"] == "unavailable"
    assert report.vector_coverage["status"] == "unavailable"
    assert report.hygiene_summary["status"] == "unavailable"


def test_build_doctor_report_uses_optional_sqlite_vec_for_real_vec0_tables(tmp_path):
    sqlite_vec = pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "doctor.db"
    writable = sqlite3.connect(db_path)
    try:
        writable.enable_load_extension(True)
        sqlite_vec.load(writable)
        writable.execute("CREATE VIRTUAL TABLE vec_working USING vec0(embedding float[2])")
        writable.commit()
    finally:
        writable.enable_load_extension(False)
        writable.close()

    report = build_doctor_report("work", db_path)

    vec_table = next(
        (table for table in report.schema_fingerprint.tables if table.name == "vec_working"),
        None,
    )
    assert vec_table is not None
    assert vec_table.status == STATUS_OK
    assert report.sqlite_health["vec0"]["status"] == "available"
    assert not any(finding.code == "sqlite.vec0_capability" for finding in report.findings)


def test_open_readonly_doctor_db_does_not_create_a_missing_path(tmp_path):
    db_path = tmp_path / "missing.db"

    with pytest.raises(sqlite3.OperationalError):
        open_readonly_doctor_db(db_path)

    assert not db_path.exists()


def test_doctor_runtime_dependency_failure_is_safe_and_unknown(tmp_path, monkeypatch):
    db_path = tmp_path / "doctor.db"
    sqlite3.connect(db_path).close()
    raw_secret = "runtime-diagnostics-private-secret"  # nosec - regression fixture

    def fail_runtime_diagnostics():
        raise RuntimeError(f"dependency capability failed: password={raw_secret}")

    monkeypatch.setattr("mnemosyne.runtime_diagnostics.collect_runtime_diagnostics", fail_runtime_diagnostics)

    report = build_doctor_report("work", db_path)
    payload = json.dumps(report.to_dict())

    assert report.runtime_diagnostics == {
        "status": STATUS_UNKNOWN,
        "error_class": "runtime_error",
    }
    assert raw_secret not in payload
    assert "dependency capability failed" not in payload


def test_runtime_diagnostics_marks_sqlite_vec_available_only_after_loading(monkeypatch):
    """The diagnostic must test the extension load, not only SQLite's toggle."""

    from mnemosyne import runtime_diagnostics
    from mnemosyne.core import beam

    calls: list[sqlite3.Connection] = []
    fake_sqlite_vec = types.SimpleNamespace(load=lambda conn: calls.append(conn))
    monkeypatch.setattr(beam, "_SQLITE_VEC_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sqlite_vec", fake_sqlite_vec)

    result = runtime_diagnostics.collect_runtime_diagnostics()

    assert len(calls) == 1
    assert any(
        check == {
            "category": "core",
            "check": "sqlite_vec_available",
            "status": "YES",
            "detail": "",
        }
        for check in result["checks"]
    )


@pytest.mark.parametrize(
    ("raw_python_path", "safe_python_executable"),
    [
        ("/home/doctor-user/.venvs/mnemosyne/bin/python3.13", "python3.13"),
        (r"C:\Users\doctor-user\venvs\mnemosyne\python.exe", "python.exe"),
        (r"\\server\share\venvs\mnemosyne\python.exe", "python.exe"),
    ],
)
def test_runtime_metadata_is_retained_in_safe_doctor_artifacts(
    monkeypatch, raw_python_path, safe_python_executable
):
    """Metadata checks use enum statuses while preserving their values as details."""

    raw_model_path = "/home/doctor-user/.cache/mnemosyne/models/bge-small-en-v1.5"
    metadata = {
        "python_version": "3.13.5",
        "platform": "Linux-6.18.34-rpt-rpi-2712-aarch64-with-glibc2.40",
        "python_executable": raw_python_path,
        "mnemosyne_version": "1.2.3",
        "embeddings_model": f"model cache: {raw_model_path}",
    }
    safe_metadata = {
        **metadata,
        "python_executable": safe_python_executable,
        "embeddings_model": "model cache: <redacted-path>",
    }
    monkeypatch.setattr(
        "mnemosyne.runtime_diagnostics.collect_runtime_diagnostics",
        lambda: {
            "status": "ok",
            "checks": [
                {"check": check, "status": "OK", "detail": detail}
                for check, detail in metadata.items()
            ],
        },
    )

    runtime = RuntimeDiagnosticsAdapter().inspect().metrics
    assert runtime["checks"] == [
        {
            "check": check,
            "status": "OK",
            "detail": safe_metadata[check],
        }
        for check in metadata
    ]

    # The canonical payload boundary must also protect a future runtime source
    # that bypasses the adapter and hands Doctor an absolute executable path.
    payload = doctor_report_payload(
        DoctorReport(
            bank_name="default",
            runtime_diagnostics={
                "status": "ok",
                "checks": [
                    {"check": check, "status": "OK", "detail": detail}
                    for check, detail in metadata.items()
                ],
            },
        )
    )
    json_artifact = render_doctor_json(payload)
    markdown_artifact = render_doctor_markdown(payload)

    assert "## Runtime" in markdown_artifact
    assert raw_python_path not in json_artifact
    assert raw_python_path not in markdown_artifact
    assert raw_model_path not in json_artifact
    assert raw_model_path not in markdown_artifact
    assert safe_python_executable in json_artifact
    assert safe_python_executable in markdown_artifact
    assert "<redacted-path>" in json_artifact
    assert "<redacted-path>" in markdown_artifact
    for detail in safe_metadata.values():
        assert detail in json_artifact
        assert detail in markdown_artifact


def test_safe_preview_caps_raw_text_before_regex_redaction(monkeypatch):
    """Unbounded input must not be handed to Doctor's regex redactors."""

    from mnemosyne import doctor

    captured: list[str] = []

    class CapturePattern:
        def sub(self, _replacement, text):
            captured.append(text)
            return text

    monkeypatch.setattr(doctor, "_PREVIEW_SECRET_ASSIGNMENT", CapturePattern())

    preview = doctor.safe_preview("x" * 100, max_length=10)

    assert captured == ["x" * 40]
    assert preview == "x" * 9 + "…"


def test_doctor_runtime_adapter_does_not_import_or_call_diagnose(tmp_path, monkeypatch):
    """Doctor must rely only on the neutral runtime adapter, never diagnose."""

    db_path = tmp_path / "doctor.db"
    sqlite3.connect(db_path).close()
    real_import = builtins.__import__

    def reject_diagnose_import(name, *args, **kwargs):
        if name == "mnemosyne.diagnose":
            raise AssertionError("doctor must not import diagnose")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_diagnose_import)

    report = build_doctor_report("work", db_path)

    assert report.runtime_diagnostics["status"] in {"ok", "warning", "unavailable"}


def test_schema_fingerprint_includes_tables_columns_and_foreign_keys(tmp_path):
    db_path = tmp_path / "doctor.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE parent (id INTEGER PRIMARY KEY);
        CREATE TABLE child (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES parent(id) ON DELETE CASCADE
        );
        """
    )
    conn.close()

    readonly = open_readonly_doctor_db(db_path)
    try:
        fingerprint = inspect_schema_fingerprint(readonly)
    finally:
        readonly.close()

    child = next(table for table in fingerprint.tables if table.name == "child")
    assert child.status == STATUS_OK
    assert [column.name for column in child.columns] == ["id", "parent_id"]
    assert child.foreign_keys == [
        {
            "from": "parent_id",
            "id": 0,
            "match": "NONE",
            "on_delete": "CASCADE",
            "on_update": "NO ACTION",
            "seq": 0,
            "table": "parent",
            "to": "id",
        }
    ]


def test_schema_fingerprint_bounds_catalog_and_table_metadata_with_stable_signature(tmp_path):
    conn = _readonly_fixture(
        tmp_path,
        """
        CREATE TABLE parent_0 (id INTEGER PRIMARY KEY);
        CREATE TABLE parent_1 (id INTEGER PRIMARY KEY);
        CREATE TABLE parent_2 (id INTEGER PRIMARY KEY);
        CREATE TABLE a_schema (
          id INTEGER PRIMARY KEY,
          first_ref TEXT REFERENCES parent_0(id),
          second_ref TEXT REFERENCES parent_1(id),
          third_ref TEXT REFERENCES parent_2(id)
        );
        CREATE TABLE b_schema (id INTEGER PRIMARY KEY);
        CREATE TABLE c_schema (id INTEGER PRIMARY KEY);
        """,
    )
    try:
        first = inspect_schema_fingerprint(conn, scan_limit=2)
        second = inspect_schema_fingerprint(conn, scan_limit=2)
    finally:
        conn.close()

    payload = DoctorReport(bank_name="work", schema_fingerprint=first).to_dict()["schema_fingerprint"]
    assert [table.name for table in first.tables] == ["a_schema", "b_schema"]
    assert first.tables_truncated is True
    assert [column.name for column in first.tables[0].columns] == ["id", "first_ref"]
    assert first.tables[0].columns_truncated is True
    assert len(first.tables[0].foreign_keys) == 2
    assert first.tables[0].foreign_keys_truncated is True
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == json.dumps(
        DoctorReport(bank_name="work", schema_fingerprint=second).to_dict()["schema_fingerprint"],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "c_schema" not in json.dumps(payload)


def test_schema_fingerprint_rejects_invalid_scan_limit(tmp_path):
    conn = _readonly_fixture(tmp_path, "CREATE TABLE items (id INTEGER PRIMARY KEY);")
    try:
        with pytest.raises(ValueError, match="scan_limit"):
            inspect_schema_fingerprint(conn, scan_limit=0)
    finally:
        conn.close()


def test_schema_introspection_only_labels_confirmed_vec0_as_unloadable(tmp_path):
    db_path = tmp_path / "doctor.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE vec_items (id INTEGER PRIMARY KEY)")
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "UPDATE sqlite_master SET sql = "
        "'CREATE VIRTUAL TABLE vec_items USING vec0(id INTEGER PRIMARY KEY)' "
        "WHERE name = 'vec_items'"
    )
    conn.execute("PRAGMA writable_schema=OFF")
    conn.commit()
    conn.close()

    readonly = open_readonly_doctor_db(db_path)
    try:
        fingerprint = inspect_schema_fingerprint(readonly)
    finally:
        readonly.close()

    vec_items = next(table for table in fingerprint.tables if table.name == "vec_items")
    assert vec_items.status == STATUS_PRESENT_BUT_UNLOADABLE
    assert vec_items.columns == []
    assert vec_items.foreign_keys == []


class _GenericOperationalErrorConnection(sqlite3.Connection):
    """Simulate an unrelated runtime failure during table introspection."""

    def execute(self, sql, parameters=()):
        if sql.startswith('PRAGMA table_xinfo("vec_items")'):
            raise sqlite3.OperationalError("disk I/O error: raw memory content")
        return super().execute(sql, parameters)


class _CatalogErrorConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if sql.startswith("SELECT name, sql FROM sqlite_master"):
            raise sqlite3.DatabaseError("catalog failure with private body")
        return super().execute(sql, parameters)


class _ColumnsErrorConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if sql.startswith('PRAGMA table_xinfo("working_memory")'):
            raise sqlite3.OperationalError("column failure with secret content")
        return super().execute(sql, parameters)


class _TrackingTableXinfoCursor:
    def __init__(self, cursor, row_counts):
        self._cursor = cursor
        self._row_counts = row_counts

    def __iter__(self):
        for row in self._cursor:
            self._row_counts[-1] += 1
            yield row


class _BoundedTableXinfoConnection(sqlite3.Connection):
    table_xinfo_row_counts: list[int]

    def execute(self, sql, parameters=()):
        cursor = super().execute(sql, parameters)
        if sql.startswith('PRAGMA table_xinfo("working_memory")'):
            self.table_xinfo_row_counts.append(0)
            return cast(
                sqlite3.Cursor,
                _TrackingTableXinfoCursor(cursor, self.table_xinfo_row_counts),
            )
        return cursor


class _CountErrorConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if sql.startswith("SELECT 1 FROM (SELECT 1 FROM memory_embeddings"):
            raise sqlite3.OperationalError("count failure with embedding_json")
        return super().execute(sql, parameters)


class _VecWorkingCapabilityErrorConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if sql == 'SELECT 1 FROM "vec_working" LIMIT 0':
            raise sqlite3.OperationalError("vec read failure with private content")
        return super().execute(sql, parameters)


class _VecEpisodesCountErrorConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if sql.startswith("SELECT 1 FROM (SELECT 1 FROM vec_episodes) LIMIT ?"):
            raise sqlite3.OperationalError("vec count failure with private embedding")
        return super().execute(sql, parameters)


def test_schema_introspection_requires_the_vec0_error_signature(tmp_path):
    db_path = tmp_path / "doctor.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE vec_items (id INTEGER PRIMARY KEY)")
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "UPDATE sqlite_master SET sql = "
        "'CREATE VIRTUAL TABLE vec_items USING vec0(id INTEGER PRIMARY KEY)' "
        "WHERE name = 'vec_items'"
    )
    conn.execute("PRAGMA writable_schema=OFF")
    conn.commit()
    conn.close()

    readonly = sqlite3.connect(
        f"{db_path.as_uri()}?mode=ro",
        uri=True,
        factory=_GenericOperationalErrorConnection,
    )
    try:
        fingerprint = inspect_schema_fingerprint(readonly)
    finally:
        readonly.close()

    vec_items = next(table for table in fingerprint.tables if table.name == "vec_items")
    assert vec_items.status == STATUS_UNKNOWN
    assert vec_items.error_class == "operational_error"
    assert "raw memory content" not in vec_items.detail


def test_schema_introspection_handles_a_non_sqlite_file_without_crashing(tmp_path):
    db_path = tmp_path / "not-a-database.db"
    db_path.write_bytes(b"not an sqlite database")

    readonly = open_readonly_doctor_db(db_path)
    try:
        fingerprint = inspect_schema_fingerprint(readonly)
    finally:
        readonly.close()

    assert fingerprint.status == STATUS_UNKNOWN
    assert fingerprint.error_class == "database_error"
    assert fingerprint.tables == []
    assert "not an sqlite database" not in fingerprint.detail


def test_adapter_catalog_errors_are_unknown_and_redacted():
    conn = sqlite3.connect(":memory:", factory=_CatalogErrorConnection)
    try:
        result = ReferenceContractRegistry(conn).inspect()
    finally:
        conn.close()

    assert set(result.metrics) == {
        "memory_embeddings",
        "vec_working",
        "graph_edges",
        "episodic_memory",
        "canonical_facts",
        "triples",
    }
    assert all(metric == {"status": STATUS_UNKNOWN, "error_class": "database_error"} for metric in result.metrics.values())
    assert "private body" not in json.dumps(result.metrics)


def test_adapter_column_errors_do_not_mask_contracts_as_empty_or_complete():
    conn = sqlite3.connect(":memory:", factory=_ColumnsErrorConnection)
    conn.execute("CREATE TABLE working_memory (id TEXT PRIMARY KEY)")
    try:
        result = ReferenceContractRegistry(conn).inspect()
    finally:
        conn.close()

    assert all(metric == {"status": STATUS_UNKNOWN, "error_class": "operational_error"} for metric in result.metrics.values())
    assert "secret content" not in json.dumps(result.metrics)


def test_table_xinfo_scan_is_bounded_and_truncation_prevents_complete_contract_claims():
    conn = sqlite3.connect(":memory:", factory=_BoundedTableXinfoConnection)
    conn.table_xinfo_row_counts = []
    conn.execute(
        "CREATE TABLE working_memory (filler_one TEXT, filler_two TEXT, id TEXT PRIMARY KEY)"
    )
    try:
        first_fingerprint = inspect_schema_fingerprint(conn, scan_limit=2)
        second_fingerprint = inspect_schema_fingerprint(conn, scan_limit=2)
        first_contracts = ReferenceContractRegistry(conn, scan_limit=2).inspect()
        second_contracts = ReferenceContractRegistry(conn, scan_limit=2).inspect()
        first_coverage = VectorCoverageAdapter(conn, scan_limit=2).inspect()
        second_coverage = VectorCoverageAdapter(conn, scan_limit=2).inspect()
    finally:
        conn.close()

    table = first_fingerprint.tables[0]
    assert [column.name for column in table.columns] == ["filler_one", "filler_two"]
    assert table.columns_truncated is True
    assert conn.table_xinfo_row_counts == [3, 3, 3, 3, 3, 3]
    assert all(
        metric == {"status": STATUS_UNKNOWN, "columns_truncated": True}
        for metric in first_contracts.metrics.values()
    )
    assert json.dumps(
        DoctorReport(bank_name="work", schema_fingerprint=first_fingerprint).to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ) == json.dumps(
        DoctorReport(bank_name="work", schema_fingerprint=second_fingerprint).to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert json.dumps(first_contracts.metrics, sort_keys=True, separators=(",", ":")) == json.dumps(
        second_contracts.metrics, sort_keys=True, separators=(",", ":")
    )
    assert first_coverage.metrics["working"] == {
        "status": STATUS_UNKNOWN,
        "columns_truncated": True,
    }
    assert json.dumps(first_coverage.metrics, sort_keys=True, separators=(",", ":")) == json.dumps(
        second_coverage.metrics, sort_keys=True, separators=(",", ":")
    )


def test_adapter_count_errors_remain_unknown_without_zero_counts():
    conn = sqlite3.connect(":memory:", factory=_CountErrorConnection)
    conn.executescript(
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        INSERT INTO working_memory VALUES ('live');
        INSERT INTO memory_embeddings VALUES ('live', '[private embedding_json]');
        """
    )
    try:
        result = ReferenceContractRegistry(conn).inspect()
    finally:
        conn.close()

    assert result.metrics["memory_embeddings"] == {
        "status": STATUS_UNKNOWN,
        "error_class": "operational_error",
    }
    assert "rows" not in result.metrics["memory_embeddings"]
    assert "embedding_json" not in json.dumps(result.metrics)


def test_report_models_serialize_without_memory_content_or_repair_execution():
    report = DoctorReport(
        bank_name="work",
        findings=[
            Finding(
                code="schema.table.vec_items",
                status=STATUS_PRESENT_BUT_UNLOADABLE,
                severity=SEVERITY_WARNING,
                message="Vector table is present but unloadable by this runtime.",
            )
        ],
        repair_candidates=[
            RepairCandidate(
                id="rebuild-vector-index",
                bank_name="work",
                description="Rebuild the vector index after an explicit future repair check.",
            )
        ],
        schema_fingerprint=SchemaFingerprint(
            tables=[TableFingerprint(name="vec_items", status=STATUS_PRESENT_BUT_UNLOADABLE)]
        ),
    )

    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["bank_name"] == "work"
    assert payload["findings"][0]["status"] == STATUS_PRESENT_BUT_UNLOADABLE


def test_report_payload_removes_all_raw_repair_candidate_fields_from_both_renderers():
    raw_values = {
        "repair-id-unique-8472",
        "bank-name-unique-8472",
        "repair-description-unique-8472",
        "finding-code-unique-8472",
    }
    report = DoctorReport(
        bank_name="work",
        repair_candidates=[
            RepairCandidate(
                id="repair-id-unique-8472",
                bank_name="bank-name-unique-8472",
                description="repair-description-unique-8472",
                finding_codes=["finding-code-unique-8472"],
            )
        ],
    )

    payload = doctor_report_payload(report, include_candidates=True)
    json_report = render_doctor_json(payload)
    markdown_report = render_doctor_markdown(payload)

    assert payload["repair_candidates"] == [
        {"candidate_class": "repair", "requires_explicit_confirmation": True}
    ]
    assert all(raw not in json_report for raw in raw_values)
    assert all(raw not in markdown_report for raw in raw_values)


def test_equivalent_payload_mapping_orders_render_byte_identical_artifacts():
    """Renderer ordering must not depend on how an equivalent payload was built."""

    payload = {
        "bank_name": "work",
        "sqlite_health": {
            "quick_check": {"status": "ok"},
            "foreign_key_check": {"status": "ok"},
        },
        "reference_contracts": {
            "zeta": {"status": "unavailable"},
            "alpha": {"status": "scan_limited"},
        },
        "vector_coverage": {
            "working": {"status": "ok"},
            "archive": {"status": "unknown"},
        },
        "hygiene_summary": {"status": "ok", "candidates": []},
        "findings": [],
        "repair_candidates": [],
    }
    reordered_payload = {
        "repair_candidates": [],
        "findings": [],
        "hygiene_summary": {"candidates": [], "status": "ok"},
        "vector_coverage": {
            "archive": {"status": "unknown"},
            "working": {"status": "ok"},
        },
        "reference_contracts": {
            "alpha": {"status": "scan_limited"},
            "zeta": {"status": "unavailable"},
        },
        "sqlite_health": {
            "foreign_key_check": {"status": "ok"},
            "quick_check": {"status": "ok"},
        },
        "bank_name": "work",
    }

    assert payload == reordered_payload
    assert render_doctor_json(payload) == render_doctor_json(reordered_payload)
    assert render_doctor_markdown(payload) == render_doctor_markdown(reordered_payload)


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("replace", "simulated second target replace failure"),
        ("file_fsync", "simulated staging file fsync failure"),
        ("directory_fsync", "simulated directory fsync failure"),
    ],
)
def test_atomic_two_target_failures_restore_existing_artifacts_and_clean_staging(
    tmp_path, monkeypatch, failure, error
):
    """A failed pair write never leaves either old artifact replaced or staged."""

    json_path = tmp_path / "doctor.json"
    markdown_path = tmp_path / "doctor.md"
    original_json = '{"previous": true}\n'
    original_markdown = "# previous\n"
    json_path.write_text(original_json)
    markdown_path.write_text(original_markdown)

    if failure == "replace":
        real_replace = os.replace

        def fail_second_replace(source, destination):
            if Path(destination) == markdown_path:
                raise OSError(error)
            return real_replace(source, destination)

        monkeypatch.setattr(doctor.os, "replace", fail_second_replace)
    else:
        real_fsync = os.fsync

        def fail_selected_fsync(descriptor):
            is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
            if (failure == "file_fsync" and not is_directory) or (
                failure == "directory_fsync" and is_directory
            ):
                raise OSError(error)
            return real_fsync(descriptor)

        monkeypatch.setattr(doctor.os, "fsync", fail_selected_fsync)

    with pytest.raises(OSError, match=error):
        write_doctor_artifacts_atomically(
            json_path=json_path,
            json_text='{"safe": true}\n',
            markdown_path=markdown_path,
            markdown_text="# safe\n",
        )

    assert json_path.read_text() == original_json
    assert markdown_path.read_text() == original_markdown
    assert not list(tmp_path.glob(".doctor-*"))


def test_finding_details_are_redacted_and_json_safe():
    report = DoctorReport(
        bank_name="work",
        findings=[
            Finding(
                code="schema.metadata",
                status=STATUS_UNKNOWN,
                severity=SEVERITY_WARNING,
                message="Schema metadata could not be inspected.",
                details={
                    "operation": "schema_metadata",
                    "error_class": "DatabaseError",
                    "raw_memory": "alice@example.com secret-token",
                    "opaque_value": object(),
                },
            )
        ],
    )

    payload = report.to_dict()
    serialized = json.dumps(payload)
    assert payload["findings"][0]["details"] == {
        "operation": "schema_metadata",
        "error_class": "database_error",
        "redacted_field_count": 2,
    }
    assert "alice@example.com" not in serialized
    assert "secret-token" not in serialized


def test_sqlite_health_adapter_reports_checks_fk_inventory_and_unloadable_vec0(tmp_path):
    db_path = tmp_path / "health.db"
    writable = sqlite3.connect(db_path)
    writable.executescript(
        """
        PRAGMA foreign_keys = OFF;
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (
          memory_id TEXT REFERENCES working_memory(id), embedding_json TEXT
        );
        INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (404, '[0]');
        CREATE TABLE vec_working (id INTEGER PRIMARY KEY);
        PRAGMA writable_schema = ON;
        UPDATE sqlite_master SET sql =
          'CREATE VIRTUAL TABLE vec_working USING vec0(embedding float[3])'
          WHERE name = 'vec_working';
        PRAGMA writable_schema = OFF;
        """
    )
    writable.commit()
    writable.close()

    readonly = open_readonly_doctor_db(db_path)
    try:
        result = SQLiteHealthAdapter(readonly).inspect()
    finally:
        readonly.close()

    assert result.metrics["quick_check"]["status"] == "ok"
    assert result.metrics["quick_check"]["runtime_cost"] == "potentially_global"
    assert result.metrics["foreign_key_check"] == {"status": "violations", "count": 1}
    assert result.metrics["foreign_key_inventory"] == [
        {"table": "memory_embeddings", "count": 1}
    ]
    assert result.metrics["vec0"] == {
        "status": STATUS_PRESENT_BUT_UNLOADABLE,
        "table_count": 1,
        "error_class": "operational_error",
    }


def test_sqlite_health_adapter_bounds_foreign_key_check_rows(tmp_path):
    conn = _readonly_fixture(
        tmp_path,
        """
        PRAGMA foreign_keys = OFF;
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT REFERENCES working_memory(id));
        INSERT INTO memory_embeddings VALUES ('missing-1');
        INSERT INTO memory_embeddings VALUES ('missing-2');
        INSERT INTO memory_embeddings VALUES ('missing-3');
        """,
    )
    try:
        result = SQLiteHealthAdapter(conn, scan_limit=2).inspect()
    finally:
        conn.close()

    assert result.metrics["foreign_key_check"] == {
        "status": "violations",
        "count": 2,
        "truncated": True,
    }


def test_reference_registry_uses_polymorphic_embedding_union_and_only_offers_vec_backfill(tmp_path):
    conn = _queryable_vec0_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY, consolidated_at TEXT);
        CREATE TABLE memories (id TEXT PRIMARY KEY);
        CREATE TABLE episodic_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY);
        INSERT INTO working_memory VALUES ('working-live', NULL);
        INSERT INTO memories VALUES ('legacy-live');
        INSERT INTO episodic_memory VALUES ('episodic-live');
        INSERT INTO memory_embeddings VALUES ('working-live', '[0]');
        INSERT INTO memory_embeddings VALUES ('legacy-live', '[0]');
        INSERT INTO memory_embeddings VALUES ('episodic-live', '[0]');
        INSERT INTO memory_embeddings VALUES ('embedding-orphan', '[0]');
        """,
    )
    try:
        result = ReferenceContractRegistry(conn).inspect()
    finally:
        conn.close()

    assert result.metrics["memory_embeddings"] == {
        "status": "checked",
        "rows": 4,
        "orphan_rows": 1,
    }
    assert result.metrics["vec_working"]["orphan_rows"] == 0
    assert result.metrics["vec_working"]["missing_backfill_rows"] == 1
    assert [candidate.id for candidate in result.repair_candidates] == ["backfill-vec-working"]


def test_reference_registry_ignores_a_normal_table_named_vec_working(tmp_path):
    conn = _readonly_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY);
        INSERT INTO working_memory VALUES ('working-live');
        INSERT INTO memory_embeddings VALUES ('working-live', '[private embedding]');
        INSERT INTO vec_working VALUES (999);
        """,
    )
    try:
        result = ReferenceContractRegistry(conn).inspect()
    finally:
        conn.close()

    assert result.metrics["vec_working"] == {
        "status": "not_configured",
        "orphan_rows": 0,
        "missing_backfill_rows": 0,
    }
    assert result.repair_candidates == []
    assert "private embedding" not in json.dumps(result.metrics)


def test_reference_registry_limits_counts_and_never_leaks_payloads(tmp_path):
    conn = _readonly_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE episodic_memory (id TEXT PRIMARY KEY, binary_vector BLOB);
        CREATE TABLE canonical_facts (
          owner_id TEXT, category TEXT, name TEXT, valid_until TEXT, body TEXT
        );
        INSERT INTO working_memory VALUES ('live', 'private content');
        INSERT INTO memory_embeddings VALUES ('orphan-1', '[private embedding_json]');
        INSERT INTO memory_embeddings VALUES ('orphan-2', '[private embedding_json]');
        INSERT INTO memory_embeddings VALUES ('orphan-3', '[private embedding_json]');
        INSERT INTO episodic_memory VALUES ('episode', X'DEADBEEF');
        INSERT INTO canonical_facts VALUES ('owner', 'category', 'name', NULL, 'private body');
        """,
    )
    try:
        result = ReferenceContractRegistry(conn, scan_limit=2).inspect()
    finally:
        conn.close()

    assert result.metrics["memory_embeddings"] == {
        "status": "scan_limited",
        "rows": 2,
        "rows_truncated": True,
        "orphan_rows": 2,
        "orphan_rows_truncated": True,
    }
    serialized = json.dumps(result.metrics)
    for secret in ("private content", "private embedding_json", "private body", "deadbeef"):
        assert secret not in serialized


def test_reference_registry_keeps_graph_nodes_and_unverifiable_stores_out_of_orphan_repairs(tmp_path):
    conn = _readonly_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE graph_edges (source TEXT, target TEXT);
        CREATE TABLE episodic_memory (id TEXT PRIMARY KEY, valid_until TEXT);
        CREATE TABLE canonical_facts (
          owner_id TEXT, category TEXT, name TEXT, valid_until TEXT, body TEXT
        );
        CREATE TABLE triples (
          subject TEXT, predicate TEXT, object TEXT, valid_from TEXT, valid_until TEXT
        );
        INSERT INTO working_memory VALUES ('live-memory');
        INSERT INTO memory_embeddings VALUES ('known-but-gone', '[0]');
        INSERT INTO graph_edges VALUES ('live-memory', 'known-but-gone');
        INSERT INTO graph_edges VALUES ('entity:alice', 'topic:sqlite');
        INSERT INTO canonical_facts VALUES ('owner', 'identity', 'name', NULL, 'private body');
        INSERT INTO canonical_facts VALUES ('owner', 'identity', 'name', NULL, 'another private body');
        INSERT INTO triples VALUES ('subject', 'predicate', 'object', '2026-02-01', '2026-01-01');
        """,
    )
    try:
        result = ReferenceContractRegistry(conn).inspect()
    finally:
        conn.close()

    assert result.metrics["graph_edges"] == {
        "status": "checked",
        "dangling_memory_endpoints": 1,
        "opaque_graph_node": 2,
    }
    assert result.metrics["episodic_memory"]["status"] == "unverifiable_parent"
    assert result.metrics["canonical_facts"] == {
        "status": "unverifiable_contract",
        "duplicate_current_slots": 1,
    }
    assert result.metrics["triples"] == {
        "status": "not_a_memory_reference",
        "invalid_temporal_ranges": 1,
    }
    assert result.repair_candidates == []


def test_vector_coverage_reports_backfill_orphans_optional_tables_and_compatibility_tiers(tmp_path):
    conn = _queryable_vec0_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (
          id TEXT PRIMARY KEY, valid_until TEXT, superseded_by TEXT
        );
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY);
        CREATE TABLE episodic_memory (id TEXT PRIMARY KEY, binary_vector BLOB);
        CREATE TABLE memories (id TEXT PRIMARY KEY);
        INSERT INTO working_memory VALUES ('working-live', NULL, NULL);
        INSERT INTO working_memory VALUES ('working-expired', '2000-01-01', NULL);
        INSERT INTO memory_embeddings VALUES ('working-live', '[0]');
        INSERT INTO vec_working VALUES (999);
        INSERT INTO episodic_memory VALUES ('episode-1', X'00');
        INSERT INTO episodic_memory VALUES ('episode-2', NULL);
        """,
    )
    try:
        coverage = VectorCoverageAdapter(conn).inspect().metrics
    finally:
        conn.close()

    assert coverage["working"] == {
        "status": "backfill_available",
        "active_source_rows": 1,
       "fallback_embedding_rows": 1,
        "vec_working_rows": 1,
        "missing_vec_working_rows": 1,
        "orphan_vec_working_rows": 1,
    }
    assert coverage["episodic"] == {
        "status": "partial",
        "source_rows": 2,
        "binary_vector_rows": 1,
        "vec_episode_rows": 0,
    }
    assert coverage["legacy"] == {"status": "compatibility_store"}
    assert coverage["canonical"] == {"status": "not_configured"}
    assert coverage["triples"] == {"status": "not_configured"}


def test_vector_coverage_ignores_a_normal_table_named_vec_working(tmp_path):
    conn = _readonly_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY);
        INSERT INTO working_memory VALUES ('working-live');
        INSERT INTO memory_embeddings VALUES ('working-live', '[private embedding_json]');
        INSERT INTO vec_working VALUES (999);
        """,
    )
    try:
        working = VectorCoverageAdapter(conn).inspect().metrics["working"]
    finally:
        conn.close()

    assert working == {
        "status": "fallback_only",
        "active_source_rows": 1,
        "fallback_embedding_rows": 1,
        "vec_working_rows": 0,
        "missing_vec_working_rows": 0,
        "orphan_vec_working_rows": 0,
    }
    assert "private embedding_json" not in json.dumps(working)


def test_vector_coverage_does_not_claim_working_coverage_after_confirmed_vec0_read_error(tmp_path):
    conn = _queryable_vec0_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY);
        INSERT INTO working_memory VALUES ('working-live');
        INSERT INTO memory_embeddings VALUES ('working-live', '[0]');
        """,
        factory=_VecWorkingCapabilityErrorConnection,
        verify_capability=False,
    )
    try:
        working = VectorCoverageAdapter(conn).inspect().metrics["working"]
    finally:
        conn.close()

    assert working == {"status": STATUS_UNKNOWN, "error_class": "operational_error"}
    assert "private content" not in json.dumps(working)


def test_vector_coverage_does_not_claim_episodic_coverage_after_confirmed_vec0_count_error(tmp_path):
    conn = _queryable_vec0_fixture(
        tmp_path,
        """
        CREATE TABLE episodic_memory (id TEXT PRIMARY KEY, binary_vector BLOB);
        CREATE TABLE vec_episodes (rowid INTEGER PRIMARY KEY);
        INSERT INTO episodic_memory VALUES ('episode-1', X'00');
        """,
        factory=_VecEpisodesCountErrorConnection,
        vector_table="vec_episodes",
    )
    try:
        episodic = VectorCoverageAdapter(conn).inspect().metrics["episodic"]
    finally:
        conn.close()

    assert episodic == {"status": STATUS_UNKNOWN, "error_class": "operational_error"}
    assert "private embedding" not in json.dumps(episodic)


def test_vector_coverage_excludes_expired_iso_timestamp_on_the_current_day(tmp_path):
    conn = _queryable_vec0_fixture(
        tmp_path,
        """
        CREATE TABLE working_memory (
          id TEXT PRIMARY KEY, valid_until TEXT, superseded_by TEXT
        );
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY);
        INSERT INTO working_memory VALUES (
          'expired-now', strftime('%Y-%m-%dT%H:%M:%S', 'now', '-1 minute'), NULL
        );
        INSERT INTO working_memory VALUES (
          'still-active', strftime('%Y-%m-%dT%H:%M:%S', 'now', '+1 minute'), NULL
        );
        INSERT INTO memory_embeddings VALUES ('still-active', '[0]');
        """,
    )
    try:
        working = VectorCoverageAdapter(conn).inspect().metrics["working"]
    finally:
        conn.close()

    assert working["active_source_rows"] == 1
    assert working["fallback_embedding_rows"] == 1
    assert working["missing_vec_working_rows"] == 1
    assert working["status"] == "backfill_available"


def test_vector_coverage_treats_unloadable_vec0_as_degraded_not_corrupt(tmp_path):
    db_path = tmp_path / "vec-unloadable.db"
    writable = sqlite3.connect(db_path)
    writable.executescript(
        """
        CREATE TABLE working_memory (id TEXT PRIMARY KEY);
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        CREATE TABLE vec_working (id INTEGER PRIMARY KEY);
        INSERT INTO working_memory VALUES ('working-live');
        INSERT INTO memory_embeddings VALUES ('working-live', '[0]');
        PRAGMA writable_schema = ON;
        UPDATE sqlite_master SET sql =
          'CREATE VIRTUAL TABLE vec_working USING vec0(embedding float[3])'
          WHERE name = 'vec_working';
        PRAGMA writable_schema = OFF;
        """
    )
    writable.commit()
    writable.close()

    readonly = open_readonly_doctor_db(db_path)
    try:
        coverage = VectorCoverageAdapter(readonly).inspect().metrics
    finally:
        readonly.close()

    assert coverage["working"]["status"] == "unavailable"
    assert coverage["working"]["vec0_status"] == STATUS_PRESENT_BUT_UNLOADABLE
    assert coverage["working"]["error_class"] == "operational_error"
    assert set(coverage["working"]) == {"status", "vec0_status", "error_class"}
