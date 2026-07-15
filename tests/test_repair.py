"""Security regression coverage for narrow Doctor-gated repair operations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

import mnemosyne.repair as repair
from mnemosyne.doctor import build_doctor_report, doctor_report_payload
from mnemosyne.repair import RepairError, run_repair


ROOT = Path(__file__).resolve().parent.parent
RAW_SECRET = "repair-output-private-secret-79d1"  # nosec - redaction fixture
RAW_CONTENT = "Only the hidden cobalt-archive content may contain this phrase."
RAW_EMBEDDING = "[0.125, 0.875]"
RAW_BLOB_HEX = "DEADBEEF"


def _create_db(path: Path, *, with_vector_cache: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE working_memory (
            id TEXT PRIMARY KEY,
            content TEXT,
            private_blob BLOB,
            valid_until TEXT,
            superseded_by TEXT
        );
        CREATE TABLE memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_json TEXT
        );
        """
    )
    if with_vector_cache:
        # This is intentionally a normal SQLite table. Stock SQLite cannot load
        # vec0, so the successful backfill test uses the controlled adapter
        # below rather than presenting this fixture as real vec0 integration.
        conn.execute("CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY, embedding TEXT)")
    conn.commit()
    conn.close()


def _insert_memory(
    path: Path,
    memory_id: str,
    *,
    valid_until: str | None = None,
    superseded_by: str | None = None,
    content: str = "safe fixture content",
    embedding: str | None = None,
) -> int:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO working_memory (id, content, private_blob, valid_until, superseded_by) "
        "VALUES (?, ?, ?, ?, ?)",
        (memory_id, content, bytes.fromhex(RAW_BLOB_HEX), valid_until, superseded_by),
    )
    if embedding is not None:
        conn.execute("INSERT INTO memory_embeddings VALUES (?, ?)", (memory_id, embedding))
    rowid = conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()[0]
    conn.commit()
    conn.close()
    return int(rowid)


def _write_manifest(db_path: Path, tmp_path: Path, *, bank_name: str = "default") -> Path:
    report_path = tmp_path / f"{bank_name}-doctor.json"
    report = build_doctor_report(bank_name, db_path)
    report_path.write_text(
        json.dumps(doctor_report_payload(report, include_candidates=True), sort_keys=True), encoding="utf-8"
    )
    return report_path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ControlledQueryableVecConnection(sqlite3.Connection):
    """Test-only complete vec0 adapter-contract connection.

    Stock SQLite cannot create vec0. The adapter supplies the two catalog
    surfaces required by the production guard (DDL plus ``table_list`` type),
    while the fixture continues to exercise the real rowid/embedding query and
    transactional insert path. It is deliberately an isolated adapter-contract
    test, not a claim that a normal SQLite table is a vec0 integration.
    """

    def execute(self, sql, parameters=()):
        if sql == "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_working'":
            return super().execute(
                "SELECT ? AS sql",
                ("CREATE VIRTUAL TABLE vec_working USING vec0(embedding float[2])",),
            )
        if sql == "PRAGMA table_list":
            return super().execute(
                "SELECT 'main' AS schema, 'vec_working' AS name, 'virtual' AS type, "
                "2 AS ncol, 0 AS wr, 0 AS strict"
            )
        return super().execute(sql, parameters)


class _UnloadableVecConnection(_ControlledQueryableVecConnection):
    """Expose confirmed vec0 DDL but fail its live capability probe."""

    def execute(self, sql, parameters=()):
        if sql == 'SELECT rowid, embedding FROM "vec_working" LIMIT 0':
            raise sqlite3.OperationalError("no such module: vec0")
        return super().execute(sql, parameters)


def _controlled_writable_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, factory=_ControlledQueryableVecConnection, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _unloadable_writable_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, factory=_UnloadableVecConnection, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _run_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "data")
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_dry_run_is_read_only_and_creates_no_backup(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    before = _hash(db_path)
    requested_backup = tmp_path / "should-not-exist.sqlite"

    result = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=["working_memory:selected"],
        action="expire",
        backup_path=requested_backup,
    )

    assert result == {
        "mode": "dry_run",
        "action": "expire",
        "backup": False,
        "applied": [{"table": "working_memory", "status": "planned"}],
        "skipped": [],
    }
    assert not requested_backup.exists()
    assert _hash(db_path) == before


def test_repair_fails_closed_before_dry_run_on_non_linux(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    report_path = tmp_path / "missing-report.json"
    before = tuple(tmp_path.iterdir())

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("platform guard must run before filesystem work")

    monkeypatch.setattr(repair.sys, "platform", "darwin")
    monkeypatch.setattr(repair, "_database_identity", fail_if_called)
    monkeypatch.setattr(repair, "_load_manifest", fail_if_called)

    with pytest.raises(RepairError, match="supported only on Linux"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
        )

    assert not report_path.exists()
    assert tuple(tmp_path.iterdir()) == before


def test_manifest_bank_fingerprint_and_parse_fail_closed_before_backup_or_write(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    requested_backup = tmp_path / "blocked.sqlite"

    with pytest.raises(RepairError, match="bank does not match"):
        run_repair(
            db_path=db_path,
            bank_name="other",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=requested_backup,
        )
    assert not requested_backup.exists()

    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE working_memory ADD COLUMN changed_after_report TEXT")
    conn.commit()
    conn.close()
    before_mismatch = _hash(db_path)
    with pytest.raises(RepairError, match="fingerprint does not match"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=requested_backup,
        )
    assert not requested_backup.exists()
    assert _hash(db_path) == before_mismatch

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not JSON", encoding="utf-8")
    with pytest.raises(RepairError, match="could not be parsed"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=malformed,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=requested_backup,
        )
    assert not requested_backup.exists()


def test_apply_revalidates_manifest_fingerprint_under_lock_before_backup_or_write(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    backup = tmp_path / "must-not-exist.sqlite"
    events: list[str] = []
    drifted_hash: str | None = None
    real_gate = repair._verify_report_gate

    def drift_after_preflight(report, database, bank_name, *, conn=None, **kwargs):
        nonlocal drifted_hash
        events.append("locked" if conn is not None else "preflight")
        result = real_gate(report, database, bank_name, conn=conn, **kwargs)
        if conn is None:
            drift = sqlite3.connect(database)
            try:
                drift.execute("ALTER TABLE working_memory ADD COLUMN changed_between_gates TEXT")
                drift.commit()
            finally:
                drift.close()
            drifted_hash = _hash(database)
        return result

    monkeypatch.setattr(repair, "_verify_report_gate", drift_after_preflight)

    with pytest.raises(RepairError, match="fingerprint does not match"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=backup,
        )

    assert events == ["preflight", "locked"]
    assert drifted_hash is not None
    assert _hash(db_path) == drifted_hash
    assert not backup.exists()
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is None
    finally:
        conn.close()


def test_apply_creates_and_quick_checks_backup_before_mutation(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    backup = tmp_path / "verified-backup.sqlite"
    events: list[str] = []
    real_backup = repair._create_validated_backup

    def checked_backup(source: Path, destination: Path) -> None:
        real_backup(source, destination)
        assert destination.is_file()
        verify = sqlite3.connect(destination)
        try:
            assert verify.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            # This proves the pre-mutation snapshot still contains the active row.
            assert verify.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is None
        finally:
            verify.close()
        events.append("backup")

    def tracked_connection(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.set_trace_callback(lambda sql: events.append("update") if sql.startswith("UPDATE working_memory") else None)
        return conn

    monkeypatch.setattr(repair, "_create_validated_backup", checked_backup)
    monkeypatch.setattr(repair, "_open_writable_repair_db", tracked_connection)
    result = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=["working_memory:selected"],
        action="expire",
        apply=True,
        backup_path=backup,
    )

    assert result["backup"] is True
    assert events.index("backup") < events.index("update")
    assert backup.is_file()
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is not None
    finally:
        conn.close()


def test_missing_stale_and_already_expired_rows_skip_without_backup_or_write(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "stale", valid_until="2000-01-01")
    _insert_memory(db_path, "already-expired", valid_until="2000-01-02")
    _insert_memory(db_path, "inactive", superseded_by="newer")
    report_path = _write_manifest(db_path, tmp_path)
    before = _hash(db_path)
    backup = tmp_path / "must-not-exist.sqlite"

    result = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=[
            "working_memory:missing",
            "working_memory:stale",
            "working_memory:already-expired",
            "working_memory:inactive",
        ],
        action="expire",
        apply=True,
        backup_path=backup,
    )

    assert result["backup"] is False
    assert result["applied"] == []
    assert len(result["skipped"]) == 4
    assert {item["reason"] for item in result["skipped"]} == {"missing", "already_inactive"}
    assert not backup.exists()
    assert _hash(db_path) == before


def test_controlled_vec_backfill_only_selected_row_and_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    _create_db(db_path, with_vector_cache=True)
    selected_rowid = _insert_memory(db_path, "selected", embedding="[3, 4]")
    other_rowid = _insert_memory(db_path, "not-selected", embedding="[5, 12]")
    report_path = _write_manifest(db_path, tmp_path)
    statements: list[str] = []
    backup_events: list[str] = []
    real_backup = repair._create_validated_backup

    def traced_controlled_connection(path: Path) -> sqlite3.Connection:
        conn = _controlled_writable_connection(path)
        conn.set_trace_callback(statements.append)
        return conn

    def checked_backup(source_fd: int, destination: Path) -> None:
        # The exact controlled vec0 insert has been attempted and rolled back
        # before the first backup is allowed to exist.
        savepoint = statements.index("SAVEPOINT repair_vec_write_preflight")
        rollback = statements.index("ROLLBACK TO repair_vec_write_preflight")
        assert any(statement.startswith("INSERT INTO vec_working") for statement in statements[savepoint:rollback])
        assert "RELEASE repair_vec_write_preflight" in statements
        backup_events.append("after_preflight")
        real_backup(source_fd, destination)

    monkeypatch.setattr(repair, "_open_writable_repair_db", traced_controlled_connection)
    monkeypatch.setattr(repair, "_create_validated_backup", checked_backup)

    first_backup = tmp_path / "first.sqlite"
    first = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=["working_memory:selected"],
        action="backfill-vec-working",
        apply=True,
        backup_path=first_backup,
    )

    assert first["backup"] is True
    assert first["applied"] == [{"table": "working_memory", "status": "applied"}]
    assert backup_events == ["after_preflight"]
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT rowid, embedding FROM vec_working ORDER BY rowid").fetchall()
    finally:
        conn.close()
    assert [row[0] for row in rows] == [selected_rowid]
    assert other_rowid != selected_rowid
    assert json.loads(rows[0][1]) == pytest.approx([0.6, 0.8])

    second_backup = tmp_path / "second.sqlite"
    second = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=["working_memory:selected"],
        action="backfill-vec-working",
        apply=True,
        backup_path=second_backup,
    )
    assert second["backup"] is False
    assert second["applied"] == []
    assert second["skipped"] == [
        {"table": "working_memory", "status": "skipped", "reason": "already_present"}
    ]
    assert not second_backup.exists()


def test_unloadable_vec_backfill_skips_without_backup_or_mutation_and_stays_content_safe(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    _create_db(db_path, with_vector_cache=True)
    selected_id = f"id-{RAW_SECRET}"
    _insert_memory(
        db_path,
        selected_id,
        content=RAW_CONTENT,
        embedding=RAW_EMBEDDING,
    )
    report_path = _write_manifest(db_path, tmp_path)
    backup = tmp_path / "must-not-exist.sqlite"
    before = _hash(db_path)
    monkeypatch.setattr(repair, "_open_writable_repair_db", _unloadable_writable_connection)

    result = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=[f"working_memory:{selected_id}"],
        action="backfill-vec-working",
        apply=True,
        backup_path=backup,
    )

    assert result == {
        "mode": "apply",
        "action": "backfill-vec-working",
        "backup": False,
        "applied": [],
        "skipped": [{"table": "working_memory", "status": "skipped", "reason": "unloadable"}],
    }
    assert _hash(db_path) == before
    assert not backup.exists()
    output = repair.render_repair_json(result)
    for raw in (RAW_SECRET, RAW_CONTENT, RAW_EMBEDDING, RAW_BLOB_HEX, selected_id):
        assert raw not in output


def test_expiry_changes_only_selected_valid_until_and_never_deletes(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected", content="selected content")
    _insert_memory(db_path, "untouched", content="untouched content")
    report_path = _write_manifest(db_path, tmp_path)
    statements: list[str] = []

    def traced_connection(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(repair, "_open_writable_repair_db", traced_connection)
    result = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=["working_memory:selected"],
        action="expire",
        apply=True,
        backup_path=tmp_path / "expiry.sqlite",
    )

    assert result["applied"] == [{"table": "working_memory", "status": "applied"}]
    assert not any("DELETE" in statement.upper() for statement in statements)
    conn = sqlite3.connect(db_path)
    try:
        selected = conn.execute(
            "SELECT content, valid_until FROM working_memory WHERE id = 'selected'"
        ).fetchone()
        untouched = conn.execute(
            "SELECT content, valid_until FROM working_memory WHERE id = 'untouched'"
        ).fetchone()
        assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 2
    finally:
        conn.close()
    assert selected[0] == "selected content"
    assert selected[1] is not None
    assert untouched == ("untouched content", None)


def test_apply_binds_connection_to_authorized_inode_during_a_to_b_to_a_swap(tmp_path, monkeypatch):
    """The writable connection must not follow the public name after binding."""

    db_path = tmp_path / "memory.db"
    replacement = tmp_path / "replacement.db"
    _create_db(db_path)
    _create_db(replacement)
    _insert_memory(db_path, "selected", content="authorized")
    _insert_memory(replacement, "selected", content="attacker replacement")
    report_path = _write_manifest(db_path, tmp_path)
    backup = tmp_path / "authorized-backup.sqlite"
    replacement_before = _hash(replacement)
    original_open = repair._open_writable_repair_db

    def open_during_a_to_b_to_a_swap(private_database: Path) -> sqlite3.Connection:
        parked = tmp_path / "authorized-parked.db"
        # The public path points at B exactly while the repair opens its write
        # connection, then returns to A before any old pathname re-stat could
        # notice.  The private hardlink must still keep the connection on A.
        os.replace(db_path, parked)
        os.replace(replacement, db_path)
        try:
            return original_open(private_database)
        finally:
            os.replace(db_path, replacement)
            os.replace(parked, db_path)

    monkeypatch.setattr(repair, "_open_writable_repair_db", open_during_a_to_b_to_a_swap)
    result = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=["working_memory:selected"],
        action="expire",
        apply=True,
        backup_path=backup,
    )

    assert result["applied"] == [{"table": "working_memory", "status": "applied"}]
    assert backup.is_file()
    assert _hash(replacement) == replacement_before
    authorized = sqlite3.connect(db_path)
    attacker = sqlite3.connect(replacement)
    try:
        assert authorized.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is not None
        assert attacker.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is None
    finally:
        authorized.close()
        attacker.close()
    assert not list(tmp_path.glob(".mnemosyne-repair-*"))


def test_wal_database_is_rejected_before_backup_or_mutation(tmp_path):
    """A private hardlink must not silently omit another pathname's WAL."""

    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode = WAL").fetchone()[0].lower() == "wal"
    finally:
        conn.close()
    report_path = _write_manifest(db_path, tmp_path)
    backup = tmp_path / "must-not-exist.sqlite"

    with pytest.raises(RepairError, match="sidecars prevent"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=backup,
        )

    assert not backup.exists()
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is None
    finally:
        conn.close()
    assert not list(tmp_path.glob(".mnemosyne-repair-*"))


def test_existing_rollback_journal_fails_closed_before_binding_backup_or_mutation(tmp_path):
    """The original rollback journal cannot be replayed through a private link."""

    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal.write_bytes(b"controlled nonempty hot-ish rollback journal")
    before = _hash(db_path)
    journal_before = _hash(journal)
    backup = tmp_path / "must-not-exist.sqlite"

    with pytest.raises(RepairError, match="sidecars prevent"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=backup,
        )

    assert _hash(db_path) == before
    assert _hash(journal) == journal_before
    assert not backup.exists()
    assert not list(tmp_path.glob(".mnemosyne-repair-*"))


def test_sidecar_created_after_preflight_is_rejected_before_inode_binding(tmp_path, monkeypatch):
    """A sidecar arriving during Doctor preflight cannot cross the bind gate."""

    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    journal = db_path.with_name(db_path.name + "-journal")
    journal_bytes = b"sidecar created after the initial preflight check"
    before = _hash(db_path)
    backup = tmp_path / "must-not-exist.sqlite"
    fd_before = len(os.listdir("/proc/self/fd"))
    real_gate = repair._verify_report_gate

    def create_sidecar_after_preflight(*args, conn=None, **kwargs):
        result = real_gate(*args, conn=conn, **kwargs)
        if conn is None:
            journal.write_bytes(journal_bytes)
        return result

    monkeypatch.setattr(repair, "_verify_report_gate", create_sidecar_after_preflight)

    with pytest.raises(RepairError, match="sidecars prevent"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=backup,
        )

    assert _hash(db_path) == before
    assert journal.read_bytes() == journal_bytes
    assert not backup.exists()
    assert not list(tmp_path.glob(".mnemosyne-repair-*"))
    assert len(os.listdir("/proc/self/fd")) == fd_before


def test_backup_parent_swap_cannot_redirect_fd_anchored_backup(tmp_path, monkeypatch):
    """The requested backup parent may be renamed only after its FD is retained."""

    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    backup_parent = tmp_path / "backup-parent"
    backup_parent.mkdir()
    attacker_parent = tmp_path / "attacker-parent"
    attacker_parent.mkdir()
    parked_parent = tmp_path / "parked-backup-parent"
    backup = backup_parent / "backup.sqlite"
    fd_before = len(os.listdir("/proc/self/fd"))
    real_backup = repair._create_validated_backup

    def swap_parent_after_reservation(source_fd: int, destination) -> None:
        # Reservation must already hold the original parent directory FD and
        # final file FD. Any later string-path use would write to attacker_parent.
        os.replace(backup_parent, parked_parent)
        os.replace(attacker_parent, backup_parent)
        real_backup(source_fd, destination)

    monkeypatch.setattr(repair, "_create_validated_backup", swap_parent_after_reservation)
    result = run_repair(
        db_path=db_path,
        bank_name="default",
        report_path=report_path,
        selections=["working_memory:selected"],
        action="expire",
        apply=True,
        backup_path=backup,
    )

    assert result["backup"] is True
    assert not backup.exists()
    retained_backup = parked_parent / "backup.sqlite"
    assert retained_backup.is_file()
    verify = sqlite3.connect(retained_backup)
    source = sqlite3.connect(db_path)
    try:
        assert verify.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert verify.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is None
        assert source.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is not None
    finally:
        verify.close()
        source.close()
    assert not list(tmp_path.glob(".mnemosyne-repair-*"))
    assert not list(backup_parent.iterdir())
    assert len(os.listdir("/proc/self/fd")) == fd_before


def test_working_memory_trigger_fails_closed_before_backup_or_selected_update(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    _insert_memory(db_path, "untouched")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TRIGGER working_memory_expiry_side_effect
            AFTER UPDATE OF valid_until ON working_memory
            BEGIN
                UPDATE working_memory
                SET valid_until = CURRENT_TIMESTAMP
                WHERE id = 'untouched';
            END;
            """
        )
        conn.commit()
    finally:
        conn.close()
    report_path = _write_manifest(db_path, tmp_path)
    manifest = json.loads(report_path.read_text(encoding="utf-8"))
    assert any(
        trigger["name"] == "working_memory_expiry_side_effect" and trigger["table_name"] == "working_memory"
        for trigger in manifest["schema_fingerprint"]["triggers"]
    )
    backup = tmp_path / "must-not-exist.sqlite"

    with pytest.raises(RepairError, match="triggers prevent"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=backup,
        )

    assert not backup.exists()
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT valid_until FROM working_memory WHERE id = 'selected'").fetchone()[0] is None
        assert conn.execute("SELECT valid_until FROM working_memory WHERE id = 'untouched'").fetchone()[0] is None
        assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 2
    finally:
        conn.close()
    assert not list(tmp_path.glob(".mnemosyne-repair-*"))


def test_rejects_unknown_selection_action_and_database_or_hardlink_backup_targets(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    _insert_memory(db_path, "selected")
    report_path = _write_manifest(db_path, tmp_path)
    before = _hash(db_path)

    with pytest.raises(RepairError, match="working_memory:ID") as unknown_selection:
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=[f"memories:{RAW_SECRET}"],
            action="expire",
        )
    assert RAW_SECRET not in str(unknown_selection.value)
    with pytest.raises(RepairError, match="--action"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="delete-all",
        )
    with pytest.raises(RepairError, match="must not be the inspected database"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=db_path,
        )

    hardlink = tmp_path / "database-hardlink.sqlite"
    os.link(db_path, hardlink)
    with pytest.raises(RepairError, match="must be a new file"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=hardlink,
        )
    symlink = tmp_path / "database-symlink.sqlite"
    symlink.symlink_to(db_path)
    with pytest.raises(RepairError, match="must not use a symlink"):
        run_repair(
            db_path=db_path,
            bank_name="default",
            report_path=report_path,
            selections=["working_memory:selected"],
            action="expire",
            apply=True,
            backup_path=symlink,
        )
    assert _hash(db_path) == before


def test_cli_output_and_errors_never_echo_raw_memory_or_embedding_data(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_db(db_path)
    selected_id = f"id-{RAW_SECRET}"
    _insert_memory(
        db_path,
        selected_id,
        content=RAW_CONTENT,
        embedding=RAW_EMBEDDING,
    )
    report_path = _write_manifest(db_path, tmp_path)

    result = _run_cli(
        [
            "repair",
            "--db",
            str(db_path),
            "--report",
            str(report_path),
            "--select",
            f"working_memory:{selected_id}",
            "--action",
            "expire",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["applied"] == [{"status": "planned", "table": "working_memory"}]
    leaked = result.stdout + result.stderr
    for raw in (RAW_SECRET, RAW_CONTENT, RAW_EMBEDDING, RAW_BLOB_HEX, selected_id):
        assert raw not in leaked

    invalid = _run_cli(
        [
            "repair",
            "--db",
            str(db_path),
            "--report",
            str(report_path),
            "--select",
            f"unknown:{RAW_SECRET}",
        ],
        tmp_path,
    )
    assert invalid.returncode == 1
    assert RAW_SECRET not in invalid.stdout + invalid.stderr

    unknown_option = _run_cli(["repair", f"--{RAW_SECRET}"], tmp_path)
    assert unknown_option.returncode == 2
    assert RAW_SECRET not in unknown_option.stdout + unknown_option.stderr

    invalid_bank = _run_cli(
        [
            "repair",
            "--bank",
            f"bad/{RAW_SECRET}",
            "--report",
            str(report_path),
            "--select",
            "working_memory:any",
        ],
        tmp_path,
    )
    assert invalid_bank.returncode == 2
    assert RAW_SECRET not in invalid_bank.stdout + invalid_bank.stderr


def test_cli_apply_requires_an_explicit_complete_selection(tmp_path):
    result = _run_cli(["repair", "--apply", "--report", "missing.json"], tmp_path)
    assert result.returncode == 2
    assert "At least one complete --select" in result.stderr
