"""Busy-timeout and lock-observability regressions for both audit providers."""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT_MODULES = (
    ROOT / "hermes_memory_provider" / "audit.py",
    ROOT / "integrations" / "hermes" / "src" / "mnemosyne_hermes" / "audit.py",
)


def _load_audit_module(path: Path) -> ModuleType:
    module_name = f"_audit_under_test_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=AUDIT_MODULES, ids=("legacy", "standalone"))
def audit_module(request: pytest.FixtureRequest) -> ModuleType:
    return _load_audit_module(request.param)


@pytest.mark.parametrize(
    ("env_value", "expected_ms"),
    (
        (None, 5000),
        ("37", 37),
        ("not-an-integer", 5000),
        ("0", 0),
        ("-1", 0),
        ("2147483647", 2147483647),
    ),
    ids=("unset", "valid", "invalid", "zero", "negative", "large"),
)
def test_audit_connection_uses_core_busy_timeout_semantics(
    audit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_value: str | None,
    expected_ms: int,
) -> None:
    if env_value is None:
        monkeypatch.delenv("MNEMOSYNE_BUSY_TIMEOUT_MS", raising=False)
    else:
        monkeypatch.setenv("MNEMOSYNE_BUSY_TIMEOUT_MS", env_value)

    audit = audit_module.AuditLog(tmp_path / "audit.db")
    try:
        assert audit._conn.execute("PRAGMA busy_timeout").fetchone()[0] == expected_ms
    finally:
        audit.close()


def test_locked_audit_write_warns_without_raising_or_long_block(
    audit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_BUSY_TIMEOUT_MS", "20")
    db_path = tmp_path / "audit.db"
    audit = audit_module.AuditLog(db_path)
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")

    caplog.set_level(logging.WARNING, logger=audit_module.__name__)
    started = time.monotonic()
    try:
        # Audit remains best-effort: the caller continues with its result.
        result = {"status": "memory-operation-completed"}
        audit.record("remember", memory_id="locked")
    finally:
        elapsed = time.monotonic() - started
        blocker.rollback()
        blocker.close()
        audit.close()

    assert result == {"status": "memory-operation-completed"}
    assert elapsed < 1.0
    assert "audit: failed to record event" in caplog.text
    assert "database is locked" in caplog.text
