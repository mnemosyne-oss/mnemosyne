"""Regression tests for cross-thread Hermes audit writes (#997)."""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MODULES = [
    pytest.param(
        PROJECT_ROOT / "hermes_memory_provider" / "audit.py",
        id="legacy-provider",
    ),
    pytest.param(
        PROJECT_ROOT
        / "integrations"
        / "hermes"
        / "src"
        / "mnemosyne_hermes"
        / "audit.py",
        id="standalone-provider",
    ),
]


def _load_audit_module(path: Path):
    module_name = f"_audit_{path.parent.name}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("module_path", AUDIT_MODULES)
def test_record_persists_when_called_from_another_thread(module_path, tmp_path):
    audit = _load_audit_module(module_path).AuditLog(tmp_path / "audit.db")
    worker = threading.Thread(
        target=audit.record,
        args=("remember",),
        kwargs={"memory_id": "cross-thread", "source_tool": "test"},
    )

    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert audit.count() == 1
    assert audit.query()[0]["memory_id"] == "cross-thread"
    audit.close()


class _TransactionProbe:
    """Connection double that rejects overlapping execute/commit pairs."""

    def __init__(self):
        self._state_lock = threading.Lock()
        self._transaction_active = False
        self.first_execute_entered = threading.Event()
        self.release_first_execute = threading.Event()
        self.commits = 0

    def execute(self, _sql, _params):
        with self._state_lock:
            if self._transaction_active:
                raise RuntimeError("overlapping audit transaction")
            self._transaction_active = True
            first = self.commits == 0
        if first:
            self.first_execute_entered.set()
            assert self.release_first_execute.wait(timeout=2)
        return self

    def commit(self):
        with self._state_lock:
            assert self._transaction_active
            self._transaction_active = False
            self.commits += 1


class _ObservableLock:
    """Lock wrapper that signals when a second caller reaches acquisition."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._entries = 0
        self.second_entry_attempted = threading.Event()

    def __enter__(self):
        with self._state_lock:
            self._entries += 1
            if self._entries == 2:
                self.second_entry_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self._lock.release()


@pytest.mark.parametrize("module_path", AUDIT_MODULES)
def test_concurrent_records_serialize_execute_and_commit(module_path, tmp_path):
    audit = _load_audit_module(module_path).AuditLog(tmp_path / "audit.db")
    audit.close()
    probe = _TransactionProbe()
    observable_lock = _ObservableLock()
    audit._conn = probe
    audit._lock = observable_lock

    first = threading.Thread(target=audit.record, args=("first",))
    second = threading.Thread(target=audit.record, args=("second",))
    first.start()
    assert probe.first_execute_entered.wait(timeout=2)
    second.start()
    second_attempted = observable_lock.second_entry_attempted.wait(timeout=2)
    probe.release_first_execute.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert second_attempted
    assert not first.is_alive()
    assert not second.is_alive()
    assert probe.commits == 2


class _FailingConnection:
    def execute(self, _sql, _params):
        raise RuntimeError("audit write failed")


@pytest.mark.parametrize("module_path", AUDIT_MODULES)
def test_first_record_failure_warns_then_later_failures_debug(
    module_path, tmp_path, caplog
):
    module = _load_audit_module(module_path)
    audit = module.AuditLog(tmp_path / "audit.db")
    audit.close()
    audit._conn = _FailingConnection()

    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        audit.record("first")
        audit.record("second")

    failures = [
        record
        for record in caplog.records
        if "failed to record event" in record.message
    ]
    assert [record.levelno for record in failures] == [logging.WARNING, logging.DEBUG]
