"""Regression tests for SQLite connection retention under thread churn."""

import gc
import importlib
import os
from pathlib import Path
import queue
import threading

import pytest


_CONNECTION_MODULES = (
    "mnemosyne.core.memory",
    "mnemosyne.core.beam",
)


def _database_fd_count(db_path: Path) -> int:
    """Count descriptors held for one SQLite database and its sidecars."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("open-descriptor assertions require /proc")

    prefix = str(db_path)
    count = 0
    for fd_path in fd_dir.iterdir():
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if target == prefix or target.startswith(f"{prefix}-"):
            count += 1
    return count


def _directory_database_fd_count(directory: Path) -> int:
    """Count descriptors held for SQLite files under one directory."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("open-descriptor assertions require /proc")

    prefix = f"{directory}{os.sep}"
    count = 0
    for fd_path in fd_dir.iterdir():
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if target.startswith(prefix):
            count += 1
    return count


def _open_on_new_thread(module, db_path: Path):
    errors = []

    def open_connection():
        try:
            module._get_connection(db_path)
        except Exception as exc:  # pragma: no cover - surfaced by the assertion
            errors.append(exc)

    thread = threading.Thread(target=open_connection)
    thread.start()
    thread.join()
    assert not errors


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_short_lived_thread_connections_keep_database_fds_bounded(
    module_name, tmp_path
):
    """Thread churn must not retain connections until process-wide FD exhaustion."""
    module = importlib.import_module(module_name)
    db_path = tmp_path / f"{module_name.rsplit('.', 1)[-1]}.db"

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(40):
            _open_on_new_thread(module, db_path)

        assert _database_fd_count(db_path) <= 20
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_connection_that_outlives_its_creating_thread_stays_usable(
    module_name, tmp_path
):
    """Cleanup must not invalidate a connection handed to another thread."""
    module = importlib.import_module(module_name)
    retained_path = tmp_path / "retained.db"
    churn_path = tmp_path / "churn.db"
    connections = queue.Queue()

    def open_retained_connection():
        connections.put(module._get_connection(retained_path))

    owner = threading.Thread(target=open_retained_connection)
    owner.start()
    owner.join()
    retained = connections.get_nowait()

    for _ in range(40):
        _open_on_new_thread(module, churn_path)

    assert retained.execute("SELECT 1").fetchone()[0] == 1
    retained.close()


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_live_thread_database_switches_keep_fds_bounded(module_name, tmp_path):
    """A persistent worker switching databases must also trigger cleanup."""
    module = importlib.import_module(module_name)
    ready = threading.Event()
    release = threading.Event()
    errors = []

    def churn_connections():
        try:
            for index in range(64):
                module._get_connection(tmp_path / f"database-{index}.db")
            ready.set()
            release.wait(timeout=5)
        except Exception as exc:  # pragma: no cover - surfaced by the assertion
            errors.append(exc)
            ready.set()

    gc_was_enabled = gc.isenabled()
    gc.disable()
    worker = threading.Thread(target=churn_connections)
    worker.start()
    try:
        assert ready.wait(timeout=5)
        assert not errors
        assert _directory_database_fd_count(tmp_path) <= 20
    finally:
        release.set()
        worker.join(timeout=5)
        if gc_was_enabled:
            gc.enable()
        gc.collect()
