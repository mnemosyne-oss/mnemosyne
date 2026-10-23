"""Regression tests for SQLite connection retention under thread churn."""

import gc
import importlib
import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
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


# ── Explicit release contract (mnemosyne#1163) ───────────────────────────────


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_close_thread_connection_releases_the_handle_deterministically(
    module_name, tmp_path
):
    """The owning thread releases its handle without waiting for collection."""
    if not Path("/proc/self/fd").exists():
        pytest.skip("open-descriptor assertions require /proc")
    module = importlib.import_module(module_name)
    db_path = tmp_path / "explicit-close.db"
    observed = queue.Queue()

    def open_use_and_close():
        module._get_connection(db_path).execute("SELECT 1").fetchone()
        before = _database_fd_count(db_path)
        closed = module.close_thread_connection(db_path)
        observed.put((before, closed, _database_fd_count(db_path)))

    worker = threading.Thread(target=open_use_and_close)
    worker.start()
    worker.join()
    before, closed, after = observed.get_nowait()

    assert before >= 1
    assert closed is True
    assert after == 0


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_connection_is_recreated_after_explicit_close(module_name, tmp_path):
    """A released thread-local must reconnect on next use, not stay poisoned."""
    module = importlib.import_module(module_name)
    db_path = tmp_path / "reopen.db"
    observed = queue.Queue()

    def close_then_reuse():
        first = module._get_connection(db_path)
        assert module.close_thread_connection(db_path) is True
        second = module._get_connection(db_path)
        observed.put(
            (second is not first, second.execute("SELECT 1").fetchone()[0])
        )

    worker = threading.Thread(target=close_then_reuse)
    worker.start()
    worker.join()

    reopened, value = observed.get_nowait()
    assert reopened is True
    assert value == 1


def test_close_thread_connection_refuses_the_main_thread(tmp_path):
    """The main-thread cache is process-lifetime and must opt in explicitly."""
    from mnemosyne.core.memory import _get_connection, close_thread_connection

    db_path = tmp_path / "main-thread.db"
    connection = _get_connection(db_path)

    assert close_thread_connection(db_path) is False
    assert connection.execute("SELECT 1").fetchone()[0] == 1

    assert close_thread_connection(db_path, allow_main_thread=True) is True
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_close_thread_connection_leaves_other_paths_cached(tmp_path):
    """Releasing one database must not displace another database's cache."""
    from mnemosyne.core.memory import _get_connection, close_thread_connection

    kept_path = tmp_path / "kept.db"
    other_path = tmp_path / "other.db"
    observed = queue.Queue()

    def close_unrelated_path():
        kept = _get_connection(kept_path)
        closed = close_thread_connection(other_path)
        observed.put((closed, kept.execute("SELECT 1").fetchone()[0]))
        close_thread_connection(kept_path)

    worker = threading.Thread(target=close_unrelated_path)
    worker.start()
    worker.join()

    closed, value = observed.get_nowait()
    assert closed is False
    assert value == 1


def test_retained_instance_alias_stays_usable_after_creator_thread_exits(tmp_path):
    """Owner lifetime is not creator-thread lifetime.

    BeamMemory keeps ``self.conn`` and hands the same handle to five sub-stores,
    and connections are opened with ``check_same_thread=False``. A retained
    instance must therefore survive its creating thread's exit, its own
    collection, and unrelated connection churn -- the difference between an
    explicit release by the owner and a release inferred from thread death.
    """
    from mnemosyne.core.beam import BeamMemory

    db_path = tmp_path / "retained-alias.db"
    churn_path = tmp_path / "churn.db"
    held = {}

    def create_and_retain():
        beam = BeamMemory(session_id="alias-test", db_path=db_path)
        held["beam"] = beam
        held["substores_share_conn"] = all(
            store.conn is beam.conn
            for store in (
                beam.annotations,
                beam.canonical,
                beam.media,
                beam.episodic_graph,
                beam.veracity_consolidator,
            )
        )

    creator = threading.Thread(target=create_and_retain)
    creator.start()
    creator.join()
    assert not creator.is_alive()

    beam_module = importlib.import_module("mnemosyne.core.beam")
    gc.collect()
    for _ in range(8):
        _open_on_new_thread(beam_module, churn_path)

    assert held["substores_share_conn"] is True
    retained = held["beam"]
    assert retained.conn.execute("SELECT 1").fetchone()[0] == 1
    assert retained.annotations.conn.execute("SELECT 1").fetchone()[0] == 1


class _CloseFailingConnection(sqlite3.Connection):
    """Connection whose close can be made to fail on demand."""

    fail_close = True

    def close(self):
        if type(self).fail_close:
            raise sqlite3.OperationalError("simulated close failure")
        return super().close()


def test_failed_close_keeps_the_handle_cached_for_retry(tmp_path):
    """A failed close must not drop the only reference to a live handle.

    ``sqlite3.Connection.close()`` can raise, and the handle is still usable
    when it does. Clearing the thread-local anyway would hide a live handle and
    leave the caller no way to retry, so the entry must survive until a later
    release succeeds.
    """
    from mnemosyne.core import memory as memory_module

    db_path = (tmp_path / "close-failure.db").resolve()
    observed = queue.Queue()

    def close_with_failure():
        beam_module = importlib.import_module("mnemosyne.core.beam")
        connection = sqlite3.connect(str(db_path), factory=_CloseFailingConnection)
        memory_module._thread_local.conn = connection
        memory_module._thread_local.db_path = str(db_path)
        try:
            first = memory_module.close_thread_connection(db_path)
            still_cached = memory_module._thread_local.conn is connection
            usable = connection.execute("SELECT 1").fetchone()[0]

            _CloseFailingConnection.fail_close = False
            second = memory_module.close_thread_connection(db_path)
            cleared = memory_module._thread_local.conn is None
            observed.put((first, still_cached, usable, second, cleared))
        finally:
            _CloseFailingConnection.fail_close = False
            beam_module._thread_local.conn = None
            beam_module._thread_local.db_path = None

    worker = threading.Thread(target=close_with_failure)
    worker.start()
    worker.join()

    assert observed.get_nowait() == (False, True, 1, True, True)


# Runs in a fresh process: the defect this guards only appears when
# mnemosyne.core.memory has not been imported yet on the releasing thread.
_COLD_RELEASE_PROBE = '''
import json
import os
import sys
import threading
import traceback
from pathlib import Path

root = Path(sys.argv[1])
import_mode = sys.argv[2]
path_mode = sys.argv[3]
entry = sys.argv[4]
os.environ["MNEMOSYNE_DATA_DIR"] = str(root / "default")

import mnemosyne.core.beam as beam_module
from mnemosyne.core.beam import BeamMemory

default_db = (root / "default" / "mnemosyne.db").resolve()
worker_db = default_db if path_mode == "match" else (root / "worker" / "mnemosyne.db").resolve()
other_db = (root / "other" / "other.db").resolve()

if import_mode == "warm":
    import mnemosyne.core.memory  # noqa: F401

observed = {}
other_ready = threading.Event()
other_done = threading.Event()
other_conn = {}


def hold_other_path():
    beam = BeamMemory(session_id="other", db_path=other_db)
    other_conn["conn"] = beam.conn
    other_ready.set()
    other_done.wait(timeout=120)


def _release_in_worker():
    beam = BeamMemory(session_id="cold-release", db_path=worker_db)
    original = beam.conn
    observed["cache_holds_original"] = (
        getattr(beam_module._thread_local, "conn", None) is original
    )
    if entry == "beam_helper":
        observed["result"] = beam_module.close_thread_connection(worker_db)
    else:
        import mnemosyne

        observed["result"] = mnemosyne.close_thread_connection(worker_db)
    cached = getattr(beam_module._thread_local, "conn", None)
    observed["cache_holds_original_after"] = cached is original
    observed["cache_cleared"] = cached is None
    observed["replacement_leftover"] = cached is not None and cached is not original
    try:
        original.execute("SELECT 1").fetchone()
        observed["original_closed"] = False
    except Exception:
        observed["original_closed"] = True
    observed["memory_imported"] = "mnemosyne.core.memory" in sys.modules


def release_from_worker():
    # Surface a thread failure as a probe failure instead of a silent exit 0.
    try:
        _release_in_worker()
    except BaseException:
        observed["error"] = traceback.format_exc()


parent = BeamMemory(session_id="parent", db_path=worker_db)
other_thread = threading.Thread(target=hold_other_path)
other_thread.start()
assert other_ready.wait(timeout=120), "other-path holder did not start"

worker = threading.Thread(target=release_from_worker)
worker.start()
worker.join(timeout=120)
assert not worker.is_alive(), "release worker did not finish"
assert "error" not in observed, observed.get("error")
assert "memory_imported" in observed, "release worker failed before recording observations"

observed["parent_usable"] = parent.conn.execute("SELECT 1").fetchone()[0] == 1
observed["other_path_usable"] = other_conn["conn"].execute("SELECT 1").fetchone()[0] == 1
other_done.set()
other_thread.join(timeout=120)
print(json.dumps(observed))
'''


def _run_cold_release_probe(root: Path, import_mode: str, path_mode: str, entry: str) -> dict:
    """Run the release probe in a fresh interpreter and return its observations."""
    script = root / "cold_release_probe.py"
    script.write_text(_COLD_RELEASE_PROBE)
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    env["HERMES_HOME"] = str(root / "hermes-home")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, str(script), str(root), import_mode, path_mode, entry],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("entry", ["beam_helper", "package_helper"])
@pytest.mark.parametrize("path_mode", ["match", "differ"])
@pytest.mark.parametrize("import_mode", ["cold", "warm"])
def test_cold_release_closes_the_original_worker_handle(tmp_path, import_mode, path_mode, entry):
    """Cleanup must release the handle the worker opened, not a replacement.

    ``mnemosyne.core.memory`` runs ``init_db()`` at import time, which opens a
    connection for the default database and republishes it into ``beam``'s
    thread-local cache. A release that imported that module therefore closed
    the import's replacement and reported success (matching default path), or
    closed nothing at all (different default path), while the worker's original
    handle stayed open. Importing the release helper must not touch SQLite.

    ``import_mode`` covers the cold process (the failing case) and the warm one
    where an unrelated caller imported ``core.memory`` first; ``path_mode``
    covers a worker database on the default path and elsewhere.
    """
    observed = _run_cold_release_probe(tmp_path, import_mode, path_mode, entry)

    assert observed["cache_holds_original"] is True
    assert observed["result"] is True
    assert observed["original_closed"] is True
    assert observed["cache_holds_original_after"] is False
    assert observed["cache_cleared"] is True
    assert observed["replacement_leftover"] is False
    assert observed["parent_usable"] is True
    assert observed["other_path_usable"] is True
    # Warm means core.memory was imported before any handle existed; cold means
    # the release path itself must not have pulled it in.
    assert observed["memory_imported"] is (import_mode == "warm")
