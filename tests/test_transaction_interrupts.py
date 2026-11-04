"""Transaction ownership and interruption regressions for #1319."""

import sqlite3

import pytest

from mnemosyne.core.beam import (
    _BeamConnection,
    _deferred_commits,
    _guarded_transaction,
    _write_locked_transaction,
)
from mnemosyne.core.veracity_consolidation import VeracityConsolidator


@pytest.fixture(params=[sqlite3.Connection, _BeamConnection], ids=["plain", "beam"])
def connections(request, tmp_path):
    path = tmp_path / "interrupt.db"
    conn = sqlite3.connect(path, factory=request.param)
    peer = sqlite3.connect(path, timeout=0)
    conn.execute("CREATE TABLE markers (value TEXT)")
    conn.commit()
    try:
        yield conn, peer
    finally:
        conn.close()
        peer.close()


@pytest.fixture(params=[RuntimeError, KeyboardInterrupt, SystemExit])
def failure(request):
    return request.param("original body failure")


def _scope(helper, conn):
    if helper == "serialized":
        return VeracityConsolidator(conn=conn, federation_probe=False)._serialized_write()
    if helper == "guarded":
        return _guarded_transaction(conn)
    return _write_locked_transaction(conn)


def _markers(conn):
    return [row[0] for row in conn.execute("SELECT value FROM markers ORDER BY rowid")]


def _assert_writer_available(peer):
    peer.execute("BEGIN IMMEDIATE")
    peer.rollback()


@pytest.mark.parametrize("helper", ["write_locked", "serialized"])
@pytest.mark.parametrize("connection_factory", [sqlite3.Connection, _BeamConnection],
                         ids=["plain", "beam"])
@pytest.mark.parametrize("mode", ["before", "after", "closed"])
def test_begin_interruption_releases_owned_writer(
    tmp_path, failure, helper, connection_factory, mode
):
    begins = []
    rollbacks = []

    def interrupt_begin(execute, sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE":
            begins.append(sql)
            if mode != "before":
                execute(sql, *args, **kwargs)
                assert conn.in_transaction
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    peer.execute("BEGIN IMMEDIATE")
            if mode == "closed":
                conn.close()
            raise failure
        return execute(sql, *args, **kwargs)

    class InterruptedCursor(sqlite3.Cursor):
        def execute(self, sql, *args, **kwargs):
            return interrupt_begin(super().execute, sql, *args, **kwargs)

    class InterruptedBeginConnection(connection_factory):
        def execute(self, sql, *args, **kwargs):
            if helper == "write_locked":
                return interrupt_begin(super().execute, sql, *args, **kwargs)
            return super().execute(sql, *args, **kwargs)

        def cursor(self, *args, **kwargs):
            # Python 3.10's Connection.execute also calls this override.
            if helper == "serialized":
                return super().cursor(factory=InterruptedCursor)
            return super().cursor(*args, **kwargs)

        def rollback(self):
            rollbacks.append("rollback")
            return super().rollback()

    path = tmp_path / "begin_interrupt.db"
    conn = sqlite3.connect(path, factory=InterruptedBeginConnection)
    peer = sqlite3.connect(path, timeout=0)
    conn.execute("CREATE TABLE markers (value TEXT)")
    scope = _scope(helper, conn)
    try:
        with pytest.raises(type(failure)) as caught:
            with scope:
                pytest.fail("interrupted acquisition must not enter the body")
        assert caught.value is failure
        assert begins == ["BEGIN IMMEDIATE"]
        if mode != "closed":
            assert not conn.in_transaction
            assert _markers(conn) == []
        if mode == "after":
            assert rollbacks == ["rollback"]
        assert _markers(peer) == []
        _assert_writer_available(peer)
    finally:
        conn.close()
        peer.close()


@pytest.mark.parametrize("helper", ["write_locked", "guarded", "serialized"])
def test_owned_failure_rolls_back_and_releases_writer(connections, failure, helper):
    conn, peer = connections
    with pytest.raises(type(failure)) as caught:
        with _scope(helper, conn):
            # The guarded helper relies on its caller to begin/write the tx.
            conn.execute("INSERT INTO markers VALUES ('body')")
            raise failure

    assert caught.value is failure
    assert not conn.in_transaction
    assert _markers(conn) == []
    assert _markers(peer) == []
    _assert_writer_available(peer)


@pytest.mark.parametrize("helper", ["write_locked", "guarded", "serialized"])
def test_caller_failure_preserves_existing_ownership_contract(connections, failure, helper):
    conn, peer = connections
    # Construct before the caller writes: consolidator initialization commits.
    scope = _scope(helper, conn)
    conn.execute("INSERT INTO markers VALUES ('caller')")
    with pytest.raises(type(failure)) as caught:
        with scope:
            conn.execute("INSERT INTO markers VALUES ('body')")
            raise failure

    assert caught.value is failure
    if helper == "guarded" and not isinstance(conn, _BeamConnection):
        # Legacy raw-connection guard owns the whole transaction, not a savepoint.
        assert not conn.in_transaction
        assert _markers(conn) == []
        _assert_writer_available(peer)
        return

    assert conn.in_transaction
    expected = ["caller", "body"] if helper == "serialized" else ["caller"]
    assert _markers(conn) == expected
    assert _markers(peer) == []
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        peer.execute("BEGIN IMMEDIATE")
    conn.commit()
    assert _markers(peer) == expected
    _assert_writer_available(peer)


@pytest.mark.parametrize("helper", ["write_locked", "guarded", "serialized"])
def test_nested_failure_keeps_outer_scope_usable(connections, failure, helper):
    conn, peer = connections
    if helper == "guarded" and not isinstance(conn, _BeamConnection):
        pytest.skip("raw guarded transactions do not own nested savepoints")
    # Reuse one consolidator to exercise its reentrant lock and entry-time conn.
    if helper == "serialized":
        scope = VeracityConsolidator(conn=conn, federation_probe=False)._serialized_write
    elif helper == "guarded":
        scope = lambda: _guarded_transaction(conn)
    else:
        scope = lambda: _write_locked_transaction(conn)

    with scope():
        conn.execute("INSERT INTO markers VALUES ('outer')")
        with pytest.raises(type(failure)) as caught:
            with scope():
                conn.execute("INSERT INTO markers VALUES ('inner')")
                raise failure
        assert caught.value is failure
        assert conn.in_transaction
        expected = ["outer", "inner"] if helper == "serialized" else ["outer"]
        assert _markers(conn) == expected
        conn.execute("INSERT INTO markers VALUES ('after')")

    assert not conn.in_transaction
    assert _markers(peer) == expected + ["after"]
    _assert_writer_available(peer)


@pytest.mark.parametrize("helper", ["write_locked", "guarded", "serialized"])
@pytest.mark.parametrize("caller_owned", [False, True], ids=["owned", "caller"])
def test_cleanup_sqlite_failure_preserves_original_exception(
    connections, failure, helper, caller_owned
):
    conn, _ = connections
    scope = _scope(helper, conn)
    if caller_owned:
        conn.execute("INSERT INTO markers VALUES ('caller')")
    denied = []

    def deny_rollback(action, operation, name, database, trigger):
        if action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT):
            if operation == "ROLLBACK":
                denied.append(operation)
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        with pytest.raises(type(failure)) as caught:
            with scope:
                conn.execute("INSERT INTO markers VALUES ('body')")
                conn.set_authorizer(deny_rollback)
                raise failure
        assert caught.value is failure
        # _serialized_write intentionally attempts no cleanup for a caller tx.
        assert bool(denied) is not (caller_owned and helper == "serialized")
    finally:
        # Passing None only disables the authorizer on Python 3.11+.
        conn.set_authorizer(lambda *args: sqlite3.SQLITE_OK)
        conn.rollback()


@pytest.mark.parametrize("helper", ["write_locked", "guarded", "serialized"])
@pytest.mark.parametrize("caller_owned", [False, True], ids=["owned", "caller"])
def test_closed_connection_cleanup_preserves_original_exception(
    connections, failure, helper, caller_owned
):
    conn, peer = connections
    scope = _scope(helper, conn)
    if caller_owned:
        conn.execute("INSERT INTO markers VALUES ('caller')")
    with pytest.raises(type(failure)) as caught:
        with scope:
            conn.execute("INSERT INTO markers VALUES ('body')")
            conn.close()
            raise failure
    assert caught.value is failure
    assert _markers(peer) == []
    _assert_writer_available(peer)


@pytest.mark.parametrize("connection_factory", [sqlite3.Connection, _BeamConnection],
                         ids=["plain", "beam"])
@pytest.mark.parametrize("error_type", [sqlite3.OperationalError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("helper,caller_owned", [
    pytest.param("write_locked", False, id="write_locked-owned"),
    pytest.param("write_locked", True, id="write_locked-caller"),
    pytest.param("guarded", False, id="guarded-owned"),
    pytest.param("serialized", False, id="serialized-owned"),
])
def test_commit_failure_or_successful_caller_release(
    tmp_path, connection_factory, error_type, helper, caller_owned
):
    failure = error_type("commit interrupted")

    class InterruptedCommitConnection(connection_factory):
        interrupt_commit = False

        def commit(self):
            if self.interrupt_commit:
                raise failure
            return super().commit()

    path = tmp_path / "commit_interrupt.db"
    conn = sqlite3.connect(path, factory=InterruptedCommitConnection)
    peer = sqlite3.connect(path, timeout=0)
    events = []
    conn.execute("CREATE TABLE markers (value TEXT)")
    try:
        # The consolidator constructor also commits; interrupt only the scope.
        scope = _scope(helper, conn)
        conn.interrupt_commit = True
        if caller_owned:
            # Successful control: caller participation must never call commit().
            # Interrupted caller finalization is exercised at RELEASE below.
            conn.execute("INSERT INTO markers VALUES ('caller')")
            with scope:
                conn.execute("INSERT INTO markers VALUES ('body')")
            assert conn.in_transaction
            assert _markers(conn) == ["caller", "body"]
            assert _markers(peer) == []
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                peer.execute("BEGIN IMMEDIATE")
        else:
            with pytest.raises(error_type) as caught:
                with scope:
                    conn.execute("INSERT INTO markers VALUES ('body')")
                    if isinstance(conn, _BeamConnection):
                        conn._after_commit_hooks.append(lambda: events.append("body"))
            assert caught.value is failure
            assert not conn.in_transaction
            assert _markers(conn) == []
            assert _markers(peer) == []
            _assert_writer_available(peer)
            if isinstance(conn, _BeamConnection):
                assert conn._after_commit_hooks == []
            assert events == []
    finally:
        conn.rollback()
        conn.close()
        peer.close()


@pytest.mark.parametrize("helper,connection_factory", [
    pytest.param("write_locked", sqlite3.Connection, id="write_locked-plain"),
    pytest.param("write_locked", _BeamConnection, id="write_locked-beam"),
    # Only Beam guarded transactions preserve caller ownership via a savepoint.
    pytest.param("guarded", _BeamConnection, id="guarded-beam"),
])
@pytest.mark.parametrize("error_type", [
    sqlite3.OperationalError, RuntimeError, KeyboardInterrupt, SystemExit,
])
@pytest.mark.parametrize("mode", ["before", "deny_rollback", "deny_release", "after"])
def test_release_failure_rolls_back_only_helper_savepoint(
    tmp_path, helper, connection_factory, error_type, mode
):
    failure = error_type("release interrupted")
    releases = []
    denied = []

    def deny_cleanup(action, operation, name, database, trigger):
        target = {"deny_rollback": "ROLLBACK", "deny_release": "RELEASE"}.get(mode)
        if action == sqlite3.SQLITE_SAVEPOINT and operation == target:
            denied.append(operation)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    class InterruptedReleaseConnection(connection_factory):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("RELEASE SAVEPOINT"):
                releases.append(sql)
                if len(releases) == 1:
                    if mode == "after":
                        # Completed release cannot be undone through this savepoint.
                        super().execute(sql, *args, **kwargs)
                    self.set_authorizer(deny_cleanup)
                    raise failure
            return super().execute(sql, *args, **kwargs)

    path = tmp_path / "release_interrupt.db"
    conn = sqlite3.connect(path, factory=InterruptedReleaseConnection)
    peer = sqlite3.connect(path, timeout=0)
    events = []
    conn.execute("CREATE TABLE markers (value TEXT)")
    try:
        conn.execute("INSERT INTO markers VALUES ('caller')")
        if isinstance(conn, _BeamConnection):
            conn._after_commit_hooks.append(lambda: events.append("caller"))
        with pytest.raises(error_type) as caught:
            with _scope(helper, conn):
                conn.execute("INSERT INTO markers VALUES ('body')")
                if isinstance(conn, _BeamConnection):
                    conn._after_commit_hooks.append(lambda: events.append("body"))
        assert caught.value is failure
        assert conn.in_transaction
        expected = ["caller", "body"] if mode in ("deny_rollback", "after") else ["caller"]
        assert _markers(conn) == expected
        assert _markers(peer) == []
        assert denied == ({"deny_rollback": ["ROLLBACK"],
                           "deny_release": ["RELEASE"]}.get(mode, []))
        if mode in ("before", "deny_release"):
            assert len(releases) == 2
            assert releases[0] == releases[1]
        else:
            assert len(releases) == 1
        conn.set_authorizer(lambda *args: sqlite3.SQLITE_OK)
        if mode in ("before", "after"):
            # Successful cleanup removes the helper savepoint. A completed
            # initial release also removes it, but leaves body writes pending.
            savepoint = releases[0].split()[-1]
            with pytest.raises(sqlite3.OperationalError, match="no such savepoint"):
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            peer.execute("BEGIN IMMEDIATE")
        assert events == []
        if isinstance(conn, _BeamConnection) and mode in ("before", "after"):
            assert conn._savepoint_hook_marks == []
        conn.execute("INSERT INTO markers VALUES ('after')")
        conn.commit()
        assert _markers(peer) == expected + ["after"]
        assert events == (expected if isinstance(conn, _BeamConnection) else [])
        _assert_writer_available(peer)
    finally:
        conn.set_authorizer(lambda *args: sqlite3.SQLITE_OK)
        conn.rollback()
        conn.close()
        peer.close()


@pytest.mark.parametrize("helper", ["write_locked", "guarded"])
def test_scoped_failure_discards_only_inner_commit_hooks(tmp_path, failure, helper):
    conn = sqlite3.connect(tmp_path / "hooks.db", factory=_BeamConnection)
    events = []
    conn.execute("CREATE TABLE markers (value TEXT)")
    try:
        with _deferred_commits(conn):
            conn.execute("INSERT INTO markers VALUES ('outer')")
            conn._after_commit_hooks.append(lambda: events.append("outer"))
            with pytest.raises(type(failure)) as caught:
                with _scope(helper, conn):
                    conn.execute("INSERT INTO markers VALUES ('inner')")
                    conn._after_commit_hooks.append(lambda: events.append("inner"))
                    raise failure
            assert caught.value is failure
            assert conn._defer_commit
            assert conn.in_transaction
            assert _markers(conn) == ["outer"]
            assert conn._savepoint_hook_marks == []
            assert events == []
        assert not conn.in_transaction
        assert not conn._defer_commit
        assert events == ["outer"]
    finally:
        conn.close()
