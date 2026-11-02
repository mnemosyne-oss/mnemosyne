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
        conn.set_authorizer(None)
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
