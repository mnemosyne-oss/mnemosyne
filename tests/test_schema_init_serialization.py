"""Real thread/process regression for issue #833; temporary databases only."""
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from mnemosyne.core import beam


def initialize(path, barrier, legacy=False):
    barrier.wait(timeout=30)
    if legacy:
        from mnemosyne.core.memory import init_db
        init_db(Path(path))
    else:
        beam.init_beam(Path(path))


def check(path):
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        columns = [r[1] for r in conn.execute('PRAGMA table_info(working_memory)')]
        assert columns.count('recall_count') == 1
    beam.init_beam(path)


@pytest.mark.parametrize('legacy', [False, True])
def test_threads(tmp_path, legacy):
    import threading
    path = tmp_path / 'threads.db'
    for _ in range(3):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(initialize, str(path), barrier, legacy) for _ in range(2)]
            for future in futures:
                future.result(timeout=60)
        check(path)


@pytest.mark.parametrize('legacy', [False, True])
def test_processes(tmp_path, legacy):
    ctx = multiprocessing.get_context('spawn')
    path = tmp_path / 'processes.db'
    barrier = ctx.Barrier(2)
    children = [ctx.Process(target=initialize, args=(str(path), barrier, legacy)) for _ in range(2)]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(60)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()
    check(path)


def test_alias(tmp_path):
    path = tmp_path / 'real.db'
    beam.init_beam(path)
    alias = tmp_path / 'alias.db'
    alias.symlink_to(path)
    beam.init_beam(alias)
    assert beam._get_connection(alias) is beam._get_connection(path)
    assert not Path(str(alias) + '.init.lock').exists()


@pytest.mark.parametrize('message', ['disk I/O error', 'database is locked', 'attempt to write a readonly database'])
def test_errors_propagate(tmp_path, monkeypatch, message):
    def fail(path):
        raise sqlite3.OperationalError(message)
    monkeypatch.setattr(beam, '_get_connection', fail)
    with pytest.raises(sqlite3.OperationalError, match=message):
        beam.init_beam(tmp_path / 'failure.db')


def test_constructor_locks_first_connection(tmp_path, monkeypatch):
    import contextlib
    from mnemosyne.core import memory
    active = False
    observations = []
    original_lock = beam._schema_init_lock
    original_get = memory._get_connection

    @contextlib.contextmanager
    def marked(path):
        nonlocal active
        with original_lock(path) as canonical:
            active = True
            try:
                yield canonical
            finally:
                active = False

    def get(path):
        observations.append(active)
        return original_get(path)

    monkeypatch.setattr(beam, '_schema_init_lock', marked)
    monkeypatch.setattr(memory, '_get_connection', get)
    memory.Mnemosyne(db_path=tmp_path / 'constructor.db')
    assert observations[0] is True


def test_sidecar_error(tmp_path):
    path = tmp_path / 'failure.db'
    Path(str(path) + '.init.lock').mkdir()
    with pytest.raises(IsADirectoryError):
        beam.init_beam(path)


def test_existing_migration_schema_type_must_match(tmp_path):
    path = tmp_path / 'schema.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE working_memory (id TEXT, consolidated_at INTEGER)')
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.OperationalError, match='schema mismatch'):
            beam._add_column_if_missing(conn, 'working_memory', 'consolidated_at', 'TEXT')


def test_existing_migration_schema_nullability_must_match(tmp_path):
    path = tmp_path / 'schema.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE working_memory (id TEXT, consolidation_claimed_at TEXT NOT NULL)')
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.OperationalError, match='schema mismatch'):
            beam._add_column_if_missing(conn, 'working_memory', 'consolidation_claimed_at', 'TEXT')


@pytest.mark.parametrize('message', ['disk I/O error', 'attempt to write a readonly database'])
def test_migration_ddl_errors_propagate(tmp_path, monkeypatch, message):
    original = beam._add_column_if_missing

    def fail(conn, table, column, col_type):
        if column == 'consolidated_at':
            raise sqlite3.OperationalError(message)
        return original(conn, table, column, col_type)

    monkeypatch.setattr(beam, '_add_column_if_missing', fail)
    with pytest.raises(sqlite3.OperationalError, match=message):
        beam.init_beam(tmp_path / 'ddl.db')


def test_duplicate_migration_is_not_false_positive(tmp_path):
    path = tmp_path / 'schema.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE working_memory (id TEXT, consolidated_at TEXT)')
    result = beam._add_column_if_missing(sqlite3.connect(path), 'working_memory', 'consolidated_at', 'TEXT')
    assert result is False


def test_existing_default_must_match_when_not_requested(tmp_path):
    path = tmp_path / 'schema.db'
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE working_memory (id TEXT, consolidated_at TEXT DEFAULT 'bad')")
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.OperationalError, match='schema mismatch'):
            beam._add_column_if_missing(conn, 'working_memory', 'consolidated_at', 'TEXT')


def test_default_different_tolerated_when_expected_has_default(tmp_path):
    # Idempotent startup: an existing column with a different default (or none)
    # is valid as-is when the expected declaration carries a DEFAULT — enforcing
    # an exact default here would break pre-existing/legacy databases.
    path = tmp_path / 'schema.db'
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (id TEXT, c TEXT DEFAULT 'a')")
    result = beam._add_column_if_missing(sqlite3.connect(path), 't', 'c', "TEXT DEFAULT 'b'")
    assert result is False


def test_default_none_tolerated_when_expected_has_default(tmp_path):
    # Regression for CI: working_memory.veracity exists without a default in
    # doctor-bank-routing DBs; expected TEXT DEFAULT 'unknown' must not make
    # startup fail.
    path = tmp_path / 'schema.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE working_memory (id TEXT, veracity TEXT)')
    result = beam._add_column_if_missing(sqlite3.connect(path), 'working_memory', 'veracity', "TEXT DEFAULT 'unknown'")
    assert result is False


def test_default_exact_match_accepted(tmp_path):
    path = tmp_path / 'schema.db'
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (id TEXT, c TEXT DEFAULT 'x')")
    result = beam._add_column_if_missing(sqlite3.connect(path), 't', 'c', "TEXT DEFAULT 'x'")
    assert result is False


def _make_race_conn(path, wrong_dup=False, winner_ddl='ALTER TABLE target ADD COLUMN added TEXT'):
    real = sqlite3.connect(path)
    real.execute('CREATE TABLE target (id INTEGER PRIMARY KEY)')
    real.commit()

    class RaceCursor:
        def __init__(self, cursor):
            self._cursor = cursor
        def execute(self, sql, params=()):
            if sql.startswith('ALTER TABLE target ADD COLUMN added'):
                if wrong_dup:
                    raise sqlite3.OperationalError('duplicate column name: unrelated_column')
                # Winner: actually add the column, then report the duplicate.
                self._cursor.execute(winner_ddl)
                raise sqlite3.OperationalError('duplicate column name: added')
            return self._cursor.execute(sql, params)
        def fetchall(self):
            return self._cursor.fetchall()

    class RaceConnection:
        def cursor(self):
            return RaceCursor(real.cursor())
        def commit(self):
            return real.commit()
        def close(self):
            return real.close()

    return RaceConnection()


def test_duplicate_race_suppressed_after_verify(tmp_path):
    # F2: ALTER races with a concurrent winner that adds the column between our
    # read and write; the duplicate must be verified and suppressed, not raised.
    conn = _make_race_conn(tmp_path / 'race.db', wrong_dup=False)
    result = beam._add_column_if_missing(conn, 'target', 'added', 'TEXT')
    assert result is False
    conn.close()


def test_duplicate_race_wrong_column_not_suppressed(tmp_path):
    # E3: a duplicate reported for an unrelated column must NOT be suppressed.
    conn = _make_race_conn(tmp_path / 'race.db', wrong_dup=True)
    with pytest.raises(sqlite3.OperationalError, match='duplicate column name'):
        beam._add_column_if_missing(conn, 'target', 'added', 'TEXT')
    conn.close()


def test_duplicate_race_wrong_default_not_suppressed(tmp_path):
    # A duplicate for the requested column still fails when the winner used a
    # different default than the requested declaration.
    conn = _make_race_conn(
        tmp_path / 'race.db',
        winner_ddl="ALTER TABLE target ADD COLUMN added TEXT DEFAULT 'wrong'",
    )
    with pytest.raises(sqlite3.OperationalError, match='duplicate column name'):
        beam._add_column_if_missing(conn, 'target', 'added', "TEXT DEFAULT 'expected'")
    conn.close()


def test_duplicate_race_missing_default_not_suppressed(tmp_path):
    # A requested default must also be present on a concurrent winner.
    conn = _make_race_conn(
        tmp_path / 'race.db',
        winner_ddl='ALTER TABLE target ADD COLUMN added TEXT',
    )
    with pytest.raises(sqlite3.OperationalError, match='duplicate column name'):
        beam._add_column_if_missing(conn, 'target', 'added', "TEXT DEFAULT 'expected'")
    conn.close()


def test_unloadable_sqlite_vec_skips_vec_tables(tmp_path, monkeypatch):
    # The Python package may be installed even when this connection could not
    # load its extension; schema init must keep the optional-vector fallback.
    path = tmp_path / 'unloadable.db'
    conn = sqlite3.connect(path)
    monkeypatch.setattr(beam, '_SQLITE_VEC_AVAILABLE', True)
    monkeypatch.setattr(beam, '_detect_vec_type', lambda _conn: 'float32')
    monkeypatch.setattr(beam, '_get_connection', lambda _path: conn)
    beam._init_beam_locked(path)
    names = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert 'vec_episodes' not in names
    assert 'vec_working' not in names
    assert 'vec_facts' not in names
    conn.close()


def test_loaded_sqlite_vec_ddl_errors_propagate(tmp_path, monkeypatch):
    path = tmp_path / 'vector-failure.db'
    class FaultCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if 'CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0' in ' '.join(str(sql).split()):
                raise sqlite3.OperationalError('disk I/O error')
            return super().execute(sql, parameters)

    class LoadedConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            kwargs['factory'] = FaultCursor
            return super().cursor(*args, **kwargs)

    conn = sqlite3.connect(path, factory=LoadedConnection)
    conn._mnemosyne_vec_loaded = True
    monkeypatch.setattr(beam, '_SQLITE_VEC_AVAILABLE', True)
    monkeypatch.setattr(beam, '_detect_vec_type', lambda _conn: 'float32')
    monkeypatch.setattr(beam, '_get_connection', lambda _path: conn)
    with pytest.raises(sqlite3.OperationalError, match='disk I/O error'):
        beam._init_beam_locked(path)
    conn.close()
