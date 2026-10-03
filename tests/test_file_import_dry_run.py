"""Regression coverage for native-export file imports in dry-run mode."""

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

import mnemosyne.core.annotations as annotations_module
import mnemosyne.core.beam as beam_module
import mnemosyne.core.canonical as canonical_module
import mnemosyne.core.memory as memory_module
import mnemosyne.core.triples as triples_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.triples import TripleStore
import mnemosyne.mcp_tools as mcp_tools


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HERMES_INTEGRATION_SRC = _REPO_ROOT / "integrations" / "hermes" / "src"
if str(_HERMES_INTEGRATION_SRC) not in sys.path:
    sys.path.insert(0, str(_HERMES_INTEGRATION_SRC))


# Use real table contents rather than SQLite's connection-level change counter:
# imports fan out across BEAM, legacy, graph, annotation, canonical, and audit
# stores, so a sorted row snapshot is the public no-mutation contract.
_REQUIRED_CORE_SNAPSHOT_TABLES = frozenset(
    {"working_memory", "episodic_memory", "memories", "scratchpad", "triples"}
)


def _table_snapshot(db_path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    with closing(sqlite3.connect(db_path)) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' "
            "AND sql NOT LIKE 'CREATE VIRTUAL TABLE%' "
            "ORDER BY name"
        ).fetchall()
        snapshot = {}
        for (table,) in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            rows = [tuple(row) for row in conn.execute(f"SELECT * FROM {quoted}")]
            snapshot[table] = tuple(sorted(rows, key=repr))
    return snapshot


def _assert_snapshot_has_required_tables(
    snapshot: dict[str, tuple[tuple[object, ...], ...]], *audit_tables: str
) -> None:
    assert _REQUIRED_CORE_SNAPSHOT_TABLES | set(audit_tables) <= snapshot.keys()


def _assert_imported_store_rows(snapshot: dict[str, tuple[tuple[object, ...], ...]]) -> None:
    assert snapshot["triples"]
    assert snapshot["annotations"]
    assert snapshot["canonical_facts"]


def test_table_snapshot_skips_virtual_tables_and_keeps_ordinary_rows(tmp_path):
    db_path = tmp_path / "snapshot-tables.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL)")
        conn.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, event TEXT NOT NULL)")
        conn.execute("INSERT INTO memories VALUES ('sentinel', 'ordinary row')")
        conn.execute("INSERT INTO audit_log VALUES (1, 'ordinary audit row')")
        conn.execute("CREATE VIRTUAL TABLE memory_search USING fts5(content)")
        conn.commit()

    snapshot = _table_snapshot(db_path)

    assert {"memories", "audit_log"} <= snapshot.keys()
    assert "memory_search" not in snapshot
    assert snapshot["memories"] == (("sentinel", "ordinary row"),)
    assert snapshot["audit_log"] == ((1, "ordinary audit row"),)


def _clone_database(source: Mnemosyne, clone_path: Path) -> None:
    with closing(sqlite3.connect(clone_path)) as clone:
        source.conn.backup(clone)
        clone.commit()


def _capture_dry_run_clone_dirs(monkeypatch) -> list[Path]:
    """Record clone directories while preserving TemporaryDirectory semantics."""
    clone_dirs = []
    original_temporary_directory = memory_module.tempfile.TemporaryDirectory

    def tracking_temporary_directory(*args, **kwargs):
        directory = original_temporary_directory(*args, **kwargs)
        clone_dirs.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        memory_module.tempfile, "TemporaryDirectory", tracking_temporary_directory
    )
    return clone_dirs


def _export_source(tmp_path: Path) -> Path:
    source = Mnemosyne(session_id="source", db_path=tmp_path / "source.db")
    memory_id = source.remember("file-import dry-run source working memory", source="test")
    source.scratchpad_write("file-import dry-run source scratchpad")
    triples = TripleStore(db_path=source.db_path)
    try:
        triples.add(
            "file-import-dry-run-source",
            "has_fixture",
            "triple",
            source="test",
        )
    finally:
        triples.conn.close()
    source.beam.annotations.add(
        memory_id,
        "mentions",
        "file-import-dry-run annotation",
        source="test",
    )
    source.beam.canonical.remember(
        "source",
        "fixture",
        "canonical",
        "file-import-dry-run canonical fact",
        source="test",
    )
    source.conn.execute(
        "INSERT INTO memories "
        "(id, content, source, timestamp, session_id, importance, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ",
        (
            "file-import-dry-run-legacy",
            "file-import dry-run source legacy memory",
            "test",
            "2026-07-25T00:00:00",
            "source",
            0.5,
            "{}",
        ),
    )
    source.conn.commit()
    export_path = tmp_path / "source-export.json"
    source.export_to_file(str(export_path))
    return export_path


def _target(tmp_path: Path, name: str = "target") -> Mnemosyne:
    target = Mnemosyne(session_id=name, db_path=tmp_path / f"{name}.db")
    target.remember("file-import dry-run target sentinel", source="test")
    return target


def _cli_args(input_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mnemosyne_cmd="import",
        from_provider=None,
        input=str(input_path),
        file=None,
        force=True,
        dry_run=True,
        session_id="cli-session",
        channel_id=None,
        list_providers=False,
        generate_script=False,
        agentic=False,
        output_script=None,
        api_key=None,
        user_id=None,
        agent_id=None,
        base_url=None,
        bank=None,
    )


def test_core_file_import_dry_run_matches_clone_and_preserves_all_rows(
    tmp_path, monkeypatch
):
    export_path = _export_source(tmp_path)
    target = _target(tmp_path)
    before = _table_snapshot(target.db_path)
    _assert_snapshot_has_required_tables(before)

    # Hold the real target's active module caches, not just the wrapper fields:
    # clone construction must not detach either caller-owned connection.
    active_core_conn = memory_module._thread_local.conn
    active_core_path = memory_module._thread_local.db_path
    active_beam_conn = beam_module._thread_local.conn
    active_beam_path = beam_module._thread_local.db_path
    assert active_core_conn is target.conn
    assert active_beam_conn is target.beam.conn

    clone_dirs = _capture_dry_run_clone_dirs(monkeypatch)
    dry_stats = target.import_from_file(str(export_path), force=True, dry_run=True)
    after_dry_run = _table_snapshot(target.db_path)

    assert memory_module._thread_local.conn is active_core_conn
    assert memory_module._thread_local.db_path == active_core_path
    assert beam_module._thread_local.conn is active_beam_conn
    assert beam_module._thread_local.db_path == active_beam_path
    assert active_core_conn.execute("SELECT 1").fetchone()[0] == 1
    assert active_beam_conn.execute("SELECT 1").fetchone()[0] == 1
    assert clone_dirs
    assert all(clone_dir.parent == target.db_path.parent for clone_dir in clone_dirs)
    assert all(not clone_dir.exists() for clone_dir in clone_dirs)

    clone_path = tmp_path / "normal-import-clone.db"
    _clone_database(target, clone_path)
    normal_target = Mnemosyne(session_id="target", db_path=clone_path)
    normal_stats = normal_target.import_from_file(str(export_path), force=True)

    assert dry_stats == normal_stats
    assert normal_stats["triples"]["inserted"] == 1
    assert normal_stats["annotations"]["inserted"] == 1
    assert normal_stats["canonical"]["inserted"] == 1
    assert after_dry_run == before
    _assert_imported_store_rows(_table_snapshot(normal_target.db_path))

    normal_stats_on_target = target.import_from_file(str(export_path), force=True)
    assert normal_stats_on_target == normal_stats
    after_normal_import = _table_snapshot(target.db_path)
    assert after_normal_import != before
    _assert_imported_store_rows(after_normal_import)


def test_core_file_import_dry_run_closes_clone_store_connections_on_error(
    tmp_path, monkeypatch
):
    export_path = _export_source(tmp_path)
    target = _target(tmp_path)
    before = _table_snapshot(target.db_path)
    _assert_snapshot_has_required_tables(before)
    active_core_conn = memory_module._thread_local.conn
    active_core_path = memory_module._thread_local.db_path
    active_beam_conn = beam_module._thread_local.conn
    active_beam_path = beam_module._thread_local.db_path
    closed_connections = []
    clone_cache_connections = []
    clone_query_caches = []
    init_connections = []
    clone_dirs = _capture_dry_run_clone_dirs(monkeypatch)

    original_query_cache = beam_module.QueryCache

    class TrackingQueryCache(original_query_cache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            clone_query_caches.append(self)

        def close(self):
            self.closed = True
            super().close()

    caller_query_cache = original_query_cache(
        db_path=target.db_path.parent / "query_cache.db"
    )
    target.beam._query_cache = caller_query_cache

    class CloseSpyConnection:
        def __init__(self, connection, tracker=closed_connections):
            self._connection = connection
            self._tracker = tracker
            self.closed = False

        def close(self):
            self.closed = True
            self._tracker.append(self)
            self._connection.close()

        def __getattr__(self, name):
            return getattr(self._connection, name)

    def tracking_store(store_type):
        class TrackingStore(store_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.conn = CloseSpyConnection(self.conn)

        return TrackingStore

    original_core_get_connection = memory_module._get_connection
    original_beam_get_connection = beam_module._get_connection

    def tracking_core_get_connection(db_path=None):
        connection = original_core_get_connection(db_path)
        if (
            db_path is not None
            and Path(db_path) != target.db_path
            and not isinstance(connection, CloseSpyConnection)
        ):
            connection = CloseSpyConnection(connection, clone_cache_connections)
            memory_module._thread_local.conn = connection
        return connection

    def tracking_beam_get_connection(db_path: Path | None = None):
        connection = (
            original_beam_get_connection()
            if db_path is None
            else original_beam_get_connection(db_path)
        )
        if (
            db_path is not None
            and Path(db_path) != target.db_path
            and not isinstance(connection, CloseSpyConnection)
        ):
            connection = CloseSpyConnection(connection, clone_cache_connections)
            beam_module._thread_local.conn = connection
        return connection

    monkeypatch.setattr(memory_module, "_get_connection", tracking_core_get_connection)
    monkeypatch.setattr(beam_module, "_get_connection", tracking_beam_get_connection)
    monkeypatch.setattr(beam_module, "QueryCache", TrackingQueryCache)

    original_import_from_dict = BeamMemory.import_from_dict

    def import_with_clone_query_cache(self, *args, **kwargs):
        if self.db_path != target.db_path:
            # Match recall_enhanced's lazy cache construction without making
            # this cleanup regression depend on recall ranking or embeddings.
            self._query_cache = beam_module.QueryCache(
                db_path=self.db_path.parent / "query_cache.db"
            )
        return original_import_from_dict(self, *args, **kwargs)

    monkeypatch.setattr(BeamMemory, "import_from_dict", import_with_clone_query_cache)

    monkeypatch.setattr(
        triples_module, "TripleStore", tracking_store(triples_module.TripleStore)
    )
    original_init_triples = triples_module.init_triples

    def tracking_init_triples(*args, **kwargs):
        original_get_conn = triples_module._get_conn

        def tracking_get_conn(*args, **kwargs):
            connection = CloseSpyConnection(
                original_get_conn(*args, **kwargs), init_connections
            )
            init_connections.append(connection)
            return connection

        triples_module._get_conn = tracking_get_conn
        try:
            return original_init_triples(*args, **kwargs)
        finally:
            triples_module._get_conn = original_get_conn

    monkeypatch.setattr(triples_module, "init_triples", tracking_init_triples)
    monkeypatch.setattr(
        annotations_module,
        "AnnotationStore",
        tracking_store(annotations_module.AnnotationStore),
    )
    tracking_canonical = tracking_store(canonical_module.CanonicalStore)

    def raise_after_clone_stores_created(self, *_args, **_kwargs):
        raise RuntimeError("forced clone import failure")

    monkeypatch.setattr(
        tracking_canonical, "import_all", raise_after_clone_stores_created
    )
    monkeypatch.setattr(canonical_module, "CanonicalStore", tracking_canonical)

    with pytest.raises(RuntimeError, match="forced clone import failure"):
        target.import_from_file(str(export_path), force=True, dry_run=True)

    assert len(closed_connections) == 3
    assert len(clone_cache_connections) == 2
    assert all(connection.closed for connection in clone_cache_connections)
    assert len(clone_query_caches) == 1
    assert clone_query_caches[0].closed
    assert init_connections
    assert all(connection.closed for connection in init_connections)
    assert memory_module._thread_local.conn is active_core_conn
    assert memory_module._thread_local.db_path == active_core_path
    assert beam_module._thread_local.conn is active_beam_conn
    assert beam_module._thread_local.db_path == active_beam_path
    assert active_core_conn.execute("SELECT 1").fetchone()[0] == 1
    assert active_beam_conn.execute("SELECT 1").fetchone()[0] == 1
    assert target.beam._query_cache is caller_query_cache
    assert caller_query_cache._conn.execute("SELECT 1").fetchone()[0] == 1
    assert _table_snapshot(target.db_path) == before
    assert clone_dirs
    assert all(not clone_dir.exists() for clone_dir in clone_dirs)
    caller_query_cache.close()


def test_core_file_import_dry_run_rejects_unsupported_export_without_mutation(
    tmp_path, monkeypatch
):
    export_path = _export_source(tmp_path)
    export = json.loads(export_path.read_text(encoding="utf-8"))
    export["mnemosyne_export"]["version"] = "unsupported"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    target = _target(tmp_path)
    before = _table_snapshot(target.db_path)
    clone_dirs = _capture_dry_run_clone_dirs(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported export version: unsupported"):
        target.import_from_file(str(export_path), force=True, dry_run=True)

    assert _table_snapshot(target.db_path) == before
    assert clone_dirs
    assert all(clone_dir.parent == target.db_path.parent for clone_dir in clone_dirs)
    assert all(not clone_dir.exists() for clone_dir in clone_dirs)


def test_mcp_file_import_dry_run_has_stable_output_and_no_mutation(tmp_path, monkeypatch):
    export_path = _export_source(tmp_path)
    target = _target(tmp_path)
    before = _table_snapshot(target.db_path)
    _assert_snapshot_has_required_tables(before)
    handler_target = [target]
    monkeypatch.setattr(
        mcp_tools, "_create_instance", lambda **_kwargs: handler_target[0]
    )

    arguments = {"input_path": str(export_path), "force": True, "dry_run": True}
    first = mcp_tools._handle_import(arguments)
    second = mcp_tools._handle_import(arguments)

    assert first == second
    assert first["status"] == "dry_run"
    assert first["dry_run"] is True
    assert _table_snapshot(target.db_path) == before

    clone_path = tmp_path / "mcp-normal-import-clone.db"
    _clone_database(target, clone_path)
    handler_target[0] = Mnemosyne(session_id="target", db_path=clone_path)
    normal = mcp_tools._handle_import(
        {"input_path": str(export_path), "force": True, "dry_run": False}
    )

    assert normal["status"] == "imported"
    assert normal["dry_run"] is False
    assert first["stats"] == normal["stats"]
    assert normal["stats"]["triples"]["inserted"] == 1
    assert _table_snapshot(target.db_path) == before


@pytest.mark.parametrize(
    ("module_name", "provider_name", "audit_table"),
    [
        ("hermes_memory_provider", "MnemosyneMemoryProvider", "audit_log"),
        ("mnemosyne_hermes", "MnemosyneMemoryProvider", "memory_audit_events"),
    ],
)
def test_hermes_file_import_dry_run_preserves_data_and_audit_rows(
    tmp_path, monkeypatch, module_name, provider_name, audit_table
):
    export_path = _export_source(tmp_path)
    db_path = tmp_path / f"{module_name.replace('.', '-')}.db"
    # A real provider DB has already received the core schema during startup.
    # Initialize it before the before/after snapshot so this test isolates the
    # import route rather than first-run schema creation.
    target = Mnemosyne(session_id="provider", db_path=db_path)
    beam = BeamMemory(session_id="provider", db_path=db_path)
    provider_class = getattr(pytest.importorskip(module_name), provider_name)
    provider = provider_class()
    provider._beam = beam
    provider._session_id = "provider"
    provider._init_audit_log()

    before = _table_snapshot(db_path)
    _assert_snapshot_has_required_tables(before, audit_table)
    calls = []
    original_import = target.import_from_file

    def import_spy(input_path, force=False, dry_run=False):
        calls.append((input_path, force, dry_run))
        return original_import(input_path, force=force, dry_run=dry_run)

    def provider_memory_factory(**kwargs):
        candidate_path = kwargs.get("db_path")
        if candidate_path is not None and Path(candidate_path) == db_path:
            return target
        return Mnemosyne(**kwargs)

    monkeypatch.setattr(target, "import_from_file", import_spy)
    monkeypatch.setattr(
        "mnemosyne.core.memory.Mnemosyne", provider_memory_factory
    )
    dry_run_result = json.loads(provider._handle_import(
        {"input_path": str(export_path), "force": True, "dry_run": True}
    ))

    assert dry_run_result["status"] == "dry_run"
    assert dry_run_result["dry_run"] is True
    assert calls == [(str(export_path), True, True)]
    assert _table_snapshot(db_path) == before

    normal_result = json.loads(provider._handle_import(
        {"input_path": str(export_path), "force": True, "dry_run": False}
    ))

    assert normal_result["status"] == "imported"
    assert normal_result["dry_run"] is False
    assert normal_result["stats"] == dry_run_result["stats"]
    assert calls == [
        (str(export_path), True, True),
        (str(export_path), True, False),
    ]
    after_normal = _table_snapshot(db_path)
    assert len(after_normal[audit_table]) == len(before[audit_table]) + 1
    _assert_imported_store_rows(after_normal)
    if provider._audit is not None:
        provider._audit.close()


@pytest.mark.parametrize(
    "module_name",
    ["hermes_memory_provider.cli", "mnemosyne_hermes.cli"],
)
def test_hermes_cli_file_import_forwards_dry_run_and_keeps_exit_compat(
    tmp_path, monkeypatch, module_name, capsys
):
    export_path = _export_source(tmp_path)
    target = _target(tmp_path, name=module_name.replace(".", "-"))
    before = _table_snapshot(target.db_path)
    _assert_snapshot_has_required_tables(before)
    calls = []
    original_import = target.import_from_file
    original_mnemosyne = Mnemosyne

    def import_spy(input_path, force=False, dry_run=False):
        calls.append((input_path, force, dry_run))
        return original_import(input_path, force=force, dry_run=dry_run)

    def cli_memory_factory(**kwargs):
        # The outer CLI construction has no explicit database path.  The core
        # dry-run implementation constructs an isolated clone with one, which
        # must remain a real Mnemosyne rather than recurse through this spy.
        if kwargs.get("db_path") is not None:
            return original_mnemosyne(**kwargs)
        return target

    monkeypatch.setattr(target, "import_from_file", import_spy)
    monkeypatch.setattr("mnemosyne.core.memory.Mnemosyne", cli_memory_factory)

    cli = pytest.importorskip(module_name)
    assert cli.mnemosyne_command(_cli_args(export_path)) == 0
    output = capsys.readouterr().out
    assert calls == [(str(export_path), True, True)]
    assert output.count("  (force mode: overwrites would be applied)") == 1
    assert "  (force mode: overwrites applied)" not in output
    assert output.count("  (dry-run mode: no memories were written)") == 1
    assert output.index("  Triples:") < output.index(
        "  (force mode: overwrites would be applied)"
    ) < output.index("  (dry-run mode: no memories were written)")
    assert _table_snapshot(target.db_path) == before

    normal_args = _cli_args(export_path)
    normal_args.dry_run = False
    assert cli.mnemosyne_command(normal_args) == 0
    normal_output = capsys.readouterr().out
    assert calls == [
        (str(export_path), True, True),
        (str(export_path), True, False),
    ]
    assert normal_output.count("  (force mode: overwrites applied)") == 1
    assert "  (force mode: overwrites would be applied)" not in normal_output
    assert "  (dry-run mode: no memories were written)" not in normal_output
    after_normal = _table_snapshot(target.db_path)
    assert after_normal != before
    _assert_imported_store_rows(after_normal)


@pytest.mark.parametrize(
    "module_name",
    ["hermes_memory_provider.cli", "mnemosyne_hermes.cli"],
)
def test_hermes_cli_file_import_dry_run_failure_preserves_data(
    tmp_path, monkeypatch, module_name
):
    export_path = _export_source(tmp_path)
    target = _target(tmp_path, name=f"failure-{module_name.replace('.', '-')}")
    before = _table_snapshot(target.db_path)
    original_mnemosyne = Mnemosyne

    def cli_memory_factory(**kwargs):
        if kwargs.get("db_path") is not None:
            return original_mnemosyne(**kwargs)
        return target

    def fail_import(*_args, **_kwargs):
        raise ValueError("forced dry-run validation failure")

    monkeypatch.setattr(target, "import_from_file", fail_import)
    monkeypatch.setattr("mnemosyne.core.memory.Mnemosyne", cli_memory_factory)
    cli = pytest.importorskip(module_name)

    assert cli.mnemosyne_command(_cli_args(export_path)) == 1
    assert _table_snapshot(target.db_path) == before
