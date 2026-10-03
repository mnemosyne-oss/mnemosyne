"""Tests for profile-isolation-aware bank resolution in the hermes CLI.

Regression coverage for #362: `hermes mnemosyne stats` (and friends) used to
always bind to the default/legacy bank, so under `profile_isolation` they
reported empty state while the profile bank held the real data.

Standalone-import coverage for #373: when Hermes loads the plugin CLI module
via ``importlib.util.spec_from_file_location()``, the module has no parent
package and the previous relative import of ``MnemosyneMemoryProvider`` failed
silently, again falling back to the default bank.
"""

import importlib.util
import json
import sqlite3
import types
from pathlib import Path

import pytest
from mnemosyne.core.annotations import AnnotationStore
from mnemosyne.core.canonical import CanonicalStore
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.triples import TripleStore

import mnemosyne_hermes as _mnh
from mnemosyne_hermes.cli import (
    _BANK_RESOLUTION_FAILED,
    _EXPORT_REQUIRED_COLUMNS,
    _EXPORT_REQUIRED_TABLES,
    _completeness_details,
    _export_schema_is_complete_read_only,
    _get_provider_class,
    _resolve_cli_bank,
    mnemosyne_command,
)


def _args(**kw):
    return types.SimpleNamespace(**kw)


def _write_config(home, isolation):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"memory:\n  mnemosyne:\n    profile_isolation: {isolation}\n"
    )


def _export_args(output, bank=None):
    return _args(mnemosyne_cmd="export", output=str(output), bank=bank)


def _file_import_args(input_path):
    return _args(
        mnemosyne_cmd="import",
        input=str(input_path),
        bank=None,
        force=False,
        dry_run=False,
        list_providers=False,
        generate_script=False,
        agentic=False,
        from_provider=None,
        output_script=None,
        session_id=None,
        channel_id=None,
    )


def _seed_export_sections(bank, label, *, include_episodic_embedding=False):
    """Seed every non-optional export section with bank-specific content."""
    marker = f"bank-marker-{label}"
    memory = Mnemosyne(session_id="hermes_default", bank=bank)
    memory_id = memory.remember(f"working-{marker}")
    memory.beam.consolidate_to_episodic(f"episodic-{marker}", [memory_id])
    memory.scratchpad_write(f"scratchpad-{marker}")
    TripleStore(db_path=memory.db_path).add(f"triple-{marker}", "has", "value")
    AnnotationStore(db_path=memory.db_path).add(memory_id, "fact", f"annotation-{marker}")
    CanonicalStore(db_path=memory.db_path).remember(
        f"owner-{marker}", "profile", "name", f"canonical-{marker}"
    )
    # The public writers above cover the normal memory sections. The exporter
    # also reads these persistent rows directly, so use a disposable test DB to
    # make their bank provenance observable without a vector backend or an
    # actual consolidation run.
    if include_episodic_embedding:
        # Use Beam's connection so a real vec0 table, when present, can be
        # replaced safely with this disposable, deterministic test table.
        episode_rowid = memory.conn.execute(
            "SELECT rowid FROM episodic_memory WHERE content = ?",
            (f"episodic-{marker}",),
        ).fetchone()[0]
        memory.conn.execute("DROP TABLE IF EXISTS vec_episodes")
        memory.conn.execute(
            "CREATE TABLE vec_episodes (rowid INTEGER PRIMARY KEY, embedding TEXT NOT NULL)"
        )
        memory.conn.execute(
            "INSERT INTO vec_episodes (rowid, embedding) VALUES (?, ?)",
            (episode_rowid, json.dumps([f"episodic-embedding-{marker}"])),
        )
        memory.conn.commit()
    with sqlite3.connect(memory.db_path) as conn:
        conn.execute(
            "INSERT INTO consolidation_log "
            "(session_id, items_consolidated, summary_preview, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("hermes_default", 1, f"consolidation-{marker}", "2026-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO memory_embeddings "
            "(memory_id, embedding_json, model, created_at) VALUES (?, ?, ?, ?)",
            (
                f"legacy-embedding-{marker}",
                json.dumps([f"legacy-embedding-{marker}"]),
                "test",
                "2026-01-01T00:00:00",
            ),
        )


def _read_export(path):
    return json.loads(path.read_text())


def _assert_export_has_only_label(payload, selected, excluded):
    """Assert every non-optional payload section is selected-bank-only."""
    marker = f"bank-marker-{selected}"
    excluded_marker = f"bank-marker-{excluded}"
    expected_sections = {
        "working_memory": f"working-{marker}",
        "episodic_memory": f"episodic-{marker}",
        "episodic_embeddings": f"episodic-embedding-{marker}",
        "scratchpad": f"scratchpad-{marker}",
        "consolidation_log": f"consolidation-{marker}",
        "legacy_memories": f"working-{marker}",
        "legacy_embeddings": f"legacy-embedding-{marker}",
        "triples": f"triple-{marker}",
        "annotations": f"annotation-{marker}",
        "canonical_facts": f"canonical-{marker}",
    }
    for section, expected_marker in expected_sections.items():
        serialized = json.dumps(payload[section])
        assert expected_marker in serialized, f"{section} omitted selected-bank content"
        assert excluded_marker not in serialized, f"{section} leaked other-bank content"
    # Section-specific presence above proves that all exporter sections were
    # actually exercised; this whole-payload check catches a future section
    # added to the exporter that accidentally binds to the default database.
    assert excluded_marker not in json.dumps(payload)


def test_explicit_bank_takes_precedence_and_is_sanitized(monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert _resolve_cli_bank(_args(bank="Work Stuff"), "stats") == "work_stuff"


def test_profile_bank_resolved_when_isolation_enabled(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "zedd"
    _write_config(home, "true")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank=None), "stats") == "zedd"


def test_default_bank_when_isolation_disabled(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "zedd"
    _write_config(home, "false")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank=None), "stats") is None


def test_default_bank_when_no_config(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "zedd"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank=None), "stats") is None


def test_root_hermes_home_is_treated_as_default(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    _write_config(home, "true")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The base profile's HERMES_HOME basename (.hermes) maps to the shared bank.
    assert _resolve_cli_bank(_args(bank=None), "stats") is None


def test_import_bank_arg_does_not_redirect_target(tmp_path, monkeypatch):
    # `import --bank` names the SOURCE provider bank (e.g. Hindsight), not the
    # Mnemosyne destination, so it must not be used as the CLI's target bank.
    home = tmp_path / "profiles" / "zedd"
    _write_config(home, "true")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank="hindsight"), "import") == "zedd"


def test_get_provider_class_returns_real_class():
    """The helper must return an actual class, not None or a dummy."""
    cls = _get_provider_class()
    assert cls is not None
    assert hasattr(cls, "_sanitize_bank_name")


def test_standalone_load_via_spec_resolves_profile_bank(tmp_path, monkeypatch):
    """End-to-end standalone load: CLI module loaded from file path
    (no __package__) resolves the active profile bank."""
    home = tmp_path / "profiles" / "work"
    _write_config(home, "true")

    # Locate the installed package's cli.py on disk
    pkg_dir = Path(_mnh.__file__).resolve().parent
    cli_py = pkg_dir / "cli.py"
    assert cli_py.exists(), f"cli.py not found next to package at {pkg_dir}"

    spec = importlib.util.spec_from_file_location("_clitest_cli", str(cli_py))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # The standalone load context should give us no package metadata
    pre_pkg = getattr(mod, "__package__", None)
    assert pre_pkg in (None, ""), f"expected no package, got {pre_pkg!r}"
    spec.loader.exec_module(mod)

    # The module should expose the patched helper + resolver
    assert hasattr(mod, "_resolve_cli_bank")

    # Verify the helper picks the absolute-import path
    cls = mod._get_provider_class()
    assert cls is not None
    assert hasattr(cls, "_sanitize_bank_name")

    # Verify bank resolution works end-to-end without leaking HERMES_HOME
    # into later tests.
    monkeypatch.setenv("HERMES_HOME", str(home))
    result = mod._resolve_cli_bank(_args(bank=None), "stats")
    assert result == "work", (
        f"standalone load: expected 'work', got {result!r}. "
        "This indicates the absolute-import fallback failed."
    )


def test_export_explicit_bank_has_no_default_content_in_seedable_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda conn: True)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    _seed_export_sections(None, "default", include_episodic_embedding=True)
    _seed_export_sections("work", "work", include_episodic_embedding=True)

    output = tmp_path / "work.json"
    assert mnemosyne_command(_export_args(output, bank="work")) == 0
    _assert_export_has_only_label(_read_export(output), "work", "default")


def test_export_profile_isolation_has_no_default_content_in_seedable_sections(
    tmp_path, monkeypatch
):
    home = tmp_path / "profiles" / "work"
    _write_config(home, "true")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda conn: True)
    _seed_export_sections(None, "default", include_episodic_embedding=True)
    _seed_export_sections("work", "work", include_episodic_embedding=True)

    output = tmp_path / "profile.json"
    assert mnemosyne_command(_export_args(output)) == 0
    _assert_export_has_only_label(_read_export(output), "work", "default")


def test_export_without_selected_bank_keeps_legacy_default_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda conn: True)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    _seed_export_sections(None, "default", include_episodic_embedding=True)
    _seed_export_sections("work", "work", include_episodic_embedding=True)

    output = tmp_path / "default.json"
    assert mnemosyne_command(_export_args(output)) == 0
    _assert_export_has_only_label(_read_export(output), "default", "work")


def test_file_export_and_import_warn_about_partial_portable_data(tmp_path, monkeypatch, capsys):
    """The standalone Hermes CLI exposes core completeness results on both paths."""
    source_data = tmp_path / "source-data"
    target_data = tmp_path / "target-data"
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(source_data))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    source = Mnemosyne(session_id="hermes_default")
    source.remember("portable source")
    with sqlite3.connect(source.db_path) as conn:
        conn.execute(
            "INSERT INTO facts (fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("omitted-fact", "hermes_default", "portable", "has", "omitted fact", 1.0),
        )
        conn.execute("UPDATE working_memory SET pinned = 1, author_id = 'export-owner'")

    export_path = tmp_path / "partial.json"
    assert mnemosyne_command(_export_args(export_path)) == 0
    export_output = capsys.readouterr().out
    assert "WARNING: portable export is partial" in export_output
    assert "facts (1)" in export_output
    assert "working_memory missing" in export_output
    assert "author_id (1)" in export_output
    # pinned survives the portable round-trip since the event-date/pinned
    # export fix; the manifest no longer reports it as omitted.
    assert "pinned" not in export_output

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(target_data))
    assert mnemosyne_command(_file_import_args(export_path)) == 0
    import_output = capsys.readouterr().out
    assert "WARNING: imported supported data only" in import_output
    assert "facts (1)" in import_output
    assert "working_memory missing" in import_output
    assert "author_id (1)" in import_output
    # pinned now round-trips; the import manifest no longer reports it.
    assert "pinned" not in import_output
    # Round-trip proof at the data level: the exported pinned=1 row must
    # persist as pinned=1 in the imported DB (output text alone cannot
    # catch a manifest regression that silently drops the field).
    with sqlite3.connect(target_data / "mnemosyne.db") as conn:
        assert conn.execute(
            "SELECT pinned FROM working_memory"
        ).fetchone() == (1,)


def test_completeness_details_omits_invalid_partial_affected_row_counts():
    """Manifest counts are terminal-safe only when they are real row counts."""
    details = _completeness_details(
        {
            "partial_surfaces": [
                {
                    "section": "working_memory",
                    "omitted_fields": [
                        {"field": "author_id", "affected_rows": 3},
                        {"field": "negative", "affected_rows": -1},
                        {"field": "boolean", "affected_rows": True},
                        {"field": "malformed", "affected_rows": "3"},
                    ],
                }
            ]
        }
    )

    assert details == "working_memory missing author_id (3), negative, boolean, malformed"


def test_file_import_completeness_warnings_are_terminal_safe_and_handle_unknown(
    tmp_path, monkeypatch, capsys
):
    """Imported manifests cannot inject terminal text; legacy manifests remain explicit."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "source-data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    source = Mnemosyne(session_id="hermes_default")
    source.remember("portable source")
    export_path = tmp_path / "source.json"
    assert mnemosyne_command(_export_args(export_path)) == 0
    capsys.readouterr()

    payload = _read_export(export_path)
    payload["mnemosyne_export"]["completeness"] = {
        "complete": False,
        "omitted_surfaces": [{"table": "facts\x1b[31m", "row_count": "untrusted"}],
        "partial_surfaces": [
            {"section": "working\n_memory", "omitted_fields": [{"field": "author\t_id"}]}
        ],
    }
    unsafe_path = tmp_path / "unsafe.json"
    unsafe_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "partial-target"))
    assert mnemosyne_command(_file_import_args(unsafe_path)) == 0
    partial_output = capsys.readouterr().out
    assert "WARNING: imported supported data only" in partial_output
    assert "\x1b" not in partial_output
    assert "\n_memory" not in partial_output
    assert "\tauthor" not in partial_output

    payload["mnemosyne_export"].pop("completeness")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "legacy-target"))
    assert mnemosyne_command(_file_import_args(legacy_path)) == 0
    assert "NOTE: source export predates completeness reporting" in capsys.readouterr().out


def test_export_explicit_default_bank_keeps_legacy_default_behavior(tmp_path, monkeypatch):
    """Explicit default selects and preflights the populated legacy default DB."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda conn: True)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    _seed_export_sections(None, "default", include_episodic_embedding=True)
    _seed_export_sections("work", "work", include_episodic_embedding=True)

    output = tmp_path / "explicit-default.json"
    assert mnemosyne_command(_export_args(output, bank="default")) == 0
    _assert_export_has_only_label(_read_export(output), "default", "work")


@pytest.mark.parametrize(
    "selection,bank",
    [("explicit", "missing"), ("implicit", "work")],
)
@pytest.mark.parametrize("state", ["missing", "incomplete"])
def test_export_selected_missing_or_incomplete_bank_has_no_artifacts(
    tmp_path, monkeypatch, capsys, selection, bank, state
):
    """A selected named bank needs its directory and SQLite DB before export.

    ``get_bank_db_path_read_only`` defines an incomplete bank as a bank directory
    without ``mnemosyne.db``; it deliberately accepts any existing SQLite file
    and does not validate that file's schema.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output = tmp_path / "export.json"
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    if selection == "implicit":
        home = tmp_path / "profiles" / bank
        _write_config(home, "true")
        monkeypatch.setenv("HERMES_HOME", str(home))
        args = _export_args(output)
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)
        args = _export_args(output, bank=bank)
    if state == "incomplete":
        (data_dir / "banks" / bank).mkdir(parents=True)

    selected_path = data_dir / "banks" / bank
    before = sorted(path.relative_to(data_dir) for path in data_dir.rglob("*"))
    assert mnemosyne_command(args) == 1
    assert "Bank not found:" in capsys.readouterr().out
    assert not output.exists()
    assert not (selected_path / "mnemosyne.db").exists()
    assert sorted(path.relative_to(data_dir) for path in data_dir.rglob("*")) == before


def test_export_bank_validation_error_does_not_disclose_exception_details(
    tmp_path, monkeypatch, capsys
):
    """Unexpected preflight errors remain concise and do not expose DB paths."""
    calls = []

    def raise_validation_error(bank):
        calls.append(bank)
        raise RuntimeError(f"cannot inspect {tmp_path / 'private' / bank / 'mnemosyne.db'}")

    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        "mnemosyne.core.banks.get_bank_db_path_read_only", raise_validation_error
    )

    output_path = tmp_path / "export.json"
    assert mnemosyne_command(_export_args(output_path, bank="work")) == 1
    output = capsys.readouterr().out
    assert output == "Bank validation failed\n"
    assert str(tmp_path) not in output
    assert calls == ["work"]
    assert not output_path.exists()


def test_export_missing_selected_bank_fails_before_host_llm_registration(
    tmp_path, monkeypatch, capsys
):
    """Rejected named exports leave the host LLM registry untouched."""
    registrations = []
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "mnemosyne_hermes.hermes_llm_adapter.register_hermes_host_llm",
        lambda: registrations.append("called"),
    )

    output_path = tmp_path / "export.json"
    assert mnemosyne_command(_export_args(output_path, bank="work")) == 1
    assert "Bank not found: work" in capsys.readouterr().out
    assert registrations == []
    assert not output_path.exists()


def test_export_invalid_explicit_bank_does_not_fall_back_to_default_bank(
    tmp_path, monkeypatch, capsys
):
    """An invalid explicit export bank never selects shared/default data."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    output_path = tmp_path / "export.json"

    assert mnemosyne_command(_export_args(output_path, bank="***")) == 1
    assert capsys.readouterr().out == "Bank resolution failed\n"
    assert not output_path.exists()


def test_export_resolution_failure_does_not_fall_back_to_default_bank(
    tmp_path, monkeypatch, capsys
):
    """A resolver failure never exports the shared/default bank."""
    monkeypatch.setattr(
        "mnemosyne_hermes.cli._resolve_cli_bank", lambda *_args: _BANK_RESOLUTION_FAILED
    )
    output_path = tmp_path / "export.json"

    assert mnemosyne_command(_export_args(output_path, bank="work")) == 1
    assert capsys.readouterr().out == "Bank resolution failed\n"
    assert not output_path.exists()


def test_explicit_default_export_requires_existing_default_bank(tmp_path, monkeypatch, capsys):
    """An explicit default bank is preflighted instead of bypassing validation."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    output_path = tmp_path / "export.json"

    assert mnemosyne_command(_export_args(output_path, bank="default")) == 1
    assert capsys.readouterr().out == "Bank not found: default\n"
    assert not output_path.exists()


def test_non_export_resolution_failure_keeps_default_fallback(monkeypatch):
    """Ordinary CLI commands retain legacy fail-soft resolver behavior."""
    monkeypatch.setattr("mnemosyne_hermes.cli._get_provider_class", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert _resolve_cli_bank(_args(bank="work"), "stats") is None


@pytest.mark.parametrize("raise_on_execute", [False, True])
def test_export_schema_probe_closes_connection_on_all_paths(
    tmp_path, monkeypatch, raise_on_execute
):
    """The read-only preflight closes even on early return or query failure."""
    class Connection:
        closed = False

        def execute(self, query):
            if raise_on_execute:
                raise sqlite3.DatabaseError("query failed")
            return []

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(
        "mnemosyne_hermes.cli.sqlite3.connect", lambda *args, **kwargs: connection
    )

    if raise_on_execute:
        with pytest.raises(sqlite3.DatabaseError, match="query failed"):
            _export_schema_is_complete_read_only(tmp_path / "selected.db")
    else:
        assert not _export_schema_is_complete_read_only(tmp_path / "selected.db")
    assert connection.closed


@pytest.mark.parametrize("selection,bank", [("explicit", "work"), ("implicit", "profile")])
def test_export_selected_bank_with_incomplete_sqlite_schema_is_untouched(
    tmp_path, monkeypatch, capsys, selection, bank
):
    """A selected SQLite file without Mnemosyne's export schema fails closed."""
    data_dir = tmp_path / "data"
    selected_dir = data_dir / "banks" / bank
    selected_dir.mkdir(parents=True)
    db_path = selected_dir / "mnemosyne.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    before_bytes = db_path.read_bytes()
    before_paths = sorted(path.relative_to(data_dir) for path in data_dir.rglob("*"))

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    if selection == "implicit":
        home = tmp_path / "profiles" / bank
        _write_config(home, "true")
        monkeypatch.setenv("HERMES_HOME", str(home))
        args = _export_args(tmp_path / "export.json")
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)
        args = _export_args(tmp_path / "export.json", bank=bank)
    output = Path(args.output)

    assert mnemosyne_command(args) == 1
    assert f"Bank schema incomplete: {bank}" in capsys.readouterr().out
    assert not output.exists()
    assert db_path.read_bytes() == before_bytes
    assert sorted(path.relative_to(data_dir) for path in data_dir.rglob("*")) == before_paths


# Mirror only the columns selected by the current unconditional JSON export
# queries. This makes a contract change in an exporter fail this boundary test.
_EXPORTER_REQUIRED_COLUMNS = {
    "working_memory": frozenset({
        "id", "content", "source", "timestamp", "session_id", "importance",
        "metadata_json", "valid_until", "superseded_by", "scope", "recall_count",
        "last_recalled", "created_at", "veracity", "consolidated_at",
        "consolidation_claimed_at",
    }),
    "episodic_memory": frozenset({
        "id", "content", "source", "timestamp", "session_id", "importance",
        "metadata_json", "summary_of", "valid_until", "superseded_by", "scope",
        "recall_count", "last_recalled", "created_at",
    }),
    "scratchpad": frozenset({"id", "content", "session_id", "created_at", "updated_at"}),
    "consolidation_log": frozenset({
        "id", "session_id", "items_consolidated", "summary_preview", "created_at",
    }),
    "memories": frozenset({
        "id", "content", "source", "timestamp", "session_id", "importance",
        "metadata_json", "created_at",
    }),
    "memory_embeddings": frozenset({"memory_id", "embedding_json", "model", "created_at"}),
    "triples": frozenset({
        "id", "subject", "predicate", "object", "valid_from", "valid_until",
        "source", "confidence", "created_at",
    }),
    "annotations": frozenset({
        "id", "memory_id", "kind", "value", "source", "confidence", "created_at",
    }),
    "canonical_facts": frozenset({
        "id", "owner_id", "category", "name", "body", "source", "confidence",
        "version", "valid_from", "valid_until", "created_at",
    }),
}
_EXPORTER_TABLES = frozenset(_EXPORTER_REQUIRED_COLUMNS)
_EXPORTER_REQUIRED_COLUMN_CASES = tuple(
    (table, column)
    for table, columns in sorted(_EXPORTER_REQUIRED_COLUMNS.items())
    for column in sorted(columns)
)


def _initialized_selected_export(tmp_path, monkeypatch, selection, bank):
    """Create a complete selected bank and configure its CLI selection mode."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    _seed_export_sections(bank, "selected")
    db_path = data_dir / "banks" / bank / "mnemosyne.db"
    if selection == "implicit":
        home = tmp_path / "profiles" / bank
        _write_config(home, "true")
        monkeypatch.setenv("HERMES_HOME", str(home))
        args = _export_args(tmp_path / "export.json")
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)
        args = _export_args(tmp_path / "export.json", bank=bank)
    return args, data_dir, db_path


@pytest.mark.parametrize("selection,bank", [("explicit", "work"), ("implicit", "profile")])
@pytest.mark.parametrize("missing_table", sorted(_EXPORTER_TABLES))
def test_export_selected_bank_rejects_each_missing_exporter_table_without_mutation(
    tmp_path, monkeypatch, capsys, selection, bank, missing_table
):
    """Every table read unconditionally by the exporter is a probe dependency."""
    assert _EXPORT_REQUIRED_COLUMNS == _EXPORTER_REQUIRED_COLUMNS
    assert _EXPORT_REQUIRED_TABLES == _EXPORTER_TABLES
    args, data_dir, db_path = _initialized_selected_export(tmp_path, monkeypatch, selection, bank)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f'DROP TABLE "{missing_table}"')
    before_bytes = db_path.read_bytes()
    before_paths = sorted(path.relative_to(data_dir) for path in data_dir.rglob("*"))
    output = Path(args.output)

    assert mnemosyne_command(args) == 1
    assert f"Bank schema incomplete: {bank}" in capsys.readouterr().out
    assert not output.exists()
    assert db_path.read_bytes() == before_bytes
    assert sorted(path.relative_to(data_dir) for path in data_dir.rglob("*")) == before_paths


@pytest.mark.parametrize("selection,bank", [("explicit", "work"), ("implicit", "profile")])
@pytest.mark.parametrize("table,column", _EXPORTER_REQUIRED_COLUMN_CASES)
def test_export_selected_bank_rejects_each_missing_exporter_column_without_mutation(
    tmp_path, monkeypatch, capsys, selection, bank, table, column
):
    """Every unconditional exporter column fails selected-bank preflight closed."""
    assert _EXPORT_REQUIRED_COLUMNS == _EXPORTER_REQUIRED_COLUMNS
    args, data_dir, db_path = _initialized_selected_export(tmp_path, monkeypatch, selection, bank)
    renamed_column = f"missing_export_column_{column}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f'ALTER TABLE "{table}" RENAME COLUMN "{column}" TO "{renamed_column}"'
        )
    before_bytes = db_path.read_bytes()
    before_paths = sorted(path.relative_to(data_dir) for path in data_dir.rglob("*"))
    output = Path(args.output)

    assert mnemosyne_command(args) == 1
    assert f"Bank schema incomplete: {bank}" in capsys.readouterr().out
    assert not output.exists()
    assert db_path.read_bytes() == before_bytes
    assert sorted(path.relative_to(data_dir) for path in data_dir.rglob("*")) == before_paths


@pytest.mark.parametrize("selection,bank", [("explicit", "work"), ("implicit", "profile")])
def test_export_selected_bank_does_not_require_optional_sync_table(
    tmp_path, monkeypatch, selection, bank
):
    """Optional sync storage is absent from the read-only probe and export path."""
    args, data_dir, db_path = _initialized_selected_export(tmp_path, monkeypatch, selection, bank)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS memory_events")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "memory_events" not in tables
    assert _export_schema_is_complete_read_only(db_path)
    before_bytes = db_path.read_bytes()
    before_paths = sorted(path.relative_to(data_dir) for path in data_dir.rglob("*"))

    assert mnemosyne_command(args) == 0
    payload = _read_export(Path(args.output))
    assert "working-bank-marker-selected" in json.dumps(payload["working_memory"])
    assert db_path.read_bytes() == before_bytes
    assert sorted(path.relative_to(data_dir) for path in data_dir.rglob("*")) == before_paths
