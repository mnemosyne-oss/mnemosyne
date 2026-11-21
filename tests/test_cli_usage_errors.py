"""CLI usage error regression tests."""

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing

import pytest


USAGE_COMMANDS = [
    (["store"], "Usage: mnemosyne store <content> [source] [importance]"),
    (["recall"], "Usage: mnemosyne recall <query> [top_k]"),
    (["update", "missing-id"], "Usage: mnemosyne update <memory_id> <new_content> [importance]"),
    (["delete"], "Usage: mnemosyne delete <memory_id>"),
    (["import"], "Usage: mnemosyne import <file.json>"),
    (["import-hindsight"], "Usage: mnemosyne import-hindsight <file.json|base_url> [bank]"),
    (["bank"], "Usage: mnemosyne bank <list|create|delete> [name]"),
]


def run_cli(args, tmp_path):
    return run_cli_in(args, tmp_path, tmp_path / "mnemosyne-data")


def run_cli_in(args, tmp_path, data_dir):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_missing_required_args_report_usage_error_without_traceback(tmp_path):
    for args, expected_usage in USAGE_COMMANDS:
        result = run_cli(args, tmp_path)

        assert result.returncode != 0, args
        assert result.stdout == ""
        assert expected_usage in result.stderr
        assert "Traceback" not in result.stderr


def test_unknown_command_reports_error_without_traceback(tmp_path):
    result = run_cli(["definitely-not-a-command"], tmp_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "Unknown command: definitely-not-a-command" in result.stderr
    assert "Run 'mnemosyne --help' for usage." in result.stderr
    assert "Traceback" not in result.stderr


def test_help_exits_successfully(tmp_path):
    result = run_cli(["--help"], tmp_path)

    assert result.returncode == 0
    assert "Usage: mnemosyne <command> [args]" in result.stdout
    assert "Traceback" not in result.stderr


def test_recall_explain_json_outputs_parseable_payload(tmp_path):
    store = run_cli(["store", "Alice prefers Vim", "cli", "0.8"], tmp_path)
    assert store.returncode == 0, store.stderr

    result = run_cli(["recall", "Alice", "--explain", "--json"], tmp_path)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query"] == "Alice"
    assert payload["top_k"] == 5
    assert isinstance(payload["results"], list)
    assert "explain" in payload


def test_reindex_rejects_unknown_option_before_opening_store(tmp_path):
    result = run_cli(["reindex", "--bogus"], tmp_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "Unknown reindex option: --bogus" in result.stderr
    assert "Traceback" not in result.stderr


def test_reindex_dry_run_honors_db_override(tmp_path):
    default_dir = tmp_path / "default-data"
    custom_dir = tmp_path / "custom-data"

    default_first = run_cli_in(["store", "Default bank memory one", "cli", "0.5"], tmp_path, default_dir)
    assert default_first.returncode == 0, default_first.stderr
    default_second = run_cli_in(["store", "Default bank memory two", "cli", "0.5"], tmp_path, default_dir)
    assert default_second.returncode == 0, default_second.stderr

    custom_store = run_cli_in(["store", "Custom bank memory", "cli", "0.5"], tmp_path, custom_dir)
    assert custom_store.returncode == 0, custom_store.stderr

    custom_db = custom_dir / "mnemosyne.db"
    assert custom_db.is_file()

    result = run_cli_in(["reindex", "--dry-run", "--db", str(custom_db)], tmp_path, default_dir)

    assert result.returncode == 0, result.stderr
    assert str(custom_db) in result.stdout
    assert "working_memory: 1" in result.stdout


def _tree(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


REINDEX_MODES = [["--dry-run"], ["--yes"]]


@pytest.mark.parametrize("mode", REINDEX_MODES, ids=["dry-run", "real-run"])
def test_reindex_rejects_missing_db_without_creating_it(tmp_path, mode):
    missing_db = tmp_path / "typo" / "mnemosyne.db"

    result = run_cli(["reindex", *mode, "--db", str(missing_db)], tmp_path)

    assert result.returncode != 0
    assert f"Database not found: {missing_db}" in result.stderr
    assert "Traceback" not in result.stderr
    assert _tree(tmp_path) == []


@pytest.mark.parametrize("mode", REINDEX_MODES, ids=["dry-run", "real-run"])
def test_reindex_rejects_missing_bank_without_creating_it(tmp_path, mode):
    seeded = run_cli(["store", "Default bank memory", "cli", "0.5"], tmp_path)
    assert seeded.returncode == 0, seeded.stderr
    before = _tree(tmp_path)

    result = run_cli(["reindex", *mode, "--bank", "typo"], tmp_path)

    assert result.returncode != 0
    assert "Bank 'typo' does not exist" in result.stderr
    assert "Traceback" not in result.stderr
    assert _tree(tmp_path) == before


def test_reindex_db_dry_run_leaves_default_store_untouched(tmp_path):
    default_dir = tmp_path / "default-data"
    custom_dir = tmp_path / "custom-data"
    custom_store = run_cli_in(["store", "Custom bank memory", "cli", "0.5"], tmp_path, custom_dir)
    assert custom_store.returncode == 0, custom_store.stderr
    custom_db = custom_dir / "mnemosyne.db"

    result = run_cli_in(["reindex", "--dry-run", "--db", str(custom_db)], tmp_path, default_dir)

    assert result.returncode == 0, result.stderr
    assert "working_memory: 1" in result.stdout
    assert not (default_dir / "mnemosyne.db").exists()
    assert not (tmp_path / "home" / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db").exists()


@pytest.mark.parametrize("selection", ["ambient", "db", "bank"])
def test_reindex_dry_run_does_not_initialize_or_backfill_target(tmp_path, monkeypatch, selection):
    monkeypatch.delenv("MNEMOSYNE_BANK", raising=False)
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "int8")
    data_dir = tmp_path / "data #%"
    target_dir = data_dir / "banks" / "test" if selection == "bank" else data_dir
    target = target_dir / "mnemosyne.db"
    seeded = run_cli_in(
        ["store", "A disposable memory backed by JSON", "cli", "0.5"], tmp_path, target_dir
    )
    assert seeded.returncode == 0, seeded.stderr

    with closing(sqlite3.connect(target)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("DELETE FROM memory_embeddings")
        row = conn.execute("SELECT id FROM working_memory").fetchone()
        vec_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_episodes'"
        ).fetchone()
        conn.execute(
            "INSERT INTO memory_embeddings(memory_id, embedding_json, model) VALUES (?, ?, ?)",
            (row[0], json.dumps([1.0] + [0.0] * 383), "fixture"),
        )
        conn.commit()
    if selection == "ambient":
        (data_dir / "banks").rmdir()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    tree_before = _tree(tmp_path)
    args = {
        "ambient": [],
        "db": ["--db", str(target)],
        "bank": ["--bank", "test"],
    }[selection]

    monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "float32")
    result = run_cli_in(["reindex", "--dry-run", *args], tmp_path, data_dir)

    assert result.returncode == 0, result.stderr
    assert str(target) in result.stdout
    assert "working_memory: 1" in result.stdout
    if vec_schema is not None:
        assert "sqlite_vec: True" in result.stdout
        assert "vec_type: int8" in result.stdout
    assert _tree(tmp_path) == tree_before
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("journal_mode, suffix", [("WAL", "-wal"), ("DELETE", "-journal")])
def test_reindex_dry_run_refuses_pending_journal_without_writes(tmp_path, journal_mode, suffix):
    target = tmp_path / "pending.db"
    with closing(sqlite3.connect(target)) as conn:
        conn.execute(f"PRAGMA journal_mode={journal_mode}")
        conn.execute("CREATE TABLE working_memory(content TEXT)")
        conn.execute("CREATE TABLE episodic_memory(content TEXT)")
        conn.execute("INSERT INTO working_memory VALUES ('initial')")
        conn.commit()
        if journal_mode == "WAL":
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("UPDATE working_memory SET content = 'new text'")
        if journal_mode == "WAL":
            conn.commit()  # A committed row exists only in the WAL.
        tree_before = _tree(tmp_path)
        before = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
        assert target.with_name(target.name + suffix).stat().st_size > 0

        result = run_cli(["reindex", "--dry-run", "--db", str(target)], tmp_path)

        assert result.returncode != 0
        assert "checkpoint" in result.stderr.lower()
        assert "Traceback" not in result.stderr
        assert _tree(tmp_path) == tree_before
        assert {path: path.read_bytes() for path in before} == before


def test_reindex_dry_run_counts_existing_rows_in_legacy_store(tmp_path):
    target = tmp_path / "legacy.db"
    with closing(sqlite3.connect(target)) as conn:
        conn.execute("CREATE TABLE working_memory(content TEXT)")
        conn.execute("INSERT INTO working_memory VALUES ('legacy memory')")
        conn.commit()
    before = target.read_bytes()

    result = run_cli(["reindex", "--dry-run", "--db", str(target)], tmp_path)

    assert result.returncode == 0, result.stderr
    assert "working_memory: 1" in result.stdout
    assert "episodic_memory: 0" in result.stdout
    assert "sqlite_vec: False" in result.stdout
    assert _tree(tmp_path) == ["legacy.db"]
    assert target.read_bytes() == before


def test_help_lists_reindex_target_options(tmp_path):
    result = run_cli(["--help"], tmp_path)

    assert result.returncode == 0
    reindex_lines = [line for line in result.stdout.splitlines() if line.strip().startswith("reindex ")]
    assert len(reindex_lines) == 1
    assert "--db PATH|--bank NAME" in reindex_lines[0]
