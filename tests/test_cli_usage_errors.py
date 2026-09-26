"""CLI usage error regression tests."""

import json
import os
import subprocess
import sys


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
