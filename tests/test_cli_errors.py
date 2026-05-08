"""CLI error handling regression tests."""

import os
import subprocess
import sys


COMMANDS = [
    (
        ["store", "hello", "cli", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["recall", "hello", "not-an-int"],
        "top_k must be an integer",
    ),
    (
        ["update", "missing-id", "new content", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["import", "missing-file.json"],
        "Import file not found",
    ),
]


def run_cli(args, tmp_path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_invalid_cli_input_reports_error_without_traceback(tmp_path):
    for args, expected_error in COMMANDS:
        result = run_cli(args, tmp_path)

        assert result.returncode != 0, args
        assert expected_error in result.stderr, result.stderr
        assert "Traceback" not in result.stderr


def test_import_malformed_json_reports_error_without_traceback(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")

    result = run_cli(["import", str(bad_json)], tmp_path)

    assert result.returncode != 0
    assert "Invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr
