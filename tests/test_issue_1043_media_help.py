"""Regression tests for issue #1043: `mnemosyne media --help` must print usage, not ingest."""

import os
import subprocess
import sys


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


def test_media_help_prints_usage_without_ingesting(tmp_path):
    """`mnemosyne media --help` should print usage and exit 0, not start ingestion."""

    result = run_cli(["media", "--help"], tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout
    assert "Usage: mnemosyne media <path|url|data:uri>" in result.stdout
    assert "--modality" in result.stdout
    assert "--json" in result.stdout
    # The old bug path printed "Asset: <uuid>" — assert it does not.
    assert "Asset:" not in result.stdout


def test_media_short_help_flag_also_prints_usage(tmp_path):
    """`-h` should behave the same as `--help`."""

    result = run_cli(["media", "-h"], tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Usage: mnemosyne media <path|url|data:uri>" in result.stdout
    assert "Asset:" not in result.stdout


def test_media_help_creates_no_database_or_asset(tmp_path):
    """`mnemosyne media --help` must be side-effect-free on disk."""

    data_dir = tmp_path / "mnemosyne-data"
    data_dir.mkdir(parents=True, exist_ok=True)

    result = run_cli(["media", "--help"], tmp_path)

    assert result.returncode == 0
    # No database file created, no asset row written.
    assert not (data_dir / "mnemosyne.db").exists()
    # Sub-directories specific to media (assets dir) should not be created.
    assert not (data_dir / "media_assets").exists()
