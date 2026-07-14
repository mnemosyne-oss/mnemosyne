"""Regression test for [Issue 2]: `mnemosyne bank list` must reflect
on-disk state, not a virtual 'default' entry.

The current `BankManager.list_banks()` in
`mnemosyne/core/banks.py:107-115` always inserts 'default' into
the list if it's not present, even when
`<DATA_DIR>/mnemosyne.db` does not exist on disk. This is a
'phantom bank' that misleads operators.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args, tmp_path, extra_env):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    env.pop("MNEMOSYNE_DATA_DIR", None)
    env.pop("HERMES_HOME", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_bank_list_no_phantom_default_when_file_missing(tmp_path):
    """bank list must not report 'default' if <DATA_DIR>/mnemosyne.db does not exist."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Ensure the default bank file does NOT exist
    default_db = data_dir / "mnemosyne.db"
    if default_db.exists():
        default_db.unlink()

    result = _run_cli(["bank", "list"], tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)})
    assert result.returncode == 0, result.stderr

    # The fix: bank list must not include 'default' when the file is absent.
    # After the fix, the list should be empty (no banks on disk).
    assert "  - default" not in result.stdout, (
        f"bank list reported a phantom 'default' bank; the file "
        f"{default_db} does not exist. stdout:\n{result.stdout}"
    )


def test_bank_list_includes_default_when_file_exists(tmp_path):
    """bank list must include 'default' when <DATA_DIR>/mnemosyne.db exists on disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    default_db = data_dir / "mnemosyne.db"
    # Create the default bank file (empty SQLite is enough)
    sqlite3.connect(str(default_db)).close()

    result = _run_cli(["bank", "list"], tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)})
    assert result.returncode == 0, result.stderr
    assert "  - default" in result.stdout, (
        f"bank list did not report 'default' but {default_db} exists; "
        f"stdout:\n{result.stdout}"
    )


def test_bank_list_reports_real_banks(tmp_path):
    """bank list must include any non-default banks that exist on disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Create two non-default banks on disk
    for bank_name in ("work", "personal"):
        bank_dir = data_dir / "banks" / bank_name
        bank_dir.mkdir(parents=True)
        sqlite3.connect(str(bank_dir / "mnemosyne.db")).close()

    result = _run_cli(["bank", "list"], tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)})
    assert result.returncode == 0, result.stderr
    assert "  - work" in result.stdout
    assert "  - personal" in result.stdout
