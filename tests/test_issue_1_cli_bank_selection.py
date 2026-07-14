"""Regression test for [Issue 1]: `mnemosyne store` must honor
`MNEMOSYNE_BANK` and write to the bank-named SQLite file, not the
default bank at `<DATA_DIR>/mnemosyne.db`.

The CLI currently always writes to the default bank regardless of
the env var, which contradicts the platform contract documented in
`docker/mnemosyne/Dockerfile`, `compose.tenant.yml`,
`docs/architecture.md`, and `scripts/mnemosyne-sleep.sh`.
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


def test_cli_store_honors_MNEMOSYNE_BANK(tmp_path):
    """store must write to the bank named by $MNEMOSYNE_BANK, not default."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bank_name = "ora"
    bank_dir = data_dir / "banks" / bank_name
    bank_db = bank_dir / "mnemosyne.db"
    assert not bank_dir.exists()

    result = _run_cli(
        ["store", "phase-1-cli-bank-test", "phase-1-init"],
        tmp_path,
        {
            "MNEMOSYNE_DATA_DIR": str(data_dir),
            "MNEMOSYNE_BANK": bank_name,
        },
    )
    assert result.returncode == 0, result.stderr

    # The bank DB must contain the stored row.
    assert bank_db.exists(), f"bank DB not at {bank_db}"
    con = sqlite3.connect(str(bank_db))
    rows = con.execute("SELECT count(*) FROM working_memory").fetchone()[0]
    con.close()
    assert rows >= 1, (
        f"bank DB at {bank_db} has 0 working_memory rows; "
        f"CLI wrote to default instead of MNEMOSYNE_BANK={bank_name}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    # The default bank must NOT contain this row.
    default_db = data_dir / "mnemosyne.db"
    if default_db.exists():
        con = sqlite3.connect(str(default_db))
        rows = con.execute("SELECT count(*) FROM working_memory").fetchone()[0]
        con.close()
        assert rows == 0, (
            f"default bank has {rows} rows; CLI wrote to default "
            f"instead of MNEMOSYNE_BANK={bank_name}"
        )


def test_cli_store_default_when_MNEMOSYNE_BANK_unset(tmp_path):
    """When MNEMOSYNE_BANK is unset, store must use the default bank.

    This is the existing default behavior (no regression). It also
    serves as a control for the MNEMOSYNE_BANK test above.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    default_db = data_dir / "mnemosyne.db"

    result = _run_cli(
        ["store", "phase-1-cli-default-test", "phase-1-init"],
        tmp_path,
        {"MNEMOSYNE_DATA_DIR": str(data_dir)},
    )
    assert result.returncode == 0, result.stderr

    assert default_db.exists()
    con = sqlite3.connect(str(default_db))
    rows = con.execute("SELECT count(*) FROM working_memory").fetchone()[0]
    con.close()
    assert rows >= 1
