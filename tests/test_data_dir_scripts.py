"""Regression tests for standalone scripts honoring MNEMOSYNE_DATA_DIR."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKFILL_SCRIPT = ROOT / "scripts" / "backfill_temporal_triples.py"
MIGRATE_SCRIPT = ROOT / "scripts" / "migrate_from_legacy.py"


def _isolated_env(tmp_path):
    home = tmp_path / "home"
    data_dir = tmp_path / "custom-data"
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    return env, home, data_dir


def _store_memory(env):
    result = subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", "store", "script data dir probe"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_backfill_temporal_triples_uses_mnemosyne_data_dir(tmp_path):
    env, home, data_dir = _isolated_env(tmp_path)
    _store_memory(env)

    result = subprocess.run(
        [sys.executable, str(BACKFILL_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Database: {data_dir / 'mnemosyne.db'}" in result.stdout
    assert "ERROR: Database not found" not in result.stdout
    assert (data_dir / "mnemosyne.db").exists()
    assert not (home / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db").exists()


def test_migrate_from_legacy_uses_mnemosyne_data_dir_as_canonical(tmp_path):
    env, home, data_dir = _isolated_env(tmp_path)

    result = subprocess.run(
        [sys.executable, str(MIGRATE_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Canonical DB: {data_dir / 'mnemosyne.db'}" in result.stdout
    assert (data_dir / "mnemosyne.db").exists()
    assert not (home / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db").exists()


def test_empty_mnemosyne_data_dir_falls_back_to_default_for_scripts(tmp_path):
    home = tmp_path / "home"
    default_db = home / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["MNEMOSYNE_DATA_DIR"] = ""
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    _store_memory(env)

    backfill = subprocess.run(
        [sys.executable, str(BACKFILL_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert backfill.returncode == 0, backfill.stdout + backfill.stderr
    assert f"Database: {default_db}" in backfill.stdout

    migrate = subprocess.run(
        [sys.executable, str(MIGRATE_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert migrate.returncode == 0, migrate.stdout + migrate.stderr
    assert f"Canonical DB: {default_db}" in migrate.stdout
    assert default_db.exists()
    assert not (ROOT / "mnemosyne.db").exists()
