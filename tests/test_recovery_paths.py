"""Tests for recovery.get_default_paths() honoring the same path configuration
as the live store (mnemosyne.core.beam).

The disaster-recovery helpers (backup/restore, and `mnemosyne reindex`'s
auto-backup) must resolve the database to the same location the store actually
uses. Previously they hardcoded ``~/.mnemosyne/data`` and ignored
MNEMOSYNE_DATA_DIR / HERMES_HOME, so they operated on (or failed to find) the
wrong database.
"""
from __future__ import annotations

from pathlib import Path

from mnemosyne.dr import recovery


def test_get_default_paths_honors_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MNEMOSYNE_BACKUP_DIR", raising=False)
    data_dir, backup_dir, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "data"
    assert db_path == tmp_path / "data" / "mnemosyne.db"
    # backups land alongside the data dir, not under ~/.mnemosyne
    assert backup_dir == tmp_path / "backups"


def test_get_default_paths_honors_hermes_home(monkeypatch, tmp_path):
    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)
    monkeypatch.delenv("MNEMOSYNE_BACKUP_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    data_dir, backup_dir, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "home" / "mnemosyne" / "data"
    assert db_path == data_dir / "mnemosyne.db"
    assert backup_dir == tmp_path / "home" / "mnemosyne" / "backups"


def test_get_default_paths_backup_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_BACKUP_DIR", str(tmp_path / "custom_backups"))
    _, backup_dir, _ = recovery.get_default_paths()
    assert backup_dir == tmp_path / "custom_backups"


def test_get_default_paths_data_dir_takes_precedence_over_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "explicit"))
    data_dir, _, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "explicit"
    assert db_path == tmp_path / "explicit" / "mnemosyne.db"
