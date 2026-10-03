"""``mnemosyne backup`` and ``mnemosyne restore`` follow MNEMOSYNE_BANK (#1034).

Both commands used to resolve the database through
``mnemosyne.dr.recovery.get_default_paths()``, which only knows the default
bank. With ``MNEMOSYNE_BANK=work`` the backup held the default bank's data and
the restore overwrote the default bank, while ``work`` stayed untouched and
both commands reported success.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from mnemosyne import cli
from mnemosyne.dr import recovery


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """A default and a ``work`` bank with distinct markers, all under tmp_path."""
    data_dir = tmp_path / "data"
    backup_root = tmp_path / "backups"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("MNEMOSYNE_BANK", raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_BACKUP_DIR", str(backup_root))
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))
    default_db = _make_store(data_dir / "mnemosyne.db", "default")
    work_db = _make_store(data_dir / "banks" / "work" / "mnemosyne.db", "work")
    return backup_root, default_db, work_db


def _make_store(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE marker (label TEXT)")
    conn.execute("INSERT INTO marker VALUES (?)", (label,))
    conn.commit()
    conn.close()
    return path


def _labels(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return [row[0] for row in conn.execute("SELECT label FROM marker")]
    finally:
        conn.close()


def _dump(path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()


def _store_dir(backup_root: Path, db: Path) -> Path:
    digest = hashlib.sha256(os.fsencode(str(db.resolve()))).hexdigest()[:32]
    return backup_root / "stores" / f"{db.stem}-{digest}"


def test_backup_snapshots_the_selected_bank(stores, monkeypatch, capsys):
    backup_root, default_db, work_db = stores
    monkeypatch.setenv("MNEMOSYNE_BANK", "work")

    cli.cmd_backup([])

    backups = sorted(_store_dir(backup_root, work_db).glob("mnemosyne_backup_*.db.gz"))
    assert len(backups) == 1
    assert f"Backup created: {backups[0]}" in capsys.readouterr().out
    dump = _dump(backups[0])
    assert "'work'" in dump
    assert "'default'" not in dump
    meta = json.loads(backups[0].with_suffix(".gz.json").read_text())
    assert meta["source_db"] == str(work_db.resolve())
    assert recovery.list_backups() == []


def test_backup_to_output_dir_snapshots_the_selected_bank(stores, monkeypatch, tmp_path):
    _, _, work_db = stores
    monkeypatch.setenv("MNEMOSYNE_BANK", "work")
    out = tmp_path / "out"

    cli.cmd_backup([str(out)])

    backups = sorted(out.glob("mnemosyne_backup_*.db.gz"))
    assert len(backups) == 1
    assert "'work'" in _dump(backups[0])
    assert "'default'" not in _dump(backups[0])


def test_backup_without_bank_still_snapshots_the_default_bank(stores, tmp_path):
    _, default_db, _ = stores
    out = tmp_path / "out"

    cli.cmd_backup([str(out)])

    backups = sorted(out.glob("mnemosyne_backup_*.db.gz"))
    assert len(backups) == 1
    assert "'default'" in _dump(backups[0])
    assert "'work'" not in _dump(backups[0])
    meta = json.loads(backups[0].with_suffix(".gz.json").read_text())
    assert meta["source_db"] == str(default_db.resolve())


def test_restore_writes_only_the_selected_bank(stores, monkeypatch, tmp_path, capsys):
    _, default_db, work_db = stores
    source = _make_store(tmp_path / "snapshot" / "source.db", "restored")
    backup = recovery.create_backup(db_path=source, backup_dir=tmp_path / "out")
    default_bytes = default_db.read_bytes()
    monkeypatch.setenv("MNEMOSYNE_BANK", "work")

    cli.cmd_restore([backup["backup_path"]])

    assert f"Database:     {work_db}" in capsys.readouterr().out
    assert _labels(work_db) == ["restored"]
    assert _labels(default_db) == ["default"]
    assert default_db.read_bytes() == default_bytes


@pytest.mark.parametrize("command", ["backup", "restore"])
def test_missing_bank_fails_before_writing(stores, monkeypatch, tmp_path, capsys, command):
    backup_root, default_db, work_db = stores
    source = _make_store(tmp_path / "snapshot" / "source.db", "restored")
    backup = recovery.create_backup(db_path=source, backup_dir=tmp_path / "out")
    default_bytes = default_db.read_bytes()
    work_bytes = work_db.read_bytes()
    monkeypatch.setenv("MNEMOSYNE_BANK", "nope")

    with pytest.raises(SystemExit) as exc:
        if command == "backup":
            cli.cmd_backup([])
        else:
            cli.cmd_restore([backup["backup_path"]])

    assert exc.value.code == 2
    assert "Bank 'nope' does not exist" in capsys.readouterr().err
    assert not (default_db.parent / "banks" / "nope").exists()
    assert not backup_root.exists()
    assert default_db.read_bytes() == default_bytes
    assert work_db.read_bytes() == work_bytes
