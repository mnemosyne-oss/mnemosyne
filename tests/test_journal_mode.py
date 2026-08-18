"""Tests for MNEMOSYNE_JOURNAL_MODE selection and its application to stores.

WAL on Linux containers over macOS virtiofs intermittently corrupts reads
("database disk image is malformed"); deployments there set
MNEMOSYNE_JOURNAL_MODE=delete. The helper must honor valid values, fall back
to wal on unset/blank, and warn (not stay silent) on a typo: silently
reverting to WAL would restore exactly the failure the variable exists to
escape.
"""
from __future__ import annotations

import pytest

from mnemosyne.core import beam, journal
from mnemosyne.core.journal import journal_mode


def test_journal_mode_default_is_wal(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_JOURNAL_MODE", raising=False)
    assert journal_mode() == "wal"


@pytest.mark.parametrize("raw", ["delete", "DELETE", " Delete "], ids=["lower", "upper", "padded"])
def test_journal_mode_honors_valid_override(monkeypatch, raw):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", raw)
    assert journal_mode() == "delete"


@pytest.mark.parametrize("raw", ["", "   "], ids=["empty", "blank"])
def test_journal_mode_blank_is_unset(monkeypatch, raw):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", raw)
    assert journal_mode() == "wal"


def test_journal_mode_invalid_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", "deleet")
    with caplog.at_level("WARNING", logger="mnemosyne.core.journal"):
        assert journal_mode() == "wal"
    assert any("MNEMOSYNE_JOURNAL_MODE" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


def test_beam_connection_applies_configured_mode(tmp_path, monkeypatch):
    """The override must land on real store connections, not just the helper:
    beam's PRAGMA read-back returns the configured mode."""
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", "delete")
    db = tmp_path / "store.db"
    beam.init_beam(db)
    mode = beam._get_connection(db).execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "delete"


def test_invalid_override_leaves_connection_on_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_JOURNAL_MODE", "bogus")
    db = tmp_path / "store.db"
    beam.init_beam(db)
    mode = beam._get_connection(db).execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_journal_module_has_no_import_cycle():
    """journal.py must stay dependency-free: every core store module imports
    it at module scope, so any import there would cycle."""
    import inspect

    src = inspect.getsource(journal)
    assert "from mnemosyne" not in src and "import mnemosyne" not in src
