"""Reject unintended all-candidate deletion before any mutation (#1074)."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from mnemosyne.cli import cmd_hygiene
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.hygiene import NoiseCandidate, clean_noise
from mnemosyne.mcp_tools import _handle_hygiene_clean


@pytest.fixture
def mixed(tmp_path):
    path = tmp_path / 'mnemosyne.db'
    beam = BeamMemory(session_id='test', db_path=path)
    for mid in ['trash', 'note']:
        beam.conn.execute('INSERT INTO working_memory (id, content, session_id) VALUES (?, ?, ?)', (mid, mid, 'test'))
    beam.conn.commit()
    candidates = [NoiseCandidate('trash', 'working_memory', '', .9, suggested_action='delete'),
                  NoiseCandidate('note', 'working_memory', '', .1, suggested_action='keep')]
    yield path, candidates
    beam.conn.close()


def ids(path):
    with sqlite3.connect(path) as conn:
        return [r[0] for r in conn.execute('SELECT id FROM working_memory ORDER BY id')]


@pytest.mark.parametrize('dry_run', [True, False])
def test_mixed_delete_refused_before_any_write(mixed, dry_run):
    path, candidates = mixed
    before = path.read_bytes()
    result = clean_noise(path, candidates, action='delete', confirm=True, dry_run=dry_run)
    assert result.errors
    assert result.deleted == result.log_entries == 0
    assert ids(path) == ['note', 'trash']
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='hygiene_audit_log'").fetchone() is None


@pytest.mark.parametrize('force', [False, 1, 'true', [], None])
def test_core_force_requires_literal_true(mixed, force):
    path, candidates = mixed
    result = clean_noise(path, candidates, action='delete', confirm=True, dry_run=False, force=force)
    assert result.errors
    assert ids(path) == ['note', 'trash']


def test_literal_force_allows_override_but_does_not_replace_confirm(mixed):
    path, candidates = mixed
    denied = clean_noise(path, candidates, action='delete', force=True, confirm=False, dry_run=False)
    assert denied.errors
    assert ids(path) == ['note', 'trash']
    applied = clean_noise(path, candidates, action='delete', force=True, confirm=True, dry_run=False)
    assert applied.errors == []
    assert applied.deleted == 2
    assert ids(path) == []


def test_per_candidate_mode_still_preserves_note(mixed):
    path, candidates = mixed
    result = clean_noise(path, candidates, action='keep', confirm=True, dry_run=False)
    assert result.errors == []
    assert result.deleted == 1
    assert ids(path) == ['note']


def test_cli_refusal_explains_force_and_force_applies(mixed, monkeypatch, capsys):
    path, candidates = mixed
    monkeypatch.setattr('mnemosyne.cli.DATA_DIR', path.parent)
    report = path.parent / 'audit.json'
    report.write_text(json.dumps([c.to_dict() for c in candidates]))
    with pytest.raises(SystemExit) as exc:
        cmd_hygiene(['clean', str(report), '--action', 'delete', '--confirm'])
    assert exc.value.code != 0
    assert '--force' in capsys.readouterr().err
    assert ids(path) == ['note', 'trash']
    cmd_hygiene(['clean', str(report), '--action', 'delete', '--force', '--confirm'])
    assert 'deleted=2' in capsys.readouterr().out
    assert ids(path) == []


@pytest.mark.parametrize('force', [False, 1, 'true', None, True])
def test_mcp_force_contract(mixed, monkeypatch, force):
    path, candidates = mixed
    monkeypatch.setattr('mnemosyne.mcp_tools._create_instance', lambda **kw: SimpleNamespace(beam=SimpleNamespace(db_path=path)))
    result = _handle_hygiene_clean({'candidates_json': [c.to_dict() for c in candidates],
                                  'action': 'delete', 'confirm': True, 'force': force})
    if force is True:
        assert result['result']['deleted'] == 2
        assert ids(path) == []
    else:
        assert result.get('error') or result['result']['errors']
        assert result.get('status') != 'applied'
        assert ids(path) == ['note', 'trash']


def test_mcp_refusal_does_not_open_bank(mixed, monkeypatch):
    _path, candidates = mixed
    def forbidden(**kwargs):
        pytest.fail('refused override must not initialize a bank')
    monkeypatch.setattr('mnemosyne.mcp_tools._create_instance', forbidden)
    result = _handle_hygiene_clean({'candidates_json': [c.to_dict() for c in candidates], 'action': 'delete'})
    assert result['error']
