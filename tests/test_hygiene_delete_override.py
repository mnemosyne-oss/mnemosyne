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


@pytest.mark.parametrize('force', [None, 0, []])
@pytest.mark.parametrize('action', ['delete', 'archive', 'flag'])
def test_force_type_is_checked_without_mixed_delete_guard(mixed, force, action):
    path, candidates = mixed
    for candidate in candidates:
        candidate.suggested_action = 'delete'
    result = clean_noise(path, candidates, action=action, confirm=True, dry_run=False, force=force)
    assert result.errors and 'boolean' in result.errors[0]
    assert ids(path) == ['note', 'trash']


def test_false_force_is_valid_for_all_delete_recommendations(mixed):
    path, candidates = mixed
    for candidate in candidates:
        candidate.suggested_action = 'delete'
    result = clean_noise(path, candidates, action='delete', force=False, confirm=True, dry_run=False)
    assert result.errors == []
    assert result.deleted == 2
    assert ids(path) == []


@pytest.mark.parametrize(
    'case,force_args,confirm,expected_error,expected_status,deleted,remaining',
    [
        ('refused', {}, True, True, None, None, ['note', 'trash']),
        ('cleanup_failed', {}, True, True, 'failed', 0, ['note', 'trash']),
        ('dry_run', {}, False, False, 'dry_run', 1, ['note', 'trash']),
        ('applied', {}, True, False, 'applied', 1, ['note']),
        ('forced', {'force': True}, True, False, 'applied', 2, []),
        ('forced', {'force': True}, False, False, 'dry_run', 2, ['note', 'trash']),
        ('forced', {'force': 1}, True, True, None, None, ['note', 'trash']),
        ('forced', {'force': 1}, False, True, None, None, ['note', 'trash']),
        ('forced', {'force': 'true'}, True, True, None, None, ['note', 'trash']),
        ('forced', {'force': 'true'}, False, True, None, None, ['note', 'trash']),
    ],
    ids=['refused', 'cleanup-failed', 'per-candidate-preview', 'per-candidate-applied',
         'force-true-confirmed', 'force-true-preview', 'force-int-confirmed',
         'force-int-preview', 'force-string-confirmed', 'force-string-preview'],
)
def test_mcp_wire_reports_hygiene_failure_and_success(
    mixed, monkeypatch, case, force_args, confirm, expected_error,
    expected_status, deleted, remaining,
):
    """Returned handler failures must set the real MCP CallToolResult flag."""
    import asyncio
    import contextlib
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams
    from mnemosyne.mcp_server import _build_mcp_server

    path, candidates = mixed
    monkeypatch.setattr('mnemosyne.mcp_tools._create_instance', lambda **kw: SimpleNamespace(beam=SimpleNamespace(db_path=path)))
    if case == 'cleanup_failed':
        candidates = candidates[:1]
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TRIGGER prevent_cleanup BEFORE DELETE ON working_memory BEGIN SELECT RAISE(ABORT, 'cleanup blocked'); END")
    arguments = {'candidates_json': json.dumps([c.to_dict() for c in candidates]),
                 'action': 'delete' if case in ('refused', 'cleanup_failed', 'forced') else 'keep',
                 'confirm': confirm, **force_args}

    async def exercise():
        server = _build_mcp_server()
        async with create_client_server_memory_streams() as (cs, ss):
            task = asyncio.create_task(server.run(*ss, server.create_initialization_options()))
            try:
                async with ClientSession(*cs) as client:
                    await client.initialize()
                    response = await client.call_tool('mnemosyne_hygiene_clean', arguments)
                    payload = json.loads(response.content[0].text)
                    assert response.is_error is expected_error
                    if expected_status is None:
                        reason = '--force' if case == 'refused' else 'boolean'
                        assert reason in payload['error']
                    else:
                        assert payload['status'] == expected_status
                        assert bool(payload['result']['errors']) is expected_error
                        assert payload['result']['deleted'] == deleted
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    asyncio.run(exercise())
    assert ids(path) == remaining
