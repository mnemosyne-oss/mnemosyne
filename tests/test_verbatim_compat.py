"""Optional-core compatibility and explicit SQLite ownership boundaries."""
import builtins
import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
import mnemosyne.core.polyphonic_recall as poly
from mnemosyne.core import verbatim_ledger as vl
from tests import test_verbatim_ledger as ledger_tests
from tests.test_verbatim_ledger import armed, capture
from tests.test_verbatim_provider_contract import load_provider

ROOT = Path(__file__).resolve().parents[1]
store = ledger_tests.store


def test_provider_compatibility_bundles_cannot_drift():
    assert (ROOT / 'hermes_memory_provider/_verbatim_compat.py').read_bytes() == (
        ROOT / 'integrations/hermes/src/mnemosyne_hermes/_verbatim_compat.py'
    ).read_bytes()


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('key', [None, 123, [], {}, '', 'x' * (vl.MAX_SESSION_KEY_CHARS + 1)])
def test_invalid_keys_abstain_at_both_entry_points(key, enabled):
    ledger = vl.VerbatimLedger(enabled)
    assert ledger.begin(key, []) is None
    assert ledger.snapshot_for(key) is None
    assert not ledger._sessions


class WithoutLimitAPI:
    """Real SQL connection with Python 3.10's missing limit-query capability."""
    def __init__(self, conn):
        self.conn = conn

    def execute(self, *args, **kwargs):
        return self.conn.execute(*args, **kwargs)


def test_no_limit_api_preserves_positive_exclusions_and_budget(store):
    ledger = armed()
    mid = capture(ledger, store, 'fresh botanical record')
    snapshot = ledger.snapshot_for('session')
    conn = WithoutLimitAPI(store.conn)
    assert vl.resolve_exclusions(conn, snapshot) == {mid}
    proof = snapshot.captures[0]
    # Historical default budget, including reserved slots, still bounds work.
    oversized = vl.ExclusionSnapshot(snapshot.generation, (proof,) * 167)
    assert vl.resolve_exclusions(conn, oversized) == set()
    snapshot.generation.valid = False
    assert vl.resolve_exclusions(conn, snapshot) == set()


def test_no_limit_api_sql_error_abstains(store):
    ledger = armed()
    capture(ledger, store, 'another fresh botanical record')
    class UnavailableSQL(WithoutLimitAPI):
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError('too many SQL variables')
    assert vl.resolve_exclusions(UnavailableSQL(store.conn), ledger.snapshot_for('session')) == set()


@pytest.mark.parametrize('name', ['root', 'packaged'])
@pytest.mark.parametrize('failure_kind', ['module', 'symbol'])
def test_missing_optional_core_preserves_initialize_capture_and_prefetch(name, failure_kind, tmp_path, monkeypatch):
    # Core Beam is already imported: simulate only the optional ledger capability
    # being absent, not a fully installed historic dependency matrix.
    original_import = builtins.__import__
    def old_core_import(module, *args, **kwargs):
        if module == 'mnemosyne.core.verbatim_ledger':
            if failure_kind == 'module':
                raise ModuleNotFoundError('old core has no optional ledger', name=module)
            raise ImportError('old core has no VerbatimLedger symbol')
        return original_import(module, *args, **kwargs)
    monkeypatch.setenv('MNEMOSYNE_SELF_ECHO_ENABLED', '1')
    monkeypatch.setenv('MNEMOSYNE_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('MNEMOSYNE_AUTO_SLEEP', '0')
    monkeypatch.setattr(builtins, '__import__', old_core_import)
    mod = load_provider(name)
    provider = mod.MnemosyneMemoryProvider()
    ledger = provider._verbatim_ledger
    assert not ledger.enabled
    assert ledger.begin([], []) is None
    assert ledger.snapshot_for([]) is None
    assert ledger.release([], []) is None
    assert ledger.reset_session([]) is None
    # Use real initialization, narrowing only DB location and optional host I/O.
    monkeypatch.setattr(provider, '_apply_provider_config', lambda _: None)
    monkeypatch.setattr(provider, '_init_audit_log', lambda: None)
    monkeypatch.setattr(provider, '_capture_identity_signals', lambda _: None)
    provider._profile_isolation_enabled = False
    provider._auto_sleep_enabled = False
    provider._sync_roles = {'user', 'assistant'}
    def make_beam(**kw):
        kw['db_path'] = tmp_path / (kw['session_id'] + '.db')
        return BeamMemory(**kw)
    monkeypatch.setattr(mod, '_get_beam_class', lambda: make_beam)
    provider.initialize('prior-session', hermes_home=str(tmp_path))
    assert provider._beam is not None
    previous = provider._beam
    provider.initialize('session', hermes_home=str(tmp_path))
    previous.conn.close()
    beam = provider._beam
    assert beam is not None
    try:
        provider.on_pre_compress([{'role':'assistant', 'content':'anchor'}])
        provider.sync_turn('new turn without transcript', '', session_id='session')
        assert beam.conn.execute('SELECT 1 FROM working_memory WHERE content=?',
                                 ('[USER] new turn without transcript',)).fetchone()
        calls = []
        actual = beam.recall
        def old_recall(query, **kw):
            assert 'exclude_captures' not in kw
            calls.append(query)
            return actual(query, **kw)
        monkeypatch.setattr(beam, 'recall', old_recall)
        provider.prefetch('new turn without transcript', session_id='session')
        assert calls
        assert ledger.snapshot_for('session') is None
    finally:
        beam.conn.close()
        provider._beam = None
        provider._deactivate_in_module()


@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_exclusion_readback_closes_only_owned_connection(tmp_path, monkeypatch, borrowed, failure):
    real_connect = sqlite3.connect
    shared = real_connect(tmp_path/'memory.db') if borrowed else None
    engine = poly.PolyphonicRecallEngine(db_path=tmp_path/'memory.db', conn=shared)
    opened = []
    def tracked(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn
    for name in ('_graph_voice','_fact_voice','_vector_voice','_temporal_voice'):
        monkeypatch.setattr(engine, name, lambda *a, **kw: [])
    monkeypatch.setattr(poly.sqlite3, 'connect', tracked)
    seen = []
    def resolve(conn, snapshot):
        seen.append(conn)
        assert tuple(conn.execute('SELECT 1').fetchone()) == (1,)
        if failure:
            raise RuntimeError('readback failed')
        return set()
    monkeypatch.setattr(poly, 'resolve_exclusions', resolve)
    try:
        if failure:
            with pytest.raises(RuntimeError, match='readback failed'):
                engine.recall('query', exclude_captures=vl.ExclusionSnapshot(vl._Generation(), ()))
        else:
            engine.recall('query', exclude_captures=vl.ExclusionSnapshot(vl._Generation(), ()))
        assert len(seen) == 1
        if borrowed:
            assert opened == [] and seen[0] is shared
            assert tuple(shared.execute('SELECT 1').fetchone()) == (1,)
        else:
            assert opened == seen
            with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                opened[0].execute('SELECT 1')
    finally:
        for conn in opened:
            conn.close()
        if shared is not None:
            shared.close()
