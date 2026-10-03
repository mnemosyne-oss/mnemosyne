"""An empty embedding is a missing optional index, never a missing summary."""
import numpy as np
import pytest

import mnemosyne.core.beam as bm
from mnemosyne.core import local_llm
from tests.test_beam_event_timestamps import _seed_old_wm


@pytest.mark.parametrize('empty', [[], np.array([]), np.empty((0, 8)), np.empty((1, 0))])
@pytest.mark.parametrize('vec_available', [False, True])
def test_empty_embedding_keeps_summary(tmp_path, monkeypatch, empty, vec_available):
    beam = bm.BeamMemory(db_path=tmp_path / 'summary.db', session_id='empty')
    monkeypatch.setattr(bm._embeddings, 'available', lambda: True)
    monkeypatch.setattr(bm._embeddings, 'embed', lambda texts: empty)
    monkeypatch.setattr(bm, '_vec_available', lambda conn: vec_available)
    writes = []
    monkeypatch.setattr(bm, '_vec_insert', lambda *a, **k: writes.append(a))
    mid = beam.consolidate_to_episodic('An independently useful summary.', [])
    row = beam.conn.execute('SELECT content FROM episodic_memory WHERE id=?', (mid,)).fetchone()
    assert row[0] == 'An independently useful summary.'
    assert not writes
    assert beam.conn.execute('SELECT COUNT(*) FROM memory_embeddings WHERE memory_id=?', (mid,)).fetchone()[0] == 0


@pytest.mark.parametrize('vec_available', [False, True])
def test_empty_embedding_sleep_writes_summary_without_stranded_claims(tmp_path, monkeypatch, vec_available):
    beam = bm.BeamMemory(db_path=tmp_path / 'sleep.db', session_id='empty-sleep')
    _seed_old_wm(beam.conn, 'empty-sleep', [
        ('older', 'The collection contains rare botanical specimens from a tropical research expedition.', '2020-01-01T00:00:00', None, None),
        ('newer', 'The archive catalog records the expedition date and the location of each preserved specimen.', '2020-01-02T00:00:00', None, None),
    ])
    monkeypatch.setattr(local_llm, 'llm_available', lambda: False)
    monkeypatch.setattr(beam, '_detect_conflicts', lambda items: [])
    monkeypatch.setattr(bm._embeddings, 'available', lambda: True)
    monkeypatch.setattr(bm._embeddings, 'embed', lambda texts: np.array([]))
    monkeypatch.setattr(bm, '_vec_available', lambda conn: vec_available)
    result = beam.sleep(force=True)
    assert result['status'] == 'consolidated'
    assert beam.conn.execute('SELECT COUNT(*) FROM episodic_memory').fetchone()[0] >= 1
    remaining = beam.conn.execute('SELECT consolidation_claimed_at FROM working_memory WHERE id IN (?,?)', ('older', 'newer')).fetchall()
    assert all(row[0] is None for row in remaining)
