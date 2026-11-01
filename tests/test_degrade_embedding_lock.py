"""External embedding latency must not hold a degradation write lock (#695)."""
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mnemosyne.core import beam as bm
from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, '_SQLITE_VEC_AVAILABLE', False)
    monkeypatch.setattr(bm, '_vec_available', lambda conn: False)
    monkeypatch.setattr(bm._embeddings, 'available', lambda: True)
    monkeypatch.setattr(bm, 'SMART_COMPRESS', False)
    from mnemosyne.core import local_llm
    monkeypatch.setattr(local_llm, 'llm_available', lambda: False)
    beam = BeamMemory(session_id='s', db_path=tmp_path / 'bank.db')
    yield beam
    beam.conn.close()


def seed(beam, tier, mid):
    content = 'original detailed context ' * 100
    old = (datetime.now(timezone.utc) - timedelta(days=400)).replace(tzinfo=None).isoformat()
    beam.conn.execute('INSERT INTO episodic_memory (id, content, tier, created_at, session_id) VALUES (?, ?, ?, ?, ?)',
                      (mid, content, tier, old, 's'))
    beam.conn.execute('INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)', (mid, '[0.5]', 'old'))
    beam.conn.commit()
    return content


@pytest.mark.parametrize('tier', [1, 2])
def test_embed_allows_other_writer_for_each_degraded_row(memory, monkeypatch, tier):
    for mid in ['first', 'second']:
        seed(memory, tier, mid)
    memory.conn.execute('CREATE TABLE concurrent_writes (id INTEGER)')
    memory.conn.commit()
    observations = []
    def embed(texts):
        with sqlite3.connect(memory.db_path, timeout=0) as peer:
            try:
                peer.execute('INSERT INTO concurrent_writes VALUES (1)')
                peer.commit()
                observations.append(True)
            except sqlite3.OperationalError:
                observations.append(False)
        return np.ones((1, 384), dtype=np.float32)
    monkeypatch.setattr(bm._embeddings, 'embed', embed)
    result = memory.degrade_episodic()
    assert observations == [True, True]
    assert result[f'tier{tier}_to_tier{tier+1}'] == 2
    assert memory.conn.execute('SELECT COUNT(*) FROM concurrent_writes').fetchone()[0] == 2
    assert not memory.conn.in_transaction


@pytest.mark.parametrize('tier', [1, 2])
def test_concurrent_content_change_is_not_overwritten(memory, monkeypatch, tier):
    seed(memory, tier, 'target')
    def embed(texts):
        with sqlite3.connect(memory.db_path, timeout=0) as peer:
            peer.execute("UPDATE episodic_memory SET content='peer replacement' WHERE id='target'")
            peer.execute("UPDATE memory_embeddings SET embedding_json='[9.0]' WHERE memory_id='target'")
            peer.commit()
        return np.ones((1, 384), dtype=np.float32)
    monkeypatch.setattr(bm._embeddings, 'embed', embed)
    result = memory.degrade_episodic()
    row = memory.conn.execute("SELECT content, tier FROM episodic_memory WHERE id='target'").fetchone()
    assert tuple(row) == ('peer replacement', tier)
    assert result[f'tier{tier}_to_tier{tier+1}'] == 0
    assert memory.conn.execute("SELECT embedding_json FROM memory_embeddings WHERE memory_id='target'").fetchone()[0] == '[9.0]'


def test_interrupted_vector_write_rolls_back_and_releases_lock(memory, monkeypatch):
    original = seed(memory, 2, 'target')
    monkeypatch.setattr(bm._embeddings, 'embed', lambda texts: np.ones((1, 384), dtype=np.float32))
    real_refresh = memory._refresh_episodic_embedding
    def interrupt(*args, **kwargs):
        real_refresh(*args, **kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(memory, '_refresh_episodic_embedding', interrupt)
    with pytest.raises(KeyboardInterrupt):
        memory.degrade_episodic()
    assert not memory.conn.in_transaction
    assert tuple(memory.conn.execute("SELECT content, tier FROM episodic_memory WHERE id='target'").fetchone()) == (original, 2)
    assert memory.conn.execute("SELECT embedding_json FROM memory_embeddings WHERE memory_id='target'").fetchone()[0] == '[0.5]'
    with sqlite3.connect(memory.db_path, timeout=0) as peer:
        peer.execute('BEGIN IMMEDIATE')
        peer.rollback()
