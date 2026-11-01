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
    row = memory.conn.execute("SELECT content, tier, degraded_at FROM episodic_memory WHERE id='target'").fetchone()
    assert tuple(row) == ('peer replacement', tier, None)
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


def test_sqlite_rollback_trigger_preserves_original_error(memory, monkeypatch):
    """RAISE(ROLLBACK) removes the savepoint; cleanup must not replace its error."""
    original = seed(memory, 2, 'target')
    monkeypatch.setattr(bm._embeddings, 'embed', lambda texts: np.ones((1, 384), dtype=np.float32))
    memory.conn.execute("CREATE TRIGGER abort_degradation BEFORE UPDATE OF tier ON episodic_memory BEGIN SELECT RAISE(ROLLBACK, 'original degradation failure'); END")
    memory.conn.commit()
    row = memory.conn.execute("SELECT id, rowid, content, tier, created_at FROM episodic_memory WHERE id='target'").fetchone()
    with pytest.raises(sqlite3.IntegrityError, match='original degradation failure'):
        memory._apply_episodic_degradation(row, 'compressed content', 2, datetime.now(timezone.utc))
    assert not memory.conn.in_transaction
    assert tuple(memory.conn.execute("SELECT content, tier, degraded_at FROM episodic_memory WHERE id='target'").fetchone()) == (original, 2, None)


def test_lost_savepoint_does_not_replace_keyboard_interrupt(memory, monkeypatch):
    original = seed(memory, 2, 'target')
    monkeypatch.setattr(bm._embeddings, 'embed', lambda texts: np.ones((1, 384), dtype=np.float32))
    def interrupt_after_rollback(*args, **kwargs):
        memory.conn.rollback()
        raise KeyboardInterrupt('original interrupt')
    monkeypatch.setattr(memory, '_refresh_episodic_embedding', interrupt_after_rollback)
    with pytest.raises(KeyboardInterrupt, match='original interrupt'):
        memory.degrade_episodic()
    assert not memory.conn.in_transaction
    assert tuple(memory.conn.execute("SELECT content, tier FROM episodic_memory WHERE id='target'").fetchone()) == (original, 2)


@pytest.mark.parametrize('tier', [1, 2])
def test_embedding_failure_warns_without_private_details_and_invalidates_stale_vector(
    memory, monkeypatch, caplog, tier,
):
    original = seed(memory, tier, 'target')

    def fail_embedding(texts):
        raise RuntimeError('private provider credential; input=' + texts[0])

    monkeypatch.setattr(bm._embeddings, 'embed', fail_embedding)
    result = memory.degrade_episodic()
    assert result[f'tier{tier}_to_tier{tier+1}'] == 1
    row = memory.conn.execute(
        "SELECT content, tier, binary_vector FROM episodic_memory WHERE id='target'"
    ).fetchone()
    assert row['content'] != original
    assert row['tier'] == tier + 1
    assert row['binary_vector'] is None
    assert memory.conn.execute(
        "SELECT 1 FROM memory_embeddings WHERE memory_id='target'"
    ).fetchone() is None
    warnings = [record for record in caplog.records if 'embedding failed' in record.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelname == 'WARNING'
    assert 'RuntimeError' in warnings[0].getMessage()
    assert warnings[0].exc_info is None
    assert 'private provider credential' not in caplog.text
    assert 'original detailed context' not in caplog.text
    assert not memory.conn.in_transaction
    with sqlite3.connect(memory.db_path, timeout=0) as peer:
        peer.execute('BEGIN IMMEDIATE')
        peer.rollback()
