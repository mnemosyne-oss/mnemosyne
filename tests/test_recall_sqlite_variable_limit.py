"""Linear recall must preserve its results on SQLite's legacy bind budget (#1307)."""
import random
import sqlite3

import numpy as np
import pytest

from mnemosyne.core import beam as bm


@pytest.fixture
def make_store(tmp_path, monkeypatch):
    monkeypatch.setenv('MNEMOSYNE_NO_EMBEDDINGS', '1')
    monkeypatch.setenv('MNEMOSYNE_POLYPHONIC_RECALL', '0')
    monkeypatch.setenv('MNEMOSYNE_ENHANCED_RECALL', '1')
    stores = []

    def make(tier):
        memory = bm.BeamMemory(session_id='owner', db_path=tmp_path / f'{tier}.db')
        stores.append(memory)
        table = f'{tier}_memory'
        memory.conn.executemany(
            f'INSERT INTO {table} (id,content,source,timestamp,importance,session_id,channel_id,author_id,author_type) '
            'VALUES (?,?,?,?,?,?,?,?,?)',
            [(f'm{i:04}', f'Zephyr launch record {i}', 'user', '2026-01-01T00:00:00',
              (i + 1) / 1300, 'owner', 'channel', 'alice', 'human') for i in range(1200)]
            + [('foreign', 'Zephyr launch foreign', 'user', '2026-01-01', 1., 'other', 'other', 'bob', 'agent'),
               ('expired', 'Zephyr launch expired', 'user', '2026-01-01', 1., 'owner', 'channel', 'alice', 'human')],
        )
        memory.conn.execute(f"UPDATE {table} SET valid_until='2000-01-01' WHERE id='expired'")
        memory.conn.commit()
        return memory

    yield make
    for memory in stores:
        if getattr(memory, '_query_cache', None) is not None:
            memory._query_cache.close()
        memory.conn.close()
        if getattr(memory, '_annotations', None) is not None:
            memory._annotations.conn.close()


def limit(memory, value):
    if not hasattr(memory.conn, 'setlimit'):
        pytest.skip('runtime SQLite limits require Python 3.11+')
    memory.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, value)
    # Statements prepared before setlimit otherwise remain executable from sqlite3's cache.
    for i in range(160):
        memory.conn.execute(f'SELECT {i}').fetchall()


def signature(rows):
    return [(r['tier'], r['id'], r['score']) for r in rows]


FILTERS = dict(_cross_session=False, source='user', channel_id='channel', author_id='alice', author_type='human')


@pytest.mark.parametrize('bind_limit', [31, 999])
@pytest.mark.parametrize('tier', ['working', 'episodic'])
@pytest.mark.parametrize('kind', ['mentions', 'fact'])
def test_annotation_fanout_respects_limit_and_scope(make_store, tier, kind, bind_limit):
    memory = make_store(tier)
    annotations = memory.annotations
    annotations.conn.executemany('INSERT INTO annotations(memory_id,kind,value) VALUES (?,?,?)',
                                 [(f'm{i:04}', kind, 'Zephyr') for i in range(1200)]
                                 + [('foreign', kind, 'Zephyr'), ('expired', kind, 'Zephyr')])
    annotations.conn.commit()
    baseline = memory.recall('Zephyr', top_k=5, _track_recall=False, **FILTERS)
    assert len(baseline) == 5
    limit(memory, bind_limit)
    actual = memory.recall('Zephyr', top_k=5, **FILTERS)
    assert signature(actual) == signature(baseline)
    assert all(r['id'].startswith('m') for r in actual)
    credited = dict(memory.conn.execute(f'SELECT id,recall_count FROM {tier}_memory'))
    assert {mid for mid, count in credited.items() if count} == {r['id'] for r in actual}
    assert max(credited.values()) == 1


@pytest.mark.parametrize('tier', ['working', 'episodic'])
def test_large_candidate_pool_and_tracking_preserve_results(make_store, tier):
    memory = make_store(tier)
    baseline = memory.recall('launch', top_k=1200, _track_recall=False, **FILTERS)
    assert len(baseline) == 1200
    limit(memory, 999)
    actual = memory.recall('launch', top_k=1200, **FILTERS)
    assert signature(actual) == signature(baseline)
    counts = dict(memory.conn.execute(f'SELECT id,recall_count FROM {tier}_memory'))
    assert all(counts[f'm{i:04}'] == 1 for i in range(1200))
    assert counts['foreign'] == counts['expired'] == 0


def test_dense_candidate_fetch_with_small_top_k(make_store, monkeypatch):
    memory = make_store('working')
    vector = np.zeros(bm.EMBEDDING_DIM, dtype=np.float32)
    vector[0] = 1.
    monkeypatch.setattr(bm._embeddings, 'available', lambda: True)
    monkeypatch.setattr(bm._embeddings, 'embed_query', lambda _: vector)
    for row in memory.conn.execute('SELECT id FROM working_memory').fetchall():
        bm._store_working_embedding(memory.conn, row['id'], vector)
    memory.conn.commit()
    baseline = memory.recall('launch', top_k=400, _track_recall=False, **FILTERS)
    assert len(baseline) == 400
    limit(memory, 999)
    actual = memory.recall('launch', top_k=400, _track_recall=False, **FILTERS)
    assert signature(actual) == signature(baseline)
    assert all(r['dense_score'] == 1. for r in actual)


def test_memoria_source_fanout_is_not_silently_lost(make_store, monkeypatch):
    memory = make_store('working')
    source_ids = [f'm{i:04}' for i in range(1200)] + ['foreign', 'expired']
    # Inject only the candidate provider; hydration, SQL and scope checks stay real.
    monkeypatch.setattr(memory, 'memoria_retrieve', lambda *a, **k: {
        'source': 'review', 'context': 'Zephyr launch', 'source_memory_ids': source_ids})
    baseline = memory.recall('Zephyr launch', top_k=2500, _track_recall=False, **FILTERS)
    expected = {f'memoria_source_m{i:04}' for i in range(1200)}
    assert expected <= {r['id'] for r in baseline}
    limit(memory, 999)
    actual = memory.recall('Zephyr launch', top_k=2500, _track_recall=False, **FILTERS)
    assert signature(actual) == signature(baseline)
    assert expected <= {r['id'] for r in actual}


@pytest.mark.parametrize('tier', ['working', 'episodic'])
def test_enhanced_weibull_large_pool(make_store, tier):
    memory = make_store(tier)
    options = dict(top_k=600, use_weibull=True, use_mmr=False, use_intent=False,
                   use_synonyms=False, use_associative=False, use_cache=False, _track_recall=False, **FILTERS)
    options.pop('_cross_session')  # Enhanced recall resolves this internally.
    baseline = memory.recall_enhanced('launch', **options)
    assert len(baseline) == 600
    limit(memory, 999)
    assert signature(memory.recall_enhanced('launch', **options)) == signature(baseline)


def test_summary_link_hydration_large_pool(make_store):
    memory = make_store('episodic')
    memory.conn.execute("UPDATE episodic_memory SET summary_of='working-source' WHERE id='m1199'")
    memory.conn.commit()
    rows = [{'tier': 'episodic', 'id': f'm{i:04}', 'score': .5} for i in range(1200)]
    rows.append({'tier': 'working', 'id': 'working-source', 'score': .1})
    limit(memory, 999)
    actual = memory._dedup_cross_tier_summary_links(rows)
    assert len(actual) == 1200
    assert actual == rows[:-1]


def test_select_reserves_filter_parameters_and_deduplicates(make_store):
    memory = make_store('working')
    limit(memory, 7)
    rows = bm._select_recall_rows_by_ids(
        memory.conn,
        'SELECT id FROM working_memory WHERE id IN ({placeholders}) '
        'AND session_id=? AND channel_id=? AND author_type=?',
        ['foreign', 'm0008', 'm0000', 'm0007', 'm0003', 'm0001', 'm0000', 'expired'],
        ('owner', 'channel', 'human'),
    )
    assert [r['id'] for r in rows] == ['expired', 'm0000', 'm0001', 'm0003', 'm0007', 'm0008']


def test_select_python310_limit_fallback_uses_real_sqlite(make_store):
    memory = make_store('working')
    if hasattr(memory.conn, 'setlimit'):
        limit(memory, 999)

    class Python310Connection:
        # Same DB, deliberately without the Python 3.11 getlimit API.
        execute = memory.conn.execute

    rows = bm._select_recall_rows_by_ids(
        Python310Connection(), 'SELECT id FROM working_memory WHERE id IN ({placeholders})',
        [f'm{i:04}' for i in range(1200)],
    )
    assert [r['id'] for r in rows] == [f'm{i:04}' for i in range(1200)]


def test_empty_id_lookup_does_not_execute_even_with_no_budget(make_store):
    memory = make_store('working')
    if hasattr(memory.conn, 'setlimit'):
        limit(memory, 0)
    assert bm._select_recall_rows_by_ids(memory.conn, 'invalid SQL {placeholders}', []) == []


@pytest.mark.parametrize('order', ['reverse', 'shuffled'])
def test_equal_score_top_k_is_independent_of_bind_budget(make_store, order):
    memory = make_store('working')
    memory.conn.execute('DELETE FROM working_memory')
    ids = [f'm{i:04}' for i in range(100)]
    if order == 'reverse':
        ids.reverse()
    else:
        random.Random(1307).shuffle(ids)
    memory.conn.executemany(
        "INSERT INTO working_memory(id,content,source,timestamp,importance,session_id) "
        "VALUES (?, 'Zephyr launch', 'user', '2026-01-01T00:00:00', .5, 'owner')",
        [(mid,) for mid in ids],
    )
    memory.conn.commit()
    options = dict(top_k=40, source='user', _cross_session=False, _track_recall=False)
    baseline = memory.recall('launch', **options)
    assert [r['id'] for r in baseline] == [f'm{i:04}' for i in range(40)]
    limit(memory, 31)
    assert signature(memory.recall('launch', **options)) == signature(baseline)
