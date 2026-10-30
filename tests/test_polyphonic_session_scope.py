"""Polyphonic recall must not let rows the caller cannot see starve rows it can.

Invariant: a scope predicate pushed into a voice may drop only rows that
``BeamMemory._polyphonic_row_passes_filters`` would drop anyway, and a row the
caller can see is never cut by rows it cannot.

Grid: cut site x visibility mode.

- Cut sites: the working-memory vector scan, the temporal voice's LIMIT 20,
  and the fused top-K cut (fed here by a graph voice, whose stores have no
  session column). The episodic vector voice is not a cell: it already
  applies ``episodic_where`` before its limits. One accepted exception: that
  predicate drops episodic rows with a NULL session_id, which the post-filter
  would keep. The linear path drops them too, so it is left as it was.
- Plus one dual-tier cell: an id whose working row is foreign but whose
  episodic row is in scope must survive the fused cut.
- Visibility modes, each a target row the caller must see, ranked below 30
  noise rows:
  own-session target, noise from another session;
  target in another session but on the caller's channel, noise on another channel;
  target in another session with scope='global';
  target with a NULL session_id (import can write one), which the post-filter keeps;
  cross-session on, where the noise is visible too and must not be filtered out;
  author-only recall, which has no session scope: a target from another
  session by that author must not be filtered out, nor starved by other authors;
  a channel recall whose noise is the caller's own session on another channel.
"""
import sqlite3

import numpy as np
import pytest

import mnemosyne.core.beam as bm
from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine, RecallResult

NOISE = 30
CALLER = "caller-session"
OTHER = "other-session"


def _unit(vector):
    return vector / np.linalg.norm(vector)


QUERY = _unit(np.ones(384, dtype=np.float32))
# Noise is strictly closer to the query than the target, so an unscoped cut
# always keeps noise over the target.
NEAR = QUERY
FAR = _unit(np.concatenate([np.ones(300), -np.ones(84)]).astype(np.float32))


def _vector_for(text):
    return FAR if text.startswith("target") else NEAR


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(bm._embeddings, "embed_query", lambda text: QUERY)
    monkeypatch.setattr(bm._embeddings, "embed",
                        lambda texts: np.stack([_vector_for(t) for t in texts]))
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.delenv("MNEMOSYNE_CROSS_SESSION", raising=False)
    for voice in ("VECTOR", "TEMPORAL", "GRAPH", "FACT"):
        monkeypatch.setenv(f"MNEMOSYNE_VOICE_{voice}", "0")
    db = tmp_path / "memory.db"
    caller = bm.BeamMemory(session_id=CALLER, db_path=db)
    other = bm.BeamMemory(session_id=OTHER, db_path=db)
    return caller, other, monkeypatch


def _set(beam, memory_id, **columns):
    for column, value in columns.items():
        beam.conn.execute(f"UPDATE working_memory SET {column} = ? WHERE id = ?", (value, memory_id))
    beam.conn.commit()


# Each mode: (target writer, target columns, noise writer, noise columns,
# recall kwargs, cross-session).
MODES = {
    "own-session": ("caller", {}, "other", {}, {}, False),
    "channel": ("other", {"channel_id": "c1"}, "other", {"channel_id": "c2"}, {"channel_id": "c1"}, False),
    "channel-own-session-noise": ("other", {"channel_id": "c1"}, "caller", {"channel_id": "c2"},
                                  {"channel_id": "c1"}, False),
    "global": ("other", {"scope": "global"}, "other", {}, {}, False),
    "null-session": ("other", {"session_id": None}, "other", {}, {}, False),
    "cross-session": ("other", {}, "other", {}, {}, True),
    "author-only": ("other", {"author_id": "a1"}, "other", {"author_id": "a2"}, {"author_id": "a1"}, False),
}


def _seed(caller, other, mode, target_first):
    writer, target_cols, noise_writer, noise_cols, _, _ = MODES[mode]
    target_beam = caller if writer == "caller" else other
    noise_beam = caller if noise_writer == "caller" else other
    noise_ids = []
    target_id = None
    if target_first:
        target_id = target_beam.remember("target recent quokka", source="user", dedupe=False)
    for i in range(NOISE):
        noise_ids.append(noise_beam.remember(f"noise recent quokka {i} n{i}", source="user", dedupe=False))
    if not target_first:
        target_id = target_beam.remember("target recent quokka", source="user", dedupe=False)
    _set(target_beam, target_id, **target_cols)
    for noise_id in noise_ids:
        _set(noise_beam, noise_id, **noise_cols)
    return target_id, set(noise_ids)


def _check(results, mode, target_id, noise_ids):
    ids = [r["id"] for r in results]
    if MODES[mode][5]:
        # Cross-session: the noise is visible, so it legitimately outranks the target.
        assert set(ids) & noise_ids
    else:
        assert target_id in ids
        assert not set(ids) & noise_ids


def _recall(caller, monkeypatch, mode, query):
    kwargs = MODES[mode][4]
    if MODES[mode][5]:
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "1")
    return caller.recall(query, top_k=1, **kwargs)


@pytest.mark.parametrize("mode", list(MODES))
def test_vector_scan_is_scoped(env, mode):
    caller, other, monkeypatch = env
    monkeypatch.setenv("MNEMOSYNE_VOICE_VECTOR", "1")
    target_id, noise_ids = _seed(caller, other, mode, target_first=False)
    _check(_recall(caller, monkeypatch, mode, "quokka"), mode, target_id, noise_ids)


@pytest.mark.parametrize("mode", list(MODES))
def test_temporal_limit_is_scoped(env, mode):
    caller, other, monkeypatch = env
    monkeypatch.setenv("MNEMOSYNE_VOICE_TEMPORAL", "1")
    # Target first, so all 30 noise rows are newer and fill an unscoped LIMIT 20.
    target_id, noise_ids = _seed(caller, other, mode, target_first=True)
    _check(_recall(caller, monkeypatch, mode, "recent quokka"), mode, target_id, noise_ids)


@pytest.mark.parametrize("mode", list(MODES))
def test_fused_cut_is_scoped(env, mode):
    caller, other, monkeypatch = env
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "1")
    target_id, noise_ids = _seed(caller, other, mode, target_first=False)
    ranked = [*sorted(noise_ids), target_id]

    def graph_voice(self, query):
        # A graph store with no session column, ranking every noise row above the target.
        return [RecallResult(memory_id=mid, score=1.0 - i / 100, voice="graph", metadata={})
                for i, mid in enumerate(ranked)]

    monkeypatch.setattr(PolyphonicRecallEngine, "_graph_voice", graph_voice)
    _check(_recall(caller, monkeypatch, mode, "quokka"), mode, target_id, noise_ids)


def test_fused_cut_keeps_id_in_scope_in_either_tier(env):
    # Untyped graph hits are hydrated episodic-first, so an id whose working
    # row is foreign but whose episodic row is the caller's own is visible.
    caller, other, monkeypatch = env
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "1")
    target_id, noise_ids = _seed(caller, other, "own-session", target_first=False)
    _set(caller, target_id, session_id=OTHER)
    caller.conn.execute(
        # scope defaults to 'global' in the table; this row must be session-scoped.
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance, session_id, scope) "
        "VALUES (?, 'target recent quokka summary', 'user', datetime('now'), 0.5, ?, 'session')",
        (target_id, CALLER),
    )
    caller.conn.commit()
    ranked = [*sorted(noise_ids), target_id]

    def graph_voice(self, query):
        return [RecallResult(memory_id=mid, score=1.0 - i / 100, voice="graph", metadata={})
                for i, mid in enumerate(ranked)]

    monkeypatch.setattr(PolyphonicRecallEngine, "_graph_voice", graph_voice)
    results = caller.recall("quokka", top_k=1)
    assert [(r["id"], r["tier"]) for r in results] == [(target_id, "episodic")]


@pytest.mark.skipif(not hasattr(sqlite3.Connection, "setlimit"),
                    reason="sqlite3.Connection.setlimit needs Python 3.11+")
def test_fused_cut_prune_stays_under_sqlite_variable_limit(env):
    # Older SQLite builds allow 999 bound variables; a graph voice can name
    # more ids than that, and the prune must neither fail nor be skipped.
    caller, other, monkeypatch = env
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "1")
    target_id = caller.remember("target recent quokka", source="user", dedupe=False)
    noise_ids = [f"noise-{i}" for i in range(1200)]
    caller.conn.executemany(
        # scope defaults to 'global' in the table; remember() writes 'session'.
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, ?, 'user', datetime('now'), ?, 0.5, 'session')",
        [(mid, f"noise quokka {mid}", OTHER) for mid in noise_ids],
    )
    caller.conn.commit()
    caller.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 1000)
    ranked = [*noise_ids, target_id]

    def graph_voice(self, query):
        return [RecallResult(memory_id=mid, score=1.0 - i / 10000, voice="graph", metadata={})
                for i, mid in enumerate(ranked)]

    monkeypatch.setattr(PolyphonicRecallEngine, "_graph_voice", graph_voice)
    assert [r["id"] for r in caller.recall("quokka", top_k=1)] == [target_id]


def test_fused_cut_drops_ids_with_no_memory_row(env):
    # Graph and fact hits can name gists or facts with no memory row. BeamMemory
    # discards them, so they must not hold fused slots above a real memory.
    caller, other, monkeypatch = env
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "1")
    target_id = caller.remember("target recent quokka", source="user", dedupe=False)
    ranked = [*(f"gist-{i}" for i in range(NOISE)), target_id]

    def graph_voice(self, query):
        return [RecallResult(memory_id=mid, score=1.0 - i / 100, voice="graph", metadata={})
                for i, mid in enumerate(ranked)]

    monkeypatch.setattr(PolyphonicRecallEngine, "_graph_voice", graph_voice)
    assert [r["id"] for r in caller.recall("quokka", top_k=1)] == [target_id]
