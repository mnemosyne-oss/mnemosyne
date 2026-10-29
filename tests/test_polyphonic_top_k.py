"""Polyphonic recall must return min(top_k, eligible) results, never fewer.

Grid: tier (working, episodic) x top_k (at or under the old per-voice cap of
20, over it, and over the point where the removed metadata-sized context budget
cut the engine's 2x over-fetch below top_k).
"""
import numpy as np
import pytest

import mnemosyne.core.beam as bm

ELIGIBLE = 100


def _distinct(i):
    # Distinct words per row so the diversity re-rank keeps every one.
    return f"record{i} alpha{i} bravo{i} charlie{i} delta{i}"


@pytest.fixture
def beam(tmp_path, monkeypatch):
    query = np.ones(384, dtype=np.float32)
    query /= np.linalg.norm(query)
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(bm._embeddings, "embed_query", lambda text: query)
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: np.stack([query for _ in texts]))
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    # Only the vector voice: the others would add rows and hide a short vector voice.
    for voice in ("TEMPORAL", "GRAPH", "FACT"):
        monkeypatch.setenv(f"MNEMOSYNE_VOICE_{voice}", "0")
    monkeypatch.delenv("MNEMOSYNE_VOICE_VECTOR", raising=False)
    return bm.BeamMemory(session_id="top-k-session", db_path=tmp_path / "memory.db")


def _seed_working(beam):
    for i in range(ELIGIBLE):
        beam.remember(_distinct(i), source="user", dedupe=False)


def _seed_episodic(beam):
    pytest.importorskip("sqlite_vec")
    for i in range(ELIGIBLE):
        beam.consolidate_to_episodic(_distinct(i), [], source="user")


@pytest.mark.parametrize("seed,tier", [(_seed_working, "working"), (_seed_episodic, "episodic")],
                         ids=["working", "episodic"])
@pytest.mark.parametrize("top_k", [10, 20, 50, 80])
def test_polyphonic_recall_returns_full_top_k(beam, seed, tier, top_k):
    seed(beam)
    results = beam.recall("unrelated opaque query", top_k=top_k)
    assert len(results) == top_k
    assert len({r["id"] for r in results}) == top_k
    assert {r["tier"] for r in results} == {tier}
