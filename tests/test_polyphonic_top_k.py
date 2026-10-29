"""Polyphonic recall must return min(top_k, eligible) results, never fewer.

Grid: tier (working, episodic) x top_k (at or under the old per-voice cap of
20, over it, over the point where the removed metadata-sized context budget
cut the engine's 2x over-fetch below top_k, and over the eligible count).
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
@pytest.mark.parametrize("top_k", [10, 20, 50, 80, 150])
def test_polyphonic_recall_returns_full_top_k(beam, seed, tier, top_k):
    seed(beam)
    results = beam.recall("unrelated opaque query", top_k=top_k)
    expected = min(top_k, ELIGIBLE)
    assert len(results) == expected
    assert len({r["id"] for r in results}) == expected
    assert {r["tier"] for r in results} == {tier}


def _reference_diversity_rerank(results, top_k):
    # The pairwise word-Jaccard selection as first written, kept as the oracle
    # for the precomputed-word-set version.
    def similarity(a, b):
        words_a = (a.content or a.metadata.get("content") or "").lower().split()
        words_b = (b.content or b.metadata.get("content") or "").lower().split()
        if not words_a or not words_b:
            return 0.0
        set_a, set_b = set(words_a), set(words_b)
        return len(set_a & set_b) / len(set_a | set_b)

    selected = []
    for result in sorted(results.values(), key=lambda x: x.combined_score, reverse=True):
        if len(selected) >= top_k:
            break
        if all(similarity(result, sel) <= 0.8 for sel in selected):
            selected.append(result)
    return selected


@pytest.mark.parametrize("seed", range(50))
def test_diversity_rerank_matches_reference(seed):
    import random

    from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine, PolyphonicResult

    rng = random.Random(seed)
    # A small vocabulary makes near-duplicates (Jaccard > 0.8) common.
    vocab = [f"w{i}" for i in range(8)]
    results = {}
    for i in range(rng.randint(0, 120)):
        words = rng.sample(vocab, rng.randint(0, 6))
        content = " ".join(words) if rng.random() < 0.8 else ""
        meta = {} if content or rng.random() < 0.5 else {"content": " ".join(words).upper()}
        results[f"m{i}"] = PolyphonicResult(memory_id=f"m{i}", combined_score=rng.random(),
                                            voice_scores={}, metadata=meta, content=content)
    top_k = rng.randint(1, 100)
    engine = PolyphonicRecallEngine.__new__(PolyphonicRecallEngine)
    got = [r.memory_id for r in engine._diversity_rerank(results, top_k)]
    want = [r.memory_id for r in _reference_diversity_rerank(results, top_k)]
    assert got == want
