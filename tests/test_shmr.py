"""SHMR regression tests for issue #762.

`_embed()` used to pass a ``str`` to ``embeddings.embed()``, which expects a
``List[str]`` and iterates it per item.  A ``str`` is iterated per character,
so the returned vector became ``dim * len(text)`` instead of ``dim``; the
clustering step then crashed with a dimension mismatch whenever two candidates
had different text lengths.  These tests pin the fixed contracts:

- ``_embed()`` hands ``embed()`` a list and returns a flat ``(EMBEDDING_DIM,)``
  vector regardless of input length.
- ``_cluster_by_similarity()`` tolerates variable-length candidate texts.
- ``harmonize()`` degrades gracefully when embeddings are disabled.
"""
import numpy as np

from mnemosyne.core import shmr
from mnemosyne.core.beam import BeamMemory


def _fake_embed_module(monkeypatch):
    """Replace ``shmr._embeddings.embed`` with a faithful fastembed stand-in.

    Mirrors the real contract that made the bug observable: a ``str`` argument
    is iterated per character (as ``embeddings.embed`` does with ``for t in
    texts``), while a list produces one vector per item.
    """
    seen = []

    def fake_embed(texts):
        seen.append(texts)
        if isinstance(texts, str):
            texts = list(texts)
        return np.stack(
            [np.full(shmr.EMBEDDING_DIM, 0.5, dtype=np.float32) for _ in texts]
        )

    monkeypatch.setattr(shmr._embeddings, "embed", fake_embed)
    return seen


def test_embed_passes_list_and_returns_fixed_dim(monkeypatch):
    seen = _fake_embed_module(monkeypatch)

    emb = shmr._embed("finished")

    assert isinstance(seen[-1], list)
    assert seen[-1] == ["finished"]
    assert emb.shape == (shmr.EMBEDDING_DIM,)


def test_embed_fixed_dim_independent_of_text_length(monkeypatch):
    _fake_embed_module(monkeypatch)

    shapes = {
        shmr._embed(t).shape
        for t in ("a", "long text here", "finished")
    }
    assert shapes == {(shmr.EMBEDDING_DIM,)}


def test_embed_falls_back_to_zero_vector_for_missing_or_empty_results(monkeypatch):
    expected = np.zeros(shmr.EMBEDDING_DIM, dtype=np.float32)

    for response in (None, np.empty((0, shmr.EMBEDDING_DIM), dtype=np.float32)):
        monkeypatch.setattr(shmr._embeddings, "embed", lambda texts: response)
        emb = shmr._embed("finished")

        assert emb.shape == (shmr.EMBEDDING_DIM,)
        assert emb.dtype == np.float32
        np.testing.assert_array_equal(emb, expected)


def test_cluster_by_similarity_tolerates_variable_length_texts(monkeypatch):
    _fake_embed_module(monkeypatch)

    candidates = [
        {"embedding": shmr._embed("short"), "content": "a"},
        {"embedding": shmr._embed("a considerably longer episodic recollection"), "content": "b"},
        {"embedding": shmr._embed("finished"), "content": "c"},
    ]

    clusters = shmr._cluster_by_similarity(candidates, threshold=0.1)

    assert isinstance(clusters, list)
    assert sum(len(c) for c in clusters) == len(candidates)


def test_harmonize_degrades_gracefully_without_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(shmr._embeddings, "embed", lambda texts: None)
    db_path = str(tmp_path / "shmr.db")
    beam = BeamMemory(session_id="shmr-test", db_path=db_path)

    conn = beam.conn
    for i, subject in enumerate(("alice", "bob", "carol")):
        conn.execute(
            "INSERT INTO facts (fact_id, session_id, subject, predicate, object, timestamp)"
            " VALUES (?, 'shmr-test', ?, 'likes', ?, datetime('now'))",
            (f"f{i}", subject, f"item {i}"),
        )
    conn.commit()

    result = shmr.harmonize(beam)

    assert isinstance(result, dict)
    assert "clusters_found" in result
