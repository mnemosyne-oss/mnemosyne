"""Input policy changes must invalidate the outer enhanced-recall cache."""
import json

import numpy as np
import pytest

from mnemosyne.core import embeddings
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.config import MnemosyneConfig
from mnemosyne.core.query_cache import QueryCache


@pytest.mark.parametrize("cap", ["MNEMOSYNE_EMBEDDING_MAX_CHARS", "MNEMOSYNE_EMBEDDING_MAX_BYTES"])
@pytest.mark.parametrize("restart", [False, True])
def test_cap_change_recomputes_real_ranking_including_persisted_cache(tmp_path, monkeypatch, cap, restart):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_URL", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_VIA_API", raising=False)

    def unexpected_api(*args, **kwargs):
        pytest.fail("local ranking test dispatched to the API")

    monkeypatch.setattr(embeddings, "_embed_api", unexpected_api)
    for name in ("MNEMOSYNE_EMBEDDING_MAX_CHARS", "MNEMOSYNE_EMBEDDING_MAX_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", "")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX", "")
    MnemosyneConfig.reset_instance()
    db = tmp_path / "memory.db"
    cache = tmp_path / "cache.db"
    memory = BeamMemory(session_id="caps", db_path=db)
    model_calls = []

    class Model:
        def embed(self, texts):
            model_calls.extend(texts)
            return [np.array(([0., 1.] if text == "anchor" else [1., 0.]) + [0.] * 382,
                             dtype=np.float32) for text in texts]

    def close():
        if hasattr(memory, "_query_cache"):
            memory._query_cache.close()
        memory.conn.close()

    try:
        alpha = memory.remember("anchor alpha knowledge", importance=.5, source="test")
        beta = memory.remember("anchor beta knowledge", importance=.5, source="test")
        for mid, vector in ((alpha, [1., 0.]), (beta, [0., 1.])):
            memory.conn.execute(
                "INSERT OR REPLACE INTO memory_embeddings(memory_id,embedding_json,model) VALUES(?,?,?)",
                (mid, json.dumps(vector + [0.] * 382), "BAAI/bge-small-en-v1.5"),
            )
        memory.conn.commit()
        monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS")
        monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "BAAI/bge-small-en-v1.5")
        monkeypatch.setattr(embeddings, "_is_fastembed_available", lambda: True)
        monkeypatch.setattr(embeddings, "_get_model", lambda: Model())
        embeddings._embed_query_cached.cache_clear()
        memory._query_cache = QueryCache(db_path=cache)
        options = dict(top_k=2, use_weibull=False, use_mmr=False, use_intent=False,
                       use_synonyms=False, use_associative=False, vec_weight=1.,
                       fts_weight=0., importance_weight=0.)
        assert memory.recall_enhanced("anchor gamma", **options)[0]["id"] == alpha
        assert model_calls == ["anchor gamma"]
        if restart:
            close()
            memory = BeamMemory(session_id="caps", db_path=db)
            memory._query_cache = QueryCache(db_path=cache)
        monkeypatch.setenv(cap, "6")
        warm = memory.recall_enhanced("anchor gamma", **options)
        assert warm[0]["id"] == beta
        fresh = memory.recall_enhanced("anchor gamma", use_cache=False, **options)
        assert [row["id"] for row in warm] == [row["id"] for row in fresh]
        calls_before_hit = list(model_calls)
        assert memory.recall_enhanced("anchor gamma", **options)[0]["id"] == beta
        assert model_calls == calls_before_hit
    finally:
        close()
        embeddings._embed_query_cached.cache_clear()
        MnemosyneConfig.reset_instance()
