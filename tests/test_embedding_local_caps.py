"""Existing opt-in embedding input caps also apply to local fastembed calls."""

import logging

import numpy as np
import pytest

from mnemosyne.core import embeddings


@pytest.fixture
def local_model(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [np.array([1.0, 2.0], dtype=np.float32) for _ in texts]

    for flag in (
        "MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF", "MNEMOSYNE_EMBEDDINGS_VIA_API",
    ):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_URL", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", raising=False)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX", "passage: ")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", "query: ")
    model = FakeModel()
    monkeypatch.setattr(embeddings, "TextEmbedding", object())
    monkeypatch.setattr(embeddings, "_get_model", lambda: model)
    embeddings._embed_query_cached.cache_clear()
    yield model
    embeddings._embed_query_cached.cache_clear()


def test_local_caps_are_disabled_by_default_and_zero(local_model, monkeypatch):
    embeddings.embed(["abcdef", "éééé"])
    embeddings.embed_query("abcdef")
    assert local_model.calls == [["passage: abcdef", "passage: éééé"], ["query: abcdef"]]
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "0")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", "-1")
    embeddings.embed(["abcdef"])
    assert local_model.calls[-1] == ["passage: abcdef"]


def test_local_document_char_and_byte_caps_keep_order_prefix_and_unicode(local_model, monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "13")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", "14")
    with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
        vectors = embeddings.embed(["éééé", "abcdef"])
    assert vectors.shape == (2, 2)
    assert local_model.calls == [["passage: éé", "passage: abcd"]]
    assert any("MNEMOSYNE_EMBEDDING_MAX_BYTES" in r.getMessage() for r in caplog.records)
    assert all("éééé" not in r.getMessage() for r in caplog.records)


def test_local_query_cache_tracks_character_byte_and_prefix_policy(local_model, monkeypatch):
    embeddings.embed_query("éééé")
    embeddings.embed_query("éééé")
    assert local_model.calls == [["query: éééé"]]
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "10")
    embeddings.embed_query("éééé")
    assert local_model.calls[-1] == ["query: ééé"]
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", "9")
    embeddings.embed_query("éééé")
    assert local_model.calls[-1] == ["query: é"]
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", "q: ")
    embeddings.embed_query("éééé")
    assert local_model.calls[-1] == ["q: ééé"]
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MAX_BYTES")
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MAX_CHARS")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", "query: ")
    embeddings.embed_query("éééé")
    assert len(local_model.calls) == 4


def test_local_byte_cap_refuses_empty_input_before_backend(local_model, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", "1")
    with pytest.raises(embeddings.EmbeddingPolicyError, match="MNEMOSYNE_EMBEDDING_MAX_BYTES"):
        embeddings.embed(["é"])
    assert local_model.calls == []


def test_local_query_caps_are_bound_to_cache_key_across_model_load(local_model, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "10")

    def change_cap_during_load():
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "9")
        return local_model

    monkeypatch.setattr(embeddings, "_get_model", change_cap_during_load)
    embeddings.embed_query("abcdef")
    assert local_model.calls == [["query: abc"]]
    embeddings.embed_query("abcdef")
    assert local_model.calls == [["query: abc"], ["query: ab"]]
    embeddings.embed_query("abcdef")
    assert len(local_model.calls) == 2
