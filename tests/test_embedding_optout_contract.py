"""ENV-only opt-out contract: strict aliases and guards before cache/dispatch."""

from itertools import permutations

import numpy as np
import pytest

from mnemosyne.core import embeddings


FLAGS = (
    "MNEMOSYNE_NO_EMBEDDINGS",
    "MNEMOSYNE_SKIP_EMBEDDINGS",
    "MNEMOSYNE_EMBEDDINGS_OFF",
)


@pytest.fixture(autouse=True)
def isolated_embeddings(monkeypatch, tmp_path):
    for name in (
        *FLAGS,
        "MNEMOSYNE_EMBEDDING_API_URL",
        "MNEMOSYNE_EMBEDDINGS_VIA_API",
        "MNEMOSYNE_EMBEDDING_DIM",
        "MNEMOSYNE_EMBEDDING_QUERY_PREFIX",
        "MNEMOSYNE_EMBEDDING_DOC_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")
    monkeypatch.setattr(embeddings, "_embedding_model", None)
    monkeypatch.setattr(embeddings, "_FASTEMBED_CACHE_DIR", str(tmp_path / "models"))
    embeddings._embed_query_cached.cache_clear()
    yield
    embeddings._embed_query_cached.cache_clear()


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        (" \t\n", False),
        ("0", False),
        (" 0 ", False),
        ("false", False),
        ("no", False),
        ("off", False),
        (" FaLsE ", False),
        (" NO ", False),
        ("\tOff\n", False),
        ("1", True),
        (" 1 ", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        (" TrUe ", True),
        (" YES ", True),
        ("\tOn\n", True),
    ],
)
def test_boolean_alias_values(monkeypatch, flag, raw, expected):
    if raw is not None:
        monkeypatch.setenv(flag, raw)
    assert embeddings._is_disabled() is expected


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("raw", ["2", "disabled", "truthy", "false extra"])
def test_invalid_alias_is_configuration_error(monkeypatch, flag, raw):
    monkeypatch.setenv(flag, raw)
    with pytest.raises(ValueError, match=flag):
        embeddings._is_disabled()


@pytest.mark.parametrize("first,second,third", list(permutations(FLAGS)))
def test_mixed_aliases_validate_before_or(monkeypatch, first, second, third):
    monkeypatch.setenv(first, "false")
    monkeypatch.setenv(second, "true")
    monkeypatch.setenv(third, "off")
    assert embeddings._is_disabled() is True
    monkeypatch.setenv(third, "invalid")
    with pytest.raises(ValueError, match=third):
        embeddings._is_disabled()
    monkeypatch.setenv(first, "invalid")
    monkeypatch.setenv(third, "false")
    with pytest.raises(ValueError, match=first):
        embeddings._is_disabled()


@pytest.fixture(params=["local", "api"])
def backend(request, monkeypatch):
    """Double only model construction/HTTP, not the embedding call paths."""
    calls = {"load": 0, "embed": [], "request": []}

    class Model:
        def embed(self, texts):
            calls["embed"].append(list(texts))
            return [np.ones(384, dtype=np.float32) for _ in texts]

    def load(**kwargs):
        calls["load"] += 1
        return Model()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            import json

            texts = calls["request"][-1]["input"]
            return json.dumps(
                {"data": [{"embedding": [1.0] * 384} for _ in texts]}
            ).encode()

    def urlopen(req, **kwargs):
        import json

        calls["request"].append(json.loads(req.data))
        return Response()

    monkeypatch.setattr(embeddings, "TextEmbedding", load)
    monkeypatch.setattr(embeddings, "_FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", urlopen)
    if request.param == "api":
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://example.test/v1")
    return calls


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("raw", ["true", "invalid"])
@pytest.mark.parametrize(
    "operation", ["available", "_get_model", "embed", "embed_query"]
)
def test_guard_before_loading_or_api_dispatch(
    monkeypatch, backend, flag, raw, operation
):
    monkeypatch.setenv(flag, raw)
    args = {
        "available": (),
        "_get_model": (),
        "embed": (["one", "two"],),
        "embed_query": ("query",),
    }[operation]
    call = getattr(embeddings, operation)
    if raw == "invalid":
        with pytest.raises(ValueError, match=flag):
            call(*args)
    else:
        assert call(*args) is (False if operation == "available" else None)
    assert backend == {"load": 0, "embed": [], "request": []}
    assert embeddings._embed_query_cached.cache_info().currsize == 0


@pytest.mark.parametrize("flag", FLAGS)
def test_warm_cache_disable_invalid_and_reenable(monkeypatch, backend, flag):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX", "passage: ")
    first = embeddings.embed_query("warm query")
    assert first.shape == (384,)
    assert first.dtype == np.float32
    assert embeddings.embed_query("warm query") is first
    before = {
        key: list(value) if isinstance(value, list) else value
        for key, value in backend.items()
    }
    cache_before = embeddings._embed_query_cached.cache_info()
    monkeypatch.setenv(flag, " ON ")
    assert embeddings.embed_query("warm query") is None
    assert embeddings.embed_query("cold query") is None
    assert embeddings.embed(["document", "other"]) is None
    assert embeddings._get_model() is None
    assert backend == before
    assert embeddings._embed_query_cached.cache_info() == cache_before
    monkeypatch.setenv(flag, "invalid")
    other_flag = next(name for name in FLAGS if name != flag)
    monkeypatch.setenv(other_flag, "true")
    with pytest.raises(ValueError, match=flag):
        embeddings.embed_query("warm query")
    assert backend == before
    assert embeddings._embed_query_cached.cache_info() == cache_before
    monkeypatch.setenv(flag, " FaLsE ")
    monkeypatch.setenv(other_flag, "off")
    assert embeddings.available() is True
    assert embeddings.embed_query("warm query") is first
    docs = embeddings.embed(["document", "other"])
    np.testing.assert_array_equal(docs, np.ones((2, 384), dtype=np.float32))
    assert docs.dtype == np.float32
    if backend["request"]:
        assert backend["request"][0]["input"] == ["query: warm query"]
        assert backend["request"][-1]["input"] == [
            "passage: document",
            "passage: other",
        ]
        assert len(backend["request"]) == 2
    else:
        assert backend["load"] == 1
        assert backend["embed"] == [
            ["query: warm query"],
            ["passage: document", "passage: other"],
        ]


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("operation", ["embed", "embed_query"])
def test_invalid_alias_precedes_empty_input(monkeypatch, backend, flag, operation):
    monkeypatch.setenv(flag, "invalid")
    with pytest.raises(ValueError, match=flag):
        getattr(embeddings, operation)([] if operation == "embed" else "")
    assert backend == {"load": 0, "embed": [], "request": []}


@pytest.mark.parametrize("flag", FLAGS)
def test_false_alias_preserves_unknown_model_failure(monkeypatch, flag):
    monkeypatch.setenv(flag, "false")
    with pytest.raises(ValueError, match="Unknown embedding model"):
        embeddings._get_embedding_dim("unknown/model")


@pytest.mark.parametrize("api", [False, True])
@pytest.mark.parametrize("operation", ["embed", "embed_query"])
def test_enabled_errors_propagate_unchanged(monkeypatch, api, operation):
    error = RuntimeError("distinct boundary failure")

    def fail(*args):
        raise error

    if api:
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://example.test/v1")
        monkeypatch.setattr(embeddings, "_embed_api", fail)
    else:
        monkeypatch.setattr(embeddings, "_get_model", fail)
    with pytest.raises(RuntimeError) as exc:
        getattr(embeddings, operation)(
            ["document"] if operation == "embed" else "query"
        )
    assert exc.value is error


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("surface", ["core", "beam"])
def test_disabled_public_recall_retains_keyword_result(
    monkeypatch, tmp_path, backend, flag, surface
):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.memory import Mnemosyne

    monkeypatch.setenv(flag, "true")
    store = None
    try:
        cls = Mnemosyne if surface == "core" else BeamMemory
        store = cls(session_id="anonymous-optout", db_path=tmp_path / "memory.db")
        memory_id = store.remember("anonymous optout keyword document", source="test")
        results = store.recall("optout keyword document", top_k=3)
        assert any(row["id"] == memory_id for row in results)
        assert backend == {"load": 0, "embed": [], "request": []}
    finally:
        if store is not None:
            store.conn.close()
