"""embed_query's cache must not serve a vector from one embedding backend to another.

The backend is resolved at call time (MNEMOSYNE_EMBEDDING_API_URL, embeddings_via_api),
so it can change inside one process. Documents are embedded with the current backend,
and a query vector cached under the old one would come from a different space.
"""

import numpy as np
import pytest

from mnemosyne.core import embeddings

URL_A = "http://127.0.0.1:11435/v1"
URL_B = "http://127.0.0.1:11436/v1"
REAL_GET_MODEL = embeddings._get_model


class FakeModel:
    def __init__(self, calls):
        self.calls = calls

    def embed(self, texts):
        self.calls.append(("local", None))
        return [np.array([-1.0], dtype=np.float32) for _ in texts]


@pytest.fixture
def calls(monkeypatch, tmp_path):
    # Config wins over env, and the default model is read at import: pin both so a
    # developer's config.yaml or MNEMOSYNE_EMBEDDING_MODEL can't pick the backend.
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "BAAI/bge-small-en-v1.5")
    for name in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
        "MNEMOSYNE_EMBEDDINGS_VIA_API",
        "MNEMOSYNE_EMBEDDING_API_URL",
        "MNEMOSYNE_EMBEDDING_QUERY_PREFIX",
        "MNEMOSYNE_EMBEDDING_MAX_CHARS",
        "MNEMOSYNE_EMBEDDING_MAX_BYTES",
        "MNEMOSYNE_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")
    log = []

    # Faked above the transport so the https (OpenRouter) case never reaches the network.
    def fake_embed_api(texts, base_url=None):
        log.append(("api", base_url))
        return np.ones((len(texts), 1), dtype=np.float32)

    monkeypatch.setattr(embeddings, "_embed_api", fake_embed_api)
    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel(log))
    # Backend/cache tests use a fake local model, without requiring fastembed.
    monkeypatch.setattr(embeddings, "_is_fastembed_available", lambda: True)
    embeddings._embed_query_cached.cache_clear()
    yield log
    embeddings._embed_query_cached.cache_clear()


def _use(monkeypatch, backend):
    if backend == "local":
        monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_URL", raising=False)
    else:
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", backend)


@pytest.mark.parametrize(
    "first, second, reused",
    [
        ("local", "local", True),
        (URL_A, URL_A, True),
        ("local", URL_A, False),
        (URL_A, "local", False),
        (URL_A, URL_B, False),
    ],
)
def test_query_cache_is_keyed_on_backend(monkeypatch, calls, first, second, reused):
    _use(monkeypatch, first)
    embeddings.embed_query("same query")
    _use(monkeypatch, second)
    vec = embeddings.embed_query("same query")

    if reused:
        assert len(calls) == 1
    else:
        assert len(calls) == 2
        kind, url = calls[1]
        if second == "local":
            assert kind == "local" and vec[0] == -1.0
        else:
            assert kind == "api" and url.startswith(second) and vec[0] == 1.0


def test_via_api_flag_switches_backend(monkeypatch, calls):
    """embeddings_via_api routes the default local model to the API without a URL change."""
    embeddings.embed_query("same query")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    vec = embeddings.embed_query("same query")

    assert [kind for kind, _ in calls] == ["local", "api"]
    assert vec[0] == 1.0


@pytest.mark.parametrize(
    "flip_env, flip_value, restore_inside_load",
    [
        ("MNEMOSYNE_EMBEDDING_API_URL", URL_A, False),
        ("MNEMOSYNE_NO_EMBEDDINGS", "1", False),
        ("MNEMOSYNE_NO_EMBEDDINGS", "1", True),
    ],
    ids=["switch-to-api", "opt-out", "opt-out-undone-before-load-returns"],
)
def test_config_change_during_local_load_is_not_cached(
    monkeypatch, calls, flip_env, flip_value, restore_inside_load
):
    """embed_query picks the local key, then the config changes before the model loads.

    The loader then answers for the new config ("api" sentinel, or None when opted out),
    so this call has no local vector. That miss must not be cached under the local key,
    or the query keeps getting None after the config changes back.
    """
    # Opt-out only looks like a config change when fastembed could have loaded.
    monkeypatch.setattr(embeddings, "_is_fastembed_available", lambda: True)
    fake_local = embeddings._get_model

    def flip_then_load():
        monkeypatch.setenv(flip_env, flip_value)
        model = REAL_GET_MODEL()
        if restore_inside_load:
            monkeypatch.delenv(flip_env)
        return model

    monkeypatch.setattr(embeddings, "_get_model", flip_then_load)
    assert embeddings.embed_query("same query") is None

    monkeypatch.delenv(flip_env, raising=False)
    monkeypatch.setattr(embeddings, "_get_model", fake_local)
    vec = embeddings.embed_query("same query")

    assert vec is not None and vec[0] == -1.0
    assert calls == [("local", None)]


def test_missing_fastembed_bypasses_cache_and_recovers(monkeypatch, calls):
    """Missing imports bypass the cache; the same query can recover later."""
    monkeypatch.setattr(embeddings, "_is_fastembed_available", lambda: False)
    fake_local = embeddings._get_model
    loads = []

    def counting_load():
        loads.append(1)
        return fake_local()

    monkeypatch.setattr(embeddings, "_get_model", counting_load)

    assert embeddings.embed_query("same query") is None
    assert embeddings.embed_query("same query") is None
    assert loads == []
    assert embeddings._embed_query_cached.cache_info().currsize == 0

    monkeypatch.setattr(embeddings, "_is_fastembed_available", lambda: True)
    vec = embeddings.embed_query("same query")
    assert vec is not None and vec[0] == -1.0
    assert loads == [1]
    assert calls == [("local", None)]
