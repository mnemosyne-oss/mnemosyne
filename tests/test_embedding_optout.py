"""Tests for the MNEMOSYNE_NO_EMBEDDINGS opt-out and rate-limit retry.

These cover the two CI failure modes reported in the 2026-06-17 CI run:
1. Hugging Face rate-limits (429) the embedding model download, which
   cascades into 48+ test failures across sleep/consolidation/recall.
2. Tests that exercise non-embedding code paths still try to download
   the model because only `available()` was checking the opt-out flag.

The opt-out flag must short-circuit `_get_model()` and the embed call
paths, returning None cleanly without raising.
"""
from __future__ import annotations

import os
import json
import urllib.error

import pytest

from mnemosyne.core import embeddings


@pytest.fixture(autouse=True)
def _clean_embedding_env(monkeypatch):
    """Make sure no embedding-related env var leaks between tests."""
    for key in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
        "MNEMOSYNE_EMBEDDING_API_URL",
        "MNEMOSYNE_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def test_is_disabled_default_false():
    """No env var set -> embeddings enabled."""
    assert embeddings._is_disabled() is False


def test_is_disabled_no_embeddings_flag(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings._is_disabled() is True


def test_is_disabled_skip_alias(monkeypatch):
    """MNEMOSYNE_SKIP_EMBEDDINGS is a shorter alias with the same intent."""
    monkeypatch.setenv("MNEMOSYNE_SKIP_EMBEDDINGS", "1")
    assert embeddings._is_disabled() is True


def test_is_disabled_off_alias(monkeypatch):
    """MNEMOSYNE_EMBEDDINGS_OFF is a longer alias with the same intent."""
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_OFF", "1")
    assert embeddings._is_disabled() is True


def test_available_returns_false_when_disabled(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings.available() is False


def test_get_model_returns_none_when_disabled(monkeypatch):
    """The opt-out must short-circuit _get_model without attempting
    a Hugging Face download. This is the core CI fix: with the env
    var set, a 429 cannot happen because the download is never
    attempted.
    """
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings._get_model() is None


def test_embed_query_returns_none_when_disabled(monkeypatch):
    """embed_query must return None cleanly when embeddings are off,
    not raise. Callers (recall, consolidation) check for None and
    degrade gracefully."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings.embed_query("hello world") is None


def test_embed_returns_none_when_disabled(monkeypatch):
    """embed() must return None cleanly when embeddings are off,
    matching the behavior callers expect from the disabled state."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings.embed(["a", "b", "c"]) is None
    assert embeddings.embed([]) is None  # Empty input is also None.


def test_get_model_raises_clear_error_on_load_failure(monkeypatch):
    """When the model download fails for a non-rate-limit reason,
    _get_model must raise a RuntimeError that names the model and
    the underlying cause, instead of silently returning None.

    Silent None returns previously masked the 429 cascade, so the
    error message has to be loud enough to be visible in the test
    failure log.
    """
    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS", raising=False)

    call_count = {"n": 0}

    def _fake_text_embedding(*args, **kwargs):
        call_count["n"] += 1
        raise ConnectionError("network unreachable: model host not found")

    monkeypatch.setattr(embeddings, "TextEmbedding", _fake_text_embedding)
    monkeypatch.setattr(embeddings, "_embedding_model", None)

    with pytest.raises(RuntimeError) as excinfo:
        embeddings._get_model()
    assert "BAAI/bge-small-en-v1.5" in str(excinfo.value)
    assert "network unreachable" in str(excinfo.value)
    assert call_count["n"] == 1  # Non-rate-limit errors fail fast.


def test_get_model_retries_on_rate_limit(monkeypatch):
    """A 429 from Hugging Face must trigger an exponential-backoff
    retry. A single 429 is recoverable; the test confirms at least
    one retry is attempted before giving up.
    """
    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS", raising=False)

    call_count = {"n": 0}

    def _fake_text_embedding(*args, **kwargs):
        call_count["n"] += 1
        # First attempt: rate limited. Second attempt: also rate limited.
        # We give up after 3 attempts.
        raise RuntimeError("HTTP 429 Too Many Requests")

    # Make the backoff sleep effectively zero so the test is fast.
    import time
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(embeddings, "TextEmbedding", _fake_text_embedding)
    monkeypatch.setattr(embeddings, "_embedding_model", None)

    with pytest.raises(RuntimeError) as excinfo:
        embeddings._get_model()
    assert "429" in str(excinfo.value) or "rate" in str(excinfo.value).lower()
    assert call_count["n"] == 3  # Retried up to 3 times before giving up.


def test_embed_api_retries_transient_http_503(monkeypatch):
    """Transient HTTP 5xx responses receive bounded retries."""
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://example.test/v1")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "text-embedding-test")
    monkeypatch.setattr(embeddings, "np", __import__("numpy"))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    calls = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"data": [{"embedding": [0.1, 0.2]}]}).encode()

    def _urlopen(_request, **_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.HTTPError(
                "https://example.test/v1/embeddings", 503, "unavailable", {}, None
            )
        return _Response()

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", _urlopen)
    result = embeddings._embed_api(["hello"])

    __import__("numpy").testing.assert_allclose(result, [[0.1, 0.2]])
    assert calls["count"] == 3


def test_embed_api_does_not_retry_permanent_http_400(monkeypatch):
    """Permanent client errors fail fast instead of hammering the endpoint."""
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://example.test/v1")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "text-embedding-test")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    calls = {"count": 0}

    def _urlopen(_request, **_kwargs):
        calls["count"] += 1
        raise urllib.error.HTTPError(
            "https://example.test/v1/embeddings", 400, "bad request", {}, None
        )

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", _urlopen)

    assert embeddings._embed_api(["hello"]) is None
    assert calls["count"] == 1
