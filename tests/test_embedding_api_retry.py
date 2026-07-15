"""Regression coverage for transient embedding endpoint failures."""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from mnemosyne.core import embeddings


class Response:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body.read()


def test_embed_api_retries_transient_network_failures(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    result = Response({"data": [{"embedding": [0.25, 0.75]}]})
    failures = [urllib.error.URLError(OSError(65, "No route to host")), TimeoutError(), result]

    with patch("urllib.request.urlopen", side_effect=failures) as request, \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0.1), \
         patch("mnemosyne.core.embeddings.time.sleep") as sleep:
        vectors = embeddings._embed_api(["retry me"])

    assert vectors.tolist() == [[0.25, 0.75]]
    assert request.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.6, 1.1]


@pytest.mark.parametrize("status", [429, 503])
def test_embed_api_retries_transient_http_errors(monkeypatch, status):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    error = urllib.error.HTTPError("http://example", status, "transient", {}, None)
    result = Response({"data": [{"embedding": [0.25, 0.75]}]})

    with patch("urllib.request.urlopen", side_effect=[error, result]) as request, \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep") as sleep:
        vectors = embeddings._embed_api(["retry me"])

    assert vectors.tolist() == [[0.25, 0.75]]
    assert request.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_embed_api_does_not_retry_nontransient_client_error(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    error = urllib.error.HTTPError("http://example", 400, "bad request", {}, None)

    with patch("urllib.request.urlopen", side_effect=error) as request, \
         patch("mnemosyne.core.embeddings.time.sleep") as sleep:
        assert embeddings._embed_api(["bad request"]) is None

    assert request.call_count == 1
    sleep.assert_not_called()


def test_embed_api_stops_after_three_transient_attempts(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    error = urllib.error.URLError(OSError(65, "No route to host"))

    with patch("urllib.request.urlopen", side_effect=error) as request, \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep") as sleep:
        assert embeddings._embed_api(["offline"]) is None

    assert request.call_count == 3
    assert sleep.call_count == 2
