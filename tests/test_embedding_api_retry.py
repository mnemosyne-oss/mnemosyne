"""Regression coverage for transient embedding endpoint failures."""

import io
import json
import logging
import urllib.error
from unittest.mock import patch

import pytest

from mnemosyne.core import embeddings
from mnemosyne.core.config import MnemosyneConfig


@pytest.fixture(autouse=True)
def _uncredentialed_embedding_client(monkeypatch, tmp_path):
    """These tests drive `_embed_api` against plain-http endpoints; the client
    refuses credentialed non-HTTPS URLs, so blank the key regardless of the
    developer shell. Tests that need a key set it themselves AFTER this (on an
    https:// URL). The config is isolated too: resolution precedence is
    config.yaml > env, so an ambient config would shadow the URLs under test.
    The import-time snapshots are re-resolved after the reset, and the
    credentialed opener is routed through `urlopen`, so tests mock a single
    transport seam (redirect refusal keeps dedicated tests elsewhere).
    """
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg_dir = tmp_path / "mnemosyne-data"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text("")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(cfg_dir))
    MnemosyneConfig.reset_instance()
    monkeypatch.setattr(
        embeddings, "_DEFAULT_MODEL", embeddings._resolve_default_model()
    )
    monkeypatch.setattr(
        embeddings, "_OPENAI_API_KEY", embeddings._resolve_api_key()
    )
    monkeypatch.setattr(
        embeddings, "_OPENAI_BASE_URL", embeddings._resolve_api_base_url()
    )

    class _UrlopenOpener:
        def open(self, req, timeout=None):
            return embeddings.urllib.request.urlopen(req, timeout=timeout)

    monkeypatch.setattr(
        embeddings.urllib.request, "build_opener",
        lambda *handlers: _UrlopenOpener(),
    )


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


def test_embed_api_stops_after_three_transient_attempts(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    error = urllib.error.URLError(OSError(65, "No route to host"))

    with patch("urllib.request.urlopen", side_effect=error) as request, \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep") as sleep:
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
            assert embeddings._embed_api(["offline"]) is None

    assert request.call_count == 3
    assert sleep.call_count == 2
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "embedding API request failed" in caplog.text
    assert "error=URLError" in caplog.text


def test_embed_api_logs_final_client_error_without_input_or_credentials(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://user:password@example.test/v1?token=secret")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_KEY", "secret-key")
    error = urllib.error.HTTPError("https://example.test/v1/embeddings", 401, "unauthorized", {}, None)

    with patch("urllib.request.urlopen", side_effect=error):
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
            assert embeddings._embed_api(["private memory content"]) is None

    assert "endpoint=https://example.test/v1 status=401" in caplog.text
    assert "private memory content" not in caplog.text
    assert "secret-key" not in caplog.text
    assert "user:password" not in caplog.text
    assert "token=secret" not in caplog.text


def test_embed_api_logs_invalid_response_schema_without_input(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")

    with patch("urllib.request.urlopen", return_value=Response({"unexpected": []})):
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
            assert embeddings._embed_api(["private memory content"]) is None

    assert "embedding API call failed" in caplog.text
    assert "KeyError" in caplog.text
    assert "private memory content" not in caplog.text


def test_embed_api_logs_final_transient_http_error(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    error = urllib.error.HTTPError("http://127.0.0.1:11435/v1/embeddings", 503, "unavailable", {}, None)

    with patch("urllib.request.urlopen", side_effect=error) as request, \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep") as sleep:
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
            assert embeddings._embed_api(["private memory content"]) is None

    assert request.call_count == 3
    assert sleep.call_count == 2
    assert "endpoint=http://127.0.0.1:11435/v1/embeddings status=503" in caplog.text
    assert "private memory content" not in caplog.text


def test_embed_api_logs_final_rate_limit_error(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    error = urllib.error.HTTPError("http://127.0.0.1:11435/v1/embeddings", 429, "rate limited", {}, None)

    with patch("urllib.request.urlopen", side_effect=error) as request, \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep") as sleep:
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
            assert embeddings._embed_api(["private memory content"]) is None

    assert request.call_count == 3
    assert sleep.call_count == 2
    assert "endpoint=http://127.0.0.1:11435/v1/embeddings status=429" in caplog.text


def test_embed_api_invalid_port_stays_fail_soft(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://example.test:port/v1")

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")), \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep"):
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
            assert embeddings._embed_api(["private memory content"]) is None

    assert "embedding API request failed" in caplog.text
    assert "private memory content" not in caplog.text


def test_safe_api_endpoint_redacts_userinfo_and_preserves_ipv6_port():
    assert embeddings._safe_api_endpoint(
        "http://user:pw@[::1]:11435/v1?token=secret#access_token=secret"
    ) == "http://[::1]:11435/v1"


def test_safe_api_endpoint_handles_unbalanced_ipv6_bracket():
    assert embeddings._safe_api_endpoint("http://[::1/v1") == "<invalid-url>"


def test_safe_api_endpoint_rejects_schemeless_credential_text():
    assert embeddings._safe_api_endpoint("user:pw@example.test/v1") == "<invalid-url>"
