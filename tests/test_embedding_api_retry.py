"""Regression coverage for transient embedding endpoint failures."""

import io
import json
import logging
import threading
import urllib.error
from unittest.mock import patch

import pytest

from mnemosyne.core import embeddings


@pytest.fixture(autouse=True)
def _uncredentialed_embedding_client(monkeypatch):
    """These tests drive `_embed_api` against plain-http endpoints; the client
    refuses credentialed non-HTTPS URLs, so blank the key regardless of the
    developer shell. Tests that need a key set it themselves AFTER this (on an
    https:// URL)."""
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")
    # These tests drive the env aliases. A config.yaml anywhere on the runner
    # (CI runs the suite with MNEMOSYNE_NO_EMBEDDINGS=1, and the first read
    # seeds that value into the file) would win over every env change below, so
    # neutralize the config resolver for this file instead of depending on the
    # machine having no config.
    monkeypatch.setattr(embeddings, "_cfg", lambda _key, default=None: default)
    monkeypatch.setattr(embeddings, "_cfg_bool", lambda _key, default=False: default)


@pytest.fixture
def _embedding_dispatch_enabled(monkeypatch):
    """The fail-loud tests exercise `embed()` end to end, but CI runs the suite
    with `MNEMOSYNE_NO_EMBEDDINGS=1`, and #1083 makes the opt-out short-circuit
    `embed()` before any dispatch. Clear the opt-out aliases so the tests reach
    the API path they are about."""
    for key in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(key, raising=False)


class Response:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body.read()


class StubOpener:
    """Stands in for the opener a key-bearing embed builds, so the
    credentialed recovery path can be exercised without network."""

    def __init__(self, response):
        self._response = response

    def open(self, req, timeout=None):
        return self._response


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

    assert "endpoint=https://example.test/v1/embeddings status=401" in caplog.text
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


# ---------------------------------------------------------------------------
# #735: embed()/embed_query() must fail loud on the API path instead of
# silently returning None. A bare None is indistinguishable from "embeddings
# unavailable", so BeamMemory.remember()'s `if vec is not None` would skip
# vector storage without its except-Exception warning ever firing.
# ---------------------------------------------------------------------------

def _api_env(monkeypatch, base_url="http://127.0.0.1:11435/v1"):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", base_url)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_SKIP_EMBEDDINGS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_OFF", raising=False)
    embeddings._embed_query_cached.cache_clear()


def test_embed_raises_when_api_request_fails_while_available(monkeypatch):
    _api_env(monkeypatch)
    error = urllib.error.URLError(OSError(65, "No route to host"))

    with patch("urllib.request.urlopen", side_effect=error), \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep"):
        with pytest.raises(RuntimeError, match="Embedding API returned no vectors") as excinfo:
            embeddings.embed(["private memory content"])

    assert "private memory content" not in str(excinfo.value)


def test_embed_raises_on_empty_data_list(monkeypatch):
    _api_env(monkeypatch)

    with patch("urllib.request.urlopen", return_value=Response({"data": []})):
        with pytest.raises(RuntimeError, match="empty vector result"):
            embeddings.embed(["anything"])


def test_embed_raises_on_zero_length_embedding_row(monkeypatch):
    # {"data": [{"embedding": []}]} yields a (1, 0) array: one response row
    # with zero elements. Checking len() (row count) alone would miss this;
    # result.size must be zero for the fail-loud guard to fire.
    _api_env(monkeypatch)

    with patch("urllib.request.urlopen", return_value=Response({"data": [{"embedding": []}]})):
        with pytest.raises(RuntimeError, match="empty vector result"):
            embeddings.embed(["anything"])


def test_embed_raises_on_scalar_embedding_row(monkeypatch):
    # A scalar "embedding" value yields a rank-1 array that looks non-empty;
    # the contract requires a rank-2 result with one vector per input.
    _api_env(monkeypatch)

    with patch("urllib.request.urlopen", return_value=Response({"data": [{"embedding": 0.5}]})):
        with pytest.raises(RuntimeError, match="unexpected rank-1 result"):
            embeddings.embed(["anything"])


def test_embed_raises_on_partial_response(monkeypatch):
    # Fewer vectors than requested inputs must not silently misalign the
    # caller's vector-to-text mapping.
    _api_env(monkeypatch)
    data = {"data": [
        {"embedding": [0.1, 0.2]},
        {"embedding": [0.3, 0.4]},
    ]}

    with patch("urllib.request.urlopen", return_value=Response(data)):
        with pytest.raises(RuntimeError, match=r"2 vector\(s\) for 3 input\(s\)"):
            embeddings.embed(["a", "b", "c"])


def test_embed_query_raises_on_partial_response(monkeypatch):
    # The query path requests exactly one vector; a multi-vector response must
    # be rejected before indexing result[0].
    _api_env(monkeypatch)
    data = {"data": [
        {"embedding": [0.1, 0.2]},
        {"embedding": [0.3, 0.4]},
    ]}

    with patch("urllib.request.urlopen", return_value=Response(data)):
        with pytest.raises(RuntimeError, match=r"2 vector\(s\) for 1 input\(s\)"):
            embeddings.embed_query("anything")

    embeddings._embed_query_cached.cache_clear()


def test_embed_query_raises_when_api_request_fails(monkeypatch):
    _api_env(monkeypatch)
    error = urllib.error.HTTPError("http://127.0.0.1:11435/v1/embeddings", 401, "unauthorized", {}, None)

    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(RuntimeError, match="Embedding API returned no vectors") as excinfo:
            embeddings.embed_query("private query text")

    assert "private query text" not in str(excinfo.value)
    embeddings._embed_query_cached.cache_clear()


def test_embed_raise_message_is_redacted_and_retries_transient_failures(
    monkeypatch, _embedding_dispatch_enabled
):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://user:password@example.test/v1?token=secret")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    embeddings._embed_query_cached.cache_clear()
    error = urllib.error.HTTPError("https://example.test/v1/embeddings", 503, "unavailable", {}, None)

    with patch("urllib.request.OpenerDirector.open", side_effect=error) as request, \
         patch("mnemosyne.core.embeddings.random.uniform", return_value=0), \
         patch("mnemosyne.core.embeddings.time.sleep"):
        with pytest.raises(RuntimeError) as excinfo:
            embeddings.embed(["secret content"])

    msg = str(excinfo.value)
    assert "endpoint=https://example.test/v1" in msg
    assert "model=" in msg
    assert "secret-key" not in msg
    assert "user:password" not in msg
    assert "token=secret" not in msg
    assert "secret content" not in msg
    assert request.call_count == 3


def test_embed_raise_message_names_missing_openrouter_key(
    monkeypatch, _embedding_dispatch_enabled
):
    # OpenRouter base + no API key is the previously-silent path: _embed_api
    # logged nothing and returned None. It must now log, and the public API
    # must name the missing key instead of a generic failure.
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")

    with pytest.raises(RuntimeError, match="no API key is configured") as excinfo:
        embeddings.embed(["content"])

    assert "openrouter.ai" in str(excinfo.value)


def test_embed_raises_redacted_runtime_error_on_malformed_endpoint_url(
    monkeypatch, _embedding_dispatch_enabled
):
    # urlsplit() raises ValueError on malformed URLs (e.g. "http://["); host
    # classification must not leak that as a raw ValueError -- the API path
    # should still fail loud with the redacted RuntimeError and never echo
    # the malformed URL. The message must name the configured model and the
    # redacted endpoint so a generic error cannot satisfy this regression test.
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://[")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "test-embedding-model")

    with patch("mnemosyne.core.embeddings.time.sleep"):
        with pytest.raises(RuntimeError, match="Embedding API returned no vectors") as excinfo:
            embeddings.embed(["content"])

    assert "http://[" not in str(excinfo.value)
    assert "<invalid-url>" in str(excinfo.value)
    assert "test-embedding-model" in str(excinfo.value)


def test_custom_endpoint_with_openrouter_ai_in_query_needs_no_key(
    monkeypatch, _embedding_dispatch_enabled
):
    # Substring-based OpenRouter detection would misclassify this custom
    # endpoint because its query contains "openrouter.ai", blocking it from
    # the keyless custom path. Hostname matching must let it through, and the
    # query string must survive route construction so the request hits
    # .../embeddings?upstream=openrouter.ai rather than a corrupted target.
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://proxy.example.com/v1?upstream=openrouter.ai")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")

    with patch("urllib.request.urlopen", return_value=Response({"data": [{"embedding": [0.1, 0.2]}]})) as request:
        result = embeddings.embed(["hello"])

    assert result is not None
    assert result.shape == (1, 2)
    req = request.call_args[0][0]
    assert req.full_url == "https://proxy.example.com/v1/embeddings?upstream=openrouter.ai"
    assert "Authorization" not in req.headers


def test_embed_api_no_key_path_logs_redacted_endpoint(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://user:password@openrouter.ai/api/v1?token=secret")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")

    with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
        assert embeddings._embed_api(["content"]) is None

    assert "no API key" in caplog.text
    assert "openrouter.ai" in caplog.text
    assert "user:password" not in caplog.text
    assert "token=secret" not in caplog.text


def test_embed_returns_none_on_non_api_path_when_model_unavailable(monkeypatch):
    # The fail-loud rule is scoped to the API branch only: when no API is
    # configured and the local model is unavailable, embed()/embed_query()
    # keep returning None so the "embeddings not available" degradation is
    # preserved.
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_URL", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_VIA_API", raising=False)
    embeddings._embed_query_cached.cache_clear()

    with patch.object(embeddings, "_get_model", return_value=None):
        assert embeddings.embed(["hello"]) is None
        assert embeddings.embed_query("hello") is None

    embeddings._embed_query_cached.cache_clear()


# ---------------------------------------------------------------------------
# #735 fail-soft consumers: when embed() raises, the best-effort call sites
# must degrade gracefully instead of aborting the memory write / import.
# ---------------------------------------------------------------------------

def _raise_embedding_error(texts):
    raise RuntimeError("endpoint down")


def test_shmr_embed_degrades_to_zero_vector_when_embedding_raises(monkeypatch):
    from mnemosyne.core import shmr

    monkeypatch.setattr(shmr._embeddings, "available", lambda: True)
    monkeypatch.setattr(shmr._embeddings, "embed", _raise_embedding_error)

    vec = shmr._embed("content")

    assert vec.shape == (shmr.EMBEDDING_DIM,)
    assert not vec.any()


def test_legacy_mnemosyne_remember_persists_without_vector_when_embedding_raises(monkeypatch, tmp_path):
    import sqlite3

    from mnemosyne.core import embeddings as emb
    from mnemosyne.core.memory import Mnemosyne

    monkeypatch.setattr(emb, "available", lambda: True)
    monkeypatch.setattr(emb, "embed", _raise_embedding_error)

    memory = Mnemosyne(session_id="legacy-735", db_path=tmp_path / "memory.db")
    memory_id = memory.remember("best effort memory", source="test")

    conn = sqlite3.connect(tmp_path / "memory.db")
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM memories WHERE id=?", (memory_id,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id=?", (memory_id,)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_hindsight_import_continues_without_vector_when_embedding_raises(monkeypatch, tmp_path):
    import json
    import sqlite3

    from mnemosyne.core import embeddings as emb
    from mnemosyne.core.importers import HindsightImporter
    from mnemosyne.core.memory import Mnemosyne

    export = tmp_path / "hs-export.json"
    export.write_text(json.dumps({"items": [{
        "id": "hs-735-1",
        "text": "memory that must still import",
        "fact_type": "world",
        "mentioned_at": "2026-04-29T01:36:00+00:00",
        "date": "2026-04-29",
        "proof_count": 1,
    }]}), encoding="utf-8")

    monkeypatch.setattr(emb, "available", lambda: True)
    monkeypatch.setattr(emb, "embed", _raise_embedding_error)

    db_path = tmp_path / "hs.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    result = HindsightImporter(file_path=str(export), bank="hermes", generate_embeddings=True).run(mem)

    assert result.imported == 1
    assert result.failed == 0

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# #735 hardening (review round): policy-error redaction, non-finite API
# vectors, and stale derived vectors left by a failed update_working() refresh.
# ---------------------------------------------------------------------------

def test_credentialed_cleartext_policy_error_redacts_url(monkeypatch, caplog):
    # EmbeddingPolicyError interpolated the raw endpoint URL, so userinfo and
    # query secrets could reach the exception text and the fail-soft consumer
    # logs. Both must carry only the redacted endpoint.
    monkeypatch.setenv(
        "MNEMOSYNE_EMBEDDING_API_URL",
        "http://user:password@example.test/v1?token=secret",
    )
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_LAST_POLICY_REFUSAL", None)

    def _no_request(*args, **kwargs):
        raise AssertionError("no request may be attempted for a cleartext credentialed endpoint")

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", _no_request)
    with caplog.at_level(logging.WARNING, logger="mnemosyne.core.embeddings"):
        with pytest.raises(embeddings.EmbeddingPolicyError) as excinfo:
            embeddings._embed_api(["private memory content"])

    message = str(excinfo.value)
    assert "non-HTTPS" in message
    assert "http://example.test/v1" in message
    for leaked in ("user:password", "token=secret", "secret-key", "private memory content"):
        assert leaked not in message
        assert leaked not in caplog.text


def test_credentialed_redirect_policy_error_redacts_target_url(monkeypatch):
    # The redirect refusal echoed the raw Location target, which can carry
    # userinfo or query secrets. It must be redacted before formatting.
    import urllib.request

    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_LAST_POLICY_REFUSAL", None)
    handler = embeddings._CredentialedNoRedirect()
    request = urllib.request.Request(
        "https://configured.example/v1/embeddings",
        headers={"Authorization": "Bearer secret"},
    )
    newurl = "http://user:password@attacker.example/embed?token=secret"

    with pytest.raises(embeddings.EmbeddingPolicyError) as excinfo:
        handler.redirect_request(request, None, 302, "Found", {"Location": newurl}, newurl)

    message = str(excinfo.value)
    assert "redirect" in message
    assert "http://attacker.example/embed" in message
    assert "user:password" not in message
    assert "token=secret" not in message


def test_policy_refusal_announces_loudly_once(monkeypatch, caplog):
    # The fail-loud contract must survive the broad except blocks up the
    # stack (beam et al. degrade any embedding failure to keyword-only
    # recall): the refusal logs at ERROR once per refusal kind per process,
    # then still raises for callers that listen; the status surface keeps
    # the latest refused endpoint.
    _api_env(monkeypatch, base_url="http://example.test/v1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_LAST_POLICY_REFUSAL", None)

    with caplog.at_level(logging.ERROR, logger="mnemosyne.core.embeddings"):
        with pytest.raises(embeddings.EmbeddingPolicyError):
            embeddings._embed_api(["private memory content"])
        # A different cleartext endpoint is the same refusal KIND: the
        # announcement stays once, while the status keeps the latest message.
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://other.example/v1")
        with pytest.raises(embeddings.EmbeddingPolicyError):
            embeddings._embed_api(["private memory content"])

    assert "other.example" in embeddings.policy_refusal_message()
    assert caplog.text.count("policy refusal") == 1


def test_policy_refusal_state_resets_after_successful_embed(monkeypatch):
    # _embed_api re-reads the environment per call, so a configuration fixed
    # mid-process must stop reporting the stale refusal on the status surface.
    # The recovery runs credentialed (the realistic shape: the key stays and
    # the URL becomes https), through the opener path a key-bearing embed
    # actually takes; a garbage response must NOT clear the refusal, only a
    # validated one may.
    _api_env(monkeypatch, base_url="http://example.test/v1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_LAST_POLICY_REFUSAL", None)

    with pytest.raises(embeddings.EmbeddingPolicyError):
        embeddings._embed_api(["anything"])
    assert embeddings.policy_refusal_message() is not None

    _api_env(monkeypatch, base_url="https://example.test/v1")

    garbage = {"data": [{"embedding": [float("nan")]}]}
    with patch("urllib.request.build_opener", return_value=StubOpener(Response(garbage))):
        with pytest.raises(RuntimeError, match="non-finite"):
            embeddings.embed(["anything"])
    # The cleartext kind is superseded by the healthy static gate, so the
    # preserved refusal is asserted on the recorded state itself.
    assert embeddings._LAST_POLICY_REFUSAL is not None

    data = {"data": [{"embedding": [0.1, 0.2]}]}
    with patch("urllib.request.build_opener", return_value=StubOpener(Response(data))):
        vectors = embeddings.embed(["anything"])

    assert vectors is not None and vectors.shape[0] == 1
    assert embeddings._LAST_POLICY_REFUSAL is None
    assert embeddings.policy_refusal_message() is None


def test_policy_refusal_failed_emit_releases_the_claim(monkeypatch):
    # A failed announcement must not consume the refusal kind: the next
    # refusal of the same kind retries the ERROR log instead of staying
    # silent for the process lifetime.
    _api_env(monkeypatch, base_url="http://example.test/v1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_LAST_POLICY_REFUSAL", None)
    attempts = []

    def _flaky_error(fmt, *args):
        attempts.append("attempt")
        if len(attempts) == 1:
            raise OSError("handler broke")
        attempts[-1] = "ok"

    monkeypatch.setattr(embeddings.logger, "error", _flaky_error)
    with pytest.raises(embeddings.EmbeddingPolicyError):
        embeddings._embed_api(["x"])
    with pytest.raises(embeddings.EmbeddingPolicyError):
        embeddings._embed_api(["x"])

    assert attempts == ["attempt", "ok"]


def test_policy_refusal_derived_statically_without_embed_call(monkeypatch):
    # doctor and diagnose never embed, so the refusal must be derivable from
    # configuration alone for their status surfaces to report it; the private
    # name from released tags keeps working as an alias of the public one.
    _api_env(monkeypatch, base_url="http://example.test/v1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_LAST_POLICY_REFUSAL", None)

    message = embeddings.policy_refusal_message()
    assert message is not None
    assert "http://example.test/v1" in message
    assert embeddings._EmbeddingPolicyError is embeddings.EmbeddingPolicyError

    _api_env(monkeypatch, base_url="https://example.test/v1")
    assert embeddings.policy_refusal_message() is None


def test_policy_refusal_snapshot_survives_concurrent_clears(monkeypatch):
    # Regression (review 5480062367): the None check and the tuple unpack
    # used two reads of _LAST_POLICY_REFUSAL, so a concurrent validated
    # embed clearing the refusal in between raised TypeError inside a
    # status read, breaking recall(..., explain=True) with it.
    _api_env(monkeypatch, base_url="https://example.test/v1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "openai/text-embedding-3-small")
    # Seeded through monkeypatch so teardown restores the pre-test value
    # even though the churner thread writes the global directly.
    monkeypatch.setattr(
        embeddings, "_LAST_POLICY_REFUSAL", ("credentialed-redirect", "redirect to attacker.example")
    )

    stop = threading.Event()

    def _churn():
        while not stop.is_set():
            embeddings._LAST_POLICY_REFUSAL = None
            embeddings._LAST_POLICY_REFUSAL = ("credentialed-redirect", "redirect to attacker.example")

    churner = threading.Thread(target=_churn)
    churner.start()
    try:
        for _ in range(500):
            message = embeddings.policy_refusal_message()
            assert message is None or "attacker.example" in message
    finally:
        stop.set()
        churner.join()


def test_policy_refusal_silent_when_embeddings_disabled(monkeypatch):
    # Regression (review 5480062367): with embeddings disabled there is no
    # active embedding route, so a stale redirect refusal must stop being
    # reported as ERROR; a malformed flag value keeps the no-throw contract.
    _api_env(monkeypatch, base_url="https://example.test/v1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setattr(
        embeddings, "_LAST_POLICY_REFUSAL", ("credentialed-redirect", "redirect to attacker.example")
    )

    assert "attacker.example" in embeddings.policy_refusal_message()

    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings.policy_refusal_message() is None

    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "maybe")
    assert embeddings.policy_refusal_message() is None

    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS", raising=False)
    assert "attacker.example" in embeddings.policy_refusal_message()


def test_policy_refusal_reports_redirect_kind_while_api_route_active(monkeypatch):
    # A credentialed-redirect refusal has no static signal: it is reported
    # while the API route stays active, cleared by a validated embed, and
    # dropped entirely once the configuration can no longer refuse anything.
    # The refusal is raised through the real handler, not by seeding state,
    # so the test fails if the raise funnel stops recording the kind.
    _api_env(monkeypatch, base_url="https://example.test/v1")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "secret-key")
    monkeypatch.setattr(embeddings, "_ANNOUNCED_POLICY_REFUSALS", set())
    monkeypatch.setattr(embeddings, "_LAST_POLICY_REFUSAL", None)
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "openai/text-embedding-3-small")

    import urllib.request

    handler = embeddings._CredentialedNoRedirect()
    request = urllib.request.Request(
        "https://example.test/v1/embeddings",
        headers={"Authorization": "Bearer secret"},
    )

    def _refuse_redirect():
        with pytest.raises(embeddings.EmbeddingPolicyError):
            handler.redirect_request(
                request, None, 302, "Found", {}, "https://attacker.example/embed"
            )

    _refuse_redirect()
    assert "attacker.example" in embeddings.policy_refusal_message()

    # An invalid response must NOT clear the recorded refusal; the redirect
    # kind is what the accessor reports, so both sides of the contract are
    # asserted through it. The credentialed opener path is stubbed, matching
    # the key-bearing shape that raised the refusal in the first place.
    garbage = {"data": [{"embedding": [float("nan")]}]}
    with patch("urllib.request.build_opener", return_value=StubOpener(Response(garbage))):
        with pytest.raises(RuntimeError, match="non-finite"):
            embeddings.embed(["anything"])
    assert "attacker.example" in embeddings.policy_refusal_message()

    data = {"data": [{"embedding": [0.1, 0.2]}]}
    with patch("urllib.request.build_opener", return_value=StubOpener(Response(data))):
        vectors = embeddings.embed(["anything"])
    assert vectors is not None and vectors.shape[0] == 1
    assert embeddings._LAST_POLICY_REFUSAL is None
    assert embeddings.policy_refusal_message() is None

    _refuse_redirect()
    assert "attacker.example" in embeddings.policy_refusal_message()

    # Leaving the API route (local model, opt-in flag and URL dropped) drops
    # the redirect report: the stale refusal must not be reported against a
    # configuration that can no longer refuse it.
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_VIA_API", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_URL", raising=False)
    assert embeddings.policy_refusal_message() is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_embed_raises_on_non_finite_api_vector(monkeypatch, value):
    # A NaN/infinity response passes the shape and count checks but serializes
    # to invalid JSON for SQLite (json_valid() = 0) and yields non-finite
    # similarity scores. Reject it before it can reach storage.
    _api_env(monkeypatch)
    data = {"data": [{"embedding": [0.1, value]}]}

    with patch("urllib.request.urlopen", return_value=Response(data)):
        with pytest.raises(RuntimeError, match="non-finite"):
            embeddings.embed(["anything"])


def _beam_vector():
    np = pytest.importorskip("numpy")
    from mnemosyne.core import beam as beam_module

    vector = np.array([1.0] + [0.0] * (beam_module.EMBEDDING_DIM - 1), dtype=np.float32)
    return beam_module, vector


def _working_embedding_count(beam, memory_id):
    return beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]


def _seed_working_vector(beam, beam_module, monkeypatch, vector):
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam_module._embeddings, "embed", lambda contents: [vector])
    memory_id = beam.remember("original content", source="test")
    assert _working_embedding_count(beam, memory_id) == 1
    return memory_id


def test_update_working_invalidates_stale_vector_when_embedding_raises(monkeypatch, tmp_path):
    beam_module, vector = _beam_vector()
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="stale-vector-735", db_path=tmp_path / "beam.db")
    memory_id = _seed_working_vector(beam, beam_module, monkeypatch, vector)

    def _raise(_contents):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(beam_module._embeddings, "embed", _raise)

    assert beam.update_working(memory_id, content="replacement content") is True
    # The stored vector described the previous content; it must not survive to
    # score the new content in dense recall.
    assert _working_embedding_count(beam, memory_id) == 0


def test_update_working_invalidates_stale_vector_when_embed_returns_none(monkeypatch, tmp_path):
    beam_module, vector = _beam_vector()
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="stale-vector-none-735", db_path=tmp_path / "beam.db")
    memory_id = _seed_working_vector(beam, beam_module, monkeypatch, vector)

    monkeypatch.setattr(beam_module._embeddings, "embed", lambda contents: None)

    assert beam.update_working(memory_id, content="replacement content") is True
    assert _working_embedding_count(beam, memory_id) == 0


def test_update_working_invalidates_stale_vector_when_embeddings_unavailable(monkeypatch, tmp_path):
    beam_module, vector = _beam_vector()
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="stale-vector-off-735", db_path=tmp_path / "beam.db")
    memory_id = _seed_working_vector(beam, beam_module, monkeypatch, vector)

    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)

    assert beam.update_working(memory_id, content="replacement content") is True
    assert _working_embedding_count(beam, memory_id) == 0


def test_update_working_without_content_change_keeps_vector(monkeypatch, tmp_path):
    beam_module, vector = _beam_vector()
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="stale-vector-keep-735", db_path=tmp_path / "beam.db")
    memory_id = _seed_working_vector(beam, beam_module, monkeypatch, vector)

    def _raise(_contents):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(beam_module._embeddings, "embed", _raise)

    assert beam.update_working(memory_id, importance=0.9) is True
    assert _working_embedding_count(beam, memory_id) == 1
