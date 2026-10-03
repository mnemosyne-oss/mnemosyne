"""MNEMOSYNE_EMBEDDING_MAX_CHARS: opt-in cap on API embedding inputs.

Covers the API payload (default off, cap applied when set), operator
observability (truncation warnings), and disable/invalid-value handling.
"""

import io
import json

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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")


def _capture_payload(monkeypatch):
    """Record each request payload; answer one distinguishable vector per input, in order."""
    payloads = []

    def fake_urlopen(request, *_args, **_kwargs):
        payload = json.loads(request.data.decode())
        payloads.append(payload)
        return Response(
            {
                "data": [
                    {"embedding": [float(i + 1)]} for i in range(len(payload["input"]))
                ]
            }
        )

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", fake_urlopen)
    return payloads


def test_default_sends_full_text(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    payloads = _capture_payload(monkeypatch)
    long_text = "x" * 20000

    assert embeddings._embed_api([long_text]) is not None
    assert payloads[0]["input"] == [long_text]


def test_cap_limits_payload_but_not_returned_texts(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "100")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    payloads = _capture_payload(monkeypatch)

    assert embeddings._cap_for_api(["y" * 50, "short"]) == ["y" * 50, "short"]
    assert embeddings._cap_for_api(["y" * 150, "short"]) == ["y" * 100, "short"]
    embeddings._embed_api(["y" * 150])
    assert payloads[0]["input"] == ["y" * 100]


def test_truncation_is_logged_with_lengths(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "100")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    _capture_payload(monkeypatch)

    with caplog.at_level("WARNING", logger="mnemosyne.core.embeddings"):
        embeddings._embed_api(["z" * 300])

    assert any(
        "embedding input truncated: 300 -> 100 chars" in rec.getMessage()
        for rec in caplog.records
    )


def test_zero_and_blank_disable_the_cap(monkeypatch):
    text = "a" * 500

    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "0")
    assert embeddings._cap_for_api([text]) == [text]

    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "  ")
    assert embeddings._cap_for_api([text]) == [text]


def test_invalid_value_warns_and_disables(monkeypatch, caplog):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "huge")
    text = "b" * 500

    with caplog.at_level("WARNING", logger="mnemosyne.core.embeddings"):
        assert embeddings._cap_for_api([text]) == [text]

    assert any(
        "invalid MNEMOSYNE_EMBEDDING_MAX_CHARS" in rec.getMessage()
        for rec in caplog.records
    )


def test_oversized_row_does_not_abort_batch_when_capped(monkeypatch):
    """The llama.cpp wedge: a huge working-memory row embedded alongside
    normal rows used to HTTP 400 the whole request. With the cap set the
    payload stays under the limit and the batch succeeds."""
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "100")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    payloads = _capture_payload(monkeypatch)

    vectors = embeddings._embed_api(["c" * 300000, "normal text"])

    assert vectors is not None
    assert len(payloads[0]["input"]) == 2
    assert len(payloads[0]["input"][0]) <= 100
    assert payloads[0]["input"][1] == "normal text"
    # one vector per input, in input order
    assert [float(v[0]) for v in vectors] == [1.0, 2.0]


def test_query_cache_follows_cap_changes(monkeypatch):
    """Review #1052: embed_query is cached by its prefixed text, while the cap is read at call time.
    A query embedded uncapped must not be served from that cache once a cap is set."""
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", raising=False)
    # CI runs with embeddings opted out; embed_query honors that before its cache.
    for flag in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(flag, raising=False)
    payloads = _capture_payload(monkeypatch)
    embeddings._embed_query_cached.cache_clear()

    embeddings.embed_query("abcdef")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "3")
    embeddings.embed_query("abcdef")

    assert [p["input"] for p in payloads] == [["abcdef"], ["abc"]]
    embeddings._embed_query_cached.cache_clear()


def test_byte_cap_cuts_on_a_code_point_boundary(monkeypatch):
    """MNEMOSYNE_EMBEDDING_MAX_BYTES bounds UTF-8 bytes, which bound tokens from above for byte-level BPE; a cut never
    leaves half a character."""
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", "5")
    assert embeddings._cap_for_api(["я" * 10, "abc"]) == ["яя", "abc"]
    assert embeddings._cap_for_api(["𝔘𝔘"]) == ["𝔘"]  # 4-byte characters


def test_byte_cap_is_off_by_default_and_combines_with_the_char_cap(monkeypatch, caplog):
    text = "ж" * 100
    assert embeddings._cap_for_api([text]) == [text]
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", "40")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", "50")
    with caplog.at_level("WARNING", logger="mnemosyne.core.embeddings"):
        assert embeddings._cap_for_api([text]) == ["ж" * 25]  # 40 chars = 80 bytes > 50
    assert any(
        "embedding input truncated: 80 -> 50 bytes" in rec.getMessage()
        for rec in caplog.records
    )


def test_query_cache_follows_byte_cap_changes(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", raising=False)
    for flag in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(flag, raising=False)
    payloads = _capture_payload(monkeypatch)
    embeddings._embed_query_cached.cache_clear()

    embeddings.embed_query("щщщщ")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MAX_BYTES", "4")
    embeddings.embed_query("щщщщ")

    assert [p["input"] for p in payloads] == [["щщщщ"], ["щщ"]]
    embeddings._embed_query_cached.cache_clear()
