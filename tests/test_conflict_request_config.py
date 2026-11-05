"""Explicit conflict-check provider options reach both supported transports."""
import json
import sys
from types import SimpleNamespace

import pytest

from mnemosyne.core import llm_conflict_detector as detector


@pytest.fixture(params=["httpx", "urllib"])
def captured_request(request, monkeypatch):
    """Capture complete conflict payloads through each supported transport."""
    monkeypatch.setattr(detector, "CONFLICT_LLM_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setattr(detector, "CONFLICT_LLM_MODEL", "gpt-test")
    monkeypatch.setattr(detector, "CONFLICT_LLM_API_KEY", "")
    monkeypatch.delenv("MNEMOSYNE_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_LLM_EXTRA_BODY", raising=False)
    captured = []
    response = {"choices": [{"message": {"content": "{}"}}]}

    class Transport:
        def __init__(self, **kwargs):
            """Accept the client's timeout and redirect settings without I/O."""
            pass

        def __enter__(self):
            """Expose the fake client/response context."""
            return self

        def __exit__(self, *args):
            """Keep test failures visible outside the fake context."""
            pass

        def post(self, url, *, json, headers):
            """Capture httpx's structured request body."""
            captured.append(json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

        def open(self, req, *, timeout):
            """Decode the actual bytes prepared by the urllib path."""
            captured.append(json.loads(req.data))
            return self

        def read(self):
            """Return a minimal successful completion response."""
            return json.dumps(response).encode()

    if request.param == "httpx":
        monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=Transport))
    else:
        import urllib.request
        monkeypatch.setitem(sys.modules, "httpx", None)
        monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Transport())

    def call():
        """Require a successful request and return its captured payload."""
        assert detector._call_conflict_llm_with_retry("probe") == ("{}", None, None)
        return captured[-1]

    return call


def test_conflict_default_remains_512(captured_request):
    """Unconfigured conflict requests retain their historical budget."""
    assert captured_request()["max_tokens"] == 512


@pytest.mark.parametrize("value, expected", [("2048", 2048), (" 768 ", 768), ("", 512), ("bad", 512), ("0", 512), ("-1", 512), ("1.5", 512)])
def test_explicit_token_budget(captured_request, monkeypatch, value, expected):
    """Accept positive explicit budgets and safely fall back for unusable ones."""
    monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", value)
    assert captured_request()["max_tokens"] == expected


def test_provider_body_options_and_owned_fields(captured_request, monkeypatch):
    """Merge provider fields last without handing over request-owned fields."""
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", json.dumps({
        "thinking": {"type": "disabled"}, "max_tokens": 1024,
        "messages": [], "model": "other", "stream": True,
    }))
    body = captured_request()
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 1024
    assert body["messages"] == [{"role": "user", "content": "probe"}]
    assert body["model"] == "gpt-test"
    assert "stream" not in body


@pytest.mark.parametrize("value", ["", "{", "[]", "null", "true", '"secret-string"'])
def test_invalid_extra_body_is_ignored(captured_request, monkeypatch, value, caplog):
    """Malformed or non-object bodies do not change requests or leak values."""
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", value)
    body = captured_request()
    assert body["max_tokens"] == 512
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "response_format"}
    assert "secret-string" not in caplog.text


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
@pytest.mark.parametrize("location", ["direct", "object", "array"])
def test_nonfinite_extra_body_is_ignored_entirely(
    captured_request, monkeypatch, caplog, number, location
):
    """Reject nonfinite numbers at any depth before either transport sees them."""
    value = number
    if location == "object":
        value = '{"nested":' + number + '}'
    elif location == "array":
        value = '[{"nested":' + number + '}]'
    monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", "768")
    monkeypatch.setenv(
        "MNEMOSYNE_LLM_EXTRA_BODY",
        '{"max_tokens":4096,"probe":' + value + ',"private":"secret-body-value"}',
    )
    body = captured_request()
    assert body["max_tokens"] == 768
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "response_format"}
    json.dumps(body, allow_nan=False)
    assert "invalid MNEMOSYNE_LLM_EXTRA_BODY" in caplog.text
    assert "secret-body-value" not in caplog.text


def test_finite_extra_body_values_are_preserved(captured_request, monkeypatch):
    """Strict number handling preserves finite floats, integers and ordinary strings."""
    monkeypatch.setenv(
        "MNEMOSYNE_LLM_EXTRA_BODY",
        '{"probe":{"fraction":1.25,"large":1e308,"integer":9007199254740993,'
        '"literal":"NaN","items":[true,null,-0.0]}}',
    )
    body = captured_request()
    assert body["probe"] == {
        "fraction": 1.25, "large": 1e308, "integer": 9007199254740993,
        "literal": "NaN", "items": [True, None, -0.0],
    }
    json.dumps(body, allow_nan=False)


def test_configuration_changes_between_requests(captured_request, monkeypatch):
    """Re-read both variables for each request, including their later removal."""
    monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", '{"temperature":0.2,"thinking":{"type":"enabled"}}')
    first = captured_request()
    assert first["max_tokens"] == 1024
    assert first["temperature"] == 0.2
    assert first["thinking"] == {"type": "enabled"}

    monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", '{"temperature":0.0,"thinking":{"type":"disabled"}}')
    second = captured_request()
    assert second["max_tokens"] == 2048
    assert second["temperature"] == 0.0
    assert second["thinking"] == {"type": "disabled"}

    monkeypatch.delenv("MNEMOSYNE_LLM_MAX_TOKENS")
    monkeypatch.delenv("MNEMOSYNE_LLM_EXTRA_BODY")
    third = captured_request()
    assert third["max_tokens"] == 512
    assert third["temperature"] == 0.0
    assert "thinking" not in third
