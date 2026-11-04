"""Explicit conflict-check provider options reach both supported transports."""
import json
import sys
from types import SimpleNamespace

import pytest

from mnemosyne.core import llm_conflict_detector as detector


@pytest.fixture(params=["httpx", "urllib"])
def captured_request(request, monkeypatch):
    monkeypatch.setattr(detector, "CONFLICT_LLM_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setattr(detector, "CONFLICT_LLM_MODEL", "gpt-test")
    monkeypatch.setattr(detector, "CONFLICT_LLM_API_KEY", "")
    monkeypatch.delenv("MNEMOSYNE_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_LLM_EXTRA_BODY", raising=False)
    captured = []
    response = {"choices": [{"message": {"content": "{}"}}]}

    class Transport:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, *, json, headers):
            captured.append(json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

        def open(self, req, *, timeout):
            captured.append(json.loads(req.data))
            return self

        def read(self):
            return json.dumps(response).encode()

    if request.param == "httpx":
        monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=Transport))
    else:
        import urllib.request
        monkeypatch.setitem(sys.modules, "httpx", None)
        monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Transport())

    def call():
        assert detector._call_conflict_llm_with_retry("probe") == ("{}", None, None)
        return captured[-1]

    return call


def test_conflict_default_remains_512(captured_request):
    assert captured_request()["max_tokens"] == 512


@pytest.mark.parametrize("value, expected", [("2048", 2048), (" 768 ", 768), ("", 512), ("bad", 512), ("0", 512), ("-1", 512), ("1.5", 512)])
def test_explicit_token_budget(captured_request, monkeypatch, value, expected):
    monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", value)
    assert captured_request()["max_tokens"] == expected


def test_provider_body_options_and_owned_fields(captured_request, monkeypatch):
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
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", value)
    body = captured_request()
    assert body["max_tokens"] == 512
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "response_format"}
    assert "secret-string" not in caplog.text
