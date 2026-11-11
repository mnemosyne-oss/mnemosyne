"""Explicit conflict-check provider options reach both supported transports."""
import json
import os
import subprocess
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
    monkeypatch.delenv("MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY", raising=False)
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
            if url.startswith("https://conflict.invalid/") and "thinking" in json:
                raise ValueError("Conflict provider rejects thinking")
            captured.append(json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

        def open(self, req, *, timeout):
            """Decode the actual bytes prepared by the urllib path."""
            body = json.loads(req.data)
            if req.full_url.startswith("https://conflict.invalid/") and "thinking" in body:
                raise ValueError("Conflict provider rejects thinking")
            captured.append(body)
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


@pytest.mark.parametrize("override", ["{}", "", "  "])
def test_conflict_body_can_suppress_shared_provider_options(
    captured_request, monkeypatch, override
):
    """A separate endpoint can opt out of incompatible shared provider fields."""
    monkeypatch.setattr(detector, "CONFLICT_LLM_BASE_URL", "https://conflict.invalid/v1")
    monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", "768")
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", '{"thinking":{"type":"disabled"},"max_tokens":4096}')
    monkeypatch.setenv("MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY", override)
    body = captured_request()
    assert "thinking" not in body
    assert body["max_tokens"] == 768


def test_conflict_body_replaces_shared_body_and_preserves_owned_fields(
    captured_request, monkeypatch
):
    """Select one body, filter protected fields, and re-read override removal."""
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", '{"shared_only":true,"max_tokens":4096}')
    monkeypatch.setenv("MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY", json.dumps({
        "temperature": 0.3, "max_tokens": 1024, "messages": [],
        "model": "other", "stream": True,
    }))
    body = captured_request()
    assert "shared_only" not in body
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 1024
    assert body["model"] == "gpt-test"
    assert body["messages"] == [{"role": "user", "content": "probe"}]
    assert "stream" not in body
    monkeypatch.delenv("MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY")
    inherited = captured_request()
    assert inherited["shared_only"] is True
    assert inherited["max_tokens"] == 4096


@pytest.mark.parametrize("override", ["{", "[]", "null", '"secret-override"',
    '{"probe":NaN}', '{"probe":{"nested":1e400}}'])
def test_invalid_conflict_body_does_not_fall_back_to_shared_options(
    captured_request, monkeypatch, caplog, override
):
    """Invalid overrides cannot leak incompatible shared settings or log values."""
    monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", '{"thinking":{"type":"disabled"}}')
    monkeypatch.setenv("MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY", override)
    body = captured_request()
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "response_format"}
    assert body["max_tokens"] == 512
    assert "MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY" in caplog.text
    assert "secret-override" not in caplog.text


@pytest.mark.parametrize("value", ["bad", "1.5", " "])
def test_token_fallback_does_not_mask_consolidation_import_error(value, tmp_path):
    """Fresh processes expose the documented boundary between the two parsers."""
    env = dict(os.environ, MNEMOSYNE_LLM_MAX_TOKENS=value,
               MNEMOSYNE_DATA_DIR=str(tmp_path), MNEMOSYNE_NO_EMBEDDINGS="1")
    env.pop("MNEMOSYNE_LLM_EXTRA_BODY", None)
    env.pop("MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY", None)
    result = subprocess.run([sys.executable, "-c", """
from mnemosyne.core.llm_conflict_detector import _conflict_request_options
assert _conflict_request_options()["max_tokens"] == 512
try:
    from mnemosyne.core import local_llm
except ValueError:
    print("consolidation-import-error")
else:
    raise AssertionError("Expected independent consolidation parser to reject value")
"""], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "consolidation-import-error"


def test_conflict_override_keeps_consolidation_body_and_default(tmp_path):
    """Conflict-only options leave the consolidation parser and budget independent."""
    env = dict(os.environ, MNEMOSYNE_DATA_DIR=str(tmp_path),
               MNEMOSYNE_LLM_EXTRA_BODY='{"thinking":{"type":"disabled"}}',
               MNEMOSYNE_CONFLICT_LLM_EXTRA_BODY='{}')
    env.pop("MNEMOSYNE_LLM_MAX_TOKENS", None)
    result = subprocess.run([sys.executable, "-c", """
from mnemosyne.core import local_llm
from mnemosyne.core.llm_conflict_detector import _conflict_request_options
assert local_llm.LLM_MAX_TOKENS == 2048
assert local_llm.LLM_EXTRA_BODY == {"thinking": {"type": "disabled"}}
assert _conflict_request_options() == {"max_tokens": 512}
"""], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
