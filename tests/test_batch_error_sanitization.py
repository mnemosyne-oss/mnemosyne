"""Regression tests for issue #832 / B15 batch error boundaries."""

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import mnemosyne.batch_tool as batch_tool
from mnemosyne.batch_tool import (
    BatchValidationError,
    apply_beam_batch,
    batch_validation_error_payload,
    validate_batch_operations,
)


class _FakeBeam:
    conn = None

    def remember(self, **kwargs):
        return "fake-id"


def _without_transactions(_conn):
    return contextlib.nullcontext()


def test_validation_payload_contains_no_raw_message():
    exc = BatchValidationError(
        "content is required\n/private/secret/path", 0, "remember"
    )
    assert batch_validation_error_payload(exc) == {
        "status": "error",
        "error": "batch_validation_failed",
        "failed_index": 0,
        "action": "remember",
    }


def test_validation_payload_has_fixed_shape_and_rejects_untrusted_context():
    canary: Any = "<script>private-index</script>"
    payload = batch_validation_error_payload(
        BatchValidationError("private message", canary, canary)
    )
    assert payload == {
        "status": "error",
        "error": "batch_validation_failed",
        "failed_index": None,
        "action": None,
    }
    assert canary not in json.dumps(payload)


def test_unknown_action_not_reflected():
    with pytest.raises(BatchValidationError) as exc_info:
        validate_batch_operations([{"action": "<script>alert(1)</script>"}])
    payload = batch_validation_error_payload(exc_info.value)
    assert payload["action"] is None
    assert "<script>" not in json.dumps(payload)


def test_execution_failure_excludes_raw_exception_text():
    canary = "/private/secret/path\nmultiline\ncanary"
    normalized = [
        {"index": 0, "action": "remember", "payload": {"content": "x"}}
    ]
    with (
        patch.object(batch_tool, "_deferred_commits", _without_transactions),
        patch.object(batch_tool, "_apply_one", side_effect=RuntimeError(canary)),
    ):
        result = apply_beam_batch(_FakeBeam(), normalized)

    assert result == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": 0,
        "action": "remember",
    }
    assert canary not in json.dumps(result)
    assert "RuntimeError" not in json.dumps(result)


def test_execution_failure_rejects_untrusted_action_and_index():
    canary = "<script>private-context</script>"
    normalized = [{"index": canary, "action": canary, "payload": {}}]
    with patch.object(batch_tool, "_deferred_commits", _without_transactions):
        result = apply_beam_batch(_FakeBeam(), normalized)

    assert result == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": None,
        "action": None,
    }
    assert canary not in json.dumps(result)


@pytest.mark.parametrize(
    "normalized",
    [
        ["not-an-operation"],
        [{"index": -1, "action": "remember", "payload": {}}],
        [{"index": 50, "action": "remember", "payload": {}}],
        [{"index": True, "action": "remember", "payload": {}}],
    ],
)
def test_execution_failure_sanitizer_handles_malformed_normalized_entries(
    normalized
):
    with patch.object(batch_tool, "_deferred_commits", _without_transactions):
        result = apply_beam_batch(_FakeBeam(), normalized)

    assert result == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": None,
        "action": None if not isinstance(normalized[0], dict) else "remember",
    }


def test_packaged_hermes_fallback_payloads_are_fixed_and_sanitized(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    code = f"""
import builtins
import json
import sys
sys.path.insert(0, {str(repo / 'integrations' / 'hermes' / 'src')!r})
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'mnemosyne.batch_tool':
        raise ImportError('forced fallback')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import mnemosyne_hermes as provider
canary = '/private/fallback\\ncanary'
print(json.dumps({{
    'validation': provider.batch_validation_error_payload(RuntimeError(canary)),
    'execution': provider.apply_beam_batch(),
}}))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "validation": {
            "status": "error",
            "error": "batch_validation_failed",
            "failed_index": None,
            "action": None,
        },
        "execution": {
            "status": "error",
            "error": "batch_failed",
            "failed_index": None,
            "action": None,
        },
    }
    assert "/private/fallback" not in result.stdout


@pytest.mark.parametrize(
    ("payload", "expected_is_error"),
    [
        ({"status": "error", "error": "batch_validation_failed"}, True),
        ({"status": "error", "error": "batch_failed"}, True),
        ({"status": "error", "error": "not_batch"}, False),
        ({"status": "error", "error": 42}, False),
        ({"status": "error", "error": ["batch_failed"]}, False),
        ({"status": "ok", "error": "batch_failed"}, False),
    ],
)
def test_mcp_batch_error_result_contract(payload, expected_is_error):
    from mnemosyne.mcp_server import _build_mcp_server

    class _Params:
        name = "mnemosyne_batch"
        arguments = {}

    server = _build_mcp_server()
    on_call_tool = server.get_request_handler("tools/call").handler
    with patch("mnemosyne.mcp_server.handle_tool_call", return_value=payload):
        result = asyncio.run(on_call_tool(ctx=None, params=_Params()))

    assert result.is_error is expected_is_error
    assert json.loads(result.content[0].text) == payload


def test_actual_mcp_batch_errors_keep_payload_and_transport_contract(
    tmp_path, monkeypatch
):
    from mnemosyne.mcp_server import _build_mcp_server

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

    class _Params:
        name = "mnemosyne_batch"
        arguments = {
            "operations": [{"action": "<script>private-action</script>"}]
        }

    on_call_tool = _build_mcp_server().get_request_handler("tools/call").handler
    validation = asyncio.run(on_call_tool(ctx=None, params=_Params()))
    validation_payload = json.loads(validation.content[0].text)

    _Params.arguments = {
        "operations": [
            {"action": "update", "memory_id": "missing", "content": "x"}
        ]
    }
    execution = asyncio.run(on_call_tool(ctx=None, params=_Params()))
    execution_payload = json.loads(execution.content[0].text)

    assert validation.is_error is True
    assert validation_payload == {
        "status": "error",
        "error": "batch_validation_failed",
        "failed_index": 0,
        "action": None,
    }
    assert execution.is_error is True
    assert execution_payload == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": 0,
        "action": "update",
    }


def test_unexpected_mcp_batch_exception_is_sanitized():
    from mnemosyne.mcp_server import _build_mcp_server

    class _Params:
        name = "mnemosyne_batch"
        arguments = {}

    canary = "/private/mcp/path\ncanary"
    on_call_tool = _build_mcp_server().get_request_handler("tools/call").handler
    with patch(
        "mnemosyne.mcp_server.handle_tool_call", side_effect=RuntimeError(canary)
    ):
        result = asyncio.run(on_call_tool(ctx=None, params=_Params()))

    payload = json.loads(result.content[0].text)
    assert result.is_error is True
    assert payload == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": None,
        "action": None,
    }
    assert canary not in result.content[0].text


def test_mcp_callback_survives_malformed_params_without_secondary_failure():
    from mnemosyne.mcp_server import _build_mcp_server

    class _Params:
        @property
        def name(self):
            raise RuntimeError("malformed params")

    on_call_tool = _build_mcp_server().get_request_handler("tools/call").handler
    result = asyncio.run(on_call_tool(ctx=None, params=_Params()))

    assert result.is_error is True
    assert json.loads(result.content[0].text) == {
        "status": "error",
        "message": "malformed params",
    }
