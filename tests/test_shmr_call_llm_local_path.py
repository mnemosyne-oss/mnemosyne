"""Regression test: shmr._call_llm must actually reach the local LLM.

Root cause (verified on upstream 85a4b88): mnemosyne/core/shmr.py::_call_llm
called `local_llm._call_local_llm(prompt, system=..., temperature=...)`, but
`_call_local_llm` accepts only `prompt`. The resulting TypeError was swallowed
by the bare `except Exception: pass`, so the local LLM path was dead and
`_call_llm` could only ever return the cloud fallback (or "").

This test exercises the real `_call_llm` with a local LLM installed via the
shared `local_llm_enabled` fixture (no monkeypatching of `_call_llm` itself)
and asserts the local callable is reached and its output returned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mnemosyne.core import shmr  # noqa: E402


def test_call_llm_uses_local_llm(local_llm_enabled):
    """_call_llm returns the local LLM's output when the local path works.

    Before the fix this raised TypeError inside the try block (system=/temperature=
    unsupported by _call_local_llm), was swallowed, and _call_llm returned "".
    """
    local_llm_enabled.response = "local-llm-produced-this-harmony"
    captured = []

    def create_chat_completion(messages, **kwargs):
        captured.append(messages)
        return {"choices": [{"message": {"content": local_llm_enabled.response}}]}

    local_llm_enabled.create_chat_completion = create_chat_completion

    out = shmr._call_llm("harmonize this cluster please", system="be concise")

    assert out == "local-llm-produced-this-harmony"
    assert captured[0][0]["content"] == "be concise\n\nharmonize this cluster please"


def test_call_llm_logs_local_failure_without_prompt(caplog, monkeypatch):
    """Local fallback failures remain diagnosable without logging prompt text."""
    from mnemosyne.core import local_llm
    from mnemosyne.extraction import client as extraction_client_mod

    def fail_local(prompt):
        raise RuntimeError("backend unavailable")

    class NoNetworkClient:
        def __init__(self, *a, **kw):
            pass

        def chat(self, messages, **kw):
            raise AssertionError("test must not send outbound requests")

    monkeypatch.setattr(local_llm, "_call_local_llm", fail_local)
    monkeypatch.setattr(extraction_client_mod, "ExtractionClient", NoNetworkClient)
    # shmr imports ExtractionClient lazily inside _call_llm, so patching the
    # package attribute is also needed for the no-network guarantee.
    import mnemosyne.extraction as extraction_pkg

    monkeypatch.setattr(extraction_pkg, "ExtractionClient", NoNetworkClient)

    with caplog.at_level("DEBUG", logger="mnemosyne.shmr"):
        shmr._call_llm("private prompt", system="private system")

    record = next(r for r in caplog.records if r.message.startswith("SHMR local LLM failed"))
    assert "RuntimeError" in record.getMessage()
    assert "private prompt" not in record.getMessage()
    assert "private system" not in record.getMessage()


def test_call_llm_without_system_passes_prompt_unchanged():
    """The default-system branch (harmonize()'s path) sends the bare prompt."""
    from mnemosyne.core import local_llm
    from mnemosyne.extraction import client as extraction_client_mod

    captured = []

    def fake_local(prompt):
        captured.append(prompt)
        return "local-harmony-result"

    class NoNetworkClient:
        def __init__(self, *a, **kw):
            pass

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(local_llm, "_call_local_llm", fake_local)
    monkeypatch.setattr(extraction_client_mod, "ExtractionClient", NoNetworkClient)
    import mnemosyne.extraction as extraction_pkg

    monkeypatch.setattr(extraction_pkg, "ExtractionClient", NoNetworkClient)
    try:
        out = shmr._call_llm("harmonize this cluster please")
    finally:
        monkeypatch.undo()

    assert out == "local-harmony-result"
    assert captured == ["harmonize this cluster please"]
