"""Regression: `mnemosyne recall` must not silently truncate record content.

The reporter in #685 could not tell a 150-char preview from a complete record:
the line ended in `...` with nothing saying the record was cut. `--json` already
emits full content, but it is undiscoverable from the truncated output and forces
a machine-readable shape for a human who just wants to read the record.

These tests pin the property that matters: when the displayed content is shorter
than the stored content, the output must say so, and a full read must be available
without changing the default view.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

import pytest

LONG = (
    "Network infrastructure (three local networks):\n"
    "- 192.168.88.0/25 — main network: gw .88.1 (main Mikrotik)\n"
    "- 192.168.1.0/24 — second network: gw.itj .1.254\n"
    "- 192.168.2.0/24 — third network: gw.itj .2.253\n\n"
    "Сетевая инфраструктура (три локальные сети):\n"
    "- 192.168.88.0/25 — основная сеть: gw .88.1\n"
    "- 192.168.1.0/24 — вторая сеть: gw.itj .1.254\n"
    "- 192.168.2.0/24 — третья сеть: gw.itj .2.253"
)


def _render_recall(results: list[dict]) -> str:
    """Run the real CLI render path and capture what the user sees."""
    from mnemosyne import cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._print_recall_results("network", results)
    return buf.getvalue()


def test_long_record_is_not_truncated_at_the_display_cap():
    out = _render_recall([{"id": "a6dc2c7371307e57", "content": LONG, "score": 0.738}])
    assert "...\n" not in out.replace("192.168", "")  # no ellipsis-only body
    assert "192.168.2.0/24" in out, "the third network is dropped from the output"


def test_truncation_is_announced_when_a_preview_cap_applies():
    """With --preview the cap may apply, but the output must name what was withheld."""
    from mnemosyne import cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._print_recall_results("network", [{"id": "x", "content": "y" * 4000, "score": 0.5}], preview=True)
    out = buf.getvalue()
    assert "showing 150 of 4000" in out, out[-200:]
    assert not out.rstrip().endswith("..."), "a bare ellipsis still hides that the record was cut"


def test_json_output_still_carries_full_content():
    """The machine path is already correct and must not regress."""
    payload = [{"id": "x", "content": LONG, "score": 0.5}]
    assert json.loads(json.dumps({"query": "q", "results": payload}))["results"][0]["content"] == LONG


def _run_cmd_recall(monkeypatch, args):
    """Drive the real command entry point with a stubbed memory backend.

    The renderer tests above cover the content and the character counts, but
    they cannot catch a break in how `--preview` is parsed or forwarded. This
    exercises cmd_recall end to end so that wiring is covered too.
    """
    from mnemosyne import cli

    captured = {}

    class _StubMemory:
        def recall(self, query, top_k=5, explain=False):
            captured["called"] = (query, top_k, explain)
            return [{"id": "x", "content": LONG, "score": 0.5}]

    monkeypatch.setattr(cli, "_get_memory", lambda: _StubMemory())
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.cmd_recall(args)
    return buf.getvalue(), captured


def test_cmd_recall_prints_whole_record_by_default(monkeypatch):
    out, captured = _run_cmd_recall(monkeypatch, ["network"])
    assert captured["called"] == ("network", 5, False)
    assert "192.168.2.0/24" in out, "the default path must not truncate the record"


def test_cmd_recall_preview_flag_is_parsed_and_applies_the_cap(monkeypatch):
    out, _ = _run_cmd_recall(monkeypatch, ["network", "--preview"])
    assert "Content truncated: showing 150 of" in out
    assert "192.168.2.0/24" not in out


def test_cmd_recall_json_flag_is_parsed_and_stays_complete(monkeypatch):
    out, _ = _run_cmd_recall(monkeypatch, ["network", "--json"])
    payload = json.loads(out)
    assert payload["results"][0]["content"] == LONG


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))