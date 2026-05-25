"""[issue #45 followup] Tests for hermes_plugin.tools.mnemosyne_recall.

Adversarial review of issue #45's PR caught that the Hermes plugin's recall
handler ALSO drops vec_weight / fts_weight / importance_weight. The
RECALL_SCHEMA at hermes_plugin/tools.py:66-110 doesn't advertise the
scoring weights, and the handler at lines 375-393 doesn't forward them
to mem.recall.

Same bug class as the MCP-side fix in PR #46 and the
hermes_memory_provider fix on the C12.b branch — schema/handler mismatch
with what BeamMemory.recall actually accepts.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_recall_schema_advertises_scoring_weights():
    """Hermes plugin's RECALL_SCHEMA must advertise vec_weight / fts_weight /
    importance_weight as type=number properties so Hermes' tool-arg validator
    accepts them instead of stripping as unknown fields."""
    from hermes_plugin.tools import RECALL_SCHEMA

    props = RECALL_SCHEMA["parameters"]["properties"]
    for key in ("vec_weight", "fts_weight", "importance_weight"):
        assert key in props, (
            f"hermes_plugin RECALL_SCHEMA missing {key!r} — schema "
            f"advertises 'hybrid vector + full-text search' but doesn't "
            f"let clients tune the weights"
        )
        assert props[key]["type"] == "number", (
            f"{key} should be type=number, got {props[key].get('type')!r}"
        )


def test_mnemosyne_recall_forwards_scoring_weights_to_mem(monkeypatch):
    """hermes_plugin.tools.mnemosyne_recall handler must forward vec_weight /
    fts_weight / importance_weight to mem.recall when caller supplies them."""
    from hermes_plugin import tools as plugin_tools

    captured = {}

    class _StubMem:
        def recall(self, query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return []

    monkeypatch.setattr(plugin_tools, "_get_memory", lambda **kw: _StubMem())

    response = plugin_tools.mnemosyne_recall({
        "query": "anything",
        "top_k": 3,
        "vec_weight": 0.55,
        "fts_weight": 0.25,
        "importance_weight": 0.20,
    })

    parsed = json.loads(response)
    assert "error" not in parsed, parsed
    assert captured.get("vec_weight") == 0.55, (
        f"mnemosyne_recall did not forward vec_weight; captured={captured!r}"
    )
    assert captured.get("fts_weight") == 0.25
    assert captured.get("importance_weight") == 0.20


def test_mnemosyne_recall_omits_weights_when_caller_does_not_supply(monkeypatch):
    """When caller omits the scoring weights, the handler must NOT pass
    spurious values to mem.recall — beam treats None as 'fall back to env
    var or default' via _normalize_weights, and forcing 0.0 / 0.5 / etc.
    would override that resolution and break MNEMOSYNE_*_WEIGHT env-var
    deployments."""
    from hermes_plugin import tools as plugin_tools

    captured = {}

    class _StubMem:
        def recall(self, query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return []

    monkeypatch.setattr(plugin_tools, "_get_memory", lambda **kw: _StubMem())

    plugin_tools.mnemosyne_recall({"query": "anything", "top_k": 3})

    # Acceptable: kwarg not in mem.recall's call OR explicitly None.
    # Failing path: a numeric default (0.5 / 0.0) leaked through.
    for key in ("vec_weight", "fts_weight", "importance_weight"):
        val = captured.get(key, "OMITTED")
        assert val in (None, "OMITTED"), (
            f"mnemosyne_recall forwarded {key}={val!r} when caller "
            f"omitted it; this overrides beam's env/default resolution. "
            f"Either pass None or omit the kwarg entirely."
        )
