"""Regression tests for the two pre-test gap closures:

1. **Per-question voice_scores threading** — `evaluate_conversation()`
   now writes a `recall_provenance` summary into each per-question
   result dict so post-hoc per-voice attribution analysis (Recipe E
   in `docs/benchmark-results-analysis.md`) works from the result
   file directly. Without this, Theses 1, 2, 3 in the experiment
   plan can't be falsified — they require knowing which voice
   contributed to which question.

2. **`MNEMOSYNE_BEAM_OPTIMIZATIONS` parser migration** — was using
   the brittle `.lower() in ("1","true","yes")` pattern that rejects
   `on` and whitespace. Now goes through `_env_truthy()` in `beam.py`,
   matching the convention from PR #91.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mnemosyne.core.beam import BeamMemory, _env_truthy


# ─────────────────────────────────────────────────────────────────
# Item 2: MNEMOSYNE_BEAM_OPTIMIZATIONS parser migration
# ─────────────────────────────────────────────────────────────────


class TestBeamOptimizationsEnvParser:
    """The env var now goes through `_env_truthy` so the parsing
    semantics match the rest of the codebase: accepts `on`, strips
    whitespace, case-insensitive."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on",
                                        "TRUE", "ON", "On",
                                        " 1 ", "  true  ", "\ton\t"])
    def test_truthy_values_accepted(self, value, monkeypatch):
        """Pre-fix, `on` and whitespace-padded values were silently
        treated as off. Post-fix they enable BEAM mode."""
        monkeypatch.setenv("MNEMOSYNE_BEAM_OPTIMIZATIONS", value)
        assert _env_truthy("MNEMOSYNE_BEAM_OPTIMIZATIONS") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off",
                                        "FALSE", "OFF",
                                        "", " ", "garbage", "maybe"])
    def test_falsy_or_garbage_rejected(self, value, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BEAM_OPTIMIZATIONS", value)
        assert _env_truthy("MNEMOSYNE_BEAM_OPTIMIZATIONS") is False

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_BEAM_OPTIMIZATIONS", raising=False)
        assert _env_truthy("MNEMOSYNE_BEAM_OPTIMIZATIONS") is False


# ─────────────────────────────────────────────────────────────────
# Item 1: Voice-attribution threading
# ─────────────────────────────────────────────────────────────────


class TestAnswerWithMemoryReturnMemoriesKwarg:
    """The `return_memories=True` kwarg returns `(answer, memories)`;
    the default `return_memories=False` returns the answer string
    unchanged (back-compat for existing tests)."""

    @pytest.fixture
    def fake_llm(self):
        llm = MagicMock()
        llm.chat = MagicMock(return_value="LLM-FALLBACK-ANSWER")
        return llm

    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test.db"

    def test_default_returns_string(self, temp_db, fake_llm, monkeypatch):
        """Default — back-compat with all existing callers."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        result = answer_with_memory(
            llm=fake_llm, beam=beam,
            question="anything",
            conversation_messages=[{"role": "user", "content": "x"}],
            top_k=5, ability="ABS",
        )
        assert isinstance(result, str)

    def test_return_memories_true_returns_tuple(self, temp_db, fake_llm, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        result = answer_with_memory(
            llm=fake_llm, beam=beam,
            question="anything",
            conversation_messages=[{"role": "user", "content": "x"}],
            top_k=5, ability="ABS",
            return_memories=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        answer, memories = result
        assert isinstance(answer, str)
        assert isinstance(memories, list)

    def test_bypass_path_returns_empty_memories(self, temp_db, fake_llm, monkeypatch):
        """TR bypass short-circuits before recall — memories list
        should be empty but the field still present (schema parity)."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from tools.evaluate_beam_end_to_end import answer_with_memory

        # TR-shaped fixture that triggers the bypass
        msgs = [
            {"role": "user", "content": "I started March 15, 2024."},
            {"role": "user", "content": "Deployed June 30, 2024."},
        ]
        answer, memories = answer_with_memory(
            llm=fake_llm, beam=beam,
            question="how many days between?",
            conversation_messages=msgs,
            top_k=5, ability="TR",
            return_memories=True,
        )
        assert memories == []


# ─────────────────────────────────────────────────────────────────
# Item 1: _summarize_recall_memories helper
# ─────────────────────────────────────────────────────────────────


class TestSummarizeRecallMemories:
    """The summary helper packs voice_scores from a list of memory
    dicts into a compact provenance object."""

    def test_empty_memories_returns_minimal_shape(self):
        from tools.evaluate_beam_end_to_end import _summarize_recall_memories
        out = _summarize_recall_memories([])
        assert out == {
            "engine": "unknown",
            "kept_count": 0,
            "voice_sums": {},
            "top_result_voices": {},
            "top_result_tier": None,
        }

    def test_polyphonic_engine_identified_by_keyset(self):
        from tools.evaluate_beam_end_to_end import _summarize_recall_memories
        memories = [
            {"voice_scores": {"vector": 0.5, "graph": 0.2}, "tier": "episodic"},
            {"voice_scores": {"vector": 0.4}, "tier": "working"},
        ]
        out = _summarize_recall_memories(memories)
        assert out["engine"] == "polyphonic"
        assert out["kept_count"] == 2
        # Sums across kept results
        assert out["voice_sums"]["vector"] == pytest.approx(0.9)
        assert out["voice_sums"]["graph"] == pytest.approx(0.2)
        # Top result is the first memory's voice_scores
        assert out["top_result_voices"]["vector"] == 0.5
        assert out["top_result_tier"] == "episodic"

    def test_linear_engine_identified_by_keyset(self):
        from tools.evaluate_beam_end_to_end import _summarize_recall_memories
        memories = [
            {"voice_scores": {"vec": 0.7, "fts": 0.3, "keyword": 0.0,
                              "importance": 0.5, "recency_decay": 0.9},
             "tier": "working"},
        ]
        out = _summarize_recall_memories(memories)
        assert out["engine"] == "linear"
        assert out["kept_count"] == 1
        assert out["voice_sums"]["vec"] == pytest.approx(0.7)
        assert out["top_result_tier"] == "working"

    def test_unknown_engine_for_voice_keys_outside_known_sets(self):
        from tools.evaluate_beam_end_to_end import _summarize_recall_memories
        # All memories have voice_scores but with unrecognized keys
        memories = [{"voice_scores": {"made_up_voice": 0.5}, "tier": "working"}]
        out = _summarize_recall_memories(memories)
        assert out["engine"] == "unknown"
        # Voice sums still tracked for unknown keys
        assert out["voice_sums"]["made_up_voice"] == 0.5

    def test_handles_missing_voice_scores_field(self):
        """Some memory dicts might not have voice_scores at all
        (e.g., bypass-path placeholders). Helper should tolerate."""
        from tools.evaluate_beam_end_to_end import _summarize_recall_memories
        memories = [
            {"tier": "working"},  # no voice_scores
            {"voice_scores": {"vector": 0.5}, "tier": "episodic"},
        ]
        out = _summarize_recall_memories(memories)
        assert out["kept_count"] == 2
        assert out["engine"] == "polyphonic"
        assert out["voice_sums"]["vector"] == 0.5

    def test_handles_non_numeric_voice_value(self):
        """Defensive: malformed voice_score values are skipped, not
        crash."""
        from tools.evaluate_beam_end_to_end import _summarize_recall_memories
        memories = [
            {"voice_scores": {"vector": 0.5, "graph": None}, "tier": "working"},
            {"voice_scores": {"vector": "not-a-number"}, "tier": "working"},
        ]
        out = _summarize_recall_memories(memories)
        # `None` and "not-a-number" silently skipped
        assert out["voice_sums"]["vector"] == 0.5
        assert "graph" not in out["voice_sums"]


# ─────────────────────────────────────────────────────────────────
# Integration: recall_provenance ends up in per-question result
# ─────────────────────────────────────────────────────────────────


class TestRecallProvenanceInResultDict:
    """End-to-end: a per-question result dict includes
    `recall_provenance` (the summary). Pinned via source-grep since
    running `evaluate_conversation()` end-to-end requires real LLM."""

    def test_evaluate_conversation_writes_recall_provenance(self):
        """The per-question result dict construction in
        `evaluate_conversation()` includes `recall_provenance`. A
        future refactor that drops the field would break post-hoc
        attribution analysis."""
        harness_src = (_REPO_ROOT / "tools" / "evaluate_beam_end_to_end.py").read_text()
        # The field is added to the per-question result dict
        assert '"recall_provenance": recall_provenance,' in harness_src, (
            "evaluate_conversation no longer writes recall_provenance "
            "into per-question result dict — Recipe E in "
            "docs/benchmark-results-analysis.md broken"
        )
        # And `return_memories=True` is passed when calling answer_with_memory
        assert "return_memories=True" in harness_src, (
            "evaluate_conversation no longer requests memories for "
            "provenance — recall_provenance will be empty"
        )
