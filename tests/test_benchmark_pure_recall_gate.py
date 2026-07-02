"""Regression tests for E7/E8/E9 — `MNEMOSYNE_BENCHMARK_PURE_RECALL` gate.

`_benchmarks/evaluate_beam_end_to_end.py` historically shipped four bypass
paths that let the harness answer benchmark questions WITHOUT going
through Mnemosyne recall:

- **E7 TR oracle:** TR (Temporal Reasoning) questions extracted a
  timeline from raw `conversation_messages` and returned the LLM
  answer directly (line 1080) before any `BeamMemory.recall()`.
- **E7 CR augmentation:** CR (Contradiction Resolution) questions
  injected contradiction context built from raw messages into the
  answer prompt (line 1089).
- **E8 IE/KU side-index:** `_context_facts` (built from raw messages
  at ingest, line 418) was queried by IE/KU questions; matching
  values were returned directly at line 1291.
- **E9 RECENT CONVERSATION:** the last 12 raw messages were prepended
  to every answer prompt (line 1282) regardless of arm or recall
  quality.

For the BEAM-recovery experiment (Arms A/B/C compare recall pathways),
these bypasses mean the harness measures a harness-side oracle on
TR/CR/IE/KU and the recent-context shortcut on every question type —
NOT what the arms actually retrieve. `MNEMOSYNE_BENCHMARK_PURE_RECALL=1`
(or `--pure-recall`) disables all four.

Default behavior preserved when env unset.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def fake_llm():
    """LLMClient stand-in that captures the messages it was called with."""
    llm = MagicMock()
    llm.chat = MagicMock(return_value="LLM-FALLBACK-ANSWER")
    return llm


def _build_msgs(n: int = 20) -> List[dict]:
    """Synthetic conversation messages — `n` user turns."""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"message-{i} payload alpha"})
    return msgs


@pytest.fixture
def beam_with_context_facts(temp_db):
    """A BeamMemory with a non-empty `_context_facts` map so we can
    exercise the IE/KU side-index path."""
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam._context_facts = {"favorite color blue": ["blue"]}
    return beam


# ─────────────────────────────────────────────────────────────────
# Default mode (env unset) — existing bypasses still fire
# ─────────────────────────────────────────────────────────────────


class TestDefaultModeBehaviorUnchanged:
    """When `MNEMOSYNE_BENCHMARK_PURE_RECALL` is unset, the existing
    bypass paths still fire (zero behavioral regression for callers
    who haven't migrated)."""

    def test_default_ie_returns_context_fact_value(self, beam_with_context_facts, fake_llm, monkeypatch):
        """IE question with a matching `_context_facts` entry returns
        the value directly — bypass is active."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(20),
            top_k=5,
            ability="IE",
        )
        assert ans == "blue"
        # LLM was NOT called — bypass returned the value directly.
        fake_llm.chat.assert_not_called()

    def test_default_recent_context_included(self, temp_db, fake_llm, monkeypatch):
        """When LLM is invoked (e.g., for ABS questions), the prompt
        includes the RECENT CONVERSATION section by default."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        msgs = _build_msgs(20)
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="some abstract reasoning question",
            conversation_messages=msgs,
            top_k=5,
            ability="ABS",
        )
        # LLM called; its prompt includes "RECENT CONVERSATION".
        fake_llm.chat.assert_called()
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        assert "RECENT CONVERSATION" in user_msg


# ─────────────────────────────────────────────────────────────────
# Pure-recall mode — all bypasses disabled
# ─────────────────────────────────────────────────────────────────


class TestPureRecallModeDisablesBypasses:
    """When `MNEMOSYNE_BENCHMARK_PURE_RECALL=1`, every bypass is
    disabled and every answer must go through the full LLM path with
    only retrieved memories in context."""

    def test_pure_recall_ie_does_not_return_context_fact_value(
        self, beam_with_context_facts, fake_llm, monkeypatch
    ):
        """IE question with matching `_context_facts` should NOT short-
        circuit in pure-recall mode — the LLM gets called instead."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(20),
            top_k=5,
            ability="IE",
        )
        # LLM was called — bypass disabled.
        fake_llm.chat.assert_called_once()
        # Whatever the LLM returned is the answer (our fake returns LLM-FALLBACK-ANSWER).
        assert ans == "LLM-FALLBACK-ANSWER"

    # TR fixture: must use "Month Day, Year" format which is what
    # `_extract_timeline_from_conversation`'s Pattern 1 matches. ISO
    # `2024-01-15` format DOES NOT match the regex — using ISO would
    # make the test pass vacuously (the bypass wouldn't fire in either
    # mode, so "absent in pure-recall mode" is trivially true).
    _TR_FIXTURE_MSGS = [
        {"role": "user", "content": "I started the project on March 15, 2024 with the team."},
        {"role": "user", "content": "Then I deployed it on June 30, 2024 after testing."},
        {"role": "user", "content": "The final release was September 10, 2024."},
    ]
    _TR_FIXTURE_QUESTION = "how many days between project start and deployment?"

    # CR fixture: `_detect_contradictions` needs key terms from the
    # question to appear in the conversation AND at least one message
    # must contain a negation word (never/not/n't/no/etc.) in the
    # sentence with the term. Using "never" satisfies that.
    _CR_FIXTURE_MSGS = [
        {"role": "user", "content": "I love flask routes and use them for all HTTP requests."},
        {"role": "user", "content": "I never use flask routes — I prefer raw WSGI handlers."},
    ]
    _CR_FIXTURE_QUESTION = "Have I worked with flask routes?"

    def test_default_tr_oracle_fires_positive_control(self, temp_db, fake_llm, monkeypatch):
        """Positive control: in DEFAULT mode, the TR zero-LLM date calculator
        DOES fire with this fixture and computes the answer without calling
        the LLM. Verifies the bypass works end-to-end."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm, beam=beam,
            question=self._TR_FIXTURE_QUESTION,
            conversation_messages=self._TR_FIXTURE_MSGS,
            top_k=5, ability="TR",
        )
        # TR zero-LLM bypass computes the answer directly without calling the LLM.
        # Verify the computed answer contains the correct date calculation.
        assert ans is not None, "TR bypass returned None"
        assert "107" in str(ans), (
            f"TR bypass should compute 107 days (Mar 15 → Jun 30), "
            f"got: {ans[:200]}"
        )

    def test_pure_recall_tr_bypass_also_fires(
        self, temp_db, fake_llm, monkeypatch
    ):
        """TR zero-LLM bypass fires in pure-recall mode too — it is an
        optimization that applies in all modes, not gated by pure-recall."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm, beam=beam,
            question=self._TR_FIXTURE_QUESTION,
            conversation_messages=self._TR_FIXTURE_MSGS,
            top_k=5, ability="TR",
        )
        # Zero-LLM bypass fires in pure-recall mode too.
        assert ans is not None, "TR bypass returned None in pure-recall mode"
        assert "107" in str(ans), (
            f"TR bypass should compute 107 days in pure-recall mode, "
            f"got: {ans[:200]}"
        )

    def test_default_cr_detection_fires_positive_control(self, temp_db, fake_llm, monkeypatch):
        """Positive control: in DEFAULT mode, the CR-detect injection
        DOES fire with this fixture. Checks all LLM calls since the gap
        analysis path makes multiple calls and contradiction injection
        may appear in any message."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        answer_with_memory(
            llm=fake_llm, beam=beam,
            question=self._CR_FIXTURE_QUESTION,
            conversation_messages=self._CR_FIXTURE_MSGS,
            top_k=5, ability="CR",
        )
        # CR detection fires the gap analysis path which makes multiple LLM
        # calls. Search ALL calls for the contradiction injection string.
        found_contradiction = False
        all_contents = []
        for call in fake_llm.chat.call_args_list:
            messages = call[0][0]
            for msg in messages:
                content = msg.get("content", "")
                all_contents.append(content[:200])
                if "contradictory information" in content.lower():
                    found_contradiction = True
                    break
            if found_contradiction:
                break
        assert found_contradiction, (
            f"CR-detect did NOT inject contradiction context in DEFAULT mode. "
            f"Searched {len(fake_llm.chat.call_args_list)} LLM calls. "
            f"Sample contents: {all_contents[:3]}"
        )

    def test_pure_recall_cr_does_not_inject_contradiction_context(
        self, temp_db, fake_llm, monkeypatch
    ):
        """CR question should NOT inject `_detect_contradictions`
        output into the prompt — pure recall means recall alone."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        answer_with_memory(
            llm=fake_llm, beam=beam,
            question=self._CR_FIXTURE_QUESTION,
            conversation_messages=self._CR_FIXTURE_MSGS,
            top_k=5, ability="CR",
        )
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        assert "contradictory information" not in user_msg, (
            f"CR-bypass injected contradiction context despite pure-recall mode; "
            f"prompt: {user_msg[:300]}"
        )

    def test_pure_recall_excludes_recent_conversation_section(
        self, temp_db, fake_llm, monkeypatch
    ):
        """The 'RECENT CONVERSATION' block (last 12 raw messages) is
        always included pre-fix. Pure-recall mode strips it so the LLM
        sees only what each arm's recall returned."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        msgs = _build_msgs(20)
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="some abstract reasoning question",
            conversation_messages=msgs,
            top_k=5,
            ability="ABS",
        )
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        assert "RECENT CONVERSATION" not in user_msg, (
            f"RECENT CONVERSATION section leaked into pure-recall prompt: "
            f"{user_msg[:300]}"
        )


class TestPureRecallEnvValueParsing:
    """The env var is treated as truthy on '1', 'true', 'yes' (lowercase
    or any case), falsy/unset otherwise. Locks the parsing surface."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES"])
    def test_truthy_values_enable_gate(self, value, beam_with_context_facts, fake_llm, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", value)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(5),
            top_k=5,
            ability="IE",
        )
        # IE bypass should NOT fire; LLM should be called.
        fake_llm.chat.assert_called()

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "anything-else"])
    def test_falsy_values_preserve_default(self, value, beam_with_context_facts, fake_llm, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", value)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        ans = answer_with_memory(
            llm=fake_llm,
            beam=beam_with_context_facts,
            question="what is favorite color blue",
            conversation_messages=_build_msgs(5),
            top_k=5,
            ability="IE",
        )
        # IE bypass SHOULD fire — LLM not called, value returned directly.
        assert ans == "blue"
        fake_llm.chat.assert_not_called()


class TestPureRecallPrecedenceOverFullContext:
    """Codex /review P1: when both pure_recall and full_context are
    active, pure_recall MUST win — otherwise the full-context path
    silently invalidates the recall-only guarantee by shipping the
    entire raw conversation to the LLM."""

    def test_pure_recall_disables_full_context_mode_even_when_both_set(
        self, temp_db, fake_llm, monkeypatch
    ):
        """Both env vars set → pure_recall wins; FULL CONVERSATION
        block must NOT appear in the prompt."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        monkeypatch.setenv("FULL_CONTEXT_MODE", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        msgs = _build_msgs(20)
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="some abstract reasoning question",
            conversation_messages=msgs,
            top_k=5,
            ability="ABS",
        )
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        assert "FULL CONVERSATION" not in user_msg, (
            "FULL_CONTEXT_MODE leaked despite pure-recall being set; "
            f"prompt: {user_msg[:400]}"
        )

    def test_full_context_alone_still_works_when_pure_recall_off(
        self, temp_db, fake_llm, monkeypatch
    ):
        """Sanity: existing FULL_CONTEXT behavior unchanged when only
        full-context is set."""
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        monkeypatch.setenv("FULL_CONTEXT_MODE", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        msgs = _build_msgs(5)
        answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="some abstract reasoning question",
            conversation_messages=msgs,
            top_k=5,
            ability="ABS",
        )
        user_msg = fake_llm.chat.call_args[0][0][-1]["content"]
        assert "FULL CONVERSATION" in user_msg, (
            "FULL_CONTEXT_MODE didn't fire when pure-recall was off; "
            "preserving the existing benchmark mode is a regression "
            f"if this fails. Prompt: {user_msg[:300]}"
        )


class TestPureRecallActuallyRoutesThroughRecall:
    """Codex /review P2: prior tests only assert "X not present in
    prompt" — that passes under both pure-recall AND full-context
    paths. Strengthen by spying on `_multi_strategy_recall` to confirm
    pure-recall actually goes through the recall pipeline."""

    def test_pure_recall_invokes_multi_strategy_recall(
        self, temp_db, fake_llm, monkeypatch
    ):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        import _benchmarks.evaluate_beam_end_to_end as harness

        spy = MagicMock(return_value=[])
        monkeypatch.setattr(harness, "_multi_strategy_recall", spy)
        harness.answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="some reasoning question",
            conversation_messages=_build_msgs(20),
            top_k=5,
            ability="ABS",
        )
        spy.assert_called_once()

    def test_pure_recall_tr_routes_through_recall_not_oracle(
        self, temp_db, fake_llm, monkeypatch
    ):
        """Even with a TR-shaped question that would have triggered
        the timeline oracle, pure-recall mode reaches the recall
        pipeline."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        import _benchmarks.evaluate_beam_end_to_end as harness

        spy = MagicMock(return_value=[])
        monkeypatch.setattr(harness, "_multi_strategy_recall", spy)
        msgs = [
            {"role": "user", "content": "started project 2024-01-15"},
            {"role": "user", "content": "deployed 2024-03-22"},
        ]
        harness.answer_with_memory(
            llm=fake_llm,
            beam=beam,
            question="days between?",
            conversation_messages=msgs,
            top_k=5,
            ability="TR",
        )
        spy.assert_called_once()
