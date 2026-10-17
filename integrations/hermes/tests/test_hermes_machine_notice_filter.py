"""Tests for Hermes machine-generated notice filtering in mnemosyne_hermes sync_turn (Issue #1102)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mnemosyne.core.beam import BeamMemory
from mnemosyne_hermes import MnemosyneMemoryProvider
from mnemosyne_hermes import _sanitize_hermes_machine_envelope


MACHINE_NOTICES = [
    "[IMPORTANT: Background process proc_1 completed normally (exit code 0).\nCommand: ls\nOutput:\nfoo]",
    "[IMPORTANT: Background process proc_2 exited (exit code 1).\nCommand: test.sh\nOutput:\nerror]",
    "[IMPORTANT: Background process proc_3 matched watch pattern \"done\".\nCommand: build.sh\nMatched output:\ndone]",
    "[IMPORTANT: 2 background processes completed.\n[IMPORTANT: Background process proc_a completed normally (exit code 0)]]",
    "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your final response will be automatically delivered.]",
    "[IMPORTANT: MCP servers have been reloaded. Added 2 tools.]",
    "[System note: The user switched models from claude to gpt]",
    "[ASYNC DELEGATION COMPLETE — deleg_123]",
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_456]",
    "[ASYNC DELEGATION TASK FAILED — deleg_789, task 2/3]",
    "[CONTEXT COMPACTION — REFERENCE ONLY]\nPrevious summary content here",
    "[CONTEXT SUMMARY]: previous context of discussion",
    "[PRIOR CONTEXT — for reference only; not a new message]",
    "[Your active task list was preserved across context compression]",
    "[Planning state preserved across model switch]",
    "A background fan-out of 3 subagent(s) you dispatched earlier has finished.",
    "A background subagent you dispatched earlier has finished. You may have moved on.",
    "[IMPORTANT: Background process proc_complex completed normally (exit code 0).\nCommand: python -c \"print([1, 2, 3])\"\nOutput:\n[1, 2, 3]\nfoo]bar]",
    "[ASYNC DELEGATION COMPLETE — deleg_abc123]\nA background subagent you dispatched earlier has finished. You may have moved on since dispatching it; the full task source is below so you can act on the result or re-dispatch if things have changed.\n\nDispatched: 2026-10-14 10:00:00 (1m ago)\nOriginal goal: verify schema migrations\nContext you provided: use test database\nToolsets: terminal, web\nRole: leaf   Model: gpt-5\nStatus: completed   API calls: 3   Duration: 12.5s\n--- RESULT ---\nMigrations verified: 0 errors found.",
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_batch789]\nA background fan-out of 2 subagent(s) you dispatched earlier has finished. All ran in parallel and waited on each other; their consolidated results are below. You may have moved on since dispatching — act on these or re-dispatch if things have changed.\n\nRole: orchestrator   Model: gpt-5   Total duration: 15.0s\n\n--- ✓ TASK 1/2: run lint  (status=completed) ---\nLint passed\n\n--- ✓ TASK 2/2: run test  (status=completed) ---\nTests passed",
    "[IMPORTANT: 2 background subagent delegations completed for this session. Treat these results as one completion batch and send at most one consolidated user-facing response. If a result does not change the current conclusion, absorb it silently.]",
]

HUMAN_CONTROLS = [
    "IMPORTANT: Background process — what does that mean?",
    "[IMPORTANT: Background process — what does that mean?]",
    "A background process I ran has finished — can you check the output?",
    "A background subagent you dispatched earlier has finished? no wait, I was asking about the report",
    "When you see PRIOR CONTEXT, treat it carefully",
    "the async delegation batch complete marker disappeared from my log",
    "I want to know about your task list",
    "[System note: what does that mean?]",
    "[System note: how to configure this?]",
    "[System note: custom note] Some user text",
    "Can you explain why [System note: The user switched models] appeared?",
    "[CONTEXT SUMMARY]: what does that mean?",
    "[PRIOR CONTEXT]: why was this preserved?",
    "[Your active task list was preserved — where are the tasks?]",
    "[Planning state preserved across model switch — what changed?]",
    "[IMPORTANT: Background process proc_1 completed normally (exit code 0) — why did it exit?]",
    "Is [CONTEXT COMPACTION] configurable?",
    "What does [ASYNC DELEGATION COMPLETE] mean?",
]

MIXED_NOTICES_AND_PROMPTS = [
    (
        "[IMPORTANT: Background process job completed]\nPlease summarize the results",
        "Please summarize the results",
    ),
    (
        "[IMPORTANT: Background process proc_1 completed normally (exit code 0).\nCommand: ls\nOutput:\nfoo]\n"
        "[System note: Your previous turn was interrupted mid-run — the app or its backend process stopped. The interrupted request was:]\n\n"
        "please build the data ingestion pipeline",
        "please build the data ingestion pipeline",
    ),
    (
        "[System note: Your previous turn was interrupted mid-run — the app or its backend process stopped. "
        "The interrupted request was:]\n\n"
        "please build the data ingestion pipeline",
        "please build the data ingestion pipeline",
    ),
    (
        "[System note: The user switched models from claude to gpt]\nNow continue with step 2",
        "Now continue with step 2",
    ),
    (
        "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your final response will be automatically delivered.]\n\n"
        "run daily checks",
        "run daily checks",
    ),
    (
        '[Replying to: "Cronjob Response: Daily DB Backup"]\n\nUpdate retention to 14 days',
        "Update retention to 14 days",
    ),
    (
        "A background subagent you dispatched earlier has finished. You may have moved on.\n\nWhat were the findings?",
        "What were the findings?",
    ),
    (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below.\n## Historical Task Snapshot\nSummary here\n\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---\n\nWhat is the status of the project?",
        "What is the status of the project?",
    ),
    (
        "[CONTEXT SUMMARY]: Previous discussion about database migration\n\nPlease add index on email column",
        "Please add index on email column",
    ),
    (
        "[PRIOR CONTEXT — for reference only; not a new message]\nContext line 1\n\nCan you explain the architecture?",
        "Can you explain the architecture?",
    ),
    (
        "[Your active task list was preserved across context compression]\n- Task 1: review code\n\nPlease start working on task 1",
        "Please start working on task 1",
    ),
    (
        "[Planning state preserved across model switch]\nPhase 1: active\n\nProceed to phase 2",
        "Proceed to phase 2",
    ),
    (
        "[IMPORTANT: Background process proc_complex completed normally (exit code 0).\nCommand: python -c \"print([1, 2, 3])\"\nOutput:\n[1, 2, 3]\nfoo]bar]\nPlease summarize the results",
        "Please summarize the results",
    ),
    (
        "[ASYNC DELEGATION COMPLETE — deleg_abc123]\nA background subagent you dispatched earlier has finished.\nDispatched: 2026-10-14 10:00:00\nOriginal goal: sensitive goal\nRole: leaf   Model: gpt-5\nStatus: completed   Duration: 5s\n--- RESULT ---\nSensitive findings\n\n[System note: Your previous turn was interrupted mid-run — the app or its backend process stopped. The interrupted request was:]\n\nplease deploy to staging",
        "please deploy to staging",
    ),
]


@pytest.fixture
def provider():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}
        prov._capture_identity_signals = MagicMock()
        prov._beam.remember = MagicMock(return_value="mem_123")
        yield prov


@pytest.mark.parametrize("notice", MACHINE_NOTICES)
def test_sync_turn_rejects_machine_notices(provider, notice):
    """Machine notice turns are rejected: neither user nor paired assistant autosaves."""
    provider.sync_turn(notice, "I see the process completed.")
    provider._beam.remember.assert_not_called()
    provider._capture_identity_signals.assert_not_called()


@pytest.mark.parametrize("notice", MACHINE_NOTICES)
def test_sync_turn_assistant_only_skips_machine_notices(provider, notice):
    """Even in assistant-only mode, paired assistant turns for machine notices are skipped."""
    provider._sync_roles = {"assistant"}
    provider.sync_turn(notice, "I see the process completed.")
    provider._beam.remember.assert_not_called()


@pytest.mark.parametrize("human_msg", HUMAN_CONTROLS)
def test_sync_turn_preserves_human_controls(provider, human_msg):
    """Genuine user prompts mentioning notice terms are preserved and autosaved."""
    provider.sync_turn(human_msg, "I can explain that.")

    calls = provider._beam.remember.call_args_list
    assert len(calls) == 2
    contents = [c.kwargs.get("content", "") for c in calls]
    assert contents[0] == f"[USER] {human_msg}"
    assert contents[1] == "[ASSISTANT] I can explain that."
    provider._capture_identity_signals.assert_called_once_with(human_msg)


def test_sync_turn_preserves_interrupted_human_prompt(provider):
    """Mixed turns with system note envelope prefix strip the prefix and save genuine input."""
    raw = (
        "[System note: Your previous turn was interrupted mid-run — the app or its backend process stopped. "
        "The interrupted request was:]\n\n"
        "please build the data ingestion pipeline"
    )
    provider.sync_turn(raw, "Working on the pipeline.")

    calls = provider._beam.remember.call_args_list
    assert len(calls) == 2
    contents = [c.kwargs.get("content", "") for c in calls]
    assert contents[0] == "[USER] please build the data ingestion pipeline"
    assert contents[1] == "[ASSISTANT] Working on the pipeline."
    provider._capture_identity_signals.assert_called_once_with("please build the data ingestion pipeline")


@pytest.mark.parametrize("raw_input,expected_prompt", MIXED_NOTICES_AND_PROMPTS)
def test_sync_turn_preserves_mixed_notice_prompts(provider, raw_input, expected_prompt):
    """Mixed turns with machine notices followed by human text strip envelopes and save prompt."""
    provider.sync_turn(raw_input, "Acknowledged.")

    calls = provider._beam.remember.call_args_list
    assert len(calls) == 2
    contents = [c.kwargs.get("content", "") for c in calls]
    assert contents[0] == f"[USER] {expected_prompt}"
    assert contents[1] == "[ASSISTANT] Acknowledged."
    provider._capture_identity_signals.assert_called_once_with(expected_prompt)


def test_sync_turn_with_real_sqlite_storage():
    """End-to-end test with real SQLite storage confirming zero rows for notices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "real.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}
        identity_mock = MagicMock()
        prov._capture_identity_signals = identity_mock

        # 1. Sync machine notice
        prov.sync_turn(
            "[IMPORTANT: Background process proc_99 completed normally (exit code 0).\nCommand: ls\nOutput:\nok]",
            "Acknowledged."
        )

        # 2. Sync process notice with bracketed output
        proc_complex = (
            "[IMPORTANT: Background process proc_complex completed normally (exit code 0).\n"
            "Command: python -c \"print([1, 2, 3])\"\n"
            "Output:\n"
            "[1, 2, 3]\n"
            "result[0] = 42\n"
            "status: done]"
        )
        prov.sync_turn(proc_complex, "Acknowledged.")

        # 3. Sync full async delegation notice
        deleg_full = (
            "[ASYNC DELEGATION COMPLETE — deleg_abc123]\n"
            "A background subagent you dispatched earlier has finished. You may "
            "have moved on since dispatching it; the full task source is below so "
            "you can act on the result or re-dispatch if things have changed.\n\n"
            "Dispatched: 2026-10-14 10:00:00 (1m ago)\n"
            "Original goal: verify schema migrations\n"
            "Context you provided: use test database\n"
            "Toolsets: terminal, web\n"
            "Role: leaf   Model: gpt-5\n"
            "Status: completed   API calls: 3   Duration: 12.5s\n"
            "--- RESULT ---\n"
            "Migrations verified: 0 errors found."
        )
        prov.sync_turn(deleg_full, "Acknowledged.")

        # Identity signals must not have been captured for any machine notices
        identity_mock.assert_not_called()

        # 4. Sync genuine message
        prov.sync_turn("My favorite programming language is Python", "Python is great!")

        # 5. Sync mixed turn with notice followed by prompt
        prov.sync_turn(
            f"{proc_complex}\nPlease analyze the test results",
            "Understood."
        )

        # Verify rows in storage via recall
        assert len(prov._beam.recall("Background process", top_k=10)) == 0
        assert len(prov._beam.recall("proc_complex", top_k=5)) == 0
        assert len(prov._beam.recall("verify schema migrations", top_k=5)) == 0
        assert len(prov._beam.recall("deleg_abc123", top_k=5)) == 0

        py_results = prov._beam.recall("favorite programming language Python", top_k=5)
        assert len(py_results) > 0
        assert "favorite programming language is Python" in py_results[0]["content"]

        mixed_results = prov._beam.recall("Please analyze the test results", top_k=5)
        assert len(mixed_results) > 0
        assert "Please analyze the test results" in mixed_results[0]["content"]
        assert "proc_complex" not in mixed_results[0]["content"]

        # Identity capture called on genuine turns only
        identity_mock.assert_any_call("My favorite programming language is Python")
        identity_mock.assert_any_call("Please analyze the test results")
        assert identity_mock.call_count == 2


@pytest.mark.parametrize("classifier_mode", ["off", "warn", "strict"])
def test_sync_turn_classifier_modes_reject_machine_notices(classifier_mode, monkeypatch):
    """Machine notice rejection is deterministic and independent of write_classifier mode."""
    monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", classifier_mode)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "real.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}

        prov.sync_turn(
            "[IMPORTANT: Background process proc_strict completed normally (exit code 0).\nCommand: ls\nOutput:\nok]",
            "Acknowledged."
        )
        assert len(prov._beam.recall("Background process proc_strict", top_k=5)) == 0


def test_sync_turn_custom_ignore_patterns_still_apply():
    """Custom ignore_patterns still filter configured user patterns alongside machine notices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "real.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}
        prov._write_policy_overrides = {"ignore_patterns": ["^SKIP_THIS_PATTERN"]}

        # Machine notice dropped
        prov.sync_turn("[IMPORTANT: 2 background processes completed]", "ack")
        # Custom ignore pattern dropped
        prov.sync_turn("SKIP_THIS_PATTERN: confidential token", "ack")
        # Genuine user prompt retained
        prov.sync_turn("KEEP_THIS_PATTERN: legitimate technical insight", "ack")

        assert len(prov._beam.recall("background processes completed", top_k=5)) == 0
        assert len(prov._beam.recall("confidential token", top_k=5)) == 0
        retained = prov._beam.recall("legitimate technical insight", top_k=5)
        assert len(retained) > 0
        assert "legitimate technical insight" in retained[0]["content"]


def test_sync_turn_no_verbatim_ledger_capture_on_machine_notice(provider):
    """Verbatim ledger records zero captures when turn is a machine notice envelope."""
    ledger = MagicMock()
    ledger.enabled = True
    ledger.begin.return_value = None
    ledger.capture.side_effect = (
        lambda _key, _ticket, beam, _raw, **kwargs: beam.remember(**kwargs)
    )
    provider._verbatim_ledger = ledger
    provider._active_session_id = "test-session"
    provider.sync_turn(
        "[IMPORTANT: Background process proc_ledger completed normally (exit code 0).\nCommand: ls\nOutput:\nok]",
        "Acknowledged.",
        session_id="test-session",
    )
    ledger.capture.assert_not_called()

    # Mixed turn preserves trailing prompt in ledger capture
    provider._sync_roles = {"user"}
    provider.sync_turn(
        "[IMPORTANT: Background process job completed]\nPlease summarize the results",
        "ok",
        session_id="test-session",
    )
    assert ledger.capture.call_count == 1
    captured_raw = ledger.capture.call_args[0][3]
    assert captured_raw == "Please summarize the results"

    provider.sync_turn("A genuine user turn", "ok", session_id="test-session")
    assert ledger.capture.call_count == 2



def test_interrupted_quote_in_human_text_not_truncated():
    # ponytail: regression for dplush r3 — precheck must only fire on the real system-note boundary
    text = "Apa arti 'The interrupted request was: X' di log?"
    assert _sanitize_hermes_machine_envelope(text) == text


def test_context_compaction_tail_keeps_human_prompt_exactly():
    # ponytail: regression for dplush r3 — removeprefix, not char-set lstrip
    text = ("[CONTEXT COMPACTION] Earlier turns were compacted.\n"
            "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---\n"
            "solve this for me please")
    assert _sanitize_hermes_machine_envelope(text) == "solve this for me please"


def test_interrupted_note_without_boundary_keeps_later_human_quote():
    # ponytail: regression for dplush r4 — only the note's own boundary may cut the text
    text = "[System note: Your previous turn was interrupted mid-run.]\n\nhuman says: The interrupted request was: X and more"
    assert _sanitize_hermes_machine_envelope(text) == "human says: The interrupted request was: X and more"


def test_prior_context_repeated_handoff_keeps_human_prompt():
    # ponytail: regression for dplush r4 — exercises the repeated PRIOR CONTEXT branch
    marker = "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
    text = (f"[PRIOR CONTEXT — earlier]\nsummary A\n{marker}\n"
            f"[PRIOR CONTEXT — earlier]\nsummary B\n{marker}\nsolve this for me please")
    assert _sanitize_hermes_machine_envelope(text) == "solve this for me please"


def test_bracketed_human_question_about_notice_is_kept():
    # ponytail: dplush r5 — notice-like bracketed question must keep the human input
    text = "[IMPORTANT: Background process — what does that mean?]"
    assert _sanitize_hermes_machine_envelope(text) == text


def test_explanation_request_without_question_mark_is_kept():
    # ponytail: dplush r5 — explicit explanation request without '?' is human input
    text = "explain the background process notice you just got"
    assert _sanitize_hermes_machine_envelope(text) == text


def test_unmatched_recursive_fallback_drops_machine_content():
    # ponytail: dplush r5 — a machine notice whose inner pass rejects must not leak raw
    text = "[Your active task list was preserved across context compression]\n\n[IMPORTANT: Background process x completed]"
    assert _sanitize_hermes_machine_envelope(text) in (None, "")


def test_r7_notice_like_explanation_kept_through_sync_turn():
    # ponytail: dplush r7 — human explanation request must reach storage, not only the sanitizer
    text = "[IMPORTANT: Background process — explain what that means]"
    assert _sanitize_hermes_machine_envelope(text) == text


def test_r7_async_bracket_question_kept():
    text = "[ASYNC DELEGATION COMPLETE: explain what the subagent did]"
    assert _sanitize_hermes_machine_envelope(text) == text


def test_r7_process_payload_tail_stays_machine():
    # ponytail: dplush r7 — Command/Output payload must not resurface as a human prompt
    text = "[IMPORTANT: Background process p1 completed]\nCommand: ls\nOutput: how do I check?"
    assert _sanitize_hermes_machine_envelope(text) is None


def test_gateway_restart_notice_with_inquiry_words_suppresses_storage():
    # ponytail: dplush r9 — anonymous emitter-shaped gateway restart note with inquiry wording
    # must not revive into storage, and must suppress paired assistant autosave and identity capture
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}
        identity_mock = MagicMock()
        prov._capture_identity_signals = identity_mock

        notice = "[System note: The previous gateway session could not be recovered after a restart (API recovery timed out). Explain what happened.]"
        prov.sync_turn(notice, "The gateway was restarted automatically.")

        assert len(prov._beam.recall("gateway session", top_k=5)) == 0
        assert len(prov._beam.recall("restarted automatically", top_k=5)) == 0
        identity_mock.assert_not_called()


def test_task_list_notice_with_inquiry_item_suppresses_storage():
    # ponytail: dplush r10 — public Hermes TodoStore task list containing inquiry words
    # must not revive into storage, and must suppress paired assistant autosave and ledger capture
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}
        identity_mock = MagicMock()
        prov._capture_identity_signals = identity_mock
        ledger = MagicMock()
        prov._verbatim_ledger = ledger

        raw = (
            "[Your active task list was preserved across context compression]\n"
            "- [ ] 1. Check the anonymous report (pending)"
        )
        prov.sync_turn(raw, "Checking report.")

        assert len(prov._beam.recall("Check the anonymous report", top_k=5)) == 0
        assert len(prov._beam.recall("Checking report", top_k=5)) == 0
        identity_mock.assert_not_called()
        ledger.capture.assert_not_called()


def test_r7_notice_like_explanation_persists_through_sync_turn_storage(tmp_path):
    # ponytail: dplush r10 / coderabbit — human explanation request survives real sync_turn SQLite path
    db_path = tmp_path / "test.db"
    prov = MnemosyneMemoryProvider()
    prov._beam = BeamMemory(db_path=db_path)
    prov._sync_roles = {"user", "assistant"}
    identity_mock = MagicMock()
    prov._capture_identity_signals = identity_mock

    text = "[IMPORTANT: Background process — explain what that means]"
    prov.sync_turn(text, "It is an execution status notification.")

    results = prov._beam.recall("explain what that means", top_k=5)
    assert len(results) > 0
    assert text in results[0]["content"]
    assert "[USER]" in results[0]["content"]
    identity_mock.assert_called_once_with(text)
