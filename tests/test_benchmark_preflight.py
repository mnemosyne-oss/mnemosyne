"""Preflight regression tests for the BEAM benchmark harness.

Pre-fix, `_benchmarks/evaluate_beam_end_to_end.py` ran with harness oracles
(TR timeline, CR injection, IE/KU `_context_facts`, RECENT CONVERSATION
raw-message injection) by default -- without pure-recall mode the
oracles produce answers that bypass `BeamMemory.recall()`, contaminating
arm-vs-arm comparisons.

Post-fix the harness refuses to run unless either:
  - `MNEMOSYNE_BENCHMARK_PURE_RECALL=1` (or `--pure-recall`), or
  - `--allow-harness-oracles` (explicit opt-in for ceiling tests / legacy
    reproduction)

These tests pin the preflight gate. They subprocess the harness in
`--help` / argument-parsing mode rather than running a full benchmark.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "_benchmarks" / "evaluate_beam_end_to_end.py"


@pytest.fixture
def clean_env(monkeypatch):
    """Clear all benchmark-mode env vars so each test starts from a
    known state."""
    monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
    monkeypatch.delenv("FULL_CONTEXT_MODE", raising=False)
    return monkeypatch


def _run_harness(*args, env_overrides=None):
    """Invoke the harness with the given CLI args + env overrides.
    Returns CompletedProcess. Uses --dry-run when possible to avoid
    actually loading the BEAM dataset."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    # Always include --sample 0 + --scales (a single small scale) so we
    # don't accidentally hit the full pipeline if dry-run isn't enough.
    return subprocess.run(
        [sys.executable, str(_HARNESS), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


class TestPreflightRefusesWithoutPureRecall:
    """When neither pure-recall nor --allow-harness-oracles is set,
    the harness must exit with a non-zero status BEFORE doing any work
    (no dataset load, no LLM calls)."""

    def test_no_flags_no_env_exits_with_error(self, clean_env):
        """Default invocation -- should refuse."""
        result = _run_harness("--sample", "1", "--scales", "100K")
        assert result.returncode == 2, (
            f"Expected exit code 2 (preflight refusal); got {result.returncode}.\n"
            f"stdout: {result.stdout[:400]}\nstderr: {result.stderr[:400]}"
        )
        assert "harness oracles are active" in result.stderr, (
            f"Expected preflight error message in stderr; got: {result.stderr[:400]}"
        )

    def test_full_context_alone_is_not_enough(self, clean_env):
        """`FULL_CONTEXT_MODE=1` doesn't disable the oracles -- it adds
        a different bypass. Should still refuse without pure-recall."""
        result = _run_harness(
            "--sample", "1", "--scales", "100K",
            env_overrides={"FULL_CONTEXT_MODE": "1"},
        )
        assert result.returncode == 2

    def test_pure_recall_flag_satisfies_preflight(self, clean_env):
        """`--pure-recall` enables the bypass-disabling mode; preflight
        should let the run proceed (it may fail later for unrelated
        reasons like missing API key, but not at preflight)."""
        result = _run_harness("--pure-recall", "--dry-run")
        # Either it proceeded past preflight (returncode != 2) or it
        # failed for some OTHER reason (dataset / network). The
        # preflight-error string is what we're checking is absent.
        assert "harness oracles are active" not in result.stderr

    def test_pure_recall_env_satisfies_preflight(self, clean_env):
        result = _run_harness(
            "--dry-run",
            env_overrides={"MNEMOSYNE_BENCHMARK_PURE_RECALL": "1"},
        )
        assert "harness oracles are active" not in result.stderr

    def test_pure_recall_env_accepts_on(self, clean_env):
        """C31 helper accepts `on`; preflight should honor that."""
        result = _run_harness(
            "--dry-run",
            env_overrides={"MNEMOSYNE_BENCHMARK_PURE_RECALL": "on"},
        )
        assert "harness oracles are active" not in result.stderr

    def test_allow_harness_oracles_explicit_opt_in(self, clean_env):
        """Operators that explicitly want the legacy bypass behavior
        (ceiling tests, pre-fix reproduction) can opt in."""
        result = _run_harness("--allow-harness-oracles", "--dry-run")
        assert "harness oracles are active" not in result.stderr

    def test_preflight_runs_before_dataset_load(self, clean_env):
        """The preflight should fail BEFORE attempting to load the
        BEAM dataset, so operators with no HuggingFace access still
        get a clean error message."""
        result = _run_harness("--sample", "1", "--scales", "100K")
        # If we got past preflight to dataset loading, we'd see
        # 'Loading BEAM dataset' in stdout. Should not be there.
        assert "Loading BEAM dataset" not in result.stdout
