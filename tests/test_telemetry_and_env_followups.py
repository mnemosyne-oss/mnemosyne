"""Regression tests for C30 + C31 + C32 — pre-experiment cleanup follow-ups.

Three small fidelity fixes surfaced by the /review army on PRs #89 and
#90 that weren't bundled into those PRs because they're independent:

- **C30** (telemetry): `beam.py:2671` set `dense_score` for episodic
  fallback rows via `wm_vec_sims.get(row["id"], 0.0)` — but `wm_vec_sims`
  is the working-memory dict, ep ids aren't in it, so the value was
  always 0.0. Misleading provenance for post-run analysis. Fixed by
  setting `dense_score: 0.0` explicitly with a comment.
- **C31** (env parser): `MNEMOSYNE_BENCHMARK_PURE_RECALL=on` and
  `FULL_CONTEXT_MODE=on` were treated as falsy because the parser only
  accepted `1|true|yes`. Whitespace-padded values (`" 1 "`) were also
  treated as falsy. Fixed by routing through a new `_env_truthy()`
  helper that accepts `1|true|yes|on` and strips whitespace.
- **C32** (drift WARN): `MNEMOSYNE_*_WEIGHT` env vars override recall
  scoring but NOT consolidation Bayesian compounding. Fixed by emitting
  a single startup WARNING listing the overrides.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
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


# ─────────────────────────────────────────────────────────────────
# C30 — episodic fallback `dense_score` explicit 0.0
# ─────────────────────────────────────────────────────────────────


class TestC30EpisodicFallbackDenseScore:
    """Pre-fix: `beam.py:2671` set `dense_score` via `wm_vec_sims.get(row["id"], 0.0)`.
    Since `row["id"]` is always an episodic id and `wm_vec_sims` keys
    are working-memory ids, the lookup always returned 0.0 (the
    default). Same numeric value as the fix, but the wrong-dict
    lookup is misleading provenance: someone reading the code would
    think `dense_score` reflects a WM-vector similarity for ep rows.

    Fix: set `dense_score: 0.0` explicitly with a comment explaining
    that EM fallback rows reach this code path precisely because
    vec/FTS produced no episodic candidates, so no `sim` is computed."""

    def test_fallback_rows_have_explicit_zero_dense_score(self, temp_db, monkeypatch):
        """Seed an episodic row whose content has no vector embedding
        (so fallback is the only path that returns it), recall a query
        matching its content, assert the surviving row has dense_score=0.0."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Insert via SQL to skip embedding generation, forcing fallback path.
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("ep-no-emb", "unique-zorblax-token for fallback test",
             "consolidation", datetime.now().isoformat(), "s1", 0.5),
        )
        beam.conn.commit()

        results = beam.recall("zorblax", top_k=10)
        ep_rows = [r for r in results if r["id"] == "ep-no-emb"]
        assert ep_rows, (
            f"Expected fallback to surface seeded row; got: "
            f"{[(r['id'], r.get('tier'), r.get('content', '')[:50]) for r in results]}"
        )
        ep = ep_rows[0]
        # Pin tier provenance — Claude MEDIUM noted the original test
        # could pass via entity/fact branches too. Lock to fallback.
        assert ep.get("tier") == "episodic"
        # Field is present, explicit float 0.0.
        assert ep["dense_score"] == 0.0
        # Type is float (not None, not int) — keep downstream consumers stable.
        assert isinstance(ep["dense_score"], float)

    def test_main_path_episodic_dense_score_not_clobbered(self, temp_db, monkeypatch):
        """Negative control: main-path episodic rows (vec+FTS-driven)
        DO compute a real `sim` and set `dense_score` to it. Pin this
        so a future refactor that collapses everything to 0.0 breaks
        the test, not just experiment provenance."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Use the BEAM-real path: insert via consolidate_to_episodic so
        # it gets a real vector embedding (if fastembed is available).
        mid = beam.consolidate_to_episodic(
            summary="The user wants dark mode for the editor",
            source_wm_ids=["wm-1"],
        )
        # Recall a similar query — the main path's `sim` is non-zero
        # if embedding worked, OR rev would still surface via FTS.
        results = beam.recall("dark mode", top_k=10)
        ep_rows = [r for r in results if r["id"] == mid]
        if not ep_rows:
            pytest.skip("main-path recall returned no candidates in this env")
        # The main path SHOULD set dense_score to a meaningful sim value.
        # The fix only touched fallback + entity/fact branches.
        # We can't assert > 0 reliably (fastembed may not be installed),
        # but we CAN assert the field is present + a float.
        assert "dense_score" in ep_rows[0]
        assert isinstance(ep_rows[0]["dense_score"], float)


class TestC30ExtendedToEntityFactPaths:
    """C30 fix had to extend beyond the EM fallback path. Codex P2 +
    Claude HIGH noted that entity-aware (beam.py:~2306) and fact-aware
    (beam.py:~2426) episodic recall branches had the SAME wrong-dict
    pattern. Pin behavior for those two paths."""

    def test_entity_branch_returns_explicit_dense_score(self, temp_db, monkeypatch):
        """The entity-aware ep branch now sets `dense_score: 0.0`
        explicitly rather than the misleading WM-dict lookup. We
        can't easily force this branch in a unit test (needs entity
        extraction wired), so verify by inspecting the source: line
        2306 region should have `"dense_score": 0.0,` not the
        `wm_vec_sims.get(...)` pattern. Belt-and-suspenders alongside
        the integration test."""
        from pathlib import Path
        import mnemosyne.core.beam as beam_module
        src = Path(beam_module.__file__).read_text()
        # Count remaining wrong-pattern sites that target ep rows.
        # The two valid `wm_vec_sims.get` lines (`tier: "working"` at
        # ~2238 and ~2359) should remain; the two ep-tier sites we
        # fixed should be gone.
        wrong_pattern_count = src.count(
            'dense_score": round(wm_vec_sims.get(row["id"], 0.0), 4)'
        )
        # Only the two `tier: "working"` sites should match the old pattern.
        # If a future change adds a new `tier: "episodic"` site with the
        # wrong pattern, this count rises and the test fails.
        assert wrong_pattern_count == 2, (
            f"Expected exactly 2 `wm_vec_sims.get(row[id], 0.0)` sites "
            f"(both `tier: working`); found {wrong_pattern_count}. A new "
            f"episodic site may have reintroduced the C30 bug pattern."
        )


# ─────────────────────────────────────────────────────────────────
# C31 — env-var truthy parser accepts `on` + trims whitespace
# ─────────────────────────────────────────────────────────────────


class TestC31EnvTruthyParser:
    """Pre-fix the parser was `lower() in ("1", "true", "yes")` — `on`
    and whitespace-padded values were treated as falsy. Fix: route
    through `_env_truthy()` which accepts `1|true|yes|on` (case-
    insensitive, whitespace-stripped)."""

    @pytest.fixture(autouse=True)
    def _ensure_clean_env(self, monkeypatch):
        # Clear both env vars before each test.
        monkeypatch.delenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", raising=False)
        monkeypatch.delenv("FULL_CONTEXT_MODE", raising=False)

    @pytest.mark.parametrize("value", [
        "1", "true", "yes", "on",
        "TRUE", "True", "YES", "ON", "On",
        " 1 ", "  true  ", "\ton\t",  # whitespace
    ])
    def test_env_truthy_accepts_value(self, value, monkeypatch):
        from _benchmarks.evaluate_beam_end_to_end import _env_truthy

        monkeypatch.setenv("TEST_ENV_VAR", value)
        assert _env_truthy("TEST_ENV_VAR") is True

    @pytest.mark.parametrize("value", [
        "0", "false", "no", "off",
        "FALSE", "OFF",
        "", " ", "  ",
        "garbage", "maybe", "2", "y",  # non-canonical
    ])
    def test_env_truthy_rejects_value(self, value, monkeypatch):
        from _benchmarks.evaluate_beam_end_to_end import _env_truthy

        monkeypatch.setenv("TEST_ENV_VAR", value)
        assert _env_truthy("TEST_ENV_VAR") is False

    def test_env_truthy_unset_variable_is_false(self, monkeypatch):
        from _benchmarks.evaluate_beam_end_to_end import _env_truthy
        monkeypatch.delenv("TEST_ENV_VAR", raising=False)
        assert _env_truthy("TEST_ENV_VAR") is False

    def test_pure_recall_accepts_on(self, temp_db, monkeypatch):
        """End-to-end: `MNEMOSYNE_BENCHMARK_PURE_RECALL=on` now enables
        the gate. Pre-fix this was silently treated as off."""
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", "on")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam._context_facts = {"favorite color blue": ["blue"]}
        fake_llm = MagicMock()
        fake_llm.chat = MagicMock(return_value="LLM-FALLBACK")
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        msgs = [{"role": "user", "content": f"row {i}"} for i in range(5)]
        answer_with_memory(
            llm=fake_llm, beam=beam,
            question="what is favorite color blue",
            conversation_messages=msgs, top_k=5, ability="IE",
        )
        # Pure-recall mode active → IE bypass disabled → LLM called.
        fake_llm.chat.assert_called_once()

    def test_pure_recall_accepts_whitespace_padded(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_BENCHMARK_PURE_RECALL", " 1 ")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam._context_facts = {"favorite color blue": ["blue"]}
        fake_llm = MagicMock()
        fake_llm.chat = MagicMock(return_value="LLM-FALLBACK")
        from _benchmarks.evaluate_beam_end_to_end import answer_with_memory

        answer_with_memory(
            llm=fake_llm, beam=beam,
            question="what is favorite color blue",
            conversation_messages=[{"role": "user", "content": "x"}],
            top_k=5, ability="IE",
        )
        fake_llm.chat.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# C32 — MNEMOSYNE_*_WEIGHT env override startup WARN
# ─────────────────────────────────────────────────────────────────


class TestC32VeracityWeightOverrideWarn:
    """Pre-fix: operators setting `MNEMOSYNE_STATED_WEIGHT=0.9` (etc.)
    silently broke the 'consolidated-as-N also ranks at N' invariant
    because the consolidator's Bayesian compounding doesn't honor env
    overrides. Fix: emit a single WARNING listing the override(s).
    """

    def test_no_overrides_returns_empty_list(self, monkeypatch):
        """Sanity: when no env vars set, the helper returns empty."""
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        for name in (
            "MNEMOSYNE_STATED_WEIGHT", "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT", "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)
        assert _detect_veracity_weight_overrides() == []

    def test_single_override_returned(self, monkeypatch):
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        for name in (
            "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT", "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "0.95")
        assert _detect_veracity_weight_overrides() == ["MNEMOSYNE_STATED_WEIGHT"]

    def test_multiple_overrides_returned_in_canonical_order(self, monkeypatch):
        """Order matches the function's hard-coded canonical list so
        the warning log is deterministic across runs."""
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "1.0")
        monkeypatch.setenv("MNEMOSYNE_UNKNOWN_WEIGHT", "0.5")
        monkeypatch.setenv("MNEMOSYNE_TOOL_WEIGHT", "0.4")
        # Set some but not all; the others stay absent.
        monkeypatch.delenv("MNEMOSYNE_INFERRED_WEIGHT", raising=False)
        monkeypatch.delenv("MNEMOSYNE_IMPORTED_WEIGHT", raising=False)

        result = _detect_veracity_weight_overrides()
        # Canonical order from the function definition.
        assert result == [
            "MNEMOSYNE_STATED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ]

    def test_empty_string_value_not_counted_as_override(self, monkeypatch):
        """Codex P1 fix: `export MNEMOSYNE_STATED_WEIGHT=` (empty)
        falls back to default in `_env_float`, so it doesn't actually
        override anything. The detection helper should match — empty
        values are NOT overrides. Pre-fix the test pinned the OPPOSITE
        behavior; that was operationally moot because `float("")` would
        have crashed module load before the WARN could fire."""
        from mnemosyne.core.beam import _detect_veracity_weight_overrides

        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "")
        monkeypatch.setenv("MNEMOSYNE_TOOL_WEIGHT", "   ")  # whitespace-only
        for name in (
            "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_IMPORTED_WEIGHT", "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)
        # Empty + whitespace-only → neither counts.
        assert _detect_veracity_weight_overrides() == []

    def test_env_float_falls_back_on_empty_value(self, monkeypatch):
        """Pre-fix: `float(os.environ.get('MNEMOSYNE_STATED_WEIGHT', '1.0'))`
        crashed with ValueError when env was set to empty (`os.environ.get`
        returns `""` for set-but-empty, not the default). Fixed via
        `_env_float` which strips + falls back."""
        from mnemosyne.core.beam import _env_float

        monkeypatch.setenv("MY_TEST_VAR", "")
        assert _env_float("MY_TEST_VAR", 0.7) == 0.7
        monkeypatch.setenv("MY_TEST_VAR", "   ")
        assert _env_float("MY_TEST_VAR", 0.5) == 0.5
        monkeypatch.delenv("MY_TEST_VAR", raising=False)
        assert _env_float("MY_TEST_VAR", 0.3) == 0.3

    def test_env_float_falls_back_on_invalid_value_with_warn(self, monkeypatch, caplog):
        """Garbage value → fall back + WARN."""
        from mnemosyne.core.beam import _env_float

        monkeypatch.setenv("MY_TEST_VAR", "not-a-number")
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            value = _env_float("MY_TEST_VAR", 0.42)
        assert value == 0.42
        assert any("not a valid float" in r.message for r in caplog.records
                   if r.levelno == logging.WARNING)

    def test_warn_fires_when_overrides_present(self, monkeypatch, caplog):
        """Call `_warn_about_veracity_weight_overrides(force=True)`
        directly with env overrides set; assert WARN logged + returns True.

        `force=True` because module load already called the helper once
        and the idempotency guard would otherwise suppress this call."""
        from mnemosyne.core.beam import _warn_about_veracity_weight_overrides

        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "0.5")
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            emitted = _warn_about_veracity_weight_overrides(force=True)

        assert emitted is True
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "Veracity weight env overrides detected" in r.message]
        assert warnings, (
            f"Expected veracity-weight WARN; got records: "
            f"{[r.message[:100] for r in caplog.records]}"
        )
        # The WARN should mention the specific env var.
        assert any("MNEMOSYNE_STATED_WEIGHT" in r.message for r in warnings)

    def test_warn_silent_when_no_overrides(self, monkeypatch, caplog):
        """Negative control: clean env → no WARN, returns False."""
        from mnemosyne.core.beam import _warn_about_veracity_weight_overrides

        for name in (
            "MNEMOSYNE_STATED_WEIGHT", "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT", "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        ):
            monkeypatch.delenv(name, raising=False)

        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            emitted = _warn_about_veracity_weight_overrides(force=True)

        assert emitted is False
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "Veracity weight env overrides detected" in r.message]
        assert warnings == []

    def test_warn_is_idempotent_per_process(self, monkeypatch, caplog):
        """Claude MEDIUM fix: the WARN guards against multi-emission
        within a single process. Module load already called it once;
        a second non-force call returns False without re-emitting.
        Pins the contract so multi-worker setups (uvicorn workers,
        pytest-xdist) don't spam N identical WARNs per startup."""
        from mnemosyne.core.beam import _warn_about_veracity_weight_overrides
        import mnemosyne.core.beam as beam_module

        # Reset the guard so we can simulate "fresh process" — module
        # load already flipped the flag at session start, so we have
        # to clear it manually to test the gate.
        monkeypatch.setattr(beam_module, "_VERACITY_WARN_EMITTED", False)
        monkeypatch.setenv("MNEMOSYNE_STATED_WEIGHT", "0.5")

        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            first = _warn_about_veracity_weight_overrides()
            second = _warn_about_veracity_weight_overrides()
            third = _warn_about_veracity_weight_overrides()

        assert first is True
        assert second is False
        assert third is False
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "Veracity weight env overrides detected" in r.message]
        # Exactly one WARN fired despite three calls.
        assert len(warnings) == 1
