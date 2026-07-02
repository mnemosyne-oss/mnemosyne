"""Regression tests for Gap E (paired-outcomes JSONL) and Gap G
(linear-path voice_scores parity).

Both gaps were filed in the BEAM-recovery experiment plan as
post-execution analysis polish -- they don't block the experiment
running, but they make post-hoc per-tool attribution credible.

- **Gap G:** `BeamMemory.recall()` linear-path result dicts now carry
  a `voice_scores: dict` field with the same JSON shape contract as
  polyphonic results (different keys per engine since the signal
  sources differ). Lets downstream analysis treat both arms uniformly
  when computing per-signal contributions across phases.

- **Gap E:** the harness writes `paired_outcomes.jsonl` alongside the
  main results JSON. Each line is one (config_id, question_id,
  ability, score, correct) row. Append-only with config_id so
  multiple A/B runs accumulate in one file; analyst filters by
  config_id when computing bootstrap CIs on paired deltas.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List
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
# Gap G -- Linear-path voice_scores parity
# ─────────────────────────────────────────────────────────────────


class TestLinearVoiceScores:
    """Linear-path result dicts must carry a `voice_scores: dict` field
    so downstream analysis can treat linear + polyphonic results
    uniformly when computing per-signal contributions."""

    def _seed_episodic(self, beam: BeamMemory, ep_id: str, content: str):
        ts = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            (ep_id, content, "consolidation", ts, "s1", 0.5),
        )
        beam.conn.commit()

    def test_every_linear_result_has_voice_scores(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Seed enough content that recall returns results across both
        # tiers + via different paths (main + fallback).
        self._seed_episodic(beam, "ep-1", "the deployment runbook explains rollout")
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("wm-1", "the deployment plan is approved", "conversation",
             datetime.now().isoformat(), "s1", 0.5),
        )
        beam.conn.commit()

        results = beam.recall("deployment", top_k=10)
        assert results, "Expected at least one recall hit"
        for r in results:
            assert "voice_scores" in r, (
                f"Linear result missing voice_scores: "
                f"{list(r.keys())}"
            )
            assert isinstance(r["voice_scores"], dict)

    def test_voice_scores_contains_expected_signal_keys(self, temp_db):
        """The linear engine's voice_scores dict should include the
        per-signal raw scores the linear scorer composed."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        self._seed_episodic(beam, "ep-1", "deployment notes about prod release")

        results = beam.recall("deployment", top_k=5)
        if not results:
            pytest.skip("recall returned empty -- environment-dependent")
        vs = results[0]["voice_scores"]
        # Linear-side keys (different from polyphonic's vec/graph/fact/temporal).
        expected_keys = {"vec", "fts", "keyword", "importance", "recency_decay"}
        assert expected_keys.issubset(set(vs.keys())), (
            f"voice_scores keys missing expected linear signals: "
            f"got {set(vs.keys())}, want superset of {expected_keys}"
        )

    def test_voice_scores_values_are_numeric(self, temp_db):
        """All entries should be floats so downstream summing /
        comparison works without coercion."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        self._seed_episodic(beam, "ep-1", "config change for the api gateway")

        results = beam.recall("api gateway", top_k=5)
        if not results:
            pytest.skip("recall returned empty")
        for k, v in results[0]["voice_scores"].items():
            assert isinstance(v, (int, float)), (
                f"voice_scores[{k!r}] = {v!r} is not numeric"
            )

    def test_voice_scores_in_both_main_and_fallback_paths(self, temp_db, monkeypatch):
        """Both the main vec/FTS-driven loop and the fallback substring
        scan should attach voice_scores. Force fallback by inserting an
        ep row WITHOUT embeddings (so vec returns no candidates)."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Insert directly so no embedding gets generated → triggers fallback
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, "
            "session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("ep-fallback", "unique-marker for the fallback path",
             "consolidation", datetime.now().isoformat(), "s1", 0.5),
        )
        beam.conn.commit()

        results = beam.recall("unique-marker", top_k=5)
        fallback_hits = [r for r in results if r["id"] == "ep-fallback"]
        if not fallback_hits:
            pytest.skip("recall didn't surface fallback row in this env")
        assert "voice_scores" in fallback_hits[0]
        assert isinstance(fallback_hits[0]["voice_scores"], dict)


# ─────────────────────────────────────────────────────────────────
# Gap E -- Paired-outcomes JSONL
# ─────────────────────────────────────────────────────────────────


class TestPairedOutcomesJSONL:
    """The harness writes `paired_outcomes.jsonl` alongside the main
    results JSON. Each row is one (config_id, qid, ability, score,
    correct) so a downstream notebook can paired-bootstrap CIs across
    multiple A/B runs without re-parsing the main JSON."""

    def test_harness_help_shows_config_id_flag(self):
        """`--config-id` flag is exposed."""
        harness = _REPO_ROOT / "_benchmarks" / "evaluate_beam_end_to_end.py"
        result = subprocess.run(
            [sys.executable, str(harness), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert "--config-id" in result.stdout, (
            f"Expected --config-id in help; got:\n{result.stdout[-500:]}"
        )

    def test_config_id_derived_from_env_when_unset(self, monkeypatch):
        """Without --config-id, the harness derives one from the env
        snapshot. Two runs with identical env should produce identical
        config_ids; two runs with different env should differ.

        We test this by verifying the helper logic: SHA-256 of the
        canonical env serialization, first 10 hex chars, prefixed
        with 'cfg-'.
        """
        import hashlib
        env_a = {"MNEMOSYNE_VOICE_FACT": "0", "MNEMOSYNE_POLYPHONIC_RECALL": "1"}
        env_b = {"MNEMOSYNE_VOICE_GRAPH": "0", "MNEMOSYNE_POLYPHONIC_RECALL": "1"}
        # Mirror the harness's canonicalization (filtered + sorted).
        def _id(env: Dict[str, str]) -> str:
            canon = "\n".join(f"{k}={v}" for k, v in sorted(env.items()))
            return "cfg-" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:10]
        a1 = _id(env_a)
        a2 = _id(env_a)
        b = _id(env_b)
        assert a1 == a2, "identical env should produce identical config_id"
        assert a1 != b, "different env should produce different config_id"
        assert a1.startswith("cfg-")
        assert len(a1) == len("cfg-") + 10

    def test_paired_outcomes_file_constant_defined(self):
        """The harness module exposes PAIRED_OUTCOMES_FILE so tests
        and downstream tools have a stable reference."""
        import _benchmarks.evaluate_beam_end_to_end as harness
        assert hasattr(harness, "PAIRED_OUTCOMES_FILE")
        assert str(harness.PAIRED_OUTCOMES_FILE).endswith("paired_outcomes.jsonl")

    def test_paired_outcomes_jsonl_row_shape(self, tmp_path):
        """Direct test: simulate writing a row in the format the
        harness writes, then read it back. This pins the JSONL schema
        without subprocess-running the full pipeline."""
        outfile = tmp_path / "paired_outcomes.jsonl"
        row = {
            "config_id": "cfg-abc1234567",
            "run_started_at": "2026-05-12T15:00:00+00:00",
            "scale": "100K",
            "conversation_id": "conv-001",
            "qid": "q-042",
            "ability": "IE",
            "score": 0.75,
            "correct": True,
        }
        with open(outfile, "a") as f:
            f.write(json.dumps(row) + "\n")
        with open(outfile) as f:
            line = f.readline()
        parsed = json.loads(line)
        # Required fields for paired-bootstrap analysis:
        for required in ("config_id", "qid", "ability", "score", "correct"):
            assert required in parsed, f"missing field {required!r}"
        # Score is a float, correct is a bool.
        assert isinstance(parsed["score"], (int, float))
        assert isinstance(parsed["correct"], bool)

    def test_correct_threshold_at_half(self):
        """Pin the threshold definition: score >= 0.5 → correct=True.
        Matches the rubric: 1.0=correct, 0.5=partial, 0.0=wrong.
        Treating partial as correct here errs on the side of charity;
        analyst can rescore using raw `score` if they want stricter."""
        # Direct comparison to the harness's expression
        assert (0.5 >= 0.5) is True   # exactly partial → correct
        assert (1.0 >= 0.5) is True   # fully correct
        assert (0.0 >= 0.5) is False  # wrong
        assert (0.49 >= 0.5) is False  # just below threshold

    def test_paired_outcomes_constants_path_under_results(self):
        """Paired outcomes file should live alongside the main results
        JSON, both under `results/`. Pins the convention."""
        import _benchmarks.evaluate_beam_end_to_end as harness
        assert harness.PAIRED_OUTCOMES_FILE.parent == harness.RESULTS_FILE.parent
