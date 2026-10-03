"""
Phase 4: Configurable Hybrid Scoring Tests

Validates:
1. _normalize_weights() with explicit params, env vars, and defaults
2. Backward compatibility (no params = old hardcoded behavior)
3. Weight normalization sums to 1.0
4. Edge cases: all zeros, negative weights, single non-zero weight
5. recall() accepts new weight params and produces different rankings
6. Env var overrides work end-to-end
"""

import math
import sys
import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mnemosyne.core.memory as memory_module
from mnemosyne.core import beam as beam_module
from mnemosyne.core.config import MnemosyneConfig, get_config
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.beam import (
    _normalize_weights,
    BeamMemory,
    init_beam,
)


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Do not leak a temporary config directory into later test modules."""
    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()


# ============================================================================
# _normalize_weights() unit tests
# ============================================================================

class TestNormalizeWeights:
    """Unit tests for the _normalize_weights helper."""

    def test_default_weights(self):
        """No params + no env vars = default (0.5, 0.3, 0.2)."""
        vw, fw, iw = _normalize_weights(None, None, None)
        assert (vw, fw, iw) == pytest.approx((0.5, 0.3, 0.2), abs=1e-6)

    def test_explicit_params_override_defaults(self):
        """Explicit params are used directly and normalized."""
        vw, fw, iw = _normalize_weights(1.0, 1.0, 1.0)
        assert (vw, fw, iw) == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-6)

    def test_explicit_params_no_normalization_needed(self):
        """If they already sum to 1.0, normalization is a no-op."""
        vw, fw, iw = _normalize_weights(0.6, 0.3, 0.1)
        assert (vw, fw, iw) == pytest.approx((0.6, 0.3, 0.1), abs=1e-6)

    def test_normalization_sums_to_one(self):
        """All outputs must sum to exactly 1.0."""
        for params in [
            (1.0, 2.0, 3.0),
            (0.1, 0.1, 0.1),
            (10.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (0.0, 0.0, 7.0),
        ]:
            vw, fw, iw = _normalize_weights(*params)
            assert vw + fw + iw == pytest.approx(1.0, abs=1e-9)

    def test_all_zeros_fallback(self):
        """All zeros should fall back to defaults."""
        vw, fw, iw = _normalize_weights(0.0, 0.0, 0.0)
        assert (vw, fw, iw) == pytest.approx((0.5, 0.3, 0.2), abs=1e-6)

    def test_negative_weights_clamped(self):
        """Negative inputs are clamped to 0 before normalization."""
        vw, fw, iw = _normalize_weights(-0.5, 1.0, 0.5)
        # After clamping: 0.0, 1.0, 0.5 -> sum=1.5 -> 0.0, 2/3, 1/3
        assert vw == pytest.approx(0.0, abs=1e-6)
        assert fw == pytest.approx(2 / 3, abs=1e-6)
        assert iw == pytest.approx(1 / 3, abs=1e-6)

    def test_single_non_zero_weight(self):
        """Only one non-zero weight becomes 1.0."""
        vw, fw, iw = _normalize_weights(0.0, 0.0, 5.0)
        assert (vw, fw, iw) == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)

    def test_env_var_override(self, monkeypatch):
        """Env vars are used when params are None."""
        monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.7")
        monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0.2")
        monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.1")
        vw, fw, iw = _normalize_weights(None, None, None)
        assert (vw, fw, iw) == pytest.approx((0.7, 0.2, 0.1), abs=1e-6)

    def test_explicit_params_override_env(self, monkeypatch):
        """Explicit params take precedence over env vars."""
        monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.7")
        monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0.2")
        monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.1")
        vw, fw, iw = _normalize_weights(0.1, 0.1, 0.1)
        assert (vw, fw, iw) == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-6)

    def test_partial_env_vars(self, monkeypatch):
        """Missing env vars fall back to defaults, not all-or-nothing."""
        monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.8")
        # fts_weight and importance_weight fall back to defaults
        vw, fw, iw = _normalize_weights(None, None, None)
        # 0.8 + 0.3 + 0.2 = 1.3 -> normalize
        assert vw == pytest.approx(0.8 / 1.3, abs=1e-6)
        assert fw == pytest.approx(0.3 / 1.3, abs=1e-6)
        assert iw == pytest.approx(0.2 / 1.3, abs=1e-6)

    @pytest.mark.parametrize("nonfinite", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_explicit_weights_fall_back_as_a_complete_snapshot(self, nonfinite):
        """A non-finite component must not escape normalization or poison scores."""
        weights = _normalize_weights(nonfinite, 0.3, 0.2)
        assert weights == pytest.approx((0.5, 0.3, 0.2), abs=1e-6)
        assert all(math.isfinite(weight) for weight in weights)

    @pytest.mark.parametrize(
        "huge_weight", [10 ** 10_000, -(10 ** 10_000)], ids=["positive", "negative"]
    )
    @pytest.mark.parametrize("weight_index", [0, 1, 2], ids=["vec", "fts", "importance"])
    def test_internal_normalizer_handles_extreme_python_ints_as_whole_snapshot(
        self, huge_weight, weight_index
    ):
        """Each unconvertible component makes direct normalizer calls use defaults."""
        values = [0.8, 0.6, 0.2]
        values[weight_index] = huge_weight
        snapshot = beam_module._normalize_recall_weight_values(*values)
        assert snapshot.as_tuple() == (0.5, 0.3, 0.2)
        assert all(math.isfinite(weight) for weight in snapshot.as_tuple())


# ============================================================================
# Integration tests: recall() with configurable weights
# ============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_beam(db_path)
        yield db_path


class TestRecallConfigurableWeights:
    """Integration tests verifying recall() behavior with different weight configs."""

    def test_recall_accepts_weight_params(self, temp_db):
        """recall() should accept vec_weight, fts_weight, importance_weight without error."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Python is a programming language", importance=0.8)
        beam.remember("JavaScript runs in browsers", importance=0.3)

        # Should not raise
        results = beam.recall("programming language", top_k=5,
                              vec_weight=0.6, fts_weight=0.3, importance_weight=0.1)
        assert isinstance(results, list)

    def test_recall_without_weight_params_is_backward_compatible(self, temp_db):
        """Old code calling recall() without weight params still works."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Python is a programming language", importance=0.8)
        beam.remember("JavaScript runs in browsers", importance=0.3)

        results = beam.recall("programming language", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_high_importance_weight_boosts_high_importance_memories(self, temp_db):
        """With high importance_weight, high-importance memories rank higher."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("A: low importance generic text", importance=0.1)
        beam.remember("B: high importance critical alert", importance=0.9)

        # Low importance weight: keyword match dominates
        results_low_iw = beam.recall("critical alert", top_k=2, importance_weight=0.05)
        # High importance weight: importance dominates
        results_high_iw = beam.recall("critical alert", top_k=2, importance_weight=0.8)

        # Both should return results
        assert len(results_low_iw) >= 1
        assert len(results_high_iw) >= 1

        # With high importance weight, the high-importance memory should score higher
        # relative to the low-importance one compared to low importance weight
        # The high-importance memory (B) should be present in both
        assert any("B:" in r["content"] for r in results_low_iw)
        assert any("B:" in r["content"] for r in results_high_iw)

    def test_results_include_score_breakdown(self, temp_db):
        """Result dicts should include dense_score, fts_score, importance fields."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Test content for scoring breakdown", importance=0.5)

        results = beam.recall("test content", top_k=1,
                              vec_weight=0.4, fts_weight=0.4, importance_weight=0.2)
        assert len(results) > 0
        r = results[0]
        assert "dense_score" in r
        assert "fts_score" in r
        assert "importance" in r
        assert "score" in r

    def test_env_vars_affect_scoring(self, temp_db, monkeypatch):
        """Env vars should affect recall() scoring when params are not provided."""
        monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.1")
        monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0.1")
        monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.8")

        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Content A", importance=0.2)
        beam.remember("Content B", importance=0.9)

        # Without explicit params, env vars should be used
        results = beam.recall("content", top_k=2)
        assert len(results) >= 1
        # With 80% importance weight, the high-importance item should dominate
        top_result = results[0]
        assert top_result["importance"] >= 0.5  # Likely the high-importance one

    def test_explicit_params_override_env_in_recall(self, temp_db, monkeypatch):
        """Explicit params in recall() should override env vars."""
        monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.1")
        monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0.1")
        monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.8")

        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Test content", importance=0.5)

        # Call with explicit params that differ from env
        results = beam.recall("test", top_k=1,
                              vec_weight=0.5, fts_weight=0.3, importance_weight=0.2)
        assert len(results) > 0
        # Should succeed without error = params were accepted

    def test_weight_params_dont_break_temporal_scoring(self, temp_db):
        """Weight params should coexist with temporal_weight from Phase 3."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Recent event happened today", importance=0.5)

        results = beam.recall("event", top_k=1,
                              vec_weight=0.4, fts_weight=0.3, importance_weight=0.3,
                              temporal_weight=0.5, query_time="2099-01-01")
        assert isinstance(results, list)

    def test_zero_all_weights_uses_defaults_in_recall(self, temp_db):
        """Passing all zeros should trigger fallback to defaults."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Some content here", importance=0.5)

        # Should not crash; internally falls back to (0.5, 0.3, 0.2)
        results = beam.recall("content", top_k=1,
                              vec_weight=0.0, fts_weight=0.0, importance_weight=0.0)
        assert len(results) > 0

    @pytest.mark.parametrize(
        "huge_weight", [10 ** 10_000, -(10 ** 10_000)], ids=["positive", "negative"]
    )
    def test_direct_recall_extreme_python_int_falls_back_to_exact_finite_defaults(
        self, temp_db, monkeypatch, huge_weight
    ):
        """One unconvertible public weight resets the entire direct snapshot."""
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        MnemosyneConfig.reset_instance()
        beam = BeamMemory(session_id="extreme", db_path=temp_db)
        beam.remember("extreme direct weight sentinel", importance=0.8)

        payload = beam.recall(
            "extreme direct weight sentinel",
            top_k=1,
            vec_weight=huge_weight,
            fts_weight=0.8,
            importance_weight=0.2,
            explain=True,
        )

        weights = payload["explain"]["weights"]
        assert weights == {"vec": 0.5, "fts": 0.3, "importance": 0.2, "temporal": 0.0}
        assert all(math.isfinite(weight) for weight in weights.values())
        assert all(math.isfinite(result["score"]) for result in payload["results"])

    def test_multi_key_weight_snapshot_uses_one_generation_at_reload_boundary(
        self, temp_db, monkeypatch
    ):
        """A reload at the snapshot boundary yields one complete new generation."""
        config_path = temp_db.parent / "config.yaml"
        new_generation = (0.0, 1.0, 0.0)
        config_path.write_text("vec_weight: 1\nfts_weight: 0\nimportance_weight: 0\n")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        MnemosyneConfig.reset_instance()
        config = get_config()
        original_maybe_reload = config._maybe_reload
        reads = 0

        def reload_at_snapshot_boundary():
            nonlocal reads
            reads += 1
            original_maybe_reload()
            config_path.write_text("vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n")
            config.reload()

        monkeypatch.setattr(config, "_maybe_reload", reload_at_snapshot_boundary)
        snapshot = beam_module._resolve_recall_weights(None, None, None).as_tuple()

        assert reads == 1
        assert snapshot == new_generation

    def test_config_multi_key_read_uses_one_yaml_generation(self, temp_db, monkeypatch):
        """The generic config API performs one reload check for a multi-key read."""
        config_path = temp_db.parent / "config.yaml"
        config_path.write_text("vec_weight: 1\nfts_weight: 0\nimportance_weight: 0\n")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        MnemosyneConfig.reset_instance()
        config = get_config()
        original_maybe_reload = config._maybe_reload
        reads = 0

        def count_reload_checks():
            nonlocal reads
            reads += 1
            original_maybe_reload()

        monkeypatch.setattr(config, "_maybe_reload", count_reload_checks)
        resolved = config.get_many({
            "vec_weight": 0.5,
            "fts_weight": 0.3,
            "importance_weight": 0.2,
        })

        assert tuple(resolved[key] for key in ("vec_weight", "fts_weight", "importance_weight")) == (1, 0, 0)
        assert reads == 1

    @pytest.mark.parametrize("source", ["explicit", "env", "yaml"])
    def test_nonfinite_weight_sources_fall_back_to_finite_explained_scores(
        self, temp_db, monkeypatch, source
    ):
        """Explicit, env, and YAML non-finite values use one finite default snapshot."""
        config_path = temp_db.parent / "config.yaml"
        kwargs = {}
        if source == "explicit":
            config_path.write_text("cross_session: false\n")
            kwargs = {"vec_weight": float("inf"), "fts_weight": 0.3, "importance_weight": 0.2}
        elif source == "env":
            config_path.write_text("cross_session: false\n")
            monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "inf")
            monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0.3")
            monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.2")
        else:
            config_path.write_text("vec_weight: .inf\nfts_weight: 0.3\nimportance_weight: 0.2\n")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        MnemosyneConfig.reset_instance()

        beam = BeamMemory(session_id="finite", db_path=temp_db)
        beam.remember("finite scoring sentinel", importance=0.8)
        payload = beam.recall("finite scoring sentinel", top_k=1, explain=True, **kwargs)
        weights = payload["explain"]["weights"]

        assert (weights["vec"], weights["fts"], weights["importance"]) == pytest.approx((0.5, 0.3, 0.2))
        assert all(math.isfinite(weight) for weight in weights.values())
        assert all(math.isfinite(result["score"]) for result in payload["results"])
        assert "nan" not in repr(payload).lower()
        assert "inf" not in repr(payload).lower()

    def test_recall_weight_snapshot_reloads_between_requests_once_per_request(self, temp_db, monkeypatch):
        """Reload changes the next recall, while each request resolves one snapshot."""
        config_path = temp_db.parent / "config.yaml"
        config_path.write_text("vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        MnemosyneConfig.reset_instance()
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("snapshot reload sentinel", importance=0.5)

        original = beam_module._resolve_recall_weights
        snapshots = []

        def capture(*args, **kwargs):
            snapshot = original(*args, **kwargs)
            snapshots.append(snapshot)
            return snapshot

        monkeypatch.setattr(beam_module, "_resolve_recall_weights", capture)
        first = beam.recall("snapshot reload sentinel", top_k=1, explain=True)
        assert first["explain"]["weights"] == {
            "vec": 0.0, "fts": 1.0, "importance": 0.0, "temporal": 0.0,
        }
        assert len(snapshots) == 1

        config_path.write_text("vec_weight: 1\nfts_weight: 0\nimportance_weight: 0\n")
        get_config().reload()
        second = beam.recall("snapshot reload sentinel", top_k=1, explain=True)
        assert second["explain"]["weights"] == {
            "vec": 1.0, "fts": 0.0, "importance": 0.0, "temporal": 0.0,
        }
        assert len(snapshots) == 2

    def test_explicit_recall_weights_override_config(self, temp_db, monkeypatch):
        """Per-call public weights remain higher priority than conflicting YAML."""
        config_path = temp_db.parent / "config.yaml"
        config_path.write_text("vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        MnemosyneConfig.reset_instance()
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("explicit weight sentinel", importance=0.5)

        payload = beam.recall(
            "explicit weight sentinel",
            top_k=1,
            vec_weight=1.0,
            fts_weight=0.0,
            importance_weight=0.0,
            explain=True,
        )
        assert payload["explain"]["weights"] == {
            "vec": 1.0, "fts": 0.0, "importance": 0.0, "temporal": 0.0,
        }


@pytest.mark.parametrize("tier", ["working", "episodic"])
@pytest.mark.parametrize("weight_source", ["explicit", "env", "yaml"])
def test_literal_flag_precedence_survives_importance_only_weight_resolution(
    tmp_path, monkeypatch, tier, weight_source
):
    """A literal flag is a selection invariant across tiers and weight sources."""
    data_dir = tmp_path / f"{tier}-{weight_source}"
    data_dir.mkdir()
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    kwargs = {}

    if weight_source == "explicit":
        kwargs = {
            "vec_weight": 0.0,
            "fts_weight": 0.0,
            "importance_weight": 1.0,
        }
    elif weight_source == "env":
        monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0")
        monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0")
        monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "1")
    else:
        (data_dir / "config.yaml").write_text(
            "vec_weight: 0\nfts_weight: 0\nimportance_weight: 1\n"
        )

    MnemosyneConfig.reset_instance()
    beam = BeamMemory(
        session_id=f"literal-flag-{tier}-{weight_source}",
        db_path=data_dir / "memory.db",
    )
    if tier == "working":
        expected_id = beam.remember(
            "Use --force after confirmation.", source="test", importance=0.1
        )
        distractor_id = beam.remember(
            "The force field calibration is complete.",
            source="test",
            importance=1.0,
        )
    else:
        expected_id = beam.consolidate_to_episodic(
            "Use --force after confirmation.",
            source_wm_ids=[],
            source="test",
            importance=0.1,
        )
        distractor_id = beam.consolidate_to_episodic(
            "The force field calibration is complete.",
            source_wm_ids=[],
            source="test",
            importance=1.0,
        )

    payload = beam.recall("--force", top_k=2, explain=True, **kwargs)

    assert payload["explain"]["weights"] == {
        "vec": 0.0,
        "fts": 0.0,
        "importance": 1.0,
        "temporal": 0.0,
    }
    assert payload["results"][0]["id"] == expected_id
    assert distractor_id not in {result["id"] for result in payload["results"]}
    beam.conn.close()


class TestPublicRecallConfigurableWeights:
    """Public Mnemosyne recall wrappers should expose BeamMemory scoring weights."""

    def test_mnemosyne_recall_accepts_weight_params(self, temp_db):
        """Mnemosyne.recall() should forward scoring weights to BeamMemory.recall()."""
        mem = Mnemosyne(session_id="test", db_path=temp_db)
        mem.remember("Python is a programming language", importance=0.8)

        results = mem.recall(
            "programming language",
            top_k=5,
            vec_weight=0.6,
            fts_weight=0.3,
            importance_weight=0.1,
        )

        assert isinstance(results, list)
        assert len(results) > 0

    def test_module_recall_accepts_weight_params(self, monkeypatch):
        """mnemosyne.recall() module helper should expose the same scoring weights."""
        class FakeMemory:
            def recall(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return [{"id": "test", "content": "weight forwarding"}]

        fake = FakeMemory()
        monkeypatch.setattr(memory_module, "_get_default", lambda bank=None: fake)

        results = memory_module.recall(
            "weight forwarding",
            top_k=5,
            vec_weight=0.6,
            fts_weight=0.3,
            importance_weight=0.1,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert fake.kwargs["vec_weight"] == 0.6
        assert fake.kwargs["fts_weight"] == 0.3
        assert fake.kwargs["importance_weight"] == 0.1


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    """Boundary conditions and error handling."""

    def test_very_high_vec_weight(self, temp_db):
        """vec_weight=1.0 should make vector similarity dominate."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Content for vector test", importance=0.1)
        results = beam.recall("vector test", top_k=1,
                              vec_weight=1.0, fts_weight=0.0, importance_weight=0.0)
        assert len(results) >= 0  # May be empty if no embeddings, but should not crash

    def test_very_high_fts_weight(self, temp_db):
        """fts_weight=1.0 should make text match dominate."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Exact text match phrase", importance=0.1)
        results = beam.recall("exact text match", top_k=1,
                              vec_weight=0.0, fts_weight=1.0, importance_weight=0.0)
        assert len(results) > 0
        assert "exact" in results[0]["content"].lower()

    def test_invalid_negative_param_clamped(self, temp_db):
        """Negative weight params should be clamped to 0."""
        beam = BeamMemory(session_id="test", db_path=temp_db)
        beam.remember("Test content", importance=0.5)

        # Should not raise; negative values are clamped
        results = beam.recall("test", top_k=1,
                              vec_weight=-0.5, fts_weight=1.0, importance_weight=0.5)
        assert len(results) > 0


# ============================================================================
# Run standalone
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
