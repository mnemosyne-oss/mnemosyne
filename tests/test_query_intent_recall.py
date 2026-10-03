"""Regression tests for query-intent recall weighting."""

import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.config import MnemosyneConfig


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path: Path):
    """Keep configuration singleton state and config seeding test-local."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("cross_session: false\n")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(config_dir))
    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()


def test_query_intent_classification_and_weight_adjustment():
    from mnemosyne.core.query_intent import adjust_weights, classify_intent

    temporal = classify_intent("what happened last week")
    assert temporal.category == "temporal"
    tv, tf, ti = adjust_weights(0.5, 0.3, 0.2, temporal)
    assert round(tv + tf + ti, 6) == 1.0
    assert tf > 0.3
    assert tv < 0.5

    procedural = classify_intent("how do I deploy this service")
    assert procedural.category == "procedural"
    pv, pf, pi = adjust_weights(0.5, 0.3, 0.2, procedural)
    assert round(pv + pf + pi, 6) == 1.0
    assert pv > 0.5
    assert pi < 0.2

    preference = classify_intent("which option should I choose")
    assert preference.category == "preference"
    _, _, pref_i = adjust_weights(0.5, 0.3, 0.2, preference)
    assert pref_i > 0.2


def test_recall_works_with_query_intent_enabled_and_disabled(monkeypatch):
    from mnemosyne.core.beam import BeamMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "mnemosyne.db"
        beam = BeamMemory(session_id="test", db_path=db_path)
        beam.remember("Yesterday we configured the deployment workflow", importance=0.8)
        beam.remember("The user prefers compact direct answers", importance=0.9)

        monkeypatch.delenv("MNEMOSYNE_QUERY_INTENT", raising=False)
        off_results = beam.recall("what happened yesterday", top_k=5)
        assert off_results

        monkeypatch.setenv("MNEMOSYNE_QUERY_INTENT", "1")
        on_results = beam.recall("what happened yesterday", top_k=5)
        assert on_results
        assert any("Yesterday" in r.get("content", "") for r in on_results)


def test_explicit_recall_weights_override_query_intent(monkeypatch):
    from mnemosyne.core.beam import BeamMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "mnemosyne.db"
        beam = BeamMemory(session_id="test", db_path=db_path)
        beam.remember("Last week we changed the deployment workflow", importance=0.7)

        monkeypatch.setenv("MNEMOSYNE_QUERY_INTENT", "1")
        results = beam.recall(
            "what happened last week",
            top_k=5,
            vec_weight=0.2,
            fts_weight=0.7,
            importance_weight=0.1,
        )
        assert results


def test_recall_explain_reports_intent_adjusted_weights(monkeypatch):
    """Explain output must report the weights used by direct recall scoring."""
    from mnemosyne.core import query_intent
    from mnemosyne.core.beam import BeamMemory

    monkeypatch.setenv("MNEMOSYNE_QUERY_INTENT", "1")
    monkeypatch.setattr(query_intent, "classify_intent", lambda query: object())
    monkeypatch.setattr(query_intent, "adjust_weights", lambda *args, **kwargs: (0.2, 0.7, 0.1))

    with tempfile.TemporaryDirectory() as tmpdir:
        beam = BeamMemory(session_id="test", db_path=Path(tmpdir) / "mnemosyne.db")
        beam.remember("Last week we changed the deployment workflow", importance=0.7)
        payload = beam.recall("what happened last week", top_k=5, explain=True)

    assert payload["explain"]["weights"] == pytest.approx({
        "vec": 0.2, "fts": 0.7, "importance": 0.1, "temporal": 0.0,
    })


def test_public_enhanced_recall_keeps_explicit_weights_from_query_intent(monkeypatch):
    """Enhanced intent classification must not rewrite public caller weights."""
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core.memory import Mnemosyne

    adjust_calls = []

    def fail_if_adjusted(*args, **kwargs):
        adjust_calls.append((args, kwargs))
        raise AssertionError("explicit recall weights must skip intent adjustment")

    monkeypatch.setattr(beam_module, "adjust_weights", fail_if_adjusted)
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_QUERY_INTENT", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Mnemosyne(session_id="test", db_path=Path(tmpdir) / "mnemosyne.db")
        try:
            memory.remember("Last week we changed the deployment workflow", importance=0.7)

            payload = memory.recall(
                "what happened last week",
                top_k=5,
                vec_weight=0.2,
                fts_weight=0.7,
                importance_weight=0.1,
                explain=True,
            )

            assert payload["explain"]["weights"] == pytest.approx({
                "vec": 0.2, "fts": 0.7, "importance": 0.1, "temporal": 0.0,
            })
            assert adjust_calls == []
        finally:
            memory.conn.close()


def test_public_enhanced_recall_resolves_weight_defaults_and_overrides(monkeypatch):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core.memory import Mnemosyne

    observed_base_weights = []
    original_adjust_weights = beam_module.adjust_weights

    def capture_adjust_weights(*args, **kwargs):
        observed_base_weights.append(
            (kwargs["base_vec"], kwargs["base_fts"], kwargs["base_importance"])
        )
        return original_adjust_weights(*args, **kwargs)

    monkeypatch.setattr(beam_module, "adjust_weights", capture_adjust_weights)
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.delenv("MNEMOSYNE_VEC_WEIGHT", raising=False)
    monkeypatch.delenv("MNEMOSYNE_FTS_WEIGHT", raising=False)
    monkeypatch.delenv("MNEMOSYNE_IMPORTANCE_WEIGHT", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("cross_session: false\n")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(config_dir))
        db_path = Path(tmpdir) / "mnemosyne.db"
        memory = Mnemosyne(session_id="test", db_path=db_path)
        try:
            memory.remember("The sprint ends on March 29.", scope="global")

            default_results = memory.recall("When does the sprint end?", top_k=3)
            explicit_results = memory.recall(
                "What is the sprint deadline?",
                top_k=3,
                vec_weight=0.0,
                fts_weight=1.0,
                importance_weight=0.0,
            )
            monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.2")
            monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0.7")
            monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.1")
            configured_results = memory.recall("When is the sprint due?", top_k=3)

            assert default_results[0]["content"] == "The sprint ends on March 29."
            assert explicit_results[0]["content"] == "The sprint ends on March 29."
            assert configured_results[0]["content"] == "The sprint ends on March 29."
            assert [
                tuple(round(weight, 6) for weight in weights)
                for weights in observed_base_weights
            ] == [
                (0.5, 0.3, 0.2),
                (0.2, 0.7, 0.1),
            ]
        finally:
            memory.conn.close()
