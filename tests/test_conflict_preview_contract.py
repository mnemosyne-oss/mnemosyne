"""Exercise aggregate previews with real SQLite, detector and verdict parsing."""

import json

import pytest

from mnemosyne.core import llm_conflict_detector as lcd
from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.cost_log import init_cost_log


@pytest.fixture
def preview_beam(tmp_path, monkeypatch):
    """Create a disposable resolver with validation enabled and no model calls."""
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    # remember() must never initialize local models or call an embedding API.
    # The detector below reads the deterministic vectors stored by _add_pair.
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)
    beam = BeamMemory(session_id="preview", db_path=tmp_path / "preview.db")
    init_cost_log(beam.db_path)
    yield beam
    beam.conn.close()


def _add_pair(beam, source="preference"):
    """Store a cross-session pair with matching deterministic detector vectors."""
    ids = []
    for index, text in enumerate((
        "I prefer orchard apples as my regular breakfast while travelling in winter.",
        "My orchard breakfast preference has changed to pears; please never serve apples.",
    )):
        memory_id = beam.remember(text, source=source, scope="global", importance=0.8)
        assert memory_id
        beam.conn.execute(
            "UPDATE working_memory SET timestamp=?, session_id=? WHERE id=?",
            (f"2026-11-01T0{index * 2}:00:00+00:00", f"{source}-{index}", memory_id),
        )
        beam.conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings(memory_id, embedding_json) VALUES(?, ?)",
            (memory_id, json.dumps([1.0] + [0.0] * 383)),
        )
        ids.append(memory_id)
    beam.conn.commit()
    return ids


def _memory_state(beam):
    """Snapshot the memory rows separately from permitted validation cost logs."""
    return {
        table: [tuple(row) for row in beam.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("working_memory", "episodic_memory", "memory_embeddings")
    }


def _assert_aggregate(result):
    """Pin the existing public counter payload without adding pair diagnostics."""
    assert set(result) == {
        "status", "pairs_flagged", "conflicts_resolved", "invalidated",
        "rows_scanned", "llm_validations", "llm_cap_reached", "candidates_truncated",
    }


def test_default_preview_never_calls_model_or_logs_costs(preview_beam, monkeypatch):
    """An enabled validator still requires explicit llm_eval for previews."""
    beam = preview_beam
    _add_pair(beam)
    before = _memory_state(beam)

    def unexpected_call(*args, **kwargs):
        pytest.fail("default preview must not call the model transport")

    monkeypatch.setattr(lcd, "_call_conflict_llm_with_retry", unexpected_call)
    result = beam.resolve_cross_session_conflicts(dry_run=True)
    _assert_aggregate(result)
    assert result["pairs_flagged"] == result["conflicts_resolved"] == 1
    assert result["llm_validations"] == result["invalidated"] == 0
    assert _memory_state(beam) == before
    assert beam.conn.execute("SELECT count(*) FROM cost_entries").fetchone()[0] == 0


@pytest.mark.parametrize("response, confirmed, costs", [
    pytest.param(None, 0, 0, id="unavailable"),
    pytest.param(("not JSON", 10, 5), 0, 0, id="malformed"),
    pytest.param((json.dumps({"is_conflict": "false", "confidence": 0.8}), 10, 5),
                 0, 0, id="string-verdict"),
    pytest.param((json.dumps({"is_conflict": True, "confidence": 2.0}), 10, 5),
                 0, 0, id="out-of-range-confidence"),
    pytest.param((json.dumps({"is_conflict": True, "confidence": 0.8,
                             "correct_fact": 123}), 10, 5),
                 0, 0, id="non-string-correct-fact"),
    pytest.param((json.dumps({"is_conflict": False, "confidence": 0.8}), 10, 5),
                 0, 1, id="negative"),
    pytest.param((json.dumps({"is_conflict": True, "confidence": 0.8}), 10, 5),
                 1, 1, id="positive"),
])
def test_evaluated_preview_counts_attempts_without_supersession(
    preview_beam, monkeypatch, response, confirmed, costs,
):
    """Real parsing distinguishes cost writes while the payload remains aggregate."""
    beam = preview_beam
    _add_pair(beam)
    before = _memory_state(beam)
    calls = []

    def transport(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(lcd, "_call_conflict_llm_with_retry", transport)
    result = beam.resolve_cross_session_conflicts(dry_run=True, llm_eval=True)
    _assert_aggregate(result)
    assert result["status"] == "dry_run"
    assert result["pairs_flagged"] == result["llm_validations"] == len(calls) == 1
    assert result["conflicts_resolved"] == confirmed and result["invalidated"] == 0
    assert _memory_state(beam) == before
    rows = beam.conn.execute(
        "SELECT session_id, memory_count, token_count, model FROM cost_entries"
    ).fetchall()
    assert len(rows) == costs
    if costs:
        assert tuple(rows[0]) == ("preview", 2, 15, lcd.CONFLICT_LLM_MODEL)


def test_no_candidates_differs_from_an_unconfirmed_candidate(preview_beam, monkeypatch):
    """The real detector reports zero attempts once stored vectors are orthogonal."""
    beam = preview_beam
    ids = _add_pair(beam)
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(lcd, "_call_conflict_llm_with_retry", transport)
    unconfirmed = beam.resolve_cross_session_conflicts(dry_run=True, llm_eval=True)
    assert unconfirmed["pairs_flagged"] == unconfirmed["llm_validations"] == 1
    beam.conn.execute(
        "UPDATE memory_embeddings SET embedding_json=? WHERE memory_id=?",
        (json.dumps([0.0, 1.0] + [0.0] * 382), ids[1]),
    )
    beam.conn.commit()
    before = _memory_state(beam)
    empty = beam.resolve_cross_session_conflicts(dry_run=True, llm_eval=True)
    _assert_aggregate(empty)
    assert empty["pairs_flagged"] == empty["llm_validations"] == 0
    assert empty["conflicts_resolved"] == unconfirmed["conflicts_resolved"] == 0
    assert len(calls) == 1
    assert _memory_state(beam) == before


def test_evaluated_preview_stops_at_validation_cap(preview_beam, monkeypatch):
    """Two source groups leave a pending pair after one real validator attempt."""
    beam = preview_beam
    _add_pair(beam, "first")
    _add_pair(beam, "second")
    before = _memory_state(beam)
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(lcd, "_call_conflict_llm_with_retry", transport)
    result = beam.resolve_cross_session_conflicts(
        dry_run=True, llm_eval=True, max_llm_validations=1,
    )
    _assert_aggregate(result)
    assert result["pairs_flagged"] == 2
    assert result["llm_validations"] == len(calls) == 1 and result["llm_cap_reached"]
    assert result["conflicts_resolved"] == result["invalidated"] == 0
    assert _memory_state(beam) == before
