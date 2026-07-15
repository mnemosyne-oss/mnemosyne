"""Recall calibration, full-content and FTS synchronization regressions."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core import polyphonic_recall as polyphonic_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.polyphonic_recall import PolyphonicResult


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, 1.0),
        (math.sqrt(0.5), 0.75),
        (math.sqrt(2.0), 0.0),
        (2.0, 0.0),
    ],
)
def test_float32_l2_distance_uses_normalized_vector_geometry(distance, expected):
    assert beam_module._vec_distance_to_similarity(distance, "float32") == pytest.approx(expected)


def test_in_memory_cosine_distance_uses_one_minus_distance():
    assert beam_module._cosine_distance_to_similarity(0.2) == pytest.approx(0.8)


def test_polyphonic_float32_distance_uses_linear_recall_geometry():
    assert polyphonic_module._vector_distance_to_similarity(
        math.sqrt(0.5), "float32"
    ) == pytest.approx(0.75)


def test_polyphonic_numpy_cosine_keeps_absolute_cosine_score():
    assert polyphonic_module._cosine_to_similarity(0.5) == pytest.approx(0.5)
    assert polyphonic_module._cosine_to_similarity(-0.2) == pytest.approx(0.0)


def test_polish_question_glue_is_not_a_lexical_relevance_signal():
    assert beam_module._recall_tokens("Jakie jest hasło dla gości") == ["hasło", "gości"]


def test_vector_only_admission_requires_calibrated_top1_and_margin(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_WM_VECTOR_ONLY_MIN_SIM", raising=False)
    monkeypatch.delenv("MNEMOSYNE_WM_VECTOR_ONLY_MIN_MARGIN", raising=False)
    ranked = [{"id": "correct", "sim": 0.82}, {"id": "runner-up", "sim": 0.79}]

    assert beam_module._admit_wm_vector_only("correct", ranked) is True
    assert beam_module._admit_wm_vector_only("runner-up", ranked) is False
    assert beam_module._admit_wm_vector_only(
        "ambiguous", [{"id": "ambiguous", "sim": 0.82}, {"id": "other", "sim": 0.81}]
    ) is False
    assert beam_module._admit_wm_vector_only(
        "weak", [{"id": "weak", "sim": 0.80}, {"id": "other", "sim": 0.70}]
    ) is False


def test_vector_only_single_candidate_uses_absolute_threshold(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_WM_VECTOR_ONLY_MIN_SIM", raising=False)
    assert beam_module._admit_wm_vector_only("only", [{"id": "only", "sim": 0.82}]) is True
    assert beam_module._admit_wm_vector_only("only", [{"id": "only", "sim": 0.80}]) is False


def test_single_episodic_candidate_keeps_absolute_dense_score(tmp_path, monkeypatch):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    now = datetime.now().isoformat()
    cursor = memory.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, importance, session_id, scope, veracity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("ep-one", "hydraulika zaworu proporcjonalnego", "test", now, 0.5, "s1", "session", "stated"),
    )
    rowid = cursor.lastrowid
    memory.conn.commit()

    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed_query",
        lambda _query: np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(beam_module, "_vec_available", lambda _conn: True)
    monkeypatch.setattr(beam_module, "_effective_vec_type", lambda _conn, table="vec_episodes": "float32")
    monkeypatch.setattr(
        beam_module,
        "_vec_search",
        lambda _conn, _embedding, k=20: [{"rowid": rowid, "distance": math.sqrt(0.4)}],
    )
    monkeypatch.setattr(beam_module, "_fts_search", lambda _conn, _query, k=20: [])

    results = memory.recall("hydraulika", top_k=3)

    episodic = next(row for row in results if row["id"] == "ep-one")
    assert episodic["dense_score"] == pytest.approx(0.8, abs=1e-4)


def test_episodic_vector_only_noise_below_calibrated_threshold_abstains(tmp_path, monkeypatch):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    now = datetime.now().isoformat()
    cursor = memory.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, importance, session_id, scope, veracity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("ep-noise", "hydraulika zaworu proporcjonalnego", "test", now, 0.5, "s1", "session", "stated"),
    )
    rowid = cursor.lastrowid
    memory.conn.commit()

    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed_query",
        lambda _query: np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(beam_module, "_vec_available", lambda _conn: True)
    monkeypatch.setattr(beam_module, "_effective_vec_type", lambda _conn, table="vec_episodes": "float32")
    monkeypatch.setattr(
        beam_module,
        "_vec_search",
        lambda _conn, _embedding, k=20: [{"rowid": rowid, "distance": math.sqrt(0.4)}],
    )
    monkeypatch.setattr(beam_module, "_fts_search", lambda _conn, _query, k=20: [])

    results = memory.recall("zzzx qqqy wwwv", top_k=3)

    assert all(row["id"] != "ep-noise" for row in results)


@pytest.mark.parametrize(("similarity", "expected_ids"), [(0.80, []), (0.82, ["wm-poly"])])
def test_polyphonic_vector_only_uses_calibrated_abstention(
    tmp_path, monkeypatch, similarity, expected_ids
):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    memory.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES ('wm-poly', 'hydraulika zaworu proporcjonalnego', 'test', ?, 's1')",
        (datetime.now().isoformat(),),
    )
    memory.conn.commit()

    class FakeEngine:
        def recall(self, **_kwargs):
            return [PolyphonicResult(
                memory_id="wm-poly",
                combined_score=1.0 / 61.0,
                voice_scores={"vector": 1.0 / 61.0},
                metadata={"raw_voice_scores": {"vector": similarity}},
            )]

    monkeypatch.setattr(memory, "_get_polyphonic_engine", lambda: FakeEngine())
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)

    results = memory._recall_polyphonic("zzzx qqqy wwwv", top_k=3)

    assert [row["id"] for row in results] == expected_ids


def test_polyphonic_vector_margin_ignores_filtered_foreign_session(tmp_path, monkeypatch):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    timestamp = datetime.now().isoformat()
    memory.conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, scope) "
        "VALUES (?, ?, 'test', ?, ?, 'session')",
        [
            ("wm-visible", "hydraulika zaworu", timestamp, "s1"),
            ("wm-foreign", "sprężarka powietrza", timestamp, "s2"),
        ],
    )
    memory.conn.commit()

    class FakeEngine:
        def recall(self, **_kwargs):
            return [
                PolyphonicResult(
                    memory_id="wm-visible",
                    combined_score=1.0 / 61.0,
                    voice_scores={"vector": 1.0 / 61.0},
                    metadata={"raw_voice_scores": {"vector": 0.82}},
                ),
                PolyphonicResult(
                    memory_id="wm-foreign",
                    combined_score=1.0 / 62.0,
                    voice_scores={"vector": 1.0 / 62.0},
                    metadata={"raw_voice_scores": {"vector": 0.81}},
                ),
            ]

    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
    monkeypatch.setattr(beam_module, "_CROSS_SESSION", False)
    monkeypatch.setattr(memory, "_get_polyphonic_engine", lambda: FakeEngine())
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)

    results = memory._recall_polyphonic("zzzx qqqy wwwv", top_k=3)

    assert [row["id"] for row in results] == ["wm-visible"]


def test_recall_returns_full_content_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_RECALL_CONTENT_CHARS", raising=False)
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    sentinel = "KONIEC-PEŁNEJ-TREŚCI"
    content = "hydraulika " + ("bardzo-długi-opis " * 45) + sentinel
    memory.remember(content, source="test", importance=0.8)

    results = memory.recall("hydraulika", top_k=3)

    row = next(item for item in results if item["tier"] == "working")
    assert len(row["content"]) > 500
    assert sentinel in row["content"]


def test_recall_content_truncation_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_RECALL_CONTENT_CHARS", "20")
    assert beam_module._format_recall_content("0123456789abcdefghijTAIL") == "0123456789abcdefghij"


def test_facts_fts_is_synchronized_after_update(tmp_path):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    memory.conn.execute(
        "INSERT INTO facts(subject, predicate, object, source_msg_id, confidence, session_id) VALUES (?, ?, ?, ?, ?, ?)",
        ("pompa", "ma", "ciśnienie", "msg-1", 1.0, "s1"),
    )
    memory.conn.commit()
    rowid = memory.conn.execute("SELECT rowid FROM facts WHERE source_msg_id='msg-1'").fetchone()[0]
    assert memory.conn.execute(
        "SELECT rowid FROM fts_facts WHERE fts_facts MATCH 'pompa'"
    ).fetchone()[0] == rowid

    memory.conn.execute(
        "UPDATE facts SET subject=?, object=? WHERE rowid=?",
        ("zawór", "przepływ", rowid),
    )
    memory.conn.commit()

    assert memory.conn.execute(
        "SELECT rowid FROM fts_facts WHERE fts_facts MATCH 'zawór'"
    ).fetchone()[0] == rowid
    assert memory.conn.execute(
        "SELECT rowid FROM fts_facts WHERE fts_facts MATCH 'pompa'"
    ).fetchone() is None
