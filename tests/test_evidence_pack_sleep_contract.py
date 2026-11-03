"""Characterize the existing evidence-pack boundaries after real Sleep.

Related: #545, #789, #979, #1266. These are current-contract tests, not an
archive-recovery policy, semantic calibration, or answer-accuracy benchmark.
Only inference is controlled; SQLite/FTS, Sleep, recall and pack assembly run.
Explicit source=fact isolates pack exclusion from default dense eligibility.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mnemosyne.core import beam as bm, local_llm, model_refresh
from mnemosyne.core.config import MnemosyneConfig

QUERY = "Orion gateway access"
ORIGINAL = "Orion gateway access uses cobaltcommand."
CURRENT = "Orion gateway access uses jadecommand."
SUMMARY = "A maintenance event was recorded."
COLLEAGUE_QUERY = "my opal gateway access"
COLLEAGUE = (
    "My colleague configures opal gateway access using silvercommand. "
    "This is about my colleague, not my own configuration."
)
UNKNOWN = "opal gateway access status remains unknown."


def _ids(rows):
    """Return row IDs in the order produced by real recall or pack assembly."""
    return [row["id"] for row in rows]


def _remember(beam, memory_id, content, *, session_id="reader", importance=0.6):
    """Store one global synthetic fact and restore the caller's session binding."""
    previous = beam.session_id, beam.channel_id
    beam.session_id = beam.channel_id = session_id
    try:
        result = beam.remember(
            content, source="fact", importance=importance, scope="global",
            memory_id=memory_id, veracity="stated", dedupe=False,
            extract_entities=False,
        )
        assert result == memory_id
    finally:
        beam.session_id, beam.channel_id = previous


@pytest.fixture(params=["linear", "polyphonic"])
def slept_memory(request, tmp_path, monkeypatch):
    """Yield an isolated store after real Sleep with controlled inference."""
    config = tmp_path / "data"
    config.mkdir()
    (config / "config.yaml").write_text("cross_session: true\n", encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(config))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("MNEMOSYNE_LLM_CONFLICT_DETECTION", "0")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", str(int(request.param == "polyphonic")))
    MnemosyneConfig.reset_instance()

    query_vector = np.zeros(bm.EMBEDDING_DIM, dtype=np.float32)
    query_vector[0] = 1
    orthogonal = np.zeros(bm.EMBEDDING_DIM, dtype=np.float32)
    orthogonal[1] = 1

    def embed(texts):
        """Give the summary, primary and other facts distinct fixed directions."""
        vectors = []
        for text in texts:
            if text == SUMMARY:
                vector = orthogonal
            elif text in {CURRENT, UNKNOWN}:
                vector = query_vector
            else:
                vector = 0.8 * query_vector + 0.6 * orthogonal
            vectors.append(vector)
        return np.array(vectors)

    # Patch inference before constructing the DB: no model load or download.
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(bm._embeddings, "embed", embed)
    monkeypatch.setattr(bm._embeddings, "embed_query", lambda query: query_vector)
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda lines, **kw: [lines])
    monkeypatch.setattr(local_llm, "_summarize_memories", lambda *a, **kw: SUMMARY)
    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", lambda items: [])

    beam = bm.BeamMemory(session_id="writer", db_path=tmp_path / "bank.db")
    try:
        _remember(beam, "archived", ORIGINAL, session_id="writer", importance=0.4)
        assert "archived" in _ids(
            beam.recall(QUERY, source="fact", top_k=5, _track_recall=False)
        )
        receipt = beam.sleep(force=True, _fleet_census=False)
        assert receipt["items_consolidated"] == 1
        assert receipt["summaries_created"] == 1
        row = beam.conn.execute(
            "SELECT content, consolidated_at, consolidation_claimed_at "
            "FROM working_memory WHERE id='archived'"
        ).fetchone()
        assert row["content"] == ORIGINAL
        assert row["consolidated_at"] is not None
        assert row["consolidation_claimed_at"] is None
        assert beam.conn.execute(
            "SELECT summary_of FROM episodic_memory"
        ).fetchone()[0] == "archived"
        beam.session_id = beam.channel_id = "reader"
        yield beam
    finally:
        beam.conn.close()
        MnemosyneConfig.reset_instance()


def _pack(beam, backend, monkeypatch, *, query=QUERY):
    """Run real pack retrieval and check primary order and WM recall credits."""
    calls = []
    recall = beam.recall

    def observe(*args, **kwargs):
        """Record recall boundaries while preserving its results and side effects."""
        rows = recall(*args, **kwargs)
        calls.append({
            "top_k": kwargs.get("top_k"),
            "track": kwargs.get("_track_recall", True),
            "ids": _ids(rows),
        })
        return rows

    # Both arms store genuine vectors. Only WM native-index availability differs.
    # Polyphonic WM already uses JSON, so its two arms are not independent algorithms.
    with monkeypatch.context() as context:
        if backend == "json":
            context.setattr(bm, "_wm_vec_available", lambda conn: False)
        elif not bm._wm_vec_available(beam.conn):
            pytest.skip("sqlite-vec extension required for the native WM arm")
        plain = recall(query, source="fact", top_k=1, _track_recall=False)
        usage_before = dict(beam.conn.execute(
            "SELECT id, COALESCE(recall_count, 0) FROM working_memory"
        ).fetchall())
        context.setattr(beam, "recall", observe)
        packed = beam.recall_with_evidence_pack(
            query, source="fact", top_k=1, candidate_k=8, pack_k=5
        )
        usage_after = dict(beam.conn.execute(
            "SELECT id, COALESCE(recall_count, 0) FROM working_memory"
        ).fetchall())

    assert _ids(plain) == _ids(packed["primary"]) == ["primary"]
    assert [(call["top_k"], call["track"]) for call in calls] == [(1, True), (8, False)]
    # Check actual storage effects, not just the spy's _track_recall flag.
    assert usage_after["primary"] == usage_before["primary"] + 1
    for memory_id in usage_before:
        if memory_id != "primary":
            assert usage_after[memory_id] == usage_before[memory_id]
    return calls[1]["ids"], packed["evidence_pack"]


@pytest.mark.parametrize("backend", ["sqlite", "json"])
def test_sleep_original_reaches_candidates_but_is_excluded_from_pack(
    slept_memory, backend, monkeypatch
):
    _remember(slept_memory, "primary", CURRENT, importance=0.99)
    candidates, pack = _pack(slept_memory, backend, monkeypatch)
    assert "archived" in candidates
    assert pack == []


@pytest.mark.parametrize("backend", ["sqlite", "json"])
def test_current_detail_from_primary_session_is_not_a_supplement(
    slept_memory, backend, monkeypatch
):
    _remember(slept_memory, "sibling", "Orion gateway access secondary detail uses amberkey.")
    _remember(slept_memory, "primary", CURRENT, importance=0.99)
    candidates, pack = _pack(slept_memory, backend, monkeypatch)
    assert "sibling" in candidates
    assert pack == []


@pytest.mark.parametrize("backend", ["sqlite", "json"])
def test_authorized_colleague_context_can_enter_pack(
    slept_memory, backend, monkeypatch
):
    _remember(slept_memory, "colleague", COLLEAGUE, session_id="colleague-writer")
    _remember(slept_memory, "primary", UNKNOWN, importance=0.99)
    candidates, pack = _pack(
        slept_memory, backend, monkeypatch, query=COLLEAGUE_QUERY
    )
    assert "colleague" in candidates
    assert _ids(pack) == ["colleague"]
    assert pack[0]["session_id"] == "colleague-writer"
    assert pack[0]["content"] == COLLEAGUE


def test_optional_hermes_lexical_gate_does_not_resolve_subject(monkeypatch):
    """Helper-only companion check, not pack-to-provider integration."""
    plugin = pytest.importorskip("mnemosyne_hermes")
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE", raising=False)
    assert plugin._prefetch_has_distinctive_lexical_evidence(COLLEAGUE_QUERY, COLLEAGUE)
