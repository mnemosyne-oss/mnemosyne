"""Working-memory vector arm: exact blob scoring for int8 stores (#982).

The working-memory dense blend consumes the ``sim`` values returned by
``_wm_vec_search``. On an int8 vec store the legacy
``1 - distance / (2 * EMBEDDING_DIM)`` mapping is a scale guess: every candidate
lands in a narrow high band (0.92-0.95 measured on int8[768]), so the blend
keeps the ordering but loses the amplitude and cannot re-rank a gold row above a
distractor. These tests pin the fixed behaviour:

* int8 candidates are scored from their stored bytes with the shared
  ``_vec_int8_blob_cosine`` helper,
* score separation is real (a controlled cosine separation survives instead of
  being squeezed to ~0.03),
* an int8 candidate whose blob cannot be read is **not** scored from its
  distance: the arm abstains and the exact compatibility scan takes over,
* other arms keep their mapping unchanged,
* ``BeamMemory.recall()`` separates a gold row from a distractor through
  ``dense_score``.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

import mnemosyne.core.beam as beam_module
from mnemosyne.core.beam import (
    BeamMemory,
    _wm_vec_row_sim,
    _wm_vec_search,
)


def _load_vec(conn):
    """Load sqlite-vec exactly like the application does (if available)."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        return True
    except Exception:
        return False


VEC_AVAILABLE = None


def vec_supports_int8():
    """sqlite-vec present AND supports int8 quantization probes."""
    global VEC_AVAILABLE
    if VEC_AVAILABLE is None:
        try:
            c = sqlite3.connect(":memory:")
            VEC_AVAILABLE = _load_vec(c) and c.execute(
                "SELECT vec_quantize_int8('[0.1, 0.2]', 'unit')"
            ).fetchone() is not None
            c.close()
        except Exception:
            VEC_AVAILABLE = False
    return VEC_AVAILABLE


requires_vec = pytest.mark.skipif(
    not vec_supports_int8(), reason="sqlite-vec int8 support not available"
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _int8_blob(values):
    return pytest.importorskip("numpy").array(values, dtype="int8").tobytes()


def _legacy_sim(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance / (2.0 * beam_module.EMBEDDING_DIM)))


def _query_vector(values):
    np = pytest.importorskip("numpy")
    v = np.array(values, dtype="float32")
    return v / np.linalg.norm(v)


def test_row_sim_scores_int8_candidates_from_blobs():
    """Blob cosine replaces the saturated legacy mapping when blobs are present."""
    q = _int8_blob([127, 0, 0, 0])
    same = _int8_blob([127, 0, 0, 0])
    angled = _int8_blob([102, 76, 0, 0])  # cos ~ 0.80
    orthogonal = _int8_blob([0, 127, 0, 0])

    # The legacy mapping reports ~1.0 for every one of these distances.
    assert _wm_vec_row_sim(0.0, "int8", q, same) == pytest.approx(1.0, abs=0.01)
    assert _wm_vec_row_sim(0.0, "int8", q, angled) == pytest.approx(0.80, abs=0.02)
    assert _wm_vec_row_sim(0.0, "int8", q, orthogonal) == pytest.approx(0.0, abs=0.01)


def test_row_sim_abstains_when_an_int8_blob_is_unavailable():
    """No blob means no cosine: the arm abstains instead of guessing.

    The legacy mapping would have reported ~0.93 for this distance, which is the
    calibration problem from #982 — a fabricated, saturated score.
    """
    distance = 107.0
    assert _legacy_sim(distance) > 0.8  # the saturated band the old mapping produced
    assert _wm_vec_row_sim(distance, "int8", None, None) is None
    assert _wm_vec_row_sim(distance, "int8", b"", None) is None
    assert _wm_vec_row_sim(distance, "int8", None, b"\x01") is None
    assert _wm_vec_row_sim(distance, "int8", b"\x01", b"\x01\x02") is None


def test_row_sim_has_no_legacy_switch(monkeypatch):
    """The old MNEMOSYNE_WM_VEC_BLOB_SCORING escape hatch is gone."""
    q = _int8_blob([127, 0, 0, 0])
    orthogonal = _int8_blob([0, 127, 0, 0])
    monkeypatch.setenv("MNEMOSYNE_WM_VEC_BLOB_SCORING", "0")

    assert _wm_vec_row_sim(107.0, "int8", q, orthogonal) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("vec_type", [None, "", "float32", "bit", "unknown"])
def test_row_sim_non_int8_arms_keep_legacy_mapping(vec_type):
    assert _wm_vec_row_sim(107.0, vec_type, b"\x7f", b"\x00") == pytest.approx(_legacy_sim(107.0))


def _seed_working_rows(beam, rows, session_id):
    """Insert working-memory rows with their embeddings, then build vec_working."""
    now = datetime.now().isoformat()
    for memory_id, content, embedding in rows:
        beam.conn.execute(
            """
            INSERT INTO working_memory
                (id, content, source, timestamp, session_id, scope, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (memory_id, content, "test", now, session_id, "session", 0.5),
        )
        beam.conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
            (memory_id, beam_module._embeddings.serialize(embedding), "test"),
        )
    beam.conn.commit()
    beam_module._backfill_vec_working_from_memory_embeddings(beam.conn)


def _enable_embeddings(monkeypatch, query_vec):
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings, "embed_query", lambda _query: query_vec
    )


def _blob_fixture(beam, session_id):
    """Three working rows: identical, ~0.8 cosine, orthogonal to the query."""
    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    same = _query_vector([1.0] + [0.0] * (dim - 1))
    angled = _query_vector([0.8, 0.6] + [0.0] * (dim - 2))
    orthogonal = _query_vector([0.0, 1.0] + [0.0] * (dim - 2))
    _seed_working_rows(
        beam,
        [
            ("wm-same", "identical row", same),
            ("wm-angled", "near row", angled),
            ("wm-orthogonal", "unrelated row", orthogonal),
        ],
        session_id,
    )
    return query


def _cosine_ladder(beam, session_id, targets):
    """Working rows whose true cosine with the query spans ``targets``."""
    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    rows = []
    for target in targets:
        theta = math.acos(target)
        vec = _query_vector([math.cos(theta), math.sin(theta)] + [0.0] * (dim - 2))
        rows.append((f"wm-{target:.2f}", f"ladder row {target:.2f}", vec))
    _seed_working_rows(beam, rows, session_id)
    return query


def _legacy_sims_from_knn(beam, query, k):
    """The mapping this fix removed, computed from the same rows via KNN.

    sqlite-vec only reports ``distance`` inside a MATCH query, so this repeats
    exactly the query the search path issues.
    """
    query_json = beam_module._embeddings.serialize(query)
    return {
        str(row[0]): _legacy_sim(float(row[1]))
        for row in beam.conn.execute(
            "SELECT wm.id, vw.distance FROM vec_working vw "
            "JOIN working_memory wm ON wm.rowid = vw.rowid "
            "WHERE vw.embedding MATCH vec_quantize_int8(?, 'unit') AND k = ? "
            "ORDER BY vw.distance",
            (query_json, k),
        )
    }


@requires_vec
def test_wm_vec_search_similarity_ordering_uses_exact_cosine(temp_db):
    """Regression for #982: `sim` spans the true cosine range, not a squeezed band.

    Same shape as the controlled reproduction in #982: the true cosine separation
    survives (blob scoring) while the removed mapping compresses it into a much
    narrower band on these same rows. The compression factor is asserted
    relatively, because the raw int8 distance scale depends on the sqlite-vec
    build rather than on this code.
    """
    targets = [0.95, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.0]
    beam = BeamMemory(session_id="wm-vec-blob", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    query = _cosine_ladder(beam, "wm-vec-blob", targets)

    results = _wm_vec_search(beam.conn, query, k=len(targets))
    sims = {r["id"]: r["sim"] for r in results}

    assert set(sims) == {f"wm-{t:.2f}" for t in targets}
    for target in targets:
        assert sims[f"wm-{target:.2f}"] == pytest.approx(target, abs=0.05), sims
    blob_spread = max(sims.values()) - min(sims.values())
    assert blob_spread > 0.9

    legacy = _legacy_sims_from_knn(beam, query, len(targets))
    assert set(legacy) == set(sims)
    legacy_spread = max(legacy.values()) - min(legacy.values())
    # The absolute band depends on the sqlite-vec build (the raw int8 distance
    # scale differs between versions), so the squeeze is asserted relatively:
    # the mapping loses most of the spread the true cosines have.
    assert blob_spread > legacy_spread * 2, f"no squeeze: {legacy}"
    # An orthogonal row kept a high similarity under the mapping, i.e. a
    # fabricated dense voice instead of 0.0.
    assert legacy["wm-0.00"] > 0.7, f"orthogonal row kept {legacy['wm-0.00']}"


@requires_vec
def test_unscorable_int8_candidates_fall_back_to_the_exact_scan(temp_db, monkeypatch):
    """An unreadable int8 blob must not degrade into a distance guess.

    With scoring unavailable the arm returns nothing, so `_wm_vec_search` serves
    the candidate set from the compatibility scan, whose similarities are exact
    cosines rather than a fabricated band.
    """
    beam = BeamMemory(session_id="wm-vec-fallback", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    query = _blob_fixture(beam, "wm-vec-fallback")

    monkeypatch.setattr(beam_module, "_wm_vec_row_sim", lambda *a, **kw: None)
    results = _wm_vec_search(beam.conn, query, k=3)
    sims = {r["id"]: r["sim"] for r in results}

    assert set(sims) == {"wm-same", "wm-angled", "wm-orthogonal"}
    assert sims["wm-same"] == pytest.approx(1.0, abs=0.05)
    assert sims["wm-angled"] == pytest.approx(0.8, abs=0.1)
    assert sims["wm-orthogonal"] == pytest.approx(0.0, abs=0.1)


@requires_vec
def test_mixed_scorable_candidates_keep_every_row_and_score(temp_db, monkeypatch):
    """A partially scorable int8 result set must not lose rows or scores.

    `_wm_vec_search_sqlite` abandons the whole vector arm as soon as one
    candidate has no usable blob, handing the complete candidate set to the
    compatibility scan. This pins that invariant for a *mixed* set: two
    candidates are scored from their stored bytes, exactly one abstains, and all
    three rows still come back with their exact cosines (review: dplush on #987).
    """
    beam = BeamMemory(session_id="wm-vec-mixed", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    query = _blob_fixture(beam, "wm-vec-mixed")
    angled_blob = bytes(beam.conn.execute(
        "SELECT vw.embedding FROM vec_working vw "
        "JOIN working_memory wm ON wm.rowid = vw.rowid WHERE wm.id = ?",
        ("wm-angled",),
    ).fetchone()[0])

    original = beam_module._wm_vec_row_sim
    scored = []
    abstained = []

    def mixed(distance, vec_type, query_blob, row_blob):
        if row_blob == angled_blob:
            abstained.append(row_blob)
            return None
        sim = original(distance, vec_type, query_blob, row_blob)
        scored.append(sim)
        return sim

    monkeypatch.setattr(beam_module, "_wm_vec_row_sim", mixed)
    results = _wm_vec_search(beam.conn, query, k=3)
    sims = {r["id"]: r["sim"] for r in results}

    assert len(abstained) == 1, abstained
    # The arm stops at the first abstention because it hands the whole set to
    # the compatibility scan; anything it scored before that is a real cosine.
    assert scored and all(s is not None for s in scored), scored
    assert set(sims) == {"wm-same", "wm-angled", "wm-orthogonal"}
    assert sims["wm-same"] == pytest.approx(1.0, abs=0.05)
    assert sims["wm-angled"] == pytest.approx(0.8, abs=0.1)
    assert sims["wm-orthogonal"] == pytest.approx(0.0, abs=0.1)


@requires_vec
def test_recall_ranks_the_gold_row_above_a_distractor(temp_db, monkeypatch):
    """`BeamMemory.recall()`: the dense voice must separate gold from distractor.

    Both rows share the query tokens (so both reach the pool through FTS) and
    differ only in direction: the gold embedding matches the query, the
    distractor is orthogonal. The legacy mapping gave both a ~0.93
    `dense_score`, i.e. the same near-constant term, so the dense voice could
    not lift the gold row at all.
    """
    # The polyphonic engine has its own path and returns before the linear
    # working-memory arm, so pin the default engine for this regression.
    monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
    beam = BeamMemory(session_id="wm-recall-gold", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")

    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    gold = _query_vector([1.0, 0.0] + [0.0] * (dim - 2))
    distractor = _query_vector([0.0, 1.0] + [0.0] * (dim - 2))
    _seed_working_rows(
        beam,
        [
            ("wm-gold", "kuma threshold fact", gold),
            ("wm-distractor", "kuma threshold noise", distractor),
        ],
        "wm-recall-gold",
    )
    _enable_embeddings(monkeypatch, query)

    results = beam.recall("kuma threshold", top_k=10)
    dense = {r["id"]: r["dense_score"] for r in results}
    ids = [r["id"] for r in results]

    assert "wm-gold" in dense and "wm-distractor" in dense, f"both rows must reach recall: {ids}"
    assert dense["wm-gold"] >= 0.9, f"gold row lost its dense voice: {dense}"
    assert dense["wm-distractor"] <= 0.1, (
        f"distractor kept a saturated dense voice ({dense['wm-distractor']}): "
        "the distance-derived mapping is back"
    )
    assert ids.index("wm-gold") < ids.index("wm-distractor")
