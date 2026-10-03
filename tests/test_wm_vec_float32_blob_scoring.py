"""Working-memory float32 vector arm: exact blob scoring (the arm left by #987).

#1069: #982/#987 made the int8 arm of ``_wm_vec_search_sqlite()`` score its
candidates from their stored bytes and abstain when it cannot. The float32 arm
was not migrated and still fell through to the legacy
``1 - distance / (2 * EMBEDDING_DIM)`` mapping, which assumes unit norms and
divides by the dimension instead of 2. On a ``float32[1024]`` store every
candidate collapsed into a ~0.9993-0.9996 band: the ordering survived, the
amplitude did not, and the working-memory dense blend received a near-constant
term, so the documented recall-first lexical admission opt-in (#937 / #886)
could not be used at all.

These tests pin the fixed behaviour:

* float32 candidates are scored from their stored bytes with the shared
  ``_vec_float32_blob_cosine`` helper, so ``sim`` spans the true cosine range,
* legacy rows written before normalization was enforced still score exactly
  (the distance mapping cannot: it assumes unit norms),
* a candidate whose blob is unavailable abstains instead of reporting a guess,
* the ``bit`` arm keeps its mapping and the int8 arm is untouched,
* the ``_wm_vec_search()`` wrapper and ``BeamMemory.recall()`` see the exact
  scores instead of the saturated band.

The float32 store is built from its own DDL instead of switching
``MNEMOSYNE_VEC_TYPE``: the module-level ``VEC_TYPE`` is resolved at import time,
so mutating it would leak into every other test in this session, while both the
insert path and the search path read the *table's* declared type.
"""

from __future__ import annotations

import json
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


def vec_supports_float32():
    """sqlite-vec present AND able to create a float32 vec0 table."""
    global VEC_AVAILABLE
    if VEC_AVAILABLE is None:
        try:
            c = sqlite3.connect(":memory:")
            VEC_AVAILABLE = _load_vec(c) and c.execute(
                "CREATE VIRTUAL TABLE probe USING vec0(embedding float32[4])"
            ) is not None
            c.close()
        except Exception:
            VEC_AVAILABLE = False
    return VEC_AVAILABLE


requires_vec = pytest.mark.skipif(
    not vec_supports_float32(), reason="sqlite-vec float32 support not available"
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _query_vector(values):
    np = pytest.importorskip("numpy")
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _float32_blob(values):
    np = pytest.importorskip("numpy")
    return np.array(values, dtype=np.float32).tobytes()


def _l2_distance(query, row) -> float:
    """The distance a float32 vec0 table reports for these two vectors."""
    np = pytest.importorskip("numpy")
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(row, dtype=np.float64)
    return float(np.linalg.norm(q - r))


def _legacy_sim(distance: float) -> float:
    """The distance mapping this fix removed, kept for contrast."""
    return max(0.0, min(1.0, 1.0 - distance / (2.0 * beam_module.EMBEDDING_DIM)))


# ---------------------------------------------------------------- unit level


def test_row_sim_scores_float32_candidates_from_blobs():
    """Blob cosine replaces the saturated legacy mapping when a blob is present."""
    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    same = _float32_blob([1.0] + [0.0] * (dim - 1))
    angled = _float32_blob([0.8, 0.6] + [0.0] * (dim - 2))
    orthogonal = _float32_blob([0.0, 1.0] + [0.0] * (dim - 2))

    d_same = _l2_distance(query, [1.0] + [0.0] * (dim - 1))
    d_angled = _l2_distance(query, [0.8, 0.6] + [0.0] * (dim - 2))
    d_orthogonal = _l2_distance(query, [0.0, 1.0] + [0.0] * (dim - 2))

    assert _wm_vec_row_sim(d_same, "float32", query, same) == pytest.approx(1.0, abs=0.01)
    assert _wm_vec_row_sim(d_angled, "float32", query, angled) == pytest.approx(0.80, abs=0.02)
    assert _wm_vec_row_sim(d_orthogonal, "float32", query, orthogonal) == pytest.approx(0.0, abs=0.01)

    # The mapping this replaces reported ~0.999 for all three distances, i.e. an
    # unrelated row kept a near-perfect dense voice (the #1069 symptom).
    for distance in (d_same, d_angled, d_orthogonal):
        assert _legacy_sim(distance) > 0.99, distance


def test_row_sim_float32_scores_legacy_rows_without_unit_norms():
    """A pre-normalization row is still scored exactly; the mapping cannot be.

    The stored blob is the raw float data, so the cosine is recoverable from the
    row's own norm. The distance-only conversion assumes unit norms, so such a
    row is mis-scored no matter how its distance is transformed.
    """
    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    legacy_row = [4.0, 3.0] + [0.0] * (dim - 2)  # norm 5, not 1
    distance = _l2_distance(query, legacy_row)

    sim = _wm_vec_row_sim(distance, "float32", query, _float32_blob(legacy_row))
    assert sim == pytest.approx(0.80, abs=0.01)  # true cosine of (1,0) vs (4,3)
    assert _legacy_sim(distance) > 0.99  # what a distance-only arm would report


def test_row_sim_float32_abstains_when_a_blob_is_unavailable():
    """No blob means no cosine: abstain instead of guessing from the distance.

    ``_wm_vec_search_sqlite()`` reacts to ``None`` by dropping the candidate and
    letting the exact compatibility scan serve the set, which is what preserves
    membership without fabricating a score.
    """
    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    distance = 1.116056

    assert _legacy_sim(distance) > 0.99  # the fabricated score this replaces
    assert _wm_vec_row_sim(distance, "float32", query, None) is None


def test_row_sim_float32_abstains_on_a_blob_that_cannot_be_a_vector():
    """A blob of the wrong length must abstain, not score 0.0.

    ``_vec_float32_blob_cosine`` returns 0.0 for a blob it cannot decode into
    the query's shape, which is indistinguishable from a genuine orthogonal
    row - so without a length check the arm would silently score an
    unreadable candidate as unrelated instead of letting the exact
    compatibility scan serve it (the int8 arm length-checks for the same
    reason).
    """
    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    distance = 1.116056

    # Not a whole number of float32 values.
    assert _wm_vec_row_sim(distance, "float32", query, b"\x00\x01\x02") is None
    # A valid float32 blob, but of the wrong dimension.
    assert _wm_vec_row_sim(distance, "float32", query, _float32_blob([1.0, 0.0])) is None
    # No query-side reference to compare against.
    assert (
        _wm_vec_row_sim(distance, "float32", None, _float32_blob([1.0] + [0.0] * (dim - 1)))
        is None
    )
    # Control: a correctly shaped pair still scores.
    ok = _float32_blob([1.0] + [0.0] * (dim - 1))
    assert _wm_vec_row_sim(0.0, "float32", query, ok) == pytest.approx(1.0, abs=0.01)


def test_other_vec_arms_keep_their_mapping():
    """The bit arm is unchanged and the int8 arm is untouched."""
    dim = beam_module.EMBEDDING_DIM
    query = _query_vector([1.0] + [0.0] * (dim - 1))
    blob = _float32_blob([1.0] + [0.0] * (dim - 1))

    assert _wm_vec_row_sim(1.116056, "bit", query, blob) == pytest.approx(_legacy_sim(1.116056))
    assert _wm_vec_row_sim(0.0, None, query, blob) == pytest.approx(_legacy_sim(0.0))

    np = pytest.importorskip("numpy")
    int8_blob = np.array([127, 0] + [0] * (dim - 2), dtype="int8").tobytes()
    assert _wm_vec_row_sim(0.0, "int8", int8_blob, int8_blob) == pytest.approx(1.0, abs=0.01)
    assert _wm_vec_row_sim(0.0, "int8", None, None) is None


# --------------------------------------------------------- integration level


def _force_float32_vec_working(beam, dim=None):
    """Recreate ``vec_working`` as float32[N] (the arm under test).

    BeamMemory creates the table with the module-level ``VEC_TYPE``, so the
    regression rebuilds it explicitly. Both the insert path and the search path
    read the declared type, so no environment mutation is needed.
    """
    dim = dim or beam_module.EMBEDDING_DIM
    conn = beam.conn
    conn.execute("DROP TABLE IF EXISTS vec_working")
    conn.execute(f"CREATE VIRTUAL TABLE vec_working USING vec0(embedding float32[{dim}])")
    conn.commit()
    beam_module._mark_vec_store_norm_bit(conn)


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
    """The mapping this fix removed, computed from the same float32 rows."""
    query_json = beam_module._embeddings.serialize(query)
    return {
        str(row[0]): _legacy_sim(float(row[1]))
        for row in beam.conn.execute(
            "SELECT wm.id, vw.distance FROM vec_working vw "
            "JOIN working_memory wm ON wm.rowid = vw.rowid "
            "WHERE vw.embedding MATCH ? AND k = ? "
            "ORDER BY vw.distance",
            (query_json, k),
        )
    }


@requires_vec
def test_wm_vec_search_reports_exact_float32_scores(temp_db):
    """Regression for #1069: `sim` spans the true cosine range on float32 stores."""
    targets = [0.95, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.0]
    beam = BeamMemory(session_id="wm-f32", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    _force_float32_vec_working(beam)
    query = _cosine_ladder(beam, "wm-f32", targets)

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
    # The absolute band depends on the sqlite-vec build, so the squeeze is
    # asserted relatively: the mapping loses most of the spread.
    assert blob_spread > legacy_spread * 2, f"no squeeze: {legacy}"
    # An orthogonal row kept a near-perfect similarity under the mapping.
    assert legacy["wm-0.00"] > 0.99, f"orthogonal row kept {legacy['wm-0.00']}"


@requires_vec
def test_unscorable_float32_candidates_fall_back_to_the_exact_scan(temp_db, monkeypatch):
    """An unavailable blob must not degrade into a distance guess.

    With scoring unavailable the arm returns nothing, so ``_wm_vec_search()``
    serves the candidate set from the compatibility scan, whose similarities are
    exact cosines rather than a fabricated band.
    """
    dim = beam_module.EMBEDDING_DIM
    beam = BeamMemory(session_id="wm-f32-fallback", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    _force_float32_vec_working(beam)

    query = _query_vector([1.0] + [0.0] * (dim - 1))
    _seed_working_rows(
        beam,
        [
            ("wm-same", "identical row", _query_vector([1.0] + [0.0] * (dim - 1))),
            ("wm-angled", "near row", _query_vector([0.8, 0.6] + [0.0] * (dim - 2))),
            ("wm-orthogonal", "unrelated row", _query_vector([0.0, 1.0] + [0.0] * (dim - 2))),
        ],
        "wm-f32-fallback",
    )

    def abstain(*args, **kwargs):
        return None

    monkeypatch.setattr(beam_module, "_wm_vec_row_sim", abstain)
    results = _wm_vec_search(beam.conn, query, k=3)
    sims = {r["id"]: r["sim"] for r in results}

    assert set(sims) == {"wm-same", "wm-angled", "wm-orthogonal"}
    assert sims["wm-same"] == pytest.approx(1.0, abs=0.05)
    assert sims["wm-angled"] == pytest.approx(0.8, abs=0.1)
    assert sims["wm-orthogonal"] == pytest.approx(0.0, abs=0.1)


@requires_vec
def test_recall_separates_gold_from_distractor_on_a_float32_store(temp_db, monkeypatch):
    """``BeamMemory.recall()``: the dense voice must separate gold from distractor.

    Both rows share the query tokens (so both reach the pool through FTS) and
    differ only in direction. The legacy mapping gave both the same saturated
    ``dense_score``, so the dense blend could not lift the gold row.
    """
    monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
    dim = beam_module.EMBEDDING_DIM
    beam = BeamMemory(session_id="wm-f32-recall", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    _force_float32_vec_working(beam)

    query = _query_vector([1.0] + [0.0] * (dim - 1))
    gold = _query_vector([1.0, 0.0] + [0.0] * (dim - 2))
    distractor = _query_vector([0.0, 1.0] + [0.0] * (dim - 2))
    _seed_working_rows(
        beam,
        [
            ("wm-gold", "kuma threshold fact", gold),
            ("wm-distractor", "kuma threshold noise", distractor),
        ],
        "wm-f32-recall",
    )
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda _query: query)

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


# ------------------------------------------------- top-k membership (#1080 review)


# More rows than the candidate window holds for k=1 (max(k*25, 500)), so the
# window cannot be read whole.
_WINDOW_FILLER_ROWS = 520


def _clear_vec_store_norm_bit(conn):
    """Put the store back into the pre-normalization ("non-pure") regime.

    ``_classify_vec_store_regime()`` routes on that format marker, and the vec
    table is created marked, so a legacy store is simulated by clearing it.
    """
    uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
    conn.execute(f"PRAGMA user_version = {uv & ~beam_module._VEC_NORM_BIT & 0xFFFFFFFF}")
    conn.commit()


def _write_raw_vec_blob(beam, memory_id, values):
    """Write a row's vec_working blob as-is, bypassing the normalizing writer.

    ``_vec_table_insert()`` normalizes before inserting, so this is the only way
    a pre-normalization row can exist in the vec table - exactly what the older
    code left behind.
    """
    rowid = beam.conn.execute(
        "SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0]
    blob = json.dumps([float(v) for v in values])
    beam.conn.execute("DELETE FROM vec_working WHERE rowid = ?", (rowid,))
    beam.conn.execute("INSERT INTO vec_working(rowid, embedding) VALUES (?, ?)", (rowid, blob))
    beam.conn.commit()


def _knn_order(beam, query, k):
    """The order the distance window alone would return."""
    return [
        r[0]
        for r in beam.conn.execute(
            "SELECT wm.id FROM vec_working vw JOIN working_memory wm ON wm.rowid = vw.rowid "
            "WHERE vw.embedding MATCH ? AND k = ? ORDER BY vw.distance",
            (beam_module._embeddings.serialize(query), k),
        )
    ]


def _raw_row(values):
    """A stored vector exactly as given: no normalization.

    The legacy rows under test are non-unit on purpose, so they must not go
    through ``_query_vector()``.
    """
    np = pytest.importorskip("numpy")
    return np.array(values, dtype=np.float32)


def _filler_rows(dim, *, unit=True):
    """Rows orthogonal to the query, each nearer in L2 than a long row.

    The window is bounded by the number of rows, not by the number of axes, so
    the fillers reuse the query axis plus one orthogonal axis instead of
    spreading over one dimension each. That keeps the scenario independent of
    EMBEDDING_DIM (the CI process runs a 384-dimension model).

    :param unit: emit unit-length fillers, as a normalized store requires.
    """
    rows = []
    for i in range(_WINDOW_FILLER_ROWS):
        vec = [0.0] * dim
        # Non-unit fillers are nearer in L2 the smaller their scale, and every
        # one of them stays nearer than a norm-5 row.
        vec[1] = 1.0 if unit else 0.5 + (i * 0.001)
        rows.append((f"wm-filler-{i:03d}", f"filler {i}", _raw_row(vec)))
    return rows


def _needs_window_scenario(dim) -> bool:
    # Only the query axis and one orthogonal axis are used.
    return dim < 2


@requires_vec
def test_top_k_membership_follows_cosine_not_l2_distance(temp_db):
    """#1080 review: the returned top-k must be the top-k by the reported score.

    ``_wm_vec_search_sqlite()`` reads candidates in L2-distance order, which only
    agrees with the cosine it reports while the rows are unit-normalized. On a
    store with a legacy (non-unit) row, truncating in distance order returned the
    orthogonal unit row (sim 0.0) and dropped the collinear norm-5 row (sim 1.0),
    even though the exact compatibility scan ranked them the other way round.
    """
    dim = beam_module.EMBEDDING_DIM
    beam = BeamMemory(session_id="wm-f32-membership", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    _force_float32_vec_working(beam)
    _clear_vec_store_norm_bit(beam.conn)

    query = _query_vector([1.0] + [0.0] * (dim - 1))
    orthogonal = _query_vector([0.0, 1.0] + [0.0] * (dim - 2))
    legacy = [5.0] + [0.0] * (dim - 1)  # norm 5, cosine 1.0, L2 distance 4.0
    _seed_working_rows(
        beam,
        [
            ("wm-orthogonal", "unrelated unit row", orthogonal),
            ("wm-collinear", "legacy norm-5 row", _raw_row(legacy)),
        ],
        "wm-f32-membership",
    )
    # The write path normalizes, so the legacy row is written as-is, the way the
    # pre-normalization code left it in the vec table.
    _write_raw_vec_blob(beam, "wm-collinear", legacy)

    # Both rows fit the window (the store is smaller than max(k*25, 500)), so the
    # only question is the order they are truncated in.
    results = _wm_vec_search(beam.conn, query, k=1)
    assert [r["id"] for r in results] == ["wm-collinear"], results
    assert results[0]["sim"] == pytest.approx(1.0, abs=0.01)

    # The order the window alone would have truncated: the orthogonal unit row is
    # nearer in L2 (sqrt(2) vs 4.0) even though its cosine is the worse one.
    assert _knn_order(beam, query, 2) == ["wm-orthogonal", "wm-collinear"]


@requires_vec
def test_bounded_l2_window_abstains_on_a_non_pure_store(temp_db):
    """#1080 review: a bounded distance window must not silently drop the best match.

    The filler rows are orthogonal to the query and far nearer in L2 than the
    collinear row, so a 500-row distance window cannot contain the best cosine
    match at all. The store is not in the normalized format, so the arm abstains
    and the caller's exact compatibility scan ranks the candidate set.
    """
    dim = beam_module.EMBEDDING_DIM
    if _needs_window_scenario(dim):
        pytest.skip("embedding dimension too small for this scenario")
    beam = BeamMemory(session_id="wm-f32-window", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    _force_float32_vec_working(beam)
    _clear_vec_store_norm_bit(beam.conn)

    query = _query_vector([1.0] + [0.0] * (dim - 1))
    legacy = [5.0] + [0.0] * (dim - 1)
    rows = [("wm-collinear", "legacy norm-5 row", _raw_row(legacy))]
    rows.extend(_filler_rows(dim, unit=False))
    _seed_working_rows(beam, rows, "wm-f32-window")
    _write_raw_vec_blob(beam, "wm-collinear", legacy)

    total = beam.conn.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0]
    assert total > 500, total  # the window is bounded below the row count
    # The scenario itself: the best cosine match is outside the 500-row window.
    window = _knn_order(beam, query, 500)
    assert len(window) == 500 and "wm-collinear" not in window, window[:5]

    assert beam_module._wm_vec_search_sqlite(beam.conn, query, k=1, where_sql="1=1") == []
    results = _wm_vec_search(beam.conn, query, k=1)
    assert [r["id"] for r in results] == ["wm-collinear"], results
    assert results[0]["sim"] == pytest.approx(1.0, abs=0.01)


@requires_vec
def test_normalized_store_keeps_the_window_fast_path(temp_db):
    """The format marker still routes a normalized store through the window.

    Only stores that may hold pre-normalization rows abstain; on a "pure" store
    every row is unit-normalized, distance order is score order, and the bounded
    window remains exact.
    """
    dim = beam_module.EMBEDDING_DIM
    if _needs_window_scenario(dim):
        pytest.skip("embedding dimension too small for this scenario")
    beam = BeamMemory(session_id="wm-f32-pure", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    _force_float32_vec_working(beam)  # keeps the norm marker written at creation
    assert beam_module._classify_vec_store_regime(beam.conn, "vec_working") == "pure"

    query = _query_vector([1.0] + [0.0] * (dim - 1))
    rows = [("wm-best", "closest row", _query_vector([0.8, 0.6] + [0.0] * (dim - 2)))]
    rows.extend(_filler_rows(dim))
    _seed_working_rows(beam, rows, "wm-f32-pure")

    total = beam.conn.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0]
    assert total > 500, total  # the window really is bounded
    results = _wm_vec_search(beam.conn, query, k=3)
    assert len(results) == 3, results
    assert results[0]["id"] == "wm-best"
