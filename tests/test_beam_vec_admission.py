"""Absolute-cosine admission for episodic vector candidates.

int8 rows are scored from their stored quantized bytes against the query's
quantized bytes (exact cosine for non-saturated rows; sign-bit fallback for
saturated legacy rows). float32/bit arms convert from distance in their own
domain. All int8 tests run against REAL vec0 tables with expectations
computed independently from the seeded float vectors — never from module
internals.
"""
import json
import math
import sqlite3

import numpy as np
import pytest

import mnemosyne.core.beam as beam_module
from mnemosyne.core.beam import (
    EM_VEC_ADMIT,
    BeamMemory,
    _vec_distance_sim,
    _vec_int8_blob_cosine,
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


def make_table(conn, dim):
    conn.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding int8[{dim}])")


def quantize_insert(conn, rid, vec):
    conn.execute(
        "INSERT INTO v(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
        (rid, json.dumps([float(x) for x in vec])),
    )


def sql_query_blob(conn, vec):
    return conn.execute(
        "SELECT vec_quantize_int8(?, 'unit')",
        (json.dumps([float(x) for x in vec]),),
    ).fetchone()[0]


def sql_knn(conn, query_vec, k):
    return conn.execute(
        "SELECT rowid, distance, embedding FROM v "
        "WHERE embedding MATCH vec_quantize_int8(?, 'unit') AND k=? ORDER BY distance",
        (json.dumps([float(x) for x in query_vec]), k),
    ).fetchall()


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def exact_cos_pair(rng, dim, target):
    """Two unit vectors with float cosine == target (Gram-Schmidt)."""
    base = rng.standard_normal(dim)
    noise = rng.standard_normal(dim)
    noise -= (np.dot(noise, base) / np.dot(base, base)) * base
    noise *= np.linalg.norm(base) / np.linalg.norm(noise)
    a = unit(base)
    b = unit(base * target + noise * math.sqrt(1 - target * target))
    return a, b



@pytest.fixture
def temp_db(tmp_path):
    return tmp_path / "beam_test.db"


class TestPerArmConversion:
    """Round-1 F1: each vector arm must convert within its own domain."""

    def test_float32_arm_orthogonal_scores_zero(self):
        # float32 vec0: raw L2 over unit vectors; orthogonal -> d=sqrt(2)
        assert _vec_distance_sim(2.0 ** 0.5, "float32") < 0.01
        assert _vec_distance_sim(0.0, "float32") == pytest.approx(1.0)
        assert _vec_distance_sim(3.0, "float32") == pytest.approx(0.0)  # d>2 -> 0

    def test_bit_arm_uses_arc_relation(self):
        # bit vec0: sign-bit disagreement d/D approximates theta/pi, so
        # cos = cos(pi * d / D) — the ARC, not the linear chord 1-2d/D
        # (the chord wrongly rejected true-cos ~0.90 vectors).
        import math as _math
        D = beam_module.EMBEDDING_DIM
        assert _vec_distance_sim(0.0, "bit") == pytest.approx(1.0)
        # true cosine 0.90 == frac = acos(0.9)/pi ≈ 0.1436 -> d ≈ 0.1436*D
        d90 = ( _math.acos(0.90) / _math.pi ) * D
        assert _vec_distance_sim(d90, "bit") == pytest.approx(0.90, abs=0.01)
        # orthogonal (half bits differ) -> 0
        assert _vec_distance_sim(D / 2.0, "bit") == pytest.approx(0.0)
        # opposite (all bits differ) -> -1 clamped to 0
        assert _vec_distance_sim(D, "bit") == pytest.approx(0.0)
        # mid-range arc beats the chord: frac=0.25 -> cos(pi/4)=0.707, chord 0.5
        assert _vec_distance_sim(0.25 * D, "bit") == pytest.approx(_math.cos(_math.pi / 4), abs=1e-6)

    def test_in_memory_fallback_uses_cosine_distance(self):
        # fallback returns distance = 1 - cosine in [0, 2]
        assert _vec_distance_sim(0.0, None) == pytest.approx(1.0)      # identical
        assert _vec_distance_sim(0.2, None) == pytest.approx(0.8)      # cos 0.8
        assert _vec_distance_sim(1.0, None) == pytest.approx(0.0)      # orthogonal
        assert _vec_distance_sim(1.5, None) == pytest.approx(0.0)      # negative cos

    def test_int8_arm_abstains_without_blobs(self):
        # An int8 distance alone cannot yield an absolute cosine (the
        # quantization scale is dimension/distribution-dependent) — the
        # distance-only arm abstains; blob callers use _vec_int8_blob_cosine.
        assert _vec_distance_sim(60.0, "int8") == pytest.approx(0.0)

    def test_unknown_arm_abstains(self):
        assert _vec_distance_sim(0.0, "quantum") == pytest.approx(0.0)

    def test_non_finite_across_arms(self):
        for arm in ("int8", "float32", "bit", None):
            assert _vec_distance_sim(float("nan"), arm) == pytest.approx(0.0)
            assert _vec_distance_sim(float("inf"), arm) == pytest.approx(0.0)


# ------------------------------------------------- integration through recall
class TestRecallAdmission:
    """e2e through recall(): stubbed search returns crafted int8 blobs so
    the wiring (arm routing, gate application, dense_score) is exercised
    with known quantized vectors; expectations computed from those same
    byte arrays."""

    def _seed_em(self, beam, rows):
        for rid, content in rows:
            beam.conn.execute(
                "INSERT INTO episodic_memory "
                "(id, content, source, timestamp, session_id, importance, scope, memory_type) "
                "VALUES (?, ?, 'sleep_consolidation', datetime('now'), 'sess', 0.6, 'global', 'fact')",
                (rid, content),
            )
        beam.conn.commit()

    def test_admission_is_stable_across_top_k(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="admit", db_path=temp_db)
        self._seed_em(beam, [("near-1", "alpha near target"), ("far-1", "bravo far target")])
        rid_map = {r[1]: r[0] for r in beam.conn.execute(
            "SELECT rowid, id FROM episodic_memory WHERE id IN ('near-1','far-1')")}
        near_rid = rid_map["near-1"]
        far_rid = rid_map["far-1"]

        # crafted 8-dim int8 vectors: near is collinear with q, far is
        # sign-flipped (sign-bit cosine 0, dot cosine < 0)
        q_arr = np.array([100, -90, 80, -70, 60, -50, 40, -30], dtype=np.int8)
        near_arr = q_arr  # identical -> cosine 1.0
        far_arr = -q_arr  # opposite -> cosine 0 after clamp
        q_blob = q_arr.tobytes()

        def fake_search_with_blobs(conn, emb, k=20):
            rows = [{"rowid": near_rid, "distance": 1.0, "blob": near_arr.tobytes()}]
            if k >= 20:
                rows.append({"rowid": far_rid, "distance": 200.0, "blob": far_arr.tobytes()})
            return rows, q_blob

        monkeypatch.setattr(beam_module, "_vec_search_with_blobs", fake_search_with_blobs)
        monkeypatch.setattr(beam_module, "_vec_available", lambda conn: True)
        monkeypatch.setattr(
            beam_module, "_vec_table_type_strict", lambda conn: "int8"
        )  # force the arm: the e2e must not depend on the host's vec tables
        monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
        monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda text: np.zeros(1024, dtype=np.float32))
        monkeypatch.setattr(beam_module._embeddings, "embed", lambda texts: [np.zeros(1024, dtype=np.float32)])

        # independent expectation from the crafted byte vectors
        exp_near = float(np.dot(q_arr.astype(float), near_arr.astype(float))) / (
            np.linalg.norm(q_arr.astype(float)) * np.linalg.norm(near_arr.astype(float))
        )

        for top_k in (1, 10, 60):
            res = beam.recall("zzzqqqwwxx", top_k=top_k)
            ids = {r["id"] for r in res}
            assert "near-1" in ids, f"near candidate missing at top_k={top_k}"
            assert "far-1" not in ids, f"far candidate admitted at top_k={top_k}"
            near = next(r for r in res if r["id"] == "near-1")
            # non-vacuous at every k: the near row's score must be the same
            # absolute value regardless of whether far-1 was in its batch
            assert near["dense_score"] == pytest.approx(exp_near, abs=1e-4)


class TestDetectionFailureAbstains:
    """A transient vec-table DDL lookup failure must ABSTAIN (score 0), not
    fall into the in-memory (1 - cosine) conversion — a native-table distance
    like 0.1 would otherwise score 0.9 and pass the admission gate."""

    def test_detection_failure_zeroes_scores(self, temp_db, monkeypatch):
        import numpy as np
        beam = BeamMemory(session_id="detfail", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO episodic_memory "
            "(id, content, source, timestamp, session_id, importance, scope, memory_type) "
            "VALUES ('r1', 'alpha near target', 'sleep_consolidation', datetime('now'), 'sess', 0.6, 'global', 'fact')",
        )
        beam.conn.commit()
        rid = beam.conn.execute("SELECT rowid FROM episodic_memory WHERE id='r1'").fetchone()[0]

        def failing_strict(conn):
            raise RuntimeError("transient DDL lookup failure")

        monkeypatch.setattr(beam_module, "_vec_search", lambda conn, emb, k=20: [
            {"rowid": rid, "distance": 0.1},  # native-float32-looking distance
        ])
        monkeypatch.setattr(beam_module, "_vec_available", lambda conn: True)
        monkeypatch.setattr(beam_module, "_vec_table_type_strict", failing_strict)
        monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
        monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda text: np.zeros(1024, dtype=np.float32))
        monkeypatch.setattr(beam_module._embeddings, "embed", lambda texts: [np.zeros(1024, dtype=np.float32)])

        res = beam.recall("zzzqqqwwxx", top_k=10)
        items = res if isinstance(res, list) else getattr(res, "results", [])
        assert "r1" not in {r["id"] for r in items}, (
            "detection failure must not admit the row via a misread conversion"
        )


class TestEnvValidation:
    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EM_VEC_ADMIT", "garbage")
        assert beam_module._env_vec_admit() == pytest.approx(0.80)

    def test_nan_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EM_VEC_ADMIT", "nan")
        assert beam_module._env_vec_admit() == pytest.approx(0.80)

    def test_inf_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EM_VEC_ADMIT", "inf")
        assert beam_module._env_vec_admit() == pytest.approx(0.80)

    def test_out_of_range_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EM_VEC_ADMIT", "2")
        assert beam_module._env_vec_admit() == pytest.approx(0.80)

    def test_valid_env_is_honored(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EM_VEC_ADMIT", "0.70")
        assert beam_module._env_vec_admit() == pytest.approx(0.70)


class TestInt8BlobCosine:
    """Real vec0 tables; expectations from the seeded float vectors."""

    @requires_vec
    def test_recovers_true_cosine_above_gate(self):
        """The maintainer's case: true cosine 0.82 at 1024d must score
        >= 0.80 (the constant model scored 0.7915 and rejected it)."""
        rng = np.random.default_rng(42)
        dim = 1024
        a, b = exact_cos_pair(rng, dim, 0.82)
        assert abs(float(np.dot(a, b)) - 0.82) < 1e-9
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        quantize_insert(conn, 1, a)
        quantize_insert(conn, 2, b)
        q_blob = sql_query_blob(conn, a)
        rows = sql_knn(conn, a, k=2)
        blobs = {r[0]: r[2] for r in rows}
        score = _vec_int8_blob_cosine(q_blob, blobs[2])
        # independently computed: true float cosine 0.82
        assert score >= EM_VEC_ADMIT, f"cos-0.82 pair rejected: {score}"
        assert abs(score - 0.82) <= 0.03
        conn.close()

    @requires_vec
    def test_orthogonal_scores_low(self):
        rng = np.random.default_rng(7)
        dim = 1024
        a = unit(rng.standard_normal(dim))
        c = rng.standard_normal(dim)
        c -= (np.dot(c, a) / np.dot(a, a)) * a
        c = unit(c)
        assert abs(float(np.dot(a, c))) < 1e-9
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        quantize_insert(conn, 1, a)
        quantize_insert(conn, 2, c)
        q_blob = sql_query_blob(conn, a)
        rows = sql_knn(conn, a, k=2)
        blobs = {r[0]: r[2] for r in rows}
        score = _vec_int8_blob_cosine(q_blob, blobs[2])
        assert score < 0.2, f"orthogonal pair scored {score}"
        assert 0.0 <= score <= 1.0
        conn.close()

    @requires_vec
    def test_dimension_portability_384(self):
        """Same battery at 384d within tolerance — no dimension-
        conditioned constants anywhere."""
        rng = np.random.default_rng(3)
        dim = 384
        a, b = exact_cos_pair(rng, dim, 0.82)
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        quantize_insert(conn, 1, a)
        quantize_insert(conn, 2, b)
        q_blob = sql_query_blob(conn, a)
        rows = sql_knn(conn, a, k=2)
        blobs = {r[0]: r[2] for r in rows}
        score = _vec_int8_blob_cosine(q_blob, blobs[2])
        assert score >= EM_VEC_ADMIT
        assert abs(score - 0.82) <= 0.03
        conn.close()

    def test_degenerate_inputs_abstain(self):
        assert _vec_int8_blob_cosine(b"", b"x" * 8) == 0.0
        assert _vec_int8_blob_cosine(b"x" * 8, b"") == 0.0
        assert _vec_int8_blob_cosine(b"x" * 8, b"x" * 9) == 0.0  # len mismatch
        zero = bytes(8)
        one = bytes([1] * 8)
        assert _vec_int8_blob_cosine(one, zero) == 0.0  # zero-norm row
        assert _vec_int8_blob_cosine(zero, one) == 0.0  # zero-norm query


class TestSaturatedLegacyRows:
    """Pre-normalization rows saturate the int8 byte range; sign bits are
    the surviving directional signal."""

    @requires_vec
    def test_saturated_collinear_scores_near_one(self):
        dim = 1024
        rng = np.random.default_rng(11)
        a = unit(rng.standard_normal(dim))
        saturated = np.sign(a) * 1e6  # bytes clip at 126/-128
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        quantize_insert(conn, 1, a)
        # legacy write path: un-normalized input through the same quantizer
        conn.execute(
            "INSERT INTO v(rowid, embedding) VALUES (2, vec_quantize_int8(?, 'unit'))",
            (json.dumps([float(x) for x in saturated]),),
        )
        blob = conn.execute("SELECT embedding FROM v WHERE rowid=2").fetchone()[0]
        arr = np.frombuffer(blob, dtype=np.int8).astype(np.float64)
        rms = np.linalg.norm(arr) / math.sqrt(dim)
        assert rms > 50.0, f"expected saturation regime, rms={rms}"
        q_blob = sql_query_blob(conn, a)
        score = _vec_int8_blob_cosine(q_blob, blob)
        assert score >= 0.78, (
            f"saturated collinear pair must be admitted (scored {score})"
        )
        assert score <= 0.95, (
            f"sign cap must prevent unbounded sign recovery (scored {score})"
        )
        conn.close()

    @requires_vec
    def test_moderate_magnitude_uses_law_of_cosines(self):
        """5x rows (RMS ~19) stay on the dot-product path and still
        score high for collinear pairs."""
        dim = 1024
        rng = np.random.default_rng(13)
        a = unit(rng.standard_normal(dim))
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        quantize_insert(conn, 1, a)
        conn.execute(
            "INSERT INTO v(rowid, embedding) VALUES (2, vec_quantize_int8(?, 'unit'))",
            (json.dumps([float(x) for x in a * 5.0]),),
        )
        blob = conn.execute("SELECT embedding FROM v WHERE rowid=2").fetchone()[0]
        arr = np.frombuffer(blob, dtype=np.int8).astype(np.float64)
        rms = np.linalg.norm(arr) / math.sqrt(dim)
        assert rms < 50.0, f"5x row unexpectedly saturated, rms={rms}"
        q_blob = sql_query_blob(conn, a)
        score = _vec_int8_blob_cosine(q_blob, blob)
        assert score >= 0.95, f"5x collinear pair scored {score}"
        conn.close()

    @requires_vec
    def test_mixed_store_scores_each_regime(self):
        """Normalized + legacy saturated rows in one table, same query:
        both score in their regime's expected band."""
        dim = 1024
        rng = np.random.default_rng(17)
        a = unit(rng.standard_normal(dim))
        noise = rng.standard_normal(dim)
        noise -= (np.dot(noise, a) / np.dot(a, a)) * a
        noise = unit(noise)
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        quantize_insert(conn, 1, a)                      # normalized target
        quantize_insert(conn, 2, noise)                  # normalized orthogonal
        conn.execute(
            "INSERT INTO v(rowid, embedding) VALUES (3, vec_quantize_int8(?, 'unit'))",
            (json.dumps([float(x) for x in np.sign(a) * 1e6]),),  # saturated collinear
        )
        q_blob = sql_query_blob(conn, a)
        blobs = {r[0]: r[2] for r in sql_knn(conn, a, k=3)}
        s_target = _vec_int8_blob_cosine(q_blob, blobs[1])
        s_orth = _vec_int8_blob_cosine(q_blob, blobs[2])
        s_sat = _vec_int8_blob_cosine(q_blob, blobs[3])
        assert s_target >= EM_VEC_ADMIT
        assert s_orth < 0.2
        assert s_sat >= 0.78, f"saturated row must be admitted: {s_sat}"
        assert s_sat <= 0.95, f"sign cap respected: {s_sat}"
        conn.close()


class TestTopKStabilityRealCorpus:
    """Real vec0 corpus; two query scenarios; absolute scores asserted
    against independently computed float cosines."""

    @requires_vec
    def test_near_row_absolute_score_invariant_across_top_k(self):
        dim = 512
        rng = np.random.default_rng(5)
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        query = unit(rng.standard_normal(dim))
        near = unit(query * 0.9 + unit(rng.standard_normal(dim)) * 0.15)
        true_near_cos = float(np.dot(query, near))
        quantize_insert(conn, 1, near)
        for rid in range(2, 202):
            quantize_insert(conn, rid, unit(rng.standard_normal(dim)))
        q_blob = sql_query_blob(conn, query)
        seen_pools = []
        for k in (20, 60, 180):
            rows = sql_knn(conn, query, k=k)
            pool = tuple(r[0] for r in rows)
            seen_pools.append(set(pool))
            near_row = next(r for r in rows if r[0] == 1)
            score = _vec_int8_blob_cosine(q_blob, near_row[2])
            assert score >= EM_VEC_ADMIT, f"near row not admitted at k={k}: {score}"
            assert abs(score - true_near_cos) <= 0.03
        # pools genuinely differ across k (anti-vacuity)
        assert seen_pools[0] < seen_pools[1] < seen_pools[2]
        # the same row must produce the SAME absolute score at every k —
        # recompute from the stored blob (not the cached value)
        scores = set()
        for k in (20, 60, 180):
            rows = sql_knn(conn, query, k=k)
            near_row = next(r for r in rows if r[0] == 1)
            scores.add(round(_vec_int8_blob_cosine(q_blob, near_row[2]), 9))
        assert len(scores) == 1, f"score varies with top_k: {scores}"
        conn.close()

    @requires_vec
    def test_far_but_nearest_not_admitted(self):
        """Query with NO close neighbor: the nearest row is far and must
        not cross the admission gate."""
        dim = 512
        rng = np.random.default_rng(23)
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        query = unit(rng.standard_normal(dim))
        for rid in range(1, 201):
            quantize_insert(conn, rid, unit(rng.standard_normal(dim)))
        q_blob = sql_query_blob(conn, query)
        rows = sql_knn(conn, query, k=20)
        best = rows[0]
        best_cos = _vec_int8_blob_cosine(q_blob, best[2])
        # Property assertion: a random corpus has no true neighbor for this
        # query, so even the nearest row must stay under the gate.
        assert best_cos < EM_VEC_ADMIT, (
            f"random-corpus nearest row admitted: {best_cos}"
        )
        conn.close()


class TestPersistentFailureDegrades:
    """R4-H1: a PERSISTENT vec failure (lock/corruption class) must degrade
    recall to no vector candidates — the plain _vec_search fallback itself
    re-raises, and the handler must not propagate that crash."""

    def test_persistent_lock_never_crashes_recall(self, temp_db, monkeypatch):
        import sqlite3 as _sq
        beam = BeamMemory(session_id="plock", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO episodic_memory "
            "(id, content, source, timestamp, session_id, importance, scope, memory_type) "
            "VALUES ('r1', 'alpha target', 'sleep_consolidation', datetime('now'), 'sess', 0.6, 'global', 'fact')",
        )
        beam.conn.commit()

        calls = {"n": 0}

        def locked_strict(conn):
            calls["n"] += 1
            raise _sq.OperationalError("database is locked")

        def locked_search(conn, emb, k=20):
            # the REAL _vec_search re-detects the type and re-raises
            raise _sq.OperationalError("database is locked")

        monkeypatch.setattr(beam_module, "_vec_search", locked_search)
        monkeypatch.setattr(beam_module, "_vec_search_with_blobs",
                            lambda conn, emb, k=20: locked_strict(conn))
        monkeypatch.setattr(beam_module, "_vec_available", lambda conn: True)
        monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
        monkeypatch.setattr(beam_module._embeddings, "embed_query",
                            lambda text: np.zeros(1024, dtype=np.float32))
        monkeypatch.setattr(beam_module._embeddings, "embed",
                            lambda texts: [np.zeros(1024, dtype=np.float32)])

        # must not raise — recall still returns (FTS-only) results
        res = beam.recall("alpha target", top_k=5)
        assert isinstance(res, list)


class TestFloat32LegacyBlobScoring:
    """R4-M3: legacy un-normalized float32 rows must score their exact
    cosine from the stored blob, not clamp to 0 via the unit-norm
    distance conversion."""

    @requires_vec
    def test_legacy_5x_float32_row_admitted(self):
        dim = 512
        rng = np.random.default_rng(29)
        a = unit(rng.standard_normal(dim))
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        conn.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")
        conn.execute(
            "INSERT INTO v(rowid, embedding) VALUES (1, ?)",
            (json.dumps([float(x) for x in a]),),
        )
        conn.execute(
            "INSERT INTO v(rowid, embedding) VALUES (2, ?)",
            (json.dumps([float(x) for x in a * 5.0]),),  # legacy un-normalized
        )
        rows = conn.execute(
            "SELECT rowid, distance, embedding FROM v "
            "WHERE embedding MATCH ? AND k=2 ORDER BY distance",
            (json.dumps([float(x) for x in a]),),
        ).fetchall()
        for rid, dist, blob in rows:
            if rid == 2:
                score = beam_module._vec_float32_blob_cosine(
                    a.astype(np.float32), blob
                )
                assert score >= EM_VEC_ADMIT, (
                    f"legacy float32 row rejected: {score}"
                )
                assert abs(score - 1.0) <= 0.01
        conn.close()


class TestLegacyBoundaryBand:
    """R4-M2: legacy rows in the partial-clipping RMS band (50-100) must
    not flip below the gate on sign-arm noise — max(dot, sign) covers."""

    @requires_vec
    def test_15x_legacy_true_cos_082_admitted(self):
        dim = 1024
        rng = np.random.default_rng(31)
        a, b = exact_cos_pair(rng, dim, 0.82)
        conn = sqlite3.connect(":memory:")
        _load_vec(conn)
        make_table(conn, dim)
        quantize_insert(conn, 1, a)
        # legacy write at 15x magnitude — partial clipping, RMS ~57
        conn.execute(
            "INSERT INTO v(rowid, embedding) VALUES (2, vec_quantize_int8(?, 'unit'))",
            (json.dumps([float(x) for x in b * 15.0]),),
        )
        blob = conn.execute("SELECT embedding FROM v WHERE rowid=2").fetchone()[0]
        arr = np.frombuffer(blob, dtype=np.int8).astype(np.float64)
        rms = np.linalg.norm(arr) / math.sqrt(dim)
        assert rms > 50.0, f"expected boundary band, rms={rms}"
        q_blob = sql_query_blob(conn, a)
        score = _vec_int8_blob_cosine(q_blob, blob)
        assert score >= EM_VEC_ADMIT, f"boundary-band legacy row rejected: {score}"
        conn.close()
