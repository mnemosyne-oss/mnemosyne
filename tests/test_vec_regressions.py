"""Regression tests: vec scoring arms and legacy-store classification.

- sign arm gated on ROW clip fraction (not RMS), zero-aware Hamming, and a
  query-zero-fraction weight
- bit-arm similarity normalizes by the LIVE bit width, not the configured
  EMBEDDING_DIM (dplush's exact numbers: 0.895966 -> 0.336890)
- legacy-store classification uses a durable normalized-format boundary and
  routes every unmarked store conservatively
- bit-blob exact scoring + per-query explain vec_mode
"""
from __future__ import annotations

import logging
import math
import queue
import threading
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "vec_regime.db"

import mnemosyne.core.beam as beam_module
from mnemosyne.core.beam import (
    BeamMemory,
    _vec_distance_sim,
    _vec_int8_blob_cosine,
)


bm = beam_module

try:
    import sqlite_vec  # noqa: F401
    _HAS_SQLITE_VEC = True
except Exception:
    _HAS_SQLITE_VEC = False


requires_vec = pytest.mark.skipif(
    not _HAS_SQLITE_VEC, reason="sqlite-vec not available"
)


def _qblob(vec) -> bytes:
    """Mirror production quantization: unit-normalize then max-abs scale to
    int8 (vec_quantize_int8 'unit' semantics at 1024 dims)."""
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    m = np.max(np.abs(v))
    if m == 0:
        return bytes(len(v))
    scaled = np.clip(v / m * 126.0, -127, 127)
    return scaled.astype(np.int8).tobytes()


def _rng():
    return np.random.default_rng(42)


class TestSignArmGate:
    def test_dplush_repro_all126_vs_zero_heavy_query_rejected(self):
        # Row [126]*1024 and a 98.4%-zero query: the old max(cos, sign)
        # manufactured 1.0; the correct byte-dot is 0.125.
        q = np.concatenate([np.full(16, 0.0625), np.zeros(1008)])
        row = np.full(1024, 1.0)
        score = _vec_int8_blob_cosine(_qblob(q), _qblob(row))
        assert score == pytest.approx(0.125, abs=0.01)

    def test_collinear_saturated_pair_recovers(self):
        # Both sides zero-free and collinear: the sign arm must recover ~1.0.
        q = np.ones(1024)
        row = np.full(1024, 1.0)
        score = _vec_int8_blob_cosine(_qblob(q), _qblob(row))
        assert score >= 0.95

    def test_d_attack_magnitude_on_zero_dims_rejected(self):
        # Magnitude placed only on query-zero dims, tiny elsewhere: the
        # manufactured sign surface is priced off by the zero-fraction
        # weight and the tiny byte-dot; score must stay far below admission.
        rng = _rng()
        q = rng.standard_normal(1024)
        qb = _qblob(q)
        qi = np.frombuffer(qb, dtype=np.int8).astype(np.float64)
        row = np.where(
            qi == 0, 126.0 * np.sign(rng.standard_normal(1024)), 1.0
        )
        row = row.astype(np.int8).tobytes()
        score = _vec_int8_blob_cosine(qb, row)
        assert score < 0.10, "the manufactured-margin attack must score ~0"

    def test_normalized_unrelated_pair_unaffected(self):
        rng = _rng()
        score = _vec_int8_blob_cosine(
            _qblob(rng.standard_normal(1024)),
            _qblob(rng.standard_normal(1024)),
        )
        assert score < 0.5


class TestBitWidth:
    def test_live_width_not_configured_dim(self):
        # dplush's exact numbers: hamming 150 over a 384-bit table with a
        # process config of 1024 must score 0.337 (rejected), not 0.896.
        score = _vec_distance_sim(150, "bit", bit_width=384)
        assert score == pytest.approx(0.336890, abs=0.001)

    def test_config_fallback_when_no_width(self):
        # bit_width=None must take the configured-dim fallback branch —
        # derive the expectation from the module's EMBEDDING_DIM so the
        # test exercises the actual fallback arithmetic, not a magic
        # 1024 constant.
        import mnemosyne.core.beam as bm
        d = bm.EMBEDDING_DIM
        expected = math.cos(math.pi * 150 / d)
        score = _vec_distance_sim(150, "bit", bit_width=None)
        assert score == pytest.approx(expected, abs=0.001)


class TestLegacyStoreClassification:
    @staticmethod
    def _seed_vec_rows(conn, dim: int, legacy: bool):
        # vec_episodes already exists (BeamMemory init creates it when
        # sqlite-vec is importable). Quantize via the production SQL path.
        # legacy=True mimics pre-normalization stores: an all-positive
        # 10x-magnitude vector saturates (max-abs scale pins 126).
        import json as _json
        import numpy as np
        rng = np.random.default_rng(7 if legacy else 42)
        vec = rng.standard_normal(dim).astype(np.float32)
        if not legacy:
            # Mirror production _vec_table_insert: numpy unit-normalize
            # BEFORE the SQL quantize (the 'unit' param silently fails at
            # 1024 dims), so bytes land in the normal ~3.9 rms band.
            n = np.linalg.norm(vec)
            vec = (vec / n).astype(np.float32)
        else:
            # Pre-normalization stores: large-magnitude vectors whose
            # max-abs quantization saturates bytes at ±126.
            vec = (vec - vec.min() + 1.0) * 10.0
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (1,"
            " vec_quantize_int8(?, 'unit'))",
            (_json.dumps(vec.tolist()),),
        )
        conn.commit()

    @requires_vec
    def test_marked_store_pure_unmarked_conservative_warned_once(
            self, temp_db, monkeypatch, caplog):
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-legacy", db_path=temp_db)
        dim = beam_module.EMBEDDING_DIM
        self._seed_vec_rows(beam.conn, dim, legacy=True)
        # Boundary routing: the store was CREATED by current code, so the
        # normalized-format marker is set -> pure routing regardless of
        # the seeded row magnitudes (no sampling).
        assert bm._classify_vec_store_regime(beam.conn, "vec_episodes") == "pure"
        # Clearing the marker flips routing conservative (unmarked store
        # may contain pre-format rows).
        uv = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        beam.conn.execute(f"PRAGMA user_version = {uv & ~bm._VEC_NORM_BIT}")
        assert bm._classify_vec_store_regime(beam.conn, "vec_episodes") == "legacy"
        monkeypatch.setattr(bm, "_legacy_warning_emitted", False)
        with caplog.at_level(30, logger="mnemosyne.core.beam"):
            before = len([r for r in caplog.records
                          if r.name == "mnemosyne.core.beam"
                          and "normalized-format marker" in r.message])
            bm._warn_vec_store_legacy_once()
            bm._warn_vec_store_legacy_once()
            after = len([r for r in caplog.records
                         if r.name == "mnemosyne.core.beam"
                         and "normalized-format marker" in r.message])
        assert after - before == 1, (
            f"expected exactly one legacy warning, got {after - before}"
        )

    def test_unknown_marker_warning_is_thread_safe(self, monkeypatch, caplog):
        callers = 8
        start = threading.Barrier(callers)
        outcomes = queue.Queue()

        def call_warning():
            try:
                start.wait(timeout=5)
                bm._warn_vec_store_unknown_once()
            except Exception as exc:
                outcomes.put(exc)
            else:
                outcomes.put(None)

        monkeypatch.setattr(bm, "_unknown_marker_warning_emitted", False)
        caplog.clear()
        threads = [threading.Thread(target=call_warning) for _ in range(callers)]
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        assert not any(thread.is_alive() for thread in threads)
        results = [outcomes.get_nowait() for _ in threads]
        assert results == [None] * callers
        marker_warnings = [
            record for record in caplog.records
            if record.levelno == logging.WARNING
            and "vec store format marker unreadable" in record.getMessage()
        ]
        assert len(marker_warnings) == 1


    @requires_vec
    def test_pure_store_classified_pure(self, temp_db):
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-pure", db_path=temp_db)
        dim = beam_module.EMBEDDING_DIM
        self._seed_vec_rows(beam.conn, dim, legacy=False)
        regime = bm._classify_vec_store_regime(beam.conn, "vec_episodes")
        assert regime == "pure"


class TestLegacyClassificationRefinements:
    def test_xor_bonus_uses_bit_width_not_byte_count(self):
        """h_dist counts bits but xor_arr is the packed byte array. Dividing
        by len(xor_arr) inflates the distance 8x and zeroes
        the bonus. Normalized distance for fully-opposite vectors must be
        1.0 (not 8.0)."""
        import numpy as np
        bits_q = np.zeros(64, dtype=bool)
        bits_r = np.ones(64, dtype=bool)
        pq = np.packbits(bits_q).tobytes()
        pr = np.packbits(bits_r).tobytes()
        xor_arr = np.frombuffer(
            bytes(a ^ b for a, b in zip(pq, pr)), dtype=np.uint8
        )
        popcount = np.array(
            [bin(i).count("1") for i in range(256)], dtype=np.uint32
        )
        h = int(np.sum(popcount[xor_arr]))
        assert h == 64                      # fully opposite
        assert len(xor_arr) == 8            # packed bytes
        assert h / (len(xor_arr) * 8) == pytest.approx(1.0)

    def test_zero_aware_hamming_recovers_honest_cosine(self):
        """Query-zero dims must be excluded from both the numerator and
        denominator. The zero-aware arm must recover at least the honest byte
        cosine of a sign-following saturated row."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal(1024)
        qb = _qblob(q)
        qi = np.frombuffer(qb, dtype=np.int8).astype(np.float64)
        row = (np.sign(qi) * 126).astype(np.int8).tobytes()
        score = _vec_int8_blob_cosine(qb, row)
        honest = float(
            np.dot(qi, np.frombuffer(row, dtype=np.int8).astype(np.float64))
            / (np.linalg.norm(qi) * np.linalg.norm(
                np.frombuffer(row, dtype=np.int8).astype(np.float64)))
        )
        assert score >= honest - 0.02
        assert score >= 0.77

    def test_few_stray_zero_bytes_do_not_kill_legacy_recovery(self):
        """The row zero-free sub-gate must tolerate a few stray quantization
        zeros. A saturated row with 8 stray zeros must still recover."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal(1024)
        qb = _qblob(q)
        qi = np.frombuffer(qb, dtype=np.int8)
        row = bytearray((np.sign(qi) * 126).astype(np.int8).tobytes())
        for pos in range(0, 1024, 128):
            row[pos] = 0
        score = _vec_int8_blob_cosine(qb, bytes(row))
        # 8 zeroed dims cost the byte arm honestly; the gate must still
        # admit (the whole point of the 0.01 relaxation).
        assert score >= 0.78

    @requires_vec


    def test_reindex_sets_norm_bit(self, temp_db, monkeypatch):
        """A reindex re-quantizes every row (all normalized), so it must
        set the normalized-format marker and flip an unmarked store back
        to the fast KNN route."""
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-bit", db_path=temp_db)
        uv = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        beam.conn.execute(f"PRAGMA user_version = {uv & ~bm._VEC_NORM_BIT}")
        assert bm._classify_vec_store_regime(beam.conn, "vec_episodes") == "legacy"
        dim = beam_module.EMBEDDING_DIM
        monkeypatch.setattr(bm._embeddings, "available", lambda: True)
        monkeypatch.setattr(
            bm._embeddings, "embed",
            lambda texts: [np.zeros(dim, dtype=np.float32) for _ in texts],
        )
        bm.reindex_vectors(beam.conn)
        uv2 = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        assert uv2 & bm._VEC_NORM_BIT, "reindex must set the normalized-format marker"
        assert bm._classify_vec_store_regime(beam.conn, "vec_episodes") == "pure"



class TestLegacyRoutingAndFallback:
    """Regression coverage for legacy routing and in-memory vector scoring."""

    def test_in_memory_search_rowid_key_and_negative_clamp(self, temp_db):
        """The fallback scan exposed em.rowid without an alias — sqlite3.Row
        reported the key as the declared PK name on some schemas, and
        row["rowid"] raised KeyError (swallowed) -> the scan silently
        returned []. Also: 1 - sim can go slightly negative on
        near-identical vectors, and the None arm rejects negative
        distances -> the best match scored 0.0. Both are pinned here."""
        import json as _json
        import sqlite3 as _sq
        import mnemosyne.core.beam as bm

        conn = _sq.connect(":memory:")
        conn.row_factory = _sq.Row
        conn.execute(
            "CREATE TABLE episodic_memory (id INTEGER PRIMARY KEY, content TEXT)"
        )
        conn.execute("INSERT INTO episodic_memory VALUES (1, 'x')")
        conn.execute(
            "CREATE TABLE memory_embeddings (memory_id TEXT, embedding_json TEXT)"
        )
        vec = [1.0, 0.0, 0.0]
        conn.execute(
            "INSERT INTO memory_embeddings VALUES ('1', ?)",
            (_json.dumps(vec),),
        )
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        rows = bm._in_memory_vec_search(conn, q, k=5)
        assert rows, "fallback scan must return the candidate row"
        assert rows[0]["rowid"] == 1
        distance = rows[0]["distance"]
        assert distance >= 0.0, (
            f"distance must be clamped non-negative, got {distance}"
        )
        sim = bm._vec_distance_sim(distance, None)
        assert sim == pytest.approx(1.0, abs=1e-6)

    @requires_vec
    def test_legacy_regime_routes_to_full_scan(self, temp_db, monkeypatch):
        """dplush public repro (review 5122480102): 30 normalized
        distractors + one collinear legacy target in a REAL float32 vec
        table. Raw-L2 KNN buries the target; regime=legacy must route
        candidates through the full-scan blob strategy so public
        recall() recovers it."""
        import json as _json
        import sqlite_vec
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-route", db_path=temp_db)
        beam.conn.enable_load_extension(True)
        sqlite_vec.load(beam.conn)
        beam.conn.enable_load_extension(False)
        beam.conn.execute("DROP TABLE IF EXISTS vec_episodes")
        dim = 64
        beam.conn.execute(
            "CREATE VIRTUAL TABLE vec_episodes "
            "USING vec0(embedding float[{d}])".format(d=dim)
        )
        # The raw re-create simulates a pre-format store: the marker the
        # BeamMemory init wrote must be cleared, or routing stays pure.
        _uv = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        beam.conn.execute(f"PRAGMA user_version = {_uv & ~bm._VEC_NORM_BIT}")
        rng = np.random.default_rng(3)
        q = rng.standard_normal(dim)
        q = (q / np.linalg.norm(q)).astype(np.float32)

        cur = beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp,"
            " session_id, importance, scope, memory_type)"
            " VALUES ('tgt', 'legacy collinear target zqz',"
            " 'sleep_consolidation', datetime('now'), 's-route', 0.9,"
            " 'session', 'episodic')"
        )
        target_rowid = cur.lastrowid
        beam.conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (target_rowid, _json.dumps([float(x) for x in (q * 8.0)])),
        )
        for i in range(30):
            v = rng.standard_normal(dim)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            c2 = beam.conn.execute(
                "INSERT INTO episodic_memory (id, content, source,"
                " timestamp, session_id, importance, scope, memory_type)"
                " VALUES (?, ?, 'sleep_consolidation', datetime('now'),"
                " 's-route', 0.5, 'session', 'episodic')",
                (f"d{i}", f"distractor {i} lorem"),
            )
            beam.conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
                (c2.lastrowid, _json.dumps([float(x) for x in v])),
            )

        regime = bm._classify_vec_store_regime(beam.conn, "vec_episodes")
        assert regime == "legacy"

        knn = beam.conn.execute(
            "SELECT rowid FROM vec_episodes WHERE embedding MATCH ? AND k=20"
            " ORDER BY distance",
            (_json.dumps([float(x) for x in q]),),
        ).fetchall()
        assert all(r[0] != target_rowid for r in knn), (
            "repro precondition broken: target not buried in KNN"
        )

        class FakeEmb:
            @staticmethod
            def available():
                return True

            @staticmethod
            def embed_query(text):
                return q

            @staticmethod
            def embed(texts):
                return [q for _ in texts]

        monkeypatch.setattr(bm._embeddings, "available", lambda: True)
        monkeypatch.setattr(bm._embeddings, "embed_query", FakeEmb.embed_query)
        monkeypatch.setattr(bm._embeddings, "embed", FakeEmb.embed)
        k_used = []
        real_knn = bm._vec_search_with_blobs
        def kno_spy(conn, emb, k=20):
            k_used.append(k)
            return real_knn(conn, emb, k)
        monkeypatch.setattr(bm, "_vec_search_with_blobs", kno_spy)

        res = beam.recall("legacy collinear target zqz", top_k=5)
        assert not k_used, (
            "legacy regime must not use the raw-L2 KNN path"
        )
        row = next(
            (r for r in res if r.get("content") == "legacy collinear target zqz"),
            None,
        )
        assert row is not None, "legacy target missing from recall results"
        assert row["dense_score"] >= bm.EM_VEC_ADMIT, (
            f"legacy target under-admitted: {row['dense_score']}"
        )


def _regime(conn):
    return beam_module._classify_vec_store_regime(conn, "vec_episodes")


@pytest.fixture
def beam_db(tmp_path):
    beam = bm.BeamMemory(session_id="s-r7", db_path=Path(tmp_path) / "m.db")
    yield beam
    beam.conn.close()


class FakeEmb:
    """Deterministic embedding model."""

    def __init__(self, dim=64, seed=0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [self.rng.standard_normal(self.dim).astype(np.float32)
                for _ in texts]

    def available(self):
        return True


@pytest.fixture
def fake_emb(monkeypatch):
    fe = FakeEmb()
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(bm._embeddings, "embed", fe.embed)
    monkeypatch.setattr(
        bm._embeddings, "embed_query",
        lambda text: fe.embed(text)[0])
    return fe


def _mk_vec_table(conn, vtype, dim, n, seed=1, quantize_rows=None):
    import sqlite_vec as sv
    conn.enable_load_extension(True)
    sv.load(conn)
    rng = np.random.default_rng(seed)
    # vec0 table + memory_embeddings rows (drop pre-existing: BeamMemory
    # init may already have created a default vec table)
    conn.execute("DROP TABLE IF EXISTS vec_episodes")
    conn.execute("DELETE FROM memory_embeddings")
    conn.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding {vtype}[{dim}])")
    import json as _json
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        emb_json = _json.dumps(vec.tolist())
        if vtype == "int8":
            conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
                (i + 1, emb_json))
        elif vtype == "bit":
            conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_binary(?))",
                (i + 1, emb_json))
        else:
            conn.execute("INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
                         (i + 1, emb_json))
        conn.execute("INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?, ?)",
                     (str(i + 1), emb_json))
    conn.commit()


@requires_vec
class TestClassifierTypeAware:
    """Boundary routing replaces the probabilistic sampler: the
    normalized-format marker decides pure vs conservative, never a
    sampled row heuristic (maintainer P1)."""

    @requires_vec
    def test_marked_store_routes_pure_every_type(self, beam_db):
        for vtype, dim in (("int8", 64), ("float32", 64), ("bit", 64)):
            _mk_vec_table(beam_db.conn, vtype, dim, 12, seed=1)
            bm._mark_vec_store_norm_bit(beam_db.conn)
            assert _regime(beam_db.conn) == "pure", vtype

    @requires_vec
    def test_unmarked_store_routes_conservative_every_type(self, beam_db):
        for vtype, dim in (("int8", 64), ("float32", 64), ("bit", 64)):
            _mk_vec_table(beam_db.conn, vtype, dim, 12, seed=2)
            uv = beam_db.conn.execute("PRAGMA user_version").fetchone()[0]
            beam_db.conn.execute(f"PRAGMA user_version = {uv & ~bm._VEC_NORM_BIT}")
            assert _regime(beam_db.conn) == "legacy", vtype

    @requires_vec
    def test_norm_bit_survives_row_rewrites(self, beam_db):
        # Writers always normalize (see _vec_table_insert), so a marked
        # store stays safely marked through row inserts/deletes/reinserts.
        _mk_vec_table(beam_db.conn, "int8", 64, 20, seed=3)
        bm._mark_vec_store_norm_bit(beam_db.conn)
        conn = beam_db.conn
        import json as _json
        import numpy as np
        rng = np.random.default_rng(9)
        v = rng.standard_normal(64).astype(np.float32)
        v = v / np.linalg.norm(v)
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
            (99, _json.dumps(v.tolist())))
        conn.execute("DELETE FROM vec_episodes WHERE rowid = 99")
        conn.commit()
        assert _regime(conn) == "pure"

    @requires_vec
    def test_historical_legacy_bit_keeps_conservative(self, beam_db):
        # Stores flagged by pre-boundary code keep routing conservatively
        # even when the marker is present, until a reindex clears the bit.
        _mk_vec_table(beam_db.conn, "int8", 64, 12, seed=4)
        bm._mark_vec_store_norm_bit(beam_db.conn)
        uv = beam_db.conn.execute("PRAGMA user_version").fetchone()[0]
        beam_db.conn.execute(f"PRAGMA user_version = {uv | 0x10000000}")
        assert _regime(beam_db.conn) == "legacy"


class TestBitBlobScoring:
    def test_bit_blob_cosine_exact(self):
        # identical bytes -> cos 1.0
        assert bm._vec_bit_blob_cosine(b"\x00\xff", b"\x00\xff") == pytest.approx(1.0)
        # opposite bytes -> cos(pi * 8/16) = 0.0
        assert bm._vec_bit_blob_cosine(b"\xff\xff", b"\x00\x00", width=16) == pytest.approx(
            np.cos(np.pi * 8 / 16))
        # 4 of 16 bits differ -> cos(pi*4/16) = cos(pi/4) ~ 0.707
        assert bm._vec_bit_blob_cosine(b"\x03\x00", b"\x0c\x00", width=16) == pytest.approx(
            np.cos(np.pi * 4 / 16))

    def test_mismatched_length_abstains(self):
        assert bm._vec_bit_blob_cosine(b"\x00\xff", b"\x00") == 0.0
        assert bm._vec_bit_blob_cosine(None, b"\x00\xff") == 0.0


@requires_vec
class TestExplainVecMode:
    def test_explain_reports_legacy_scan(self, beam_db, fake_emb):
        # build a store that classifies legacy
        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[64])")
        _uv = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.execute(f"PRAGMA user_version = {_uv & ~bm._VEC_NORM_BIT}")
        import json as _json
        for i in range(10):
            conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
                (i + 1, _json.dumps([300.0] * 64)))
        conn.commit()
        # direct episodic row (FTS anchor) for the recall to return:
        conn.execute(
            "INSERT OR REPLACE INTO episodic_memory "
            "(id, session_id, content, source, timestamp, importance, created_at, event_date_precision) "
            "VALUES ('ep1', 's-r7', 'zebra animal striped', 'conversation', "
            "'2020-01-01T00:00:00', 0.5, '2020-01-01T00:00:00', 'unknown')")
        conn.commit()
        exp = beam_db.recall("zebra", explain=True)
        assert exp["explain"].get("vec_mode") == "legacy_scan", exp

    def test_mark_norm_bit_skips_write_when_already_set(self, beam_db):
        """Marking an already-marked store must not issue a second
        PRAGMA user_version write: the no-op write still acquires the
        write lock and bumps other connections' data_version, which
        invalidates their caches and forces extra scans."""
        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[8])")
        conn.commit()
        bm._mark_vec_store_norm_bit(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] & bm._VEC_NORM_BIT

        writes = []
        real_execute = conn.execute

        def spy_execute(sql, *args, **kwargs):
            if isinstance(sql, str) and sql.startswith("PRAGMA user_version ="):
                writes.append(sql)
            return real_execute(sql, *args, **kwargs)

        conn.execute = spy_execute
        try:
            bm._mark_vec_store_norm_bit(conn)
        finally:
            conn.execute = real_execute
        assert writes == [], f"no-op PRAGMA write issued: {writes}"

    def test_fresh_connection_skips_noop_norm_write(self, beam_db):
        # The no-op-write guard must hold across connections (the
        # user_version bits live in the file).
        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[8])")
        conn.commit()
        bm._mark_vec_store_norm_bit(conn)
        import sqlite3 as _sqlite3
        fresh = _sqlite3.connect(str(beam_db.db_path), factory=type(conn))
        writes = []
        real_execute = fresh.execute

        def spy_execute(sql, *args, **kwargs):
            if isinstance(sql, str) and sql.startswith("PRAGMA user_version ="):
                writes.append(sql)
            return real_execute(sql, *args, **kwargs)

        fresh.execute = spy_execute
        try:
            bm._mark_vec_store_norm_bit(fresh)
        finally:
            fresh.execute = real_execute
            fresh.close()
        assert writes == [], f"fresh-connection no-op PRAGMA write: {writes}"


class TestLegacyScanFullCoverage:
    @requires_vec
    def test_rows_beyond_old_cap_are_scored(self, beam_db):
        """The legacy scan streams the FULL table: a high-cosine row at
        rowid > 10000 (the old unordered-LIMIT cap) must reach the
        top-k blob-scored results."""
        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[64])")
        import json as _json
        rng = np.random.default_rng(21)
        # 9999 random filler rows (rowids 1..9999) + 51 more filler
        # (10000..10050): total 10050 rows, so the old unordered
        # LIMIT-10000 prefix excluded every rowid >= 10000.
        rows = []
        for i in range(10050):
            v = rng.standard_normal(64).astype(np.float32)
            v = v / np.linalg.norm(v)
            rows.append((i + 1, _json.dumps(v.tolist())))
        # target row at rowid 10051 (beyond any 10000-row prefix)
        target = np.zeros(64, dtype=np.float32)
        target[0] = 1.0
        rows.append((10051, _json.dumps(target.tolist())))
        conn.executemany(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
            rows)
        conn.commit()
        query = target  # exact match: cosine 1.0
        got = bm._vec_legacy_scan_with_blobs(conn, query.tolist(), k=20)
        rowids = {r["rowid"] for r in got[0]}
        assert 10051 in rowids, "target beyond the old cap never reached the scorer"
        # and the top of the heap is the target itself (sorted desc)
        assert got[0][0]["rowid"] == 10051, "target not ranked first"


class TestBoundaryRoutingRegression:
    """Maintainer P1: a legacy target in the MIDDLE of a large unmarked
    store must be retrieved. Routing is the format boundary — no sampling
    to force-miss — so an unmarked store always uses the conservative
    exact-cosine full scan."""

    @requires_vec
    def test_mid_table_legacy_target_found_by_recall(
            self, temp_db, monkeypatch, caplog):
        import json as _json
        import sqlite3 as _sqlite3
        import sqlite_vec
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-mid", db_path=temp_db)
        beam.conn.enable_load_extension(True)
        sqlite_vec.load(beam.conn)
        beam.conn.enable_load_extension(False)
        beam.conn.execute("DROP TABLE IF EXISTS vec_episodes")
        beam.conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float[64])")
        # unmarked store (pre-format): conservative route
        _uv = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        beam.conn.execute(f"PRAGMA user_version = {_uv & ~bm._VEC_NORM_BIT}")

        rng = np.random.default_rng(11)
        q = rng.standard_normal(64)
        q = (q / np.linalg.norm(q)).astype(np.float32)
        # 1499 normalized distractors (rowids 1..1499)
        for i in range(1499):
            v = rng.standard_normal(64)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            c2 = beam.conn.execute(
                "INSERT INTO episodic_memory (id, content, source, timestamp,"
                " session_id, importance, scope, memory_type)"
                " VALUES (?, ?, 'sleep_consolidation', datetime('now'),"
                " 's-mid', 0.5, 'session', 'episodic')",
                (f"d{i}", f"distractor {i} lorem ipsum"),
            )
            beam.conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
                (c2.lastrowid, _json.dumps([float(x) for x in v])),
            )
        # legacy target: 8x collinear magnitude, rowid lands MID-TABLE
        cur = beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp,"
            " session_id, importance, scope, memory_type)"
            " VALUES ('tgt', 'mid-table legacy target zqxz',"
            " 'sleep_consolidation', datetime('now'), 's-mid', 0.9,"
            " 'session', 'episodic')"
        )
        target_rowid = cur.lastrowid
        beam.conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (target_rowid, _json.dumps([float(x) for x in (q * 8.0)])),
        )
        for i in range(1499, 2999):
            v = rng.standard_normal(64)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            c2 = beam.conn.execute(
                "INSERT INTO episodic_memory (id, content, source, timestamp,"
                " session_id, importance, scope, memory_type)"
                " VALUES (?, ?, 'sleep_consolidation', datetime('now'),"
                " 's-mid', 0.5, 'session', 'episodic')",
                (f"d{i}", f"distractor {i} lorem ipsum"),
            )
            beam.conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
                (c2.lastrowid, _json.dumps([float(x) for x in v])),
            )
        beam.conn.commit()
        # precondition: raw-L2 KNN buries the 8x target
        knn = beam.conn.execute(
            "SELECT rowid FROM vec_episodes WHERE embedding MATCH ? AND k=20"
            " ORDER BY distance",
            (_json.dumps([float(x) for x in q]),),
        ).fetchall()
        assert all(r[0] != target_rowid for r in knn), (
            "repro precondition broken: target not buried in KNN"
        )

        class FakeEmb:
            @staticmethod
            def available():
                return True

            @staticmethod
            def embed_query(text):
                return q

            @staticmethod
            def embed(texts):
                return [q for _ in texts]

        monkeypatch.setattr(bm._embeddings, "available", lambda: True)
        monkeypatch.setattr(bm._embeddings, "embed_query", FakeEmb.embed_query)
        monkeypatch.setattr(bm._embeddings, "embed", FakeEmb.embed)

        knn_calls = []
        real_knn = bm._vec_search_with_blobs

        def knn_spy(conn, emb, k=20):
            knn_calls.append(k)
            return real_knn(conn, emb, k)

        monkeypatch.setattr(bm, "_vec_search_with_blobs", knn_spy)

        legacy = beam.recall("mid-table legacy target zqxz", top_k=5)

        marker_reads = []
        conn_type = type(beam.conn)
        real_execute = conn_type.execute

        def execute_without_user_version(
                conn, sql, parameters=(), *args, **kwargs):
            normalized = (
                sql.strip().rstrip(";").casefold()
                if isinstance(sql, str)
                else ""
            )
            if normalized == "pragma user_version":
                marker_reads.append(sql)
                raise _sqlite3.OperationalError("marker header unreadable")
            return real_execute(conn, sql, parameters, *args, **kwargs)

        monkeypatch.setattr(conn_type, "execute", execute_without_user_version)
        monkeypatch.setattr(bm, "_unknown_marker_warning_emitted", False)
        caplog.clear()
        with caplog.at_level(10, logger="mnemosyne.core.beam"):
            unknown_explained = beam.recall(
                "mid-table legacy target zqxz", top_k=5, explain=True
            )
            unknown_again = beam.recall("mid-table legacy target zqxz", top_k=5)

        assert len(marker_reads) == 2, "regression did not deny both marker reads"
        assert unknown_explained["explain"].get("vec_mode") == "legacy_scan"
        unknown = unknown_explained["results"]

        def result_signature(results):
            return [
                (
                    result["id"],
                    {
                        key: value
                        for key, value in result.items()
                        if key == "score" or key.endswith("_score")
                    },
                )
                for result in results
            ]

        assert result_signature(unknown) == result_signature(legacy)
        assert result_signature(unknown_again) == result_signature(legacy)
        assert not knn_calls, "non-pure regimes must not use the raw-L2 KNN path"

        marker_warnings = [
            record for record in caplog.records
            if record.levelno == 30
            and "vec store format marker unreadable" in record.getMessage()
        ]
        assert len(marker_warnings) == 1
        assert any(
            record.levelno == logging.DEBUG
            and "full-scan blob scoring this call" in record.getMessage()
            for record in caplog.records
        )
        assert not any(
            record.levelno == 20
            and "full-scan blob scoring this call" in record.getMessage()
            for record in caplog.records
        )

        row = next(
            (r for r in unknown if r.get("content") == "mid-table legacy target zqxz"),
            None,
        )
        assert row is not None, "mid-table legacy target missing from recall"
        assert row["dense_score"] >= bm.EM_VEC_ADMIT, (
            f"target under-admitted: {row['dense_score']}"
        )


class TestConservativeScanCoverage:
    """Bit-store coverage through the conservative scan, including close,
    distant and forced-failure cases."""

    @requires_vec
    def test_bit_store_scan_ranks_close_above_distant(self, beam_db):
        import json as _json
        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding bit[64])")
        rng = np.random.default_rng(13)
        q = rng.standard_normal(64)
        q = (q / np.linalg.norm(q)).astype(np.float32)
        close = q + rng.standard_normal(64).astype(np.float32) * 0.05  # near-duplicate
        far = -q  # opposite direction
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (1, vec_quantize_binary(?))",
            (_json.dumps(close.tolist()),))
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (2, vec_quantize_binary(?))",
            (_json.dumps(far.tolist()),))
        conn.commit()
        rows, qb = bm._vec_legacy_scan_with_blobs(conn, q.tolist(), k=10)
        assert rows[0]["rowid"] == 1, "close row must rank first"
        sims = {r["rowid"]: bm._vec_bit_blob_cosine(qb, r["blob"]) for r in rows}
        assert sims[1] > sims[2]
        assert sims[1] >= 0.8, f"close row under-admitted: {sims[1]}"

    @requires_vec
    def test_scan_failure_keeps_fts_row_available(self, temp_db, monkeypatch):
        """A conservative-scan failure must abstain from vector
        candidates while the matching FTS row remains available."""
        import sqlite3 as _sqlite3
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-ftsfail", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO episodic_memory "
            "(id, content, source, timestamp, session_id, importance, scope, memory_type) "
            "VALUES ('f1', 'fts anchor phrase zebrafoam', 'sleep_consolidation', "
            "datetime('now'), 's-ftsfail', 0.6, 'global', 'fact')")
        beam.conn.commit()
        # The conservative route is what invokes the legacy scan: a fresh
        # BeamMemory marks the store (pure -> KNN), so the marker must be
        # cleared or the patched scan is never reached.
        _uv = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        beam.conn.execute(f"PRAGMA user_version = {_uv & ~bm._VEC_NORM_BIT}")
        calls = {"n": 0}

        def failing_scan(conn, embedding, k=20, **kwargs):
            calls["n"] += 1
            assert kwargs["eligibility_sql"]
            assert kwargs["eligibility_params"]
            raise _sqlite3.OperationalError("simulated scan failure")

        monkeypatch.setattr(bm, "_vec_legacy_scan_with_blobs", failing_scan)
        monkeypatch.setattr(bm._embeddings, "available", lambda: True)
        monkeypatch.setattr(bm._embeddings, "embed_query",
                            lambda text: np.zeros(64, dtype=np.float32))
        monkeypatch.setattr(bm._embeddings, "embed",
                            lambda texts: [np.zeros(64, dtype=np.float32)])

        res = beam.recall("zebrafoam", top_k=5)
        items = res if isinstance(res, list) else getattr(res, "results", [])
        ids = {r["id"] for r in items}
        assert calls["n"] >= 1, (
            "test never exercised the failing scan path (marker left set?)"
        )
        assert "f1" in ids, "FTS row must remain available when vec scan fails"
        row = next(r for r in items if r["id"] == "f1")
        assert row.get("dense_score", 0.0) == 0.0


class TestBinaryBonusProduction:
    """The binary bonus divisor is asserted through the production helper,
    not local reimplemented arithmetic."""

    def test_bonus_divides_by_live_bit_width(self):
        import numpy as _np
        # 2 bytes, 2 differing bits (1111_0000 xor 1100_0000 = 0011_0000):
        # h_dist=2, live width=16.
        q = _np.array([0b1111_0000, 0b0000_0000], dtype=_np.uint8).tobytes()
        m = _np.array([0b1100_0000, 0b0000_0000], dtype=_np.uint8).tobytes()
        bonus = bm._binary_bonus(q, m)
        expected = 0.08 * (1.0 - _np.tanh((2 / 16) * 3.0))
        assert bonus == pytest.approx(expected, abs=1e-6), (
            "divisor must be the live BIT width (len*8), not the byte count"
        )
        # byte-count divisor would give 2/2=1 -> tanh(3) ~ 0.995 -> bonus ~0
        assert bonus > 0.01, "bonus zeroed: byte-count divisor regression"

    def test_bonus_identical_and_opposite(self):
        import numpy as _np
        v = _np.array([0b1010_1010] * 8, dtype=_np.uint8).tobytes()
        assert bm._binary_bonus(v, v) == pytest.approx(0.08, abs=1e-6)
        opp = bytes(b ^ 0xFF for b in v)
        assert bm._binary_bonus(v, opp) < 0.001

    def test_bonus_edge_inputs(self):
        # identical empty buffers = distance 0 = max bonus (same semantics
        # as identical non-empty vectors)
        assert bm._binary_bonus(b"", b"") == pytest.approx(0.08, abs=1e-6)
        # malformed input must degrade to 0, never raise
        assert bm._binary_bonus(None, b"\x00") == 0.0


class TestPolyphonicVecVoiceBoundary:
    """Unmarked stores scan vec rows without raw-L2 KNN; JSON-only fusion
    while sqlite-vec is usable remains the separate #950 contract."""

    @requires_vec
    def test_unmarked_store_abstains_from_knn(self, beam_db, monkeypatch):
        import json as _json
        import mnemosyne.core.beam as bm
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[32])")
        _uv = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.execute(f"PRAGMA user_version = {_uv & ~bm._VEC_NORM_BIT}")  # unmarked
        rng = np.random.default_rng(5)
        for i in range(1, 20):
            conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
                (i, _json.dumps((rng.standard_normal(32) * 0.1).tolist())))
        conn.commit()

        knn_queries = []
        real_execute = conn.execute

        def spy_execute(sql, *args, **kwargs):
            if isinstance(sql, str) and "MATCH" in sql and "vec_episodes" in sql:
                knn_queries.append(sql)
            return real_execute(sql, *args, **kwargs)

        # JSON-only candidate fusion while sqlite-vec is usable is #950's
        # explicit non-goal. Keep rowids disjoint so this test proves only
        # that an unmarked store avoids raw-L2 KNN; successful vec scanning
        # remains authoritative even when it yields no eligible candidates.
        import json as _json2
        for i in range(1, 6):
            conn.execute(
                "INSERT INTO episodic_memory (rowid, id, content, source, timestamp,"
                " session_id, importance, scope, memory_type)"
                " VALUES (?, ?, ?, 'conversation', datetime('now'), 's-poly',"
                " 0.5, 'session', 'episodic')",
                (100 + i, f"pm{i}", f"poly memory {i} lorem"),
            )
            v = rng.standard_normal(32).astype(np.float32)
            v = v / np.linalg.norm(v)
            conn.execute(
                "INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?, ?)",
                (f"pm{i}", _json2.dumps(v.tolist())))
        conn.commit()
        conn.execute = spy_execute
        try:
            engine = PolyphonicRecallEngine(conn=conn)
            q = rng.standard_normal(32).astype(np.float32)
            res = engine._vector_voice(q)
        finally:
            conn.execute = real_execute
        assert knn_queries == [], (
            f"unmarked store must abstain from raw-L2 KNN: {knn_queries}"
        )
        assert res == [], (
            "#911 must not add JSON-only candidate fusion while sqlite-vec "
            f"is usable; got: {[getattr(r, 'memory_id', None) for r in res]}"
        )

    @requires_vec
    def test_marked_store_runs_knn(self, beam_db):
        import json as _json
        import mnemosyne.core.beam as bm
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[32])")
        bm._mark_vec_store_norm_bit(conn)
        rng = np.random.default_rng(6)
        for i in range(1, 20):
            conn.execute(
                "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
                (i, _json.dumps((rng.standard_normal(32) * 0.1).tolist())))
        conn.commit()
        knn_queries = []
        real_execute = conn.execute

        def spy_execute(sql, *args, **kwargs):
            if isinstance(sql, str) and "MATCH" in sql and "vec_episodes" in sql:
                knn_queries.append(sql)
            return real_execute(sql, *args, **kwargs)

        conn.execute = spy_execute
        try:
            engine = PolyphonicRecallEngine(conn=conn)
            q = rng.standard_normal(32).astype(np.float32)
            engine._vector_voice(q)
        finally:
            conn.execute = real_execute
        assert len(knn_queries) == 1, f"marked store must run KNN: {knn_queries}"


class TestReindexMarkerLifecycle:
    """A failed rebuild must not leave the store on a stale pure verdict;
    the marker is cleared before the rebuild starts."""

    @requires_vec
    def test_failed_rebuild_leaves_store_unmarked(self, beam_db, monkeypatch):
        import mnemosyne.core.beam as bm

        conn = beam_db.conn
        conn.execute("DROP TABLE IF EXISTS vec_episodes")
        conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[32])")
        bm._mark_vec_store_norm_bit(conn)
        assert _regime(conn) == "pure"
        # two embedded episodic rows so the rebuild gets going
        for rid in ("e1", "e2"):
            conn.execute(
                "INSERT INTO episodic_memory (id, content, source, timestamp,"
                " session_id, importance, scope, memory_type)"
                " VALUES (?, ?, 'conversation', datetime('now'), 's-re',"
                " 0.5, 'session', 'episodic')",
                (rid, f"content {rid} lorem ipsum"),
            )
        conn.commit()
        calls = {"n": 0}

        def failing_embed(texts):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("simulated embedder outage mid-rebuild")
            import numpy as np
            return [np.zeros(32, dtype=np.float32) for _ in texts]

        monkeypatch.setattr(bm._embeddings, "available", lambda: True)
        monkeypatch.setattr(bm._embeddings, "embed", failing_embed)
        with __import__("pytest").raises(RuntimeError):
            bm.reindex_vectors(conn)
        uv = conn.execute("PRAGMA user_version").fetchone()[0]
        assert not (uv & bm._VEC_NORM_BIT), (
            "failed rebuild must leave the marker cleared (conservative routing)"
        )
        assert _regime(conn) == "legacy"
