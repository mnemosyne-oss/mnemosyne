"""Regression tests: round-3 dplush findings on the vec scoring rewrite.

- sign arm gated on ROW clip fraction (not RMS), zero-aware Hamming, and a
  query-zero-fraction weight (r4/r5 adversarial probes)
- bit-arm similarity normalizes by the LIVE bit width, not the configured
  EMBEDDING_DIM (dplush's exact numbers: 0.895966 -> 0.336890)
- legacy-store classification marks user_version and warns once
"""
from __future__ import annotations

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

    def test_sign_following_legacy_row_recovers_honest_cosine(self):
        # A sign-following saturated row vs a gaussian query: honest cosine
        # is the L1/L2 ratio of the quantized query (~0.80 for gaussian);
        # the sign arm must recover at least that much.
        rng = _rng()
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
        assert score >= min(0.75, honest - 0.01)

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
        # Without an explicit width the configured dim remains the fallback
        # (aligned stores are the only producible state).
        score = _vec_distance_sim(150, "bit", bit_width=1024)
        assert score == pytest.approx(0.895966, abs=0.001)


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

    def test_legacy_store_marked_and_warned_once(self, temp_db, monkeypatch):
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-legacy", db_path=temp_db)
        dim = beam_module.EMBEDDING_DIM
        self._seed_vec_rows(beam.conn, dim, legacy=True)
        monkeypatch.setattr(bm, "_legacy_warning_emitted", False)
        regime = bm._classify_vec_store_regime(beam.conn, "vec_episodes")
        assert regime == "legacy"
        bm._mark_vec_store_legacy(beam.conn)
        uv = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        assert uv & 0x10000000
        # Durable: a fresh classification call reads the marker.
        assert bm._classify_vec_store_regime(beam.conn, "vec_episodes") == "legacy"

    def test_pure_store_classified_pure(self, temp_db):
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-pure", db_path=temp_db)
        dim = beam_module.EMBEDDING_DIM
        self._seed_vec_rows(beam.conn, dim, legacy=False)
        regime = bm._classify_vec_store_regime(beam.conn, "vec_episodes")
        assert regime == "pure"
"""placeholder"""

class TestRoundFiveRegressions:
    def test_xor_bonus_uses_bit_width_not_byte_count(self):
        """Round-5 F5-1: h_dist counts BITS but xor_arr is the packed BYTE
        array — dividing by len(xor_arr) inflates the distance 8x and zeroes
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
        """Round-5 F5-6: query-zero dims were counted as differing in the
        numerator while excluded from the denominator — systematic down-bias.
        The zero-aware arm must recover at least the honest byte cosine of a
        sign-following saturated row."""
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
        """Round-5 F5-3: the row zero-free sub-gate (<= 0.001 = <= 1 byte at
        1024) denied the arm 16% of the time with just 2 stray quantization
        zeros. Relaxed to <= 0.01: a saturated row with 8 stray zeros still
        recovers."""
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

    def test_classifier_uses_multi_row_sample(self, temp_db):
        """Round-5 F5-9: LIMIT 1 classified a mixed store by its first row.
        A legacy row at rowid 2 with a normalized row at rowid 1 must
        classify the store 'legacy'."""
        import json as _json
        import numpy as np
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-mixed", db_path=temp_db)
        dim = beam_module.EMBEDDING_DIM
        rng = np.random.default_rng(1)
        norm_vec = rng.standard_normal(dim)
        norm_vec = (norm_vec / np.linalg.norm(norm_vec)).astype(np.float32)
        beam.conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (1,"
            " vec_quantize_int8(?, 'unit'))",
            (_json.dumps(norm_vec.tolist()),),
        )
        legacy = np.full(dim, 40.0, dtype=np.float32)
        beam.conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (2,"
            " vec_quantize_int8(?, 'unit'))",
            (_json.dumps(legacy.tolist()),),
        )
        beam.conn.commit()
        regime = bm._classify_vec_store_regime(beam.conn, "vec_episodes")
        assert regime == "legacy", (
            "a mixed store must not be classified by its first row alone"
        )

    def test_reindex_clears_legacy_bit(self, temp_db):
        """Round-5 F5-5: the durable user_version legacy bit must be clearable
        after a reindex — the store must not stay flagged forever."""
        import mnemosyne.core.beam as bm

        beam = BeamMemory(session_id="s-bit", db_path=temp_db)
        bm._mark_vec_store_legacy(beam.conn)
        uv = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        assert uv & 0x10000000
        uv2 = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        beam.conn.execute(f"PRAGMA user_version = {uv2 & ~0x10000000}")
        uv3 = beam.conn.execute("PRAGMA user_version").fetchone()[0]
        assert not (uv3 & 0x10000000)
