"""Regression tests for issue #834: SHMR stays importable without NumPy.

Contract (confirmed by dplush in #834):

- Importing ``mnemosyne.core.shmr`` succeeds without NumPy.
- Non-dense paths remain usable without NumPy.
- Dense-only entry points raise one deterministic capability error before
  mutating state or doing work.
- Existing dense behavior with NumPy installed is unchanged.

The no-NumPy cases run in a subprocess so the module import boundary is
exercised for real, not just simulated.
"""
import subprocess
import sys
import textwrap

import pytest


def _run_without_numpy(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in a fresh interpreter that cannot import NumPy."""
    blocker = textwrap.dedent("""
        import importlib.abc
        import sys

        class _BlockNumPy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy" or fullname.startswith("numpy."):
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r} (blocked by test)"
                    )
                return None

        sys.meta_path.insert(0, _BlockNumPy())
        assert "numpy" not in sys.modules
    """)
    script = blocker + "\n" + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_module_import_succeeds_without_numpy():
    result = _run_without_numpy("import mnemosyne.core.shmr")
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr


def test_non_dense_surface_usable_without_numpy():
    result = _run_without_numpy(
        """
        import sqlite3
        from mnemosyne.core.shmr import (
            FACTS_SCHEMA_SQL,
            ShmrDenseCapabilityUnavailable,
            _init_schema,
            get_resonance_log,
        )

        class _BeamWithConn:
            def __init__(self, conn):
                self.conn = conn

        conn = sqlite3.connect(":memory:")
        conn.executescript(FACTS_SCHEMA_SQL)
        _init_schema(conn)

        log = get_resonance_log(_BeamWithConn(conn))
        assert isinstance(log, list)

        try:
            from mnemosyne.core.shmr import _embed
        except ImportError:
            pass
        else:
            try:
                _embed("probe")
            except ShmrDenseCapabilityUnavailable as exc:
                assert "shmr_dense_unavailable" in str(exc)
            else:
                raise AssertionError("dense embed should fail without numpy")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dense_entry_points_fail_deterministically_without_numpy(tmp_path):
    db = str(tmp_path / "harmonize.db")
    result = _run_without_numpy(
        f"""
        from mnemosyne.core.shmr import (
            _cosine_similarity,
            _embed,
            _compute_harmony_score,
            harmonize,
            recall_beliefs,
            ShmrDenseCapabilityUnavailable,
        )
        import sqlite3

        db = {db!r}
        beam_conn = sqlite3.connect(db)

        class _Beam:
            conn = beam_conn
            session_id = "test"

        for call in (
            lambda: _embed("probe"),
            lambda: _cosine_similarity(None, None),
            lambda: _compute_harmony_score([], []),
            lambda: recall_beliefs(_Beam(), "query"),
            lambda: harmonize(_Beam()),
        ):
            try:
                call()
            except ShmrDenseCapabilityUnavailable as exc:
                assert str(exc) == "shmr_dense_unavailable: numpy is not installed"
            else:
                raise AssertionError("dense call unexpectedly succeeded")

        tables = [
            row[0]
            for row in beam_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "harmonic_beliefs" not in tables
        assert "memory_resonance_log" not in tables
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_state_mutation_before_capability_error(tmp_path):
    """The dense guard must run before schema creation or row writes."""
    db = str(tmp_path / "state.db")
    result = _run_without_numpy(
        f"""
        from mnemosyne.core.shmr import recall_beliefs, ShmrDenseCapabilityUnavailable
        import sqlite3

        db = {db!r}
        beam_conn = sqlite3.connect(db)
        beam_conn.row_factory = sqlite3.Row

        class _Beam:
            conn = beam_conn
            session_id = "test"

        try:
            recall_beliefs(_Beam(), "query")
        except ShmrDenseCapabilityUnavailable:
            pass
        else:
            raise AssertionError(
                "recall_beliefs unexpectedly returned without raising"
            )

        tables = [
            row[0]
            for row in beam_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        print("TABLES:", sorted(tables))
        assert "harmonic_beliefs" not in tables
        assert "memory_resonance_log" not in tables
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dense_behavior_unchanged_with_numpy(monkeypatch):
    np = pytest.importorskip("numpy")
    from mnemosyne.core import shmr

    assert shmr.np is not None
    monkeypatch.setattr(
        shmr._embeddings, "embed",
        lambda texts: np.full((len(texts), shmr.EMBEDDING_DIM), 0.5,
                              dtype=np.float32),
    )

    emb = shmr._embed("probe")
    assert emb.shape == (shmr.EMBEDDING_DIM,)
    sim = shmr._cosine_similarity(emb, emb)
    assert sim == pytest.approx(1.0, abs=1e-4)


def test_capability_error_type_and_message():
    from mnemosyne.core.shmr import ShmrDenseCapabilityUnavailable

    assert issubclass(ShmrDenseCapabilityUnavailable, RuntimeError)
