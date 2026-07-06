from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from hermes_memory_provider import MnemosyneMemoryProvider


def _provider_with_beam(tmp_path: Path) -> tuple[MnemosyneMemoryProvider, Path]:
    db_path = tmp_path / "banks" / "sisyphus" / "mnemosyne.db"
    beam = BeamMemory(session_id="diagnose-test", db_path=db_path)
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider._session_id = "diagnose-test"
    provider._agent_context = "primary"
    provider._profile_isolation_enabled = True
    return provider, db_path


def test_diagnose_reports_active_provider_db_path_and_counts(tmp_path, monkeypatch):
    provider, db_path = _provider_with_beam(tmp_path)
    provider._beam.remember("active bank row", source="fact", importance=0.7)

    legacy_path = tmp_path / "legacy" / "mnemosyne.db"
    monkeypatch.setattr(
        "mnemosyne.diagnose.run_diagnostics",
        lambda **_kwargs: {
            "checks_total": 1,
            "checks_passed": 1,
            "key_findings": [],
            "entries": [
                {
                    "category": "db",
                    "check": "db_path",
                    "status": str(legacy_path),
                    "detail": "",
                }
            ],
        },
    )

    result = json.loads(provider._handle_diagnose({}))

    assert result["active_provider_db_path"] == str(db_path)
    assert result["profile_isolation_enabled"] is True
    assert result["active_provider_counts"]["working_memory"] == 1
    assert result["active_provider_counts"]["episodic_memory"] == 0
    assert result["active_provider_counts"]["facts"] == 0
    assert any(
        str(db_path) in finding
        for finding in result["key_findings"]
    )
    assert result["entries"][0]["status"] == str(legacy_path)


def test_diagnose_without_active_beam_keeps_base_diagnostics(monkeypatch):
    provider = MnemosyneMemoryProvider()
    monkeypatch.setattr(
        "mnemosyne.diagnose.run_diagnostics",
        lambda **_kwargs: {"checks_total": 1, "key_findings": ["base finding"]},
    )

    result = json.loads(provider._handle_diagnose({}))

    assert result == {"checks_total": 1, "key_findings": ["base finding"]}


def test_diagnose_reports_count_error_without_failing(tmp_path, monkeypatch):
    provider, db_path = _provider_with_beam(tmp_path)
    provider._beam.conn.execute("DROP TABLE facts")
    provider._beam.conn.commit()
    monkeypatch.setattr(
        "mnemosyne.diagnose.run_diagnostics",
        lambda **_kwargs: {"checks_total": 1, "key_findings": []},
    )

    result = json.loads(provider._handle_diagnose({}))

    assert result["active_provider_db_path"] == str(db_path)
    assert "active_provider_counts_error" in result
    assert "no such table: facts" in result["active_provider_counts_error"]



def test_diagnose_reports_active_provider_orphans_without_mutating_rows(tmp_path, monkeypatch):
    provider, _db_path = _provider_with_beam(tmp_path)
    assert provider._beam is not None
    conn = provider._beam.conn
    conn.execute(
        "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
        ("wm-live", "working row", "test"),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
        ("gist-orphan", "orphan gist", "missing-memory"),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        ("missing-memory", "[0.0]", "test"),
    )
    conn.commit()
    monkeypatch.setattr(
        "mnemosyne.diagnose.run_diagnostics",
        lambda **_kwargs: {"checks_total": 1, "key_findings": [], "entries": []},
    )
    before_counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("working_memory", "gists", "memory_embeddings")
    }

    result = json.loads(provider._handle_diagnose({}))

    after_counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("working_memory", "gists", "memory_embeddings")
    }
    assert after_counts == before_counts
    assert result["active_provider_orphan_diagnostics"]["foreign_keys_enabled"] == 0
    assert result["active_provider_orphan_diagnostics"]["gists_orphan_memory_id"] == 1
    assert result["active_provider_orphan_diagnostics"]["memory_embeddings_orphan_memory_id"] == 1
    assert result["active_provider_orphan_diagnostics"]["orphan_memory_id_overlap"] == 1


def test_diagnose_can_repair_active_provider_vec_working_gap(tmp_path, monkeypatch):
    provider, _db_path = _provider_with_beam(tmp_path)
    np = pytest.importorskip("numpy")
    np  # keep importorskip side effect explicit
    pytest.importorskip("sqlite_vec")
    if not beam_module._wm_vec_available(provider._beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    now = "2026-01-01T00:00:00"
    memory_id = "active-gap"
    embedding = np.array([1.0] + [0.0] * (beam_module.EMBEDDING_DIM - 1), dtype=np.float32)
    provider._beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "active provider vector gap", "test", now, "diagnose-test", "session", 0.5),
    )
    provider._beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, beam_module._embeddings.serialize(embedding), "test"),
    )
    provider._beam.conn.commit()
    monkeypatch.setattr(
        "mnemosyne.diagnose.run_diagnostics",
        lambda **_kwargs: {"checks_total": 1, "key_findings": [], "entries": []},
    )

    dry_run = json.loads(provider._handle_diagnose({"repair_vec_working": True, "dry_run": True}))
    assert dry_run["active_provider_vec_working_repair"]["status"] == "dry_run"
    assert dry_run["active_provider_vec_working"]["missing_vec_working_rows"] == 1

    repaired = json.loads(provider._handle_diagnose({"repair_vec_working": True}))
    assert repaired["active_provider_vec_working_repair"]["status"] == "repaired"
    assert repaired["active_provider_vec_working_repair"]["inserted"] == 1
    assert repaired["active_provider_vec_working"]["missing_vec_working_rows"] == 0
    assert repaired["active_provider_vec_working"]["status"] == "complete"
