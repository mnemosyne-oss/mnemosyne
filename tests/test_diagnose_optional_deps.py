import sqlite3

import mnemosyne

from mnemosyne import diagnose
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.memory import init_db


def _entry(summary, check):
    return next(item for item in summary["entries"] if item["check"] == check)


def test_diagnose_version_falls_back_to_distribution_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delattr(mnemosyne, "__version__", raising=False)
    monkeypatch.setattr(diagnose.importlib.metadata, "version", lambda name: "9.9.9" if name == "mnemosyne-memory" else "0")

    summary = diagnose.run_diagnostics()

    assert _entry(summary, "mnemosyne_version")["status"] == "9.9.9"


def test_diagnose_treats_ctransformers_as_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

    summary = diagnose.run_diagnostics()
    ctransformers = _entry(summary, "ctransformers")

    assert ctransformers["status"] in {"OK", "OPTIONAL"}
    if ctransformers["status"] == "OPTIONAL":
        assert "local-GGUF fallback" in ctransformers["detail"]
    assert ctransformers["status"] not in {"MISSING", "ERROR"}


def test_memory_orphan_diagnostics_tolerates_missing_optional_tables(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO memories (id, content) VALUES (?, ?)", ("legacy-live", "legacy row"))
    conn.commit()

    result = diagnose._memory_orphan_diagnostics(conn)

    assert result["foreign_keys_enabled"] == 0
    assert result["gists_total"] == 0
    assert result["gists_with_memory_id"] == 0
    assert result["gists_orphan_memory_id"] == 0
    assert result["memory_embeddings_total"] == 0
    assert result["memory_embeddings_orphan_memory_id"] == 0
    assert result["orphan_memory_id_overlap"] == 0


def test_diagnose_reports_memory_orphans_without_mutating_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    db_path = data_dir / "mnemosyne.db"
    init_db(db_path)
    beam = BeamMemory(session_id="diagnose-orphans", db_path=db_path)
    conn = beam.conn

    conn.execute(
        "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
        ("wm-live", "working row", "test"),
    )
    conn.execute(
        "INSERT INTO memories (id, content, source) VALUES (?, ?, ?)",
        ("legacy-live", "legacy row", "test"),
    )
    conn.execute(
        "INSERT INTO episodic_memory (id, content, source) VALUES (?, ?, ?)",
        ("em-live", "episodic row", "test"),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
        ("gist-live", "valid gist", "wm-live"),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
        ("gist-null", "gist with no source", None),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
        ("gist-orphan", "orphan gist", "missing-memory"),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        ("legacy-live", "[1.0]", "test"),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        ("missing-memory", "[0.0]", "test"),
    )
    conn.commit()

    before_counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("working_memory", "memories", "episodic_memory", "gists", "memory_embeddings")
    }

    summary = diagnose.run_diagnostics()

    after_counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("working_memory", "memories", "episodic_memory", "gists", "memory_embeddings")
    }
    assert after_counts == before_counts
    assert _entry(summary, "foreign_keys_enabled")["status"] == "NO"
    assert _entry(summary, "gists_total")["status"] == "3"
    assert _entry(summary, "gists_orphan_memory_id")["status"] == "1"
    assert _entry(summary, "memory_embeddings_total")["status"] == "2"
    assert _entry(summary, "memory_embeddings_orphan_memory_id")["status"] == "1"
    assert _entry(summary, "orphan_memory_id_overlap")["status"] == "1"
