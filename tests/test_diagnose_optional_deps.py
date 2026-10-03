import sqlite3

import pytest

import mnemosyne
from mnemosyne import cli, diagnose
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

    version = _entry(summary, "mnemosyne_version")
    assert version["status"] == "OK"
    assert version["detail"] == "9.9.9"


def test_diagnose_treats_ctransformers_as_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

    summary = diagnose.run_diagnostics()
    ctransformers = _entry(summary, "ctransformers")

    assert ctransformers["status"] in {"OK", "OPTIONAL"}
    if ctransformers["status"] == "OPTIONAL":
        assert "local-GGUF fallback" in ctransformers["detail"]
    assert ctransformers["status"] not in {"MISSING", "ERROR"}


def test_run_diagnostics_counts_lowercase_error_status(tmp_path, monkeypatch):
    """Summary failures include lowercase statuses emitted by diagnostics."""
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        diagnose,
        "collect_runtime_diagnostics",
        lambda: {
            "checks": [
                {"category": "test", "check": "lowercase_error", "status": "error", "detail": ""},
            ]
        },
    )

    summary = diagnose.run_diagnostics()

    assert _entry(summary, "lowercase_error")["status"] == "error"
    assert summary["checks_failed"] == sum(
        str(entry["status"]).upper() in {"MISSING", "NO", "ERROR"}
        for entry in summary["entries"]
    )


def test_diagnose_vector_guidance_uses_valid_sleep_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        diagnose,
        "collect_runtime_diagnostics",
        lambda: {
            "checks": [
                {"category": "deps", "check": "embeddings_available", "status": "YES", "detail": ""},
                {"category": "deps", "check": "sqlite_vec_available", "status": "YES", "detail": ""},
            ]
        },
    )

    summary = diagnose.run_diagnostics()

    finding = next(
        finding for finding in summary["key_findings"] if "episodic vectors=0" in finding
    )
    assert "mnemosyne_sleep" in finding
    assert "BeamMemory.sleep()" in finding
    assert "hermes" + " mnemosyne sleep" not in finding


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
    # Build an intentionally inconsistent legacy fixture so diagnostics can
    # verify orphan reporting without mutating rows. Modern connections may
    # enable foreign-key enforcement, so disable it for this fixture setup.
    conn.execute("PRAGMA foreign_keys = OFF")

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
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
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
    assert _entry(summary, "sqlite_quick_check")["status"] == "OK"
    assert _entry(summary, "gists_total")["status"] == "3"
    assert _entry(summary, "gists_orphan_memory_id")["status"] == "1"
    assert _entry(summary, "memory_embeddings_total")["status"] == "2"
    assert _entry(summary, "memory_embeddings_orphan_memory_id")["status"] == "1"
    assert _entry(summary, "orphan_memory_id_overlap")["status"] == "1"
    assert _entry(summary, "hygiene_noise_scanned")["status"] == "3"
    assert _entry(summary, "hygiene_noise_candidates")["status"] == "0"


@pytest.mark.parametrize(
    "repair_statuses", [["error"], ["skipped"], [None], ["repaired", "error"], []]
)
def test_cli_diagnose_unsuccessful_repair_is_a_process_failure(monkeypatch, repair_statuses):
    entries = []
    for repair_status in repair_statuses:
        entry = {"check": "vec_working_repair_status"}
        if repair_status is not None:
            entry["status"] = repair_status
        entries.append(entry)
    run_diagnostics_calls = []

    def run_diagnostics(**kwargs):
        run_diagnostics_calls.append(kwargs)
        return {
            "checks_passed": 1,
            "checks_total": 2,
            "key_findings": ["vec_working repair did not complete"],
            "entries": entries,
        }

    monkeypatch.setattr(diagnose, "run_diagnostics", run_diagnostics)

    with pytest.raises(SystemExit) as raised:
        cli.cmd_diagnose(["--repair-vec-working"])

    assert raised.value.code == 1
    assert run_diagnostics_calls == [{"repair_vec_working": True, "dry_run": False}]


def test_cli_diagnose_repaired_vec_working_succeeds(monkeypatch):
    run_diagnostics_calls = []

    def run_diagnostics(**kwargs):
        run_diagnostics_calls.append(kwargs)
        return {
            "checks_passed": 2,
            "checks_total": 2,
            "key_findings": [],
            "entries": [{"check": "vec_working_repair_status", "status": "repaired"}],
        }

    monkeypatch.setattr(diagnose, "run_diagnostics", run_diagnostics)

    cli.cmd_diagnose(["--repair-vec-working"])

    assert run_diagnostics_calls == [{"repair_vec_working": True, "dry_run": False}]


def test_cli_diagnose_skipped_repair_dry_run_does_not_exit(monkeypatch):
    run_diagnostics_calls = []

    def run_diagnostics(**kwargs):
        run_diagnostics_calls.append(kwargs)
        return {
            "checks_passed": 1,
            "checks_total": 2,
            "key_findings": ["vec_working repair skipped"],
            "entries": [{"check": "vec_working_repair_status", "status": "skipped"}],
        }

    monkeypatch.setattr(diagnose, "run_diagnostics", run_diagnostics)

    cli.cmd_diagnose(["--repair-vec-working", "--dry-run"])

    assert run_diagnostics_calls == [{"repair_vec_working": True, "dry_run": True}]
