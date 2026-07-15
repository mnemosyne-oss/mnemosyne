"""CLI and renderer coverage for the read-only doctor report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from mnemosyne import cli, doctor
from mnemosyne.doctor import write_doctor_artifact_atomically, write_doctor_artifacts_atomically


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_SECRET = "doctor-cli-fixture-secret-9f7c"  # nosec - redaction regression fixture
FIXTURE_ORDINARY_MEMORY = "Release coordinator confirms the blue-orchard rehearsal is Friday."


def _create_fixture_db(path: Path) -> int:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE working_memory (
            id TEXT PRIMARY KEY,
            content TEXT,
            source TEXT,
            timestamp TEXT,
            session_id TEXT,
            importance REAL,
            valid_until TEXT,
            superseded_by TEXT
        );
        CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);
        """
    )
    conn.execute(
        "INSERT INTO working_memory (id, content) VALUES (?, ?)",
        ("safe-id", f"password={FIXTURE_SECRET}"),
    )
    conn.execute(
        "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
        ("ordinary-id", FIXTURE_ORDINARY_MEMORY, "heartbeat"),
    )
    conn.execute(
        "INSERT INTO memory_embeddings VALUES (?, ?)", ("safe-id", "[0.1, 0.2]"))
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
    conn.close()
    return count


def _run_cli(args: list[str], tmp_path: Path, *, data_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir or (tmp_path / "data"))
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _tree_snapshot(path: Path) -> tuple[bool, tuple[str, ...]]:
    if not path.exists():
        return False, ()
    return True, tuple(sorted(item.relative_to(path).as_posix() for item in path.rglob("*")))


def test_doctor_cli_writes_safe_both_reports_without_mutating_db(tmp_path):
    db_path = tmp_path / "fixture.db"
    expected_count = _create_fixture_db(db_path)
    before_hash = hashlib.sha256(db_path.read_bytes()).hexdigest()
    json_path = tmp_path / "doctor.json"
    markdown_path = tmp_path / "doctor.md"

    result = _run_cli(
        [
            "doctor",
            "--db", str(db_path),
            "--format", "both",
            "--json-out", str(json_path),
            "--markdown-out", str(markdown_path),
            "--scan-limit", "10",
            "--sample-limit", "2",
            "--include-candidates",
        ],
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text())
    markdown = markdown_path.read_text()
    assert payload["bank_name"] == "default"
    assert payload["execution"] == {"dry_run": True, "query_only": True, "read_only": True}
    candidates = payload["hygiene_summary"]["candidates"]
    assert len(candidates) == 2
    assert all(
        set(candidate) == {
            "candidate_class",
            "noise_score",
            "reason_count",
            "secret_flag_count",
            "suggested_action",
            "table",
        }
        for candidate in candidates
    )
    assert "# Mnemosyne Doctor Report" in markdown
    assert "Read-only / query_only / dry-run" in markdown
    assert "## SQLite" in markdown
    assert "## References" in markdown
    assert "## Vector tiers" in markdown
    assert "## Hygiene" in markdown
    assert "Review the findings and explicit candidates before any future repair." in markdown
    assert FIXTURE_SECRET not in json_path.read_text()
    assert FIXTURE_SECRET not in markdown
    assert FIXTURE_ORDINARY_MEMORY not in json_path.read_text()
    assert "blue-orchard rehearsal" not in json_path.read_text()
    assert FIXTURE_ORDINARY_MEMORY not in markdown
    assert "blue-orchard rehearsal" not in markdown
    assert "safe-id" not in json_path.read_text()
    assert "ordinary-id" not in json_path.read_text()
    assert "safe-id" not in markdown
    assert "ordinary-id" not in markdown
    assert "[0.1, 0.2]" not in json_path.read_text()
    assert "[0.1, 0.2]" not in markdown
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_hash
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == expected_count
    finally:
        conn.close()


def test_doctor_cli_writes_content_free_markdown_candidates_without_json_artifact(tmp_path):
    db_path = tmp_path / "fixture.db"
    expected_count = _create_fixture_db(db_path)
    before_hash = hashlib.sha256(db_path.read_bytes()).hexdigest()
    markdown_path = tmp_path / "doctor.md"
    json_path = tmp_path / "doctor.json"

    result = _run_cli(
        [
            "doctor",
            "--db", str(db_path),
            "--format", "markdown",
            "--markdown-out", str(markdown_path),
            "--scan-limit", "10",
            "--sample-limit", "2",
            "--include-candidates",
        ],
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert markdown_path.exists()
    assert not json_path.exists()
    markdown = markdown_path.read_text()
    assert "# Mnemosyne Doctor Report" in markdown
    assert "Read-only / query_only / dry-run" in markdown
    assert FIXTURE_SECRET not in markdown
    assert FIXTURE_ORDINARY_MEMORY not in markdown
    assert "blue-orchard rehearsal" not in markdown
    assert "safe-id" not in markdown
    assert "ordinary-id" not in markdown
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_hash
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == expected_count
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--db", "one.db", "--bank", "work"], "cannot be used together"),
        (["--db", "one.db", "--scan-limit", "0"], "--scan-limit must be a positive integer"),
        (["--db", "one.db", "--sample-limit", "-1"], "--sample-limit must be a non-negative integer"),
        (["--db", "one.db", "--format", "html"], "--format must be one of"),
        (["--db", "one.db", "--format", "json", "--markdown-out", "doctor.md"], "--markdown-out requires --format markdown or both"),
    ],
)
def test_doctor_cli_rejects_conflicts_and_invalid_limits(tmp_path, args, expected):
    result = _run_cli(["doctor", *args], tmp_path)

    assert result.returncode == 2
    assert expected in result.stderr
    assert "Traceback" not in result.stderr


def test_doctor_cli_resolves_named_bank_without_initializing_memory(tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "banks" / "work" / "mnemosyne.db"
    db_path.parent.mkdir(parents=True)
    _create_fixture_db(db_path)
    json_path = tmp_path / "bank.json"

    result = _run_cli(
        ["doctor", "--bank", "work", "--format", "json", "--json-out", str(json_path)],
        tmp_path,
        data_dir=data_dir,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(json_path.read_text())["bank_name"] == "work"


def test_doctor_cli_db_and_bank_resolution_leave_data_tree_unchanged(tmp_path):
    """Doctor must not materialize DATA_DIR/banks while resolving its target."""

    db_path = tmp_path / "fixture.db"
    _create_fixture_db(db_path)
    fresh_data_dir = tmp_path / "fresh-data"
    before_fresh = _tree_snapshot(fresh_data_dir)
    before_fresh_root = _tree_snapshot(tmp_path)

    by_db = _run_cli(
        ["doctor", "--db", str(db_path), "--format", "json"], tmp_path, data_dir=fresh_data_dir
    )

    assert by_db.returncode == 0, by_db.stderr
    assert _tree_snapshot(fresh_data_dir) == before_fresh
    assert _tree_snapshot(tmp_path) == before_fresh_root

    missing_db = _run_cli(
        ["doctor", "--db", str(fresh_data_dir / "missing.db"), "--format", "json"],
        tmp_path,
        data_dir=fresh_data_dir,
    )

    assert missing_db.returncode == 1
    assert "Database not found" in missing_db.stderr
    assert _tree_snapshot(fresh_data_dir) == before_fresh
    assert _tree_snapshot(tmp_path) == before_fresh_root

    missing_bank = _run_cli(
        ["doctor", "--bank", "missing", "--format", "json"], tmp_path, data_dir=fresh_data_dir
    )

    assert missing_bank.returncode == 1
    assert "does not exist" in missing_bank.stderr
    assert _tree_snapshot(fresh_data_dir) == before_fresh
    assert _tree_snapshot(tmp_path) == before_fresh_root

    existing_data_dir = tmp_path / "existing-data"
    bank_db = existing_data_dir / "banks" / "work" / "mnemosyne.db"
    bank_db.parent.mkdir(parents=True)
    _create_fixture_db(bank_db)
    before_existing = _tree_snapshot(existing_data_dir)
    before_existing_root = _tree_snapshot(tmp_path)

    existing_bank = _run_cli(
        ["doctor", "--bank", "work", "--format", "json"], tmp_path, data_dir=existing_data_dir
    )

    assert existing_bank.returncode == 0, existing_bank.stderr
    assert _tree_snapshot(existing_data_dir) == before_existing
    assert _tree_snapshot(tmp_path) == before_existing_root


def test_doctor_cli_rejects_conflicting_or_database_overwriting_output_paths(tmp_path):
    db_path = tmp_path / "fixture.db"
    _create_fixture_db(db_path)
    shared_path = tmp_path / "shared.report"

    conflict = _run_cli(
        [
            "doctor",
            "--db", str(db_path),
            "--json-out", str(shared_path),
            "--markdown-out", str(shared_path),
        ],
        tmp_path,
    )
    overwrite = _run_cli(
        ["doctor", "--db", str(db_path), "--format", "json", "--json-out", str(db_path)],
        tmp_path,
    )

    assert conflict.returncode == 2
    assert "output paths must be different" in conflict.stderr
    assert overwrite.returncode == 2
    assert "must not overwrite the inspected database" in overwrite.stderr


@pytest.mark.parametrize("output_format", ["json", "markdown", "both"])
def test_doctor_cli_rejects_hardlinked_database_output_targets(tmp_path, output_format):
    db_path = tmp_path / "fixture.db"
    _create_fixture_db(db_path)
    before_bytes = db_path.read_bytes()
    before_hash = hashlib.sha256(before_bytes).hexdigest()

    json_path = tmp_path / "database-hardlink.json"
    markdown_path = tmp_path / "database-hardlink.md"
    output_args = []
    output_paths = []
    if output_format in {"json", "both"}:
        os.link(db_path, json_path)
        output_args.extend(["--json-out", str(json_path)])
        output_paths.append(json_path)
    if output_format in {"markdown", "both"}:
        os.link(db_path, markdown_path)
        output_args.extend(["--markdown-out", str(markdown_path)])
        output_paths.append(markdown_path)
    before_stat = db_path.stat()

    result = _run_cli(
        ["doctor", "--db", str(db_path), "--format", output_format, *output_args], tmp_path
    )

    assert result.returncode == 2
    assert "must not overwrite the inspected database" in result.stderr
    after_stat = db_path.stat()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_hash
    assert (after_stat.st_dev, after_stat.st_ino, after_stat.st_nlink) == (
        before_stat.st_dev,
        before_stat.st_ino,
        before_stat.st_nlink,
    )
    for output_path in output_paths:
        assert os.path.samefile(output_path, db_path)
        assert output_path.read_bytes() == before_bytes
    assert not list(tmp_path.glob(".doctor-*"))


def test_atomic_both_output_write_rolls_back_first_target_if_second_replace_fails(tmp_path, monkeypatch):
    json_path = tmp_path / "doctor.json"
    markdown_path = tmp_path / "doctor.md"
    json_path.write_text('{"previous": true}\n')
    markdown_path.write_text("# previous\n")
    real_replace = os.replace

    def fail_markdown_replace(source, destination):
        if Path(destination) == markdown_path:
            raise OSError("simulated second target failure")
        return real_replace(source, destination)

    monkeypatch.setattr("mnemosyne.doctor.os.replace", fail_markdown_replace)

    with pytest.raises(OSError, match="simulated second target failure"):
        write_doctor_artifacts_atomically(
            json_path=json_path,
            json_text='{"safe": true}\n',
            markdown_path=markdown_path,
            markdown_text="# safe\n",
        )

    assert json_path.read_text() == '{"previous": true}\n'
    assert markdown_path.read_text() == "# previous\n"
    assert not list(tmp_path.glob(".doctor-*"))


def test_atomic_both_output_write_cleans_temps_when_staging_fsync_fails(tmp_path, monkeypatch):
    json_path = tmp_path / "doctor.json"
    markdown_path = tmp_path / "doctor.md"
    json_path.write_text('{"previous": true}\n')
    markdown_path.write_text("# previous\n")

    def fail_fsync(_descriptor):
        raise OSError("simulated staging fsync failure")

    monkeypatch.setattr("mnemosyne.doctor.os.fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated staging fsync failure"):
        write_doctor_artifacts_atomically(
            json_path=json_path,
            json_text='{"safe": true}\n',
            markdown_path=markdown_path,
            markdown_text="# safe\n",
        )

    assert json_path.read_text() == '{"previous": true}\n'
    assert markdown_path.read_text() == "# previous\n"
    assert not list(tmp_path.glob(".doctor-*"))


def test_atomic_both_output_write_cleans_restore_temp_when_rollback_replace_fails(tmp_path, monkeypatch):
    json_path = tmp_path / "doctor.json"
    markdown_path = tmp_path / "doctor.md"
    json_path.write_text('{"previous": true}\n')
    markdown_path.write_text("# previous\n")
    real_replace = os.replace

    def fail_second_and_restore(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == markdown_path:
            raise OSError("simulated second target failure")
        if destination_path == json_path and "-restore-" in source_path.name:
            raise OSError("simulated rollback restore failure")
        return real_replace(source, destination)

    monkeypatch.setattr("mnemosyne.doctor.os.replace", fail_second_and_restore)

    with pytest.raises(OSError, match="simulated second target failure"):
        write_doctor_artifacts_atomically(
            json_path=json_path,
            json_text='{"safe": true}\n',
            markdown_path=markdown_path,
            markdown_text="# safe\n",
        )

    # The failed second target was never replaced; the first cannot be restored
    # only because this test injects failure in its rollback replacement.
    assert json_path.read_text() == '{"safe": true}\n'
    assert markdown_path.read_text() == "# previous\n"
    assert not list(tmp_path.glob(".doctor-*"))


@pytest.mark.parametrize(
    ("output_name", "failure"),
    [("doctor.json", "replace"), ("doctor.md", "fsync")],
)
def test_atomic_single_output_write_preserves_target_and_cleans_temps(tmp_path, monkeypatch, output_name, failure):
    target = tmp_path / output_name
    previous = "previous artifact\n"
    target.write_text(previous)

    if failure == "replace":
        real_replace = os.replace

        def fail_replace(source, destination):
            if Path(destination) == target:
                raise OSError("simulated single-target replace failure")
            return real_replace(source, destination)

        monkeypatch.setattr(doctor.os, "replace", fail_replace)
        error = "simulated single-target replace failure"
    else:

        def fail_fsync(_descriptor):
            raise OSError("simulated single-target fsync failure")

        monkeypatch.setattr(doctor.os, "fsync", fail_fsync)
        error = "simulated single-target fsync failure"

    with pytest.raises(OSError, match=error):
        write_doctor_artifact_atomically(path=target, text="safe artifact\n")

    assert target.read_text() == previous
    assert not list(tmp_path.glob(".doctor-*"))


@pytest.mark.parametrize(
    ("output_format", "output_flag", "output_name", "failure"),
    [
        ("json", "--json-out", "doctor.json", "replace"),
        ("markdown", "--markdown-out", "doctor.md", "fsync"),
    ],
)
def test_doctor_cli_single_output_failure_preserves_target_and_cleans_temps(
    tmp_path, monkeypatch, capsys, output_format, output_flag, output_name, failure
):
    db_path = tmp_path / "fixture.db"
    _create_fixture_db(db_path)
    target = tmp_path / output_name
    previous = "previous artifact\n"
    target.write_text(previous)

    if failure == "replace":
        real_replace = os.replace

        def fail_replace(source, destination):
            if Path(destination) == target:
                raise OSError("simulated CLI single-target replace failure")
            return real_replace(source, destination)

        monkeypatch.setattr(doctor.os, "replace", fail_replace)
        error = "simulated CLI single-target replace failure"
    else:

        def fail_fsync(_descriptor):
            raise OSError("simulated CLI single-target fsync failure")

        monkeypatch.setattr(doctor.os, "fsync", fail_fsync)
        error = "simulated CLI single-target fsync failure"

    with pytest.raises(SystemExit) as raised:
        cli.cmd_doctor(["--db", str(db_path), "--format", output_format, output_flag, str(target)])

    assert raised.value.code == 1
    assert error in capsys.readouterr().err
    assert target.read_text() == previous
    assert not list(tmp_path.glob(".doctor-*"))
