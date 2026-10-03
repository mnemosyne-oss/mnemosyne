"""
Task 18: Contain trial CLI operational-failure boundary.

Every covered operational CLI failure must exit ``1``, write only
``Error: <static_code>`` to stderr, and never surface the raw exception text,
a private canary, or the temporary data-directory path. Validation/usage
failures keep their existing exit codes and curated messages. ``run_cli()``
re-raises ``SystemExit`` unchanged and contains every other ``Exception`` with
``Error: cli_unexpected_failure``.

Each case patches a real imported seam and then invokes
``mnemosyne.cli.run_cli`` through a public command in a subprocess, following
the subprocess seam pattern from ``tests/test_dream_boundary.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CANARY = "TASK18_PRIVATE_EXCEPTION_CANARY"


def _run_cli_script(
    script_body: str, tmp_path: Path, *, argv_tail: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run an inline python script that invokes ``mnemosyne.cli.run_cli``.

    The script patches an internal seam to raise ``RuntimeError(CANARY)``,
    seeds ``sys.argv`` for the chosen public command, and delegates to the real
    ``run_cli()`` entry point so the dispatcher boundary is exercised exactly.
    """
    env = os.environ.copy()
    data_dir = tmp_path / "mnemosyne-data"
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    env["HOME"] = str(tmp_path / "home")
    env.pop("MNEMOSYNE_BANK", None)
    argv_repr = ", ".join(repr(a) for a in argv_tail)
    script = (
        "import sys\n"
        "import mnemosyne.cli as _cli\n"
        "import os as _os\n"
        "from pathlib import Path as _Path\n"
        f"{script_body}\n"
        f"_cli.DATA_DIR = _os.environ['MNEMOSYNE_DATA_DIR']\n"
        f"sys.argv = ['mnemosyne', {argv_repr}]\n"
        "_cli.run_cli()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _assert_static_failure(result, code, tmp_path):
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert result.stdout == "", output
    assert result.stderr == f"Error: {code}\n", output
    assert "Traceback" not in output, output
    assert CANARY not in output, output
    assert str(tmp_path) not in output, output


def _sqlite_fixture(path: Path) -> None:
    """Create a minimal SQLite database for doctor/verify/migrate fixtures."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE working_memory ("
        "id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,"
        "session_id TEXT, importance REAL, valid_until TEXT, superseded_by TEXT);"
        "CREATE TABLE memory_embeddings (memory_id TEXT PRIMARY KEY, embedding_json TEXT);"
    )
    conn.commit()
    conn.close()


def _ensure_default_bank_db(data_dir_parent: Path) -> Path:
    """Materialize the default bank DB so migrate can reach migrate_311_tables."""
    import sqlite3

    db_path = data_dir_parent / "mnemosyne-data" / "mnemosyne.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE _task18_seed (a)")
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_report_failure_is_static(tmp_path):
    fixture = tmp_path / "fixture.db"
    _sqlite_fixture(fixture)
    script = (
        "import mnemosyne.doctor as _doctor\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_doctor.build_doctor_report = _boom\n"
    )
    result = _run_cli_script(
        script, tmp_path, argv_tail=["doctor", "--db", str(fixture), "--format", "json"]
    )
    _assert_static_failure(result, "doctor_report_failed", tmp_path)


# ---------------------------------------------------------------------------
# backup
# ---------------------------------------------------------------------------


def test_backup_failure_is_static(tmp_path):
    script = (
        "import mnemosyne.dr.recovery as _rec\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_rec.create_backup = _boom\n"
    )
    result = _run_cli_script(script, tmp_path, argv_tail=["backup"])
    _assert_static_failure(result, "backup_failed", tmp_path)


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def test_restore_genuinely_missing_backup_is_static(tmp_path):
    """Unpatched: a genuinely absent path reaches the real FileNotFoundError
    branch in ``restore_backup``. No seam patch, so this exercises the true
    operational failure path, not an injected canary.
    """
    missing = tmp_path / "absent-backup.db.gz"
    assert not missing.exists()
    result = _run_cli_script("", tmp_path, argv_tail=["restore", str(missing)])
    _assert_static_failure(result, "restore_failed", tmp_path)


def test_restore_injected_failure_is_static(tmp_path):
    """Patched: ``restore_backup`` raising RuntimeError(CANARY) is contained
    with the same static contract as the real missing-path branch.
    """
    target = tmp_path / "backup.db.gz"
    script = (
        "import mnemosyne.dr.recovery as _rec\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_rec.restore_backup = _boom\n"
    )
    result = _run_cli_script(script, tmp_path, argv_tail=["restore", str(target)])
    _assert_static_failure(result, "restore_failed", tmp_path)


def test_restore_integrity_failure_does_not_leak_result_paths(tmp_path):
    script = (
        "import mnemosyne.dr.recovery as _rec\n"
        "def _failed_restore(*a, **k):\n"
        "    return {'integrity_check': False, "
        "'backup_used': 'TASK18_BACKUP_PATH', "
        "'database_path': 'TASK18_DATABASE_PATH'}\n"
        "_rec.restore_backup = _failed_restore\n"
    )
    result = _run_cli_script(
        script, tmp_path, argv_tail=["restore", "ignored-backup.db.gz"]
    )
    _assert_static_failure(result, "restore_failed", tmp_path)
    assert "TASK18_BACKUP_PATH" not in result.stdout + result.stderr
    assert "TASK18_DATABASE_PATH" not in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_full_path_failure_is_static(tmp_path):
    fixture = tmp_path / "fixture.db"
    _sqlite_fixture(fixture)
    script = (
        "import mnemosyne.dr.recovery as _rec\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_rec.verify_integrity = _boom\n"
    )
    result = _run_cli_script(script, tmp_path, argv_tail=["verify", str(fixture)])
    _assert_static_failure(result, "verify_failed", tmp_path)


def test_verify_quick_path_corrupt_fixture_is_static(tmp_path):
    fixture = tmp_path / "corrupt.db"
    fixture.write_text("not a database")
    result = _run_cli_script(
        "", tmp_path, argv_tail=["verify", str(fixture), "--quick"]
    )
    _assert_static_failure(result, "verify_failed", tmp_path)


# ---------------------------------------------------------------------------
# hygiene audit / status
# ---------------------------------------------------------------------------


def test_hygiene_audit_missing_database_is_static(tmp_path):
    # Fresh data dir: no mnemosyne.db present (preflight "Database not found").
    result = _run_cli_script("", tmp_path, argv_tail=["hygiene", "audit"])
    _assert_static_failure(result, "hygiene_audit_failed", tmp_path)


def test_hygiene_audit_corrupt_database_is_static(tmp_path):
    """A corrupt on-disk database (garbage bytes) reaches the real
    ``open_readonly_doctor_db`` / sqlite failure path during audit and is
    contained as ``hygiene_audit_failed`` with no path, canary, or traceback.
    """
    data_dir = tmp_path / "mnemosyne-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "mnemosyne.db").write_text("not a database - task18 corrupt fixture")
    result = _run_cli_script("", tmp_path, argv_tail=["hygiene", "audit"])
    _assert_static_failure(result, "hygiene_audit_failed", tmp_path)


def test_hygiene_status_missing_database_is_static(tmp_path):
    result = _run_cli_script("", tmp_path, argv_tail=["hygiene", "status"])
    _assert_static_failure(result, "hygiene_status_failed", tmp_path)


def test_hygiene_audit_oserror_is_command_specific(tmp_path):
    script = (
        "_Path(_os.environ['MNEMOSYNE_DATA_DIR']).mkdir(parents=True)\n"
        "(_Path(_os.environ['MNEMOSYNE_DATA_DIR']) / 'mnemosyne.db').touch()\n"
        "import mnemosyne.doctor as _doctor\n"
        f"def _boom(*a, **k):\n    raise OSError('{CANARY}')\n"
        "_doctor.open_readonly_doctor_db = _boom\n"
    )
    result = _run_cli_script(script, tmp_path, argv_tail=["hygiene", "audit"])
    _assert_static_failure(result, "hygiene_audit_failed", tmp_path)


def test_hygiene_status_oserror_is_command_specific(tmp_path):
    script = (
        "_Path(_os.environ['MNEMOSYNE_DATA_DIR']).mkdir(parents=True)\n"
        "(_Path(_os.environ['MNEMOSYNE_DATA_DIR']) / 'mnemosyne.db').touch()\n"
        "import mnemosyne.doctor as _doctor\n"
        f"def _boom(*a, **k):\n    raise OSError('{CANARY}')\n"
        "_doctor.open_readonly_doctor_db = _boom\n"
    )
    result = _run_cli_script(script, tmp_path, argv_tail=["hygiene", "status"])
    _assert_static_failure(result, "hygiene_status_failed", tmp_path)


def test_hygiene_clean_backend_failure_is_command_specific(tmp_path):
    data_dir = tmp_path / "mnemosyne-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "mnemosyne.db").touch()
    candidates = tmp_path / "candidates.json"
    candidates.write_text("[]")
    script = (
        "import mnemosyne.core.hygiene as _hygiene\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_hygiene.clean_noise = _boom\n"
    )
    result = _run_cli_script(
        script,
        tmp_path,
        argv_tail=["hygiene", "clean", "--dry-run", str(candidates)],
    )
    _assert_static_failure(result, "hygiene_clean_failed", tmp_path)


def test_hygiene_clean_returned_error_is_static_and_redacted(tmp_path):
    data_dir = tmp_path / "mnemosyne-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "mnemosyne.db").touch()
    candidates = tmp_path / "candidates.json"
    candidates.write_text("[]")
    script = (
        "import mnemosyne.core.hygiene as _hygiene\n"
        "from mnemosyne.core.hygiene import CleanResult\n"
        "def _failed(*a, **k):\n"
        "    return CleanResult(errors=['TASK18_BACKEND_ERROR'])\n"
        "_hygiene.clean_noise = _failed\n"
    )
    result = _run_cli_script(
        script,
        tmp_path,
        argv_tail=["hygiene", "clean", "--dry-run", str(candidates)],
    )
    _assert_static_failure(result, "hygiene_clean_failed", tmp_path)
    assert "TASK18_BACKEND_ERROR" not in result.stdout + result.stderr


def test_hygiene_clean_directory_candidate_file_is_static(tmp_path):
    candidates = tmp_path / "candidates-directory"
    candidates.mkdir()
    result = _run_cli_script(
        "", tmp_path, argv_tail=["hygiene", "clean", str(candidates)]
    )
    _assert_static_failure(result, "hygiene_clean_failed", tmp_path)


def test_hygiene_clean_invalid_utf8_candidate_file_is_static(tmp_path):
    candidates = tmp_path / "candidates-invalid.json"
    candidates.write_bytes(b'["\xe9"]')
    result = _run_cli_script(
        "", tmp_path, argv_tail=["hygiene", "clean", str(candidates)]
    )
    _assert_static_failure(result, "hygiene_clean_failed", tmp_path)


def test_hygiene_clean_unexpected_decode_failure_is_command_specific(tmp_path):
    candidates = tmp_path / "candidates.json"
    candidates.write_text("[]", encoding="utf-8")
    script = (
        "import json as _json\n"
        f"def _boom(*a, **k):\n    raise RecursionError('{CANARY}')\n"
        "_json.load = _boom\n"
    )
    result = _run_cli_script(
        script, tmp_path, argv_tail=["hygiene", "clean", str(candidates)]
    )
    _assert_static_failure(result, "hygiene_clean_failed", tmp_path)


def test_hygiene_restore_backend_failure_is_command_specific(tmp_path):
    data_dir = tmp_path / "mnemosyne-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "mnemosyne.db").touch()
    script = (
        "import mnemosyne.core.hygiene as _hygiene\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_hygiene.restore_archived = _boom\n"
    )
    result = _run_cli_script(
        script, tmp_path, argv_tail=["hygiene", "restore"]
    )
    _assert_static_failure(result, "hygiene_restore_failed", tmp_path)


@pytest.mark.parametrize("alias", ["--help", "-h", "help"])
def test_help_alias_with_file_data_dir_is_side_effect_free(tmp_path, alias):
    data_dir = tmp_path / "mnemosyne-data"
    data_dir.write_text("not a directory")
    result = _run_cli_script("", tmp_path, argv_tail=[alias])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("alias", ["--help", "-h", "help"])
def test_help_alias_with_absent_data_dir_does_not_create_it(tmp_path, alias):
    data_dir = tmp_path / "mnemosyne-data"
    assert not data_dir.exists()
    result = _run_cli_script("", tmp_path, argv_tail=[alias])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert not data_dir.exists()


@pytest.mark.parametrize("alias", ["--version", "version"])
def test_version_alias_with_file_data_dir_is_side_effect_free(tmp_path, alias):
    data_dir = tmp_path / "mnemosyne-data"
    data_dir.write_text("not a directory")
    result = _run_cli_script("", tmp_path, argv_tail=[alias])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("Mnemosyne "), result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("alias", ["--version", "version"])
def test_version_alias_with_absent_data_dir_does_not_create_it(tmp_path, alias):
    data_dir = tmp_path / "mnemosyne-data"
    assert not data_dir.exists()
    result = _run_cli_script("", tmp_path, argv_tail=[alias])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("Mnemosyne "), result.stdout
    assert result.stderr == ""
    assert not data_dir.exists()


def test_existing_data_dir_file_is_contained(tmp_path):
    script = "_Path(_os.environ['MNEMOSYNE_DATA_DIR']).write_text('not a directory')"
    result = _run_cli_script(script, tmp_path, argv_tail=["store", "content"])
    _assert_static_failure(result, "cli_unexpected_failure", tmp_path)


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def test_migrate_failure_is_static(tmp_path):
    _ensure_default_bank_db(tmp_path)
    script = (
        "import mnemosyne.migrations.e7_311_tables as _m\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_m.migrate_311_tables = _boom\n"
    )
    result = _run_cli_script(
        script, tmp_path, argv_tail=["migrate", "--bank", "default"]
    )
    _assert_static_failure(result, "migrate_failed", tmp_path)


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


def test_diagnose_failure_is_static(tmp_path):
    script = (
        "import mnemosyne.diagnose as _diag\n"
        "def _boom(*a, **k):\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_diag.run_diagnostics = _boom\n"
    )
    result = _run_cli_script(script, tmp_path, argv_tail=["diagnose"])
    _assert_static_failure(result, "diagnose_failed", tmp_path)


# ---------------------------------------------------------------------------
# cli_unexpected_failure: store (memory construction) + sync secret-file
# ---------------------------------------------------------------------------


def test_store_memory_construction_failure_is_cli_unexpected_failure(tmp_path):
    script = (
        "def _broken_get_memory():\n"
        f"    raise RuntimeError('{CANARY}')\n"
        "_cli._get_memory = _broken_get_memory\n"
    )
    result = _run_cli_script(script, tmp_path, argv_tail=["store", "content"])
    _assert_static_failure(result, "cli_unexpected_failure", tmp_path)


def test_sync_status_secret_file_read_failure_is_cli_unexpected_failure(tmp_path):
    """A secret-file read failure must not leak its path or a traceback.

    The injected ``PermissionError`` from ``_read_secret_file`` must be
    contained as ``cli_unexpected_failure`` rather than a raw traceback.
    Uses the parser's real ``--api-key-file`` flag form.
    """
    key_file = tmp_path / "api-key.txt"
    key_file.write_text("portable-test-secret")
    db_path = tmp_path / "sync.db"
    script = (
        "def _deny_secret_file(*a, **k):\n"
        f"    raise PermissionError('{CANARY}')\n"
        "_cli._read_secret_file = _deny_secret_file\n"
    )
    result = _run_cli_script(
        script,
        tmp_path,
        argv_tail=[
            "sync-status",
            "--db-path",
            str(db_path),
            "--api-key-file",
            str(key_file),
        ],
    )
    _assert_static_failure(result, "cli_unexpected_failure", tmp_path)
    # The key-file path itself must not appear in any output.
    assert str(key_file) not in (result.stdout + result.stderr)
