"""CLI error handling regression tests."""

import json
import os
import sqlite3
import subprocess
import sys
import types

from mnemosyne import cli


COMMANDS = [
    (
        ["store", "hello", "cli", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["recall", "hello", "not-an-int"],
        "top_k must be an integer",
    ),
    (
        ["update", "missing-id", "new content", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["import", "missing-file.json"],
        "Import file not found",
    ),
]


def run_cli(args, tmp_path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_import_hindsight_errors_return_nonzero_exit(tmp_path):
    missing_file = tmp_path / "missing-hindsight-export.json"

    result = run_cli(["import-hindsight", str(missing_file)], tmp_path)

    assert result.returncode != 0
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["provider"] == "hindsight"
    assert payload["errors"]
    assert "No such file or directory" in payload["errors"][0]


def test_invalid_cli_input_reports_error_without_traceback(tmp_path):
    for args, expected_error in COMMANDS:
        result = run_cli(args, tmp_path)

        assert result.returncode != 0, args
        assert expected_error in result.stderr, result.stderr
        assert "Traceback" not in result.stderr


def test_import_non_object_json_reports_error_without_traceback(tmp_path):
    for payload in ("[]", '"not an export"'):
        bad_export = tmp_path / "not-an-export.json"
        bad_export.write_text(payload, encoding="utf-8")

        result = run_cli(["import", str(bad_export)], tmp_path)

        assert result.returncode != 0
        assert "Import file must contain a Mnemosyne export object" in result.stderr
        assert "Traceback" not in result.stderr


def test_import_malformed_json_reports_error_without_traceback(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")

    result = run_cli(["import", str(bad_json)], tmp_path)

    assert result.returncode != 0
    assert "Invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_export_reports_actual_exported_memory_counts(tmp_path):
    store_result = run_cli(["store", "exported memory", "cli", "0.7"], tmp_path)
    assert store_result.returncode == 0, store_result.stderr

    export_path = tmp_path / "export.json"
    result = run_cli(["export", str(export_path)], tmp_path)

    assert result.returncode == 0, result.stderr
    # Post-E6: occurred_on / has_source annotations land in the annotations
    # table, not in triples. Export message surfaces both counts so operators
    # see the split clearly.
    assert "Exported 1 working, 0 episodic, 1 legacy, 0 triples, 2 annotations" in result.stdout
    assert "Exported 0 memories" not in result.stdout

    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(exported["working_memory"]) == 1
    assert len(exported["legacy_memories"]) == 1
    assert len(exported["triples"]) == 0
    assert len(exported["annotations"]) == 2


def test_export_manifest_reports_omitted_and_partial_persisted_data(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    assert run_cli(["store", "portable source", "cli", "0.7"], source_dir).returncode == 0

    source_db = source_dir / "mnemosyne-data" / "mnemosyne.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "INSERT INTO facts (fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("omitted-fact", "default", "portable", "has", "omitted fact", 1.0),
        )
        conn.execute(
            "UPDATE working_memory SET pinned = 1, author_id = 'export-owner', "
            "event_date = '2026-04-01', event_date_precision = 'day'"
        )

    export_path = tmp_path / "partial.json"
    export_result = run_cli(["export", str(export_path)], source_dir)
    assert export_result.returncode == 0, export_result.stderr
    assert "WARNING: portable export is partial" in export_result.stdout
    assert "facts (1)" in export_result.stdout
    assert "working_memory missing" in export_result.stdout
    assert "author_id (1)" in export_result.stdout

    manifest = json.loads(export_path.read_text(encoding="utf-8"))["mnemosyne_export"]["completeness"]
    assert manifest["complete"] is False
    assert {surface["table"] for surface in manifest["omitted_surfaces"]} >= {"facts"}
    partial = {surface["section"]: surface for surface in manifest["partial_surfaces"]}
    fields = {field["field"]: field for field in partial["working_memory"]["omitted_fields"]}
    assert fields["author_id"]["affected_rows"] == 1
    # pinned survives the portable round-trip since the event-date/pinned
    # export fix; the manifest no longer reports it as omitted.
    assert "pinned" not in fields

    import_result = run_cli(["import", str(export_path)], target_dir)
    assert import_result.returncode == 0, import_result.stderr
    assert "WARNING: imported supported data only" in import_result.stdout
    target_db = target_dir / "mnemosyne-data" / "mnemosyne.db"
    with sqlite3.connect(target_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
        # pinned=1 + event_date survive the portable round-trip
        # (event-date/pinned export fix); author_id stays omitted.
        assert conn.execute(
            "SELECT pinned, author_id, event_date, event_date_precision FROM working_memory"
        ).fetchone() == (1, None, "2026-04-01", "day")


def test_export_warning_omits_invalid_partial_affected_row_counts(monkeypatch, capsys):
    """Only non-boolean non-negative integer counts belong in CLI output."""
    monkeypatch.setattr(
        cli,
        "_get_memory",
        lambda: types.SimpleNamespace(
            export_to_file=lambda *_args, **_kwargs: {
                "complete": False,
                "omitted_surfaces": [],
                "partial_surfaces": [
                    {
                        "section": "working_memory",
                        "omitted_fields": [
                            {"field": "author_id", "affected_rows": 3},
                            {"field": "negative", "affected_rows": -1},
                            {"field": "boolean", "affected_rows": True},
                            {"field": "malformed", "affected_rows": "3"},
                        ],
                    }
                ],
            }
        ),
    )

    cli.cmd_export(["partial.json"])

    assert (
        "working_memory missing author_id (3), negative, boolean, malformed"
        in capsys.readouterr().out
    )


def test_export_manifest_ignores_default_and_null_omitted_fields(tmp_path):
    assert run_cli(["store", "default-only source", "cli", "0.7"], tmp_path).returncode == 0
    source_db = tmp_path / "mnemosyne-data" / "mnemosyne.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "UPDATE working_memory SET memory_type = 'unknown', channel_id = NULL"
        )
        conn.execute(
            'ALTER TABLE working_memory ADD COLUMN quoted_default TEXT DEFAULT "unknown"'
        )
        conn.execute(
            "ALTER TABLE working_memory ADD COLUMN true_default INTEGER DEFAULT TRUE"
        )
        conn.execute(
            "ALTER TABLE working_memory ADD COLUMN false_default INTEGER DEFAULT FALSE"
        )
        conn.execute(
            "ALTER TABLE working_memory ADD COLUMN blob_default BLOB DEFAULT X'00'"
        )
    export_path = tmp_path / "default-only.json"
    result = run_cli(["export", str(export_path)], tmp_path)
    assert result.returncode == 0, result.stderr

    manifest = json.loads(export_path.read_text(encoding="utf-8"))["mnemosyne_export"]["completeness"]
    partial = {surface["section"]: surface for surface in manifest["partial_surfaces"]}
    assert "working_memory" not in partial


def test_import_reports_actual_imported_memory_counts(tmp_path):
    source_dir = tmp_path / "source"
    import_dir = tmp_path / "imported"

    store_result = run_cli(["store", "imported memory", "cli", "0.7"], source_dir)
    assert store_result.returncode == 0, store_result.stderr

    export_path = tmp_path / "export.json"
    export_result = run_cli(["export", str(export_path)], source_dir)
    assert export_result.returncode == 0, export_result.stderr

    result = run_cli(["import", str(export_path)], import_dir)

    assert result.returncode == 0, result.stderr
    # Post-E6: annotations imported alongside triples (the temporal anchor
    # rows moved to the annotations table).
    # Post-C28: stores now report each stats bucket separately
    # (e.g. "2 new annotations") instead of the bare count, so the
    # imported_renumbered count from an id-collision import isn't
    # silently dropped from the summary.
    # CR-A (round-7): working/episodic summaries flow through
    # _format_store_stats, so counts carry their bucket label
    # ("1 new working") and a quarantined count would surface here too.
    assert "Imported 1 new working, 0 episodic, 1 legacy" in result.stdout
    assert "0 triples" in result.stdout
    assert "2 new annotations" in result.stdout
    assert "Imported 0 memories" not in result.stdout


def test_bank_cli_list_create_delete_uses_configured_data_dir(tmp_path):
    # After [Issue 2] fix: when no mnemosyne.db exists on disk and
    # no bank subdirs exist,  must NOT report a phantom
    #  bank. (The old assertion asserted the buggy behavior.)
    result = run_cli(["bank", "list"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "  - default" not in result.stdout, (
        f"bank list reported a phantom 'default' bank: {result.stdout!r}"
    )
    assert "Traceback" not in result.stderr

    result = run_cli(["bank", "create", "project_a"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Created bank: project_a" in result.stdout
    assert "Traceback" not in result.stderr

    result = run_cli(["bank", "list"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "project_a" in result.stdout

    result = run_cli(["bank", "delete", "project_a"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Deleted bank: project_a" in result.stdout
    assert "Traceback" not in result.stderr


def test_bank_cli_validation_errors_are_user_facing(tmp_path):
    cases = [
        (["bank", "create", "bad/name"], "Invalid bank name", 2),
        (["bank", "create"], "Usage: mnemosyne bank create <name>", 2),
        (["bank", "delete"], "Usage: mnemosyne bank delete <name>", 2),
        (["bank", "nope"], "Unknown bank command: nope", 2),
        (["bank", "delete", "missing_bank"], "Bank not found: missing_bank", 1),
    ]

    for args, expected_error, expected_returncode in cases:
        result = run_cli(args, tmp_path)

        assert result.returncode == expected_returncode, args
        assert result.stdout == ""
        assert expected_error in result.stderr
        assert "Traceback" not in result.stderr
