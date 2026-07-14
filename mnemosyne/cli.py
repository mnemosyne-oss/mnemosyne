#!/usr/bin/env python3
"""
Mnemosyne CLI - v2
==================
Command-line interface for the Mnemosyne memory system.
All commands use the v2 BEAM architecture (Mnemosyne/BeamMemory).
"""

import os
import sys
import json
from pathlib import Path
from typing import NoReturn

def _default_data_dir() -> str:
    """Resolve the default data directory used by the CLI.

    Keep the standalone CLI aligned with Hermes integrations:
    MNEMOSYNE_DATA_DIR wins, then HERMES_HOME/mnemosyne/data, then the
    historical ~/.hermes/mnemosyne/data fallback.
    """
    if data_dir := os.environ.get("MNEMOSYNE_DATA_DIR"):
        return data_dir
    if hermes_home := os.environ.get("HERMES_HOME"):
        return str(Path(hermes_home).expanduser() / "mnemosyne" / "data")
    return str(Path.home() / ".hermes" / "mnemosyne" / "data")


DATA_DIR = _default_data_dir()


def _fail(message: str, exit_code: int = 2) -> NoReturn:
    """Print a CLI error and exit without a Python traceback."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _usage(message: str, exit_code: int = 2) -> NoReturn:
    """Print command usage for invalid invocations and exit."""
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def _require_value(rest, i, flag, parser):
    """Extract a value for a CLI flag at position i. Returns (parsed_value, i+2).

    Raises SystemExit if the flag is at the end of the argument list (missing
    value). Used by hygiene audit/status/restore parsers.
    """
    if i + 1 >= len(rest) or rest[i + 1].startswith("--"):
        _fail(f"{flag} requires a value")
    return parser(rest[i + 1], flag.lstrip("-")), i + 2


def _parse_float(value: str, name: str) -> float:
    """Parse a float argument or exit with a user-facing CLI error."""
    try:
        return float(value)
    except ValueError:
        _fail(f"{name} must be a number: {value}")


def _parse_int(value: str, name: str) -> int:
    """Parse an integer argument or exit with a user-facing CLI error."""
    try:
        return int(value)
    except ValueError:
        _fail(f"{name} must be an integer: {value}")


def _resolve_bank_name(bank_override: str | None = None) -> str:
    """Resolve the selected memory bank from an override or MNEMOSYNE_BANK."""
    value = bank_override if bank_override is not None else os.environ.get("MNEMOSYNE_BANK", "")
    return value.strip() or "default"


def _resolve_doctor_bank_db_path(bank_name: str) -> Path:
    """Resolve an existing bank DB without constructing ``BankManager``.

    ``BankManager`` eagerly creates ``data_dir/banks`` in its constructor,
    which is correct for mutating CLI commands but violates doctor's strict
    read-only contract.  Keep this small lookup local to the doctor command:
    validation and path inspection are the only filesystem operations.
    """

    from mnemosyne.core.banks import _validate_bank_name

    _validate_bank_name(bank_name)
    data_dir = Path(DATA_DIR)
    if bank_name == "default":
        return data_dir / "mnemosyne.db"
    bank_dir = data_dir / "banks" / bank_name
    if not bank_dir.is_dir():
        _fail(f"Bank '{bank_name}' does not exist", exit_code=1)
    return bank_dir / "mnemosyne.db"


def _doctor_output_target_is_database(output_path: Path, db_path: Path) -> bool:
    """Return whether an existing output target is the inspected DB itself.

    Path resolution protects aliases for absent targets and symlinks, but it
    cannot identify a hardlink. For an existing directory entry, compare
    filesystem identity and fail closed if that comparison is unavailable.
    """
    try:
        output_path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        _fail(f"Unable to safely inspect Doctor output path {output_path}: {error}")

    try:
        return os.path.samefile(output_path, db_path)
    except OSError as error:
        _fail(
            "Unable to determine filesystem identity for existing Doctor "
            f"output path {output_path}: {error}"
        )


def _get_memory():
    """Get a Mnemosyne v2 instance, honoring MNEMOSYNE_BANK."""
    from mnemosyne.core.memory import Mnemosyne
    from mnemosyne.core.banks import BankManager

    bank = _resolve_bank_name()
    bm = BankManager(Path(DATA_DIR))
    try:
        db_path = bm.get_bank_db_path(bank)
    except ValueError as e:
        _fail(str(e))

    return Mnemosyne(db_path=str(db_path), bank=bank)


def _resolve_default_scope() -> str:
    """Resolve the default scope for CLI store calls."""
    raw = os.environ.get("MNEMOSYNE_DEFAULT_SCOPE", "").strip().lower()
    if raw in ("session", "global"):
        return raw
    return "session"


def cmd_store(args):
    """Store a new memory."""
    if not args:
        _usage("Usage: mnemosyne store <content> [source] [importance]")
    content = args[0]
    source = args[1] if len(args) > 1 else "cli"
    importance = _parse_float(args[2], "importance") if len(args) > 2 else 0.5

    mem = _get_memory()
    memory_id = mem.remember(
        content,
        source=source,
        importance=importance,
        scope=_resolve_default_scope(),
        extract_entities=True,
    )
    print(f"Stored: {memory_id}")


def cmd_recall(args):
    """Search memories."""
    if not args:
        _usage("Usage: mnemosyne recall <query> [top_k] [--explain] [--json]")

    explain = False
    json_output = False
    positionals = []
    for arg in args:
        if arg == "--explain":
            explain = True
        elif arg == "--json":
            json_output = True
        else:
            positionals.append(arg)

    if not positionals:
        _usage("Usage: mnemosyne recall <query> [top_k] [--explain] [--json]")
    query = positionals[0]
    top_k = _parse_int(positionals[1], "top_k") if len(positionals) > 1 else 5

    mem = _get_memory()
    payload = mem.recall(query, top_k=top_k, explain=explain)
    if explain:
        results = payload.get("results", [])
    else:
        results = payload

    if json_output:
        if explain:
            print(json.dumps(payload, ensure_ascii=False, default=str))
        else:
            print(json.dumps({"query": query, "top_k": top_k, "results": results}, ensure_ascii=False, default=str))
        return

    print(f"\nResults for: {query}\n")
    for r in results:
        content = r.get("content", "")
        score = r.get("score", 0)
        print(f"  ID: {r.get('id', '?')}")
        print(f"  Content: {content[:150]}{'...' if len(content) > 150 else ''}")
        print(f"  Score: {score:.3f}")
        if r.get("entity_match"):
            print(f"  [entity match]")
        print()


def cmd_update(args):
    """Update an existing memory."""
    if len(args) < 2:
        _usage("Usage: mnemosyne update <memory_id> <new_content> [importance]")
    memory_id = args[0]
    content = args[1]
    importance = _parse_float(args[2], "importance") if len(args) > 2 else None

    mem = _get_memory()
    success = mem.update(memory_id, content=content, importance=importance)
    if success:
        print(f"Updated: {memory_id}")
    else:
        _fail(f"Memory not found: {memory_id}", exit_code=1)


def cmd_delete(args):
    """Delete a memory."""
    if not args:
        _usage("Usage: mnemosyne delete <memory_id>")
    memory_id = args[0]

    mem = _get_memory()
    success = mem.forget(memory_id)
    if success:
        print(f"Deleted: {memory_id}")
    else:
        _fail(f"Memory not found: {memory_id}", exit_code=1)


def cmd_stats(args):
    """Show memory system statistics."""
    mem = _get_memory()
    stats = mem.get_stats()
    beam = stats.get("beam", {})
    wm = beam.get("working_memory", {})
    ep = beam.get("episodic_memory", {})
    triples = beam.get("triples", {})
    print("\nMnemosyne Stats\n")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  Working memory: {wm.get('total', 0)}")
    print(f"  Episodic memory: {ep.get('total', 0)}")
    print(f"  Knowledge triples: {triples.get('total', 0)}")
    if stats.get("banks"):
        print(f"\n  Banks: {', '.join(stats['banks'])}")
    print(f"  DB path: {stats.get('database', 'N/A')}")


def cmd_sleep(args):
    """Run consolidation cycle."""
    mem = _get_memory()
    force = "--force" in args or "-f" in args
    all_sessions = "--all-sessions" in args
    dry_run = "--dry-run" in args
    result = mem.sleep_all_sessions(dry_run=dry_run, force=force) if all_sessions else mem.sleep(dry_run=dry_run, force=force)
    print(f"Consolidation complete: {result}")


def cmd_diagnose(args):
    """Run PII-safe diagnostics. Use --fix to auto-install missing dependencies."""
    fix_mode = "--fix" in args
    dry_run = "--dry-run" in args
    repair_vec_working = "--repair-vec-working" in args
    clean_args = [a for a in args if not a.startswith("--")]

    try:
        from mnemosyne.diagnose import run_diagnostics, auto_fix
        result = run_diagnostics(repair_vec_working=repair_vec_working, dry_run=dry_run)
        print("\nMnemosyne Diagnostics\n")
        print(f"  Checks passed: {result.get('checks_passed', 0)}/{result.get('checks_total', 0)}")
        if result.get("key_findings"):
            print("\n  Key findings:")
            for finding in result["key_findings"]:
                print(f"    - {finding}")
        else:
            print("\n  No issues detected")

        if repair_vec_working:
            print("\n  vec_working repair requested")

        if fix_mode or (dry_run and not repair_vec_working):
            print("\n--- Auto-fix ---")
            fix_result = auto_fix(result.get("entries", []), dry_run=dry_run)
            if fix_result["fixed"]:
                label = "Would fix" if dry_run else "Fixed"
                for item in fix_result["fixed"]:
                    print(f"  ✅ {item}")
            if fix_result["failed"]:
                for item in fix_result["failed"]:
                    print(f"  ❌ {item['label']}: {item['error']}")
            if not fix_result["fixed"] and not fix_result["failed"]:
                print("  Nothing to fix - all dependencies are healthy.")
    except Exception as e:
        print(f"Diagnostic failed: {e}")


def cmd_doctor(args):
    """Render a bounded, read-only health report for one existing database."""

    usage = (
        "Usage: mnemosyne doctor [--db PATH | --bank NAME] "
        "[--format json|markdown|both] [--json-out PATH] [--markdown-out PATH] "
        "[--scan-limit N] [--sample-limit N] [--all] [--include-candidates]"
    )
    db_override = None
    bank_override = None
    output_format = "both"
    json_out = None
    markdown_out = None
    scan_limit = 200
    sample_limit = 20
    scan_all = False
    include_candidates = False
    scan_limit_set = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--db":
            db_override, i = _require_value(args, i, "--db", lambda value, _name: value)
        elif arg == "--bank":
            bank_override, i = _require_value(args, i, "--bank", lambda value, _name: value)
        elif arg == "--format":
            output_format, i = _require_value(args, i, "--format", lambda value, _name: value)
        elif arg == "--json-out":
            json_out, i = _require_value(args, i, "--json-out", lambda value, _name: value)
        elif arg == "--markdown-out":
            markdown_out, i = _require_value(args, i, "--markdown-out", lambda value, _name: value)
        elif arg == "--scan-limit":
            scan_limit, i = _require_value(args, i, "--scan-limit", _parse_int)
            scan_limit_set = True
        elif arg == "--sample-limit":
            sample_limit, i = _require_value(args, i, "--sample-limit", _parse_int)
        elif arg == "--all":
            scan_all = True
            i += 1
        elif arg == "--include-candidates":
            include_candidates = True
            i += 1
        else:
            _usage(f"{usage}\nUnknown doctor option: {arg}")

    if db_override is not None and bank_override is not None:
        _fail("--db and --bank cannot be used together")
    if output_format not in {"json", "markdown", "both"}:
        _fail("--format must be one of: json, markdown, both")
    if not isinstance(scan_limit, int) or scan_limit < 1:
        _fail("--scan-limit must be a positive integer")
    if not isinstance(sample_limit, int) or sample_limit < 0:
        _fail("--sample-limit must be a non-negative integer")
    if scan_limit > 10_000:
        _fail("--scan-limit must not exceed 10000")
    if sample_limit > 100:
        _fail("--sample-limit must not exceed 100")
    if output_format == "json" and markdown_out is not None:
        _fail("--markdown-out requires --format markdown or both")
    if output_format == "markdown" and json_out is not None:
        _fail("--json-out requires --format json or both")

    # ``--all`` remains bounded: it requests the largest safe report scan
    # budget unless the operator selected an explicit, smaller/larger limit.
    if scan_all and not scan_limit_set:
        scan_limit = 10_000

    if db_override is not None:
        db_path = Path(db_override).expanduser()
        bank_name = "default"
    else:
        bank_name = _resolve_bank_name(bank_override)
        try:
            db_path = _resolve_doctor_bank_db_path(bank_name)
        except ValueError as error:
            _fail(str(error))
    if not db_path.is_file():
        _fail(f"Database not found: {db_path}", exit_code=1)

    resolved_db = db_path.resolve()
    resolved_json_path = (
        Path(json_out).expanduser() if json_out else Path.cwd() / "mnemosyne-doctor.json"
    )
    resolved_markdown_path = (
        Path(markdown_out).expanduser() if markdown_out else Path.cwd() / "mnemosyne-doctor.md"
    )
    if output_format == "both":
        output_paths = [resolved_json_path, resolved_markdown_path]
    elif output_format == "json" and json_out:
        output_paths = [resolved_json_path]
    elif output_format == "markdown" and markdown_out:
        output_paths = [resolved_markdown_path]
    else:
        output_paths = []
    if output_format == "both" and resolved_json_path.resolve() == resolved_markdown_path.resolve():
        _fail("JSON and Markdown output paths must be different")
    if any(path.resolve() == resolved_db for path in output_paths):
        _fail("Doctor output path must not overwrite the inspected database")
    if any(_doctor_output_target_is_database(path, db_path) for path in output_paths):
        _fail("Doctor output path must not overwrite the inspected database")

    from mnemosyne.doctor import (
        build_doctor_report,
        doctor_report_payload,
        render_doctor_json,
        render_doctor_markdown,
        write_doctor_artifact_atomically,
        write_doctor_artifacts_atomically,
    )

    try:
        report = build_doctor_report(
            bank_name,
            db_path,
            scan_limit=scan_limit,
            candidate_limit=sample_limit if include_candidates else 0,
        )
        payload = doctor_report_payload(report, include_candidates=include_candidates)
        json_text = render_doctor_json(payload)
        markdown_text = render_doctor_markdown(payload)
        if output_format == "both":
            write_doctor_artifacts_atomically(
                json_path=resolved_json_path,
                json_text=json_text,
                markdown_path=resolved_markdown_path,
                markdown_text=markdown_text,
            )
            print(f"Doctor JSON: {resolved_json_path}")
            print(f"Doctor Markdown: {resolved_markdown_path}")
        elif output_format == "json":
            if json_out:
                write_doctor_artifact_atomically(path=resolved_json_path, text=json_text)
                print(f"Doctor JSON: {resolved_json_path}")
            else:
                print(json_text, end="")
        elif markdown_out:
            write_doctor_artifact_atomically(path=resolved_markdown_path, text=markdown_text)
            print(f"Doctor Markdown: {resolved_markdown_path}")
        else:
            print(markdown_text, end="")
    except (OSError, ValueError) as error:
        _fail(f"Doctor report failed: {error}", exit_code=1)


def cmd_repair(args):
    """Apply one narrow, explicitly selected doctor-gated repair action."""

    usage = (
        "Usage: mnemosyne repair --report REPORT.json --select working_memory:ID "
        "[--select working_memory:ID ...] [--db PATH | --bank NAME] "
        "[--action backfill-vec-working|expire] [--dry-run|--apply] [--backup PATH]"
    )
    db_override = None
    bank_override = None
    report_path = None
    selections = []
    action = "backfill-vec-working"
    apply = False
    dry_run_seen = False
    backup_path = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--db":
            db_override, i = _require_value(args, i, "--db", lambda value, _name: value)
        elif arg == "--bank":
            bank_override, i = _require_value(args, i, "--bank", lambda value, _name: value)
        elif arg == "--report":
            report_path, i = _require_value(args, i, "--report", lambda value, _name: value)
        elif arg == "--select":
            selection, i = _require_value(args, i, "--select", lambda value, _name: value)
            selections.append(selection)
        elif arg == "--action":
            action, i = _require_value(args, i, "--action", lambda value, _name: value)
        elif arg == "--backup":
            backup_path, i = _require_value(args, i, "--backup", lambda value, _name: value)
        elif arg == "--apply":
            apply = True
            i += 1
        elif arg == "--dry-run":
            dry_run_seen = True
            i += 1
        else:
            _usage(f"{usage}\nUnknown repair option")

    if db_override is not None and bank_override is not None:
        _fail("--db and --bank cannot be used together")
    if apply and dry_run_seen:
        _fail("--apply and --dry-run cannot be used together")
    if report_path is None:
        _fail("--report is required")
    if not selections:
        _fail("At least one complete --select working_memory:ID is required")
    if db_override is not None:
        db_path = Path(db_override).expanduser()
        bank_name = "default"
    else:
        bank_name = _resolve_bank_name(bank_override)
        try:
            db_path = _resolve_doctor_bank_db_path(bank_name)
        except ValueError:
            _fail("Invalid bank name")
    if not db_path.is_file():
        _fail("Database not found", exit_code=1)

    from mnemosyne.repair import RepairError, render_repair_json, run_repair

    try:
        result = run_repair(
            db_path=db_path,
            bank_name=bank_name,
            report_path=report_path,
            selections=selections,
            action=action,
            apply=apply,
            backup_path=backup_path,
        )
    except RepairError as error:
        _fail(str(error), exit_code=1)
    print(render_repair_json(result), end="")


def cmd_export(args):
    """Export memories to JSON.

    Supports --include-sync-events to include the sync event log
    alongside memory data (schema v1.2).
    """
    include_sync = "--include-sync-events" in args
    # Filter out flag args to get positional arguments
    pos_args = [a for a in args if not a.startswith("--")]
    output_path = pos_args[0] if pos_args else os.path.join(DATA_DIR, "mnemosyne_export.json")
    mem = _get_memory()
    result = mem.export_to_file(output_path, include_sync_events=include_sync)
    print(
        f"Exported "
        f"{result.get('working_memory_count', 0)} working, "
        f"{result.get('episodic_memory_count', 0)} episodic, "
        f"{result.get('legacy_memories_count', 0)} legacy, "
        f"{result.get('triples_count', 0)} triples, "
        f"{result.get('annotations_count', 0)} annotations"
    )
    sync_count = result.get("sync_events_count", 0)
    if sync_count:
        print(f"  + {sync_count} sync events")
    print(f"  to {output_path}")


def cmd_import(args):
    """Import memories from JSON."""
    if not args:
        _usage("Usage: mnemosyne import <file.json>")
    mem = _get_memory()
    try:
        result = mem.import_from_file(args[0])
    except FileNotFoundError:
        _fail(f"Import file not found: {args[0]}")
    except json.JSONDecodeError as e:
        _fail(f"Invalid JSON in import file {args[0]}: {e}")
    except ValueError as e:
        _fail(str(e))
    beam_stats = result.get("beam", {})

    def _format_store_stats(stats, label):
        """Format an import_all stats dict, exposing every bucket so the
        renumbered count from C28 (rows preserved under a fresh id after
        an id collision) doesn't silently disappear from the CLI summary.

        Returns the label preceded by the count breakdown, e.g.
        '3 new + 2 renumbered triples' or '5 triples'.
        """
        if not isinstance(stats, dict):
            return f"0 {label}"
        new = stats.get("inserted", 0)
        renumbered = stats.get("imported_renumbered", 0)
        skipped = stats.get("skipped", 0)
        overwritten = stats.get("overwritten", 0)
        parts = []
        if new:
            parts.append(f"{new} new")
        if renumbered:
            parts.append(f"{renumbered} renumbered")
        if overwritten:
            parts.append(f"{overwritten} overwritten")
        if skipped:
            parts.append(f"{skipped} skipped")
        if not parts:
            return f"0 {label}"
        return f"{' + '.join(parts)} {label}"

    print(
        f"Imported "
        f"{beam_stats.get('working_memory', {}).get('inserted', 0)} working, "
        f"{beam_stats.get('episodic_memory', {}).get('inserted', 0)} episodic, "
        f"{result.get('legacy', {}).get('inserted', 0)} legacy, "
        f"{_format_store_stats(result.get('triples', {}), 'triples')}, "
        f"{_format_store_stats(result.get('annotations', {}), 'annotations')}"
    )
    # Sync events import stats (silently populated for v1.2 exports)
    se_stats = result.get("sync_events", {})
    if se_stats and se_stats.get("inserted", 0):
        print(
            f"        "
            f"{_format_store_stats(se_stats, 'sync events')}"
        )
    print(f"        from {args[0]}")


def cmd_import_hindsight(args):
    """Import memories from a Hindsight JSON export or API."""
    if not args:
        _usage("Usage: mnemosyne import-hindsight <file.json|base_url> [bank]")
    target = args[0]
    bank = args[1] if len(args) > 1 else "hermes"
    mem = _get_memory()
    from mnemosyne.core.importers.hindsight import import_from_hindsight
    if target.startswith("http://") or target.startswith("https://"):
        result = import_from_hindsight(mem, base_url=target, bank=bank)
    else:
        result = import_from_hindsight(mem, file_path=target, bank=bank)
    print(result.to_json())
    if result.errors:
        raise SystemExit(1)


def cmd_mcp(args):
    """Start MCP server."""
    try:
        from mnemosyne.mcp_server import main as mcp_main
        mcp_main(args)
    except ImportError:
        print("MCP not available. Install with: pip install mnemosyne-memory[mcp]")
        sys.exit(1)


def cmd_sync(args):
    """Sync memories with a remote Mnemosyne instance."""
    import argparse
    parser = argparse.ArgumentParser(prog="mnemosyne sync")
    parser.add_argument("--remote", required=True, help="Remote sync server URL (e.g. http://192.168.1.50:8765)")
    parser.add_argument("--mode", choices=["push", "pull", "bidirectional"], default="bidirectional",
                        help="Sync direction (default: bidirectional)")
    parser.add_argument("--encrypt", help="Encryption key (base64) or path to key file")
    parser.add_argument("--api-key", help="API key for remote server auth")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification (not implemented in stdlib impl)")
    parser.add_argument("--interval", type=float, default=0, help="Sync interval in seconds (repeat mode)")

    # Parse known flags from args; remaining are for internal use
    parsed, _ = parser.parse_known_args(args)

    mem = _get_memory()
    from mnemosyne.core.sync import SyncEngine, SyncEncryption

    encryption = None
    if parsed.encrypt:
        encryption = SyncEncryption.from_config(parsed.encrypt)

    engine = SyncEngine(mem, encryption=encryption)

    if parsed.interval > 0:
        # Repeating sync
        import time
        cycle = 0
        print(f"Starting repeating sync every {parsed.interval}s to {parsed.remote}")
        try:
            while True:
                cycle += 1
                print(f"\n--- Sync cycle {cycle} ---")
                result = engine.sync_with(
                    remote_url=parsed.remote,
                    mode=parsed.mode,
                    api_key=parsed.api_key,
                )
                _print_sync_result(result)
                time.sleep(parsed.interval)
        except KeyboardInterrupt:
            print("\nSync stopped.")
    else:
        result = engine.sync_with(
            remote_url=parsed.remote,
            mode=parsed.mode,
            api_key=parsed.api_key,
        )
        _print_sync_result(result)


def _print_sync_result(result: dict) -> None:
    """Print sync results to console."""
    if result.get("interrupted"):
        print("\n  Interrupted by user.")
    print(f"\nSync to {result.get('remote', '?')}")
    print(f"  Mode: {result.get('mode', '?')}")

    push = result.get("push")
    if push is not None:
        print(f"  Push:")
        print(f"    Accepted:   {push.get('accepted', 0)}")
        print(f"    Duplicates: {push.get('duplicates', 0)}")
        print(f"    Conflicts:  {push.get('conflicts', 0)}")

    pull = result.get("pull")
    if pull is not None:
        print(f"  Pull:")
        print(f"    Events fetched: {pull.get('events_fetched', 0)}")
        print(f"    Accepted:       {pull.get('accepted', 0)}")
        print(f"    Duplicates:     {pull.get('duplicates', 0)}")
        print(f"    Conflicts:      {pull.get('conflicts', 0)}")

    errors = result.get("errors", [])
    if errors:
        print(f"  Errors ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")


def cmd_sync_serve(args):
    """Start the Mnemosyne sync HTTP server."""
    import argparse
    parser = argparse.ArgumentParser(prog="mnemosyne sync-serve")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--api-key", help="API key for bearer-token auth")
    parser.add_argument("--jwt-secret", help="JWT secret for token auth")
    parser.add_argument("--tls-cert", help="TLS certificate file path")
    parser.add_argument("--tls-key", help="TLS key file path")
    parser.add_argument("--device-id", help="Custom device identifier")

    parsed = parser.parse_args(args)

    mem = _get_memory()
    from mnemosyne.core.sync import SyncEngine as _SyncEngine
    from mnemosyne.core.sync_server import run_sync_server as _run_server

    _run_server(
        host=parsed.host,
        port=parsed.port,
        beam_instance=mem,
        device_id=parsed.device_id,
        api_key=parsed.api_key,
        jwt_secret=parsed.jwt_secret,
        tls_cert=parsed.tls_cert,
        tls_key=parsed.tls_key,
    )


def cmd_sync_status(args):
    """Show sync status and statistics."""
    import argparse
    parser = argparse.ArgumentParser(prog="mnemosyne sync-status")
    parser.add_argument("--remote", help="Remote sync server URL to check")
    parser.add_argument("--api-key", help="API key for remote server auth")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    parsed = parser.parse_args(args)

    mem = _get_memory()
    from mnemosyne.core.sync import SyncEngine

    engine = SyncEngine(mem)

    # Log a heartbeat event so we have data to show
    status = engine.get_status(remote_url=parsed.remote if parsed.remote else None)

    if parsed.json:
        print(json.dumps(status, indent=2, default=str))
        return

    print("\nMnemosyne Sync Status\n")
    print(f"  Device ID:        {status.get('device_id', 'N/A')}")
    print(f"  Total events:     {status.get('total_events', 0)}")
    print(f"  Unique devices:   {status.get('device_count', 0)}")
    print(f"  Last event:       {status.get('last_event_time', 'N/A')}")

    if status.get('last_sync'):
        print(f"  Last sync:        {status.get('last_sync')}")

    print(f"  Synced events:    {status.get('synced_events', 0)}")

    op_breakdown = status.get("operation_breakdown", {})
    if op_breakdown:
        print(f"\n  Operations breakdown:")
        for op, cnt in sorted(op_breakdown.items(), key=lambda x: -x[1]):
            print(f"    {op}: {cnt}")

    # Security: show encryption status
    pending = status.get('total_events', 0) - status.get('synced_events', 0)
    if pending:
        print(f"\n  Pending push:     {pending} events")

    if "remote" in status:
        print(f"\n  Remote:           {status.get('remote')}")
        remote_st = status.get("remote_status", {})
        if remote_st:
            pull_info = remote_st.get("pull", {})
            if pull_info:
                print(f"  Remote events:    {pull_info.get('events_fetched', 'N/A')}")
            errors = remote_st.get("errors", [])
            if errors:
                print(f"  Remote errors:")
                for err in errors:
                    print(f"    - {err}")


def cmd_sync_generate_key(args):
    """Generate a random encryption key for sync."""
    from mnemosyne.core.sync import SyncEncryption as _Enc
    key = _Enc.generate_key()
    print(key)
    print(f"\nStore this key securely. It is the only way to decrypt synced payloads.", file=sys.stderr)


def cmd_backup(args):
    """Create a compressed backup of the database."""
    from mnemosyne.dr.recovery import create_backup
    output_dir = Path(args[0]) if args else None
    try:
        result = create_backup(backup_dir=output_dir)
        print(f"Backup created: {result['backup_path']}")
        print(f"  Original size: {result['original_size']:,} bytes")
        print(f"  Backup size:   {result['backup_size']:,} bytes")
        print(f"  Checksum:      {result['db_checksum']}")
    except Exception as e:
        _fail(str(e))


def cmd_restore(args):
    """Restore database from a backup file."""
    if not args:
        _usage("Usage: mnemosyne restore <backup_file.db.gz>")
    from mnemosyne.dr.recovery import restore_backup
    try:
        result = restore_backup(Path(args[0]))
        status = "valid" if result["integrity_check"] else "corrupt"
        print(f"Restored from: {result['backup_used']}")
        print(f"  Database:     {result['database_path']}")
        print(f"  Integrity:    {status}")
        if not result["integrity_check"]:
            _fail("Restored database failed integrity check. Emergency backup preserved.")
    except FileNotFoundError as e:
        _fail(str(e))


def cmd_verify(args):
    """Verify database integrity."""
    from mnemosyne.dr.recovery import verify_integrity
    db_path = Path(args[0]) if args else None
    quick = "--quick" in args
    try:
        if quick:
            import sqlite3
            db = db_path or Path(DATA_DIR) / "mnemosyne.db"
            conn = sqlite3.connect(str(db))
            cursor = conn.cursor()
            cursor.execute("PRAGMA quick_check")
            result = cursor.fetchone()
            conn.close()
            ok = result[0] == "ok"
        else:
            ok = verify_integrity(db_path)
        if ok:
            print("Database integrity check passed")
        else:
            print("Database is corrupt. Run 'mnemosyne restore' from a backup.")
            raise SystemExit(1)
    except Exception as e:
        _fail(str(e))


def cmd_backups_list(args):
    """List available backups."""
    from mnemosyne.dr.recovery import list_backups
    backup_dir = Path(args[0]) if args else None
    backups = list_backups(backup_dir=backup_dir)
    if not backups:
        print("No backups found.")
        print(f"  Backups directory: {backup_dir or Path.home() / '.mnemosyne' / 'backups'}")
        return
    print(f"\nBackups ({len(backups)} total):\n")
    for b in backups:
        meta = b.get("metadata", {})
        print(f"  {b['name']}")
        print(f"    Size:       {b['size']:,} bytes")
        print(f"    Created:    {meta.get('timestamp', b['modified'])}")
        if meta.get("db_checksum"):
            print(f"    Checksum:   {meta['db_checksum']}")
        print()


def cmd_bank(args):
    """Manage memory banks."""
    if not args:
        _usage("Usage: mnemosyne bank <list|create|delete> [name]")

    from mnemosyne.core.banks import BankManager
    bm = BankManager(Path(DATA_DIR))

    subcmd = args[0]
    try:
        if subcmd == "list":
            banks = bm.list_banks()
            # CLI output should reflect on-disk state: if the default
            # bank file does not exist yet, suppress the phantom
            # virtual 'default' entry here while keeping BankManager's
            # core virtual-default contract unchanged.
            default_db = bm.get_bank_db_path("default")
            if not default_db.exists():
                banks = [b for b in banks if b != "default"]
            print("\nMemory Banks:\n")
            for b in banks:
                print(f"  - {b}")
        elif subcmd == "create":
            if len(args) < 2:
                _fail("Usage: mnemosyne bank create <name>")
            bm.create_bank(args[1])
            print(f"Created bank: {args[1]}")
        elif subcmd == "delete":
            if len(args) < 2:
                _fail("Usage: mnemosyne bank delete <name>")
            if bm.delete_bank(args[1]):
                print(f"Deleted bank: {args[1]}")
            else:
                _fail(f"Bank not found: {args[1]}", exit_code=1)
        else:
            _fail(f"Unknown bank command: {subcmd}")
    except ValueError as e:
        _fail(str(e))


def cmd_reindex(args):
    """Rebuild vector indexes from source text with the active embedding model.

    Usage: mnemosyne reindex [--model NAME] [--dry-run] [--yes] [--no-backup]

    Use after changing the embedding model/dimension. Synchronous and blocking —
    re-embeds working + episodic memory, so it can take minutes on a large DB.
    Run it with any provider/gateway stopped.
    """
    dry_run = "--dry-run" in args
    assume_yes = "--yes" in args or "-y" in args
    no_backup = "--no-backup" in args

    # --model has to win before the embedding module is imported: it freezes the
    # model + dimension from the env at import time.
    if "--model" in args:
        try:
            os.environ["MNEMOSYNE_EMBEDDING_MODEL"] = args[args.index("--model") + 1]
        except IndexError:
            _usage("Usage: mnemosyne reindex [--model NAME] [--dry-run] [--yes] [--no-backup]")

    from mnemosyne.core import embeddings as _emb

    mem = _get_memory()

    if dry_run:
        plan = mem.reindex_vectors(dry_run=True)
        print("Reindex plan (dry run -- nothing written):")
        for key in ("model", "dim", "vec_type", "sqlite_vec",
                    "working_memory", "episodic_memory"):
            if key in plan:
                print(f"  {key}: {plan[key]}")
        return

    print(
        f"Reindex will re-embed working + episodic memory with "
        f"'{_emb._DEFAULT_MODEL}' (dim {_emb.EMBEDDING_DIM}) and rebuild the sqlite-vec "
        f"tables. This is synchronous and may take several minutes on a large DB -- "
        f"run it with any provider/gateway stopped."
    )
    if not assume_yes:
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = "n"
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    if not no_backup:
        try:
            from mnemosyne.dr.recovery import create_backup
            backup = create_backup()
            print(f"Backup created: {backup['backup_path']}")
        except Exception as e:
            _fail(f"Backup failed (use --no-backup to skip): {e}")

    import time
    started = time.time()

    def _progress(store, done, total):
        print(f"  {store}: {done}/{total}", flush=True)

    try:
        result = mem.reindex_vectors(progress=_progress)
    except Exception as e:
        _fail(str(e))

    print(f"Reindex complete in {time.time() - started:.1f}s:")
    print(f"  model: {result['model']} (dim {result['dim']})")
    print(f"  working_memory reindexed: {result.get('working_memory_reindexed', 0)}")
    print(f"  episodic_memory reindexed: {result.get('episodic_memory_reindexed', 0)}")
    print(f"  sqlite-vec tables recreated at dim {result['dim']}")


def cmd_hygiene(args):
    """hygiene audit|clean|restore — noise detection and safe cleanup (issue #428)."""
    import sqlite3

    from mnemosyne.core.hygiene import (
        NoiseCandidate,
        audit_noise,
        clean_noise,
        hygiene_status,
        restore_archived,
    )

    if not args or args[0] in ("--help", "-h"):
        print("Usage: mnemosyne hygiene audit|status|clean|restore [options]")
        print("  audit [--limit N] [--offset N] [--all [--batch-size N]] [--min-score F] [--json]")
        print("                                          Scan for noise (dry-run; --batch-size only affects --all)")
        print("  status [--limit N] [--json]             Show PII-safe hygiene status")
        print("  clean --action delete|archive|flag [--confirm] [--dry-run] <candidates.json>")
        print("  restore [--limit N]                     Restore archived memories")
        return

    sub = args[0]
    rest = args[1:]

    if sub == "audit":
        limit = 200
        min_score = 0.3
        offset = 0
        batch_size = 1000
        scan_all = False
        as_json = False
        i = 0
        while i < len(rest):
            if rest[i] == "--limit":
                limit, i = _require_value(rest, i, "--limit", _parse_int)
            elif rest[i] == "--offset":
                offset, i = _require_value(rest, i, "--offset", _parse_int)
            elif rest[i] == "--batch-size":
                batch_size, i = _require_value(rest, i, "--batch-size", _parse_int)
            elif rest[i] == "--min-score":
                min_score, i = _require_value(rest, i, "--min-score", _parse_float)
            elif rest[i] == "--all":
                scan_all = True
                i += 1
            elif rest[i] == "--json":
                as_json = True
                i += 1
            else:
                _fail(f"Unknown hygiene audit option: {rest[i]}")

        db_path = Path(DATA_DIR) / "mnemosyne.db"
        if not db_path.exists():
            _fail(f"Database not found at {db_path}")

        try:
            report = audit_noise(
                db_path=db_path,
                limit=limit,
                min_score=min_score,
                offset=offset,
                scan_all=scan_all,
                batch_size=batch_size,
            )
        except (ValueError, sqlite3.Error) as e:
            _fail(str(e))

        if as_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            mode = "all rows" if scan_all else f"limit={limit} offset={offset}"
            print(f"Audited {report.total_scanned} rows across {report.tables_scanned} ({mode})")
            print(f"Found {len(report.candidates)} noise candidates")
            print(f"  with secrets: {report.summary.get('with_secrets', 0)}")
            for table, count in report.summary.get("by_table", {}).items():
                print(f"  candidates in {table}: {count}")
            for action, count in report.summary.get("by_action", {}).items():
                print(f"  suggested {action}: {count}")
            print()
            for c in report.candidates[:20]:
                print(f"  [{c.noise_score:.2f}] {c.suggested_action:8} {c.table_name}:{c.memory_id[:12]}")
                print(f"         {c.content_preview[:80]}")
                if c.secret_flags:
                    print(f"         SECRETS: {', '.join(c.secret_flags)}")
            if len(report.candidates) > 20:
                print(f"  ... and {len(report.candidates) - 20} more (use --json for full list)")

    elif sub == "status":
        as_json = False
        limit = 200
        i = 0
        while i < len(rest):
            if rest[i] == "--json":
                as_json = True
                i += 1
            elif rest[i] == "--limit":
                limit, i = _require_value(rest, i, "--limit", _parse_int)
            else:
                _fail(f"Unknown hygiene status option: {rest[i]}")
        db_path = Path(DATA_DIR) / "mnemosyne.db"
        if not db_path.exists():
            _fail(f"Database not found at {db_path}")
        try:
            status = hygiene_status(db_path=db_path, limit=limit)
        except (ValueError, sqlite3.Error) as e:
            _fail(str(e))
        if as_json:
            print(json.dumps(status, indent=2))
        else:
            audit_log = status.get("audit_log", {})
            print("Hygiene status")
            print(f"  audit log present: {audit_log.get('present', False)}")
            print(f"  audit log entries: {audit_log.get('total_entries', 0)}")
            if audit_log.get("by_action"):
                for action, count in audit_log["by_action"].items():
                    print(f"  logged {action}: {count}")
            noise = status.get("noise_summary", {})
            print(f"  scanned rows: {noise.get('total_scanned', 0)}")
            print(f"  noise candidates: {noise.get('total_candidates', 0)}")
            print(f"  candidate ratio: {noise.get('candidate_ratio', 0.0)}")
            print(f"  with secrets: {noise.get('with_secrets', 0)}")

    elif sub == "clean":
        action = "keep"
        confirm = False
        dry_run = True
        candidates_file = None

        i = 0
        while i < len(rest):
            if rest[i] == "--action" and i + 1 < len(rest):
                action = rest[i + 1]
                i += 2
            elif rest[i] == "--confirm":
                confirm = True
                dry_run = False
                i += 1
            elif rest[i] == "--dry-run":
                dry_run = True
                i += 1
            else:
                candidates_file = rest[i]
                i += 1

        if not candidates_file:
            _fail("candidates JSON file required: mnemosyne hygiene clean <candidates.json>")

        try:
            with open(candidates_file) as f:
                raw = json.load(f)
        except FileNotFoundError:
            _fail(f"Candidates file not found: {candidates_file}")
        except json.JSONDecodeError as e:
            _fail(f"Invalid JSON in candidates file: {e}")

        candidates = []
        for idx, c in enumerate(raw):
            try:
                candidates.append(NoiseCandidate(
                    memory_id=c["memory_id"],
                    table_name=c["table_name"],
                    content_preview=c.get("content_preview", ""),
                    noise_score=c.get("noise_score", 0.0),
                    noise_reasons=c.get("noise_reasons", []),
                    secret_flags=c.get("secret_flags", []),
                    importance=c.get("importance", 0.5),
                    source=c.get("source", ""),
                    timestamp=c.get("timestamp", ""),
                    suggested_action=c.get("suggested_action", "keep"),
                    content_length=c.get("content_length", 0),
                ))
            except KeyError as e:
                _fail(f"Missing required field {e} in candidate #{idx}")

        db_path = Path(DATA_DIR) / "mnemosyne.db"
        if not db_path.exists():
            _fail(f"Database not found at {db_path}")

        result = clean_noise(
            db_path=db_path,
            candidates=candidates,
            action=action,
            confirm=confirm,
            dry_run=dry_run,
        )

        mode = "DRY RUN" if dry_run else "APPLIED"
        print(f"[{mode}] deleted={result.deleted} archived={result.archived} "
              f"flagged={result.flagged} kept={result.kept}")
        if result.errors:
            print(f"Errors ({len(result.errors)}):")
            for e in result.errors[:10]:
                print(f"  {e}")

    elif sub == "restore":
        restore_limit = 100
        i = 0
        while i < len(rest):
            if rest[i] == "--limit":
                restore_limit, i = _require_value(rest, i, "--limit", _parse_int)
            else:
                _fail(f"Unknown hygiene restore option: {rest[i]}")

        db_path = Path(DATA_DIR) / "mnemosyne.db"
        if not db_path.exists():
            _fail(f"Database not found at {db_path}")

        restored = restore_archived(db_path=db_path, limit=restore_limit)
        print(f"Restored {restored} archived memories.")

    else:
        _fail(f"Unknown hygiene subcommand: {sub}. Use 'audit', 'status', 'clean', or 'restore'.")


def cmd_profile(args):
    """profile list|apply|show|create — gamified config templates."""
    from mnemosyne.core.profiles import list_profiles, get_profile, apply_profile, create_profile

    if not args or args[0] in ("--help", "-h"):
        print("Usage: mnemosyne profile <list|apply|show|create> [options]")
        print("  list                           Show all available profiles")
        print("  apply <name> [--dry-run]       Apply a profile to config.yaml")
        print("  show <name>                    Inspect a profile's settings")
        print("  create <name> [description]    Save current config as a profile")
        return

    sub = args[0]
    rest = args[1:]

    if sub == "list":
        profiles = list_profiles()
        print(f"\n{'Name':15s} {'Description'}")
        print("-" * 70)
        for name, meta in profiles.items():
            print(f"{name:15s} {meta.description}")
            # Rating bars
            ratings_str = "  "
            for label, val in meta.ratings.items():
                bar = "█" * (val // 2) + "░" * (10 - val // 2)
                ratings_str += f"  {label:15s} {bar} {val:2d}/20"
            print(ratings_str)
            print(f"{'':15s} Use case: {meta.use_case}")
            print()

    elif sub == "apply":
        if len(rest) < 1:
            _fail("Profile name required: mnemosyne profile apply <name> [--dry-run]")
        name = rest[0]
        dry_run = "--dry-run" in rest
        config_path_arg = None
        if "--config" in rest:
            idx = rest.index("--config")
            if idx + 1 < len(rest):
                config_path_arg = rest[idx + 1]

        success, errors = apply_profile(name, config_path=config_path_arg, dry_run=dry_run)
        if not success:
            print(f"Failed to apply profile '{name}':", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            raise SystemExit(1)
        mode = "DRY RUN" if dry_run else "APPLIED"
        print(f"[{mode}] Profile '{name}' — {len(get_profile(name))} settings")
        if not dry_run:
            print("Run 'mnemosyne config reload' to apply changes to a running process.")

    elif sub == "show":
        if len(rest) < 1:
            _fail("Profile name required: mnemosyne profile show <name>")
        name = rest[0]
        settings = get_profile(name)
        if settings is None:
            _fail(f"Unknown profile: '{name}'. Run 'mnemosyne profile list' for options.")
        print(f"\nProfile: {name}\n")
        for key in sorted(settings.keys()):
            print(f"  {key:45s} = {settings[key]}")

    elif sub == "create":
        if len(rest) < 1:
            _fail("Profile name required: mnemosyne profile create <name> [description]")
        name = rest[0]
        desc = " ".join(rest[1:]) if len(rest) > 1 else ""
        success = create_profile(name, description=desc)
        if success:
            print(f"Created profile '{name}' from current configuration.")
        else:
            _fail(f"Failed to create profile '{name}' — validation failed.")

    else:
        _fail(f"Unknown profile subcommand: {sub}. Use 'list', 'apply', 'show', or 'create'.")


def cmd_config(args):
    """config reload|get|set|migrate — manage config.yaml."""
    if not args or args[0] in ("--help", "-h"):
        print("Usage: mnemosyne config <reload|get|set|migrate> [options]")
        print("  reload                         Re-read config.yaml (hot-reload)")
        print("  get <key>                      Read a single config value")
        print("  set <key> <value>              Write a value to config.yaml")
        print("  migrate                        Export current env vars to config.yaml")
        return

    sub = args[0]
    rest = args[1:]
    from mnemosyne.core.config import get_config

    if sub == "reload":
        config = get_config()
        changed = config.reload()
        if changed:
            print(f"Reloaded config.yaml — {len(changed)} key(s) changed:")
            for key in sorted(changed):
                print(f"  {key}")
        else:
            print("Config unchanged.")

    elif sub == "get":
        if len(rest) < 1:
            _fail("Key required: mnemosyne config get <key>")
        key = rest[0]
        config = get_config()
        val = config.get(key)
        if val is None:
            print("(not set)")
        else:
            print(f"{key} = {val}")

    elif sub == "set":
        if len(rest) < 2:
            _fail("Key and value required: mnemosyne config set <key> <value>")
        key = rest[0]
        value = rest[1]
        config = get_config()
        config.set(key, value)
        print(f"Set {key} = {value}")
        if key in config.REQUIRES_RESTART:
            print(f"  ⚠ {key} requires restart to take effect.")

    elif sub == "migrate":
        config = get_config()
        migrated = config.migrate_from_env()
        print(f"Migrated {len(migrated)} env vars to config.yaml:")
        for key in migrated:
            print(f"  {key}")

    else:
        _fail(f"Unknown config subcommand: {sub}. Use 'reload', 'get', 'set', or 'migrate'.")


def cmd_migrate(args):
    """Add the 3.11.1 schema tables to an existing bank.

    Bank selection: ``--bank <name>`` flag, else ``$MNEMOSYNE_BANK``,
    else the default bank.
    """
    usage = "Usage: mnemosyne migrate [--bank <name>]"
    bank_override = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--bank":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                _usage(usage)
            bank_override = args[i + 1]
            i += 2
        else:
            _usage(usage)

    bank = _resolve_bank_name(bank_override)

    from mnemosyne.core.banks import BankManager
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    bm = BankManager(Path(DATA_DIR))
    try:
        db_path = bm.get_bank_db_path(bank)
    except ValueError as e:
        _fail(str(e))

    if not db_path.exists():
        _fail(f"Bank '{bank}' does not exist (no db at {db_path})", exit_code=1)

    try:
        report = migrate_311_tables(db_path)
    except Exception as e:
        _fail(f"Migration failed: {e}", exit_code=1)

    print(f"migrate 311: bank={bank} db={db_path}")
    print(f"  tables added: {', '.join(report['tables_added']) or '(none)'}")
    print(
        "  tables already present: "
        f"{', '.join(report['tables_already_present']) or '(none)'}"
    )
    print(f"  indices added: {report['indices_added']}")


COMMANDS = {
    "store": cmd_store,
    "remember": cmd_store,
    "recall": cmd_recall,
    "search": cmd_recall,
    "update": cmd_update,
    "edit": cmd_update,
    "delete": cmd_delete,
    "forget": cmd_delete,
    "stats": cmd_stats,
    "sleep": cmd_sleep,
    "consolidate": cmd_sleep,
    "diagnose": cmd_diagnose,
    "doctor": cmd_doctor,
    "repair": cmd_repair,
    "export": cmd_export,
    "import": cmd_import,
    "import-hindsight": cmd_import_hindsight,
    "mcp": cmd_mcp,
    "bank": cmd_bank,
    "reindex": cmd_reindex,
    "backup": cmd_backup,
    "restore": cmd_restore,
    "verify": cmd_verify,
    "backups": cmd_backups_list,
    "sync": cmd_sync,
    "sync-serve": cmd_sync_serve,
    "sync-server": cmd_sync_serve,
    "sync-status": cmd_sync_status,
    "sync-generate-key": cmd_sync_generate_key,
    "migrate": cmd_migrate,
    "hygiene": cmd_hygiene,
    "profile": cmd_profile,
    "config": cmd_config,
}


def run_cli():
    """Main CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "help"):
        # Keep historical setup behavior for non-doctor CLI entry points while
        # leaving module import and the doctor path free of mkdir side effects.
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Mnemosyne - Local AI Memory System\n")
        print("Usage: mnemosyne <command> [args]\n")
        print("Commands:")
        print("  store <content> [source] [importance]  Store a memory")
        print("  recall <query> [top_k]                 Search memories")
        print("  update <id> <content> [importance]     Update a memory")
        print("  delete <id>                            Delete a memory")
        print("  stats                                  Show statistics")
        print("  sleep                                  Run consolidation")
        print("  diagnose [--fix] [--dry-run] [--repair-vec-working]  Run diagnostics / optional repairs")
        print("  doctor [--db PATH|--bank NAME] [--format json|markdown|both]  Read-only report")
        print("  repair --report REPORT.json --select working_memory:ID [--apply]  Narrow doctor-gated repair")
        print("  export [--include-sync-events] [file.json]    Export memories")
        print("  import <file.json>                     Import memories")
        print("  import-hindsight <file|url> [bank]     Import Hindsight memories")
        print("  bank list|create|delete [name]         Manage memory banks")
        print("  reindex [--model NAME] [--dry-run] [--yes] [--no-backup]")
        print("                                      Rebuild vector indexes with the active model")
        print("  backup [output_dir]                    Create database backup")
        print("  restore <backup.db.gz>                 Restore from backup")
        print("  verify [db_path] [--quick]             Verify database integrity")
        print("  backups [backup_dir]                   List available backups")
        print("  mcp [--transport sse] [--port 8080]    Start MCP server")
        print("  sync --remote <url> [--mode push|pull|bidirectional]")
        print("                                      Sync with remote server")
        print("  sync-serve [--port 8765] [--host 0.0.0.0]")
        print("                                      Start sync server")
        print("  sync-status [--remote <url>] [--json]")
        print("                                      Show sync status")
        print("  sync-generate-key                    Generate encryption key")
        print("  migrate [--bank <name>]                Add 3.11.1 schema tables to an existing bank")
        print("  hygiene audit|clean                  Noise audit and safe cleanup")
        print("  profile list|apply|show|create       Config templates (gamified)")
        print("  config reload|get|set|migrate         Manage config.yaml")
        return

    command = sys.argv[1]
    if command not in {"doctor", "repair"}:
        os.makedirs(DATA_DIR, exist_ok=True)
    handler = COMMANDS.get(command)

    if handler:
        handler(sys.argv[2:])
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'mnemosyne --help' for usage.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    run_cli()
