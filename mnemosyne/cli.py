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

# Data directory - respects MNEMOSYNE_DATA_DIR env var
DATA_DIR = os.environ.get("MNEMOSYNE_DATA_DIR") or str(
    Path.home() / ".hermes" / "mnemosyne" / "data"
)
os.makedirs(DATA_DIR, exist_ok=True)


def _fail(message: str, exit_code: int = 2) -> NoReturn:
    """Print a CLI error and exit without a Python traceback."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _usage(message: str, exit_code: int = 2) -> NoReturn:
    """Print command usage for invalid invocations and exit."""
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


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


def _get_memory():
    """Get a Mnemosyne v2 instance."""
    from mnemosyne.core.memory import Mnemosyne
    return Mnemosyne(db_path=os.path.join(DATA_DIR, "mnemosyne.db"))


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
        extract_entities=True,
    )
    print(f"Stored: {memory_id}")


def cmd_recall(args):
    """Search memories."""
    if not args:
        _usage("Usage: mnemosyne recall <query> [top_k]")
    query = args[0]
    top_k = _parse_int(args[1], "top_k") if len(args) > 1 else 5

    mem = _get_memory()
    results = mem.recall(query, top_k=top_k)
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
    result = mem.sleep_all_sessions()
    print(f"Consolidation complete: {result}")


def cmd_diagnose(args):
    """Run PII-safe diagnostics. Use --fix to auto-install missing dependencies."""
    fix_mode = "--fix" in args
    dry_run = "--dry-run" in args
    clean_args = [a for a in args if not a.startswith("--")]

    try:
        from mnemosyne.diagnose import run_diagnostics, auto_fix
        result = run_diagnostics()
        print("\nMnemosyne Diagnostics\n")
        print(f"  Checks passed: {result.get('checks_passed', 0)}/{result.get('checks_total', 0)}")
        if result.get("key_findings"):
            print("\n  Key findings:")
            for finding in result["key_findings"]:
                print(f"    - {finding}")
        else:
            print("\n  No issues detected")

        if fix_mode or dry_run:
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


def cmd_export(args):
    """Export memories to JSON."""
    output_path = args[0] if args else os.path.join(DATA_DIR, "mnemosyne_export.json")
    mem = _get_memory()
    result = mem.export_to_file(output_path)
    print(
        f"Exported "
        f"{result.get('working_memory_count', 0)} working, "
        f"{result.get('episodic_memory_count', 0)} episodic, "
        f"{result.get('legacy_memories_count', 0)} legacy, "
        f"{result.get('triples_count', 0)} triples, "
        f"{result.get('annotations_count', 0)} annotations "
        f"to {output_path}"
    )


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
        f"{_format_store_stats(result.get('annotations', {}), 'annotations')} "
        f"from {args[0]}"
    )


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
    "doctor": cmd_diagnose,
    "export": cmd_export,
    "import": cmd_import,
    "import-hindsight": cmd_import_hindsight,
    "mcp": cmd_mcp,
    "bank": cmd_bank,
    "backup": cmd_backup,
    "restore": cmd_restore,
    "verify": cmd_verify,
    "backups": cmd_backups_list,
}


def run_cli():
    """Main CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "help"):
        print("Mnemosyne - Local AI Memory System\n")
        print("Usage: mnemosyne <command> [args]\n")
        print("Commands:")
        print("  store <content> [source] [importance]  Store a memory")
        print("  recall <query> [top_k]                 Search memories")
        print("  update <id> <content> [importance]     Update a memory")
        print("  delete <id>                            Delete a memory")
        print("  stats                                  Show statistics")
        print("  sleep                                  Run consolidation")
        print("  diagnose [--fix] [--dry-run]           Run diagnostics (--fix auto-installs deps)")
        print("  export [file.json]                     Export memories")
        print("  import <file.json>                     Import memories")
        print("  import-hindsight <file|url> [bank]     Import Hindsight memories")
        print("  bank list|create|delete [name]         Manage memory banks")
        print("  backup [output_dir]                    Create database backup")
        print("  restore <backup.db.gz>                 Restore from backup")
        print("  verify [db_path] [--quick]             Verify database integrity")
        print("  backups [backup_dir]                   List available backups")
        print("  mcp [--transport sse] [--port 8080]    Start MCP server")
        return

    command = sys.argv[1]
    handler = COMMANDS.get(command)

    if handler:
        handler(sys.argv[2:])
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'mnemosyne --help' for usage.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    run_cli()
