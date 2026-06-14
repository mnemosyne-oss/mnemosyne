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
        print("  diagnose [--fix] [--dry-run] [--repair-vec-working]  Run diagnostics / optional repairs")
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
