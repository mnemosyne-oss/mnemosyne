#!/usr/bin/env python3
"""
Mnemosyne CLI — v2
==================
Command-line interface for the Mnemosyne memory system.
All commands use the v2 BEAM architecture (Mnemosyne/BeamMemory).
"""

import os
import sys
import json
from pathlib import Path
from typing import NoReturn

# Data directory — respects MNEMOSYNE_DATA_DIR env var
DATA_DIR = os.environ.get(
    "MNEMOSYNE_DATA_DIR",
    str(Path.home() / ".hermes" / "mnemosyne" / "data"),
)
os.makedirs(DATA_DIR, exist_ok=True)


def _fail(message: str, exit_code: int = 2) -> NoReturn:
    """Print a CLI error and exit without a Python traceback."""
    print(f"Error: {message}", file=sys.stderr)
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
        print("Usage: mnemosyne store <content> [source] [importance]")
        return
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
        print("Usage: mnemosyne recall <query> [top_k]")
        return
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
        print("Usage: mnemosyne update <memory_id> <new_content> [importance]")
        return
    memory_id = args[0]
    content = args[1]
    importance = _parse_float(args[2], "importance") if len(args) > 2 else None

    mem = _get_memory()
    success = mem.update(memory_id, content=content, importance=importance)
    if success:
        print(f"Updated: {memory_id}")
    else:
        print(f"Memory not found: {memory_id}")


def cmd_delete(args):
    """Delete a memory."""
    if not args:
        print("Usage: mnemosyne delete <memory_id>")
        return
    memory_id = args[0]

    mem = _get_memory()
    success = mem.forget(memory_id)
    if success:
        print(f"Deleted: {memory_id}")
    else:
        print(f"Memory not found: {memory_id}")


def cmd_stats(args):
    """Show memory system statistics."""
    mem = _get_memory()
    stats = mem.get_stats()
    print("\nMnemosyne Stats\n")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  Working memory: {stats.get('working_count', 0)}")
    print(f"  Episodic memory: {stats.get('episodic_count', 0)}")
    if stats.get("triple_count"):
        print(f"  Knowledge triples: {stats['triple_count']}")
    if stats.get("banks"):
        print(f"\n  Banks: {', '.join(stats['banks'])}")
    print(f"  DB path: {stats.get('db_path', 'N/A')}")


def cmd_sleep(args):
    """Run consolidation cycle."""
    mem = _get_memory()
    # Use sleep_all_sessions to consolidate across ALL sessions, not just "default"
    # The per-session sleep() uses the Mnemosyne instance's session_id which is
    # always "default" when created from CLI — causing the phantom session bug.
    result = mem.sleep_all_sessions()
    print(f"Consolidation complete: {result}")


def cmd_diagnose(args):
    """Run PII-safe diagnostics."""
    try:
        from mnemosyne.diagnose import run_diagnostics
        result = run_diagnostics()
        print("\nMnemosyne Diagnostics\n")
        print(f"  Checks passed: {result.get('checks_passed', 0)}/{result.get('checks_total', 0)}")
        if result.get("key_findings"):
            print("\n  Key findings:")
            for finding in result["key_findings"]:
                print(f"    - {finding}")
        else:
            print("\n  No issues detected")
    except Exception as e:
        print(f"Diagnostic failed: {e}")


def cmd_export(args):
    """Export memories to JSON."""
    output_path = args[0] if args else os.path.join(DATA_DIR, "mnemosyne_export.json")
    mem = _get_memory()
    result = mem.export_to_file(output_path)
    print(f"Exported {result.get('count', 0)} memories to {output_path}")


def cmd_import(args):
    """Import memories from JSON."""
    if not args:
        print("Usage: mnemosyne import <file.json>")
        return
    mem = _get_memory()
    try:
        result = mem.import_from_file(args[0])
    except FileNotFoundError:
        _fail(f"Import file not found: {args[0]}")
    except json.JSONDecodeError as e:
        _fail(f"Invalid JSON in import file {args[0]}: {e}")
    except ValueError as e:
        _fail(str(e))
    print(f"Imported {result.get('count', 0)} memories from {args[0]}")


def cmd_import_hindsight(args):
    """Import memories from a Hindsight JSON export or API."""
    if not args:
        print("Usage: mnemosyne import-hindsight <file.json|base_url> [bank]")
        return
    target = args[0]
    bank = args[1] if len(args) > 1 else "hermes"
    mem = _get_memory()
    from mnemosyne.core.importers.hindsight import import_from_hindsight
    if target.startswith("http://") or target.startswith("https://"):
        result = import_from_hindsight(mem, base_url=target, bank=bank)
    else:
        result = import_from_hindsight(mem, file_path=target, bank=bank)
    print(result.to_json())


def cmd_mcp(args):
    """Start MCP server."""
    try:
        from mnemosyne.mcp_server import main as mcp_main
        mcp_main()
    except ImportError:
        print("MCP not available. Install with: pip install mnemosyne-memory[mcp]")
        sys.exit(1)


def cmd_bank(args):
    """Manage memory banks."""
    if not args:
        print("Usage: mnemosyne bank <list|create|delete> [name]")
        return

    from mnemosyne.core.banks import BankManager
    bm = BankManager(db_path=os.path.join(DATA_DIR, "mnemosyne.db"))

    subcmd = args[0]
    if subcmd == "list":
        banks = bm.list_banks()
        print("\nMemory Banks:\n")
        for b in banks:
            print(f"  - {b}")
    elif subcmd == "create":
        if len(args) < 2:
            print("Usage: mnemosyne bank create <name>")
            return
        bm.create_bank(args[1])
        print(f"Created bank: {args[1]}")
    elif subcmd == "delete":
        if len(args) < 2:
            print("Usage: mnemosyne bank delete <name>")
            return
        bm.delete_bank(args[1])
        print(f"Deleted bank: {args[1]}")
    else:
        print(f"Unknown bank command: {subcmd}")


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
    "export": cmd_export,
    "import": cmd_import,
    "import-hindsight": cmd_import_hindsight,
    "mcp": cmd_mcp,
    "bank": cmd_bank,
}


def run_cli():
    """Main CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "help"):
        print("Mnemosyne — Local AI Memory System\n")
        print("Usage: mnemosyne <command> [args]\n")
        print("Commands:")
        print("  store <content> [source] [importance]  Store a memory")
        print("  recall <query> [top_k]                 Search memories")
        print("  update <id> <content> [importance]     Update a memory")
        print("  delete <id>                            Delete a memory")
        print("  stats                                  Show statistics")
        print("  sleep                                  Run consolidation")
        print("  diagnose                               Run diagnostics")
        print("  export [file.json]                     Export memories")
        print("  import <file.json>                     Import memories")
        print("  import-hindsight <file|url> [bank]      Import Hindsight memories")
        print("  bank list|create|delete [name]         Manage memory banks")
        print("  mcp [--transport sse] [--port 8080]    Start MCP server")
        return

    command = sys.argv[1]
    handler = COMMANDS.get(command)

    if handler:
        handler(sys.argv[2:])
    else:
        print(f"Unknown command: {command}")
        print("Run 'mnemosyne --help' for usage.")


if __name__ == "__main__":
    run_cli()
