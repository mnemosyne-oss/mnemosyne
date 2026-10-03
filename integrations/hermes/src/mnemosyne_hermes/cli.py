"""CLI commands for Mnemosyne memory provider.

Available via: hermes mnemosyne <subcommand>
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sqlite3
from pathlib import Path

_BANK_HELP = (
    "Mnemosyne bank to operate on. Defaults to the active Hermes profile's bank "
    "when profile_isolation is enabled, otherwise the shared default bank."
)

# Persistent table columns read unconditionally by Mnemosyne's JSON exporter:
# Mnemosyne.export_to_file(), BeamMemory.export_to_dict(), and the TripleStore,
# AnnotationStore, and CanonicalStore export_all() methods. Checking this
# contract through SQLite's read-only URI gives selected banks a fail-closed
# boundary before constructing Beam/Mnemosyne (whose schema setup is intentionally
# write-capable). It deliberately excludes optional sync and vector storage.
_EXPORT_REQUIRED_COLUMNS = {
    "working_memory": frozenset({
        "id", "content", "source", "timestamp", "session_id", "importance",
        "metadata_json", "valid_until", "superseded_by", "scope", "recall_count",
        "last_recalled", "created_at", "veracity", "consolidated_at",
        "consolidation_claimed_at",
    }),
    "episodic_memory": frozenset({
        "id", "content", "source", "timestamp", "session_id", "importance",
        "metadata_json", "summary_of", "valid_until", "superseded_by", "scope",
        "recall_count", "last_recalled", "created_at",
    }),
    "scratchpad": frozenset({"id", "content", "session_id", "created_at", "updated_at"}),
    "consolidation_log": frozenset({
        "id", "session_id", "items_consolidated", "summary_preview", "created_at",
    }),
    "memories": frozenset({
        "id", "content", "source", "timestamp", "session_id", "importance",
        "metadata_json", "created_at",
    }),
    "memory_embeddings": frozenset({"memory_id", "embedding_json", "model", "created_at"}),
    "triples": frozenset({
        "id", "subject", "predicate", "object", "valid_from", "valid_until",
        "source", "confidence", "created_at",
    }),
    "annotations": frozenset({
        "id", "memory_id", "kind", "value", "source", "confidence", "created_at",
    }),
    "canonical_facts": frozenset({
        "id", "owner_id", "category", "name", "body", "source", "confidence",
        "version", "valid_from", "valid_until", "created_at",
    }),
}
_EXPORT_REQUIRED_TABLES = frozenset(_EXPORT_REQUIRED_COLUMNS)
# Distinct from ``None``: ``None`` is the valid shared/default-bank selection.
_BANK_RESOLUTION_FAILED = object()


def _safe_completeness_label(value: object, fallback: str) -> str:
    """Return a bounded terminal-safe manifest label, never raw import text."""
    if not isinstance(value, str):
        return fallback
    label = "".join(
        char for char in value if char.isascii() and (char.isalnum() or char in "_.-")
    )[:80]
    return label or fallback


def _completeness_details(result: object) -> str:
    """Format bounded, terminal-safe portable-export omission details."""
    if not isinstance(result, dict):
        return ""
    details = []
    omitted = result.get("omitted_surfaces", [])
    if isinstance(omitted, list):
        for surface in omitted[:20]:
            if not isinstance(surface, dict):
                continue
            table = _safe_completeness_label(surface.get("table"), "omitted-surface")
            row_count = surface.get("row_count")
            if isinstance(row_count, int) and not isinstance(row_count, bool) and row_count >= 0:
                details.append(f"{table} ({row_count})")
            else:
                details.append(table)
    partial = result.get("partial_surfaces", [])
    if isinstance(partial, list):
        for surface in partial[:20]:
            if not isinstance(surface, dict):
                continue
            section = _safe_completeness_label(surface.get("section"), "partial-surface")
            fields = surface.get("omitted_fields", [])
            field_names = []
            if isinstance(fields, list):
                for field in fields[:20]:
                    if isinstance(field, dict):
                        field_name = _safe_completeness_label(field.get("field"), "field")
                        affected_rows = field.get("affected_rows")
                        if (
                            isinstance(affected_rows, int)
                            and not isinstance(affected_rows, bool)
                            and affected_rows >= 0
                        ):
                            field_names.append(f"{field_name} ({affected_rows})")
                        else:
                            field_names.append(field_name)
            details.append(
                f"{section} missing {', '.join(field_names)}" if field_names else section
            )
    return "; ".join(details)


def _print_completeness_warning(result: object, *, imported: bool) -> None:
    """Print portable export completeness state without echoing unsafe manifest text."""
    if not isinstance(result, dict):
        return
    complete_key = "restore_complete" if imported else "complete"
    complete = result.get(complete_key)
    if complete is False:
        details = _completeness_details(result)
        if imported:
            omitted = result.get("omitted_surfaces", [])
            partial = result.get("partial_surfaces", [])
            omitted_count = len(omitted) if isinstance(omitted, list) else 0
            partial_count = len(partial) if isinstance(partial, list) else 0
            warning = (
                "  WARNING: imported supported data only; source export reported "
                f"{omitted_count} omitted and {partial_count} partial surface(s)"
            )
        else:
            warning = "  WARNING: portable export is partial"
        print(f"{warning}; {details}" if details else warning)
    elif imported and complete is None:
        print("  NOTE: source export predates completeness reporting")


def _export_schema_is_complete_read_only(db_path: Path) -> bool:
    """Check selected export tables and read columns without opening it writable."""
    db_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        for table, required_columns in _EXPORT_REQUIRED_COLUMNS.items():
            # Table names are from the local fixed contract, not user input.
            columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
            if not required_columns <= columns:
                return False
    finally:
        conn.close()
    return True


def _distribution_version(distribution: str) -> str:
    """Return an installed distribution version without importing package globals."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def register_cli(subparser):
    """Register CLI subcommands for ``hermes mnemosyne``."""
    mn_cmds = subparser.add_subparsers(dest="mnemosyne_cmd")

    stats_cmd = mn_cmds.add_parser("stats", help="Show memory statistics")
    stats_cmd.add_argument("--global", "-g", action="store_true", help="Show global stats across all sessions")
    stats_cmd.add_argument("--bank", type=str, help=_BANK_HELP)

    sleep_cmd = mn_cmds.add_parser("sleep", help="Run consolidation cycle")
    sleep_cmd.add_argument("--all-sessions", action="store_true", help="Consolidate eligible old working memories across all sessions")
    sleep_cmd.add_argument("--dry-run", action="store_true", help="Report what would be consolidated without writing changes")
    sleep_cmd.add_argument("--bank", type=str, help=_BANK_HELP)
    mn_cmds.add_parser("version", help="Show Mnemosyne version")

    inspect_cmd = mn_cmds.add_parser("inspect", help="Search memories")
    inspect_cmd.add_argument("query", nargs="?", default="", help="Search query")
    inspect_cmd.add_argument("--limit", type=int, default=10, help="Max results")
    inspect_cmd.add_argument("--bank", type=str, help=_BANK_HELP)

    mn_cmds.add_parser("clear", help="Clear scratchpad")

    doctor_cmd = mn_cmds.add_parser("doctor", help="Run diagnostics and auto-fix missing dependencies")
    doctor_cmd.add_argument("--dry-run", action="store_true", help="Show what would be fixed without installing")
    doctor_cmd.add_argument("--no-fix", action="store_true", help="Diagnose only, do not fix")
    doctor_cmd.add_argument(
        "--bank", type=str,
        help="Mnemosyne bank to diagnose. Defaults to the active Hermes profile's bank when profile_isolation is enabled, otherwise the shared default bank.",
    )

    export_cmd = mn_cmds.add_parser("export", help="Export all memories to a JSON file")
    export_cmd.add_argument("--output", "-o", type=str, required=True, help="Output JSON file path")
    export_cmd.add_argument("--bank", type=str, help=_BANK_HELP)

    import_cmd = mn_cmds.add_parser("import", help="Import memories from a JSON file or another provider")
    import_cmd.add_argument("--input", "-i", type=str, help="Input JSON file path (for file imports)")
    import_cmd.add_argument("--file", type=str, help="Provider file input, e.g. Hindsight JSON export")
    import_cmd.add_argument("--force", action="store_true", help="Overwrite existing records (file import)")
    import_cmd.add_argument("--from", dest="from_provider", type=str, help="Provider to import from (e.g., 'mem0')")
    import_cmd.add_argument("--api-key", type=str, help="Provider API key (or set env var)")
    import_cmd.add_argument("--user-id", type=str, help="Filter by user ID (provider-specific)")
    import_cmd.add_argument("--agent-id", type=str, help="Filter by agent ID (provider-specific)")
    import_cmd.add_argument("--base-url", type=str, help="Provider base URL (for self-hosted)")
    import_cmd.add_argument("--bank", type=str, help="Provider memory bank, e.g. Hindsight bank")
    import_cmd.add_argument("--dry-run", action="store_true", help="Validate but don't import")
    import_cmd.add_argument("--session-id", type=str, help="Override session for imported memories")
    import_cmd.add_argument("--channel-id", type=str, help="Channel for imported memories")
    import_cmd.add_argument("--list-providers", action="store_true", help="List supported import providers")
    import_cmd.add_argument("--generate-script", action="store_true", help="Generate a migration script for the provider")
    import_cmd.add_argument("--agentic", action="store_true", help="Generate agent migration instructions (prompt to give your AI agent)")
    import_cmd.add_argument("--output-script", type=str, help="Save generated script to file")
    import_cmd.add_argument("--db-path", type=str, help="Holographic memory store path (default: ~/.hermes/memory_store.db)")
    import_cmd.add_argument("--min-trust", type=float, default=0.0, help="Minimum trust score threshold 0.0-1.0 (holographic only)")

    subparser.set_defaults(func=mnemosyne_command)


def _profile_isolation_enabled(hermes_home: str) -> bool:
    """True when ``memory.mnemosyne.profile_isolation`` is set in config.yaml."""
    try:
        import yaml
        with open(os.path.join(hermes_home, "config.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
        val = (cfg.get("memory", {}) or {}).get("mnemosyne", {}).get("profile_isolation", False)
    except Exception:
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


def _get_provider_class():
    """Return MnemosyneMemoryProvider, tolerating standalone plugin loads.

    Hermes may load plugin CLI modules directly from a file path using
    ``importlib.util.spec_from_file_location()``. In that context the module
    has no parent package, so relative imports raise ``ImportError``. Fall
    back to the absolute import — which works as long as the package is on
    ``sys.path`` (true for both pip installs and Hermes-managed plugin
    symlinks).
    """
    try:
        from . import MnemosyneMemoryProvider
    except ImportError:
        from mnemosyne_hermes import MnemosyneMemoryProvider
    return MnemosyneMemoryProvider


def _resolve_cli_bank(args, cmd):
    """Resolve which Mnemosyne bank the CLI beam should bind to.

    Under ``profile_isolation`` the provider writes to a per-profile bank
    (``<data_dir>/banks/<profile>/mnemosyne.db``), but the CLI historically
    always bound to the default/legacy bank — so ``stats`` and friends reported
    empty state even when the profile bank held data. Resolve the same bank the
    provider would, so the CLI operates on what the agent actually wrote.

    Precedence:
      1. explicit ``--bank`` (ignored for ``import``, whose ``--bank`` names the
         *source* provider bank, not the Mnemosyne target)
      2. the active Hermes profile bank, when ``profile_isolation`` is enabled
         (mirrors the provider's HERMES_HOME-basename fallback)
      3. ``None`` -> default/legacy bank (unchanged behavior)

    Never raises: resolution failures use a private sentinel so named exports
    can fail closed rather than silently selecting the default bank.
    """
    try:
        MnemosyneMemoryProvider = _get_provider_class()
        sanitize = MnemosyneMemoryProvider._sanitize_bank_name

        if cmd != "import":
            explicit = getattr(args, "bank", None)
            if explicit:
                bank = sanitize(explicit)
                if (
                    cmd == "export"
                    and bank == "default"
                    and str(explicit).strip().lower() != "default"
                ):
                    return _BANK_RESOLUTION_FAILED
                if bank == "default":
                    return "default" if cmd == "export" else None
                return bank

        hermes_home = os.environ.get("HERMES_HOME", "")
        if not hermes_home or not _profile_isolation_enabled(hermes_home):
            return None
        basename = Path(hermes_home).name
        if not basename or basename.lower() in (".hermes", "hermes", "default", ""):
            return None
        bank = sanitize(basename)
        return bank if bank != "default" else None
    except Exception:
        return _BANK_RESOLUTION_FAILED if cmd == "export" else None


def mnemosyne_command(args):
    """Dispatch ``hermes mnemosyne <subcommand>``."""
    cmd = getattr(args, "mnemosyne_cmd", None)
    if not cmd:
        print("Usage: hermes mnemosyne {stats|sleep|version|inspect|clear|export|import}")
        return 1

    if cmd == "version":
        print(f"Mnemosyne {_distribution_version('mnemosyne-memory')}")
        print(f"Mnemosyne Hermes {_distribution_version('mnemosyne-hermes')}")
        return 0

    bank = _resolve_cli_bank(args, cmd)
    if cmd == "export" and bank is _BANK_RESOLUTION_FAILED:
        print("Bank resolution failed")
        return 1

    # Reject unknown named banks BEFORE touching the filesystem. Mnemosyne(bank=)
    # would otherwise lazily create an empty bank directory + DB on first access
    # (e.g. via the beam below), defeating the guard and silently writing junk.
    # Use the side-effect-free existence check so we do NOT create the parent
    # `banks/` directory either (BankManager.__init__ eagerly mkdirs it).
    # Both explicit `--bank` and profile-derived implicit banks are covered:
    # a missing implicit profile bank must also fail closed rather than let
    # `Mnemosyne(bank=...)` materialize an empty bank mid-diagnostic.
    if cmd == "doctor" and bank:
        try:
            from mnemosyne.core.banks import bank_exists_read_only
            if not bank_exists_read_only(bank):
                print(f"Bank not found: {bank}")
                return 1
        except ValueError:
            # Raised by _validate_bank_name for malformed bank names.
            print(f"Bank not found: {bank}")
            return 1
        except Exception as e:
            # Import, permission, or probe runtime failures must surface as a
            # clean error, never escape as a raw traceback.
            print(f"Bank validation failed: {e}")
            return 1

    if cmd == "export" and bank:
        try:
            from mnemosyne.core.banks import get_bank_db_path_read_only
            db_path = get_bank_db_path_read_only(bank)
            if not db_path.is_file():
                print(f"Bank not found: {bank}")
                return 1
            if not _export_schema_is_complete_read_only(db_path):
                print(f"Bank schema incomplete: {bank}")
                return 1
        except (FileNotFoundError, ValueError):
            # Named exports require an existing bank database before the beam
            # or output path can be initialized.
            print(f"Bank not found: {bank}")
            return 1
        except Exception:
            print("Bank validation failed")
            return 1

    # Register Hermes host LLM only after export validation. A rejected named
    # export must not alter host-level runtime state before it exits fail-closed.
    # Use a try/except fallback chain: the relative import works when loaded
    # as part of the mnemosyne_hermes package; the absolute import is needed
    # when this module is loaded standalone via importlib discovery.
    try:
        try:
            from .hermes_llm_adapter import register_hermes_host_llm
        except ImportError:
            from mnemosyne_hermes.hermes_llm_adapter import register_hermes_host_llm
        register_hermes_host_llm()
    except Exception:
        pass

    try:
        if bank:
            # Bank-aware beam (Mnemosyne routes the bank to its own SQLite DB),
            # mirroring how the provider builds its beam under profile_isolation.
            from mnemosyne.core.memory import Mnemosyne
            beam = Mnemosyne(session_id="hermes_default", bank=bank).beam
        else:
            from mnemosyne.core.beam import BeamMemory
            beam = BeamMemory(session_id="hermes_default")
    except Exception as e:
        print(f"Error: Mnemosyne not available: {e}")
        return 1

    if cmd == "stats":
        if getattr(args, "global", False):
            working = beam.get_global_working_stats()
        else:
            working = beam.get_working_stats()
        episodic = beam.get_episodic_stats()
        memoria = beam.get_memoria_stats()
        print(json.dumps({"working": working, "episodic": episodic, "memoria": memoria}, indent=2))

    elif cmd == "sleep":
        dry_run = bool(getattr(args, "dry_run", False))
        if getattr(args, "all_sessions", False):
            result = beam.sleep_all_sessions(dry_run=dry_run)
        else:
            result = beam.sleep(dry_run=dry_run)
        print(json.dumps(result, indent=2))

    elif cmd == "inspect":
        query = getattr(args, "query", "") or ""
        limit = getattr(args, "limit", 10)
        if not query:
            query = input("Search query: ")
        results = beam.recall(query, top_k=limit)
        print(f"Results for '{query}': {len(results)}")
        for i, r in enumerate(results, 1):
            content = r.get("content", "")[:120]
            imp = r.get("importance", 0.0)
            print(f"  {i}. [{imp:.2f}] {content}")

    elif cmd == "clear":
        confirm = input("Clear scratchpad? This cannot be undone. [y/N]: ")
        if confirm.lower() in ("y", "yes"):
            beam.scratchpad_clear()
            print("Scratchpad cleared.")
        else:
            print("Cancelled.")

    elif cmd == "doctor":
        dry_run = bool(getattr(args, "dry_run", False))
        no_fix = bool(getattr(args, "no_fix", False))
        # Unknown-bank guard now runs before the beam is built (see top of
        # mnemosyne_command), so bank is guaranteed to exist here.
        try:
            from mnemosyne.diagnose import auto_fix, run_diagnostics
            result = run_diagnostics(bank=bank)
            resolved_bank = bank or "default"
            print("\nMnemosyne Diagnostics")
            print("=" * 40)
            print(f"  resolved_bank: {resolved_bank}")
            if result.get("resolved_db"):
                print(f"  resolved_db: {result.get('resolved_db')}")
            print(f"  Checks passed: {result.get('checks_passed', 0)}/{result.get('checks_total', 0)}")
            if result.get("key_findings"):
                print("\n  Key findings:")
                for finding in result["key_findings"]:
                    print(f"    - {finding}")
            else:
                print("\n  No issues detected.")

            if not no_fix:
                print("\n--- Auto-fix ---")
                fix_result = auto_fix(result.get("entries", []), dry_run=dry_run)
                if fix_result["fixed"]:
                    for item in fix_result["fixed"]:
                        print(f"  ✅ {item}")
                if fix_result["failed"]:
                    for item in fix_result["failed"]:
                        print(f"  ❌ {item['label']}: {item['error']}")
                if not fix_result["fixed"] and not fix_result["failed"]:
                    print("  Nothing to fix - all dependencies are healthy.")
            print(f"\nFull log: {result.get('log_path', 'unknown')}")
        except Exception as e:
            print(f"Diagnostic failed: {e}")
            return 1

    elif cmd == "export":
        output_path = getattr(args, "output", None)
        if not output_path:
            print("Usage: hermes mnemosyne export --output <path>")
            return 1
        try:
            from mnemosyne.core.memory import Mnemosyne
            mem = Mnemosyne(session_id="hermes_default", bank=bank)
            result = mem.export_to_file(output_path)
            print(f"Exported {result['working_memory_count']} working, {result['episodic_memory_count']} episodic, {result['legacy_memories_count']} legacy, {result['triples_count']} triples to {output_path}")
            _print_completeness_warning(result, imported=False)
        except Exception as e:
            print(f"Export failed: {e}")
            return 1

    elif cmd == "import":
        # --list-providers
        if getattr(args, "list_providers", False):
            from mnemosyne.core.importers import PROVIDERS
            print("Supported import providers:")
            for name, info in PROVIDERS.items():
                print(f"  {name}: {info['description']}")
                print(f"         docs: {info['docs']}")
                print(f"         env key: {info['env_key']}")
                print(f"         pip: {info['pypi_package']}")
            return 0

        # --agentic: generate instructions for user's AI agent
        generate_script_flag = getattr(args, "generate_script", False)
        agentic_flag = getattr(args, "agentic", False)
        from_provider = getattr(args, "from_provider", None)
        output_script = getattr(args, "output_script", None)

        if agentic_flag and from_provider:
            from mnemosyne.core.importers.agentic import generate_agent_instructions
            instructions = generate_agent_instructions(from_provider)
            if output_script:
                Path(output_script).write_text(instructions)
                print(f"Agent instructions saved to {output_script}")
            else:
                print(instructions)
            return 0

        if generate_script_flag and from_provider:
            from mnemosyne.core.importers.agentic import generate_migration_script
            api_key = getattr(args, "api_key", None)
            user_id = getattr(args, "user_id", None)
            script = generate_migration_script(
                from_provider,
                api_key=api_key or "",
                user_id=user_id or "",
            )
            if output_script:
                Path(output_script).write_text(script)
                print(f"Migration script saved to {output_script}")
            else:
                print(script)
            return 0

        cross_provider = from_provider
        input_path = getattr(args, "input", None)
        dry_run = getattr(args, "dry_run", False)
        session_id = getattr(args, "session_id", None)
        channel_id = getattr(args, "channel_id", None)

        try:
            from mnemosyne.core.memory import Mnemosyne
            mem = Mnemosyne(session_id=session_id or "import_session",
                            channel_id=channel_id)
        except Exception as e:
            print(f"Error: Mnemosyne not available: {e}")
            return 1

        # Cross-provider import
        if cross_provider:
            api_key = getattr(args, "api_key", None)
            user_id = getattr(args, "user_id", None)
            agent_id = getattr(args, "agent_id", None)
            base_url = getattr(args, "base_url", None)

            def _print_import_result(result):
                print("\nImport complete:")
                print(f"  Total found: {result.total}")
                print(f"  Imported:    {result.imported}")
                print(f"  Skipped:     {result.skipped}")
                print(f"  Failed:      {result.failed}")
                if result.errors:
                    print("  Errors:")
                    for err in result.errors[:10]:
                        print(f"    - {err}")
                    if len(result.errors) > 10:
                        print(f"    ... and {len(result.errors) - 10} more")

            if cross_provider == "hindsight":
                file_path = getattr(args, "file", None) or input_path
                bank = getattr(args, "bank", None) or "hermes"
                if not file_path and not base_url:
                    print("Error: Hindsight import requires --file/--input or --base-url.")
                    return 1

                print("Importing from hindsight...")
                if dry_run:
                    print("  (dry-run mode: no memories will be written)")

                try:
                    from mnemosyne.core.importers import import_from_provider
                    import_kwargs = {
                        "file_path": file_path,
                        "base_url": base_url,
                        "bank": bank,
                        "dry_run": dry_run,
                        "session_id": session_id,
                        "channel_id": channel_id,
                    }
                    result = import_from_provider(
                        "hindsight", mem,
                        **import_kwargs,
                    )
                    _print_import_result(result)
                    return 0 if result.failed == 0 else 1
                except ValueError as e:
                    print(f"Error: {e}")
                    return 1
                except Exception as e:
                    print(f"Import failed: {e}")
                    return 1

            if cross_provider == "holographic":
                db_path = getattr(args, "db_path", None)
                min_trust = getattr(args, "min_trust", 0.0)

                print("Importing from holographic memory...")
                if dry_run:
                    print("  (dry-run mode: no memories will be written)")

                try:
                    from mnemosyne.core.importers import import_from_provider
                    import_kwargs = {
                        "db_path": db_path,
                        "min_trust": min_trust,
                        "dry_run": dry_run,
                        "session_id": session_id,
                        "channel_id": channel_id,
                    }
                    result = import_from_provider(
                        "holographic", mem,
                        **import_kwargs,
                    )
                    _print_import_result(result)
                    return 0 if result.failed == 0 else 1
                except ValueError as e:
                    print(f"Error: {e}")
                    return 1
                except Exception as e:
                    print(f"Import failed: {e}")
                    return 1

            # Try env var fallback
            import os
            if not api_key:
                info = __import__("mnemosyne.core.importers", fromlist=["PROVIDERS"]).PROVIDERS
                pk = info.get(cross_provider, {}).get("env_key", "")
                if pk:
                    api_key = os.environ.get(pk)
            if not api_key:
                print(f"Error: --api-key required for {cross_provider} import. "
                      f"Or set the {cross_provider.upper()}_API_KEY env var.")
                return 1

            print(f"Importing from {cross_provider}...")
            if dry_run:
                print("  (dry-run mode: no memories will be written)")

            try:
                from mnemosyne.core.importers import import_from_provider
                result = import_from_provider(
                    cross_provider, mem,
                    api_key=api_key,
                    user_id=user_id,
                    agent_id=agent_id,
                    base_url=base_url,
                    dry_run=dry_run,
                    session_id=session_id,
                    channel_id=channel_id,
                )
                _print_import_result(result)
                return 0 if result.failed == 0 else 1
            except ValueError as e:
                print(f"Error: {e}")
                return 1
            except Exception as e:
                print(f"Import failed: {e}")
                return 1

        # File import
        force = getattr(args, "force", False)
        if not input_path:
            print("Usage: hermes mnemosyne import --input <path> [--force] [--dry-run]")
            print("       hermes mnemosyne import --from <provider> --api-key <key> [--dry-run]")
            print("       hermes mnemosyne import --list-providers")
            return 1
        try:
            stats = mem.import_from_file(input_path, force=force, dry_run=dry_run)
            beam_stats = stats.get("beam", {})
            legacy_stats = stats.get("legacy", {})
            triples_stats = stats.get("triples", {})
            print("Import complete:")
            print(f"  Working: +{beam_stats.get('working_memory', {}).get('inserted', 0)}")
            print(f"  Episodic: +{beam_stats.get('episodic_memory', {}).get('inserted', 0)}")
            print(f"  Legacy: +{legacy_stats.get('inserted', 0)}")
            print(f"  Triples: +{triples_stats.get('inserted', 0)}")
            _print_completeness_warning(stats, imported=True)
            if force:
                print(
                    "  (force mode: overwrites would be applied)"
                    if dry_run
                    else "  (force mode: overwrites applied)"
                )
            if dry_run:
                print("  (dry-run mode: no memories were written)")
        except Exception as e:
            print(f"Import failed: {e}")
            return 1

    return 0
