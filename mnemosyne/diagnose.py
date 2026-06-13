"""
Mnemosyne Diagnostics
=====================
PII-safe debug logging for troubleshooting installation and runtime issues.

Logs to ~/.hermes/mnemosyne/logs/diagnose_YYYY-MM-DD_HHMMSS.jsonl
Never includes memory content, user queries, or API keys.

Supports --fix mode: auto-installs missing dependencies.
"""

import importlib.metadata
import json
import os
import subprocess
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List

LOG_DIR = Path.home() / ".hermes" / "mnemosyne" / "logs"

# Map of missing dependency checks to pip install commands
FIX_MAP = {
    "fastembed": {
        "check": lambda e: e["check"] == "fastembed" and e["status"] == "MISSING",
        "install": ["pip", "install", "mnemosyne-memory[embeddings]"],
        "label": "fastembed (embeddings engine)",
    },
    "sqlite_vec": {
        "check": lambda e: e["check"] == "sqlite_vec" and e["status"] == "MISSING",
        "install": ["pip", "install", "sqlite-vec"],
        "label": "sqlite-vec (vector search)",
    },
    "numpy": {
        "check": lambda e: e["check"] == "numpy" and e["status"] == "MISSING",
        "install": ["pip", "install", "numpy"],
        "label": "numpy",
    },
    "huggingface_hub": {
        "check": lambda e: e["check"] == "huggingface_hub" and e["status"] == "MISSING",
        "install": ["pip", "install", "huggingface_hub"],
        "label": "huggingface_hub",
    },
}


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_path() -> Path:
    _ensure_log_dir()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return LOG_DIR / f"diagnose_{ts}.jsonl"


def _safe_env(name: str) -> str:
    """Return env var presence indicator, never the value."""
    val = os.environ.get(name, "")
    return "set" if val else "unset"


def run_diagnostics() -> Dict:
    """
    Run full diagnostic scan and write PII-safe log.
    Returns summary dict for display.
    """
    log_path = _log_path()
    entries: List[Dict] = []

    def log(category: str, check: str, status: str, detail: str = ""):
        entry = {
            "ts": datetime.now().isoformat(),
            "category": category,
            "check": check,
            "status": status,
            "detail": detail
        }
        entries.append(entry)
        return entry

    # --- Python environment ---
    log("env", "python_version", sys.version.split()[0])
    log("env", "platform", platform.platform())
    log("env", "python_executable", sys.executable)

    # --- Mnemosyne package ---
    try:
        import mnemosyne
        version = getattr(mnemosyne, "__version__", None)
        if not version:
            version = importlib.metadata.version("mnemosyne-memory")
        log("package", "mnemosyne_version", str(version))
    except Exception as e:
        log("package", "mnemosyne_version", "ERROR", str(e))

    # --- Core dependencies ---
    required_deps = {
        "fastembed": "fastembed",
        "sqlite_vec": "sqlite_vec",
        "numpy": "numpy",
        "huggingface_hub": "huggingface_hub",
    }
    optional_deps = {
        # Optional local-GGUF fallback only. Host/remote LLM paths and the
        # non-LLM fallback work without it, so absence should not fail the
        # installation health check.
        "ctransformers": "ctransformers",
    }
    for name, module in required_deps.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "unknown")
            log("deps", name, "OK", f"version={ver}")
        except ImportError:
            log("deps", name, "MISSING")
        except Exception as e:
            log("deps", name, "ERROR", str(e))
    for name, module in optional_deps.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "unknown")
            log("deps", name, "OK", f"version={ver}")
        except ImportError:
            log("deps", name, "OPTIONAL", "optional local-GGUF fallback dependency not installed")
        except Exception as e:
            log("deps", name, "ERROR", str(e))

    # --- Mnemosyne core components ---
    try:
        from mnemosyne.core import embeddings as _embeddings
        log("core", "embeddings_available", "YES" if _embeddings.available() else "NO")
        log("core", "embeddings_model", _embeddings._DEFAULT_MODEL)
    except Exception as e:
        log("core", "embeddings", "ERROR", str(e))

    try:
        from mnemosyne.core.beam import _SQLITE_VEC_AVAILABLE
        # _SQLITE_VEC_AVAILABLE only checks whether the pip package imports.
        # It doesn't verify that the running sqlite3 module can actually load
        # the extension (required for Python builds without
        # --enable-loadable-sqlite-extensions). Do a runtime check here.
        _vec_can_load = False
        if _SQLITE_VEC_AVAILABLE:
            try:
                import sqlite3 as _sqlite3
                _test_conn = _sqlite3.connect(":memory:")
                _test_conn.enable_load_extension(True)
                _vec_can_load = True
                _test_conn.close()
            except Exception:
                _vec_can_load = False
        log("core", "sqlite_vec_available", "YES" if _vec_can_load else "NO")
        if _SQLITE_VEC_AVAILABLE and not _vec_can_load:
            log("core", "sqlite_vec_warning", "Package imports but extension cannot load. Rebuild Python with --enable-loadable-sqlite-extensions.")
    except Exception as e:
        log("core", "sqlite_vec", "ERROR", str(e))

    # --- Database state ---
    try:
        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne()
        stats = mem.get_stats()

        # PII-safe: counts and config only, never content
        log("db", "legacy_total", str(stats.get("total_memories", 0)))
        log("db", "total_sessions", str(stats.get("total_sessions", 0)))

        beam = stats.get("beam", {})
        wm = beam.get("working_memory", {})
        ep = beam.get("episodic_memory", {})

        log("db", "working_total", str(wm.get("total", 0)))
        log("db", "episodic_total", str(ep.get("total", 0)))
        log("db", "episodic_vectors", str(ep.get("vectors", 0)))
        log("db", "episodic_vec_type", ep.get("vec_type", "none"))
        log("db", "db_path", stats.get("database", "unknown"))
    except Exception as e:
        log("db", "stats", "ERROR", str(e))

    # --- Environment variables (presence only, never values) ---
    env_vars = [
        "MNEMOSYNE_DATA_DIR",
        "MNEMOSYNE_LLM_ENABLED",
        "MNEMOSYNE_LLM_BASE_URL",
        "MNEMOSYNE_VEC_TYPE",
        "MNEMOSYNE_WM_MAX_ITEMS",
        "HERMES_HOME",
    ]
    for var in env_vars:
        log("env", var, _safe_env(var))

    # --- Write log file ---
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # --- Build summary ---
    non_failure_statuses = ("OK", "YES", "set", "OPTIONAL")
    summary = {
        "log_path": str(log_path),
        "checks_total": len(entries),
        "checks_passed": sum(1 for e in entries if e["status"] in non_failure_statuses),
        "checks_failed": sum(1 for e in entries if e["status"] in ("MISSING", "NO", "ERROR")),
        "key_findings": [],
        "fixable": [],
        "entries": entries,
    }

    # Auto-detect common problems
    embed_ok = any(e["check"] == "embeddings_available" and e["status"] == "YES" for e in entries)
    vec_ok = any(e["check"] == "sqlite_vec_available" and e["status"] == "YES" for e in entries)
    ep_vec = next((e for e in entries if e["check"] == "episodic_vectors"), None)
    ep_vec_type = next((e for e in entries if e["check"] == "episodic_vec_type"), None)

    if not embed_ok:
        summary["key_findings"].append("fastembed not available - install with: pip install mnemosyne-memory[embeddings]")
        summary["fixable"].append("fastembed")
    if not vec_ok:
        summary["key_findings"].append("sqlite-vec not available - install with: pip install sqlite-vec")
        summary["fixable"].append("sqlite_vec")
    if embed_ok and vec_ok and ep_vec and ep_vec["status"] == "0":
        summary["key_findings"].append("Both fastembed and sqlite-vec are available but episodic vectors=0 - memories may not have been consolidated yet. Run: hermes mnemosyne sleep")
    if embed_ok and vec_ok and ep_vec and int(ep_vec["status"]) > 0:
        vtype = ep_vec_type["status"] if ep_vec_type else "unknown"
        msg = f"Semantic search is active with {ep_vec['status']} vectors in episodic memory (backend: {vtype})"
        if vtype in ("binary", "json"):
            # vec_episodes (the sqlite-vec ANN table) is absent -- usually
            # because this Python's sqlite3 can't load the sqlite-vec
            # extension. Recall still works via the binary/JSON fallback;
            # the ANN index only matters at much larger scale.
            msg += " - the sqlite-vec ANN index is not in use (extension not loadable); the fallback is fine at small/medium scale"
        summary["key_findings"].append(msg)

    return summary


def auto_fix(entries: List[Dict] = None, dry_run: bool = False) -> Dict:
    """
    Auto-install missing dependencies detected by diagnostics.

    Args:
        entries: Optional list of diagnostic entries. If None, runs diagnostics first.
        dry_run: If True, report what would be fixed without installing.

    Returns:
        Dict with 'fixed', 'failed', 'skipped' lists and 'ran' bool.
    """
    if entries is None:
        summary = run_diagnostics()
        entries = summary.get("entries", [])

    result = {"fixed": [], "failed": [], "skipped": [], "ran": True}

    for fix_key, fix_info in FIX_MAP.items():
        # Check if this dependency is MISSING
        is_missing = any(fix_info["check"](e) for e in entries)
        if not is_missing:
            continue

        label = fix_info["label"]
        cmd = fix_info["install"]

        if dry_run:
            result["fixed"].append(f"WOULD install: {label} ({' '.join(cmd)})")
            continue

        print(f"🔧 Installing {label}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            result["fixed"].append(label)
            print(f"   ✅ {label} installed")
        except subprocess.CalledProcessError as e:
            result["failed"].append({"label": label, "error": e.stderr.strip()})
            print(f"   ❌ Failed: {e.stderr.strip()[:200]}")
        except FileNotFoundError:
            result["failed"].append({"label": label, "error": "pip not found"})
            print(f"   ❌ pip not found in PATH")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mnemosyne diagnostics")
    parser.add_argument("--fix", action="store_true", help="Auto-install missing dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without installing")
    args = parser.parse_args()

    result = run_diagnostics()
    print(json.dumps(result, indent=2))

    if args.fix or args.dry_run:
        fix_result = auto_fix(result.get("entries", []), dry_run=args.dry_run)
        print("\n--- Auto-fix ---")
        print(json.dumps(fix_result, indent=2))
