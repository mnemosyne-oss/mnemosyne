"""Pure runtime diagnostics shared by doctor and legacy diagnose.

This module intentionally has no logging, database, provider, repair, or CLI
dependencies.  Keeping it neutral lets ``mnemosyne.doctor`` report runtime
capabilities without importing the mutable ``mnemosyne.diagnose`` command.
"""

import importlib.metadata
import platform
import sqlite3
import sys
from typing import Any


def collect_runtime_diagnostics() -> dict[str, Any]:
    """Run pure runtime, dependency, and capability checks without a provider."""

    checks: list[dict[str, str]] = []

    def add(category: str, check: str, status: str, detail: str = "") -> None:
        checks.append({"category": category, "check": check, "status": status, "detail": detail})

    add("env", "python_version", sys.version.split()[0])
    add("env", "platform", platform.platform())
    add("env", "python_executable", sys.executable)

    try:
        import mnemosyne

        version = getattr(mnemosyne, "__version__", None)
        if not version:
            version = importlib.metadata.version("mnemosyne-memory")
        add("package", "mnemosyne_version", str(version))
    except Exception:
        add("package", "mnemosyne_version", "ERROR", "package version unavailable")

    for name, module in {
        "fastembed": "fastembed",
        "sqlite_vec": "sqlite_vec",
        "numpy": "numpy",
        "huggingface_hub": "huggingface_hub",
    }.items():
        try:
            dependency = __import__(module)
            add("deps", name, "OK", f"version={getattr(dependency, '__version__', 'unknown')}")
        except ImportError:
            add("deps", name, "MISSING")
        except Exception:
            add("deps", name, "ERROR", "dependency import failed")

    try:
        dependency = __import__("ctransformers")
        add("deps", "ctransformers", "OK", f"version={getattr(dependency, '__version__', 'unknown')}")
    except ImportError:
        add("deps", "ctransformers", "OPTIONAL", "optional local-GGUF fallback dependency not installed")
    except Exception:
        add("deps", "ctransformers", "ERROR", "dependency import failed")

    try:
        from mnemosyne.core import embeddings as _embeddings

        add("core", "embeddings_available", "YES" if _embeddings.available() else "NO")
        add("core", "embeddings_model", _embeddings._DEFAULT_MODEL)
    except Exception:
        add("core", "embeddings_available", "ERROR", "embeddings capability unavailable")

    try:
        from mnemosyne.core.beam import _SQLITE_VEC_AVAILABLE

        vec_can_load = False
        if _SQLITE_VEC_AVAILABLE:
            try:
                test_conn = sqlite3.connect(":memory:")
                try:
                    test_conn.enable_load_extension(True)
                    vec_can_load = True
                finally:
                    test_conn.close()
            except Exception:
                vec_can_load = False
        add("core", "sqlite_vec_available", "YES" if vec_can_load else "NO")
        if _SQLITE_VEC_AVAILABLE and not vec_can_load:
            add("core", "sqlite_vec_warning", "NO", "extension loading unavailable")
    except Exception:
        add("core", "sqlite_vec", "ERROR", "sqlite-vec capability unavailable")

    statuses = {entry["status"] for entry in checks}
    overall = "unavailable" if "ERROR" in statuses else "warning" if statuses & {"MISSING", "NO"} else "ok"
    return {"status": overall, "checks": checks}