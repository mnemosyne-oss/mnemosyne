"""Pure runtime diagnostics shared by doctor and legacy diagnose.

This module intentionally has no logging, database, provider, repair, or CLI
dependencies.  Keeping it neutral lets ``mnemosyne.doctor`` report runtime
capabilities without importing the mutable ``mnemosyne.diagnose`` command.
"""

import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any


def resolve_python_executable(python_executable: str) -> str:
    """Resolve PATH names as subprocesses do while preserving explicit paths."""
    expanded = os.path.expanduser(python_executable)
    if os.path.dirname(expanded):
        return os.path.abspath(expanded)
    return shutil.which(expanded) or expanded


def _collect_target_runtime_diagnostics(python_executable: str) -> dict[str, Any]:
    """Collect a selected interpreter's checks in an isolated subprocess.

    Validate the JSON response before exposing it to diagnostic consumers.
    A failed launch, timeout, or invalid response reports an unavailable target
    without falling back to the caller's interpreter or revealing its path.
    """
    executable = resolve_python_executable(python_executable)
    display_name = Path(executable).name or "python"
    script = (
        "import json; "
        "from mnemosyne.runtime_diagnostics import collect_runtime_diagnostics; "
        "print('MNEMOSYNE_DIAGNOSTICS=' + json.dumps(collect_runtime_diagnostics()))"
    )
    try:
        result = subprocess.run(
            [executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None

    payload = None
    if result is not None and result.returncode == 0:
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("MNEMOSYNE_DIAGNOSTICS="):
                try:
                    payload = json.loads(line.partition("=")[2])
                except json.JSONDecodeError:
                    pass
                break

    required_keys = {"category", "check", "status", "detail"}
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("checks"), list)
        or not all(
            isinstance(check, dict) and required_keys <= check.keys()
            for check in payload["checks"]
        )
    ):
        return {
            "status": "unavailable",
            "checks": [
                {
                    "category": "runtime",
                    "check": "target_interpreter",
                    "status": "ERROR",
                    "detail": "selected interpreter could not run Mnemosyne diagnostics",
                }
            ],
            "scope": "target_interpreter",
            "executable": display_name,
            "unavailable": True,
        }

    executable_check = next(
        (
            check
            for check in payload["checks"]
            if isinstance(check, dict) and check.get("check") == "python_executable"
        ),
        {},
    )
    executable_name = str(executable_check.get("detail") or display_name)
    executable_name = executable_name.replace("\\", "/").rsplit("/", 1)[-1]
    payload["scope"] = "target_interpreter"
    payload["executable"] = executable_name
    return payload


def collect_runtime_diagnostics(python_executable: str | None = None) -> dict[str, Any]:
    """Run pure runtime, dependency, and capability checks without a provider.

    By default, inspect the current interpreter. When ``python_executable`` is
    supplied, resolve PATH names or explicit paths and run checks there with
    Python's isolated mode. An unavailable target never falls back to the
    current interpreter.
    """

    if python_executable is not None:
        return _collect_target_runtime_diagnostics(python_executable)

    checks: list[dict[str, str]] = []

    def add(category: str, check: str, status: str, detail: str = "") -> None:
        checks.append({"category": category, "check": check, "status": status, "detail": detail})

    add("env", "python_version", "OK", sys.version.split()[0])
    add("env", "platform", "OK", platform.platform())
    # Report only the executable name: an absolute interpreter path can reveal
    # a user's home directory or virtual-environment layout in diagnostics.
    add("env", "python_executable", "OK", Path(sys.executable).name)

    try:
        import mnemosyne

        version = getattr(mnemosyne, "__version__", None)
        if not version:
            version = importlib.metadata.version("mnemosyne-memory")
        add("package", "mnemosyne_version", "OK", str(version))
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
        # available() reports env/key presence, so it stays YES while every
        # credentialed call is being refused; surface the refusal itself.
        refusal = _embeddings.policy_refusal_message()
        if refusal:
            add("core", "embedding_policy_refusal", "ERROR", refusal)
        add("core", "embeddings_model", "OK", _embeddings._DEFAULT_MODEL)
        # Surface the resolved dimension so operators can confirm their
        # MNEMOSYNE_EMBEDDING_DIM / model-table resolution via the doctor,
        # complementing the fail-loud unknown-model resolver.
        add("core", "embeddings_dim", "OK", str(_embeddings.EMBEDDING_DIM))
    except Exception:
        add("core", "embeddings_available", "ERROR", "embeddings capability unavailable")

    try:
        from mnemosyne.core.beam import _SQLITE_VEC_AVAILABLE

        vec_can_load = False
        if _SQLITE_VEC_AVAILABLE:
            try:
                import sqlite_vec

                test_conn = sqlite3.connect(":memory:")
                try:
                    test_conn.enable_load_extension(True)
                    sqlite_vec.load(test_conn)
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
