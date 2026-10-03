"""The legacy provider loads when the gateway cwd holds a ``mnemosyne/`` dir (#1056).

Hermes executes a directory plugin's sibling modules before its ``__init__``,
and ``__init__`` is what puts the source checkout on ``sys.path``. A sibling
that imports ``mnemosyne`` at module level therefore resolves it against the
gateway's cwd first; with the default ``~/.hermes/mnemosyne/`` data directory
and cwd ``~/.hermes`` that is a namespace package, the import fails, the
failure is cached, and the provider never registers.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "hermes_memory_provider"

_LOADER = """
import importlib.util, sys, types
plugin_dir, name = sys.argv[1], "hermes_plugins_mnemosyne"
pkg = types.ModuleType(name); pkg.__path__ = [plugin_dir]
sys.modules[name] = pkg
spec = importlib.util.spec_from_file_location(name + ".audit", plugin_dir + "/audit.py")
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
spec = importlib.util.spec_from_file_location(
    name, plugin_dir + "/__init__.py", submodule_search_locations=[plugin_dir])
init = importlib.util.module_from_spec(spec); sys.modules[name] = init
spec.loader.exec_module(init)
assert callable(init.register)
import mnemosyne
print(mnemosyne.__file__)
"""


def test_provider_loads_siblings_first_from_a_shadowing_cwd(tmp_path):
    (tmp_path / "mnemosyne").mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "MNEMOSYNE_DATA_DIR": str(tmp_path / "mnemosyne"),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", _LOADER, str(PLUGIN)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert Path(result.stdout.strip().splitlines()[-1]) == REPO / "mnemosyne" / "__init__.py"


def test_no_sibling_imports_mnemosyne_at_module_level():
    offenders = []
    for path in sorted(PLUGIN.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            if any(n == "mnemosyne" or n.startswith("mnemosyne.") for n in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], offenders
