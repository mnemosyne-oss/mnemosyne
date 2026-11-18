"""Console entry points must parse their arguments before touching anything (#1048).

``mnemosyne-install --help`` once ran the installer, and ``mnemosyne-uninstall``
was wired to a function that never read its arguments, so ``--help`` removed
the provider and reset ``memory.provider`` in the Hermes config. These tests
resolve each script exactly as packaging does (``pyproject.toml`` and
``setup.py``), run it in a subprocess against a seeded, isolated Hermes home,
and require that ``--help`` and unknown arguments change nothing.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DESTRUCTIVE = ("mnemosyne-install", "mnemosyne-uninstall")


def _pyproject_scripts() -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return dict(data["project"]["scripts"])


def _setup_py_scripts() -> dict:
    tree = ast.parse((REPO / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "console_scripts":
                    entries = ast.literal_eval(value)
                    return {e.split("=", 1)[0].strip(): e.split("=", 1)[1].strip() for e in entries}
    raise AssertionError("console_scripts not found in setup.py")


def _seed_hermes_home(root: Path) -> Path:
    home = root / "home"
    hermes = home / ".hermes"
    plugin = hermes / "plugins" / "mnemosyne"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: mnemosyne\n", encoding="utf-8")
    (hermes / "config.yaml").write_text("memory:\n  provider: mnemosyne\n", encoding="utf-8")
    return home


def _snapshot(home: Path) -> dict:
    return {
        str(p.relative_to(home)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(home.rglob("*"))
    }


def _run(target: str, argv: list, home: Path) -> subprocess.CompletedProcess:
    module, func = target.split(":")
    code = (
        "import sys\n"
        f"sys.argv = [{argv[0]!r}, *{argv[1:]!r}]\n"
        f"from {module} import {func} as entry\n"
        "rc = entry()\n"
        "sys.exit(rc if isinstance(rc, int) else 0)\n"
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "HERMES_HOME": str(home / ".hermes"),
        "MNEMOSYNE_DATA_DIR": str(home / "mnemosyne-data"),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
        "PYTHONPATH": str(REPO),
    }
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=120, cwd=str(home))


def test_both_packaging_files_declare_the_same_console_scripts():
    assert _pyproject_scripts() == _setup_py_scripts()


@pytest.mark.parametrize("script", DESTRUCTIVE)
def test_help_prints_usage_and_changes_nothing(script, tmp_path):
    home = _seed_hermes_home(tmp_path)
    before = _snapshot(home)

    result = _run(_pyproject_scripts()[script], [script, "--help"], home)

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert script in result.stdout
    assert _snapshot(home) == before


@pytest.mark.parametrize("script", DESTRUCTIVE)
def test_unknown_argument_is_rejected_and_changes_nothing(script, tmp_path):
    home = _seed_hermes_home(tmp_path)
    before = _snapshot(home)

    result = _run(_pyproject_scripts()[script], [script, "--definitely-not-an-option"], home)

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert _snapshot(home) == before


def test_every_console_script_target_reads_its_arguments(tmp_path):
    """A console-script target that takes a required positional, or that
    ignores argv entirely, is how #1048 happened. Every target must import and
    be a callable whose parameters are all optional, and the two destructive
    ones must be the argument-parsing entry points.

    Checked in a clean interpreter: inside the pytest process
    ``mnemosyne.integrations`` can resolve to the repository's top-level
    ``integrations/`` directory, because the repository root carries its own
    ``__init__.py``. An installed script never sees that layout, so the probe
    runs from an unrelated directory.
    """
    scripts = _pyproject_scripts()
    assert scripts["mnemosyne-install"] == "mnemosyne.install:main"
    assert scripts["mnemosyne-uninstall"] == "mnemosyne.install:uninstall_main"
    code = (
        "import importlib, inspect, sys\n"
        f"targets = {scripts!r}\n"
        "bad = []\n"
        "for script, target in targets.items():\n"
        "    module_name, func_name = target.split(':')\n"
        "    func = getattr(importlib.import_module(module_name), func_name)\n"
        "    for p in inspect.signature(func).parameters.values():\n"
        "        if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):\n"
        "            bad.append(f'{script} -> {target}: required {p.name}')\n"
        "print('\\n'.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO),
             "MNEMOSYNE_NO_EMBEDDINGS": "1", "HOME": str(tmp_path)},
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_uninstall_without_arguments_still_uninstalls(tmp_path):
    home = _seed_hermes_home(tmp_path)

    result = _run(_pyproject_scripts()["mnemosyne-uninstall"], ["mnemosyne-uninstall"], home)

    assert result.returncode == 0, result.stderr
    config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
    assert re.search(r"provider:\s*null", config)
