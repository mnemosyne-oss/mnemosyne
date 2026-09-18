"""The Hermes plugin-catalog directory (integrations/hermes-catalog) stays loadable and honest.

It is a thin wrapper: no implementation, only a manifest, a dependency declaration and a
shim that re-exports the package's registration hooks. These tests pin the contract from
hermes-agent#113851 and dplush's #859 decision: the catalog root is separate from the
PyPI project in integrations/hermes, and what the manifest declares matches the package.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "integrations" / "hermes-catalog"
PACKAGE_SRC = REPO / "integrations" / "hermes" / "src"


def _manifest() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((CATALOG / "plugin.yaml").read_text())


def test_catalog_dir_has_every_loadable_piece():
    for name in (
        "plugin.yaml",
        "__init__.py",
        "cli.py",
        "pyproject.toml",
        "README.md",
    ):
        assert (CATALOG / name).is_file(), name


def test_manifest_is_an_exclusive_memory_provider_named_like_the_wrapper():
    m = _manifest()
    assert m["name"] == "mnemosyne", "catalog install dir must match the wrapper's plugin name"
    assert m["kind"] == "exclusive", "memory providers must not be imported by the general loader"
    assert m["version"] == "0.7.1"
    assert m["provides_hooks"] == [] and m["provides_middleware"] == [] and m["requires_env"] == []


def _toml_loads(text: str) -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(text)


def test_wrapper_pyproject_declares_the_package_and_is_not_a_distribution():
    data = _toml_loads((CATALOG / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert any(d.startswith("mnemosyne-hermes>=0.7.1") for d in deps), deps
    assert any(d.startswith("mnemosyne-memory[embeddings]") for d in deps), deps
    assert "build-system" not in data, "the catalog wrapper must never build as a package"
    assert data["project"]["version"] == _manifest()["version"]


def test_declared_tools_are_real_package_tools():
    sys.path.insert(0, str(PACKAGE_SRC))
    try:
        from mnemosyne_hermes import tools
    finally:
        sys.path.pop(0)
    real = {s["name"] for s in tools.ALL_TOOL_SCHEMAS}
    declared = _manifest()["provides_tools"]
    assert len(declared) == len(set(declared)), "duplicate tool declarations"
    assert set(declared) <= real, sorted(set(declared) - real)
    assert "mnemosyne_recall" in declared and "mnemosyne_remember" in declared


def test_shim_loads_in_a_fresh_process_and_exports_both_hooks():
    """What a catalog install does: import the directory with the package on the path."""
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(PACKAGE_SRC)!r})\n"
        f"spec = importlib.util.spec_from_file_location('mnemosyne_catalog', {str(CATALOG / '__init__.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "assert callable(mod.register) and callable(mod.register_memory_provider)\n"
        "print('shim-ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        env={"MNEMOSYNE_NO_EMBEDDINGS": "1", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    assert "shim-ok" in out.stdout


def test_shim_source_is_discoverable_as_a_memory_provider_without_import():
    """plugins/memory discovery greps __init__.py for the provider contract before importing."""
    src = (CATALOG / "__init__.py").read_text()
    assert "register_memory_provider" in src[:8192]
    assert re.search(r"^from mnemosyne_hermes import", src, re.M)
