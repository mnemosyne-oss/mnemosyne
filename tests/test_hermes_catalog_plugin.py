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
    """Keep catalog identity, exclusivity, and wrapper version aligned."""
    m = _manifest()
    assert m["name"] == "mnemosyne", "catalog install dir must match the wrapper's plugin name"
    assert m["kind"] == "exclusive", "memory providers must not be imported by the general loader"
    assert m["version"] == "0.7.3"
    assert m["provides_hooks"] == [] and m["provides_middleware"] == [] and m["requires_env"] == []


# The floor is a contract with an external system, not a tunable threshold: 0.21.4 is the first
# Hermes release that carries declared-Python-dependency install (hermes-agent#113851) and
# automatic catalog install of memory providers that leave core (hermes-agent#114569). Neither
# is in 0.21.3, so a 0.21.3 install would load the plugin without the install path it needs.
MIN_HERMES_RELEASE = ">=0.21.4"


def test_manifest_gates_loading_on_a_lower_bound_hermes_release():
    """`requires_hermes` must name 0.21.4, the first release that carries the whole catalog path.

    Hermes' `plugins_manifest.requires_hermes_error()` blocks the plugin and reports the reason
    when the running version fails this specifier, so an exact pin (`==0.21.4`) would uninstall
    the plugin's future rather than describe it -- the next Hermes release would load-block a
    plugin that works. That is the part this test can decide on its own: the specifier must be a
    plain floor.

    The floor is 0.21.4 and not 0.21.3 because both halves of the catalog path reached Hermes
    `main` after the 0.21.3 release (tag v2026.9.14, 2026-09-14): declared Python dependencies
    installed and re-applied after `hermes update` (hermes-agent#113851, commit `96e8a23222`) and
    automatic catalog install of memory providers that leave core (hermes-agent#114569, commit
    `d177b119`). `hermes_cli/plugin_python_deps.py` does not exist at the v2026.9.14 tag, so a
    0.21.3 install would load this plugin without the install path it needs. dplush asked for
    exactly this floor while reviewing the catalog entry on hermes-agent#113581.

    A release that is not out yet is the point rather than a defect: below 0.21.4 the gate fails
    closed, which is safer than loading a plugin whose declared dependencies never get installed.
    This repository cannot enumerate Hermes' published releases, so raising the floor stays a
    review-time obligation against hermes-agent, and this constant is what a future bump must
    edit on purpose.
    """
    spec = _manifest()["requires_hermes"]
    assert re.fullmatch(r">=\s*\d+(\.\d+){1,2}", spec), (
        f"requires_hermes must be a '>=X.Y' or '>=X.Y.Z' lower bound, got {spec!r}: an exact "
        "pin would load-block the plugin on the very next Hermes release, and a specifier that "
        "is not a plain floor is harder to keep honest as Hermes moves."
    )
    assert spec == MIN_HERMES_RELEASE, (
        f"requires_hermes must be {MIN_HERMES_RELEASE}, the first Hermes release that carries "
        f"hermes-agent#113851 and #114569; got {spec!r}. Lowering it re-ships eb9ffde, which let "
        "the plugin load on 0.21.3 where its declared dependencies are never installed."
    )


def _toml_loads(text: str) -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(text)


def test_wrapper_pyproject_declares_the_package_and_is_not_a_distribution():
    """Require the catalog wrapper to declare packages without becoming one."""
    data = _toml_loads((CATALOG / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert "mnemosyne-hermes>=0.7.3,<0.8" in deps, deps
    assert "mnemosyne-memory[embeddings]>=4.0.0b3" in deps, deps
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
