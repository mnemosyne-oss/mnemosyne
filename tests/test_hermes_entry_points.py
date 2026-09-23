"""Contract tests for the Hermes wrapper entry-points and version surfaces.

The plugin was renamed to ``hermes-mnemosyne`` (see the #972 revert), so
every distribution entry-point key must use the hyphenated name, and every
wrapper version surface must agree on a single version.

Run with: MNEMOSYNE_NO_EMBEDDINGS=1 pytest tests/test_hermes_entry_points.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

import yaml

ROOT = Path(__file__).resolve().parents[1]
HERMES_PYPROJECT = ROOT / "integrations" / "hermes" / "pyproject.toml"
HERMES_PLUGIN = ROOT / "integrations" / "hermes" / "plugin.yaml"
HERMES_SRC_PLUGIN = (
    ROOT / "integrations" / "hermes" / "src" / "mnemosyne_hermes" / "plugin.yaml"
)
HERMES_INIT = (
    ROOT / "integrations" / "hermes" / "src" / "mnemosyne_hermes" / "__init__.py"
)
CATALOG_PYPROJECT = ROOT / "integrations" / "hermes-catalog" / "pyproject.toml"
CATALOG_PLUGIN = ROOT / "integrations" / "hermes-catalog" / "plugin.yaml"

EXPECTED_NAME = "hermes-mnemosyne"
ENTRY_POINT_GROUPS = (
    "hermes_agent.plugins",
    "hermes_agent.memory_providers",
)


def _wrapper_version() -> str:
    data = tomllib.loads(HERMES_PYPROJECT.read_text(encoding="utf-8"))
    version = data["project"]["version"]
    assert isinstance(version, str)
    return version


class TestEntryPointNames:
    def test_entry_point_keys_use_hyphenated_plugin_name(self):
        data = tomllib.loads(HERMES_PYPROJECT.read_text(encoding="utf-8"))
        entry_points = data["project"]["entry-points"]
        for group in ENTRY_POINT_GROUPS:
            assert group in entry_points, f"missing entry-point group {group}"
            assert set(entry_points[group]) == {EXPECTED_NAME}, (
                f"{group}: expected only {EXPECTED_NAME!r}, "
                f"got {sorted(entry_points[group])}"
            )

    def test_entry_point_targets_resolve(self):
        data = tomllib.loads(HERMES_PYPROJECT.read_text(encoding="utf-8"))
        entry_points = data["project"]["entry-points"]
        assert entry_points["hermes_agent.plugins"][EXPECTED_NAME] == (
            "mnemosyne_hermes:register"
        )
        assert entry_points["hermes_agent.memory_providers"][EXPECTED_NAME] == (
            "mnemosyne_hermes"
        )

    def test_plugin_manifests_use_hyphenated_name(self):
        for manifest in (HERMES_PLUGIN, HERMES_SRC_PLUGIN):
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            assert data["name"] == EXPECTED_NAME, manifest


class TestVersionSurfaces:
    def test_all_wrapper_surfaces_agree(self):
        """pyproject, both plugin.yaml copies and __init__ share one version."""
        version = _wrapper_version()
        for manifest in (HERMES_PLUGIN, HERMES_SRC_PLUGIN):
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            assert str(data["version"]) == version, manifest
        tree = ast.parse(HERMES_INIT.read_text(encoding="utf-8"))
        dunder = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__version__":
                        dunder = ast.literal_eval(node.value)
        assert dunder == version, (HERMES_INIT, dunder, version)

    def test_catalog_tracks_wrapper_release(self):
        """The catalog wrapper pins the same wrapper release it ships."""
        version = _wrapper_version()
        catalog = tomllib.loads(CATALOG_PYPROJECT.read_text(encoding="utf-8"))
        assert catalog["project"]["version"] == version
        deps = catalog["project"]["dependencies"]
        assert f"mnemosyne-hermes>={version},<0.8" in deps
        plugin = yaml.safe_load(CATALOG_PLUGIN.read_text(encoding="utf-8"))
        assert str(plugin["version"]) == version
