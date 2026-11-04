from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
import yaml

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CORE_INIT = ROOT / "mnemosyne" / "__init__.py"
STANDALONE_PROJECT = ROOT / "integrations" / "hermes" / "pyproject.toml"
CATALOG_PROJECT = ROOT / "integrations" / "hermes-catalog" / "pyproject.toml"
STANDALONE_CORE_FLOOR = "4.0.0"
CATALOG_CORE_FLOOR = "4.0.0"
LEGACY_HERMES_MANIFEST = ROOT / "integrations" / "hermes" / "plugin.yaml"
CORE_HERMES_MANIFEST = ROOT / "hermes_memory_provider" / "plugin.yaml"
REQUIRED_CORE_API = (
    ("mnemosyne.core.query_sanitize", "sanitize_prefetch_query"),
    ("mnemosyne.core.filters", "make_write_policy"),
    ("mnemosyne.core.filters", "resolve_write_policy"),
    ("mnemosyne.core.filters", "active_write_policy"),
    ("mnemosyne.core.filters", "current_write_policy"),
    ("mnemosyne.core.filters", "write_policy_operation"),
    ("mnemosyne.core.filters", "_SYSTEM_DERIVED_WRITE_CAPABILITY"),
    ("mnemosyne.core.filters", "admit_memory_write"),
    ("mnemosyne.core.verbatim_ledger", "VerbatimLedger"),
    ("mnemosyne.upgrade_hermes", "upgrade_command"),
    ("mnemosyne.core.media_tool", "remember_media_tool"),
)


def _core_version() -> str:
    """Return the core version declared by the source package."""
    tree = ast.parse(CORE_INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    value = ast.literal_eval(node.value)
                    assert isinstance(value, str)
                    return value
    raise AssertionError("mnemosyne.__version__ not found")


def _mnemosyne_memory_specs(project: Path) -> dict[str, list[Requirement]]:
    """Keep parsed Core requirements associated with their dependency group."""
    data = tomllib.loads(project.read_text(encoding="utf-8"))
    groups = {"base": data["project"].get("dependencies", [])}
    groups.update(data["project"].get("optional-dependencies", {}))
    return {group: _core_requirements(specs) for group, specs in groups.items()}


def _core_requirements(specs: list[str]) -> list[Requirement]:
    """Parse dependency declarations while allowing unrelated packages."""
    requirements = [Requirement(spec) for spec in specs]
    return [r for r in requirements if canonicalize_name(r.name) == "mnemosyne-memory"]


def _assert_core_requirement(requirements: list[Requirement], expected: str) -> None:
    """Require one exact Core contract, including extras and parsed specifiers."""
    assert len(requirements) == 1, requirements
    actual, wanted = requirements[0], Requirement(expected)
    assert canonicalize_name(actual.name) == canonicalize_name(wanted.name)
    assert actual.extras == wanted.extras, (actual, wanted)
    assert actual.specifier == wanted.specifier, (actual, wanted)


def test_hermes_packages_require_their_supported_core_api_release() -> None:
    """Standalone prerelease support and catalog promotion have separate floors."""
    core_version = _core_version()
    parsed_version = Version(core_version)
    for project, floor, groups in (
        (STANDALONE_PROJECT, STANDALONE_CORE_FLOOR,
         {"base": "embeddings", "llm": "llm", "mcp": "mcp", "all": "all"}),
        (CATALOG_PROJECT, CATALOG_CORE_FLOOR, {"base": "embeddings"}),
    ):
        requirements = _mnemosyne_memory_specs(project)
        for group, extra in groups.items():
            specs = requirements.get(group, [])
            _assert_core_requirement(specs, f"mnemosyne-memory[{extra}]>={floor}")
            requirement = specs[0]
            if project == STANDALONE_PROJECT:
                # Core is mandatory on every supported standalone environment.
                assert requirement.marker is None, (group, requirement)
            assert parsed_version in requirement.specifier, (
                project,
                core_version,
                requirement,
            )
    manifest = yaml.safe_load(LEGACY_HERMES_MANIFEST.read_text(encoding="utf-8"))
    _assert_core_requirement(
        _core_requirements(manifest.get("pip_dependencies", [])),
        f"mnemosyne-memory[embeddings]>={STANDALONE_CORE_FLOOR}",
    )
    assert f"version: {core_version}" in CORE_HERMES_MANIFEST.read_text(
        encoding="utf-8"
    )


def test_standalone_core_requirements_admit_final_without_admitting_betas() -> None:
    """Every extra must require Core 4.0.0 final, so no install pairs 0.7.7 with a beta."""
    for requirements in _mnemosyne_memory_specs(STANDALONE_PROJECT).values():
        for requirement in requirements:
            for version in ("4.0.0", "4.0.1"):
                assert Version(version) in requirement.specifier, (requirement, version)
            for version in ("3.15.1", "4.0.0b5", "4.0.0b6", "4.0.0rc1"):
                assert Version(version) not in requirement.specifier, (requirement, version)


def test_hermes_required_core_api_is_present() -> None:
    """Require the source core to export every API imported by Hermes."""
    for module_name, symbol_name in REQUIRED_CORE_API:
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol_name), (module_name, symbol_name)


def _build_wheel(source: Path, wheel_dir: Path) -> Path:
    """Build one project wheel and return its path."""
    before = set(wheel_dir.glob("*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    created = set(wheel_dir.glob("*.whl")) - before
    assert len(created) == 1, created
    return created.pop()


def _extract_wheel(wheel: Path, target: Path) -> None:
    """Extract one wheel into an isolated import root."""
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)


def test_built_release_pair_completes_the_provider_lifecycle(tmp_path: Path) -> None:
    """Build the declared release pair and exercise its public lifecycle."""
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    core_wheel = _build_wheel(ROOT, wheel_dir)
    hermes_wheel = _build_wheel(ROOT / "integrations" / "hermes", wheel_dir)
    site = tmp_path / "site"
    site.mkdir()
    _extract_wheel(core_wheel, site)
    _extract_wheel(hermes_wheel, site)
    home = tmp_path / "hermes-home"
    yaml_spec = importlib.util.find_spec("yaml")
    assert yaml_spec is not None and yaml_spec.origin is not None
    dependency_site = Path(yaml_spec.origin).resolve().parents[1]
    code = textwrap.dedent(
        f"""
        import json
        import importlib
        import sys
        from pathlib import Path
        site = Path({str(site)!r}).resolve()
        sys.path.insert(0, str(site))
        sys.path.insert(1, {str(dependency_site)!r})
        import mnemosyne
        import mnemosyne_hermes
        for module in (mnemosyne, mnemosyne_hermes):
            assert Path(module.__file__).resolve().is_relative_to(site), module.__file__
        required_core_api = {REQUIRED_CORE_API!r}
        for module_name, symbol in required_core_api:
            api_module = importlib.import_module(module_name)
            assert Path(api_module.__file__).resolve().is_relative_to(site), api_module.__file__
            assert hasattr(api_module, symbol), (module_name, symbol)
        from mnemosyne_hermes import MnemosyneMemoryProvider

        def call(provider, name, args):
            return json.loads(provider.handle_tool_call(name, args))

        provider = MnemosyneMemoryProvider()
        provider.initialize(
            "release-pair-primary",
            agent_context="primary",
            hermes_home={str(home)!r},
            auto_sleep=False,
        )
        assert provider._beam is not None, provider._init_error
        token = "release_contract_1014_unique_token"
        assert call(provider, "mnemosyne_remember", {{"content": token}})["status"] == "stored"
        recalled = call(provider, "mnemosyne_recall", {{"query": token, "limit": 5}})
        assert any(token in row.get("content", "") for row in recalled["results"])
        provider.initialize(
            "release-pair-skip",
            agent_context="subagent",
            hermes_home={str(home)!r},
            auto_sleep=False,
        )
        skipped = call(provider, "mnemosyne_stats", {{}})
        assert skipped["status"] == "memory_unavailable"
        assert skipped["reason_code"] == "reset_by_reinit"
        provider.initialize(
            "release-pair-recovery",
            agent_context="primary",
            hermes_home={str(home)!r},
            auto_sleep=False,
        )
        assert provider._beam is not None, provider._init_error
        assert call(provider, "mnemosyne_stats", {{}}).get("status") != "memory_unavailable"
        provider.shutdown()
        """
    )
    env = os.environ.copy()
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
