"""Regression coverage for legacy Hermes provider and CLI runtime selection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _copy_legacy_entrypoint(tmp_path: Path, entrypoint: str) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    site_packages = tmp_path / "side-venv" / "lib" / "python-current" / "site-packages"
    source = repo_root / "hermes_memory_provider" / entrypoint
    target = site_packages / "hermes_memory_provider" / entrypoint
    target.parent.mkdir(parents=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return site_packages, target


def _copy_legacy_source_entrypoint(tmp_path: Path, entrypoint: str) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = tmp_path / "source-checkout"
    source = repo_root / "hermes_memory_provider" / entrypoint
    target = source_root / "hermes_memory_provider" / entrypoint
    target.parent.mkdir(parents=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return source_root, target


def _run_isolated(tmp_path: Path, code: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _legacy_provider_stubs() -> str:
    """Provide only the import names needed to load the legacy provider."""
    return """
import types

for name in (
    'mnemosyne',
    'mnemosyne.core',
    'mnemosyne.core.episodic_graph',
    'mnemosyne.core.beam',
    'mnemosyne.batch_tool',
    'mnemosyne.hermes_config',
    'mnemosyne.integrations',
    'mnemosyne.integrations.hermes_persona_prompt',
):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module

sys.modules['mnemosyne.core.episodic_graph'].GraphEdge = type('GraphEdge', (), {})
sys.modules['mnemosyne.core.beam'].WORKING_MEMORY_TTL_HOURS = 1
batch_tool = sys.modules['mnemosyne.batch_tool']
batch_tool.BatchValidationError = Exception
batch_tool.apply_beam_batch = lambda *args, **kwargs: None
batch_tool.batch_validation_error_payload = lambda *args, **kwargs: None
batch_tool.dry_run_batch = lambda *args, **kwargs: None
batch_tool.validate_batch_operations = lambda *args, **kwargs: None
sys.modules['mnemosyne.hermes_config'].read_hermes_config_key = lambda *args, **kwargs: None
sys.modules['mnemosyne.integrations.hermes_persona_prompt'].HermesPersonaPromptMixin = type(
    'HermesPersonaPromptMixin', (), {}
)
"""


def _load_legacy_entrypoint_code(entrypoint: str, target: Path, site_packages: Path) -> str:
    provider_stubs = _legacy_provider_stubs() if entrypoint == "__init__.py" else ""
    return f"""
import importlib.util
import sys
from pathlib import Path

{provider_stubs}

target = Path({str(target)!r})
site_packages = {str(site_packages.resolve())!r}
spec = importlib.util.spec_from_file_location('synthetic_legacy_{entrypoint}', target)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert sys.path[0] == site_packages
"""


@pytest.mark.parametrize("entrypoint", ["__init__.py", "cli.py"], ids=["provider", "standalone-cli"])
@pytest.mark.parametrize(
    "metadata_kind",
    ["different-minor", "malformed-version-info", "pyvenv-cfg-directory"],
)
def test_legacy_entrypoint_rejects_incompatible_metadata_before_import_or_path_activation(
    tmp_path, entrypoint, metadata_kind
):
    site_packages, target = _copy_legacy_entrypoint(tmp_path, entrypoint)
    selected_version = f"{sys.version_info.major}.{sys.version_info.minor + 1}"
    config_path = site_packages.parents[2] / "pyvenv.cfg"
    expected_selected_version = selected_version
    if metadata_kind == "different-minor":
        config_path.write_text(f"version = {selected_version}.0\n", encoding="utf-8")
    elif metadata_kind == "malformed-version-info":
        config_path.write_text("version_info = not-a-version\n", encoding="utf-8")
        expected_selected_version = "unknown"
    else:
        config_path.mkdir()
        expected_selected_version = "unknown"

    code = f"""
import importlib.util
import sys
from pathlib import Path

target = Path({str(target)!r})
site_packages = {str(site_packages.resolve())!r}
before = list(sys.path)
spec = importlib.util.spec_from_file_location('synthetic_legacy_{entrypoint}', target)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
try:
    spec.loader.exec_module(module)
except RuntimeError as exc:
    message = str(exc)
    assert 'runtime Python' in message
    assert 'selected Mnemosyne environment Python {expected_selected_version}' in message
else:
    raise AssertionError('incompatible selected virtualenv should fail legacy entrypoint import')
assert site_packages not in sys.path
assert sys.path == before
assert 'mnemosyne' not in sys.modules
assert not any(name.startswith('mnemosyne.') for name in sys.modules)
"""
    _run_isolated(tmp_path, code)


@pytest.mark.parametrize("entrypoint", ["__init__.py", "cli.py"], ids=["provider", "standalone-cli"])
def test_legacy_entrypoint_accepts_matching_nearby_version_info(tmp_path, entrypoint):
    site_packages, target = _copy_legacy_entrypoint(tmp_path, entrypoint)
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro + 1}"
    (site_packages.parents[2] / "pyvenv.cfg").write_text(
        f"version_info = {current_version}\n", encoding="utf-8"
    )

    _run_isolated(tmp_path, _load_legacy_entrypoint_code(entrypoint, target, site_packages))


@pytest.mark.parametrize("entrypoint", ["__init__.py", "cli.py"], ids=["provider", "standalone-cli"])
def test_legacy_entrypoint_accepts_missing_nearby_pyvenv_cfg(tmp_path, entrypoint):
    site_packages, target = _copy_legacy_entrypoint(tmp_path, entrypoint)
    # A config beyond the standard venv ancestors must not turn a no-config
    # selected environment into a compatibility failure.
    (site_packages.parents[3] / "pyvenv.cfg").write_text("version = 99.0.0\n", encoding="utf-8")

    _run_isolated(tmp_path, _load_legacy_entrypoint_code(entrypoint, target, site_packages))


@pytest.mark.parametrize("entrypoint", ["__init__.py", "cli.py"], ids=["provider", "standalone-cli"])
def test_legacy_source_checkout_ignores_nearby_mismatched_pyvenv_cfg(tmp_path, entrypoint):
    source_root, target = _copy_legacy_source_entrypoint(tmp_path, entrypoint)
    selected_version = f"{sys.version_info.major}.{sys.version_info.minor + 1}"
    (source_root / "pyvenv.cfg").write_text(
        f"version = {selected_version}.0\n", encoding="utf-8"
    )

    _run_isolated(tmp_path, _load_legacy_entrypoint_code(entrypoint, target, source_root))
