"""Regression coverage for self-contained Hermes wrapper imports."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from mnemosyne_hermes import install


def test_generated_wrapper_bootstraps_init_and_cli_from_selected_site_packages(tmp_path):
    site_packages = tmp_path / "side-venv" / "site-packages"
    package = site_packages / "mnemosyne_hermes"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "SIDE_VALUE = 'selected-side-package'\n"
        "def register_memory_provider(*args): pass\n",
        encoding="utf-8",
    )
    (package / "cli.py").write_text(
        "def register_cli(*args): return 'selected-cli'\n",
        encoding="utf-8",
    )

    wrapper = tmp_path / "hermes-home" / "plugins" / "mnemosyne"
    install._write_wrapper_plugin(
        wrapper,
        python=Path(sys.executable),
        site_packages=site_packages,
    )

    code = f"""
import importlib.util
import sys
import types
from pathlib import Path

wrapper = Path({str(wrapper)!r})
parent = types.ModuleType('synthetic_hermes_plugins')
parent.__path__ = []
sys.modules[parent.__name__] = parent

init_name = 'synthetic_hermes_plugins.mnemosyne'
init_spec = importlib.util.spec_from_file_location(
    init_name, wrapper / '__init__.py', submodule_search_locations=[str(wrapper)]
)
init_module = importlib.util.module_from_spec(init_spec)
sys.modules[init_name] = init_module
init_spec.loader.exec_module(init_module)
assert init_module.SIDE_VALUE == 'selected-side-package'

cli_name = init_name + '.cli'
cli_spec = importlib.util.spec_from_file_location(cli_name, wrapper / 'cli.py')
cli_module = importlib.util.module_from_spec(cli_spec)
sys.modules[cli_name] = cli_module
cli_spec.loader.exec_module(cli_module)
assert cli_module.register_cli() == 'selected-cli'

side = sys.modules['mnemosyne_hermes']
assert Path(side.__file__).parent == Path({str(package)!r})
assert sys.path[0] == {str(site_packages)!r}
"""
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


def test_generated_cli_bootstraps_selected_site_packages_when_loaded_standalone_first(tmp_path):
    site_packages = tmp_path / "side-venv" / "site-packages"
    package = site_packages / "mnemosyne_hermes"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("SIDE_VALUE = 'selected-side-package'\n", encoding="utf-8")
    (package / "cli.py").write_text(
        "def register_cli(*args): return 'selected-cli'\n"
        "def mnemosyne_command(*args): return 'selected-command'\n",
        encoding="utf-8",
    )

    wrapper = tmp_path / "hermes-home" / "plugins" / "mnemosyne"
    install._write_wrapper_plugin(
        wrapper,
        python=Path(sys.executable),
        site_packages=site_packages,
    )

    code = f"""
import importlib.util
import sys
from pathlib import Path

wrapper = Path({str(wrapper)!r})
site_packages = {str(site_packages)!r}
for name in list(sys.modules):
    if name == 'mnemosyne_hermes' or name.startswith('mnemosyne_hermes.'):
        del sys.modules[name]
assert site_packages not in sys.path
assert 'mnemosyne_hermes' not in sys.modules

cli_spec = importlib.util.spec_from_file_location('standalone_mnemosyne_cli', wrapper / 'cli.py')
assert cli_spec.name == 'standalone_mnemosyne_cli'
cli_module = importlib.util.module_from_spec(cli_spec)
cli_spec.loader.exec_module(cli_module)

assert sys.path[0] == site_packages
assert Path(sys.modules['mnemosyne_hermes'].__file__).parent == Path({str(package)!r})
assert cli_module.register_cli() == 'selected-cli'
assert cli_module.mnemosyne_command() == 'selected-command'
"""
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


@pytest.mark.parametrize("invalid_python", ["relative", "missing", "directory", "non_executable"])
def test_generated_bootstrap_rejects_invalid_manifest_python_without_sys_path_mutation(tmp_path, invalid_python):
    site_packages = tmp_path / "side-venv" / "site-packages"
    site_packages.mkdir(parents=True)
    wrapper = tmp_path / "hermes-home" / "plugins" / "mnemosyne"
    install._write_wrapper_plugin(
        wrapper,
        python=Path(sys.executable),
        site_packages=site_packages,
    )
    manifest_path = wrapper / "mnemosyne-wrapper.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if invalid_python == "relative":
        manifest["python"] = "relative-python"
    elif invalid_python == "missing":
        manifest["python"] = str(tmp_path / "missing-python")
    elif invalid_python == "directory":
        manifest["python"] = str(tmp_path)
    else:
        non_executable = tmp_path / "non-executable-python"
        non_executable.write_text("not an executable\n", encoding="utf-8")
        non_executable.chmod(non_executable.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        if os.access(non_executable, os.X_OK):
            pytest.skip("platform cannot create a non-executable regular file")
        manifest["python"] = str(non_executable)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    code = f"""
import importlib.util
import sys
from pathlib import Path

wrapper = Path({str(wrapper)!r})
bootstrap_spec = importlib.util.spec_from_file_location(
    'standalone_mnemosyne_bootstrap', wrapper / '_mnemosyne_bootstrap.py'
)
bootstrap_module = importlib.util.module_from_spec(bootstrap_spec)
bootstrap_spec.loader.exec_module(bootstrap_module)
before = list(sys.path)
try:
    bootstrap_module.activate()
except RuntimeError as exc:
    assert str(exc) == 'Invalid Mnemosyne wrapper Python executable'
else:
    raise AssertionError('invalid manifest python should fail activation')
assert sys.path == before
"""
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
