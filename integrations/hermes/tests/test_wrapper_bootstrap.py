"""Regression coverage for self-contained Hermes wrapper imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
