"""Regression tests for the mnemosyne-hermes console entry point."""

from __future__ import annotations

import importlib.abc
import importlib.metadata
import subprocess
import sys

import mnemosyne_hermes
from mnemosyne_hermes import install


class _BlockMnemosyneCore(importlib.abc.MetaPathFinder):
    """Simulate an environment where mnemosyne.core is unavailable."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if fullname == "mnemosyne.core" or fullname.startswith("mnemosyne.core."):
            raise ModuleNotFoundError("No module named 'mnemosyne.core'")
        return None


def test_main_help_exits_successfully(capsys):
    try:
        install.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "mnemosyne-hermes" in out
    assert "install" in out
    assert "status" in out


def test_package_version_matches_distribution_metadata():
    assert mnemosyne_hermes.__version__ == importlib.metadata.version("mnemosyne-hermes")


def test_package_and_install_module_import_without_mnemosyne_core():
    code = r'''
import importlib.abc
import sys
class BlockMnemosyneCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mnemosyne.core" or fullname.startswith("mnemosyne.core."):
            raise ModuleNotFoundError("No module named 'mnemosyne.core'")
        return None
sys.meta_path.insert(0, BlockMnemosyneCore())
import mnemosyne_hermes
from mnemosyne_hermes import install
assert callable(install.main)
print("ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
