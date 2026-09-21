"""Delegate Hermes CLI discovery without importing the provider runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_package_spec = importlib.util.find_spec("mnemosyne_hermes")
_package_locations = tuple(_package_spec.submodule_search_locations or ()) if _package_spec else ()
if not _package_locations:
    raise ImportError("Cannot locate the installed mnemosyne_hermes package")

_cli_path = Path(_package_locations[0]) / "cli.py"
_cli_spec = importlib.util.spec_from_file_location("_mnemosyne_catalog_cli_delegate", _cli_path)
if _cli_spec is None or _cli_spec.loader is None:
    raise ImportError("Cannot load the installed mnemosyne_hermes CLI")
_cli_module = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(_cli_module)

mnemosyne_command = _cli_module.mnemosyne_command
register_cli = _cli_module.register_cli

__all__ = ["mnemosyne_command", "register_cli"]
