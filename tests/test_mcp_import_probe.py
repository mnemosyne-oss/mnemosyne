"""Importing the core package must not execute the optional mcp package."""
from __future__ import annotations

import importlib
import importlib.util
import sys

import mnemosyne


def test_importing_mnemosyne_does_not_import_mcp():
    sys.modules.pop("mcp", None)
    importlib.reload(mnemosyne)
    assert "mcp" not in sys.modules
    if importlib.util.find_spec("mcp") is not None:
        assert "run_mcp_server" in mnemosyne.__all__
    else:
        assert "run_mcp_server" not in mnemosyne.__all__
