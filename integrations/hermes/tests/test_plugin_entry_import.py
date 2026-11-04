"""The plugins-group target must stay importable without loading Mnemosyne."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUB = ROOT / "integrations/hermes/src/_mnemosyne_hermes_plugin.py"


def test_plugins_group_target_does_not_import_mnemosyne():
    for name in list(sys.modules):
        if name == "mnemosyne" or name.startswith(("mnemosyne.", "mnemosyne_hermes")):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("_mnemosyne_hermes_plugin", STUB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.register.__code__.co_argcount == 1
    assert "mnemosyne" not in sys.modules
    assert "mnemosyne_hermes" not in sys.modules
    assert "mnemosyne.batch_tool" not in sys.modules
