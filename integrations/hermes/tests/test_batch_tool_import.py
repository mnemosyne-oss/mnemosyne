"""Importing the Hermes plugin must not import the batch/embedding stack."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "integrations/hermes/src"


def test_importing_hermes_plugin_does_not_import_batch_tool():
    sys.path.insert(0, str(SRC))
    sys.modules.pop("mnemosyne.batch_tool", None)
    sys.modules.pop("mnemosyne_hermes", None)
    try:
        importlib.import_module("mnemosyne_hermes")
    except Exception as exc:
        # A missing Hermes install must not hide a batch_tool import that did happen.
        assert "mnemosyne.batch_tool" not in sys.modules, exc
        return
    assert "mnemosyne.batch_tool" not in sys.modules
