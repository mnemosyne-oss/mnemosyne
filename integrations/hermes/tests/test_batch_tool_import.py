"""Importing the Hermes plugin must not import the batch/embedding stack."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "integrations/hermes/src"


def test_importing_hermes_plugin_does_not_import_batch_tool():
    sys.path.insert(0, str(SRC))
    absent = (
        "mnemosyne.batch_tool",
        "mnemosyne.core.beam",
        "mnemosyne.core.embeddings",
        "fastembed",
    )
    for name in absent:
        sys.modules.pop(name, None)
    sys.modules.pop("mnemosyne_hermes", None)
    try:
        importlib.import_module("mnemosyne_hermes")
    except Exception as exc:
        # A missing Hermes install must not hide a deferred import that did happen.
        imported = [name for name in absent if name in sys.modules]
        assert not imported, (imported, exc)
        return
    imported = [name for name in absent if name in sys.modules]
    assert not imported, imported


def test_package_getattr_still_imports_cli_submodule():
    sys.path.insert(0, str(SRC))
    sys.modules.pop("mnemosyne_hermes.cli", None)
    provider = importlib.import_module("mnemosyne_hermes")
    provider.__dict__.pop("cli", None)
    module = provider.cli
    assert module.__name__ == "mnemosyne_hermes.cli"
    assert "mnemosyne.batch_tool" not in sys.modules
