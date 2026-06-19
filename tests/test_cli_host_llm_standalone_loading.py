"""Regression test: CLI host LLM backend registration under standalone module loading.

When Hermes' plugin discovery loads cli.py via
``importlib.util.spec_from_file_location()`` (without registering the
parent package), relative imports like ``from .hermes_llm_adapter import ...``
fail silently. The try/except in ``mnemosyne_command()`` swallows the
ImportError, so ``register_hermes_host_llm()`` never runs and
``MNEMOSYNE_HOST_LLM_ENABLED`` is silently ignored.

This test loads both CLI copies the same way ``discover_plugin_cli_commands``
does and verifies the registration path is reached.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnemosyne.core import llm_backends
from mnemosyne.core.llm_backends import get_host_llm_backend


# ---------------------------------------------------------------------------
# Fake `agent` package — same pattern as test_hermes_llm_adapter.py
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_agent_module(monkeypatch):
    agent_pkg = types.ModuleType("agent")
    aux_client = types.ModuleType("agent.auxiliary_client")
    agent_pkg.auxiliary_client = aux_client
    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux_client)
    yield aux_client
    # cleanup
    llm_backends.set_host_llm_backend(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

CLI_COPIES = [
    REPO_ROOT / "integrations" / "hermes" / "src" / "mnemosyne_hermes" / "cli.py",
    REPO_ROOT / "hermes_memory_provider" / "cli.py",
]


def _load_cli_standalone(cli_path: Path, module_name: str):
    """Load a cli.py as a standalone module via spec_from_file_location,
    exactly like Hermes' discover_plugin_cli_commands() does.

    The parent package is NOT registered in sys.modules, so relative
    imports (``from .hermes_llm_adapter import ...``) will fail.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(cli_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_backend():
    """Ensure each test starts with no backend registered."""
    llm_backends.set_host_llm_backend(None)
    yield
    llm_backends.set_host_llm_backend(None)


@pytest.mark.parametrize("cli_path", CLI_COPIES, ids=[str(p.relative_to(REPO_ROOT)) for p in CLI_COPIES])
def test_register_host_llm_reached_under_standalone_loading(fake_agent_module, cli_path):
    """Loading cli.py via spec_from_file_location must still reach
    register_hermes_host_llm() — the fallback import chain must succeed
    even though the relative import fails.
    """
    fake_agent_module.call_llm = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mod_name = f"_test_standalone_{cli_path.stem}_{hash(str(cli_path)) & 0xFFFFFFFF:x}"
    mod = _load_cli_standalone(cli_path, mod_name)

    # Verify mnemosyne_command exists and is callable
    assert hasattr(mod, "mnemosyne_command"), f"{cli_path} does not define mnemosyne_command"

    # Call mnemosyne_command with args that pass the early-return guard
    # (mnemosyne_cmd must be set) but exercise a lightweight subcommand.
    # "version" only needs to import mnemosyne.__version__ — no DB access.
    # The registration happens before subcommand dispatch, so any valid cmd
    # triggers it. We capture stdout to suppress the version output.
    import argparse, io, contextlib
    args = argparse.Namespace(mnemosyne_cmd="version")
    with contextlib.redirect_stdout(io.StringIO()):
        mod.mnemosyne_command(args)

    # The backend should be registered — if the import fallback failed
    # silently, this assertion fails.
    backend = get_host_llm_backend()
    assert backend is not None, (
        f"register_hermes_host_llm() was not reached when loading {cli_path} "
        f"via spec_from_file_location — the relative import likely failed "
        f"silently and the fallback import was not attempted."
    )
    assert backend.name == "hermes"