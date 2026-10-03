"""Regression test: CLI host LLM backend registration under standalone module loading.

When Hermes' plugin discovery loads cli.py via
``importlib.util.spec_from_file_location()`` (without registering the
parent package), relative imports like ``from .hermes_llm_adapter import ...``
fail silently. The try/except in ``mnemosyne_command()`` swallows the
ImportError, so ``register_hermes_host_llm()`` never runs and
``MNEMOSYNE_HOST_LLM_ENABLED`` is silently ignored.

This test loads both CLI copies the same way ``discover_plugins``
(in `mnemosyne/core/plugins.py`) does and verifies the registration path is reached.
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
    exactly like Hermes' discover_plugins() (mnemosyne/core/plugins.py) does.

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
def test_register_host_llm_reached_under_standalone_loading(fake_agent_module, monkeypatch, cli_path):
    """Standalone non-version commands must still register the host backend."""
    fake_agent_module.call_llm = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mod_name = f"_test_standalone_{cli_path.stem}_{hash(str(cli_path)) & 0xFFFFFFFF:x}"
    mod = _load_cli_standalone(cli_path, mod_name)

    assert hasattr(mod, "mnemosyne_command"), f"{cli_path} does not define mnemosyne_command"

    from mnemosyne.core import beam as beam_module

    class FakeBeam:
        def get_working_stats(self):
            return {}

        def get_episodic_stats(self):
            return {}

        def get_memoria_stats(self):
            return {}

    monkeypatch.setattr(beam_module, "BeamMemory", lambda *_args, **_kwargs: FakeBeam())

    import argparse, contextlib, io

    with contextlib.redirect_stdout(io.StringIO()):
        mod.mnemosyne_command(argparse.Namespace(mnemosyne_cmd="stats"))

    backend = get_host_llm_backend()
    assert backend is not None
    assert backend.name == "hermes"