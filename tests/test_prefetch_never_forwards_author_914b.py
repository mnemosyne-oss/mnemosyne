"""CWE-200 regression: automatic prefetch must never inject author identity.

Removed in this PR: both Hermes provider surfaces resolved an author from
``beam.author_id`` or ``MNEMOSYNE_AUTHOR_ID`` and forwarded it to
``beam.recall()``. A non-empty ``author_id`` makes recall() replace
session/channel filtering with ``(1=1)``, silently widening prefetch scope
across gateway threads and leaking memories across sessions.

Author identity is a per-write stamp only; the automatic prefetch/recall path
must never receive it. Parametrized over both provider surfaces.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"

HERMES_AUTHOR_ID = "hermes-env-author"
HERMES_AUTHOR_TYPE = "agent"


@pytest.fixture(autouse=True)
def _restore_provider_modules():
    """Restore provider + core module identities after each test.

    ``_import_provider`` drops the ``mnemosyne`` namespace to load the real
    core, which would otherwise orphan module references bound at collection
    time in other test modules.
    """
    saved_providers = {
        name: mod for name, mod in list(sys.modules.items())
        if name.split(".")[0] in ("hermes_memory_provider", "mnemosyne_hermes", "mnemosyne")
    }
    yield
    for name in [n for n in list(sys.modules)
                 if n.split(".")[0] in ("hermes_memory_provider", "mnemosyne_hermes", "mnemosyne")]:
        del sys.modules[name]
    sys.modules.update(saved_providers)


def _import_provider(package: str):
    """Import a provider package from its own source root."""
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name == "mnemosyne" or name.startswith("mnemosyne."):
            del sys.modules[name]
    import_root = INTEGRATION_SRC if package == "mnemosyne_hermes" else PROJECT_ROOT
    inserted = [str(import_root)]
    if import_root != PROJECT_ROOT:
        inserted.append(str(PROJECT_ROOT))
    for path in reversed(inserted):
        sys.path.insert(0, path)
    try:
        module = importlib.import_module(package)
        import mnemosyne.core.beam as _beam_mod
        module._BEAM_CLS = _beam_mod.BeamMemory
        return module
    finally:
        for path in inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


class _RecordingBeam:
    """Records recall() kwargs; simulates a beam carrying a read-side author
    identity — the identity prefetch must ignore."""

    author_id = "beam-author"
    author_type = "agent"

    def __init__(self) -> None:
        self.last_kwargs = None

    def recall(self, **kwargs):
        self.last_kwargs = kwargs
        return []


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_prefetch_never_passes_author_to_recall(provider_module_name, monkeypatch):
    """Automatic prefetch must NOT forward author_id/author_type to recall(),
    even when MNEMOSYNE_AUTHOR_ID is set AND the beam itself carries an author
    identity."""
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", HERMES_AUTHOR_TYPE)

    provider = module.MnemosyneMemoryProvider()
    provider._beam = _RecordingBeam()
    provider._agent_context = "primary"
    provider._skip_contexts = set()

    block = provider.prefetch("query for active session", session_id="session")

    assert block == ""
    assert provider._beam.last_kwargs is not None, "prefetch must reach recall()"
    assert "author_id" not in provider._beam.last_kwargs
    assert "author_type" not in provider._beam.last_kwargs
