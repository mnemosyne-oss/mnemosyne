"""Tests for the polyphonic_recall provider default.

The linear recall scorer only vector-searches episodic memory ("Uses
sqlite-vec + FTS5 for episodic, FTS5 for working" -- BeamMemory.recall).
Every memory starts in working memory, so until consolidation promotes it,
recall is keyword-only and a paraphrased question returns nothing even when
the stored fact is a strong embedding match. The Hermes providers expose
polyphonic recall as a config key so that gap can be closed; it defaults to
off (current behavior), and an explicit MNEMOSYNE_POLYPHONIC_RECALL env var
stays authoritative over both config and the default.

Following the convention in test_ab_toggles.py, the toggle is pinned at three
levels: the default, the falsy/override paths, and -- so a refactor that drops
the gate fails here instead of silently reverting -- actual recall results.

Both provider paths are covered:

1. hermes_memory_provider — legacy plugin
2. mnemosyne_hermes — pip-installable integration provider
   (`pip install -e ./integrations/hermes`, as CI does)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ENV_VAR = "MNEMOSYNE_POLYPHONIC_RECALL"

from hermes_memory_provider import MnemosyneMemoryProvider as LegacyProvider

_PIP_PROVIDER_AVAILABLE = importlib.util.find_spec("mnemosyne_hermes") is not None
if _PIP_PROVIDER_AVAILABLE:
    from mnemosyne_hermes import MnemosyneMemoryProvider as PipProvider
else:  # pragma: no cover - exercised only on installs without the integration
    PipProvider = None

# Both shipped providers are required coverage. The pip package is skipped
# *visibly* when it is not installed, rather than silently shrinking the
# parametrization to a single implementation.
PROVIDER_PARAMS = [
    pytest.param(LegacyProvider, id="hermes_memory_provider"),
    pytest.param(
        PipProvider,
        id="mnemosyne_hermes",
        marks=pytest.mark.skipif(
            not _PIP_PROVIDER_AVAILABLE,
            reason="mnemosyne_hermes not installed (pip install -e ./integrations/hermes)",
        ),
    ),
]


def _reset_provider_managed_state(provider_cls) -> None:
    """Forget the value the provider module last wrote to the env var.

    Resolved through the function's own ``__globals__`` rather than
    ``sys.modules[...]``: another test module may reload the provider, leaving
    a stale module object in ``sys.modules`` while the bound method still reads
    the original namespace. Targeting the namespace the function actually uses
    keeps this reset correct whatever else the suite has imported.
    """
    namespace = provider_cls._apply_provider_config.__globals__
    if "_provider_managed_polyphonic" in namespace:
        namespace["_provider_managed_polyphonic"] = None


@pytest.fixture(autouse=True)
def _clean_env():
    """Run each test with the gate env var unset, and restore it after."""
    original = os.environ.get(ENV_VAR)
    os.environ.pop(ENV_VAR, None)
    for param in PROVIDER_PARAMS:
        if param.values[0] is not None:
            _reset_provider_managed_state(param.values[0])
    try:
        yield
    finally:
        os.environ.pop(ENV_VAR, None)
        if original is not None:
            os.environ[ENV_VAR] = original


@pytest.mark.parametrize("provider_cls", PROVIDER_PARAMS)
class TestPolyphonicRecallDefault:
    def test_disabled_by_default(self, provider_cls):
        """With no config and no env var, current behavior is preserved."""
        provider = provider_cls()
        with patch.object(provider_cls, "_read_config_key", return_value=None):
            provider._apply_provider_config({})
        assert os.environ[ENV_VAR] == "0"

    def test_config_true_enables(self, provider_cls):
        """polyphonic_recall: true in config.yaml opts into the engine."""
        provider = provider_cls()
        with patch.object(
            provider_cls,
            "_read_config_key",
            side_effect=lambda key: True if key == "polyphonic_recall" else None,
        ):
            provider._apply_provider_config({})
        assert os.environ[ENV_VAR] == "1"

    def test_kwargs_take_precedence_over_config(self, provider_cls):
        """kwargs beat config.yaml, matching the other provider config keys."""
        provider = provider_cls()
        with patch.object(
            provider_cls,
            "_read_config_key",
            side_effect=lambda key: False if key == "polyphonic_recall" else None,
        ):
            provider._apply_provider_config({"polyphonic_recall": True})
        assert os.environ[ENV_VAR] == "1"

    @pytest.mark.parametrize(
        "raw,expected",
        [("true", "1"), ("ON", "1"), ("Yes", "1"), ("1", "1"),
         ("false", "0"), ("off", "0"), ("no", "0"), ("0", "0")],
    )
    def test_string_values_are_coerced(self, provider_cls, raw, expected):
        """String config values are parsed like the other boolean keys."""
        provider = provider_cls()
        with patch.object(
            provider_cls,
            "_read_config_key",
            side_effect=lambda key: raw if key == "polyphonic_recall" else None,
        ):
            provider._apply_provider_config({})
        assert os.environ[ENV_VAR] == expected

    @pytest.mark.parametrize("preset", ["0", "1"])
    def test_explicit_env_var_wins(self, provider_cls, preset):
        """An operator-set env var is never overwritten by the config default."""
        os.environ[ENV_VAR] = preset
        provider = provider_cls()
        with patch.object(
            provider_cls,
            "_read_config_key",
            side_effect=lambda key: (preset == "0") if key == "polyphonic_recall" else None,
        ):
            provider._apply_provider_config({})
        assert os.environ[ENV_VAR] == preset

    def test_reinitialization_applies_new_config(self, provider_cls):
        """A provider-written value is refreshed on re-init, not mistaken for
        an operator override.

        Regression: the first initialization writes the env var, so a naive
        "is it already set?" guard makes every later initialization in the same
        process a no-op -- a second profile, a session switch, or an edited
        config.yaml would silently keep the first value.
        """
        first = provider_cls()
        with patch.object(
            provider_cls,
            "_read_config_key",
            side_effect=lambda key: True if key == "polyphonic_recall" else None,
        ):
            first._apply_provider_config({})
        assert os.environ[ENV_VAR] == "1"

        second = provider_cls()
        with patch.object(
            provider_cls,
            "_read_config_key",
            side_effect=lambda key: False if key == "polyphonic_recall" else None,
        ):
            second._apply_provider_config({})
        assert os.environ[ENV_VAR] == "0"

    def test_external_override_survives_reinitialization(self, provider_cls):
        """An operator override still wins after the provider has run once."""
        first = provider_cls()
        with patch.object(provider_cls, "_read_config_key", return_value=None):
            first._apply_provider_config({})
        assert os.environ[ENV_VAR] == "0"

        # Operator intervenes mid-process (e.g. a later .env load).
        os.environ[ENV_VAR] = "1"

        second = provider_cls()
        with patch.object(provider_cls, "_read_config_key", return_value=None):
            second._apply_provider_config({})
        assert os.environ[ENV_VAR] == "1"

    def test_config_schema_advertises_the_key(self, provider_cls):
        """`hermes memory setup` can discover and document the new key."""
        schema = provider_cls().get_config_schema()
        entry = next(
            (field for field in schema if field.get("key") == "polyphonic_recall"),
            None,
        )
        assert entry is not None, "polyphonic_recall missing from get_config_schema()"
        assert entry.get("default") is False


# Retrieval coverage runs in a child process. Loading the embedding backend
# opens a large number of file descriptors and leaves module-level thread-local
# database connections behind, which perturbs suite-wide assertions elsewhere
# (tests/test_repair.py compares process-wide /proc/self/fd counts). A
# subprocess gives real end-to-end coverage while guaranteeing that none of it
# survives into the pytest process.
_RETRIEVAL_CHILD = '''
import json, os, sys

provider_module = sys.argv[1]
home = sys.argv[2]
__import__(provider_module)
Provider = sys.modules[provider_module].MnemosyneMemoryProvider

provider = Provider()
provider.initialize("polyphonic-retrieval", hermes_home=home, platform="cli",
                    agent_context="primary", agent_identity="test")


def recall_count(query):
    raw = provider.handle_tool_call("mnemosyne_recall", {"query": query, "limit": 3})
    payload = json.loads(raw) if isinstance(raw, str) else raw
    return len(payload.get("results") or [])


try:
    provider.handle_tool_call("mnemosyne_remember", {
        "content": "Casey is vegan; shared meals need genuinely vegan options.",
        "source": "fact", "importance": 0.9, "scope": "global"})

    paraphrase = "plant based diet, avoid animal products"
    # recall() reads the gate per call, so it can be toggled directly.
    os.environ["MNEMOSYNE_POLYPHONIC_RECALL"] = "1"
    result = {"keyword": recall_count("vegan"), "enabled": recall_count(paraphrase)}
    os.environ["MNEMOSYNE_POLYPHONIC_RECALL"] = "0"
    result["disabled"] = recall_count(paraphrase)
finally:
    provider.shutdown()

print("RESULT:" + json.dumps(result))
'''


@pytest.mark.parametrize("provider_cls", PROVIDER_PARAMS)
class TestPolyphonicRecallRetrieval:
    """The toggle must surface in real recall results, not just an env var."""

    @staticmethod
    def _run_child(provider_cls, home, repo_root) -> dict:
        import json
        import subprocess

        env = dict(os.environ)
        env["HERMES_HOME"] = str(home)
        env["MNEMOSYNE_DATA_DIR"] = str(home / "mnemosyne" / "data")
        env.pop("MNEMOSYNE_POLYPHONIC_RECALL", None)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in [str(repo_root), str(repo_root / "integrations" / "hermes" / "src"),
                        env.get("PYTHONPATH", "")] if p
        )
        completed = subprocess.run(
            [sys.executable, "-c", _RETRIEVAL_CHILD, provider_cls.__module__, str(home)],
            capture_output=True, text=True, env=env, timeout=600,
        )
        marker = [ln for ln in completed.stdout.splitlines() if ln.startswith("RESULT:")]
        assert marker, (
            f"retrieval child failed (rc={completed.returncode})\n"
            f"stdout:\n{completed.stdout[-2000:]}\nstderr:\n{completed.stderr[-2000:]}"
        )
        return json.loads(marker[-1][len("RESULT:"):])

    def test_paraphrase_retrieves_unconsolidated_memory(self, provider_cls, tmp_path):
        """Opting in makes a working-memory fact reachable by paraphrase.

        The opt-out preserves the linear path, where the vector voice never
        covers working memory and only literal keyword overlap matches. Both
        directions are asserted so a refactor that drops the gate fails here.
        """
        embeddings = pytest.importorskip(
            "mnemosyne.core.embeddings", reason="embeddings extra not installed"
        )
        if not embeddings.available():
            pytest.skip("embeddings backend unavailable")

        # HERMES_HOME is pinned for the child too: the hermes_home kwarg alone
        # is not enough, because parts of the database path resolve through the
        # environment, which would read and write a real Mnemosyne store.
        home = tmp_path / "hermes_home"
        home.mkdir()
        repo_root = Path(__file__).resolve().parents[1]
        counts = self._run_child(provider_cls, home, repo_root)

        # Literal keyword overlap works either way -- it is the FTS5 path.
        assert counts["keyword"] >= 1

        # The paraphrase shares no content words with the stored memory.
        assert counts["enabled"] >= 1, (
            "paraphrased query should retrieve the unconsolidated memory "
            "with polyphonic recall enabled"
        )

        # Opt-out (the default) keeps the current linear behavior.
        assert counts["disabled"] == 0
