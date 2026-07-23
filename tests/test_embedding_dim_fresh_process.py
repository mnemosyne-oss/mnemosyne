"""Fresh-process coverage for the fail-loud embedding-dimension resolver.

The helper-level tests in ``test_embeddings_multilingual.py`` call
``_get_embedding_dim`` *after* ``mnemosyne.core.embeddings`` is already
imported, so they do not prove the actionable error fires at the
import/startup boundary -- before any ``vec0`` DDL or vector write. These
tests spawn a fresh interpreter per case so the failure is observed exactly
where an operator would see it: at process startup.

They also assert parity: direct core and both Hermes provider surfaces
(``hermes_memory_provider`` and ``mnemosyne_hermes``) surface the same
actionable configuration failure for an unknown model with no explicit
``MNEMOSYNE_EMBEDDING_DIM``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"

_ERROR_MARKERS = ("Unknown embedding model", "MNEMOSYNE_EMBEDDING_DIM")


def _run_fresh(code: str, tmp_path: Path, *, pythonpath: str | None = None, **env_overrides: str):
    """Run ``code`` in a fresh interpreter with an isolated data dir.

    ``MNEMOSYNE_EMBEDDING_DIM`` is removed so the unknown-model path is
    exercised (no explicit override). The data dir points under ``tmp_path``
    so we can assert no database was written.
    """
    env = os.environ.copy()
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "data")
    env["HOME"] = str(tmp_path / "home")
    env.pop("MNEMOSYNE_EMBEDDING_DIM", None)
    # Strip embedding-disable flags so the subprocess exercises the fail-loud
    # path, not the _is_disabled() 384 fallback (these are set in some CI).
    for _flag in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        env.pop(_flag, None)
    if pythonpath:
        # Prepend so the provider package resolves before any installed copy.
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _assert_fail_fast(result: subprocess.CompletedProcess, tmp_path: Path) -> None:
    """The process crashed at import with the actionable error, before any DB write."""
    assert result.returncode != 0, result.stderr
    assert any(marker in result.stderr for marker in _ERROR_MARKERS), result.stderr
    assert not list((tmp_path / "data").rglob("*.db")), "a database was written before the error"


@pytest.mark.parametrize(
    "model,extra_env",
    [
        pytest.param("some/unknown-local-model", {}, id="unknown-local"),
        pytest.param(
            "custom-endpoint-model",
            {"MNEMOSYNE_EMBEDDING_API_URL": "http://localhost:8000/v1"},
            id="custom-endpoint",
        ),
        pytest.param("openai/text-embedding-fake", {}, id="api-model"),
    ],
)
def test_unknown_model_fails_at_import_before_any_write(tmp_path, model, extra_env):
    """A fresh process with an unknown embedding model (local, custom-endpoint,
    or API) and no explicit dimension must fail at import -- before any vec0 DDL
    or vector write -- with the actionable error, not silently assume 384."""
    result = _run_fresh(
        "from mnemosyne.core import beam",
        tmp_path,
        MNEMOSYNE_EMBEDDING_MODEL=model,
        **extra_env,
    )
    _assert_fail_fast(result, tmp_path)


@pytest.mark.parametrize(
    "code,pythonpath,fail_fast",
    [
        pytest.param("from mnemosyne.core import beam", None, True, id="direct-core"),
        pytest.param("import hermes_memory_provider", str(PROJECT_ROOT), True, id="hermes_memory_provider"),
        pytest.param("import mnemosyne_hermes", str(INTEGRATION_SRC), False, id="mnemosyne_hermes"),
    ],
)
def test_unknown_model_parity_across_surfaces(tmp_path, code, pythonpath, fail_fast):
    """Direct core and both Hermes provider surfaces expose the actionable
    configuration failure for an unknown model with no explicit dim.

    Direct core and ``hermes_memory_provider`` fail fast (non-zero exit at
    import). ``mnemosyne_hermes`` graceful-degrades by design: it catches the
    configuration error, logs it with the actionable message, and disables the
    affected batch tools rather than crashing the provider. Every surface must
    expose the actionable error; the two fail-fast surfaces must also exit
    non-zero, so a future regression that swallows the error is caught.
    """
    result = _run_fresh(
        code,
        tmp_path,
        pythonpath=pythonpath,
        MNEMOSYNE_EMBEDDING_MODEL="some/unknown-model",
    )
    assert any(marker in result.stderr for marker in _ERROR_MARKERS), result.stderr
    if fail_fast:
        assert result.returncode != 0, result.stderr
