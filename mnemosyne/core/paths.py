"""Runtime resolution of the Mnemosyne data directory.

These helpers read the environment on every call. Import-time snapshots of
HERMES_HOME go stale when a process rebinds the active profile.

``default_root`` / ``default_data_dir`` resolve, per call and without caching:

1. ``MNEMOSYNE_DATA_DIR`` if set and non-empty. That value is the data dir.
2. A resolver registered with ``register_data_dir_resolver``, when one is set
   and returns a truthy path or string. The result is coerced with ``Path``.
   A hook that raises, or returns a falsy value, is ignored. The hook exists
   so a host application can supply a context-local scope the environment
   cannot express.
3. ``<HERMES_HOME>/mnemosyne/data`` when ``HERMES_HOME`` is set and non-empty.
4. ``~/.hermes/mnemosyne/data``.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_Resolver = Callable[[], Path | str | None]
_RESOLVER: _Resolver | None = None


def register_data_dir_resolver(fn: _Resolver | None) -> _Resolver | None:
    """Install ``fn`` as the data-dir hook. ``None`` clears it.

    Returns the previously registered resolver, or ``None``.
    """
    global _RESOLVER
    previous = _RESOLVER
    _RESOLVER = fn
    return previous


def _resolver_data_dir() -> Path | None:
    """Return the hook's path, or ``None`` to fall through to the env chain."""
    resolver = _RESOLVER
    if resolver is None:
        return None
    try:
        result = resolver()
    except Exception:
        logger.debug("data-dir resolver failed; using the environment chain", exc_info=True)
        return None
    if not result:
        return None
    return Path(result)


def default_root() -> Path:
    """Return the active data directory.

    Evaluated per call. Order is the module docstring: ``MNEMOSYNE_DATA_DIR``,
    then the registered resolver, then ``HERMES_HOME``, then ``~/.hermes``.
    The hook exists so a host application can supply a context-local scope
    the environment cannot express.
    """
    override = os.environ.get("MNEMOSYNE_DATA_DIR")
    if override:
        return Path(override)
    hooked = _resolver_data_dir()
    if hooked is not None:
        return hooked
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home) / "mnemosyne" / "data"
    return Path.home() / ".hermes" / "mnemosyne" / "data"


def default_data_dir() -> Path:
    """Return the active data directory. Same order as ``default_root``."""
    return default_root()


def default_db_path() -> Path:
    """Return ``<data dir>/mnemosyne.db`` for the active data directory."""
    return default_data_dir() / "mnemosyne.db"


def _hermes_home() -> Path:
    """Hermes home behind the legacy ``_DEFAULT_ROOT`` name.

    ``MNEMOSYNE_DATA_DIR`` does not change this. The old override rewrote only
    the data directory and the database path.
    """
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home)
    return Path.home() / ".hermes"
