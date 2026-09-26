"""Runtime resolution of the Mnemosyne data directory.

These helpers read the environment on every call. Import-time snapshots of
HERMES_HOME go stale when a process rebinds the active profile.
"""

import os
from pathlib import Path


def default_root() -> Path:
    """Return the active data directory.

    Evaluated per call:

    1. ``MNEMOSYNE_DATA_DIR`` if set and non-empty. That value is the data dir.
    2. Otherwise ``<HERMES_HOME>/mnemosyne/data`` when ``HERMES_HOME`` is set
       and non-empty.
    3. Otherwise ``~/.hermes/mnemosyne/data``.
    """
    override = os.environ.get("MNEMOSYNE_DATA_DIR")
    if override:
        return Path(override)
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
