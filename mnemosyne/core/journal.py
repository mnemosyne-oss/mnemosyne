"""SQLite journal-mode selection for store connections.

Default is WAL, unchanged from historical behavior. Deployments on
filesystems where WAL is unsafe (notably Linux containers on macOS
virtiofs, where WAL readback intermittently surfaces as
"database disk image is malformed") can set MNEMOSYNE_JOURNAL_MODE
(e.g. ``delete``) to override every connection mnemosyne opens.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Values sqlite3 accepts for PRAGMA journal_mode (case-insensitive).
_VALID_MODES = {"delete", "truncate", "persist", "memory", "wal", "off"}


def journal_mode() -> str:
    """Return the journal mode to set on store connections.

    Reads MNEMOSYNE_JOURNAL_MODE from the environment; falls back to
    "wal" when unset. Invalid values are ignored (with a warning: this
    override exists for filesystems where WAL corrupts reads, so
    silently reverting to WAL on a typo would restore exactly the
    failure the deployment set the variable to escape).

    Connections that do not set a journal mode inherit the mode
    persisted in the database file, so one honoring connection flips
    the whole store.
    """
    raw = os.environ.get("MNEMOSYNE_JOURNAL_MODE")
    if raw is None or not raw.strip():
        return "wal"
    mode = raw.strip().lower()
    if mode in _VALID_MODES:
        return mode
    logger.warning(
        "Ignoring invalid MNEMOSYNE_JOURNAL_MODE=%r (valid: %s); "
        "falling back to wal",
        raw, ", ".join(sorted(_VALID_MODES)),
    )
    return "wal"
