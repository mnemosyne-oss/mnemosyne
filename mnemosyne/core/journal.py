"""SQLite journal-mode selection for store connections.

Default is WAL, unchanged from historical behavior. Deployments on
filesystems where WAL is unsafe (notably Linux containers on macOS
virtiofs, where WAL readback intermittently surfaces as
"database disk image is malformed") can set MNEMOSYNE_JOURNAL_MODE
(e.g. ``delete``) to override every connection mnemosyne opens.
"""

from __future__ import annotations

import os

#: Values sqlite3 accepts for PRAGMA journal_mode (case-insensitive).
_VALID_MODES = {"delete", "truncate", "persist", "memory", "wal", "off"}


def journal_mode() -> str:
    """Return the journal mode to set on store connections.

    Reads MNEMOSYNE_JOURNAL_MODE from the environment; falls back to
    "wal" when unset or not a valid sqlite journal mode (invalid values
    are ignored, keeping historical behavior rather than failing open
    on a typo).
    """
    mode = os.environ.get("MNEMOSYNE_JOURNAL_MODE", "").strip().lower()
    return mode if mode in _VALID_MODES else "wal"
