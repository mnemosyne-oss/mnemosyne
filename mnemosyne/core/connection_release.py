"""Release a thread's cached SQLite handles (mnemosyne#1163).

Explicit, opt-in inverse of ``_get_connection``. This module exists so the
release can be reached without importing ``mnemosyne.core.memory``: that module
runs ``init_db()`` while it is being imported, which opens a connection for the
default database and republishes it into this thread's ``mnemosyne.core.beam``
cache. A release helper that imported it therefore inspected a cache the import
had already rewritten -- the original worker handle survived while the helper
reported success (matching default path) or nothing was closed at all
(different default path).

Everything here works off caches that already exist, looked up in
``sys.modules``, so this import can neither create nor displace a handle. Keep
it free of imports that touch SQLite.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_db_path() -> Path:
    """Return the default DB path without importing ``mnemosyne.core.memory``.

    ``mnemosyne.core.beam`` owns an identical resolver and performs no database
    work at import time, so it is the safe source for this value.
    """
    from mnemosyne.core.beam import _default_db_path as beam_default_db_path

    return beam_default_db_path()


def _cached_connection_caches() -> list:
    """Return the thread-local caches that may hold a handle on this thread.

    Modules are read from ``sys.modules`` rather than imported: a module that
    was never imported cannot hold a handle for this thread, and importing it
    is exactly the side effect this helper exists to avoid.
    """
    caches = []
    for module_name in ("mnemosyne.core.beam", "mnemosyne.core.memory"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        cache = getattr(module, "_thread_local", None)
        if cache is not None:
            caches.append(cache)
    return caches


def close_thread_connection(db_path=None, *, allow_main_thread: bool = False) -> bool:
    """Close this thread's cached SQLite connection.

    Explicit, opt-in inverse of ``_get_connection``. It exists for callers that
    can *assert* no consumer will use the cached handle again: a short-lived
    worker thread that opened the connection, finished with it, and has
    abandoned every ``Mnemosyne``/``BeamMemory`` instance and sub-store alias
    (``annotations``, ``canonical``, ``media``, ``episodic_graph``,
    ``veracity_consolidator``) built on the same handle.

    "Abandoned" means no further use, not that no reference exists. A local
    variable may still point at the instance graph after the worker's last
    statement -- for example while a ``finally`` block runs -- provided that
    graph is irrevocably retired. The release also assumes no open cursor,
    statement-cache entry or deferred work will touch the handle afterwards,
    and it covers every thread-local cache to which ``_get_connection``
    publishes (``mnemosyne.core.memory`` and ``mnemosyne.core.beam``).

    The library cannot verify that assertion. Connections are created with
    ``check_same_thread=False`` and instances keep ``self.conn``, so a retained
    instance may legitimately outlive the thread that created it. Closing on
    thread exit, cache displacement or collection would therefore release a
    handle a live alias may still use. Only an explicit call from the owner may
    release it (mnemosyne#1163).

    Calling this never imports ``mnemosyne.core.memory``, so cleanup cannot run
    that module's import-time ``init_db()`` and displace the handle it is about
    to close.

    Args:
        db_path: Database whose cached connection to close. Defaults to the
            configured default database; a cached connection for any other
            path is left untouched.
        allow_main_thread: Refuse on the main thread by default, because that
            cache is process-lifetime and shared with long-lived singletons.

    Returns:
        True if at least one cached connection was closed. False if this
        thread held no matching connection, the call was refused, or every
        matching close failed. A handle whose ``close()`` raised is left in its
        cache: it is still usable, so dropping the reference here would discard
        the caller's only retry authority. A later call may try again.
    """
    if threading.current_thread() is threading.main_thread() and not allow_main_thread:
        logger.debug("close_thread_connection refused on the main thread")
        return False

    path = (Path(db_path) if db_path else _default_db_path()).expanduser().resolve()
    wanted = str(path)

    closed = False
    # _get_connection publishes core's handle into beam's cache, so one object
    # can be reachable from both thread-locals. Close is idempotent per handle.
    for cache in _cached_connection_caches():
        connection = getattr(cache, "conn", None)
        if connection is None or getattr(cache, "db_path", None) != wanted:
            continue
        try:
            connection.close()
        except Exception:
            # A failed close (for example a subclass raising from an overridden
            # close) leaves the handle usable. Keeping the cache entry preserves
            # the only retry authority; clearing it would hide a live handle.
            logger.warning("cached connection close failed; handle retained", exc_info=True)
            continue
        closed = True
        cache.conn = None
        cache.db_path = None
    return closed
