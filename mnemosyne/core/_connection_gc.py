"""Bound delayed SQLite cleanup under thread-local connection churn."""

import gc
import threading


_CONNECTIONS_PER_COLLECTION = 8
_connection_count = 0
_connection_count_lock = threading.Lock()


def collect_connection_cycles() -> None:
    """Run full cyclic collection after bounded SQLite connection churn.

    sqlite3 connections displaced by thread exit or a database-path switch can
    remain in cyclic garbage until a full collection. ``gc.collect()`` is
    process-wide, so this also collects other unreachable cycles and can add
    occasional heap-size-dependent tail latency. Running it periodically
    prevents descriptor exhaustion without closing connections still referenced
    by live objects.
    """
    global _connection_count

    with _connection_count_lock:
        _connection_count += 1
        if _connection_count < _CONNECTIONS_PER_COLLECTION:
            return
        _connection_count = 0

    gc.collect()
