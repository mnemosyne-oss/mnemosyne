"""Persistent recall provenance (query -> returned memory ids).

`recall_diagnostics` (mnemosyne/core/recall_diagnostics.py) keeps
in-process counters: per-tier hit counts and fallback rates over a
measurement window. Those counters answer "where does recall signal
come from?" but are lost when the process exits and never record
WHICH query pulled WHICH memories.

This module complements them with a persistent, queryable mapping:
when `BeamMemory.recall()` runs with MNEMOSYNE_RECALL_PROVENANCE=1,
one JSONL line is appended to `<db_path>.recall_provenance.jsonl`,
one file per database, recording the query and the returned memory
ids with their scores. Operators can later audit which memories
shaped a given answer ("why did the agent say X?"). Records stay
lean: ids and scores only, no content previews.

Latency and failure contract: provenance is default-OFF and the flag
is read per call. Every failure is swallowed and logged at debug
level (fail-open: a provenance problem never breaks recall). When
enabled, the cost is one small locked append per call: the record is
bounded, the lock is per store (recalls against different databases
never serialize against each other), there is no fsync, and the whole
line goes out in a single os.write on an O_APPEND fd, which keeps
records intact across processes on local Linux filesystems
(interleaving can only happen BETWEEN records, never within one).
NFS and other network filesystems are not supported.

Coverage note: the hook lives at the single final return of the
LINEAR recall path in beam.py. The `recall_enhanced()` and
polyphonic (MNEMOSYNE_POLYPHONIC_RECALL=1) delegation paths return
before reaching it and are NOT logged. Calls with `explain=True` are
ALSO not logged, and that is intentional: an explain call returns its
own trace object, which is the audit surface for that call, so a
provenance line would duplicate semantics rather than add them.

Lifecycle: the audit file can outlive its database. Call
`cleanup_orphaned_provenance()` at BeamMemory init/upgrade time to
remove it when the db file is gone (not wired into the recall path).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)

# One lock per provenance file, created on demand. The meta lock only
# guards the dict itself, never an append.
_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

# Hard cap on stored query text. Queries can be long prompts; the
# full text lives with the caller, the audit trail only needs to
# identify the call.
_MAX_QUERY_CHARS = 200

# The audit trail records the head of the result list, nothing more.
_MAX_RESULTS_PER_RECORD = 20

# Rotation threshold for the current generation. When the file has
# reached this size it is renamed to `<file>.1` (overwriting any
# previous `.1`) before the next append, so retention is the current
# file plus one rotated generation, hard-capped at roughly 2 MB.
_MAX_FILE_BYTES = 1_000_000

# The reader never scans the whole file: at most this many bytes of
# the CURRENT generation (a rotated `.1` is not read) are examined.
_MAX_TAIL_BYTES = 262_144

# Upper bound on records parsed per read, regardless of `limit`.
_MAX_READ_RECORDS = 10_000

# 0600 survives any umask: there are no group/other bits left to
# strip, so the file is private no matter how permissive the umask.
_FILE_MODE = 0o600

RESULT_FIELDS = ("id", "tier", "score", "importance", "timestamp")


def _provenance_path(db_path: Any) -> Path:
    """Provenance file for one database: `<db>.recall_provenance.jsonl`.

    Per-database by name, so two databases in the same directory never
    share an audit file.
    """
    return Path(str(db_path) + ".recall_provenance.jsonl")


def _lock_for(path_str: str) -> threading.Lock:
    """Return the append lock for one provenance file."""
    with _locks_guard:
        lock = _locks.get(path_str)
        if lock is None:
            lock = threading.Lock()
            _locks[path_str] = lock
        return lock


def append_recall_provenance(db_path: Any, query: str,
                             results: List[Dict], top_k: int) -> None:
    """Append one JSONL provenance record for a recall() call.

    Writes `{"ts", "query", "top_k", "results"}` to
    `<db_path>.recall_provenance.jsonl`, where each entry in `results`
    carries only id/tier/score/importance/timestamp (no content
    preview); only the first 20 results are recorded. Never raises:
    failures are logged at debug level and swallowed so recall
    behavior is unaffected.

    Privacy note: the JSONL file is plaintext on local disk, is
    created 0600, and stores up to 200 chars of raw query text per
    call. Retention is bounded by rotation: at 1 MB the current file
    becomes `.1` (overwriting the previous one), so at most two
    generations exist. Operators control deletion.
    """
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": str(query or "")[:_MAX_QUERY_CHARS],
            "top_k": int(top_k),
            "results": [
                {field: r.get(field) for field in RESULT_FIELDS}
                for r in (results or [])[:_MAX_RESULTS_PER_RECORD]
                if isinstance(r, dict)
            ],
        }
        data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        path = _provenance_path(db_path)
        path_str = str(path)
        with _lock_for(path_str):
            fd = os.open(path_str, os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                         _FILE_MODE)
            try:
                needs_rotation = os.fstat(fd).st_size >= _MAX_FILE_BYTES
            finally:
                os.close(fd)
            if needs_rotation:
                # os.replace overwrites an existing `.1`: a single
                # rotated generation, never .2, .3, ...
                os.replace(path_str, path_str + ".1")
            fd = os.open(path_str, os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                         _FILE_MODE)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
    except Exception:
        logger.debug("recall provenance append failed (non-fatal)", exc_info=True)


def cleanup_orphaned_provenance(db_path: Any) -> bool:
    """Delete the provenance file when its database is gone.

    Call at BeamMemory init/upgrade time. Removes the current file and
    its `.1` rotation only when the provenance file exists and the db
    file does not. Returns True when files were removed, False when
    nothing needed removal or cleanup failed (best effort).
    """
    path = _provenance_path(db_path)
    try:
        if Path(db_path).exists():
            return False
        rotated = Path(str(path) + ".1")
        if not path.exists() and not rotated.exists():
            return False
        if path.exists():
            path.unlink()
        if rotated.exists():
            rotated.unlink()
        return True
    except Exception:
        logger.debug("recall provenance cleanup failed (non-fatal)", exc_info=True)
        return False


def read_recall_provenance(db_path: Any, limit: int = 20) -> List[Dict]:
    """Read provenance records, newest first.

    Reads only the last `_MAX_TAIL_BYTES` bytes of the CURRENT
    generation; a rotated `.1` file is never read. When the file is
    larger than that window the first line of the window may be
    partial and is dropped. At most `_MAX_READ_RECORDS` records are
    parsed regardless of `limit`; `limit` is a cap on what is
    returned, not a guarantee that that many exist. Malformed lines
    are skipped. Missing file or any read failure -> [] (debug
    logged).
    """
    path = _provenance_path(db_path)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _MAX_TAIL_BYTES))
            lines = f.read().splitlines()
    except Exception:
        logger.debug("recall provenance read failed (non-fatal)", exc_info=True)
        return []
    if size > _MAX_TAIL_BYTES and lines:
        # The window starts mid-line: the first line is partial.
        lines = lines[1:]
    cap = min(max(0, int(limit)), _MAX_READ_RECORDS)
    records: List[Dict] = []
    for line in reversed(lines):
        if len(records) >= cap:
            break
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    # Reversed iteration already yields newest first.
    return records
