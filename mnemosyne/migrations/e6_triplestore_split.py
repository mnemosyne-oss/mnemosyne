"""
Mnemosyne E6 Migration — TripleStore Split
==========================================

Idempotent migration that moves annotation-flavored rows out of the
legacy `triples` table and into the new `annotations` table. Fixes the
silent-destruction bug where adding multiple mentions / facts for one
memory silently invalidated prior rows via `(subject, predicate)`
auto-invalidation.

This module ships INSIDE the package so pip-installed deployments can
auto-migrate on first BeamMemory init. The CLI wrapper at
`scripts/migrate_triplestore_split.py` imports from here.

Behavior
--------
1. Detect rows in `triples` whose predicate is in ANNOTATION_KINDS
   (mentions, fact, occurred_on, has_source).
2. Insert each row into `annotations` using the (memory_id, kind, value)
   mapping (subject → memory_id, predicate → kind, object → value).
3. Skip rows that already have a corresponding row in `annotations`
   (idempotent — safe to re-run).
4. Do NOT delete the source rows from `triples`. They remain as legacy
   data; `annotations` is the canonical store post-migration. This
   makes the migration reversible — restore the DB from backup if needed.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from mnemosyne.core.annotations import ANNOTATION_KINDS, init_annotations


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cursor.fetchone() is not None


def has_pending_migration(conn: sqlite3.Connection) -> bool:
    """Cheap check: is there ANY annotation-flavored row in triples that's
    not already present in annotations?

    Used by the auto-migrate hook in BeamMemory to short-circuit on every
    init without doing the full classify scan. SQL anti-join lands on the
    `(kind, value)` and `(memory_id, kind)` indexes on the annotations
    table; the only work is matching index probes per candidate row, and
    `LIMIT 1` aborts on the first hit.
    """
    if not _has_table(conn, "triples"):
        return False

    placeholders = ",".join("?" * len(ANNOTATION_KINDS))

    # If the annotations table doesn't exist yet, any annotation-flavored
    # row in triples needs migration.
    if not _has_table(conn, "annotations"):
        cursor = conn.execute(
            f"SELECT 1 FROM triples WHERE predicate IN ({placeholders}) LIMIT 1",
            tuple(ANNOTATION_KINDS),
        )
        return cursor.fetchone() is not None

    cursor = conn.execute(
        f"""
        SELECT 1
        FROM triples t
        WHERE t.predicate IN ({placeholders})
          AND NOT EXISTS (
              SELECT 1 FROM annotations a
              WHERE a.memory_id = t.subject
                AND a.kind = t.predicate
                AND a.value = t.object
          )
        LIMIT 1
        """,
        tuple(ANNOTATION_KINDS),
    )
    return cursor.fetchone() is not None


def _classify_rows(
    conn: sqlite3.Connection,
) -> Tuple[List[sqlite3.Row], int]:
    """Return (rows-to-migrate, total-triples-row-count).

    Rows-to-migrate are those whose predicate is in ANNOTATION_KINDS and
    that do not already have a matching row in `annotations`.
    """
    if not _has_table(conn, "triples"):
        return [], 0

    conn.row_factory = sqlite3.Row

    # Total for reporting
    total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    # Candidates: annotation-flavored predicates only
    placeholders = ",".join("?" * len(ANNOTATION_KINDS))
    candidates = conn.execute(
        f"""
        SELECT id, subject, predicate, object, source, confidence, created_at
        FROM triples
        WHERE predicate IN ({placeholders})
        ORDER BY id ASC
        """,
        tuple(ANNOTATION_KINDS),
    ).fetchall()

    # Filter out those already migrated.
    # Idempotency key: (memory_id=subject, kind=predicate, value=object).
    # Tuple identity here is good enough; we're matching one-to-one across
    # the data move.
    if not _has_table(conn, "annotations"):
        return list(candidates), total

    existing_keys = set()
    for row in conn.execute(
        "SELECT memory_id, kind, value FROM annotations"
    ).fetchall():
        existing_keys.add((row[0], row[1], row[2]))

    needs_migration = [
        row
        for row in candidates
        if (row["subject"], row["predicate"], row["object"]) not in existing_keys
    ]
    return needs_migration, total


def _migrate_rows(
    conn: sqlite3.Connection, rows: List[sqlite3.Row]
) -> int:
    """Insert rows into annotations. Returns count written.

    Uses executemany for first-run migrations on databases with thousands
    of annotation-flavored triples rows — meaningfully faster than per-row
    execute() for large legacy datasets.
    """
    if not rows:
        return 0

    params = [
        (
            row["subject"],
            row["predicate"],
            row["object"],
            row["source"],
            row["confidence"] if row["confidence"] is not None else 1.0,
            row["created_at"],
        )
        for row in rows
    ]

    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO annotations
            (memory_id, kind, value, source, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        params,
    )
    return len(params)


def _kind_counts(rows: List[sqlite3.Row]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        kind = row["predicate"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def migrate(
    db_path: Path,
    dry_run: bool = False,
    backup: bool = True,
    log_fn=print,
) -> int:
    """Run the migration. Returns the number of rows migrated.

    - `dry_run=True` reports what would change without writing.
    - `backup=True` copies the DB file to `{db}.pre_e6_backup` first.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log_fn(f"ERROR: database not found: {db_path}")
        raise FileNotFoundError(db_path)

    # Pre-flight inspection (read-only)
    conn = sqlite3.connect(str(db_path))
    try:
        rows, total = _classify_rows(conn)
    finally:
        conn.close()

    log_fn(f"Database: {db_path}")
    log_fn(f"  triples rows (total):        {total}")
    log_fn(f"  rows-to-migrate (this run):  {len(rows)}")
    if rows:
        for kind, count in sorted(_kind_counts(rows).items()):
            log_fn(f"    {kind:<14} {count}")

    if not rows:
        log_fn("Nothing to migrate. Schema is already split or no annotation rows exist.")
        return 0

    if dry_run:
        log_fn("Dry run: no changes written.")
        return len(rows)

    # Backup (file-level copy)
    if backup:
        backup_path = db_path.with_suffix(db_path.suffix + ".pre_e6_backup")
        if backup_path.exists():
            # Don't overwrite an existing backup — that's likely from an
            # earlier migration attempt and is more valuable than the
            # current DB state.
            log_fn(
                f"Backup already exists at {backup_path}; leaving as-is."
            )
        else:
            shutil.copy2(db_path, backup_path)
            log_fn(f"Backup written to {backup_path}")

    # Transactional write. busy_timeout matches BeamMemory's connection
    # so concurrent writers don't immediately fail with SQLITE_BUSY.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        init_annotations(db_path)
        conn.execute("BEGIN")
        try:
            conn.row_factory = sqlite3.Row
            written = _migrate_rows(conn, rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    log_fn(f"Migration complete: {written} rows moved to annotations table.")
    return written
