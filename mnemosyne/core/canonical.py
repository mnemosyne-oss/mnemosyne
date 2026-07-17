"""
Mnemosyne CanonicalStore
========================
Owner-scoped single-source-of-truth ("canonical") facts.

Motivation (issue #256)
-----------------------
Long-running companion personas need *identity consistency over time*: a
persona's stable self-facts ("my name is X", "I speak in Y register") must not
contradict themselves months apart. Stored as ordinary working/episodic
entries those facts have no uniqueness guarantee — restating one accumulates
near-duplicates, and a later change coexists with the stale value rather than
superseding it, so recall surfaces conflicting copies.

The TripleStore already solves this for clean relational ``(subject, predicate,
object)`` facts via ``valid_until`` supersession. CanonicalStore brings the same
single-current-truth discipline to *free-text identity cards* that do not reduce
to one S-P-O triple, and adds the missing **owner scope**: a
``UNIQUE(owner_id, category, name)`` slot that holds exactly one current value.

Two orthogonal axes
-------------------
"Canonical-ness" (a uniqueness/upsert discipline) is independent of
"shared-ness" (who can read it). CanonicalStore fills the *private + canonical*
cell:

    |                     | free-text / episodic        | canonical (single truth)  |
    | private (one owner) | per-profile lived recall    | CanonicalStore (this file)|
    | shared (all owners) | the shared surface          | (out of scope)            |

Owner isolation is automatic: every read and write is keyed by ``owner_id``, so
two personas each get their own namespace and a non-persona profile simply has
none. They can still exchange memory through the shared surface, which this does
not touch.

Design
------
- **One table + a partial unique index** — no new dependency, no FTS table.
  ``canonical_facts`` holds both the current value and its superseded history;
  a partial unique index over ``(owner_id, category, name) WHERE valid_until IS
  NULL`` guarantees exactly one *current* row per slot while letting historical
  rows accumulate in the same table (mirrors the TripleStore ``valid_until``
  pattern, with an owner dimension added).
- **Upsert-in-place** — ``remember`` closes the prior current row (stamps
  ``valid_until``) and inserts the new value with ``version + 1``. Re-storing an
  identical body is a no-op, so restating a stable fact never accumulates
  duplicates.
- **Point lookup** — ``recall`` is an indexed ``(owner_id, category, name)``
  read of the single current row; cheaper than hybrid vector search for a
  known-key identity read.

This complements, not replaces, episodic memory and the TripleStore: relational
facts still belong in triples; free-text identity cards get a deduped
authoritative slot here.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# Default DB location. Mirrors BeamMemory's default (the main mnemosyne.db) so a
# standalone CanonicalStore() lands in the same database the provider wires it
# into via a shared connection. Resolved at call time so MNEMOSYNE_DATA_DIR is
# honored without importing beam (avoids a circular import).
def _default_db_path() -> Path:
    data_dir = os.environ.get("MNEMOSYNE_DATA_DIR")
    base = Path(data_dir) if data_dir else (Path.home() / ".hermes" / "mnemosyne" / "data")
    return base / "mnemosyne.db"


def _now() -> str:
    """ISO timestamp used for valid_from / valid_until. Second precision is
    enough for an identity store and keeps history rows human-readable."""
    return datetime.now().isoformat(timespec="seconds")


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_canonical(db_path: Optional[Path] = None) -> None:
    """Create the canonical_facts table and indexes if absent.

    Idempotent. Safe on databases that already have the table. Opens and
    closes its own connection; does not leak file descriptors.
    """
    conn = _get_conn(db_path)
    try:
        _init_canonical_with_conn(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _init_canonical_with_conn(conn: sqlite3.Connection) -> None:
    """Run schema DDL on an existing connection. Caller owns conn lifetime."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS canonical_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            body TEXT NOT NULL,
            source TEXT,
            confidence REAL DEFAULT 1.0,
            version INTEGER NOT NULL DEFAULT 1,
            valid_from TEXT NOT NULL,
            valid_until TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Exactly-one-current-value-per-slot. A PARTIAL unique index (WHERE
    # valid_until IS NULL) is what reconciles "single source of truth" with
    # "keep history in the same table": only the live row participates in the
    # constraint, superseded rows (valid_until set) are unconstrained. Using a
    # unique INDEX rather than a table constraint lets pre-existing tables
    # acquire the guarantee on next init via IF NOT EXISTS — no migration.
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_current "
        "ON canonical_facts(owner_id, category, name) WHERE valid_until IS NULL"
    )
    # Slot history reads ("all versions of this fact") and point lookups.
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_canonical_slot "
        "ON canonical_facts(owner_id, category, name)"
    )
    # Owner/category listing ("the persona's whole self-card set").
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_canonical_owner_category "
        "ON canonical_facts(owner_id, category)"
    )

    conn.commit()


class CanonicalStore:
    """
    Owner-scoped single-source-of-truth facts.

    Each ``(owner_id, category, name)`` slot holds exactly one current value;
    re-storing the same body is a no-op, and storing a new body supersedes the
    old one (preserved as history).

    Example:
        >>> store = CanonicalStore()
        >>> store.remember("jessi", "identity", "name", "My name is Jessi.")
        >>> store.remember("jessi", "identity", "name", "My name is Jessi.")  # no-op
        >>> store.recall("jessi", "identity", "name")["body"]
        'My name is Jessi.'
        >>> store.remember("jessi", "identity", "name", "I go by Jess now.")  # supersede
        >>> store.recall("jessi", "identity", "name")["body"]
        'I go by Jess now.'
        >>> [h["body"] for h in store.history("jessi", "identity", "name")]
        ['I go by Jess now.', 'My name is Jessi.']

    Owner isolation is enforced on every method — a read or write for one
    ``owner_id`` never sees or touches another's rows.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        conn: Optional[sqlite3.Connection] = None,
    ):
        """Create a CanonicalStore handle.

        When ``conn`` is provided the store reuses that connection — this is how
        BeamMemory shares its thread-local connection with the store, avoiding a
        per-call file-descriptor cost. The caller owns the connection's
        lifetime. When ``conn`` is None, CanonicalStore opens its own connection.
        """
        self.db_path = db_path or _default_db_path()
        if conn is not None:
            self.conn = conn
            _init_canonical_with_conn(conn)
        else:
            init_canonical(self.db_path)
            self.conn = _get_conn(self.db_path)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def remember(
        self,
        owner_id: str,
        category: str,
        name: str,
        body: str,
        source: str = "",
        confidence: float = 1.0,
    ) -> Dict:
        """Upsert the canonical value for ``(owner_id, category, name)``.

        - If the slot is empty, insert version 1.
        - If the current body is identical, no-op (returns the existing row
          unchanged) so restating a stable fact never accumulates duplicates.
        - Otherwise supersede: stamp ``valid_until`` on the current row and
          insert the new body with ``version + 1``.

        Returns the resulting current row as a dict, with an added
        ``status`` key: ``"created"``, ``"unchanged"``, or ``"updated"``.

        Raises ``ValueError`` if owner_id / category / name / body is empty —
        the slot key and value must all be non-blank for the uniqueness
        guarantee to be meaningful.
        """
        if not (owner_id and category and name):
            raise ValueError("owner_id, category, and name are required")
        if not body or not body.strip():
            raise ValueError("body is required and cannot be blank")

        cursor = self.conn.cursor()
        # BEGIN IMMEDIATE so the read-current + supersede + insert sequence is
        # atomic against a concurrent writer racing on the same slot (the
        # partial unique index would otherwise reject a second live row, but we
        # want a clean transaction rather than an IntegrityError).
        cursor.execute("BEGIN IMMEDIATE")
        try:
            current = cursor.execute(
                "SELECT * FROM canonical_facts "
                "WHERE owner_id = ? AND category = ? AND name = ? "
                "AND valid_until IS NULL",
                (owner_id, category, name),
            ).fetchone()

            if current is not None and current["body"] == body:
                self.conn.commit()
                row = dict(current)
                row["status"] = "unchanged"
                return row

            now = _now()
            # Version climbs monotonically from the slot's whole history, so it
            # keeps increasing even across a forget()+re-remember() gap (where
            # there is no current row to read a version from). status reflects
            # whether a live value was superseded: "created" when the slot had
            # no current value (brand-new or previously retired), else "updated".
            prior_max = cursor.execute(
                "SELECT MAX(version) FROM canonical_facts "
                "WHERE owner_id = ? AND category = ? AND name = ?",
                (owner_id, category, name),
            ).fetchone()[0]
            version = (prior_max or 0) + 1
            if current is None:
                status = "created"
            else:
                cursor.execute(
                    "UPDATE canonical_facts SET valid_until = ? WHERE id = ?",
                    (now, current["id"]),
                )
                status = "updated"

            cursor.execute(
                """
                INSERT INTO canonical_facts
                    (owner_id, category, name, body, source, confidence,
                     version, valid_from, valid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (owner_id, category, name, body, source, confidence, version, now),
            )
            new_id = cursor.lastrowid
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        row = dict(
            self.conn.execute(
                "SELECT * FROM canonical_facts WHERE id = ?", (new_id,)
            ).fetchone()
        )
        row["status"] = status
        return row

    def forget(self, owner_id: str, category: str, name: str) -> bool:
        """Retire a canonical slot without replacing it.

        Stamps ``valid_until`` on the current row (preserving it as history),
        mirroring TripleStore.end() — there is then no current value for the
        slot. Returns True if a current row was retired, False if the slot was
        already empty. Auditable: nothing is deleted.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE canonical_facts SET valid_until = ? "
            "WHERE owner_id = ? AND category = ? AND name = ? "
            "AND valid_until IS NULL",
            (_now(), owner_id, category, name),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def recall(self, owner_id: str, category: str, name: str) -> Optional[Dict]:
        """Return the single current value for a slot, or None if unset."""
        row = self.conn.execute(
            "SELECT * FROM canonical_facts "
            "WHERE owner_id = ? AND category = ? AND name = ? "
            "AND valid_until IS NULL",
            (owner_id, category, name),
        ).fetchone()
        return dict(row) if row is not None else None

    def list(self, owner_id: str, category: Optional[str] = None) -> List[Dict]:
        """All current canonical values for an owner, optionally one category.

        Ordered by ``(category, name)`` so a persona's self-card set reads
        stably.
        """
        if category is None:
            rows = self.conn.execute(
                "SELECT * FROM canonical_facts "
                "WHERE owner_id = ? AND valid_until IS NULL "
                "ORDER BY category, name",
                (owner_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM canonical_facts "
                "WHERE owner_id = ? AND category = ? AND valid_until IS NULL "
                "ORDER BY name",
                (owner_id, category),
            ).fetchall()
        return [dict(r) for r in rows]

    def history(self, owner_id: str, category: str, name: str) -> List[Dict]:
        """All versions of a slot (current first, then superseded), newest-first."""
        rows = self.conn.execute(
            "SELECT * FROM canonical_facts "
            "WHERE owner_id = ? AND category = ? AND name = ? "
            "ORDER BY version DESC",
            (owner_id, category, name),
        ).fetchall()
        return [dict(r) for r in rows]

    def search(self, owner_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Owner-scoped substring search over current canonical values.

        Matches ``query`` (case-insensitive) against body, name, and category
        of the owner's *current* rows. A LIKE scan is the right tool here: the
        per-owner set is small (identity cards, not a memory firehose), and it
        keeps the store at "one table + a unique index" — no FTS dependency. For
        known-key reads use ``recall`` instead; this is the fuzzy fallback that
        lets full-text lookups span canonical entries.
        """
        if not query or not query.strip():
            return []
        like = f"%{query.strip()}%"
        rows = self.conn.execute(
            "SELECT * FROM canonical_facts "
            "WHERE owner_id = ? AND valid_until IS NULL "
            "AND (body LIKE ? COLLATE NOCASE "
            "     OR name LIKE ? COLLATE NOCASE "
            "     OR category LIKE ? COLLATE NOCASE) "
            "ORDER BY category, name LIMIT ?",
            (owner_id, like, like, like, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def model_card(
        self,
        owner_id: str,
        category: str,
        title: Optional[str] = None,
        names: Optional[List[str]] = None,
    ) -> Dict:
        """Render current canonical slots as a compact model card.

        This is a deterministic view over existing canonical facts, not a new
        storage layer. It is useful for identity/user/project/workflow models
        that need one current value per slot, history through canonical
        supersession, and stable prompt-ready text.
        """
        rows = self.list(owner_id, category)
        if names is not None:
            wanted = [name for name in names if name]
            by_name = {row["name"]: row for row in rows}
            rows = [by_name[name] for name in wanted if name in by_name]

        card_title = title or category.replace(":", " ").replace("_", " ").title()
        lines = [f"## {card_title}"] if rows else []
        for row in rows:
            label = str(row.get("name") or "").replace("_", " ").strip().title()
            body = str(row.get("body") or "").strip()
            if body:
                lines.append(f"- {label}: {body}")

        return {
            "owner_id": owner_id,
            "category": category,
            "title": card_title,
            "slots": rows,
            "body": "\n".join(lines),
        }

    # ------------------------------------------------------------------
    # Export / import (parity with TripleStore / AnnotationStore)
    # ------------------------------------------------------------------

    def export_all(self) -> List[Dict]:
        """Export every row (current + history) as a list of dicts."""
        rows = self.conn.execute(
            """
            SELECT id, owner_id, category, name, body, source, confidence,
                   version, valid_from, valid_until, created_at
            FROM canonical_facts
            ORDER BY id
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def import_all(self, rows: List[Dict], force: bool = False) -> Dict:
        """Import canonical rows from a list of dicts.

        Mirrors TripleStore/AnnotationStore.import_all semantics:

        - **No id collision**: insert with the imported ``id``
          (``stats["inserted"]``).
        - **Id collision + identical content**: skip (``stats["skipped"]``).
        - **Id collision + different content**: insert with a fresh
          auto-assigned id (``stats["imported_renumbered"]``).
        - **No id supplied**: insert with a fresh id (``stats["inserted"]``).
        - ``force=True``: on id collision, overwrite
          (``stats["overwritten"]``).

        A renumber INSERT that would create a *second live row* for an existing
        slot (partial unique index on ``(owner_id, category, name) WHERE
        valid_until IS NULL``) raises IntegrityError; we catch it and bucket the
        row into ``stats["skipped"]`` — the slot is already represented in the
        destination. Sum of stats equals ``len(rows)``.
        """
        stats = {"inserted": 0, "skipped": 0, "overwritten": 0,
                 "imported_renumbered": 0}
        cursor = self.conn.cursor()

        _CONTENT_FIELDS = ("owner_id", "category", "name", "body", "source",
                           "confidence", "version", "valid_from", "valid_until",
                           "created_at")
        _INSERT_DEFAULTS = {"source": "imported", "confidence": 1.0, "version": 1}

        def _normalized(item):
            return {
                f: item.get(f) if item.get(f) is not None else _INSERT_DEFAULTS.get(f)
                for f in _CONTENT_FIELDS
            }

        seen_ids = set()
        for item in rows:
            row_id = item.get("id")
            if row_id is not None:
                if row_id in seen_ids:
                    raise ValueError(
                        f"import_all: duplicate id {row_id!r} in the imported "
                        f"batch. Deduplicate the input before calling."
                    )
                seen_ids.add(row_id)

        cursor.execute("BEGIN IMMEDIATE")
        try:
            existing = cursor.execute(
                "SELECT id, owner_id, category, name, body, source, confidence, "
                "version, valid_from, valid_until, created_at FROM canonical_facts"
            ).fetchall()
            existing_snapshot = {
                r[0]: dict(zip(_CONTENT_FIELDS, r[1:])) for r in existing
            }

            def _insert_with_id(item, row_id):
                cursor.execute(
                    """
                    INSERT INTO canonical_facts
                        (id, owner_id, category, name, body, source, confidence,
                         version, valid_from, valid_until, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id, item.get("owner_id"), item.get("category"),
                        item.get("name"), item.get("body"),
                        item.get("source", "imported"),
                        item.get("confidence", 1.0), item.get("version", 1),
                        item.get("valid_from") or _now(), item.get("valid_until"),
                        item.get("created_at"),
                    ),
                )

            def _insert_without_id(item):
                cursor.execute(
                    """
                    INSERT INTO canonical_facts
                        (owner_id, category, name, body, source, confidence,
                         version, valid_from, valid_until, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.get("owner_id"), item.get("category"),
                        item.get("name"), item.get("body"),
                        item.get("source", "imported"),
                        item.get("confidence", 1.0), item.get("version", 1),
                        item.get("valid_from") or _now(), item.get("valid_until"),
                        item.get("created_at"),
                    ),
                )

            explicit_no_collision = []
            no_id = []
            collisions = []
            for item in rows:
                row_id = item.get("id")
                if row_id is None:
                    no_id.append(item)
                elif row_id in existing_snapshot:
                    collisions.append(item)
                else:
                    explicit_no_collision.append(item)

            for item in explicit_no_collision:
                try:
                    _insert_with_id(item, item["id"])
                    stats["inserted"] += 1
                except sqlite3.IntegrityError:
                    stats["skipped"] += 1
            for item in no_id:
                try:
                    _insert_without_id(item)
                    stats["inserted"] += 1
                except sqlite3.IntegrityError:
                    stats["skipped"] += 1

            for item in collisions:
                row_id = item["id"]
                existing_content = existing_snapshot[row_id]
                if force:
                    cursor.execute(
                        "DELETE FROM canonical_facts WHERE id = ?", (row_id,)
                    )
                    _insert_with_id(item, row_id)
                    stats["overwritten"] += 1
                    continue
                if _normalized(item) == existing_content:
                    stats["skipped"] += 1
                    continue
                try:
                    _insert_without_id(item)
                    stats["imported_renumbered"] += 1
                except sqlite3.IntegrityError:
                    stats["skipped"] += 1

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return stats


# ---------------------------------------------------------------------------
# Module-level convenience functions (mirror triples.py / annotations.py shape)
# ---------------------------------------------------------------------------

def remember_canonical(
    owner_id: str,
    category: str,
    name: str,
    body: str,
    source: str = "",
    confidence: float = 1.0,
    db_path: Optional[Path] = None,
) -> Dict:
    """Upsert a canonical fact without instantiating CanonicalStore manually."""
    store = CanonicalStore(db_path=db_path)
    return store.remember(owner_id, category, name, body,
                          source=source, confidence=confidence)


def recall_canonical(
    owner_id: str,
    category: str,
    name: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict]:
    """Read a single canonical fact without instantiating CanonicalStore."""
    store = CanonicalStore(db_path=db_path)
    return store.recall(owner_id, category, name)


def forget_canonical(
    owner_id: str,
    category: str,
    name: str,
    db_path: Optional[Path] = None,
) -> bool:
    """Retire a canonical slot without replacing it.

    Stamps ``valid_until`` on the current row, preserving it as history.
    Returns True if a current row was retired, False if the slot was
    already empty.
    """
    store = CanonicalStore(db_path=db_path)
    return store.forget(owner_id, category, name)
