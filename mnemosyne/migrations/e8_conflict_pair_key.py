"""
Mnemosyne E8 Migration — order-normalized unique key on ``conflicts``
=====================================================================

Idempotent migration that adds ONE expression index to an existing bank:

  - ``idx_conflicts_pair_norm`` — UNIQUE on (min(fact_a_id, fact_b_id),
    max(fact_a_id, fact_b_id))

Why an expression index and not a two-column one: the detector does not
canonicalize orientation. 15 of 32 persisted conflict rows measured
2026-09-26 violate ``fact_a_id < fact_b_id``, so a plain UNIQUE on
(fact_a_id, fact_b_id) cannot see the swapped pair ``(b, a)`` and the same
contradiction re-lands as a third row.

Canonical DDL source: ``mnemosyne/core/veracity_consolidation.py``
(``_init_conflicts_table``, ``CREATE TABLE IF NOT EXISTS conflicts``). The
index DDL below belongs beside that statement upstream — and because the
table DDL is ``IF NOT EXISTS``, changing it reaches fresh installs only,
which is exactly why existing banks need this migration. Per the E7
precedent we do NOT invent DDL here; if upstream's canonical index
definition changes, this migration is updated to match.

PAIRING REQUIREMENT (do not ship this migration alone): core's
``_record_conflict`` currently executes a bare
``INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) VALUES (?,?,?)``.
With the unique index in place that statement raises IntegrityError inside
the consolidation loop — the partial-state class the ``_record_conflict``
docstring already flags (fact INSERT durable, later conflict-record failure
leaks partial state). The insert must gain
``ON CONFLICT DO NOTHING`` in the same change. A bare upsert target is
deliberate: it needs no expression repeated verbatim and covers every
uniqueness violation on the table.

Safety: creates an index only. No table is dropped, no row is rewritten,
and no row is deleted. If pre-existing duplicate pairs are found the index
is NOT created (SQLite would fail) — the migration reports them and leaves
the bank untouched for adjudication instead of guessing a winner.

Safe to re-run: an existing same-named index is VALIDATED against the
canonical definition, and a matching one is a no-op. A same-named index
carrying a DIFFERENT definition is an explicit failure
(``IndexDefinitionMismatchError``) — never a reported success.

An existing ``conflicts`` table that lacks the pair columns — or declares
either one nullable — is likewise an explicit failure
(``ConflictSchemaUnreadableError``), in real and dry runs alike: a dry
run that promises ``index_added`` over a schema the constraint cannot
fully enforce is the same false green this module refuses everywhere
else. Nullable is not cosmetic: SQLite unique indexes treat NULL keys as
distinct, so NULL-bearing rows would re-grow the ledger under an index
reported applied. The canonical core DDL declares both pair columns
``NOT NULL`` — this refusal only ever names foreign schemas. This schema
check runs FIRST, before any same-named index is accepted: a canonical
index already sitting on a nullable table is refused (reported applied
would be false), not silently treated as an already-done no-op.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Literal, Tuple, TypedDict, Union, overload


# Canonical DDL — mirrors the index that belongs beside
# ``CREATE TABLE IF NOT EXISTS conflicts`` in
# ``mnemosyne/core/veracity_consolidation.py``.
_INDEX_NAME = "idx_conflicts_pair_norm"
_INDEX_DDL = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS {_INDEX_NAME} "
    "ON conflicts (min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id))"
)

# What SQLite stores in sqlite_master for _INDEX_DDL: the IF NOT EXISTS
# clause is stripped from the stored text while the caller's whitespace
# otherwise survives verbatim. Definition validation therefore compares
# canonical token streams (punctuation split, whitespace collapsed,
# casefolded) rather than raw text.
_STORED_INDEX_DDL = (
    f"CREATE UNIQUE INDEX {_INDEX_NAME} "
    "ON conflicts (min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id))"
)


class IndexDefinitionMismatchError(RuntimeError):
    """A same-named ``conflicts`` index enforces a different definition.

    E8 refuses such a bank instead of accepting the name: reporting
    ``applied=True`` over an index that does not enforce the normalized
    pair key would be a false green, and silently dropping or shadowing
    an index object this migration does not own is out of its authority.
    The message carries the stored and expected definitions; fix the
    named index, then re-run.
    """


class ConflictSchemaUnreadableError(RuntimeError):
    """The ``conflicts`` table exists but cannot host the pair index.

    Two shapes are refused. Missing pair columns: the duplicate sweep
    and the index DDL both read ``fact_a_id``/``fact_b_id``, so against
    a table without them every E8 statement raises
    ``sqlite3.OperationalError``. Nullable pair columns: the index can
    be created but cannot enforce — SQLite unique indexes treat NULL
    keys as distinct, so NULL-bearing rows slip the constraint under an
    index reported applied. Either way a dry run promising
    ``index_added`` would predict a success the schema defeats — the
    same false green as accepting an index by name alone. Both modes
    therefore fail loudly, bank untouched. Repair the table, then
    re-run.
    """

# Normalized-pair predicate, reused for the pre-flight duplicate sweep.
# Group by the two min/max ID EXPRESSIONS, not a slash-joined string: the
# join is ambiguous — (cf_a/b, cf_c) and (cf_a, cf_b/c) both stringify to
# cf_a/b/cf_c, so a joined key would report two genuinely distinct pairs as
# duplicates and wrongly refuse to build the index. SQLite's 2-arg
# min()/max() (scalar) give the same orientation normalization the unique
# index uses, and the pair is formatted for reporting only after grouping.
_DUPLICATE_PAIRS_SQL = """
    SELECT lo, hi, COUNT(*) AS n FROM (
        SELECT min(fact_a_id, fact_b_id) AS lo,
               max(fact_a_id, fact_b_id) AS hi
          FROM conflicts
    )
     GROUP BY lo, hi
    HAVING n > 1
"""


class MigrationReport(TypedDict):
    """Outcome of a real (non-dry-run) E8 pass."""

    applied: bool
    conflicts_table_missing: bool
    index_already_present: bool
    index_added: bool
    duplicate_pairs: List[str]


class MigrationDryRunReport(MigrationReport):
    """Outcome of a dry run: reports what a real pass would do."""

    dry_run: Literal[True]


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


# The pair columns both the duplicate sweep and the index DDL read.
# SQLite resolves column names case-insensitively, so the gate below
# compares casefolded names: a differently-cased legacy table is still
# indexable and must not be refused.
_REQUIRED_CONFLICT_COLUMNS = ("fact_a_id", "fact_b_id")


def _conflicts_columns(conn: sqlite3.Connection) -> List[Tuple[str, int]]:
    """(name, notnull) per column of the existing ``conflicts`` table.

    ``PRAGMA table_info`` reports the NOT NULL declaration flag as 0/1;
    the gate below uses it because a nullable pair column defeats the
    unique index (NULL keys are distinct) even though the columns exist.
    """
    return [(row[1], row[3]) for row in conn.execute("PRAGMA table_info(conflicts)")]


def _index_ddl(conn: sqlite3.Connection, name: str) -> Union[str, None]:
    """Stored DDL text of a named index, or None when absent."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
    ).fetchone()
    return row[0] if row is not None else None


def _ddl_tokens(sql: str) -> str:
    # Canonical token stream: split punctuation from its neighbours,
    # collapse all whitespace, casefold. Two statements compare equal only
    # when they index the same expressions in the same order — layout is
    # free, meaning is not. Quoted/bracketed identifiers are NOT unquoted
    # here; a name-shaped-but-different spelling is treated as a mismatch
    # and fails loudly, which is the conservative direction.
    spaced = sql.replace("(", " ( ").replace(")", " ) ").replace(",", " , ")
    return " ".join(spaced.split()).casefold()


def _ddl_equivalent(stored: str, canonical: str) -> bool:
    return _ddl_tokens(stored) == _ddl_tokens(canonical)


def _duplicate_pairs(conn: sqlite3.Connection) -> List[str]:
    # The caller has verified the table exists AND exposes the pair
    # columns, so an SQL failure here (lock, corruption) is NOT "no
    # duplicates" — it propagates rather than masquerading as a clean
    # sweep that green-lights the index.
    rows = conn.execute(_DUPLICATE_PAIRS_SQL).fetchall()
    # Display form only; grouping already happened on the unambiguous
    # (lo, hi) expression pair.
    return [f"{r[0]}/{r[1]}" for r in rows]


@overload
def migrate_conflict_pair_key(db_path: Path, dry_run: Literal[True]) -> MigrationDryRunReport: ...


@overload
def migrate_conflict_pair_key(
    db_path: Path, dry_run: Literal[False] = False
) -> MigrationReport: ...


def migrate_conflict_pair_key(
    db_path: Path, dry_run: bool = False
) -> Union[MigrationReport, MigrationDryRunReport]:
    """Add the order-normalized unique index to ``conflicts``.

    Idempotent; index-only; never writes or deletes a row. Returns a report
    instead of raising when the bank is not yet at the conflicts schema or
    when duplicate pairs make a unique index impossible today. A same-named
    existing index is validated against the canonical definition; a
    mismatch raises IndexDefinitionMismatchError (explicit failure, bank
    untouched) rather than falsely reporting ``applied=True``. An
    existing table that lacks the pair columns, or declares either one
    nullable, raises ConflictSchemaUnreadableError in EITHER mode — a
    dry run never promises an index the schema cannot fully enforce.
    That schema gate runs before the existing-index check, so a
    canonical index already sitting on an unreadable table is refused
    rather than passing as an already-applied no-op.
    """
    report: MigrationReport = {
        "applied": False,
        "conflicts_table_missing": False,
        "index_already_present": False,
        "index_added": False,
        "duplicate_pairs": [],
    }
    # The discriminator is set up front: EVERY early return below is still
    # a MigrationDryRunReport when dry_run was requested, and consumers
    # (CLI JSON) read report["dry_run"] unconditionally on that branch.
    if dry_run:
        report["dry_run"] = True  # type: ignore[typeddict-item]

    if not db_path.exists():
        report["conflicts_table_missing"] = True
        return report

    conn = sqlite3.connect(str(db_path))
    try:
        if not _has_table(conn, "conflicts"):
            report["conflicts_table_missing"] = True
            return report  # type: ignore[return-value]

        # An existing table with an unreadable shape is NOT the
        # missing-table case reported above, and this gate runs BEFORE
        # any index is accepted — whether E8 would create the index or
        # finds one already present, ``applied=True`` over a schema the
        # constraint cannot fully enforce is the same false green.
        # Missing pair columns: every E8 statement raises. Nullable pair
        # columns: the index builds (and could already exist) but cannot
        # enforce — SQLite unique indexes treat NULL keys as distinct, so
        # NULL-bearing rows slip it. Either way refuse explicitly, bank
        # untouched, in both dry-run and real modes.
        columns = _conflicts_columns(conn)
        found = {name.casefold() for name, _ in columns}
        missing = [c for c in _REQUIRED_CONFLICT_COLUMNS if c not in found]
        if missing:
            raise ConflictSchemaUnreadableError(
                "conflicts table exists but does not expose the pair "
                f"columns E8 needs; missing: {', '.join(missing)}; found: "
                f"{', '.join(sorted(found)) or '(no columns)'}"
            )
        nullable = sorted(
            name
            for name, notnull in columns
            if not notnull and name.casefold() in _REQUIRED_CONFLICT_COLUMNS
        )
        if nullable:
            raise ConflictSchemaUnreadableError(
                "conflicts table exposes the pair columns but declares "
                "them nullable; a UNIQUE index does not constrain NULL "
                "keys, so E8 cannot honestly report the normalized pair "
                f"key as enforced; nullable: {', '.join(nullable)}"
            )

        stored_ddl = _index_ddl(conn, _INDEX_NAME)
        if stored_ddl is not None:
            if not _ddl_equivalent(stored_ddl, _STORED_INDEX_DDL):
                raise IndexDefinitionMismatchError(
                    f"index '{_INDEX_NAME}' exists but does not enforce the "
                    "order-normalized unique pair key; refusing to report E8 "
                    f"as applied. stored: {stored_ddl.strip()!r}; "
                    f"expected: {_STORED_INDEX_DDL!r}"
                )
            report["index_already_present"] = True
            report["applied"] = True
            return report  # type: ignore[return-value]

        duplicates = _duplicate_pairs(conn)
        if duplicates:
            # A unique index cannot be created over duplicates. Report and
            # leave the bank untouched: choosing a winner is adjudication,
            # not migration.
            report["duplicate_pairs"] = duplicates
            return report  # type: ignore[return-value]

        if dry_run:
            report["index_added"] = True
            report["applied"] = True
            return report  # type: ignore[return-value]

        conn.execute(_INDEX_DDL)
        conn.commit()
        report["index_added"] = True
        report["applied"] = True
    finally:
        conn.close()
    return report
