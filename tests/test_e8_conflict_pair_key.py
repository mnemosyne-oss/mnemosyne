"""Regression test for the conflict pair key (E8).

The `conflicts` table records detected contradictions as rows, and the
detector does not canonicalize orientation: measured 2026-09-26, 15 of 32
persisted rows violated `fact_a_id < fact_b_id`. A plain UNIQUE on
(fact_a_id, fact_b_id) therefore cannot see the swapped re-land `(b, a)`,
and the same contradiction accumulates rows.

Two halves must ship together:

1. An order-normalized unique index on the pair,
   `min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id)` — created by the
   canonical DDL in `VeracityConsolidator._init_tables` for fresh banks and
   by `mnemosyne.migrations.e8_conflict_pair_key` for existing banks
   (`CREATE TABLE IF NOT EXISTS` never retrofits a DDL change).
2. `_record_conflict`'s insert gaining `ON CONFLICT DO NOTHING`. Without
   it, the index converts silent re-growth into an IntegrityError inside a
   caller's `_serialized_write` scope — the partial-state class the method's
   own docstring guards.

The test asserts the relationship (a re-detected pair cannot become a second
row), never a row count of any live bank.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne import cli
from mnemosyne.core.banks import BankManager
from mnemosyne.core.veracity_consolidation import VeracityConsolidator
from mnemosyne.migrations.e8_conflict_pair_key import (
    ConflictSchemaUnreadableError,
    IndexDefinitionMismatchError,
    migrate_conflict_pair_key,
)

INDEX_NAME = "idx_conflicts_pair_norm"

CONFLICTS_DDL = """
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a_id TEXT NOT NULL,
    fact_b_id TEXT NOT NULL,
    conflict_type TEXT,
    resolution TEXT,
    resolved_at TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _legacy_bank(path: Path, rows) -> Path:
    """A bank whose conflicts table predates the index (the migration's input)."""
    con = sqlite3.connect(str(path))
    try:
        con.execute(CONFLICTS_DDL)
        con.executemany(
            "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) VALUES (?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()
    return path


def _index_present(db_path: Path) -> bool:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (INDEX_NAME,)
        ).fetchone()
    finally:
        con.close()
    return row is not None


def _row_count(db_path: Path) -> int:
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
    finally:
        con.close()


def _ids(db_path: Path) -> set:
    con = sqlite3.connect(str(db_path))
    try:
        return {r[0] for r in con.execute("SELECT id FROM conflicts").fetchall()}
    finally:
        con.close()


def test_migration_adds_index_and_preserves_rows(tmp_path):
    # The third row shares its low member with the second but is a distinct
    # pair — it guards that the sweep groups the FULL normalized pair, not
    # one member. It is NOT an orientation swap of the second row (that would
    # be (cf_d, cf_c)); none of these rows are duplicates.
    bank = _legacy_bank(
        tmp_path / "bank.db",
        [
            ("cf_a", "cf_b", "contradiction"),
            ("cf_c", "cf_d", "contradiction"),
            ("cf_z", "cf_c", "contradiction"),
        ],
    )
    before_ids = _ids(bank)

    report = migrate_conflict_pair_key(bank)

    assert report["applied"] is True
    assert report["index_added"] is True
    assert report["duplicate_pairs"] == []
    assert _index_present(bank) is True
    assert _ids(bank) == before_ids  # index-only: no row written, none deleted

    # The MIGRATED index must enforce normalized uniqueness, not merely
    # exist under the right name: re-landing the reverse orientation of an
    # existing pair is the same contradiction and must be refused.
    con = sqlite3.connect(str(bank))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) "
                "VALUES ('cf_b', 'cf_a', 'contradiction')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) "
                "VALUES ('cf_a', 'cf_b', 'contradiction')"
            )
        # A distinct pair still inserts cleanly (the key is pair-normalized,
        # not member-global).
        con.execute(
            "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) "
            "VALUES ('cf_e', 'cf_f', 'contradiction')"
        )
    finally:
        con.close()


def test_migration_is_idempotent(tmp_path):
    bank = _legacy_bank(tmp_path / "bank.db", [("cf_a", "cf_b", "contradiction")])

    first = migrate_conflict_pair_key(bank)
    second = migrate_conflict_pair_key(bank)

    assert first["index_added"] is True
    assert second["index_already_present"] is True
    assert second["index_added"] is False
    assert _row_count(bank) == 1


def test_migration_refuses_duplicate_laden_bank_without_writing(tmp_path):
    # Same normalized pair under both orderings: a unique index cannot be
    # created over this, and choosing a winner is adjudication, not migration.
    bank = _legacy_bank(
        tmp_path / "bank.db",
        [("cf_x", "cf_y", "contradiction"), ("cf_y", "cf_x", "contradiction")],
    )

    report = migrate_conflict_pair_key(bank)

    assert report["applied"] is False
    assert report["index_added"] is False
    assert report["duplicate_pairs"] == ["cf_x/cf_y"]
    assert _index_present(bank) is False
    assert _row_count(bank) == 2  # nothing rewritten, nothing deleted


def test_migration_reports_missing_conflicts_table(tmp_path):
    bank = tmp_path / "other.db"
    con = sqlite3.connect(str(bank))
    con.execute("CREATE TABLE facts (id TEXT)")
    con.commit()
    con.close()

    report = migrate_conflict_pair_key(bank)

    assert report["conflicts_table_missing"] is True
    assert report["applied"] is False


def test_dry_run_creates_nothing(tmp_path):
    bank = _legacy_bank(tmp_path / "bank.db", [("cf_a", "cf_b", "contradiction")])

    report = migrate_conflict_pair_key(bank, dry_run=True)

    assert report["index_added"] is True  # would add
    assert report["dry_run"] is True
    assert _index_present(bank) is False  # but did not


def test_canonical_init_creates_index_on_fresh_bank(tmp_path):
    """The DDL path and the migration path converge: a fresh bank is born indexed."""
    db_path = tmp_path / "fresh.db"
    consolidator = VeracityConsolidator(db_path=db_path)
    try:
        consolidator._init_tables()
    finally:
        consolidator.conn.close()

    assert _index_present(db_path) is True
    # And the migration is then a no-op.
    report = migrate_conflict_pair_key(db_path)
    assert report["index_already_present"] is True


def test_record_conflict_swapped_pair_does_not_grow_the_ledger(tmp_path):
    """The regression this ships against: a re-detected pair, either orientation.

    First insert lands. The swapped re-detection must be a no-op and must not
    raise — a raise here lands inside a caller's `_serialized_write` scope and
    leaks partial state (fact row durable, conflict record lost).
    """
    db_path = tmp_path / "bank.db"
    consolidator = VeracityConsolidator(db_path=db_path)
    try:
        consolidator._init_tables()
        consolidator._record_conflict("cf_a", "cf_b", "contradiction")

        # Same pair, inverted orientation, and the original orientation again.
        consolidator._record_conflict("cf_b", "cf_a", "contradiction")
        consolidator._record_conflict("cf_a", "cf_b", "contradiction")

        assert _row_count(db_path) == 1
    finally:
        consolidator.conn.close()


def test_index_is_what_makes_the_bare_insert_unsafe(tmp_path):
    """Why the index must not ship alone: without the upsert, it raises.

    This pins the pairing requirement from the other side — the constraint is
    real, so an index-only PR would present as IntegrityError on the ambient
    detection path rather than as silent re-growth.
    """
    db_path = tmp_path / "bank.db"
    consolidator = VeracityConsolidator(db_path=db_path)
    try:
        consolidator._init_tables()
        consolidator._record_conflict("cf_a", "cf_b", "contradiction")

        with pytest.raises(sqlite3.IntegrityError):
            consolidator.conn.execute(
                "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) VALUES (?, ?, ?)",
                ("cf_b", "cf_a", "contradiction"),
            )
        consolidator.conn.rollback()
        assert _row_count(db_path) == 1
    finally:
        consolidator.conn.close()


def test_consolidator_opens_duplicate_laden_legacy_bank(tmp_path):
    """Fresh-bank gate (CodeRabbit Major on the head): a pre-existing bank
    holding duplicate normalized pairs must still OPEN. Before the gate,
    CREATE UNIQUE INDEX in _init_tables raised IntegrityError for such a
    bank — every VeracityConsolidator(db_path=...) on it died, including
    the E8 migration's own reporter path. Indexing that bank is the E8
    migration's job, and E8's job is to REPORT the duplicates, not to be
    pre-empted by a constructor crash."""
    db_path = tmp_path / "legacy.db"
    _legacy_bank(db_path, [
        ("cf_x", "cf_y", "contradiction"),
        ("cf_y", "cf_x", "contradiction"),  # same normalized pair, swapped
    ])

    consolidator = VeracityConsolidator(db_path=db_path)  # must not raise
    try:
        assert _index_present(db_path) is False, (
            "the inline DDL must not force an index onto a populated legacy bank"
        )
    finally:
        consolidator.conn.close()

    # And the migration still sees the bank honestly: it reports the
    # duplicate pair and refuses to write, rather than the opener exploding.
    report = migrate_conflict_pair_key(db_path)
    assert report["applied"] is False
    assert report["duplicate_pairs"] == ["cf_x/cf_y"]


def test_duplicate_sweep_groups_by_pair_not_by_joined_string(tmp_path):
    """Slash-joined pair keys are ambiguous (CodeRabbit Major on the head):
    rows (b/c, a) and (a/b, c) normalize to the distinct pairs (a, b/c) and
    (a/b, c) — but BOTH stringify to "a/b/c", so a joined GROUP BY reports a
    false duplicate and the migration refuses forever. Grouping now happens
    on the (min, max) expressions, which only two genuinely identical
    normalized pairs can collide under."""
    db_path = tmp_path / "slashy.db"
    _legacy_bank(db_path, [
        ("b/c", "a", "contradiction"),
        ("a/b", "c", "contradiction"),
    ])

    report = migrate_conflict_pair_key(db_path)
    assert report["duplicate_pairs"] == [], (
        "distinct slash-containing pairs were mistaken for duplicates: "
        f"{report['duplicate_pairs']}"
    )
    assert report["index_added"] is True
    assert _index_present(db_path) is True


def test_dry_run_report_carries_discriminator_on_every_branch(tmp_path):
    """MigrationDryRunReport promises report["dry_run"] on EVERY dry-run
    outcome (CodeRabbit Minor on the head): table-missing, index-present,
    duplicate-laden and would-add branches all must carry the key — the
    CLI consumer reads it unconditionally."""
    missing = tmp_path / "nope.db"
    dup = tmp_path / "dup.db"
    _legacy_bank(dup, [("cf_x", "cf_y", "c"), ("cf_y", "cf_x", "c")])
    migrated = tmp_path / "mig.db"
    _legacy_bank(migrated, [("cf_a", "cf_b", "c")])
    assert migrate_conflict_pair_key(migrated)["index_added"] is True

    for path in (missing, dup, migrated):
        report = migrate_conflict_pair_key(path, dry_run=True)
        assert report["dry_run"] is True, path
    # Real (non-dry) reports must not claim the discriminator.
    assert "dry_run" not in migrate_conflict_pair_key(dup)


# ---------------------------------------------------------------------------
# Round 2 (review, 2026-09-27): definition validation + CLI reachability.
# ---------------------------------------------------------------------------


def _wrong_same_named_index(db_path: Path) -> None:
    """Drop an index under E8's name that enforces something ELSE.

    A plain (non-unique) index on the raw columns is the realistic
    lookalike: same name, no normalized-pair constraint. Accepting it by
    name alone made the migration report a false success.
    """
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE INDEX idx_conflicts_pair_norm ON conflicts (fact_a_id, fact_b_id)"
    )
    con.commit()
    con.close()


def _stored_index_ddl(db_path: Path) -> str:
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
            (INDEX_NAME,),
        ).fetchone()[0]
    finally:
        con.close()


def test_same_named_index_with_other_definition_fails_explicitly(tmp_path):
    bank = _legacy_bank(tmp_path / "bank.db", [("cf_a", "cf_b", "contradiction")])
    _wrong_same_named_index(bank)

    with pytest.raises(IndexDefinitionMismatchError):
        migrate_conflict_pair_key(bank)

    # Explicit refusal, not a reported success — and the bank is left as
    # found: the lookalike index untouched, no canonical one added, rows
    # intact. Repairing a schema object E8 does not own is not E8's call.
    assert "min(" not in _stored_index_ddl(bank)
    assert _row_count(bank) == 1


def test_same_named_index_mismatch_fails_explicitly_on_dry_run(tmp_path):
    bank = _legacy_bank(tmp_path / "bank.db", [("cf_a", "cf_b", "contradiction")])
    _wrong_same_named_index(bank)

    with pytest.raises(IndexDefinitionMismatchError):
        migrate_conflict_pair_key(bank, dry_run=True)

    assert _row_count(bank) == 1


def test_whitespace_variant_of_canonical_definition_is_a_validated_noop(tmp_path):
    """Validation must compare WHAT is indexed, not HOW the DDL was typed:
    the canonical statement written across several lines with stray spaces
    is the same constraint and stays a no-op."""
    bank = _legacy_bank(tmp_path / "bank.db", [("cf_a", "cf_b", "contradiction")])
    con = sqlite3.connect(str(bank))
    con.execute(
        "CREATE UNIQUE INDEX idx_conflicts_pair_norm\n"
        "            ON conflicts (  min(fact_a_id, fact_b_id),\n"
        "                            max(fact_a_id, fact_b_id) )"
    )
    con.commit()
    con.close()

    report = migrate_conflict_pair_key(bank)

    assert report["index_already_present"] is True
    assert report["applied"] is True


def _cli_bank(tmp_path, monkeypatch, rows=None):
    """Bank on disk wired into the CLI module (mirrors the e7 CLI fixture
    in tests/test_migration_dry_run_fingerprint.py). ``rows`` seeds the
    legacy conflicts table without the index; None guarantees the table is
    absent."""
    data_dir = tmp_path / "data"
    db_path = BankManager(data_dir).create_bank("e8bank")
    con = sqlite3.connect(str(db_path))
    con.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    if rows is None:
        con.execute("DROP TABLE IF EXISTS conflicts")
    con.commit()
    con.close()
    if rows is not None:
        _legacy_bank(db_path, rows)
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_BANK", "e8bank")
    return db_path


def test_cli_migrate_applies_e8_to_existing_bank(tmp_path, monkeypatch, capsys):
    """The reviewer's gap: `mnemosyne migrate` on an existing bank must
    actually deliver the pair constraint, and it must then ENFORCE."""
    db_path = _cli_bank(
        tmp_path,
        monkeypatch,
        rows=[("cf_a", "cf_b", "contradiction"), ("cf_d", "cf_c", "contradiction")],
    )

    cli.cmd_migrate([])

    out = capsys.readouterr().out
    assert "migrate e8 [APPLIED]" in out
    assert "index added: idx_conflicts_pair_norm" in out
    assert _index_present(db_path) is True
    con = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) "
                "VALUES ('cf_b', 'cf_a', 'contradiction')"
            )
    finally:
        con.close()


def test_cli_migrate_reports_blocking_duplicates_and_fails(
    tmp_path, monkeypatch, capsys
):
    """Duplicate pairs make the unique index impossible: the command names
    the pairs, leaves rows untouched, and exits non-zero — 'migrate
    succeeded' over an unapplied constraint is exactly the false green."""
    db_path = _cli_bank(
        tmp_path,
        monkeypatch,
        rows=[("cf_x", "cf_y", "contradiction"), ("cf_y", "cf_x", "contradiction")],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_migrate([])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "NOT APPLIED" in captured.out
    assert "cf_x/cf_y" in captured.out
    assert "migrate_incomplete" in captured.err
    assert _index_present(db_path) is False
    assert _row_count(db_path) == 2


def test_cli_dry_run_reports_e8_without_writing(tmp_path, monkeypatch, capsys):
    db_path = _cli_bank(tmp_path, monkeypatch, rows=[("cf_a", "cf_b", "contradiction")])

    cli.cmd_migrate(["--dry-run"])

    out = capsys.readouterr().out
    assert "migrate e8 [DRY RUN]" in out
    assert "would add index: idx_conflicts_pair_norm" in out
    assert _index_present(db_path) is False
    assert _row_count(db_path) == 1


def test_cli_migrate_skips_e8_when_conflicts_table_absent(
    tmp_path, monkeypatch, capsys
):
    """A bank that never ran veracity consolidation has no conflicts table.
    That is an honest 'nothing to index', not a failure, and must not
    corrupt the E7 result the command already printed."""
    db_path = _cli_bank(tmp_path, monkeypatch)

    cli.cmd_migrate([])

    out = capsys.readouterr().out
    assert "conflicts table absent — nothing to index" in out
    assert _index_present(db_path) is False


def test_cli_migrate_fails_explicitly_on_same_named_mismatch(
    tmp_path, monkeypatch, capsys
):
    db_path = _cli_bank(tmp_path, monkeypatch, rows=[("cf_a", "cf_b", "contradiction")])
    _wrong_same_named_index(db_path)

    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_migrate([])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "does not enforce the order-normalized unique pair key" in captured.err
    assert _row_count(db_path) == 1
    assert "min(" not in _stored_index_ddl(db_path)


# ---------------------------------------------------------------------------
# Round 3 (CodeRabbit review on 25edba72, 2026-09-28): an existing conflicts
# table with an unreadable shape is NOT the missing-table case. The sweep
# used to swallow the resulting OperationalError as "no duplicates", so the
# DRY RUN reported index_added/applied over a schema a real pass could only
# reject on the DDL. Both modes must now fail identically, bank untouched.
# ---------------------------------------------------------------------------


def _wrong_shape_conflicts_bank(db_path: Path) -> Path:
    """Give the bank a `conflicts` table that lacks the pair columns.

    Divergent-legacy shape: both the duplicate sweep and the index DDL
    read fact_a_id/fact_b_id, so every statement E8 runs against this
    bank raises OperationalError. There is no schema E8 can honestly
    report success for here.
    """
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE conflicts (id INTEGER PRIMARY KEY, note TEXT)")
    con.execute("INSERT INTO conflicts (note) VALUES ('legacy row')")
    con.commit()
    con.close()
    return db_path


def test_dry_run_refuses_unreadable_conflicts_schema(tmp_path):
    bank = _wrong_shape_conflicts_bank(tmp_path / "bank.db")

    with pytest.raises(ConflictSchemaUnreadableError) as excinfo:
        migrate_conflict_pair_key(bank, dry_run=True)

    # The message names both halves of the gap: what is missing, what was
    # found — enough to repair the table without opening a sqlite shell.
    assert "fact_a_id" in str(excinfo.value)
    assert "note" in str(excinfo.value)
    assert _index_present(bank) is False  # and nothing was written


def test_real_run_refuses_unreadable_conflicts_schema_explicitly(tmp_path):
    """Real mode already failed, but only as a raw OperationalError from
    the DDL — two layers below its cause. Explicit, named, explained."""
    bank = _wrong_shape_conflicts_bank(tmp_path / "bank.db")

    with pytest.raises(ConflictSchemaUnreadableError):
        migrate_conflict_pair_key(bank)

    assert _index_present(bank) is False
    con = sqlite3.connect(str(bank))
    try:
        assert con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 1
    finally:
        con.close()


def test_missing_table_still_reports_rather_than_raises(tmp_path):
    """The distinction the review asked for: table ABSENT is an honest
    report (nothing to index), table unreadable is an explicit failure.
    They are never conflated in either mode."""
    absent = tmp_path / "absent.db"
    con = sqlite3.connect(str(absent))
    con.execute("CREATE TABLE facts (id TEXT)")
    con.commit()
    con.close()

    report = migrate_conflict_pair_key(absent, dry_run=True)
    assert report["conflicts_table_missing"] is True
    assert report["applied"] is False
    assert report["index_added"] is False

    present = _wrong_shape_conflicts_bank(tmp_path / "present.db")
    with pytest.raises(ConflictSchemaUnreadableError):
        migrate_conflict_pair_key(present, dry_run=True)


def test_legacy_column_casing_is_not_a_refusal(tmp_path):
    """SQLite column names are case-insensitive; the gate compares
    casefolded, so a differently-cased canonical table is indexable and
    must pass, not be refused as unreadable. Both columns are NOT NULL
    here on purpose — the nullable-declaration refusal is a separate,
    orthogonal gate (Round 4 below); this test isolates casing."""
    con_db = tmp_path / "mixed.db"
    con = sqlite3.connect(str(con_db))
    con.execute(
        "CREATE TABLE conflicts (id INTEGER PRIMARY KEY, "
        "Fact_A_Id TEXT NOT NULL, Fact_B_Id TEXT NOT NULL, conflict_type TEXT)"
    )
    con.execute(
        "INSERT INTO conflicts (Fact_A_Id, Fact_B_Id, conflict_type) "
        "VALUES ('cf_a', 'cf_b', 'contradiction')"
    )
    con.commit()
    con.close()

    report = migrate_conflict_pair_key(con_db)
    assert report["index_added"] is True
    assert _index_present(con_db) is True


def test_cli_dry_run_does_not_promise_index_on_unreadable_schema(
    tmp_path, monkeypatch, capsys
):
    db_path = _cli_bank(tmp_path, monkeypatch)  # drops the conflicts table
    _wrong_shape_conflicts_bank(db_path)  # then seeds the divergent shape

    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_migrate(["--dry-run"])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "would add index" not in captured.out
    assert "does not expose the pair columns" in captured.err
    assert _index_present(db_path) is False


def test_cli_migrate_fails_explicitly_on_unreadable_schema(
    tmp_path, monkeypatch, capsys
):
    db_path = _cli_bank(tmp_path, monkeypatch)
    _wrong_shape_conflicts_bank(db_path)

    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_migrate([])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "does not expose the pair columns" in captured.err
    assert _index_present(db_path) is False


# ---------------------------------------------------------------------------
# Round 4 (CodeRabbit review on cda66e75, 2026-09-28): nullable pair
# columns. The table exposes fact_a_id/fact_b_id, so the Round 3 gate
# lets it pass and the UNIQUE index creates cleanly — and enforces
# nothing against NULL-bearing rows: SQLite unique indexes treat NULL
# keys as distinct, so duplicates re-grow under an index E8 reported
# applied. The canonical core DDL declares both columns NOT NULL, so a
# nullable pair column is a foreign shape no honest E8 run can
# green-light. Refuse it in both modes, bank untouched, like every
# other unreadable schema.
# ---------------------------------------------------------------------------


def _nullable_pair_bank(db_path: Path, nullable: str) -> Path:
    """A conflicts table with ONE pair column declared nullable.

    Everything else is canonical, isolating the NOT NULL flag as the
    only difference from an indexable bank.
    """
    a = "fact_a_id TEXT" if nullable == "fact_a_id" else "fact_a_id TEXT NOT NULL"
    b = "fact_b_id TEXT" if nullable == "fact_b_id" else "fact_b_id TEXT NOT NULL"
    con = sqlite3.connect(str(db_path))
    con.execute(
        f"CREATE TABLE conflicts (id INTEGER PRIMARY KEY, {a}, {b}, "
        "conflict_type TEXT)"
    )
    con.execute(
        "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) "
        "VALUES ('cf_a', 'cf_b', 'contradiction')"
    )
    con.commit()
    con.close()
    return db_path


def test_unique_index_genuinely_misses_nullable_rows(tmp_path):
    """Grounding for the refusal, asserted on SQLite itself: on a
    nullable-pair table the canonical index CREATES fine yet does not
    stop duplicate NULL-bearing rows — reporting applied=True there is
    a false green, not a technicality."""
    bank = _nullable_pair_bank(tmp_path / "bank.db", "fact_a_id")
    con = sqlite3.connect(str(bank))
    con.execute(
        f"CREATE UNIQUE INDEX {INDEX_NAME} "
        "ON conflicts (min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id))"
    )
    con.execute("INSERT INTO conflicts (fact_a_id, fact_b_id) VALUES (NULL, 'cf_x')")
    con.execute("INSERT INTO conflicts (fact_a_id, fact_b_id) VALUES (NULL, 'cf_x')")
    con.commit()
    n = con.execute(
        "SELECT COUNT(*) FROM conflicts WHERE fact_a_id IS NULL"
    ).fetchone()[0]
    con.close()
    assert n == 2


@pytest.mark.parametrize("column", ["fact_a_id", "fact_b_id"])
def test_dry_run_refuses_nullable_pair_column(tmp_path, column):
    bank = _nullable_pair_bank(tmp_path / "bank.db", column)

    with pytest.raises(ConflictSchemaUnreadableError) as excinfo:
        migrate_conflict_pair_key(bank, dry_run=True)

    # Message names the gap precisely enough to repair without a shell.
    assert "nullable" in str(excinfo.value)
    assert column in str(excinfo.value)
    assert _index_present(bank) is False


@pytest.mark.parametrize("column", ["fact_a_id", "fact_b_id"])
def test_real_run_refuses_nullable_pair_column(tmp_path, column):
    bank = _nullable_pair_bank(tmp_path / "bank.db", column)

    with pytest.raises(ConflictSchemaUnreadableError):
        migrate_conflict_pair_key(bank)

    assert _index_present(bank) is False
    assert _row_count(bank) == 1  # bank untouched


def test_nullable_refusal_precedes_the_duplicate_sweep(tmp_path):
    """The refusal is about the schema, not the data: the sweep's
    GROUP BY treats NULLs as equal, so before the gate these rows would
    have produced a 'duplicate pairs' report, not the named schema
    refusal. The gate answers first."""
    bank = _nullable_pair_bank(tmp_path / "bank.db", "fact_a_id")
    con = sqlite3.connect(str(bank))
    con.execute("INSERT INTO conflicts (fact_b_id) VALUES ('cf_c')")
    con.execute(
        "INSERT INTO conflicts (fact_a_id, fact_b_id) VALUES (NULL, 'cf_c')"
    )
    con.commit()
    con.close()

    with pytest.raises(ConflictSchemaUnreadableError) as excinfo:
        migrate_conflict_pair_key(bank, dry_run=True)

    assert "nullable" in str(excinfo.value)


def test_cli_dry_run_fails_explicitly_on_nullable_pair_column(
    tmp_path, monkeypatch, capsys
):
    db_path = _cli_bank(tmp_path, monkeypatch)
    _nullable_pair_bank(db_path, "fact_b_id")

    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_migrate(["--dry-run"])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "would add index" not in captured.out
    assert "nullable" in captured.err
    assert _index_present(db_path) is False


# ---------------------------------------------------------------------------
# Round 5 (CodeRabbit review on 34e94af0, 2026-09-28): the existing-index
# path. The Round 4 gate refused nullable pair columns, but ran AFTER the
# already-present-index early return — so a nullable table that already
# carried the canonical index short-circuited to applied=True without the
# schema ever being consulted. Same false green class: an index that
# cannot constrain NULL keys is not honestly "already applied." The gate
# now runs before ANY index is accepted; assert refusal in both modes.
# ---------------------------------------------------------------------------


def _nullable_bank_with_canonical_index(db_path: Path, nullable: str) -> Path:
    """A Round 4 nullable-pair bank with the canonical index created on it.

    SQLite builds the unique index happily over nullable columns — that
    is exactly the trap: creation succeeds while enforcement is defeated.
    """
    bank = _nullable_pair_bank(db_path, nullable)
    con = sqlite3.connect(str(bank))
    con.execute(
        f"CREATE UNIQUE INDEX {INDEX_NAME} "
        "ON conflicts (min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id))"
    )
    con.commit()
    con.close()
    return bank


@pytest.mark.parametrize("column", ["fact_a_id", "fact_b_id"])
def test_dry_run_refuses_canonical_index_on_nullable_table(tmp_path, column):
    bank = _nullable_bank_with_canonical_index(tmp_path / "bank.db", column)

    with pytest.raises(ConflictSchemaUnreadableError) as excinfo:
        migrate_conflict_pair_key(bank, dry_run=True)

    assert "nullable" in str(excinfo.value)
    assert column in str(excinfo.value)


@pytest.mark.parametrize("column", ["fact_a_id", "fact_b_id"])
def test_real_run_refuses_canonical_index_on_nullable_table(tmp_path, column):
    bank = _nullable_bank_with_canonical_index(tmp_path / "bank.db", column)

    with pytest.raises(ConflictSchemaUnreadableError):
        migrate_conflict_pair_key(bank)

    # The refusal is metadata, not demolition: the foreign index object
    # is left as found and no row moved. What E8 refuses is the claim,
    # not the schema object it never owned.
    assert _index_present(bank) is True
    assert _row_count(bank) == 1
