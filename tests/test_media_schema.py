"""Schema tests for media_assets / media_moments (RFC 0003 §2, §1.6)."""

import sqlite3

import pytest

from mnemosyne.core.media import MediaStore, init_media


def _tables(conn):
    return {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _indexes(conn):
    return {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "media.db"


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

def test_init_creates_both_tables_and_every_index(db_path):
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        assert {"media_assets", "media_moments"}.issubset(_tables(conn))
        assert {
            "idx_media_ref", "idx_media_hash", "idx_media_modality",
            "idx_media_session", "idx_moment_asset", "idx_moment_time",
            "idx_moment_memory", "idx_moment_span",
        }.issubset(_indexes(conn))
    finally:
        conn.close()


def test_init_is_idempotent(db_path):
    init_media(db_path)
    init_media(db_path)
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        assert {"media_assets", "media_moments"}.issubset(_tables(conn))
    finally:
        conn.close()


def test_init_does_not_leak_the_connection_it_opens(db_path):
    """``init_media`` opens and closes its own connection. If it leaked one,
    every BeamMemory open would cost a file descriptor forever."""
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("ROLLBACK")
    finally:
        conn.close()


def test_no_column_is_a_blob(db_path):
    """RFC 0004 invariant 3: the brain stores references and text, never bytes.

    A BLOB column here would make byte storage possible by accident, which is
    exactly how a memory system becomes an ambient recorder.
    """
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        for table in ("media_assets", "media_moments"):
            declared = [
                (row[1], (row[2] or "").upper())
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            assert declared, f"{table} has no columns"
            offenders = [name for name, decl in declared if "BLOB" in decl]
            assert offenders == [], (
                f"{table} declares BLOB column(s) {offenders}; media bytes must "
                "never live in the database (RFC 0004 invariant 3)"
            )
    finally:
        conn.close()


def test_no_schema_level_foreign_keys(db_path):
    """Issue #503: ``PRAGMA foreign_keys=ON`` broke 22 tests that intentionally
    create orphan rows. Integrity here is application-layer and reported by
    ``doctor``, not enforced by SQLite."""
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        for table in ("media_assets", "media_moments"):
            assert conn.execute(f"PRAGMA foreign_key_list({table})").fetchall() == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def test_reference_uniqueness_is_enforced_by_index(db_path):
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO media_assets (asset_id, ref_kind, ref_value, modality) "
            "VALUES ('a', 'url', 'http://x/1', 'image')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO media_assets (asset_id, ref_kind, ref_value, modality) "
                "VALUES ('b', 'url', 'http://x/1', 'image')"
            )
    finally:
        conn.close()


def test_content_hash_uniqueness_is_partial_so_nulls_may_repeat(db_path):
    """Reference-only assets whose bytes were never fetched all carry a NULL
    content_hash; the partial index (canonical.py:131-132's trick) is what lets
    many of them coexist while a *known* hash stays unique."""
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "INSERT INTO media_assets (asset_id, ref_kind, ref_value, modality) "
            "VALUES ('a', 'url', 'http://x/1', 'image');"
            "INSERT INTO media_assets (asset_id, ref_kind, ref_value, modality) "
            "VALUES ('b', 'url', 'http://x/2', 'image');"
            "INSERT INTO media_assets (asset_id, content_hash, ref_kind, ref_value, modality) "
            "VALUES ('c', 'deadbeef', 'url', 'http://x/3', 'image');"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO media_assets (asset_id, content_hash, ref_kind, ref_value, modality) "
                "VALUES ('d', 'deadbeef', 'url', 'http://x/4', 'image')"
            )
    finally:
        conn.close()


def test_span_uniqueness_is_per_asset_and_kind(db_path):
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        def insert(moment_id, asset_id, kind, key):
            conn.execute(
                "INSERT INTO media_moments "
                "(moment_id, asset_id, kind, text, span_kind, span_key) "
                "VALUES (?, ?, ?, 't', 'whole', ?)",
                (moment_id, asset_id, kind, key),
            )

        insert("m1", "a", "caption", "v1:whole")
        insert("m2", "b", "caption", "v1:whole")      # different asset — fine
        insert("m3", "a", "ocr", "v1:whole")          # different kind — fine
        with pytest.raises(sqlite3.IntegrityError):
            insert("m4", "a", "caption", "v1:whole")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Migration story
# ---------------------------------------------------------------------------

def test_existing_database_acquires_the_tables_with_no_migration_module(db_path):
    """The destructive downgrade: prove "no migration module needed" rather
    than assume it.

    Build a database, drop the media tables as an older version would have left
    it, then reopen. ``BeamMemory.__init__`` runs the store's DDL on every open
    and every statement is ``IF NOT EXISTS``, so the tables come back -- which
    is exactly how ``canonical_facts`` reached existing databases.
    """
    store = MediaStore(db_path=db_path)
    up = store.upsert_asset(ref_kind="url", ref_value="http://x/1", modality="image")
    assert up.status == "created"
    store.conn.close()

    conn = sqlite3.connect(db_path)
    conn.executescript(
        "DROP TABLE media_moments; DROP TABLE media_assets;"
    )
    conn.commit()
    assert not {"media_assets", "media_moments"} & _tables(conn)
    conn.close()

    reopened = MediaStore(db_path=db_path)
    try:
        assert {"media_assets", "media_moments"}.issubset(_tables(reopened.conn))
        # The data is gone (it was dropped) but the schema is whole again.
        assert reopened.get_asset(up.asset_id) is None
        again = reopened.upsert_asset(
            ref_kind="url", ref_value="http://x/1", modality="image"
        )
        assert again.asset_id == up.asset_id, "asset_id must be deterministic"
    finally:
        reopened.conn.close()


def test_store_shares_a_caller_owned_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        store = MediaStore(db_path=db_path, conn=conn)
        assert store.conn is conn
        assert {"media_assets", "media_moments"}.issubset(_tables(conn))
    finally:
        conn.close()


def test_beam_exposes_media_on_the_shared_connection(tmp_path):
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="media-wiring", db_path=tmp_path / "beam.db")
    try:
        assert hasattr(beam, "media")
        # Same connection object — no extra file descriptor per bank.
        assert beam.media.conn is beam.conn
        assert {"media_assets", "media_moments"}.issubset(_tables(beam.conn))
    finally:
        try:
            beam.conn.close()
        except Exception:
            pass


def test_no_vec_table_is_created(db_path):
    """A moment is an ordinary working_memory row. A vec0 table would cost four
    whitelist edits and a permanent rowid-alignment discipline (RFC 0003 §1.3)."""
    init_media(db_path)
    conn = sqlite3.connect(db_path)
    try:
        names = _tables(conn) | _indexes(conn)
        assert not any("moment" in n and n.startswith("vec") for n in names)
        assert "vec_moments" not in names
    finally:
        conn.close()
