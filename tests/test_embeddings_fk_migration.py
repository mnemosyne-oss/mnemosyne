"""Tests for legacy memory_embeddings FK migration (#451).

Verifies that:
1. Databases with the old FK constraint are migrated automatically on init_beam()
2. Embedding writes succeed with PRAGMA foreign_keys=ON after migration
3. Fresh databases created via memory.py do NOT carry the FK
4. The migration is idempotent (safe to run multiple times)
5. Existing embedding data is preserved through migration (including payload)
"""
import sqlite3
import tempfile

from pathlib import Path


def _create_legacy_db(db_path) -> None:
    """Create a database with the OLD memory_embeddings schema (with FK)."""
    conn = sqlite3.connect(str(db_path))
    # Create memories table first (FK target)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            timestamp TEXT,
            session_id TEXT DEFAULT 'default',
            importance REAL DEFAULT 0.5,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create memory_embeddings WITH the legacy FK
    conn.execute("""
        CREATE TABLE memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            model TEXT DEFAULT 'bge-small-en-v1.5',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)
    # Insert some test data
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
        ("test-embedding-1", "[0.1, 0.2, 0.3]"),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
        ("test-embedding-2", "[0.4, 0.5, 0.6]"),
    )
    conn.commit()
    conn.close()


def _get_schema(conn, table):
    """Get the CREATE statement for a table."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
    ).fetchone()
    return row[0] if row else ""


def test_legacy_fk_migrated_on_init():
    """Databases with the old FK constraint should be migrated automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        _create_legacy_db(db_path)

        # Verify FK exists before init_beam
        conn = sqlite3.connect(str(db_path))
        assert "REFERENCES memories" in _get_schema(conn, "memory_embeddings")
        conn.close()

        # Run init_beam (which includes the migration)
        from mnemosyne.core.beam import init_beam
        init_beam(db_path)

        # Verify FK is gone after init_beam
        conn = sqlite3.connect(str(db_path))
        schema = _get_schema(conn, "memory_embeddings")
        assert "REFERENCES memories" not in schema, (
            f"FK should be removed, but schema still has it: {schema}"
        )
        conn.close()


def test_embeddings_save_with_foreign_keys_on():
    """After migration, embeddings save successfully with PRAGMA foreign_keys=ON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        _create_legacy_db(db_path)

        from mnemosyne.core.beam import init_beam
        init_beam(db_path)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys=ON")

        # Try to insert a new embedding (should succeed after migration)
        conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model) "
            "VALUES (?, ?, ?)",
            ("new-test-id", "[0.7, 0.8, 0.9]", "test-model"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT memory_id, model FROM memory_embeddings WHERE memory_id = ?",
            ("new-test-id",),
        ).fetchone()
        assert row is not None, "Embedding insert should succeed"
        assert row[0] == "new-test-id"
        assert row[1] == "test-model"
        conn.close()


def test_existing_data_preserved_through_migration():
    """Existing embedding rows (including payload) should survive the migration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        _create_legacy_db(db_path)

        from mnemosyne.core.beam import init_beam
        init_beam(db_path)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT memory_id, embedding_json FROM memory_embeddings ORDER BY memory_id"
        ).fetchall()
        assert len(rows) >= 2, f"Expected at least 2 rows, got {len(rows)}"
        assert rows[0][0] == "test-embedding-1"
        assert rows[1][0] == "test-embedding-2"
        # Verify the actual embedding payload survived intact (CodeRabbit finding #2)
        assert rows[0][1] == "[0.1, 0.2, 0.3]"
        assert rows[1][1] == "[0.4, 0.5, 0.6]"
        conn.close()


def test_migration_is_idempotent():
    """Running init_beam multiple times should be safe (no data loss)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        _create_legacy_db(db_path)

        from mnemosyne.core.beam import init_beam
        init_beam(db_path)  # First migration
        init_beam(db_path)  # Second run — should be no-op

        conn = sqlite3.connect(str(db_path))
        schema = _get_schema(conn, "memory_embeddings")
        assert "REFERENCES memories" not in schema
        rows = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
        assert rows >= 2, f"Data should be preserved, got {rows} rows"
        conn.close()


def test_fresh_db_has_no_fk():
    """Fresh databases created via memory.py should NOT have the FK."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        from mnemosyne.core.memory import init_db
        init_db(db_path)

        conn = sqlite3.connect(str(db_path))
        schema = _get_schema(conn, "memory_embeddings")
        assert "REFERENCES memories" not in schema, (
            f"Fresh DB should not have FK, but schema has it: {schema}"
        )
        conn.close()


def test_migration_handles_column_order_mismatch():
    """Migration should work even if legacy table has columns in different order.

    Regression test for explicit column list (CodeRabbit finding):
    ensures INSERT uses named columns, not positional SELECT *.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create a legacy table with columns in REVERSED order + extra column
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                timestamp TEXT,
                session_id TEXT DEFAULT 'default',
                importance REAL DEFAULT 0.5,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Reversed order: created_at first, then model, embedding_json, memory_id
        # Plus an extra legacy column that shouldn't exist in the new schema
        conn.execute("""
            CREATE TABLE memory_embeddings (
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model TEXT DEFAULT 'bge-small-en-v1.5',
                embedding_json TEXT NOT NULL,
                memory_id TEXT PRIMARY KEY,
                legacy_extra TEXT DEFAULT 'should_not_appear',
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)
        conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding_json, model, created_at) VALUES (?, ?, ?, ?)",
            ("reordered-1", "[0.1, 0.2]", "test-model", "2026-01-01 00:00:00"),
        )
        conn.commit()
        conn.close()

        from mnemosyne.core.beam import init_beam
        init_beam(db_path)

        conn = sqlite3.connect(str(db_path))
        schema = _get_schema(conn, "memory_embeddings")
        assert "REFERENCES memories" not in schema, "FK should be removed"

        # Data should be mapped to correct columns, not positional
        row = conn.execute(
            "SELECT memory_id, embedding_json, model FROM memory_embeddings WHERE memory_id = ?",
            ("reordered-1",),
        ).fetchone()
        assert row is not None, "Row should survive migration"
        assert row[0] == "reordered-1", f"memory_id mismatch: {row[0]}"
        assert row[1] == "[0.1, 0.2]", f"embedding_json mismatch: {row[1]}"
        assert row[2] == "test-model", f"model mismatch: {row[2]}"
        conn.close()
