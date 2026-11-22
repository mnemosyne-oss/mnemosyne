#!/usr/bin/env python3
"""
Mnemosyne Legacy Migration Script
=================================

Migrates memories from ephemeral/legacy databases to the PERSISTED canonical path.

CRITICAL for Fly.io / ephemeral VMs: Only ~/.hermes is persisted across restarts!
- Source: ~/.mnemosyne/data/mnemosyne.db (ephemeral — lost on restart)
- Target: ~/.hermes/mnemosyne/data/mnemosyne.db (persisted)

Also migrates legacy mnemosyne_native.db files from earlier versions.

Usage:
    python scripts/migrate_from_legacy.py [--dry-run]

What it does:
1. Scans ephemeral and legacy database paths
2. Copies missing memories into the persisted canonical DB
3. Migrates meaningful non-tool memories into BEAM episodic_memory
4. Promotes high-importance memories into working_memory
5. Preserves all existing data (idempotent — safe to run multiple times)
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Current canonical path (matches mnemosyne.core.beam DEFAULT_DB_PATH)
# NOTE: On Fly.io and other ephemeral VMs, ~/.hermes is the only persisted path
# unless MNEMOSYNE_DATA_DIR explicitly points elsewhere.
_HOME_ROOT = Path(os.environ["HOME"]) if os.environ.get("HOME") else Path.home()
CANONICAL_DATA_DIR = Path(
    os.environ.get("MNEMOSYNE_DATA_DIR")
    or _HOME_ROOT / ".hermes" / "mnemosyne" / "data"
)
CANONICAL_DB = CANONICAL_DATA_DIR / "mnemosyne.db"

# Legacy / ephemeral paths to scan and migrate from
LEGACY_CANDIDATES = [
    _HOME_ROOT / ".mnemosyne" / "data" / "mnemosyne.db",  # ephemeral BEAM data
    _HOME_ROOT / ".mnemosyne" / "data" / "mnemosyne_native.db",
    _HOME_ROOT / ".hermes" / "mnemosyne" / "data" / "mnemosyne_native.db",
]


def ensure_schema(conn: sqlite3.Connection):
    """Ensure the canonical DB has all required BEAM + legacy tables."""
    cursor = conn.cursor()

    # Legacy memories table
    cursor.execute("""
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
    # Some old canonical DBs were created without created_at
    cursor.execute("PRAGMA table_info(memories)")
    mem_cols = [r[1] for r in cursor.fetchall()]
    if "created_at" not in mem_cols:
        cursor.execute("ALTER TABLE memories ADD COLUMN created_at TIMESTAMP")
        cursor.execute("UPDATE memories SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON memories(source)")

    # Legacy embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            model TEXT DEFAULT 'bge-small-en-v1.5',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)

    # BEAM working_memory
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS working_memory (
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_timestamp ON working_memory(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_source ON working_memory(source)")

    # BEAM episodic_memory
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            timestamp TEXT,
            session_id TEXT DEFAULT 'default',
            importance REAL DEFAULT 0.5,
            metadata_json TEXT,
            summary_of TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_session ON episodic_memory(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_timestamp ON episodic_memory(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_source ON episodic_memory(source)")

    # BEAM scratchpad
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scratchpad (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            session_id TEXT DEFAULT 'default',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_session ON scratchpad(session_id)")

    # FTS5 for episodic memory
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes USING fts5(
            content,
            content='episodic_memory',
            content_rowid='rowid'
        )
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS em_ai AFTER INSERT ON episodic_memory BEGIN
            INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS em_ad AFTER DELETE ON episodic_memory BEGIN
            INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
        END
    """)

    conn.commit()


def get_existing_ids(conn: sqlite3.Connection, table: str) -> set:
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM {table}")
    return {row[0] for row in cursor.fetchall()}


def migrate_legacy_db(legacy_path: Path, canonical_conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """Migrate a single legacy database into the canonical one."""
    stats = {"memories_copied": 0, "embeddings_copied": 0, "episodic_migrated": 0, "working_migrated": 0}

    legacy_conn = sqlite3.connect(str(legacy_path))
    legacy_cursor = legacy_conn.cursor()

    # Check what tables exist
    legacy_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in legacy_cursor.fetchall()}

    if "memories" not in tables:
        print(f"  ⚠️  No memories table in {legacy_path} — skipping")
        legacy_conn.close()
        return stats

    canonical_cursor = canonical_conn.cursor()
    existing_memory_ids = get_existing_ids(canonical_conn, "memories")

    # 1. Copy memories
    legacy_cursor.execute("""
        SELECT id, content, source, timestamp, session_id, importance, metadata_json, created_at
        FROM memories
    """)
    rows = legacy_cursor.fetchall()
    to_insert = [row for row in rows if row[0] not in existing_memory_ids]

    if dry_run:
        print(f"  [DRY-RUN] Would copy {len(to_insert)} memories from {legacy_path}")
    else:
        for row in to_insert:
            canonical_cursor.execute("""
                INSERT INTO memories (id, content, source, timestamp, session_id, importance, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, row)
        canonical_conn.commit()
        stats["memories_copied"] = len(to_insert)

    # 2. Copy embeddings if present
    if "memory_embeddings" in tables:
        legacy_cursor.execute("SELECT memory_id, embedding_json, model, created_at FROM memory_embeddings")
        embeddings = legacy_cursor.fetchall()
        canonical_cursor.execute("SELECT memory_id FROM memory_embeddings")
        existing_emb_ids = {row[0] for row in canonical_cursor.fetchall()}
        emb_to_insert = [row for row in embeddings if row[0] not in existing_emb_ids]

        if dry_run:
            print(f"  [DRY-RUN] Would copy {len(emb_to_insert)} embeddings from {legacy_path}")
        else:
            for row in emb_to_insert:
                canonical_cursor.execute("""
                    INSERT INTO memory_embeddings (memory_id, embedding_json, model, created_at)
                    VALUES (?, ?, ?, ?)
                """, row)
            canonical_conn.commit()
            stats["embeddings_copied"] = len(emb_to_insert)

    # 3. Migrate meaningful non-tool memories into episodic_memory
    if not dry_run:
        meaningful = [row for row in rows if row[2] != 'tool_execution' and row[0] not in get_existing_ids(canonical_conn, "episodic_memory")]
        for row in meaningful:
            mid, content, source, timestamp, session_id, importance, metadata_json, created_at = row
            canonical_cursor.execute("""
                INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, metadata_json, summary_of)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (mid, content, source, timestamp, 'hermes_default', importance, metadata_json or '{}', ''))
        canonical_conn.commit()
        stats["episodic_migrated"] = len(meaningful)

        # 4. Promote top high-importance ones into working_memory
        hot = [row for row in meaningful if row[0] not in get_existing_ids(canonical_conn, "working_memory")]
        hot.sort(key=lambda r: (r[5] or 0.5), reverse=True)
        hot = hot[:30]
        for row in hot:
            mid, content, source, timestamp, session_id, importance, metadata_json, created_at = row
            canonical_cursor.execute("""
                INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (mid, content, source, timestamp, 'hermes_default', importance, metadata_json or '{}'))
        canonical_conn.commit()
        stats["working_migrated"] = len(hot)
    else:
        meaningful_count = sum(1 for row in rows if row[2] != 'tool_execution')
        print(f"  [DRY-RUN] Would migrate {meaningful_count} memories to episodic + up to 30 to working")

    legacy_conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy Mnemosyne databases to the current canonical path")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--purge-tools", action="store_true", help="Remove legacy auto-logged tool_execution memories after migration")
    args = parser.parse_args()

    print("=" * 60)
    print("Mnemosyne Legacy Database Migration")
    print("=" * 60)
    print(f"Canonical DB: {CANONICAL_DB}")
    print()

    CANONICAL_DB.parent.mkdir(parents=True, exist_ok=True)
    from mnemosyne.core.beam import _register_cjk_bigram

    canonical_conn = sqlite3.connect(str(CANONICAL_DB))
    _register_cjk_bigram(canonical_conn)
    ensure_schema(canonical_conn)

    # Pre-check stats
    cursor = canonical_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memories")
    pre_total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM memories WHERE source = 'tool_execution'")
    pre_tools = cursor.fetchone()[0]
    print(f"Current canonical DB has {pre_total} memories ({pre_tools} tool_execution)")

    total_stats = {"memories_copied": 0, "embeddings_copied": 0, "episodic_migrated": 0, "working_migrated": 0}
    any_found = False

    for legacy_path in LEGACY_CANDIDATES:
        if legacy_path.exists() and legacy_path.resolve() != CANONICAL_DB.resolve():
            any_found = True
            print(f"\n📁 Found legacy DB: {legacy_path}")
            stats = migrate_legacy_db(legacy_path, canonical_conn, dry_run=args.dry_run)
            for k in total_stats:
                total_stats[k] += stats[k]

    if not any_found and pre_total == 0:
        print("\n✅ No legacy databases found and canonical DB is empty. Nothing to migrate.")
        canonical_conn.close()
        return 0

    # Purge tool_execution noise if requested
    purged_tools = 0
    if args.purge_tools and not args.dry_run:
        cursor.execute("DELETE FROM memories WHERE source = 'tool_execution'")
        cursor.execute("DELETE FROM working_memory WHERE source = 'tool_execution'")
        purged_tools = cursor.rowcount
        canonical_conn.commit()
        print(f"\n🧹 Purged {purged_tools} tool_execution memories from canonical DB")
    elif args.purge_tools and args.dry_run:
        cursor.execute("SELECT COUNT(*) FROM memories WHERE source = 'tool_execution'")
        would_purge = cursor.fetchone()[0]
        print(f"\n[DRY-RUN] Would purge {would_purge} tool_execution memories")

    if args.dry_run:
        print("\n🏁 Dry-run complete. No changes were made.")
    else:
        cursor.execute("SELECT COUNT(*) FROM memories")
        post_total = cursor.fetchone()[0]
        print("\n🏁 Migration complete!")
        print(f"  Memories copied:    {total_stats['memories_copied']}")
        print(f"  Embeddings copied:  {total_stats['embeddings_copied']}")
        print(f"  Episodic migrated:  {total_stats['episodic_migrated']}")
        print(f"  Working promoted:   {total_stats['working_migrated']}")
        if purged_tools:
            print(f"  Tool memories purged: {purged_tools}")
        print(f"  Total in canonical: {post_total} (was {pre_total})")

    canonical_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
