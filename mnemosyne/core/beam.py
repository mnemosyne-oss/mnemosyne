"""
Mnemosyne BEAM Architecture
============================
Bilevel Episodic-Associative Memory

Three SQLite tables:
- working_memory: hot, recent context (auto-injected into prompts)
- episodic_memory: long-term storage with native vector + FTS5 search
- scratchpad: temporary agent reasoning workspace

Native sqlite-vec for vector search.
FTS5 for full-text retrieval.
Hybrid ranking: 50% vector + 30% FTS rank + 20% importance.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import json
import hashlib
import logging
import threading
import math

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set, Union
from pathlib import Path


logger = logging.getLogger(__name__)

# Typed memory classification (Phase 1 -- zero overhead, pattern-based)
try:
    from mnemosyne.core.typed_memory import classify_memory, MemoryType
except ImportError:
    classify_memory = None
    MemoryType = None

# Binary vector compression (Phase 2 -- Moorcheh ITS)
try:
    from mnemosyne.core.binary_vectors import (
        BinaryVectorStore,
        maximally_informative_binarization as _mib,
        hamming_distance as _hamming,
        EMBEDDING_DIM,
        BYTES_PER_VECTOR,
    )
except ImportError:
    _mib = None
    _hamming = None

# Episodic graph + veracity consolidation (Phases 3-4)
try:
    from mnemosyne.core.episodic_graph import EpisodicGraph, GraphEdge
except ImportError:
    EpisodicGraph = None
    GraphEdge = None
try:
    from mnemosyne.core.veracity_consolidation import (
        VeracityConsolidator,
        VERACITY_WEIGHTS,
        clamp_veracity,
        aggregate_veracity,
    )
    # Alias used below to construct STATED_WEIGHT et al. -- same dict as
    # the canonical VERACITY_WEIGHTS so changes propagate.
    _VW_DEFAULTS = VERACITY_WEIGHTS
except ImportError:
    VeracityConsolidator = None
    VERACITY_WEIGHTS = {}
    # Hardcoded backstop so degraded-import mode doesn't crash module
    # load when constructing STATED_WEIGHT et al. (C1 review fix).
    _VW_DEFAULTS = {
        "stated": 1.0,
        "inferred": 0.7,
        "tool": 0.5,
        "imported": 0.6,
        "unknown": 0.8,
    }

    # Surface degraded mode at import time so operators see ONE signal
    # in startup logs that the canonical helper isn't available. Without
    # this, the fallback silently clamps every bad label with no audit
    # trail across the run.
    logger.warning(
        "mnemosyne.core.veracity_consolidation unavailable; using fallback "
        "clamp_veracity. Non-canonical veracity labels will be clamped "
        "silently (no per-call WARNING). Operators should resolve the "
        "import to restore full audit logging."
    )

    def aggregate_veracity(source_veracities) -> str:
        """Fallback aggregator when veracity_consolidation is unavailable.
        Returns 'unknown' unconditionally so consolidation doesn't crash."""
        return "unknown"

    def clamp_veracity(raw, *, context: str = "veracity") -> str:
        """Fallback when veracity_consolidation is unavailable.
        Mirrors the canonical helper's API and clamps non-canonical
        labels to 'unknown'. Does NOT log per-call warnings -- the
        import-time warning above is the audit signal. Operators
        should fix the import to restore full observability.
        """
        if raw is None:
            return "unknown"
        norm = str(raw).strip().lower()
        if not norm:
            return "unknown"
        # Without the canonical allowlist available, fall back to the
        # known-safe set inline. Drift between this literal and the
        # canonical set is bounded by the fact that this branch
        # only fires when the import is broken.
        if norm in {"stated", "inferred", "tool", "imported", "unknown"}:
            return norm
        return "unknown"

# ------------------------------------------------------------------
# Trust tier derivation from ingestion source (plugin-first architecture)
# ------------------------------------------------------------------
TRUST_TIER_MAP = {
    "conversation": "STATED",       # Direct user input via agent
    "user":          "STATED",       # Explicit user action
    "cli":           "STATED",       # CLI direct user input
    "mcp":           "EXTERNAL_WRITE",  # External MCP tool calls
    "import":        "IMPORTED",     # Bulk import from file
    "mem0":          "IMPORTED",     # External service import
    "honcho_import": "IMPORTED",     # Honcho data migration
    "honcho_summary":"IMPORTED",     # Honcho auto-summary
    "consolidation": "DERIVED",      # System sleep/summarize output
    "sleep_consolidation": "DERIVED", # Sleep cycle output
    "regex":         "DERIVED",      # Automated regex extraction
    "extraction":    "DERIVED",      # LLM fact extraction
    "unknown":       "STATED",       # Unknown source, conservative default
}

def _source_to_trust_tier(source: str) -> str:
    """Map ingestion source to trust_tier for prompt-injection defense.

    Plugin-first design: callers describe WHAT they are (via `source`),
    Mnemosyne decides HOW to trust it (via trust_tier mapping). New
    ingestion paths only need to set `source` honestly — the mapping
    centralizes the trust policy.
    """
    if not source:
        return "STATED"
    # Direct match first
    if source in TRUST_TIER_MAP:
        return TRUST_TIER_MAP[source]
    # Heuristic fallback: any source containing 'import' → IMPORTED
    if "import" in source.lower():
        return "IMPORTED"
    # Any source containing 'mcp' → EXTERNAL_WRITE
    if "mcp" in source.lower():
        return "EXTERNAL_WRITE"
    # Conservative default: treat as direct user input
    return "STATED"

try:
    import numpy as np
except ImportError:
    np = None

from mnemosyne.core import embeddings as _embeddings
from mnemosyne.core import plugins as _plugins

# sqlite-vec optional dependency
try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except Exception:
    _SQLITE_VEC_AVAILABLE = False
    sqlite_vec = None

_thread_local = threading.local()

# On Fly.io and other ephemeral VMs, only ~/.hermes is persisted.
# Default to the legacy Hermes path so memories survive restarts.
DEFAULT_DATA_DIR = Path.home() / ".hermes" / "mnemosyne" / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"

import os

def _env_truthy(name: str) -> bool:
    """Parse an env var as truthy. Accepts `1`/`true`/`yes`/`on`
    (case-insensitive, whitespace-stripped). Everything else
    (including unset, empty, garbage) is False.

    Complement of `_env_disabled` (defined below) -- they exist for
    different default-state use cases. Use `_env_truthy` when the
    feature is default-OFF and an env var opts it on; use
    `_env_disabled` when the feature is default-ON and an env var
    opts it off.

    Mirrors the helper of the same name in `tools/evaluate_beam_end_to_end.py`
    for env-parsing consistency across the codebase.
    """
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


# BEAM benchmark optimizations (opt-in via env var, zero impact on production)
# When enabled: broader FTS5 OR semantics, larger vector scan limits, always-include vectors.
# Set MNEMOSYNE_BEAM_OPTIMIZATIONS=1 to activate for BEAM benchmarking only.
_BEAM_MODE = _env_truthy("MNEMOSYNE_BEAM_OPTIMIZATIONS")

if os.environ.get("MNEMOSYNE_DATA_DIR"):
    DEFAULT_DATA_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR"))
    DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"


def _default_data_dir() -> Path:
    """Return the current default data directory, honoring runtime env changes."""
    if os.environ.get("MNEMOSYNE_DATA_DIR"):
        return Path(os.environ["MNEMOSYNE_DATA_DIR"])
    return DEFAULT_DATA_DIR


def _default_db_path() -> Path:
    """Return the current default DB path, honoring runtime env changes."""
    return _default_data_dir() / "mnemosyne.db"

# Config
try:
    _emb_dim = int(os.environ.get("MNEMOSYNE_EMBEDDING_DIM", "384"))
    EMBEDDING_DIM = _emb_dim if _emb_dim > 0 else 384
except ValueError:
    EMBEDDING_DIM = 384  # bge-small-en-v1.5
WORKING_MEMORY_MAX_ITEMS = int(os.environ.get("MNEMOSYNE_WM_MAX_ITEMS", "10000"))
WORKING_MEMORY_TTL_HOURS = int(os.environ.get("MNEMOSYNE_WM_TTL_HOURS", "24"))
EPISODIC_RECALL_LIMIT = int(os.environ.get("MNEMOSYNE_EP_LIMIT", "50000"))
SLEEP_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_SLEEP_BATCH", "5000"))
SCRATCHPAD_MAX_ITEMS = int(os.environ.get("MNEMOSYNE_SP_MAX", "1000"))
RECENCY_HALFLIFE_HOURS = float(os.environ.get("MNEMOSYNE_RECENCY_HALFLIFE", "168"))  # 1 week default

# Tiered episodic degradation
TIER2_DAYS = int(os.environ.get("MNEMOSYNE_TIER2_DAYS", "30"))
TIER3_DAYS = int(os.environ.get("MNEMOSYNE_TIER3_DAYS", "180"))
TIER1_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER1_WEIGHT", "1.0"))
TIER2_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER2_WEIGHT", "0.5"))
TIER3_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER3_WEIGHT", "0.25"))
DEGRADE_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_DEGRADE_BATCH", "100"))
SMART_COMPRESS = os.environ.get("MNEMOSYNE_SMART_COMPRESS", "1") not in ("0", "false", "no")
TIER3_MAX_CHARS = int(os.environ.get("MNEMOSYNE_TIER3_MAX_CHARS", "300"))


def _env_disabled(name: str) -> bool:
    """A/B toggle helper: return True iff the env var is explicitly
    set to a falsy value (`0`/`false`/`no`/`off`, case-insensitive,
    whitespace-stripped).

    Used by experiment ablation toggles where the feature is ON by
    default (production behavior) and operators can disable it
    explicitly via env var. Distinct from `_env_truthy` from the
    benchmark harness -- that one defaults to OFF, this one defaults
    to ON. See `docs/benchmarking.md` for the full toggle reference.

    Unset / empty / non-falsy → False (feature enabled).
    """
    val = os.environ.get(name, "").strip().lower()
    return val in ("0", "false", "no", "off")

# Veracity weighting (memory confidence). C29: defaults come from
# `_VW_DEFAULTS` which mirrors `veracity_consolidation.VERACITY_WEIGHTS`
# in normal mode and falls back to a hardcoded literal in degraded-import
# mode (the import block above sets it). Single source of truth for the
# consolidator's Bayesian compounding and recall's veracity multiplier.
# Env-var overrides remain so operators can tune ranking; documented
# drift risk: if `MNEMOSYNE_*_WEIGHT` is set, recall scoring diverges
# from consolidation confidence math (consolidator doesn't honor env).
def _env_float(name: str, default: float) -> float:
    """Parse an env var as float; fall back to `default` on empty or
    invalid values rather than crashing at module load.

    Pre-fix `float(os.environ.get("MNEMOSYNE_STATED_WEIGHT", "1.0"))`
    raised ValueError when the env var was set to empty (`export
    MNEMOSYNE_STATED_WEIGHT=`) because `os.environ.get` returns `""`
    (the value), not the default -- `float("")` then crashed import
    BEFORE the C32 override-WARN could fire. Restored from PR #91
    after the merge stripped it.
    """
    raw = os.environ.get(name, "")
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a valid float; falling back to default %s",
            name, raw[:80], default,
        )
        return default


# Veracity weighting (memory confidence)
STATED_WEIGHT = _env_float("MNEMOSYNE_STATED_WEIGHT", _VW_DEFAULTS["stated"])
INFERRED_WEIGHT = _env_float("MNEMOSYNE_INFERRED_WEIGHT", _VW_DEFAULTS["inferred"])
TOOL_WEIGHT = _env_float("MNEMOSYNE_TOOL_WEIGHT", _VW_DEFAULTS["tool"])
IMPORTED_WEIGHT = _env_float("MNEMOSYNE_IMPORTED_WEIGHT", _VW_DEFAULTS["imported"])
UNKNOWN_WEIGHT = _env_float("MNEMOSYNE_UNKNOWN_WEIGHT", _VW_DEFAULTS["unknown"])


def _detect_veracity_weight_overrides() -> List[str]:
    """C32: return a list of `MNEMOSYNE_*_WEIGHT` env vars set to a
    non-empty value. Filters out empty-string values (`export
    MNEMOSYNE_STATED_WEIGHT=`) since `_env_float` falls back to default
    on empties -- counting them would confuse the WARN message.
    """
    return [
        name for name in (
            "MNEMOSYNE_STATED_WEIGHT",
            "MNEMOSYNE_INFERRED_WEIGHT",
            "MNEMOSYNE_TOOL_WEIGHT",
            "MNEMOSYNE_IMPORTED_WEIGHT",
            "MNEMOSYNE_UNKNOWN_WEIGHT",
        )
        if os.environ.get(name, "").strip()
    ]


_VERACITY_WARN_EMITTED = False


def _warn_about_veracity_weight_overrides(force: bool = False) -> bool:
    """Log a WARNING if any `MNEMOSYNE_*_WEIGHT` env var is overridden.

    Idempotent per-process: subsequent calls return False without
    re-emitting unless `force=True` (tests use this to verify the WARN
    fires per call). Multi-worker setups (uvicorn `--workers`,
    pytest-xdist) get one WARN per process instead of N per startup.
    """
    global _VERACITY_WARN_EMITTED
    if _VERACITY_WARN_EMITTED and not force:
        return False
    overrides = _detect_veracity_weight_overrides()
    if not overrides:
        return False
    logger.warning(
        "Veracity weight env overrides detected: %s. Recall scoring will "
        "honor the override, but consolidation Bayesian compounding "
        "(veracity_consolidation.VERACITY_WEIGHTS) does NOT -- the two "
        "will drift. Set matching values in veracity_consolidation.py "
        "OR accept that 'consolidated-as-N also ranks at N' invariant "
        "is broken until the consolidator is taught the same overrides.",
        ", ".join(overrides),
    )
    _VERACITY_WARN_EMITTED = True
    return True


_warn_about_veracity_weight_overrides()

# Vector compression: float32 | int8 | bit
VEC_TYPE = os.environ.get("MNEMOSYNE_VEC_TYPE", "int8").lower()
if VEC_TYPE not in ("float32", "int8", "bit"):
    VEC_TYPE = "float32"


def _get_connection(db_path: Path = None) -> sqlite3.Connection:
    """Get thread-local database connection with extensions loaded.

    Returns a `_BeamConnection` (sqlite3.Connection subclass) so
    `remember_batch`'s enrichment loop can defer commits via
    `_deferred_commits`. Connection is otherwise identical to a
    plain sqlite3.Connection.
    """
    path = Path(db_path) if db_path else _default_db_path()
    if not hasattr(_thread_local, 'conn') or _thread_local.conn is None or getattr(_thread_local, 'db_path', None) != str(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
            factory=_BeamConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if _SQLITE_VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
            except Exception:
                pass  # Some environments don't support load_extension
        _thread_local.conn = conn
        _thread_local.db_path = str(path)
    return _thread_local.conn


def _detect_vec_type(conn: sqlite3.Connection) -> str:
    """
    Detect whether sqlite-vec supports int8/bit.
    Falls back to float32 if the requested type is unavailable.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return "float32"
    if VEC_TYPE == "float32":
        return "float32"
    cursor = conn.cursor()
    test_type = VEC_TYPE  # int8 or bit
    try:
        cursor.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS _vec_test USING vec0(embedding {test_type}[{EMBEDDING_DIM}])")
        cursor.execute("DROP TABLE IF EXISTS _vec_test")
        conn.commit()
        return test_type
    except Exception:
        conn.rollback()
        # Try int8 as fallback from bit
        if test_type == "bit":
            try:
                cursor.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS _vec_test USING vec0(embedding int8[{EMBEDDING_DIM}])")
                cursor.execute("DROP TABLE IF EXISTS _vec_test")
                conn.commit()
                return "int8"
            except Exception:
                conn.rollback()
        return "float32"


def init_beam(db_path: Path = None):
    """Initialize BEAM schema."""
    conn = _get_connection(db_path)
    cursor = conn.cursor()

    # --- WORKING MEMORY ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS working_memory (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            timestamp TEXT,
            session_id TEXT DEFAULT 'default',
            importance REAL DEFAULT 0.5,
            metadata_json TEXT,
            veracity TEXT DEFAULT 'unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_timestamp ON working_memory(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_source ON working_memory(source)")

    # --- EPISODIC MEMORY ---
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
            veracity TEXT DEFAULT 'unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_session ON episodic_memory(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_timestamp ON episodic_memory(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_source ON episodic_memory(source)")

    # --- Tiered degradation migration (v2.3) ---
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN tier INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN degraded_at TEXT")
    except sqlite3.OperationalError:
        pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_tier ON episodic_memory(tier)")

    # --- Veracity migration (v2.4) ---
    try:
        cursor.execute("ALTER TABLE working_memory ADD COLUMN veracity TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN veracity TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass

    # --- Typed memory migration (Phase 1) ---
    try:
        cursor.execute("ALTER TABLE working_memory ADD COLUMN memory_type TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN memory_type TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass

    # --- Binary vector migration (Phase 2) ---
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN binary_vector BLOB")
    except sqlite3.OperationalError:
        pass

    # --- E3 additive sleep migration ---
    # Working memories that sleep() has consolidated into an episodic
    # summary get this timestamp set. Pre-E3 sleep() DELETEd those rows;
    # post-E3 the originals remain so they're still recallable, and
    # consolidated_at IS NULL is the predicate sleep uses to find
    # not-yet-consolidated rows.
    #
    # Naming note: episodic_memory.metadata_json["consolidated_at"]
    # (introduced in 2.5 by the heal-quality pipeline) records when a
    # summary row was finalized; this column records when a SOURCE row
    # was marked done by sleep. Same concept, different angle.
    _e3_column_added = False
    try:
        cursor.execute("ALTER TABLE working_memory ADD COLUMN consolidated_at TEXT")
        _e3_column_added = True
    except sqlite3.OperationalError as exc:
        # Only swallow "duplicate column" -- every other OperationalError
        # (database locked, disk I/O, readonly, missing table) must
        # surface so callers don't proceed with a broken schema.
        if "duplicate column" not in str(exc).lower():
            raise

    if _e3_column_added:
        # Pre-E3 backfill: existing rows are treated as already-consolidated.
        # Without this, the first post-upgrade sleep would treat the entire
        # pre-existing backlog as "not yet consolidated" and try to summarize
        # everything at once -- including rows pre-E3 sleep would have already
        # DELETEd. The backfill preserves the pre-E3 expectation that "old
        # rows are gone." Cost: a single UPDATE on existing rows at upgrade
        # time. Idempotent: this branch only fires when the column was just
        # added, so re-running init_beam is a no-op.
        cursor.execute(
            "UPDATE working_memory SET consolidated_at = ? "
            "WHERE consolidated_at IS NULL",
            (datetime.now().isoformat(),),
        )

    # Partial index for the sleep eligibility predicate. Sleep scans
    # WHERE session_id = ? AND timestamp < ? AND consolidated_at IS NULL
    # on every cycle; once consolidated rows accumulate the predicate
    # becomes the dominant filter. The partial index lets the planner
    # skip already-consolidated rows in O(eligible) instead of O(session).
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_unconsolidated "
        "ON working_memory(session_id, timestamp) "
        "WHERE consolidated_at IS NULL"
    )

    # --- SCRATCHPAD ---
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

    # Detect supported vector type
    effective_vec_type = _detect_vec_type(conn)

    # --- sqlite-vec VIRTUAL TABLE ---
    if _SQLITE_VEC_AVAILABLE:
        try:
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(
                    embedding {effective_vec_type}[{EMBEDDING_DIM}]
                )
            """)
        except sqlite3.OperationalError:
            pass  # May already exist or extension not loadable

    # --- FTS5 VIRTUAL TABLE for episodic ---
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes USING fts5(
            content,
            content='episodic_memory',
            content_rowid='rowid'
        )
    """)

    # --- FTS5 VIRTUAL TABLE for working memory (autonomous) ---
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_working USING fts5(
            id UNINDEXED,
            content
        )
    """)

    # --- FTS5 Triggers for episodic ---
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
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS em_au AFTER UPDATE ON episodic_memory BEGIN
            INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
            INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
        END
    """)

    # --- FTS5 Triggers for working memory ---
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS wm_ai AFTER INSERT ON working_memory BEGIN
            INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS wm_ad AFTER DELETE ON working_memory BEGIN
            DELETE FROM fts_working WHERE id = old.id;
        END
    """)
    # The wm_au trigger restricts to UPDATE OF content so sleep's
    # consolidated_at marker writes don't churn the FTS index. Pre-E3
    # this trigger fired on every UPDATE -- fine when UPDATEs were rare;
    # post-E3 sleep marks SLEEP_BATCH_SIZE rows per cycle and would
    # otherwise generate 2*N FTS round-trips per sleep with no content
    # delta. SQLite column-list triggers handle the perf concern.
    cursor.execute("DROP TRIGGER IF EXISTS wm_au")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS wm_au AFTER UPDATE OF content ON working_memory BEGIN
            DELETE FROM fts_working WHERE id = old.id;
            INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
        END
    """)

    # --- Consolidation Log ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            items_consolidated INTEGER,
            summary_preview TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- memory_embeddings: fallback for environments without sqlite-vec ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # --- Migration: recall tracking columns (v2.1) ---
    _add_column_if_missing(conn, "working_memory", "recall_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "working_memory", "last_recalled", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "recall_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "episodic_memory", "last_recalled", "TIMESTAMP DEFAULT NULL")

    # --- Migration: temporal validity + scope (v2.2) ---
    _add_column_if_missing(conn, "working_memory", "valid_until", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "superseded_by", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "scope", "TEXT DEFAULT 'global'")
    _add_column_if_missing(conn, "episodic_memory", "valid_until", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "superseded_by", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "scope", "TEXT DEFAULT 'global'")

    # --- NAI-0 Covering Indexes (v2.5) ---
    cursor.execute("""CREATE INDEX IF NOT EXISTS idx_em_scope_imp
        ON episodic_memory(scope, importance) WHERE superseded_by IS NULL""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS idx_wm_session_recall
        ON working_memory(session_id, last_recalled) WHERE valid_until IS NULL""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS idx_mem_emb_type
        ON memory_embeddings(memory_id, model)""")

    # --- Migration: multi-agent identity layer (v2.1) ---
    _add_column_if_missing(conn, "working_memory", "author_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "author_type", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "channel_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "author_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "author_type", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "channel_id", "TEXT DEFAULT NULL")
    # --- Migration: trust tier for prompt-injection defense (v2.6) ---
    _add_column_if_missing(conn, "working_memory", "trust_tier", "TEXT DEFAULT 'STATED'")
    _add_column_if_missing(conn, "episodic_memory", "trust_tier", "TEXT DEFAULT 'STATED'")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_author ON working_memory(author_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_channel ON working_memory(channel_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_author ON episodic_memory(author_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_channel ON episodic_memory(channel_id)")

    # --- FACTS (LLM-extracted structured knowledge) ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            fact_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            timestamp TEXT,
            source_msg_id TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_msg_id)")

    # FTS5 for full-text search on facts
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts USING fts5(
            subject, predicate, object, content='facts'
        )
    """)
    # Triggers to keep FTS5 in sync
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO fts_facts(rowid, subject, predicate, object)
            VALUES (new.rowid, new.subject, new.predicate, new.object);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
            INSERT INTO fts_facts(fts_facts, rowid, subject, predicate, object)
            VALUES ('delete', old.rowid, old.subject, old.predicate, old.object);
        END
    """)

    # Vector table for facts (sqlite-vec)
    try:
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                embedding {effective_vec_type}[{EMBEDDING_DIM}]
            )
        """)
    except (sqlite3.OperationalError, RuntimeError):
        pass  # sqlite-vec not available


class _BeamConnection(sqlite3.Connection):
    """sqlite3.Connection subclass that supports deferring commits.

    Used by BeamMemory so `remember_batch`'s enrichment loop can wrap
    many sub-helper commits in a single transaction. The substores
    (AnnotationStore, EpisodicGraph, VeracityConsolidator) each call
    `self.conn.commit()` after their per-row writes; pre-E2-hardening
    that produced 10-15 commits per batch row × 250K rows = millions
    of fsync round-trips. /review army (4-source CRITICAL on commit 1)
    estimated 3-10 hours wall clock for the BEAM-recovery benchmark.

    When `_defer_commit` is True, `commit()` becomes a no-op. The
    `_deferred_commits` context manager flips the flag, runs the
    block, then calls `_real_commit()` once at the end (or rolls back
    on exception).

    Subclassing is required because `sqlite3.Connection.commit` is a
    read-only C-level method -- monkey-patching it raises
    `AttributeError`. The factory= parameter on `sqlite3.connect` is
    the supported integration point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._defer_commit = False

    def commit(self) -> None:
        if self._defer_commit:
            return
        super().commit()

    def _real_commit(self) -> None:
        """Force a real commit regardless of the defer flag.
        Used by `_deferred_commits` on successful exit."""
        super().commit()


@contextlib.contextmanager
def _deferred_commits(conn: sqlite3.Connection):
    """Suppress nested commit() calls so the caller can wrap many
    sub-helpers in a single transaction.

    Pairs with `_BeamConnection`'s `_defer_commit` flag. If the
    passed connection isn't a `_BeamConnection` (e.g., a test
    constructed `BeamMemory` with a raw sqlite3 connection, or a
    legacy caller built its own conn), the context manager degrades
    to a no-op -- inner commits still fire, performance regression
    isn't fixed for that code path but correctness is preserved.

    Threading: `_BeamConnection._defer_commit` is per-connection.
    BeamMemory uses thread-local connections (see _get_connection),
    so the flag is visible only to the calling thread. A future
    refactor that shares the connection across threads would need
    a lock here.
    """
    is_beam_conn = isinstance(conn, _BeamConnection)
    if not is_beam_conn:
        # Degrade gracefully: inner commits fire as before. This
        # keeps the path callable from tests that build conns
        # manually but loses the batching perf win on that code path.
        yield
        return

    conn._defer_commit = True
    try:
        yield
    except Exception:
        conn._defer_commit = False
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise
    else:
        conn._defer_commit = False
        try:
            conn._real_commit()
        except sqlite3.Error as exc:
            logger.error(
                "_deferred_commits: final commit failed: %s; "
                "rolling back the buffered transaction",
                exc,
            )
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            raise
    finally:
        # Defense in depth: clear the flag on any control-flow path.
        conn._defer_commit = False


def _generate_id(content: str) -> str:
    return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str):
    """Safely add a column if it doesn't already exist (SQLite migration helper)."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()


def _normalize_weights(vec_weight: Optional[float], fts_weight: Optional[float],
                       importance_weight: Optional[float]) -> tuple[float, float, float]:
    """
    Normalize hybrid scoring weights to sum to 1.0.

    Falls back to env vars, then defaults:
        vec_weight      -> MNEMOSYNE_VEC_WEIGHT      -> 0.5
        fts_weight      -> MNEMOSYNE_FTS_WEIGHT      -> 0.3
        importance_weight -> MNEMOSYNE_IMPORTANCE_WEIGHT -> 0.2

    After normalization: vw + fw + iw == 1.0
    """
    vw = vec_weight if vec_weight is not None else float(os.environ.get("MNEMOSYNE_VEC_WEIGHT", "0.5"))
    fw = fts_weight if fts_weight is not None else float(os.environ.get("MNEMOSYNE_FTS_WEIGHT", "0.3"))
    iw = importance_weight if importance_weight is not None else float(os.environ.get("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.2"))

    # Clamp to non-negative
    vw = max(0.0, vw)
    fw = max(0.0, fw)
    iw = max(0.0, iw)

    total = vw + fw + iw
    if total == 0.0:
        # All zero = revert to defaults
        return (0.5, 0.3, 0.2)

    return (vw / total, fw / total, iw / total)


def _recency_decay(timestamp_str: str, halflife_hours: float = RECENCY_HALFLIFE_HOURS) -> float:
    """Calculate recency decay factor. 1.0 = brand new, ~0.5 = one halflife old.
    
    Exponential decay based on age. Returns 0.5 for unknown/invalid timestamps.
    """
    if not timestamp_str:
        return 0.5  # Unknown age = neutral
    try:
        ts = datetime.fromisoformat(timestamp_str)
        age_hours = (datetime.now() - ts).total_seconds() / 3600.0
        return math.exp(-age_hours / halflife_hours)
    except Exception:
        return 0.5


def _parse_query_time(query_time: Optional[Union[str, datetime]]) -> datetime:
    """Parse query_time parameter into a datetime object.

    - None -> datetime.now()
    - str  -> parsed from ISO format
    - datetime -> returned as-is
    """
    if query_time is None:
        return datetime.now()
    if isinstance(query_time, datetime):
        return query_time
    if isinstance(query_time, str):
        # Try ISO format with various precisions
        try:
            return datetime.fromisoformat(query_time)
        except ValueError:
            # Try appending time if only date provided
            try:
                return datetime.fromisoformat(f"{query_time}T00:00:00")
            except ValueError:
                raise ValueError(f"Invalid query_time format: {query_time!r}. Expected ISO datetime string.")
    raise TypeError(f"query_time must be str, datetime, or None; got {type(query_time).__name__}")


# Fast-path timestamp parsing cache
_TS_CACHE: Dict[str, datetime] = {}
_TS_CACHE_MAX = 2000


def _parse_ts_fast(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp with LRU-style cache for performance."""
    if not ts:
        return None
    cached = _TS_CACHE.get(ts)
    if cached is not None:
        return cached
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if len(_TS_CACHE) >= _TS_CACHE_MAX:
        _TS_CACHE.clear()
    _TS_CACHE[ts] = dt
    return dt


def _temporal_boost(memory_timestamp_str: str, query_time: datetime,
                    halflife_hours: float = 24.0) -> float:
    """Temporal boost factor based on proximity to query_time.

    Formula: exp(-hours_delta / halflife)
    - memory at query_time -> boost = 1.0
    - memory 1 halflife away -> boost = exp(-1) ≈ 0.368
    - memory 3 halflives away -> boost = exp(-3) ≈ 0.050

    Returns 0.0 for invalid timestamps or future timestamps (clamped to now).
    """
    ts = _parse_ts_fast(memory_timestamp_str)
    if ts is None:
        return 0.0

    # Clamp future timestamps to query_time (no negative deltas)
    if ts > query_time:
        ts = query_time

    hours_delta = (query_time - ts).total_seconds() / 3600.0
    return math.exp(-hours_delta / halflife_hours)


def _vec_available(conn: sqlite3.Connection) -> bool:
    if not _SQLITE_VEC_AVAILABLE:
        return False
    try:
        conn.execute("SELECT 1 FROM vec_episodes LIMIT 0")
        return True
    except Exception:
        return False


def _extract_and_store_entities(beam: "BeamMemory", memory_id: str, content: str):
    """
    Extract entities from content and store as annotations (post-E6).
    Called internally by remember() when extract_entities=True.

    Pre-E6 wrote to TripleStore with predicate="mentions", which silently
    invalidated prior mentions on the same memory via auto-invalidation
    on (subject, predicate). Post-E6, writes go to AnnotationStore where
    multiple mentions per memory coexist.
    """
    try:
        from mnemosyne.core.entities import extract_entities_regex

        entities = extract_entities_regex(content)
        if not entities:
            return

        # Reuse BeamMemory's shared AnnotationStore (cached on the beam
        # instance, shares the thread-local connection). UNIQUE constraint
        # on (memory_id, kind, value) plus INSERT OR IGNORE makes this
        # idempotent -- re-extraction on duplicate-content writes is a no-op.
        beam.annotations.add_many(
            memory_id=memory_id,
            kind="mentions",
            values=entities,
            source="regex",
            confidence=0.8,
        )
    except Exception:
        # Entity extraction is best-effort; never fail remember() because of it
        pass


def _extract_and_store_facts(beam: "BeamMemory", memory_id: str, content: str, source: str = ""):
    """
    Extract structured facts from content using LLM and store as annotations
    + facts table. Called internally by remember() when extract=True.

    Stores in TWO places:
    1. AnnotationStore with kind="fact" (post-E6; was TripleStore pre-E6)
    2. facts table (structured SPO facts for fact_recall())

    Post-E6 note: writes formerly used TripleStore.add_facts() which
    silently invalidated each prior fact via (subject, predicate) auto-
    invalidation. AnnotationStore.add_many is append-only so all facts
    coexist.
    """
    try:
        from mnemosyne.core.extraction import extract_facts_safe
        from mnemosyne.core.annotations import filter_facts

        facts = extract_facts_safe(content)
        if not facts:
            return

        # Filter to match the legacy filtering applied by TripleStore.add_facts.
        kept = filter_facts(facts)
        if kept:
            beam.annotations.add_many(
                memory_id=memory_id,
                kind="fact",
                values=kept,
                source=source,
                confidence=0.7,
            )

        # ALSO store in facts table (new cloud extraction path) -- uses the
        # full facts list (matching pre-E6 behavior).
        _store_facts_in_table(beam, memory_id, content, source, facts)

    except Exception:
        # Fact extraction is best-effort; never fail remember() because of it
        pass


def _store_facts_in_table(beam: "BeamMemory", memory_id: str,
                          content: str, source: str, facts: list):
    """Store extracted free-text facts as simple SPO entries in the facts table."""
    import hashlib
    cursor = beam.conn.cursor()
    timestamp = __import__('datetime').datetime.now().isoformat()
    
    for i, fact_text in enumerate(facts):
        # Derive subject from source, predicate = "stated", object = fact text
        subject = source or "user"
        fact_id = hashlib.sha256(
            f"{memory_id}:fact:{i}:{fact_text[:50]}".encode()
        ).hexdigest()[:24]

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO facts
                (fact_id, session_id, subject, predicate, object,
                 timestamp, source_msg_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id,
                beam.session_id,
                subject,
                "stated",
                fact_text,
                timestamp,
                memory_id,
                0.7,
            ))
        except Exception:
            continue  # Best-effort per fact
    
    beam.conn.commit()


def _find_memories_by_entity(beam: "BeamMemory", entity_name: str, threshold: float = 0.8) -> List[str]:
    """
    Find memory IDs that mention an entity (or similar entity via fuzzy match).
    Returns list of memory_id strings.

    Post-E6: reads from AnnotationStore. Memories with multiple mentions
    now all surface (silent-destruction bug fixed) -- the pre-E6 path
    against TripleStore returned only the last-written mention per memory
    because of auto-invalidation on (subject, predicate).
    """
    try:
        from mnemosyne.core.entities import find_similar_entities

        # Get all known entities (uses BeamMemory's cached AnnotationStore)
        known_entities = beam.annotations.get_distinct_values("mentions")
        if not known_entities:
            return []

        # Find similar entities
        matches = find_similar_entities(entity_name, known_entities, threshold=threshold)

        # Collect memory IDs for all matched entities
        memory_ids: Set[str] = set()
        for matched_entity, _ in matches:
            results = beam.annotations.query_by_kind("mentions", value=matched_entity)
            for row in results:
                memory_ids.add(row["memory_id"])

        return list(memory_ids)
    except Exception:
        return []


def _find_memories_by_fact(beam: "BeamMemory", query: str) -> List[str]:
    """
    Find memory IDs that have extracted facts matching the query.
    Does simple keyword matching against stored fact annotations.
    Returns list of memory_id strings.

    Post-E6: reads from AnnotationStore. Memories with multiple extracted
    facts now all surface (silent-destruction bug fixed).
    """
    try:
        # Get all fact annotations (uses BeamMemory's cached AnnotationStore)
        all_facts = beam.annotations.query_by_kind("fact")
        if not all_facts:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Simple keyword matching against fact text
        memory_ids: Set[str] = set()
        for fact_row in all_facts:
            fact_text = fact_row.get("value", "").lower()
            # Check if any query word appears in the fact
            if any(word in fact_text for word in query_words):
                memory_ids.add(fact_row["memory_id"])
            # Also check if the full query is a substring of the fact
            elif query_lower in fact_text:
                memory_ids.add(fact_row["memory_id"])

        return list(memory_ids)
    except Exception:
        return []


def _in_memory_vec_search(conn: sqlite3.Connection, query_embedding: np.ndarray, k: int = 20) -> List[Dict]:
    """Fallback vector search using memory_embeddings table + numpy cosine similarity."""
    if np is None:
        return []
    cursor = conn.cursor()
    # Join with episodic_memory (not memories) since that's where BEAM stores consolidated data
    cursor.execute("""
        SELECT em.rowid, me.memory_id, me.embedding_json
        FROM memory_embeddings me
        JOIN episodic_memory em ON me.memory_id = em.id
        LIMIT 10000
    """)
    rows = cursor.fetchall()
    if not rows:
        return []

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    query_unit = query_embedding / query_norm

    results = []
    for row in rows:
        try:
            vec = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = float(np.dot(query_unit, vec / vec_norm))
            # Convert similarity to distance-like metric (1 - sim) for consistent ranking
            results.append({"rowid": row["rowid"], "distance": 1.0 - sim})
        except Exception:
            continue

    results.sort(key=lambda x: x["distance"])
    return results[:k]


def _effective_vec_type(conn: sqlite3.Connection) -> str:
    """Re-detect the actual vector type used by vec_episodes."""
    if not _vec_available(conn):
        return "float32"
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_episodes'"
        ).fetchone()
        if row and "int8" in row[0]:
            return "int8"
        if row and "bit" in row[0]:
            return "bit"
    except Exception:
        pass
    return "float32"


def _vec_insert(conn: sqlite3.Connection, rowid: int, embedding: List[float]):
    """Insert embedding into sqlite-vec table with quantization via SQL functions."""
    vec_type = _effective_vec_type(conn)
    emb_json = json.dumps(embedding)
    if vec_type == "bit":
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_binary(?))",
            (rowid, emb_json)
        )
    elif vec_type == "int8":
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
            (rowid, emb_json)
        )
    else:
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (rowid, emb_json)
        )


def _vec_search(conn: sqlite3.Connection, embedding: List[float], k: int = 20) -> List[Dict]:
    """Search sqlite-vec and return rowids with distances."""
    vec_type = _effective_vec_type(conn)
    emb_json = json.dumps(embedding)
    # NOTE: sqlite-vec requires the KNN limit to be known at query planning time.
    # Parameter binding (LIMIT ?) fails on some versions because xBestIndex
    # can't resolve the parameter value. We inline k safely since it's
    # always an integer computed internally.
    k = int(k)
    if vec_type == "bit":
        rows = conn.execute(
            f"SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH vec_quantize_binary(?) ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    elif vec_type == "int8":
        rows = conn.execute(
            f"SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH vec_quantize_int8(?, 'unit') ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH ? ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    return [{"rowid": r["rowid"], "distance": r["distance"]} for r in rows]


def _fts_search(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Dict]:
    """Search FTS5 episodes and return rowids with ranks.
    Strips FTS5-special characters, keeps alphanumeric + spaces.
    In BEAM mode: filters stop-words, uses OR semantics for broader recall."""
    import re as _re
    safe_query = _re.sub(r'[^\w\s]', ' ', query)
    safe_query = ' '.join(safe_query.split())  # Collapse whitespace
    if not safe_query.strip():
        return []
    
    # BEAM mode: OR semantics with stop-word filtering for benchmark recall breadth
    if _BEAM_MODE:
        _stop_words = {'when','does','do','did','what','how','where','which','who','why',
                       'is','are','was','were','can','will','would','should','could','may',
                       'the','a','an','in','on','at','to','for','of','with','my','me','i','you'}
        content_words = [w for w in safe_query.split() if w.lower() not in _stop_words and len(w) > 1]
        if not content_words:
            content_words = [w for w in safe_query.split() if len(w) > 1]
        # BEAM mode: if stop-word filtering leaves only 1 word, include ALL original
        # non-stop-word tokens (not just content_words) to broaden recall
        original_words = [w for w in query.split() if w.lower() not in _stop_words and len(w) > 1]
        if len(content_words) <= 1 and len(original_words) > 1:
            fts_query = " OR ".join(original_words)
        else:
            fts_query = " OR ".join(content_words)
        if not fts_query:
            return []
    else:
        fts_query = safe_query
    
    rows = conn.execute(
        "SELECT rowid, rank FROM fts_episodes WHERE fts_episodes MATCH ? ORDER BY rank, rowid LIMIT ?",
        (fts_query, k)
    ).fetchall()
    return [{"rowid": r["rowid"], "rank": r["rank"]} for r in rows]


def _fts_search_working(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Dict]:
    """Search FTS5 working memory and return ids with ranks.
    Strips FTS5-special characters, keeps alphanumeric + spaces.
    In BEAM mode: filters stop-words, uses OR semantics for broader recall."""
    import re as _re
    safe_query = _re.sub(r'[^\w\s]', ' ', query)
    safe_query = ' '.join(safe_query.split())  # Collapse whitespace
    if not safe_query.strip():
        return []
    
    # BEAM mode: OR semantics with stop-word filtering for benchmark recall breadth
    if _BEAM_MODE:
        _stop_words = {'when','does','do','did','what','how','where','which','who','why',
                       'is','are','was','were','can','will','would','should','could','may',
                       'the','a','an','in','on','at','to','for','of','with','my','me','i','you'}
        content_words = [w for w in safe_query.split() if w.lower() not in _stop_words and len(w) > 1]
        if not content_words:
            content_words = [w for w in safe_query.split() if len(w) > 1]
        fts_query = " OR ".join(content_words)
        if not fts_query:
            return []
    else:
        fts_query = safe_query
    
    rows = conn.execute(
        "SELECT id, rank FROM fts_working WHERE fts_working MATCH ? ORDER BY rank, id LIMIT ?",
        (fts_query, k)
    ).fetchall()

    # BEAM mode: if phrase query returns 0, fall back to individual word OR search
    # This handles cases like "What operating system" where no single entry has
    # all content words but individual words like "operating" or "system" may match
    if not rows and _BEAM_MODE and len(content_words) > 1:
        fts_query_fallback = " OR ".join(content_words)
        rows = conn.execute(
            "SELECT id, rank FROM fts_working WHERE fts_working MATCH ? ORDER BY rank, id LIMIT ?",
            (fts_query_fallback, k)
        ).fetchall()

    return [{"id": r["id"], "rank": r["rank"]} for r in rows]


def _wm_vec_search(conn: sqlite3.Connection, query_embedding, k: int = 20) -> List[Dict]:
    """Vector search against working_memory via memory_embeddings table.
    Returns list of dicts with 'id' (memory_id) and 'sim' (cosine similarity)."""
    if np is None:
        return []
    cursor = conn.cursor()
    try:
        # BEAM mode: scan up to 500K rows for broad vector recall on large benchmark datasets
        _vec_limit = 500000 if _BEAM_MODE else 50000
        cursor.execute("""
            SELECT wm.id, me.embedding_json
            FROM memory_embeddings me
            JOIN working_memory wm ON me.memory_id = wm.id
            WHERE wm.superseded_by IS NULL
              AND (wm.valid_until IS NULL OR wm.valid_until > ?)
            LIMIT ?
        """, (datetime.now().isoformat(), _vec_limit))
    except Exception:
        return []
    rows = cursor.fetchall()
    if not rows:
        return []

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    query_unit = query_embedding / query_norm

    results = []
    for row in rows:
        try:
            vec = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = float(np.dot(query_unit, vec / vec_norm))
            results.append({"id": row["id"], "sim": sim})
        except Exception:
            continue

    results.sort(key=lambda x: x["sim"], reverse=True)
    return results[:k]


class BeamMemory:
    """
    BEAM memory interface.
    """

    def __init__(self, session_id: str = "default", db_path: Path = None,
                 author_id: str = None, author_type: str = None,
                 channel_id: str = None, use_cloud: bool = False,
                 event_emitter: "Optional[Callable[[Any], None]]" = None):
        self.session_id = session_id
        self.author_id = author_id
        self.author_type = author_type
        self.channel_id = channel_id or session_id  # default channel = session
        # Coerce path-like inputs (e.g. tempfile-produced strings) to a
        # Path so downstream consumers like _get_connection that do
        # `path.parent.mkdir(...)` don't blow up with
        # ``AttributeError: 'str' object has no attribute 'parent'``.
        # The previous contract was implicit (Path-only); making it
        # explicit here is backward-compatible -- a real Path stays a
        # Path -- and unbreaks every caller that passes a string,
        # including the test_identity_memory.py fixtures added by
        # PR #106 which now run on every PR's CI matrix.
        if db_path is not None and not isinstance(db_path, Path):
            db_path = Path(db_path)
        self.db_path = db_path or _default_db_path()
        self.use_cloud = use_cloud  # Enable LLM fact extraction during remember()
        self._extraction_client = None  # Lazy-loaded ExtractionClient
        self._extraction_buffer = []  # Buffer for batch extraction
        self._event_emitter = event_emitter  # Streaming event callback
        self.conn = _get_connection(self.db_path)
        init_beam(self.db_path)

        # E6: ensure schema split + auto-migrate legacy TripleStore rows
        # to AnnotationStore. Honors MNEMOSYNE_AUTO_MIGRATE=0 for operators
        # who want explicit control. See:
        # - mnemosyne/migrations/e6_triplestore_split.py
        # - .hermes/ledger/memory-contract.md (E6)
        # Also ensure the legacy `triples` table exists -- the post-E6
        # production path no longer writes to it, but external scripts
        # (scripts/backfill_temporal_triples.py) and deprecation-period
        # callers of TripleStore still expect the table to be present.
        try:
            from mnemosyne.core.triples import init_triples
            init_triples(db_path=self.db_path)
        except Exception:
            pass
        self._ensure_e6_schema_with_migration()

        # E6: shared AnnotationStore handle reusing this BeamMemory's
        # thread-local connection. Production call sites use `self.annotations`
        # instead of constructing fresh AnnotationStore(...) per call --
        # eliminates the per-call file-descriptor cost the post-E6 review
        # surfaced (every extraction/recall opened 2 connections + ran DDL).
        from mnemosyne.core.annotations import AnnotationStore
        self.annotations = AnnotationStore(db_path=self.db_path, conn=self.conn)

        # Phase 3: Episodic graph (shared connection)
        self.episodic_graph = None
        if EpisodicGraph is not None:
            try:
                self.episodic_graph = EpisodicGraph(conn=self.conn, db_path=self.db_path)
            except Exception:
                pass

        # Phase 4: Veracity consolidator (shared connection)
        self.veracity_consolidator = None
        if VeracityConsolidator is not None:
            try:
                self.veracity_consolidator = VeracityConsolidator(conn=self.conn, db_path=self.db_path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # E6 schema split + auto-migration
    # ------------------------------------------------------------------
    def _ensure_e6_schema_with_migration(self) -> None:
        """Ensure the AnnotationStore schema exists; auto-migrate legacy
        TripleStore rows on first run with a pre-E6 database.

        Idempotent. Safe to call on fresh installs (no triples table to
        migrate) and on databases that have already been migrated.

        Respects ``MNEMOSYNE_AUTO_MIGRATE=0`` for operators who want
        explicit control over schema migrations. When auto-migration is
        disabled and a migration would have been required, log a clear
        warning pointing at the manual migration script -- the AnnotationStore
        schema is still created so downstream code can run, but legacy rows
        remain in the triples table until the operator runs the script.

        Failures are caught and logged; init does not raise. The provider
        layer's silent-fail pattern (C27) would mask any exception we
        raised here, so logging is the visible channel for now. The user-
        facing pattern is "migration ran (or didn't), continue with
        whatever schema state we have."
        """
        import os
        from mnemosyne.core.annotations import ANNOTATION_KINDS, init_annotations

        logger = logging.getLogger(__name__)

        # Always ensure the annotations table exists (cheap, idempotent).
        try:
            init_annotations(self.db_path)
        except Exception as e:
            logger.error("E6: failed to initialize annotations schema: %s", e)
            return

        # Honor opt-out for operators who want explicit migrations only.
        if os.environ.get("MNEMOSYNE_AUTO_MIGRATE", "1") == "0":
            # If a migration would be needed, leave a warning so operators
            # see something concrete in their logs.
            try:
                cursor = self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='triples'"
                )
                if cursor.fetchone() is not None:
                    placeholders = ",".join("?" * len(ANNOTATION_KINDS))
                    cursor = self.conn.execute(
                        f"SELECT COUNT(*) FROM triples WHERE predicate IN ({placeholders})",
                        tuple(ANNOTATION_KINDS),
                    )
                    pending = cursor.fetchone()[0]
                    if pending > 0:
                        logger.warning(
                            "E6: MNEMOSYNE_AUTO_MIGRATE=0 and %d annotation "
                            "rows remain in the legacy triples table. Run "
                            "`python scripts/migrate_triplestore_split.py "
                            "--db %s` to migrate manually.",
                            pending,
                            self.db_path,
                        )
            except Exception as e:
                logger.debug("E6: opt-out probe failed: %s", e)
            return

        # Auto-migrate path. The migration logic lives inside the package
        # (mnemosyne.migrations.e6_triplestore_split) so pip-installed
        # deployments get the same auto-migrate behavior as source checkouts.
        # No filesystem-relative path resolution; just import.
        try:
            from mnemosyne.migrations.e6_triplestore_split import (
                migrate as _e6_migrate,
                has_pending_migration as _e6_has_pending,
            )

            # Fast-path: cheap index-driven existence check before any
            # heavyweight classify scan / Python-side set diff. Most BeamMemory
            # inits on a post-migration DB end here in microseconds.
            if not _e6_has_pending(self.conn):
                return

            # Flush any pending writes on our connection (init_beam commits
            # internally, but be defensive). The migration opens its own
            # connection; under WAL mode multiple connections to the same
            # SQLite file coexist without us closing ours.
            try:
                self.conn.commit()
            except Exception:
                pass

            written = _e6_migrate(
                db_path=self.db_path,
                dry_run=False,
                backup=True,
                log_fn=lambda line: logger.info("E6 migrate: %s", line),
            )
            if written > 0:
                logger.warning(
                    "E6: auto-migrated %d annotation rows from triples → "
                    "annotations. Backup is at %s.pre_e6_backup "
                    "(from this run if newly created, or an earlier run if "
                    "the file already existed). "
                    "Set MNEMOSYNE_AUTO_MIGRATE=0 to disable auto-migration.",
                    written,
                    self.db_path,
                )
        except Exception as e:
            logger.error(
                "E6: auto-migration failed (continuing init with current schema "
                "state). Run `python scripts/migrate_triplestore_split.py "
                "--db %s` manually. Error: %s",
                self.db_path,
                e,
            )

    # ------------------------------------------------------------------
    # Working Memory
    # ------------------------------------------------------------------
    def _find_duplicate(self, content: str) -> Optional[str]:
        """Check if exact same content already exists in working_memory for this session.
        Returns the existing memory_id if found, else None."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM working_memory
            WHERE session_id = ? AND content = ?
            LIMIT 1
        """, (self.session_id, content))
        row = cursor.fetchone()
        return row["id"] if row else None

    def _emit_event(self, event_type, memory_id: str, content: str = None,
                    source: str = None, importance: float = None,
                    metadata: Dict = None, delta: Dict = None) -> None:
        """Fire a streaming event if an emitter is registered."""
        if self._event_emitter is None:
            return
        try:
            from mnemosyne.core.streaming import MemoryEvent, EventType
            evt_type = EventType[event_type] if isinstance(event_type, str) else event_type
            event = MemoryEvent(
                event_type=evt_type,
                memory_id=memory_id,
                session_id=self.session_id,
                content=content,
                source=source,
                importance=importance,
                metadata=metadata,
                delta=delta,
            )
            self._event_emitter(event)
        except Exception:
            pass  # Streaming failures must never block memory operations

    def remember(self, content: str, source: str = "conversation",
                 importance: float = 0.5, metadata: Dict = None,
                 valid_until: str = None, scope: str = "session",
                 memory_id: str = None,
                 extract_entities: bool = False,
                 extract: bool = False,
                 veracity: str = "unknown",
                 trust_tier: str = None) -> str:
        """Store into working_memory. Deduplicates exact content matches.

        When called from the legacy-compatible Mnemosyne.remember() path,
        memory_id is passed through so the legacy memories row and BEAM
        working_memory row stay addressable by the same ID. Direct BEAM calls
        still generate their own deterministic ID.

        Args:
            content: The text to remember
            source: Origin of the memory (e.g., "conversation", "document")
            importance: 0.0-1.0 relevance score
            metadata: Optional dict of additional fields
            valid_until: ISO timestamp when this memory expires
            scope: "session" or "global"
            memory_id: Optional pre-generated ID from legacy layer
            extract_entities: If True, extract and store entity mentions as triples
            extract: If True, extract structured facts from content using LLM
                and store as triples. Default False.
            veracity: Confidence level -- 'stated', 'inferred', 'tool', 'imported', 'unknown'.
                Non-canonical labels are clamped to 'unknown' with a WARNING
                (mirrors the C12.b clamp at the hermes_memory_provider boundary).
        """
        # Clamp veracity at the BeamMemory.remember entry too -- the
        # method is the lowest-level public ingest path under BeamMemory,
        # so consistency with remember_batch and the provider
        # boundary requires clamping here. Pre-E4 the column was raw;
        # the new recall multiplier means non-canonical labels would
        # silently fall through to UNKNOWN_WEIGHT at scoring time.
        veracity = clamp_veracity(veracity, context="remember")

    # --- Content sanitization: extract binary payloads to blob storage ---
        from mnemosyne.core.content_sanitizer import sanitize_content as _sanitize
        sanitized_content, blob_meta = _sanitize(content)
        if blob_meta:
            metadata = (metadata or {}).copy()
            metadata["_blob"] = blob_meta
            content = sanitized_content

        # --- Auto-derive trust_tier from source if not explicitly set ---
        if trust_tier is None:
            trust_tier = _source_to_trust_tier(source)
        # Clamp to known tiers
        if trust_tier not in ("STATED", "DERIVED", "EXTERNAL_WRITE", "IMPORTED"):
            trust_tier = "STATED"

        # --- Typed memory classification (Phase 1 -- zero overhead) ---
        memory_type = None
        if classify_memory is not None:
            try:
                result = classify_memory(content)
                memory_type = result.memory_type.value
            except Exception:
                pass  # Classifier failures are non-blocking

        # --- Deduplication: exact match ---
        existing_id = self._find_duplicate(content)
        if existing_id:
            cursor = self.conn.cursor()
            # Dedup-update clears consolidated_at so a re-remembered row
            # becomes eligible for sleep again. Without this, an already-
            # consolidated row that the user reasserts is permanently
            # skipped -- its fresher timestamp/source/scope never produces
            # a fresh summary. Pre-E3 this scenario didn't exist because
            # consolidated rows were deleted; the additive design has to
            # opt back in.
            # E4.a.1 review fix (P1): refresh veracity on dedup-update too.
            # Without this, a row first stored as 'unknown' and later
            # re-remembered as 'stated' kept the stale 'unknown' label,
            # which E4.a.1's sleep-time aggregator then propagates into
            # the episodic summary -- defeating the trust-signal refresh.
            # Conservative policy: only upgrade if the new call passes a
            # non-'unknown' veracity (preserves per-row trust on
            # backfills that don't carry a meaningful veracity arg).
            cursor.execute("""
                UPDATE working_memory
                SET importance = MAX(importance, ?), timestamp = ?, source = ?,
                    valid_until = COALESCE(?, valid_until),
                    scope = COALESCE(?, scope),
                    author_id = COALESCE(?, author_id),
                    author_type = COALESCE(?, author_type),
                    channel_id = COALESCE(?, channel_id),
                    memory_type = COALESCE(?, memory_type),
                    veracity = CASE WHEN ? != 'unknown' THEN ? ELSE veracity END,
                    trust_tier = COALESCE(?, trust_tier),
                    consolidated_at = NULL
                WHERE id = ? AND session_id = ?
            """, (importance, datetime.now().isoformat(), source,
                  valid_until, scope,
                  self.author_id, self.author_type, self.channel_id,
                  memory_type,
                  veracity, veracity,
                  trust_tier,
                  existing_id, self.session_id))
            self.conn.commit()
            # Run the same entity/fact extraction the new-row path runs, so
            # backfill calls -- `mem.remember(same_content, extract=True)` on
            # an already-existing row -- actually populate the triples and
            # facts tables. Without this the dedup early-return silently
            # skips everything `extract=True` advertises, breaking the
            # contract on duplicate-content writes (see C12.a /review note).
            if extract_entities:
                _extract_and_store_entities(self, existing_id, content)
            if extract:
                _extract_and_store_facts(self, existing_id, content, source)
            # Phase 3-4: Extract graph and consolidate veracity for dedup update
            self._ingest_graph_and_veracity(existing_id, content, source, veracity)
            self._emit_event("MEMORY_UPDATED", existing_id, content=content,
                             source=source, importance=importance, metadata=metadata)
            return existing_id

        memory_id = memory_id or _generate_id(content)
        timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO working_memory
            (id, content, source, timestamp, session_id, importance, metadata_json, valid_until, scope,
             author_id, author_type, channel_id, veracity, memory_type, trust_tier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, content, source, timestamp, self.session_id, importance,
              json.dumps(metadata or {}), valid_until, scope,
              self.author_id, self.author_type, self.channel_id, veracity, memory_type, trust_tier))
        self.conn.commit()
        self._trim_working_memory()

        # Auto-generate temporal triple
        self._add_temporal_triple(memory_id, timestamp, source, content)

        # --- Entity extraction ---
        if extract_entities:
            _extract_and_store_entities(self, memory_id, content)

        # --- Structured fact extraction ---
        if extract:
            _extract_and_store_facts(self, memory_id, content, source)

        # Phase 3-4: Extract graph and consolidate veracity for new memory
        self._ingest_graph_and_veracity(memory_id, content, source, veracity)

        self._emit_event("MEMORY_ADDED", memory_id, content=content,
                         source=source, importance=importance, metadata=metadata)
        return memory_id

    def remember_batch(self, items: List[Dict],
                       *,
                       veracity: Optional[str] = None,
                       force_veracity: bool = False,
                       trust_tier: str = "IMPORTED",
                       extract_entities: bool = False,
                       extract: bool = False) -> List[str]:
        """
        Batch insert into working_memory for high-throughput ingestion.
        Each item dict should have keys: content, source, importance,
        metadata (optional), veracity (optional).

        Legal veracity values: 'stated', 'inferred', 'tool', 'imported',
        'unknown'. None / empty / whitespace silently → 'unknown'.
        Non-canonical non-empty labels emit a WARNING and clamp to
        'unknown'.

        veracity (method-level kwarg): default applied to items that
            don't supply their own `veracity` key.

        force_veracity (default False): security knob. When True, the
            method-level `veracity` is applied to EVERY row uniformly
            and per-item `item["veracity"]` is IGNORED (warning logged
            per item if present so the operator sees the override).
            Use this when the caller is the authority on trust --
            e.g., an importer ingesting LLM-generated content that
            shouldn't be able to self-elevate its label. Pre-E4 the
            per-item override was harmless because veracity didn't
            affect ranking; post-E4 it gates a real ranking signal
            so callers consuming untrusted content need this knob.
            When False (default), per-item `veracity` keys override
            the method default -- preserves the legitimate use case
            of mixed-trust batches (e.g., user messages='stated',
            tool observations='tool').

        All values are clamped to the canonical allowlist via
        `clamp_veracity` (mirrors C12.b at the hermes_memory_provider
        trust boundary). remember_batch is the high-throughput path
        used by importers, the BEAM benchmark adapter, and batch
        ingest CLIs where label quality varies.

        Pre-E4 the column defaulted to 'unknown' for every batch row;
        recall's veracity multiplier collapsed to a constant 0.8
        (global scale factor instead of rank signal). The recall
        scorer at beam.py::recall now applies the multiplier to
        working_memory hits too, so per-row veracity differentiates
        scores at the experiment level.

        E2 -- Enrichment parity with `remember()`:
            Post-E2 this method runs the same post-insert enrichment
            pipeline `remember()` runs unconditionally:
              - `_add_temporal_triple` writes the row's date as an
                `occurred_on` annotation + the source kind as a
                `has_source` annotation (zero-LLM, just date string
                slicing).
              - `_ingest_graph_and_veracity` runs pattern-based gist +
                fact extraction via `EpisodicGraph` and consolidates
                the extracted facts into `consolidated_facts` weighted
                by per-row veracity (`VeracityConsolidator`). Zero LLM
                -- rule-based / regex pattern matching only.

            Without this fix any high-throughput ingest path bypassed
            the enrichment layer entirely, leaving the polyphonic
            engine's `graph` and `fact` voices with no data to fuse --
            E5's RRF over 4 voices collapsed to 2 voices in practice.

        extract_entities (default False): opt-in regex entity scan
            via `_extract_and_store_entities`. Cheap but generates
            additional annotation rows; off by default to keep batch
            ingest stable for non-experiment callers.

        extract (default False): opt-in LLM-based structured fact
            extraction via `_extract_and_store_facts`. Real cloud-API
            cost per row; off by default. The BEAM-recovery experiment
            arm that tests LLM enrichment sets this True.

        New behavior change for existing batch callers: the always-on
        pattern-based enrichment now adds ~ms-per-row CPU cost (regex
        + a few SQLite inserts). For typical importers (10k-100k
        rows) this is a few seconds of additional latency; for the
        BEAM benchmark's 250k-message ingest, ~minutes. Documented in
        CHANGELOG.
        """
        cursor = self.conn.cursor()
        ids = []
        # Carry per-row source + veracity through to enrichment so we
        # don't re-derive them post-insert. Keyed by memory_id rather
        # than indexed-by-position (post-/review M4 -- dict eliminates
        # the parallel-list class of refactor bug, and works under
        # python -O where the prior `assert mid_check == memory_id`
        # would have stripped).
        meta_by_id: Dict[str, Tuple[str, str]] = {}  # mid → (source, veracity)
        timestamp = datetime.now().isoformat()
        # Clamp the method-level default once, not per row -- operators
        # who pass a bad default should see one warning, not N.
        default_veracity = clamp_veracity(
            veracity, context="remember_batch.default"
        )
        for item in items:
            # --- Content sanitization: extract binary payloads to blob storage ---
            from mnemosyne.core.content_sanitizer import sanitize_content as _sanitize
            raw_content = item["content"]
            sanitized_content, blob_meta = _sanitize(raw_content)
            if blob_meta:
                item["content"] = sanitized_content
                item_meta = item.get("metadata") or {}
                item["metadata"] = {**item_meta, "_blob": blob_meta}

            memory_id = _generate_id(item["content"])
            ids.append(memory_id)
            # Typed memory classification
            item_type = None
            if classify_memory is not None:
                try:
                    result = classify_memory(item["content"])
                    item_type = result.memory_type.value
                except Exception:
                    pass
            # Per-item override semantics gated by force_veracity. In
            # strict mode (force_veracity=True) per-item keys are
            # ignored -- the caller is the trust authority. Otherwise
            # per-item overrides the method-level default. Either way
            # the final value passes through clamp_veracity at the
            # trust boundary.
            if force_veracity:
                if "veracity" in item:
                    logger.warning(
                        "remember_batch.force_veracity=True; "
                        "ignoring per-item veracity %r in favor of "
                        "method-level default %r",
                        item["veracity"], default_veracity,
                    )
                item_veracity = default_veracity
            elif "veracity" in item:
                item_veracity = clamp_veracity(
                    item["veracity"], context="remember_batch.per_item"
                )
            else:
                item_veracity = default_veracity
            item_source = item.get("source", "conversation")
            meta_by_id[memory_id] = (item_source, item_veracity)
            cursor.execute("""
                INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, metadata_json,
                author_id, author_type, channel_id, memory_type, veracity, trust_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                item["content"],
                item_source,
                timestamp,
                self.session_id,
                item.get("importance", 0.5),
                json.dumps(item.get("metadata") or {}),
                item.get("author_id", self.author_id),
                item.get("author_type", self.author_type),
                item.get("channel_id", self.channel_id),
                item_type,
                item_veracity,
                trust_tier,
            ))
        self.conn.commit()
        
        # Generate vector embeddings for working memory hybrid search.
        # E2.a.10: pre-fix this block had two silent-failure modes:
        # (a) if `embed()` returns a shorter array than inputs (partial
        # fastembed failure), `vectors[i]` raises IndexError mid-loop →
        # caught by bare except → the entire batch's embeddings are
        # lost since cursor execution stops at the IndexError row;
        # (b) any embed failure during ingest was silent. At 250K-row
        # scale the vector voice (35% RRF weight) would silently bias
        # toward earlier-ingested rows without any operator signal.
        # Fix: length-mismatch check + WARNING log on the swallow.
        if _embeddings.available():
            try:
                contents = [item["content"] for item in items]
                vectors = _embeddings.embed(contents)
                if vectors is None:
                    logger.warning(
                        "remember_batch: _embeddings.embed returned None for "
                        "batch of %d items -- no vectors stored, vector voice "
                        "will miss these rows",
                        len(contents),
                    )
                elif len(vectors) != len(contents):
                    logger.warning(
                        "remember_batch: embedding count mismatch (%d vectors "
                        "for %d inputs) -- skipping vector storage for this "
                        "batch to avoid partial-alignment errors",
                        len(vectors), len(contents),
                    )
                else:
                    model = _embeddings._DEFAULT_MODEL
                    for i, memory_id in enumerate(ids):
                        emb_json = _embeddings.serialize(vectors[i])
                        cursor.execute(
                            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
                            (memory_id, emb_json, model)
                        )
            except Exception as exc:
                # M3 review fix: include exception type name so operators
                # can distinguish sqlite3.OperationalError from RuntimeError
                # etc. without parsing the message string.
                logger.warning(
                    "remember_batch: embedding storage failed for batch of "
                    "%d items (vector voice will miss these rows) (%s): %s",
                    len(items), type(exc).__name__, exc,
                )

        # E2 -- enrichment parity with `remember()`. The merge of PR #82
        # accidentally stripped these calls during conflict resolution;
        # `_add_temporal_triple` and `_ingest_graph_and_veracity` exist
        # but were not being called per row. Without them the polyphonic
        # engine's graph + fact voices have no data to fuse and recall's
        # multi-voice RRF collapses. Each call is non-blocking
        # (try/except around per-row metadata access prevents one bad
        # row from killing the rest of the batch). Runs after the bulk
        # working_memory + embedding writes so a failure here doesn't
        # poison the per-row source / veracity bookkeeping.
        for memory_id in ids:
            item_source, item_veracity = meta_by_id.get(
                memory_id, ("conversation", "unknown")
            )
            try:
                # Look up the just-written row to find its content +
                # timestamp; cheap (PK lookup).
                row = cursor.execute(
                    "SELECT content, timestamp FROM working_memory WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    continue
                row_content = row["content"] if hasattr(row, "keys") else row[0]
                row_timestamp = row["timestamp"] if hasattr(row, "keys") else row[1]
                self._add_temporal_triple(
                    memory_id, row_timestamp, item_source, row_content
                )
                self._ingest_graph_and_veracity(
                    memory_id, row_content, item_source, item_veracity
                )
                if extract_entities:
                    _extract_and_store_entities(self, memory_id, row_content)
                if extract:
                    _extract_and_store_facts(self, memory_id, row_content, item_source)
                # MEMORY_ADDED parity with remember() -- streaming
                # observers + DeltaSync see batch rows the same way
                # they see single-row writes.
                self._emit_event(
                    "MEMORY_ADDED", memory_id,
                    content=row_content,
                    source=item_source,
                    importance=0.5,
                    metadata=None,
                )
            except Exception as exc:
                # Defensive: a single row's enrichment failure must not
                # poison the rest of the batch. Log + continue.
                logger.warning(
                    "remember_batch: per-row enrichment failed for %s (%s): %s",
                    memory_id, type(exc).__name__, exc,
                )

        self._trim_working_memory()
        return ids

    def _ingest_graph_and_veracity(self, memory_id: str, content: str,
                                    source: str, veracity: str = "unknown"):
        """Phase 3-4: Extract gists + facts, store in graph, consolidate veracity.
        Non-blocking -- failures in graph/veracity don't affect memory storage."""

        gist = None
        facts = []

        # Phase 3: Episodic graph extraction
        if self.episodic_graph is not None:
            try:
                gist = self.episodic_graph.extract_gist(content, memory_id)
                self.episodic_graph.store_gist(gist, memory_id)

                facts = self.episodic_graph.extract_facts(content, memory_id)
                for fact in facts:
                    self.episodic_graph.store_fact(fact, memory_id)

                # Link graph edges between gist and facts
                for fact in facts:
                    self.episodic_graph.add_edge(GraphEdge(
                        source=gist.id,
                        target=fact.id,
                        edge_type="ctx",
                        weight=fact.confidence,
                        timestamp=datetime.now().isoformat()
                    ))
            except Exception:
                pass  # Graph failures are non-blocking

        # Phase 4: Veracity-weighted consolidation (reuses facts from above)
        if self.veracity_consolidator is not None and facts:
            try:
                for fact in facts:
                    self.veracity_consolidator.consolidate_fact(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        veracity=veracity,
                        source=memory_id
                    )
            except Exception:
                pass  # Veracity failures are non-blocking

    def _add_temporal_triple(self, memory_id: str, timestamp: str, source: str, content: str):
        """Auto-generate temporal annotations for a memory.

        Post-E6: writes occurred_on / has_source as annotations rather
        than triples. These are inherently single-valued per memory
        today, but `annotations` is the correct home -- they describe a
        memory rather than expressing a current-truth fact like
        "user prefers X". Method name kept for backward compat.
        """
        try:
            date_str = timestamp[:10]  # YYYY-MM-DD
            # Reuse the cached AnnotationStore handle on self.
            self.annotations.add(
                memory_id=memory_id,
                kind="occurred_on",
                value=date_str,
            )
            # Also tag source type
            if source and source not in ("conversation", "user", "assistant"):
                self.annotations.add(
                    memory_id=memory_id,
                    kind="has_source",
                    value=source,
                )
        except Exception:
            # Annotation writes are optional; don't fail memory write if they fail
            pass

    def _trim_working_memory(self):
        """Keep working_memory within size/time limits.

        Post-E3: consolidated rows (consolidated_at IS NOT NULL) are
        exempt from trim. The "originals stay" contract means they
        remain queryable until explicit forget(); the TTL window only
        bounds NOT-YET-consolidated content. Without this exemption,
        the additive promise expires at WORKING_MEMORY_TTL_HOURS and
        the experiment Arm B's "ADD-only" guarantee collapses at 24h.
        """
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS)).isoformat()
        self.conn.execute("""
            DELETE FROM working_memory
            WHERE session_id = ?
              AND consolidated_at IS NULL
              AND (
                timestamp < ? OR
                id NOT IN (
                    SELECT id FROM working_memory
                    WHERE session_id = ? AND consolidated_at IS NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
              )
        """, (self.session_id, cutoff, self.session_id, WORKING_MEMORY_MAX_ITEMS))
        self.conn.commit()

    def get_context(self, limit: int = 10) -> List[Dict]:
        """Get working_memory for prompt injection.
        Global memories first, then sorted by importance (high first),
        then by recency. High-importance rules/bans surface reliably."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            SELECT id, content, source, timestamp, importance, scope
            FROM working_memory
            WHERE (session_id = ? OR scope = 'global')
              AND (valid_until IS NULL OR valid_until > ?)
              AND superseded_by IS NULL
            ORDER BY
                CASE WHEN scope = 'global' THEN 0 ELSE 1 END,
                importance DESC,
                timestamp DESC
            LIMIT ?
        """, (self.session_id, now, limit))
        return [dict(row) for row in cursor.fetchall()]

    def invalidate(self, memory_id: str, replacement_id: str = None) -> bool:
        """
        Mark a memory as invalid/superseded.
        If replacement_id is provided, sets superseded_by.
        Otherwise sets valid_until to now (immediate expiry).
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        # Try working_memory first
        cursor.execute("""
            UPDATE working_memory
            SET valid_until = ?, superseded_by = ?
            WHERE id = ? AND (session_id = ? OR scope = 'global')
        """, (now, replacement_id, memory_id, self.session_id))
        if cursor.rowcount > 0:
            self.conn.commit()
            return True
        # Try episodic_memory
        cursor.execute("""
            UPDATE episodic_memory
            SET valid_until = ?, superseded_by = ?
            WHERE id = ? AND (session_id = ? OR scope = 'global')
        """, (now, replacement_id, memory_id, self.session_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_working_stats(self, author_id: str = None, author_type: str = None,
                          channel_id: str = None) -> Dict:
        cursor = self.conn.cursor()
        where_clauses = []
        params = []
        if author_id:
            where_clauses.append("author_id = ?")
            params.append(author_id)
        if author_type:
            where_clauses.append("author_type = ?")
            params.append(author_type)
        if channel_id:
            where_clauses.append("channel_id = ?")
            params.append(channel_id)
        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        cursor.execute(f"SELECT COUNT(*) FROM working_memory{where_str}", params)
        total = cursor.fetchone()[0]
        cursor.execute(f"SELECT timestamp FROM working_memory{where_str} ORDER BY timestamp DESC LIMIT 1", params)
        last = cursor.fetchone()
        return {"total": total, "last": last[0] if last else None}

    # DEPRECATED -- kept for backward compatibility with hermes_memory_provider/cli.py
    def get_global_working_stats(self) -> Dict:
        """DEPRECATED: Use get_working_stats() instead. Kept for backward compatibility."""
        return self.get_working_stats()

    def update_working(self, memory_id: str, content: str = None,
                       importance: float = None) -> bool:
        """Update a working_memory entry.

        After updating content, reindexes FTS5 (via wm_au trigger) and
        recomputes the vector embedding in memory_embeddings so recall()
        returns the corrected content instead of stale derived state.
        """
        cursor = self.conn.cursor()
        updates = []
        params = []
        content_changed = False
        if content is not None:
            updates.append("content = ?")
            params.append(content)
            content_changed = True
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        if not updates:
            return False
        params.extend([memory_id, self.session_id])
        cursor.execute(
            f"UPDATE working_memory SET {', '.join(updates)} WHERE id = ? AND session_id = ?",
            params
        )
        affected = cursor.rowcount

        # Refresh derived state when content changed.
        # FTS5 is handled by the wm_au trigger (AFTER UPDATE OF content),
        # but memory_embeddings must be recomputed explicitly.
        if content_changed and affected > 0 and _embeddings.available():
            try:
                vec = _embeddings.embed([content])
                if vec is not None and len(vec) > 0:
                    model = _embeddings._DEFAULT_MODEL
                    emb_json = _embeddings.serialize(vec[0])
                    cursor.execute(
                        "INSERT OR REPLACE INTO memory_embeddings"
                        " (memory_id, embedding_json, model)"
                        " VALUES (?, ?, ?)",
                        (memory_id, emb_json, model),
                    )
            except Exception as exc:
                logger.warning(
                    "update_working: embedding refresh failed for %s"
                    " (%s): %s",
                    memory_id, type(exc).__name__, exc,
                )

        self.conn.commit()
        return affected > 0

    def forget_working(self, memory_id: str) -> bool:
        # E6.a: the cascade-delete of annotations must be authorized by the
        # session-scoped working_memory DELETE. The annotations table has no
        # session_id column, so an unconditional `DELETE FROM annotations
        # WHERE memory_id = ?` lets a hostile caller in session B pass a
        # memory_id from session A and silently wipe session A's annotations
        # -- adversarial /review found this. The session-scoped working_memory
        # DELETE is the trust boundary: if it matches a row, the caller is
        # authorized to delete the row's annotations. If it matches zero
        # rows (wrong session, or already-forgotten), we skip the cascade.
        #
        # Wrapped in an explicit transaction with rollback so a mid-cascade
        # failure (corrupted table, lock contention, future FK trigger)
        # rolls back the working_memory DELETE rather than leaving it
        # uncommitted on the connection for a later unrelated commit to
        # silently include.
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM working_memory WHERE id = ? AND session_id = ?",
                (memory_id, self.session_id),
            )
            wm_rows = cursor.rowcount
            if wm_rows > 0:
                cursor.execute(
                    "DELETE FROM annotations WHERE memory_id = ?", (memory_id,)
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return wm_rows > 0

    # ------------------------------------------------------------------
    # Episodic Memory
    # ------------------------------------------------------------------
    def consolidate_to_episodic(self, summary: str, source_wm_ids: List[str],
                                source: str = "consolidation", importance: float = 0.6,
                                metadata: Dict = None, valid_until: str = None,
                                scope: str = "session",
                                veracity: Optional[str] = None) -> str:
        """
        Store a consolidated summary into episodic_memory with optional embedding.

        E4.a.1: `veracity` kwarg threads the aggregated source-row veracity
        into the episodic INSERT. Pre-fix the INSERT didn't include the
        veracity column at all, so post-sleep rows took the schema default
        'unknown' -- destroying the per-row veracity signal `remember_batch`
        had populated. Callers (typically `sleep()`) should compute the
        aggregate via `aggregate_veracity()` over the source rows' veracity
        values and pass it here. `None` falls back to 'unknown' (matches
        legacy behavior + schema default).
        """
        memory_id = _generate_id(summary)
        timestamp = datetime.now().isoformat()
        # Typed memory classification
        ep_type = None
        if classify_memory is not None:
            try:
                result = classify_memory(summary)
                ep_type = result.memory_type.value
            except Exception:
                pass
        # Clamp to canonical allowlist at the trust boundary. Defaults to
        # 'unknown' if not provided (back-compat with pre-E4.a.1 callers).
        if veracity is None:
            row_veracity = "unknown"
        else:
            row_veracity = clamp_veracity(
                veracity, context="consolidate_to_episodic.veracity"
            )
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO episodic_memory
            (id, content, source, timestamp, session_id, importance, metadata_json, summary_of, valid_until, scope,
             author_id, author_type, channel_id, memory_type, veracity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, summary, source, timestamp, self.session_id, importance,
              json.dumps(metadata or {}), ",".join(source_wm_ids), valid_until, scope,
              self.author_id, self.author_type, self.channel_id, ep_type, row_veracity))
        rowid = cursor.lastrowid

        if _embeddings.available():
            vec = _embeddings.embed([summary])
            if vec is not None:
                if _vec_available(self.conn):
                    _vec_insert(self.conn, rowid, vec[0].tolist())
                else:
                    # Fallback: store in memory_embeddings table for in-memory search
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                        VALUES (?, ?, ?)
                    """, (memory_id, _embeddings.serialize(vec[0]), _embeddings._DEFAULT_MODEL))

                # Binary vector compression (Phase 2 -- 32x reduction)
                if _mib is not None:
                    try:
                        bv = _mib(vec[0])
                        cursor.execute(
                            "UPDATE episodic_memory SET binary_vector = ? WHERE rowid = ?",
                            (bv, rowid)
                        )
                    except Exception:
                        pass  # Non-blocking

        self.conn.commit()

        # Phase 3-4: Graph + veracity for consolidated episodic memory
        # E4.a.1 review fix (H2): thread the aggregated row_veracity into
        # graph + fact extraction so Bayesian compounding on consolidated
        # facts uses the source-aggregated signal, not a hardcoded
        # 'inferred'. Pre-fix this line passed 'inferred' regardless, which
        # the consolidator's `consolidate_fact` then used as the veracity
        # weight in its confidence update -- undermining the very signal
        # we just preserved in the episodic INSERT.
        self._ingest_graph_and_veracity(memory_id, summary, source, veracity=row_veracity)

        self._emit_event("MEMORY_CONSOLIDATED", memory_id, content=summary,
                         source=source, importance=importance,
                         metadata={"summary_of": source_wm_ids, **(metadata or {})})
        return memory_id

    def recall(self, query: str, top_k: int = 40, *,
               from_date: Optional[str] = None, to_date: Optional[str] = None,
               source: Optional[str] = None, topic: Optional[str] = None,
               author_id: Optional[str] = None,
               author_type: Optional[str] = None,
               channel_id: Optional[str] = None,
               veracity: Optional[str] = None,
               memory_type: Optional[str] = None,
               temporal_weight: float = 0.0,
               query_time: Optional[Any] = None,
               temporal_halflife: Optional[float] = None,
               vec_weight: float = None,
               fts_weight: float = None,
               importance_weight: float = None) -> List[Dict]:
        """
        Hybrid recall across working_memory + episodic_memory.
        Uses sqlite-vec + FTS5 for episodic, FTS5 for working.
        Falls back to recency-only for working memory if FTS5 unavailable.

        Temporal filtering:
            from_date/to_date: ISO date strings (YYYY-MM-DD) to filter by timestamp.
            source: Filter by memory source (e.g., 'cron', 'user', 'conversation').
            topic: Filter by topic tag (stored in source field for now, pending dedicated column).

        Multi-agent identity filtering (v2.1):
            author_id: Filter by author (e.g., 'abdias', 'codex-agent').
            author_type: Filter by author type ('human', 'agent', 'system').
            channel_id: Filter by channel/group (e.g., 'fluxspeak-team').

        Temporal scoring (Phase 3):
            temporal_weight: Float 0.0-1.0. Soft boost for memories near query_time.
                0.0 = no temporal boost (default, backward compatible).
            query_time: Target time for temporal scoring. None = now().
            temporal_halflife: Hours for temporal decay. None = env var or 24h default.

        Temporal scoring (Phase 3):
            temporal_weight: Float 0.0-1.0. Soft boost for memories near query_time.
                0.0 = no temporal boost (default, backward compatible).
            query_time: Target time for temporal scoring. None = now().
            temporal_halflife: Hours for temporal decay. None = env var or 24h default.

        Configurable hybrid scoring (Phase 4):
            vec_weight: Weight for vector (dense) similarity in episodic scoring.
                None = use env var MNEMOSYNE_VEC_WEIGHT or default 0.5.
            fts_weight: Weight for FTS5 text relevance in episodic scoring.
                None = use env var MNEMOSYNE_FTS_WEIGHT or default 0.3.
            importance_weight: Weight for importance score in all scoring.
                None = use env var MNEMOSYNE_IMPORTANCE_WEIGHT or default 0.2.

            The three episodic weights are automatically normalized to sum to 1.0.
            Working memory uses a derived split: keyword gets (1 - importance_weight) * 0.6,
            recency gets (1 - importance_weight) * 0.4.

        Polyphonic recall (E5, gated by MNEMOSYNE_POLYPHONIC_RECALL=1):
            When the env flag is set to "1", recall delegates to
            PolyphonicRecallEngine (mnemosyne/core/polyphonic_recall.py).
            The engine runs 4 voices in parallel -- vector / graph /
            fact / temporal -- fuses them via RRF (k=60), diversity-
            reranks the combined results, and assembles within a
            context budget. Each result dict carries `voice_scores`
            for per-signal provenance.

            Flag unset or "0" (default): the existing linear scorer
            below runs unchanged. Zero behavior change for production.
        """
        # E5 feature flag -- read per call so operators can toggle
        # without rebuilding BeamMemory (critical for A/B experiments
        # in the same process). All recall filter kwargs flow through
        # so the engine path enforces the same isolation/validity
        # contract as the linear path. /review found that omitting
        # them under flag=ON was a data-isolation regression (P1).
        if os.environ.get("MNEMOSYNE_POLYPHONIC_RECALL", "0") == "1":
            return self._recall_polyphonic(
                query, top_k,
                from_date=from_date, to_date=to_date,
                source=source, topic=topic,
                author_id=author_id, author_type=author_type,
                channel_id=channel_id,
                veracity=veracity, memory_type=memory_type,
            )

        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        # ---- Configurable hybrid scoring setup (Phase 4) ----
        vw, fw, iw = _normalize_weights(vec_weight, fts_weight, importance_weight)

        # ---- Temporal scoring setup ----
        parsed_query_time = _parse_query_time(query_time)
        if temporal_halflife is not None:
            th_halflife = temporal_halflife
        else:
            th_halflife = float(os.environ.get("MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS", "24"))

        # [C4] Recall path diagnostics -- lazy import to avoid module-
        # load coupling. Counters are recorded AFTER the per-row
        # scoring loops below so they reflect POST-FILTER kept rows
        # (not pre-filter candidate sets). /review caught the
        # pre-filter shape as misleading.
        from mnemosyne.core.recall_diagnostics import get_diagnostics as _get_recall_diag
        _recall_diag = _get_recall_diag()
        # Per-call kept-row accumulators. Incremented as the scoring
        # loops append to `results`. Final values recorded to the
        # diagnostics in the try/finally at the end of recall().
        _wm_fts_kept = 0
        _wm_vec_kept = 0
        _wm_fallback_kept = 0
        _em_fts_kept = 0
        _em_vec_kept = 0
        _em_fallback_kept = 0
        _wm_fallback_used = False
        _em_fallback_used = False
        _wm_had_candidates = False
        _em_had_candidates = False

        # ---- Working memory (FTS5 fast path) ----
        try:
            wm_fts = _fts_search_working(self.conn, query, k=max(top_k * 3, 50))
        except Exception:
            wm_fts = []

        wm_ids = {r["id"] for r in wm_fts}
        wm_ranks = {r["id"]: r["rank"] for r in wm_fts}

        # ---- Working memory (vector search) ----
        wm_vec_sims = {}
        if _embeddings.available():
            try:
                emb_result = _embeddings.embed_query(query)
                if emb_result is not None:
                    wm_vec = _wm_vec_search(self.conn, emb_result,
                                              k=max(top_k, 20) if _BEAM_MODE else max(top_k * 3, 50))
                    for vr in wm_vec:
                        wm_vec_sims[vr["id"]] = vr["sim"]
                        wm_ids.add(vr["id"])  # Merge vector results with FTS5 results
            except Exception:
                pass
        # Track whether the FTS+vec layer produced any candidates
        # at all (signal source for the truly_empty gate later).
        if wm_ids:
            _wm_had_candidates = True
        # If both FTS and vec produced nothing, the WM fallback at
        # the else-branch below fires. Recording the fallback signal
        # uses a boolean (per-call), not a per-row count -- that
        # avoids double-counting against the kept-row accumulators.
        _wm_fallback_used = not wm_ids
        if _wm_fallback_used:
            _recall_diag.record_fallback_used(wm=True)

        # Build temporal filter clause for working memory
        wm_where_clauses = [
            "(valid_until IS NULL OR valid_until > ?)",
            "superseded_by IS NULL"
        ]
        wm_params = [datetime.now().isoformat()]
        
        # Session scope: channel filter only when explicitly specified.
        # Author-only searches have no session/channel restriction.
        if channel_id:
            wm_where_clauses.append("(session_id = ? OR scope = 'global' OR channel_id = ?)")
            wm_params.extend([self.session_id, channel_id])
        elif author_id or author_type:
            wm_where_clauses.append("(1=1)")
        else:
            wm_where_clauses.append("(session_id = ? OR scope = 'global')")
            wm_params.append(self.session_id)
        
        if from_date:
            wm_where_clauses.append("timestamp >= ?")
            wm_params.append(f"{from_date}T00:00:00")
        if to_date:
            wm_where_clauses.append("timestamp <= ?")
            wm_params.append(f"{to_date}T23:59:59")
        if source:
            wm_where_clauses.append("source = ?")
            wm_params.append(source)
        if topic:
            # Topic stored in source field for now (pending dedicated topic column)
            wm_where_clauses.append("source = ?")
            wm_params.append(topic)
        if veracity:
            wm_where_clauses.append("veracity = ?")
            wm_params.append(veracity)
        if memory_type:
            wm_where_clauses.append("memory_type = ?")
            wm_params.append(memory_type)
        if author_id:
            wm_where_clauses.append("author_id = ?")
            wm_params.append(author_id)
        if author_type:
            wm_where_clauses.append("author_type = ?")
            wm_params.append(author_type)
        if channel_id:
            wm_where_clauses.append("channel_id = ?")
            wm_params.append(channel_id)
        
        wm_where = " AND ".join(wm_where_clauses)

        if wm_ids:
            placeholders = ",".join("?" * len(wm_ids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity, memory_type
                FROM working_memory
                WHERE id IN ({placeholders})
                  AND {wm_where}
            """, (*tuple(wm_ids), *wm_params))
            rows = cursor.fetchall()
        else:
            # Fallback: fetch recent items and score in Python (old path)
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity, memory_type
                FROM working_memory
                WHERE {wm_where}
                ORDER BY timestamp DESC
                LIMIT {min(EPISODIC_RECALL_LIMIT, 2000)}
            """, wm_params)
            rows = cursor.fetchall()
            # [C4] _wm_fallback_kept incremented per-row inside the
            # scoring loop above. record_fallback_used was already
            # called when wm_ids was empty.

        # Precompute min_rank/rng for wm_ranks normalization
        if wm_ranks:
            min_rank = min(wm_ranks.values())
            max_rank = max(wm_ranks.values())
            rng = max_rank - min_rank if max_rank != min_rank else 1.0
        else:
            min_rank = 0.0
            rng = 1.0

        for row in rows:
            content_lower = row["content"].lower()
            content_words_list = content_lower.split()
            content_words_set = set(content_words_list)
            if wm_ranks and row["id"] in wm_ranks:
                normalized = 1.0 - ((wm_ranks[row["id"]] - min_rank) / rng)
                relevance = normalized
            else:
                # exact: query words appearing in content (substring match, not token equality)
                exact = sum(1 for w in query_words if w in content_lower)
                # partial: unique query words with substring match in content words (set-based, not cartesian)
                partial = sum(1 for w in query_words if len(w) >= 2 and any(w in cw or cw in w for cw in content_words_set if len(cw) >= 2))
                # cross: query substrings matched against content word substrings (set-based)
                query_substr = {w for w in query_words if len(w) >= 2}
                content_substr = {cw for cw in content_words_set if len(cw) >= 2}
                cross = sum(1 for q in query_substr for c in content_substr if q in c or c in q)
                # Also check if the full query is a substring of content (handles spaceless languages)
                full_match = 1.0 if query_lower in content_lower else 0.0
                if not full_match and content_lower in query_lower:
                    full_match = 0.5
                # Character-level overlap for spaceless languages (e.g. Chinese)
                query_chars = set(query_lower)
                content_chars = set(content_lower)
                char_overlap = len(query_chars & content_chars) / max(len(query_chars), 1) if query_chars else 0.0
                relevance = (exact * 1.0 + partial * 0.3 + cross * 0.5 + full_match + char_overlap * 0.8) / max(len(query_words), 1)
            if relevance > 0.02 or wm_ranks:
                decay = _recency_decay(row["timestamp"])
                # Phase 4: configurable scoring for working memory
                # keyword_share = (1 - importance_weight) * 0.6, recency_share = (1 - importance_weight) * 0.4
                kw_share = (1.0 - iw) * 0.6
                rc_share = (1.0 - iw) * 0.4
                base_score = relevance * kw_share + row["importance"] * iw
                # Blend vector similarity into working memory score (weighted toward keyword precision)
                vec_sim = wm_vec_sims.get(row["id"], 0.0)
                if vec_sim > 0:
                    base_score = base_score * 0.80 + vec_sim * 0.20
                score = base_score * (rc_share + (1.0 - rc_share) * decay)
                # Temporal boost (Phase 3)
                if temporal_weight > 0.0:
                    t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                    score *= (1.0 + temporal_weight * t_boost)
                # [C4] Per-row tier attribution. Credit FTS for any
                # row in wm_ranks (overlap with vec credited to FTS
                # so the union sum stays consistent). Vec-only rows
                # (in wm_vec_sims but not wm_ranks) credit wm_vec.
                # Rows reached via the fallback branch (wm_ids empty)
                # credit wm_fallback. Each row credits exactly one
                # tier so `wm_fts + wm_vec + wm_fallback` = total
                # kept WM rows for this call.
                if _wm_fallback_used:
                    _wm_fallback_kept += 1
                elif wm_ranks and row["id"] in wm_ranks:
                    _wm_fts_kept += 1
                elif row["id"] in wm_vec_sims:
                    _wm_vec_kept += 1
                results.append({
                    "id": row["id"],
                    "content": row["content"][:500],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "tier": "working",
                    "score": round(score, 4),
                    "keyword_score": round(relevance, 4),
                    "dense_score": round(vec_sim, 4),
                    "fts_score": round(relevance, 4) if wm_ranks else 0.0,
                    "importance": row["importance"],
                    "recall_count": row["recall_count"] or 0,
                    "last_recalled": row["last_recalled"],
                    "recency_decay": round(decay, 4),
                    "scope": row["scope"] if "scope" in row.keys() else "session",
                    "author_id": row["author_id"] if "author_id" in row.keys() else None,
                    "author_type": row["author_type"] if "author_type" in row.keys() else None,
                    "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                    "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                    "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                    "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
                })

        # ---- Entity-aware recall ----
        entity_memory_ids = _find_memories_by_entity(self, query)
        if entity_memory_ids:
            # Fetch entity-matched memories and boost their scores
            placeholders = ",".join("?" * len(entity_memory_ids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity, memory_type
                FROM working_memory
                WHERE id IN ({placeholders})
                  AND {wm_where}
            """, (*tuple(entity_memory_ids), *wm_params))
            entity_rows = cursor.fetchall()
            
            # Add entity-matched memories with boosted scores
            existing_ids = {r["id"] for r in results}
            for row in entity_rows:
                if row["id"] in existing_ids:
                    # Boost existing result
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.3, 1.0), 4)
                            r["entity_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.6 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "working",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        "dense_score": round(wm_vec_sims.get(row["id"], 0.0), 4),
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "entity_match": True
                    })
            
            # Also check episodic memory for entity matches
            em_placeholders = ",".join("?" * len(entity_memory_ids))
            if channel_id:
                em_entity_scope = "(session_id = ? OR scope = 'global' OR channel_id = ?)"
                em_entity_params = [*tuple(entity_memory_ids), self.session_id, channel_id]
            elif author_id or author_type:
                em_entity_scope = "(1=1)"
                em_entity_params = [*tuple(entity_memory_ids)]
            else:
                em_entity_scope = "(session_id = ? OR scope = 'global')"
                em_entity_params = [*tuple(entity_memory_ids), self.session_id]
            em_entity_params.extend([datetime.now().isoformat()])
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity, memory_type
                FROM episodic_memory
                WHERE id IN ({em_placeholders})
                  AND {em_entity_scope}
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
            """, (*em_entity_params,))
            em_entity_rows = cursor.fetchall()
            
            em_existing_ids = {r["id"] for r in results}
            for row in em_entity_rows:
                if row["id"] in em_existing_ids:
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.3, 1.0), 4)
                            r["entity_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.6 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "episodic",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        # C30: episodic rows never key into wm_vec_sims
                        # (that dict holds working_memory ids only). Set
                        # 0.0 explicitly rather than lookup-that-always-
                        # returns-default, so post-run analysis isn't
                        # misled into thinking dense similarity was
                        # computed. The entity/fact-matched episodic
                        # paths don't compute ep dense sim themselves.
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "entity_match": True
                    })

        # ---- Fact-aware recall ----
        fact_memory_ids = _find_memories_by_fact(self, query)
        if fact_memory_ids:
            placeholders = ",".join("?" * len(fact_memory_ids))
            cursor = self.conn.cursor()
            # Check working_memory for fact matches
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity, memory_type
                FROM working_memory
                WHERE id IN ({placeholders})
                  AND {wm_where}
            """, (*tuple(fact_memory_ids), *wm_params))
            fact_rows = cursor.fetchall()
            
            existing_ids = {r["id"] for r in results}
            for row in fact_rows:
                if row["id"] in existing_ids:
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.2, 1.0), 4)
                            r["fact_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.5 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "working",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        "dense_score": round(wm_vec_sims.get(row["id"], 0.0), 4),
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "fact_match": True
                    })
            
            # Also check episodic memory for fact matches
            if channel_id:
                fact_em_scope = "(session_id = ? OR scope = 'global' OR channel_id = ?)"
                fact_em_params = [*tuple(fact_memory_ids), self.session_id, channel_id]
            elif author_id or author_type:
                fact_em_scope = "(1=1)"
                fact_em_params = [*tuple(fact_memory_ids)]
            else:
                fact_em_scope = "(session_id = ? OR scope = 'global')"
                fact_em_params = [*tuple(fact_memory_ids), self.session_id]
            fact_em_params.extend([datetime.now().isoformat()])
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity, memory_type
                FROM episodic_memory
                WHERE id IN ({placeholders})
                  AND {fact_em_scope}
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
            """, (*fact_em_params,))
            em_fact_rows = cursor.fetchall()
            
            em_existing_ids = {r["id"] for r in results}
            for row in em_fact_rows:
                if row["id"] in em_existing_ids:
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.2, 1.0), 4)
                            r["fact_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.5 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "episodic",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        # C30: episodic rows never key into wm_vec_sims
                        # (that dict holds working_memory ids only). Set
                        # 0.0 explicitly rather than lookup-that-always-
                        # returns-default, so post-run analysis isn't
                        # misled into thinking dense similarity was
                        # computed. The entity/fact-matched episodic
                        # paths don't compute ep dense sim themselves.
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "fact_match": True
                    })

        # ---- Pre-compute query binary vector (Phase 5 binary voice) ----
        query_bv = None
        query_emb_for_bv = None
        if _embeddings.available() and _mib is not None:
            emb_result = _embeddings.embed_query(query)
            if emb_result is not None:
                query_emb_for_bv = emb_result
                query_bv = _mib(emb_result)

        # ---- Episodic memory (vec + FTS5 hybrid) ----
        vec_results = {}
        max_distance = 0.0
        if _embeddings.available():
            emb_result = _embeddings.embed_query(query)
            if emb_result is not None:
                if _vec_available(self.conn):
                    vec_rows = _vec_search(self.conn, emb_result.tolist(), k=max(top_k * 3, 20))
                else:
                    # Fallback: in-memory cosine similarity search
                    vec_rows = _in_memory_vec_search(self.conn, emb_result, k=max(top_k * 3, 20))
                if vec_rows:
                    max_distance = max(vr["distance"] for vr in vec_rows)
                    for vr in vec_rows:
                        sim = max(0.0, 1.0 - (vr["distance"] / max_distance)) if max_distance > 0 else 1.0
                        vec_results[vr["rowid"]] = sim

        fts_results = {}
        fts_rows = _fts_search(self.conn, query, k=max(top_k * 3, 20))
        if fts_rows:
            min_rank = min(r["rank"] for r in fts_rows)
            max_rank = max(r["rank"] for r in fts_rows)
            rng = max_rank - min_rank if max_rank != min_rank else 1.0
            for fr in fts_rows:
                normalized = 1.0 - ((fr["rank"] - min_rank) / rng)
                fts_results[fr["rowid"]] = normalized

        episodic_rowids = set(vec_results.keys()) | set(fts_results.keys())
        if episodic_rowids:
            _em_had_candidates = True
        # [C4] em_fts/em_vec kept counts are accumulated per-row
        # inside the scoring loop below so the counters reflect
        # post-filter results, not pre-filter candidate sets.
        # /review caught the pre-filter recording as misleading --
        # rows that pass FTS but get dropped by wm_where/em_where
        # (session/channel/date) inflated the counter.
        
        # Build temporal filter for episodic memory
        em_where_clauses = [
            "(valid_until IS NULL OR valid_until > ?)",
            "superseded_by IS NULL"
        ]
        em_params = [datetime.now().isoformat()]
        
        # Session scope: channel filter only when explicitly specified.
        # Author-only searches have no session/channel restriction.
        if channel_id:
            em_where_clauses.append("(session_id = ? OR scope = 'global' OR channel_id = ?)")
            em_params.extend([self.session_id, channel_id])
        elif author_id or author_type:
            em_where_clauses.append("(1=1)")
        else:
            em_where_clauses.append("(session_id = ? OR scope = 'global')")
            em_params.append(self.session_id)
        
        if from_date:
            em_where_clauses.append("timestamp >= ?")
            em_params.append(f"{from_date}T00:00:00")
        if to_date:
            em_where_clauses.append("timestamp <= ?")
            em_params.append(f"{to_date}T23:59:59")
        if source:
            em_where_clauses.append("source = ?")
            em_params.append(source)
        if topic:
            em_where_clauses.append("source = ?")
            em_params.append(topic)
        if veracity:
            em_where_clauses.append("veracity = ?")
            em_params.append(veracity)
        if memory_type:
            em_where_clauses.append("memory_type = ?")
            em_params.append(memory_type)
        if author_id:
            em_where_clauses.append("author_id = ?")
            em_params.append(author_id)
        if author_type:
            em_where_clauses.append("author_type = ?")
            em_params.append(author_type)
        if channel_id:
            em_where_clauses.append("channel_id = ?")
            em_params.append(channel_id)
        
        em_where = " AND ".join(em_where_clauses)
        
        if episodic_rowids:
            placeholders = ",".join("?" * len(episodic_rowids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT rowid, id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, memory_type, binary_vector
                FROM episodic_memory
                WHERE rowid IN ({placeholders})
                  AND {em_where}
            """, (*tuple(episodic_rowids), *em_params))
        for row in cursor.fetchall():
            rid = row["rowid"]
            sim = vec_results.get(rid, 0.0)
            fts = fts_results.get(rid, 0.0)
            decay = _recency_decay(row["timestamp"])
            # Phase 4: configurable hybrid scoring for episodic memory
            # vec_weight + fts_weight + importance_weight are normalized to sum to 1.0
            base_score = sim * vw + fts * fw + row["importance"] * iw

            # Phase 5: Graph + fact voices (polyphonic recall bonus).
            # Each block gated by an A/B toggle: `MNEMOSYNE_GRAPH_BONUS=0`,
            # `MNEMOSYNE_FACT_BONUS=0`, `MNEMOSYNE_BINARY_BONUS=0` to
            # disable individually for ablation. Default ON -- production
            # behavior unchanged.
            graph_bonus = 0.0
            fact_bonus = 0.0
            binary_bonus = 0.0
            memory_id = row["id"]
            content_lower = row["content"].lower()
            bv = row["binary_vector"]
            if self.episodic_graph is not None and not _env_disabled("MNEMOSYNE_GRAPH_BONUS"):
                try:
                    # Count graph edges for this memory (well-connected = more relevant)
                    cursor2 = self.conn.cursor()
                    cursor2.execute(
                        "SELECT COUNT(*) FROM graph_edges WHERE source LIKE ? OR target LIKE ?",
                        (f"%{memory_id}%", f"%{memory_id}%"))
                    edge_count = cursor2.fetchone()[0]
                    graph_bonus = min(edge_count * 0.02, 0.08)
                except Exception:
                    pass
            if self.episodic_graph is not None and not _env_disabled("MNEMOSYNE_FACT_BONUS"):
                try:
                    # Check if facts from graph match query terms via set-overlap
                    cursor2 = self.conn.cursor()
                    cursor2.execute(
                        "SELECT subject, predicate, object FROM facts WHERE source_msg_id = ?",
                        (memory_id,))
                    query_word_set = {w for w in query.lower().split() if len(w) > 2}
                    match_count = 0
                    for frow in cursor2.fetchall():
                        fact_tokens = {t.lower() for t in (f"{frow['subject']} {frow['predicate']} {frow['object']}").split() if len(t) > 2}
                        if query_word_set & fact_tokens:
                            match_count += 1
                    fact_bonus = min(match_count * 0.04, 0.1)
                except Exception:
                    pass
            # Binary vector voice (Phase 5): re-enabled -- binary vectors are now
            # backfilled for all episodic entries. ITS discriminability improves at
            # scale (1033 entries); clustering concern was for small synthetic sets.
            if query_bv is not None and bv is not None and not _env_disabled("MNEMOSYNE_BINARY_BONUS"):
                try:
                    # Compute hamming distance via XOR + popcount
                    q_arr = np.frombuffer(query_bv, dtype=np.uint8)
                    m_arr = np.frombuffer(bv, dtype=np.uint8)
                    xor_arr = np.bitwise_xor(q_arr, m_arr)
                    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)
                    h_dist = int(np.sum(popcount_table[xor_arr]))
                    # Sigmoid: max bonus at distance=0, bonus ~0 at distance=EMBEDDING_DIM
                    # Use tanh for smooth falloff; bonus range [0, 0.08]
                    normalized_dist = h_dist / EMBEDDING_DIM  # 0.0 (identical) to 1.0 (opposite)
                    binary_bonus = 0.08 * (1.0 - np.tanh(normalized_dist * 3.0))
                except Exception:
                    binary_bonus = 0.0
            else:
                binary_bonus = 0.0

            score = base_score * (0.7 + 0.3 * decay)
            score += graph_bonus + fact_bonus + binary_bonus  # Phase 5: polyphonic bonuses
            # Temporal boost (Phase 3)
            if temporal_weight > 0.0:
                t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                score *= (1.0 + temporal_weight * t_boost)
            # [C4] Per-row tier attribution. FTS gets the overlap;
            # vec gets vec-only rows. One increment per kept row.
            rid = row["rowid"]
            if rid in fts_results:
                _em_fts_kept += 1
            elif rid in vec_results:
                _em_vec_kept += 1
            results.append({
                "id": row["id"],
                "content": row["content"][:500],
                "source": row["source"],
                "timestamp": row["timestamp"],
                "tier": "episodic",
                "score": round(score, 4),
                "keyword_score": 0.0,
                "dense_score": round(sim, 4),
                "fts_score": round(fts, 4),
                "importance": row["importance"],
                "recall_count": row["recall_count"] or 0,
                "last_recalled": row["last_recalled"],
                "recency_decay": round(decay, 4),
                "scope": row["scope"] if "scope" in row.keys() else "session",
                "author_id": row["author_id"] if "author_id" in row.keys() else None,
                "author_type": row["author_type"] if "author_type" in row.keys() else None,
                "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
            })

        # Fallback: if no episodic matches from vec/FTS, scan recent episodic entries
        if not episodic_rowids:
            _em_fallback_used = True
            # [C4] Record EM fallback firing so operators see how
            # often recall comes from the weak-signal substring path
            # rather than vec/FTS. High em_fallback_rate during a
            # benchmark means recall scores aren't measuring what
            # the experiment thinks they're measuring.
            _recall_diag.record_fallback_used(em=True)
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT rowid, id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, memory_type, binary_vector
                FROM episodic_memory
                WHERE {em_where}
                ORDER BY timestamp DESC
                LIMIT {min(EPISODIC_RECALL_LIMIT, 500)}
            """, em_params)
            _em_fallback_rows = cursor.fetchall()
            # [C4] em_fallback kept count is incremented per-row
            # inside the loop below (only rows passing the
            # relevance>0.02 threshold) so the counter reflects
            # results-attributable contributions, not scanned rows.
            for row in _em_fallback_rows:
                content_lower = row["content"].lower()
                content_words_set = set(content_lower.split())
                # exact: query words appearing as complete tokens in content
                exact = sum(1 for w in query_words if w in content_words_set)
                # partial: unique query words with substring match in content words (set-based, not cartesian)
                partial = sum(1 for w in query_words if len(w) >= 2 and any(w in cw or cw in w for cw in content_words_set if len(cw) >= 2))
                # cross: query substrings matched against content word substrings (set-based)
                query_substr = {w for w in query_words if len(w) >= 2}
                content_substr = {cw for cw in content_words_set if len(cw) >= 2}
                cross = sum(1 for q in query_substr for c in content_substr if q in c or c in q)
                full_match = 1.0 if query_lower in content_lower else 0.0
                if not full_match and content_lower in query_lower:
                    full_match = 0.5
                # Character-level overlap for spaceless languages (e.g. Chinese)
                query_chars = set(query_lower)
                content_chars = set(content_lower)
                char_overlap = len(query_chars & content_chars) / max(len(query_chars), 1) if query_chars else 0.0
                relevance = (exact * 1.0 + partial * 0.3 + cross * 0.5 + full_match + char_overlap * 0.8) / max(len(query_words), 1)
                if relevance > 0.02:
                    decay = _recency_decay(row["timestamp"])
                    # Phase 4: configurable scoring for episodic fallback
                    kw_share = (1.0 - iw) * 0.6
                    rc_share = (1.0 - iw) * 0.4
                    base_score = relevance * kw_share + row["importance"] * iw
                    score = base_score * (rc_share + (1.0 - rc_share) * decay)

                    # Phase 5: Graph + fact + binary bonuses for fallback.
                    # Gated by the same toggles as the main loop above
                    # so ablation behavior is consistent across both
                    # episodic paths.
                    graph_b = 0.0
                    fact_b = 0.0
                    binary_b = 0.0
                    if not _env_disabled("MNEMOSYNE_GRAPH_BONUS"):
                        try:
                            cursor2 = self.conn.cursor()
                            cursor2.execute(
                                "SELECT COUNT(*) FROM graph_edges WHERE source LIKE ? OR target LIKE ?",
                                (f"%{row['id']}%", f"%{row['id']}%"))
                            graph_b = min(cursor2.fetchone()[0] * 0.02, 0.08)
                        except Exception:
                            pass
                    if not _env_disabled("MNEMOSYNE_FACT_BONUS"):
                        try:
                            cursor2 = self.conn.cursor()
                            cursor2.execute(
                                "SELECT subject, predicate, object FROM facts WHERE source_msg_id = ?",
                                (row["id"],))
                            q_word_set = {w for w in query.lower().split() if len(w) > 2}
                            mc = 0
                            for frow in cursor2.fetchall():
                                f_tokens = {t.lower() for t in (f"{frow['subject']} {frow['predicate']} {frow['object']}").split() if len(t) > 2}
                                if q_word_set & f_tokens:
                                    mc += 1
                            fact_b = min(mc * 0.04, 0.1)
                        except Exception:
                            pass
                    # Binary vector bonus disabled (same reason as main path -- ITS clustering)
                    binary_b = 0.0
                    score += graph_b + fact_b + binary_b
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
                    # [C4] Kept-row credit for em_fallback tier.
                    _em_fallback_kept += 1
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "episodic",
                        "score": round(score, 4),
                        "keyword_score": round(relevance, 4),
                        # C30: dense_score is 0.0 by design for EM
                        # fallback rows -- they reach this loop precisely
                        # because the vec/FTS-driven episodic path
                        # produced no candidates (no `sim` is computed
                        # here). Pre-fix this line looked up
                        # `wm_vec_sims[row["id"]]` which always returned
                        # 0.0 since `row["id"]` is an episodic id, not
                        # a working-memory id -- same numeric value,
                        # misleading provenance. Now explicit.
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
                    })

        # --- Tiered degradation weighting: apply tier multiplier to episodic scores ---
        weight_map = {1: TIER1_WEIGHT, 2: TIER2_WEIGHT, 3: TIER3_WEIGHT}
        veracity_map = {"stated": STATED_WEIGHT, "inferred": INFERRED_WEIGHT,
                        "tool": TOOL_WEIGHT, "imported": IMPORTED_WEIGHT,
                        "unknown": UNKNOWN_WEIGHT}
        em_ids_for_tier = [r["id"] for r in results if r.get("tier") == "episodic"]
        # E3.a.3: pull summary_of in the same round trip so the dedup
        # helper can use a precomputed map instead of issuing a second
        # SELECT for the same ep ids.
        ep_summary_of_map: Dict[str, str] = {}
        if em_ids_for_tier:
            placeholders = ",".join("?" * len(em_ids_for_tier))
            tier_rows = cursor.execute(
                f"SELECT id, tier, veracity, summary_of FROM episodic_memory WHERE id IN ({placeholders})",
                em_ids_for_tier
            ).fetchall()
            tier_lookup = {r["id"]: (r["tier"] or 1) for r in tier_rows}
            veracity_lookup = {r["id"]: (r["veracity"] or "unknown") for r in tier_rows}
            ep_summary_of_map = {r["id"]: (r["summary_of"] or "") for r in tier_rows}
            # A/B toggle: `MNEMOSYNE_VERACITY_MULTIPLIER=0` short-circuits
            # the multiplier so ranking depends on hybrid score alone.
            # Useful for Phase 0/1 ablation in the BEAM-recovery
            # experiment. Default ON.
            apply_veracity = not _env_disabled("MNEMOSYNE_VERACITY_MULTIPLIER")
            for r in results:
                if r.get("tier") == "episodic":
                    ep_tier = tier_lookup.get(r["id"], 1)
                    ep_veracity = veracity_lookup.get(r["id"], "unknown")
                    r["degradation_tier"] = ep_tier
                    r["veracity"] = ep_veracity
                    r["score"] *= weight_map.get(ep_tier, 1.0)
                    if apply_veracity:
                        r["score"] *= veracity_map.get(ep_veracity, UNKNOWN_WEIGHT)

        # [E4] Apply the veracity multiplier to working_memory results
        # too. Pre-E4 the multiplier was episodic-only, so per-row
        # veracity on working_memory rows (now populated by
        # remember_batch with per-row labels) had no scoring effect --
        # batch-ingested 'stated' content didn't rank above 'unknown'.
        # The row dicts already carry "veracity" from the SELECT
        # populated earlier in this function, so no second query needed.
        if not _env_disabled("MNEMOSYNE_VERACITY_MULTIPLIER"):
            for r in results:
                if r.get("tier") == "working":
                    wm_veracity = r.get("veracity") or "unknown"
                    r["score"] *= veracity_map.get(wm_veracity, UNKNOWN_WEIGHT)

        # Gap G: linear-path voice_scores parity with the polyphonic
        # engine. Each result already carries per-signal fields
        # (dense_score, fts_score, keyword_score) and ranking inputs
        # (importance, recency_decay). Collapse them into a
        # `voice_scores` dict so downstream analysis can treat linear
        # + polyphonic results uniformly when computing per-signal
        # contributions across arms. The polyphonic engine sets the
        # same field at beam.py:~3544 -- same contract, different keys
        # because the engines have different signal sources.
        for r in results:
            r.setdefault("voice_scores", {
                "vec": r.get("dense_score", 0.0),
                "fts": r.get("fts_score", 0.0),
                "keyword": r.get("keyword_score", 0.0),
                "importance": r.get("importance", 0.0),
                "recency_decay": r.get("recency_decay", 0.0),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        # E3.a.3: collapse (episodic_summary, working_memory_source)
        # duplicates before top-K truncation and recall_count attribution.
        # Post-E3 additive sleep leaves originals alongside summaries, so
        # a query matching both compounds recall_count twice per fact.
        # Pass the precomputed summary_of map from the tier-lookup
        # SELECT above so the helper doesn't issue a redundant query.
        results = self._dedup_cross_tier_summary_links(
            results, ep_summary_of_map=ep_summary_of_map
        )
        final_results = results[:top_k]

        # --- Recall tracking: increment counts + set last_recalled ---
        now_iso = datetime.now().isoformat()
        wm_ids = [r["id"] for r in final_results if r.get("tier") == "working"]
        em_ids = [r["id"] for r in final_results if r.get("tier") == "episodic"]
        cursor = self.conn.cursor()
        if channel_id:
            rec_scope = "(session_id = ? OR scope = 'global' OR channel_id = ?)"
        elif author_id or author_type:
            rec_scope = "(1=1)"
        else:
            rec_scope = "(session_id = ? OR scope = 'global')"
        if wm_ids:
            placeholders = ",".join("?" * len(wm_ids))
            rec_params = [now_iso, *tuple(wm_ids)]
            if channel_id:
                rec_params.extend([self.session_id, channel_id])
            elif not (author_id or author_type):
                rec_params.append(self.session_id)
            cursor.execute(f"""
                UPDATE working_memory
                SET recall_count = recall_count + 1, last_recalled = ?
                WHERE id IN ({placeholders}) AND {rec_scope}
            """, (*rec_params,))
        if em_ids:
            placeholders = ",".join("?" * len(em_ids))
            rec_params = [now_iso, *tuple(em_ids)]
            if channel_id:
                rec_params.extend([self.session_id, channel_id])
            elif not (author_id or author_type):
                rec_params.append(self.session_id)
            cursor.execute(f"""
                UPDATE episodic_memory
                SET recall_count = recall_count + 1, last_recalled = ?
                WHERE id IN ({placeholders}) AND {rec_scope}
            """, (*rec_params,))
        self.conn.commit()

        # [C4] Final tier-attribution records. Each counter holds the
        # number of kept rows attributed to that tier on this call.
        # Summing across tiers gives total kept rows for the call.
        # `truly_empty` is gated on whether ANY layer (primary OR
        # fallback) produced candidates -- distinct from "final
        # results empty after top_k slicing / post-filter dropouts."
        _recall_diag.record_tier_hits("wm_fts", _wm_fts_kept)
        _recall_diag.record_tier_hits("wm_vec", _wm_vec_kept)
        _recall_diag.record_tier_hits("wm_fallback", _wm_fallback_kept)
        _recall_diag.record_tier_hits("em_fts", _em_fts_kept)
        _recall_diag.record_tier_hits("em_vec", _em_vec_kept)
        _recall_diag.record_tier_hits("em_fallback", _em_fallback_kept)
        # truly_empty = final results empty AND no tier attributed
        # a kept row. Distinguishes "post-filter dropouts" (some
        # tier counted hits but they got filtered) from "no signal
        # anywhere" (zero kept across all tiers). top_k=0 callers
        # also land here, but that's an artifact of the caller's
        # choice, not a recall failure -- operators wanting to
        # exclude artifact cases can check top_k > 0 from their
        # side.
        _total_kept = (
            _wm_fts_kept + _wm_vec_kept + _wm_fallback_kept
            + _em_fts_kept + _em_vec_kept + _em_fallback_kept
        )
        _truly_empty = (len(final_results) == 0) and (_total_kept == 0)
        _recall_diag.record_call(truly_empty=_truly_empty)

        return final_results

    def _dedup_cross_tier_summary_links(
        self,
        results: List[Dict],
        *,
        ep_summary_of_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """E3.a.3: drop the lower-scored side of any (episodic_summary,
        working_memory_sources) cluster where both surface in the same recall.

        Pre-E3, `sleep()` DELETEd source `working_memory` rows when creating
        a summary, so dual-surface duplication couldn't happen. Post-E3
        (additive sleep), sources survive alongside summaries by design.
        A recall whose query matches both raw and summary text ranks them
        side-by-side AND compounds `recall_count` twice for the same
        logical fact -- the row's history boost double-counts on every call.

        Dedup rule (per-cluster, not per-edge):
          - For each episodic row with non-empty `summary_of`, collect the
            wm_ids it covers that are also present in `results`.
          - If the ep's score is >= the score of EVERY covered wm in
            results, drop those wms and keep the ep (summary wins the
            whole cluster).
          - Otherwise -- some covered wm beats the ep -- drop the ep and
            keep all covered wms (sources win; the dropped ep no longer
            represents those wms in the result set).

        Per-cluster decisions avoid the per-edge bug where a wm could
        lose to a summary that itself was being dropped by a different
        wm. Example fixed by this shape: ep covers wm-1 (0.9) + wm-2 (0.3)
        with ep at 0.6. Per-edge would drop ep (lost to wm-1) AND wm-2
        (lost to ep) -- but wm-2's representative ep is itself gone, so
        wm-2 was being dropped against a phantom. Per-cluster correctly
        keeps both wms.

        Ties (ep_score == wm_score) keep the episodic side (later-stage
        representation; matches polyphonic engine's diversity-rerank
        posture). The comparison runs on the post-multiplier `score`
        field, so the dedup decision reflects the rank the user would
        have seen.

        Preserves input order on retained rows. Returns the input list
        unchanged (same object) if no episodic rows are present or no
        summary_of linkage exists.

        Args:
            results: scored row dicts; each carries `id`, `tier`, `score`.
            ep_summary_of_map: optional precomputed `{ep_id: summary_of_str}`
                from a caller that already SELECT-ed `episodic_memory`
                rows. When provided, skips the helper's own SELECT -- keeps
                a single source of truth and avoids one round-trip per
                recall on paths that have already fetched the data.

        Caller pattern: linear path passes the tier-lookup SELECT's
        precomputed `summary_of` rows; polyphonic path lets the helper
        do its own SELECT since it has no prior per-ep query.

        Caveats:
          - Does NOT dedup ep ↔ ep (two summaries covering overlapping
            wm sets). `sleep()` doesn't re-summarize already-consolidated
            rows by design (it skips `consolidated_at IS NOT NULL` per
            E3), so this is rare in practice. If it happens via external
            re-consolidation tooling, both summaries survive.
          - The summary_of SELECT and the subsequent recall_count UPDATE
            are NOT wrapped in a single transaction. A concurrent
            `sleep()` / `forget()` between them could yield stale linkage
            data. Acceptable under SQLite WAL + busy_timeout: the worst
            case is a one-call dedup miss, not data loss.

        A/B toggle: `MNEMOSYNE_CROSS_TIER_DEDUP=0` disables the dedup,
        returning the input list unchanged. Used by the BEAM-recovery
        Phase 4 ablation to isolate the dedup's contribution.
        """
        if _env_disabled("MNEMOSYNE_CROSS_TIER_DEDUP"):
            return results
        ep_ids = [r["id"] for r in results if r.get("tier") == "episodic"]
        if not ep_ids:
            return results

        # Build summary_map from either precomputed map or own SELECT.
        summary_map: Dict[str, set] = {}
        if ep_summary_of_map is not None:
            for ep_id in ep_ids:
                raw = ep_summary_of_map.get(ep_id) or ""
                wm_ids = {s.strip() for s in raw.split(",") if s.strip()}
                if wm_ids:
                    summary_map[ep_id] = wm_ids
        else:
            placeholders = ",".join("?" * len(ep_ids))
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT id, summary_of FROM episodic_memory WHERE id IN ({placeholders})",
                tuple(ep_ids),
            )
            for row in cursor.fetchall():
                raw = row["summary_of"] or ""
                wm_ids = {s.strip() for s in raw.split(",") if s.strip()}
                if wm_ids:
                    summary_map[row["id"]] = wm_ids

        if not summary_map:
            return results

        # Per-tier score lookups disambiguate cross-tier id collisions
        # (theoretically possible since `id TEXT PRIMARY KEY` is per-table).
        wm_scores = {r["id"]: r.get("score", 0.0) for r in results
                     if r.get("tier") == "working"}
        ep_scores = {r["id"]: r.get("score", 0.0) for r in results
                     if r.get("tier") == "episodic"}
        drop_wm_ids: set = set()
        drop_ep_ids: set = set()

        for ep_id, covered_wm_ids in summary_map.items():
            if ep_id not in ep_scores:
                continue
            ep_score = ep_scores[ep_id]
            # Filter to wms actually present in results.
            present_wms = [w for w in covered_wm_ids if w in wm_scores]
            if not present_wms:
                continue
            # Per-cluster: ep wins only if it beats or ties EVERY present wm.
            ep_wins_cluster = all(ep_score >= wm_scores[w] for w in present_wms)
            if ep_wins_cluster:
                drop_wm_ids.update(present_wms)
            else:
                drop_ep_ids.add(ep_id)

        if not (drop_wm_ids or drop_ep_ids):
            return results

        return [
            r for r in results
            if not (
                (r.get("tier") == "working" and r["id"] in drop_wm_ids)
                or (r.get("tier") == "episodic" and r["id"] in drop_ep_ids)
            )
        ]

    # ── Phase NAI-0: Context Formatting ────────────────────────────

    def _sandwich_order(self, results: List[Dict], top_k: int = 10) -> dict:
        """Sort by score and partition into high/medium/closing for sandwich ordering.

        U-shaped attention: LLMs pay most attention to first AND last items.
        High-scored facts go first, medium in the middle, high-scored again at end.
        """
        scored = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        high = [r for r in scored if r.get("score", 0) > 0.7][:3]
        medium = [r for r in scored if 0.3 < r.get("score", 0) <= 0.7][:5]
        # Closing: last few high-scored items (not already in high)
        closing_pool = [r for r in scored if r not in high][:3]
        closing = closing_pool if closing_pool else high[:2]
        return {"high": high, "medium": medium, "closing": closing}

    def _fact_line(self, result: Dict) -> str:
        """Clean one-line fact: 'User prefers dark mode (2026-05-09, user, c:0.9)'"""
        content = (result.get("content") or "")[:200].strip()
        ts_raw = result.get("timestamp") or ""
        ts = ts_raw[:10] if ts_raw else "?"
        source = result.get("source", "unknown")
        score = result.get("score") or result.get("importance") or 0
        return f"{content} ({ts}, {source}, c:{score:.1f})"

    def format_context(self, results: List[Dict], format: str = "bullet") -> str:
        """Format recall results as structured context for LLM injection.

        Args:
            results: List of recall result dicts (from recall() or polyphonic recall)
            format: 'bullet' (default) for markdown bullets, 'json' for structured JSON

        Returns:
            Formatted context string ready for LLM prompt injection.
        """
        sandwich = self._sandwich_order(results)

        if format == "json":
            return self._format_context_json(sandwich)
        return self._format_context_bullet(sandwich)

    def _format_context_json(self, sandwich: dict) -> str:
        """JSON structured context with sandwich ordering."""
        import json as _json
        context = {
            "top_facts": [self._fact_line(r) for r in sandwich["high"]],
            "supporting_context": [self._fact_line(r) for r in sandwich["medium"]],
            "recent_memories": [self._fact_line(r) for r in sandwich["closing"]],
            "total_memories": sum(len(v) for v in sandwich.values()),
        }
        return _json.dumps(context, indent=2, ensure_ascii=False)

    def _format_context_bullet(self, sandwich: dict) -> str:
        """Bullet-point context with sandwich ordering (U-shaped attention).

        Highest-scored first, medium middle, high-scored again at end.
        """
        lines = []
        lines.append("## Top Facts")
        for r in sandwich["high"]:
            lines.append(f"- {self._fact_line(r)}")
        if sandwich["medium"]:
            lines.append("")
            lines.append("## Supporting Context")
            for r in sandwich["medium"]:
                lines.append(f"- {self._fact_line(r)}")
        if sandwich["closing"]:
            lines.append("")
            lines.append("## Recent Signals")
            for r in sandwich["closing"]:
                lines.append(f"- {self._fact_line(r)}")
        total = sum(len(v) for v in sandwich.values())
        lines.append(f"\n_({total} memories retrieved)_")
        return "\n".join(lines)

    def _recall_polyphonic(self, query: str, top_k: int,
                           *,
                           from_date: Optional[str] = None,
                           to_date: Optional[str] = None,
                           source: Optional[str] = None,
                           topic: Optional[str] = None,
                           author_id: Optional[str] = None,
                           author_type: Optional[str] = None,
                           channel_id: Optional[str] = None,
                           veracity: Optional[str] = None,
                           memory_type: Optional[str] = None) -> List[Dict]:
        """[E5] Polyphonic recall path.

        Delegates to PolyphonicRecallEngine when MNEMOSYNE_POLYPHONIC_RECALL=1.
        Engine runs vector + graph + fact + temporal voices, fuses via RRF,
        diversity-reranks, assembles within a context budget. Maps
        PolyphonicResult objects back to recall()'s dict shape.

        Each result carries `voice_scores` for per-signal provenance.
        Engine's RRF combined_score lands in the `score` field; the
        post-E4 veracity multiplier and tier-degradation multiplier
        are then composed on top so flag=ON callers don't lose the
        recent ranking work.

        Filters from the caller (session/scope/valid_until/superseded/
        author/channel/source/veracity/memory_type/from_date/to_date)
        are applied during row fetch -- same isolation contract as the
        linear path. /review caught the original implementation
        bypassing these (data-isolation regression, P1).

        Synthetic fact-voice ids (`cf_...`) currently can't be mapped
        back to source rows (the engine returns the fact key, not the
        producing memory_id). They're skipped; the fact voice still
        contributes ranking signal via RRF onto real memory_ids
        surfaced by other voices. Known limitation, documented in
        CHANGELOG.
        """
        engine = self._get_polyphonic_engine()

        query_embedding = None
        if _embeddings.available():
            try:
                vecs = _embeddings.embed([query])
                if vecs is not None and len(vecs) > 0:
                    query_embedding = vecs[0]
            except Exception:
                query_embedding = None

        try:
            polyphonic_results = engine.recall(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k * 2,  # over-fetch for filter dropouts
            )
        except Exception as exc:
            logger.exception("polyphonic recall engine failed: %s", exc)
            return []

        # Map → recall's dict shape with filters + multipliers applied.
        weight_map = {"stated": STATED_WEIGHT, "inferred": INFERRED_WEIGHT,
                      "tool": TOOL_WEIGHT, "imported": IMPORTED_WEIGHT,
                      "unknown": UNKNOWN_WEIGHT}
        tier_weight_map = {1: TIER1_WEIGHT, 2: TIER2_WEIGHT, 3: TIER3_WEIGHT}

        final = []
        cursor = self.conn.cursor()
        now_iso = datetime.now().isoformat()

        for r in polyphonic_results:
            memory_id = r.memory_id
            if memory_id.startswith("cf_"):
                continue
            row_dict = self._fetch_polyphonic_row(cursor, memory_id)
            if row_dict is None:
                continue

            # Apply caller-supplied filters and the always-on
            # isolation/validity contract.
            if not self._polyphonic_row_passes_filters(
                row_dict, from_date=from_date, to_date=to_date,
                source=source, topic=topic, author_id=author_id,
                author_type=author_type, channel_id=channel_id,
                veracity=veracity, memory_type=memory_type, now_iso=now_iso,
            ):
                continue

            # Compose RRF combined_score with post-E4 multipliers so
            # flag=ON callers don't silently lose the veracity rank
            # signal or tier degradation. Veracity multiplier applies
            # to both tiers (matching the post-E4 linear behavior).
            # A/B toggle: `MNEMOSYNE_VERACITY_MULTIPLIER=0` disables
            # veracity scaling here too -- mirroring the linear path so
            # both arms ablate identically.
            score = r.combined_score
            row_veracity = row_dict.get("veracity") or "unknown"
            if not _env_disabled("MNEMOSYNE_VERACITY_MULTIPLIER"):
                score *= weight_map.get(row_veracity, UNKNOWN_WEIGHT)
            if row_dict.get("tier") == "episodic":
                ep_tier = row_dict.get("degradation_tier") or 1
                score *= tier_weight_map.get(ep_tier, 1.0)

            row_dict["score"] = score
            row_dict["voice_scores"] = dict(r.voice_scores)
            final.append(row_dict)
            # No early-break: dedup needs to see all candidates before
            # truncation, otherwise a wm row dropped from top-K by an
            # earlier-arriving ep summary can't be re-promoted when the
            # ep summary itself gets deduped away. The engine already
            # caps engine.recall(top_k=top_k * 2) above, bounding the loop.

        # Re-sort post-multiplier composition so the final order reflects
        # both RRF and the veracity/tier weights.
        final.sort(key=lambda x: x["score"], reverse=True)
        # E3.a.3: apply identical cross-tier dedup as the linear path --
        # keeps experiment Arm A vs Arm B comparison apples-to-apples
        # rather than relying on the diversity rerank to handle
        # summary↔source duplicates implicitly.
        final = self._dedup_cross_tier_summary_links(final)
        final = final[:top_k]
        # Rebuild recall_count attribution lists from the deduped final
        # so dropped duplicates aren't credited with a recall.
        recalled_episodic_ids = [r["id"] for r in final if r.get("tier") == "episodic"]
        recalled_working_ids = [r["id"] for r in final if r.get("tier") == "working"]

        # E3.a.3 review fix: apply the same session/channel/scope guard
        # the linear path uses (beam.py:~2734-2763). Pre-fix the
        # polyphonic UPDATEs ran on `WHERE id IN (...)` with no scope
        # check, so a recall returning a foreign-session row would bump
        # that row's recall_count, polluting cross-session ranking. The
        # in-loop construction this commit removed was reinforcing the
        # gap; rebuilding from `final` post-dedup is the right shape, so
        # add the scope guard here too.
        if channel_id:
            rec_scope = "(session_id = ? OR scope = 'global' OR channel_id = ?)"
        elif author_id or author_type:
            rec_scope = "(1=1)"
        else:
            rec_scope = "(session_id = ? OR scope = 'global')"

        def _rec_scope_params() -> List:
            if channel_id:
                return [self.session_id, channel_id]
            if author_id or author_type:
                return []
            return [self.session_id]

        # Update recall_count / last_recalled for engine results too --
        # the linear path updates them and downstream features (decay
        # scheduling, importance reinforcement) depend on the signal.
        # /review caught the missing update as a silent telemetry loss.
        if recalled_episodic_ids:
            placeholders = ",".join("?" * len(recalled_episodic_ids))
            params = [now_iso, *recalled_episodic_ids, *_rec_scope_params()]
            self.conn.execute(
                f"UPDATE episodic_memory SET recall_count = recall_count + 1, "
                f"last_recalled = ? WHERE id IN ({placeholders}) AND {rec_scope}",
                tuple(params),
            )
        if recalled_working_ids:
            placeholders = ",".join("?" * len(recalled_working_ids))
            params = [now_iso, *recalled_working_ids, *_rec_scope_params()]
            self.conn.execute(
                f"UPDATE working_memory SET recall_count = recall_count + 1, "
                f"last_recalled = ? WHERE id IN ({placeholders}) AND {rec_scope}",
                tuple(params),
            )
        if recalled_episodic_ids or recalled_working_ids:
            self.conn.commit()

        return final

    def _get_polyphonic_engine(self):
        """Lazy-cached engine instance per BeamMemory.

        Pre-fix: a fresh engine was constructed on every recall call,
        which re-ran BinaryVectorStore + EpisodicGraph + VeracityConsolidator
        constructors and their schema-ensure SQL (`CREATE TABLE IF NOT
        EXISTS` + `CREATE INDEX IF NOT EXISTS` + commit) on every call.
        Under prefetch/A/B-flag workloads that's a wasteful commit
        storm. /review caught the per-call instantiation.

        Engine state is read-only between calls; the shared
        `self.conn` is the only mutable dependency and it's pinned at
        BeamMemory init.
        """
        if getattr(self, "_polyphonic_engine", None) is None:
            from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine
            self._polyphonic_engine = PolyphonicRecallEngine(
                db_path=self.db_path, conn=self.conn,
            )
        return self._polyphonic_engine

    def _polyphonic_row_passes_filters(self, row_dict: Dict, *,
                                       from_date: Optional[str],
                                       to_date: Optional[str],
                                       source: Optional[str],
                                       topic: Optional[str],
                                       author_id: Optional[str],
                                       author_type: Optional[str],
                                       channel_id: Optional[str],
                                       veracity: Optional[str],
                                       memory_type: Optional[str],
                                       now_iso: str) -> bool:
        """Mirror the linear path's filter set for the engine path.
        Always-on filters: session scope, valid_until, superseded_by.
        Conditional filters: caller-supplied kwargs.
        """
        # Always-on session/scope isolation: only return rows visible
        # to this session. Matches the linear path's WHERE clauses for
        # both working_memory (session_id = self.session_id OR scope =
        # 'global') and episodic_memory (same shape).
        row_session = row_dict.get("session_id") if "session_id" in row_dict else None
        # Some rows don't carry session_id in the engine row dict -- re-fetch
        # via the cursor to enforce. For now, treat global scope as
        # always-visible and same-session as visible.
        row_scope = row_dict.get("scope") or "session"
        if row_scope != "global" and row_session is not None and row_session != self.session_id:
            return False

        # Validity filters.
        valid_until = row_dict.get("valid_until")
        if valid_until and valid_until <= now_iso:
            return False
        if row_dict.get("superseded_by"):
            return False

        # Caller-supplied filters.
        if from_date and (row_dict.get("timestamp") or "") < from_date:
            return False
        if to_date and (row_dict.get("timestamp") or "") > to_date:
            return False
        if source and row_dict.get("source") != source:
            return False
        if topic and topic not in (row_dict.get("source") or ""):
            return False
        if author_id and row_dict.get("author_id") != author_id:
            return False
        if author_type and row_dict.get("author_type") != author_type:
            return False
        if channel_id and row_dict.get("channel_id") != channel_id:
            return False
        if veracity and row_dict.get("veracity") != veracity:
            return False
        if memory_type and row_dict.get("memory_type") != memory_type:
            return False

        return True

    def _fetch_polyphonic_row(self, cursor, memory_id: str) -> Optional[Dict]:
        """Resolve a memory_id from the polyphonic engine to a row
        dict matching recall()'s existing return shape. Tries episodic
        first, then working_memory; returns None if neither table
        has the row (engine returned a stale or synthetic id).

        Includes session_id in the SELECT so the filter pass can
        enforce session-scope isolation post-fetch.
        """
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance,
                   recall_count, last_recalled, valid_until,
                   superseded_by, scope, author_id, author_type,
                   channel_id, veracity, memory_type, tier
            FROM episodic_memory WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        if row is not None:
            return self._polyphonic_row_to_dict(row, tier_label="episodic")
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance,
                   recall_count, last_recalled, valid_until,
                   superseded_by, scope, author_id, author_type,
                   channel_id, veracity, memory_type
            FROM working_memory WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._polyphonic_row_to_dict(row, tier_label="working")

    def _polyphonic_row_to_dict(self, row, *, tier_label: str) -> Dict:
        """Shared row → recall-dict mapper. /review caught the
        near-duplicate column mapping across episodic/working
        branches -- single helper now."""
        d = {
            "id": row["id"],
            "content": row["content"],
            "source": row["source"],
            "timestamp": row["timestamp"],
            "session_id": row["session_id"] if "session_id" in row.keys() else None,
            "importance": row["importance"],
            "recall_count": row["recall_count"] or 0,
            "last_recalled": row["last_recalled"],
            "scope": row["scope"] if "scope" in row.keys() else "session",
            "author_id": row["author_id"] if "author_id" in row.keys() else None,
            "author_type": row["author_type"] if "author_type" in row.keys() else None,
            "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
            "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
            "memory_type": row["memory_type"] if "memory_type" in row.keys() else "unknown",
            "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
            "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
            "tier": tier_label,
        }
        if tier_label == "episodic":
            d["degradation_tier"] = row["tier"] if "tier" in row.keys() else 1
        return d

    def fact_recall(self, query: str, top_k: int = 30) -> List[Dict]:
        """Search the facts table (LLM-extracted structured knowledge).

        Returns facts as list of dicts with: content, score, fact_id, subject, predicate.

        Falls back gracefully if facts table is empty or sqlite-vec unavailable.
        """
        cursor = self.conn.cursor()
        results = []
        query_lower = query.lower()

        # Try FTS5 search first
        try:
            fts_rows = cursor.execute(
                "SELECT rowid, rank FROM fts_facts WHERE fts_facts MATCH ? ORDER BY rank, rowid LIMIT ?",
                (query, top_k * 3)
            ).fetchall()
        except Exception:
            fts_rows = []

        if not fts_rows:
            # Fallback: simple LIKE scan
            for word in query_lower.split()[:6]:
                if len(word) < 3:
                    continue
                try:
                    like_rows = cursor.execute(
                        "SELECT rowid FROM facts WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? LIMIT ?",
                        (f"%{word}%", f"%{word}%", f"%{word}%", top_k)
                    ).fetchall()
                except Exception:
                    continue
                for row in like_rows:
                    if row["rowid"] not in {r["rowid"] for r in fts_rows}:
                        fts_rows.append({"rowid": row["rowid"], "rank": 0})

        if not fts_rows:
            return []

        # Get full rows for matched fact IDs
        fact_ids = [r["rowid"] for r in fts_rows[:top_k]]
        placeholders = ",".join("?" * len(fact_ids))

        try:
            cursor.execute(f"""
                SELECT fact_id, subject, predicate, object,
                       timestamp, confidence
                FROM facts
                WHERE rowid IN ({placeholders})
                ORDER BY confidence DESC
                LIMIT ?
            """, (*fact_ids, top_k))
            fact_rows = cursor.fetchall()
        except Exception:
            return []

        for raw_row in fact_rows:
            # sqlite3.Row supports bracket access but not .get(); convert to
            # dict so the column-with-default reads below work. Without this
            # conversion fact_recall crashes the moment the facts table
            # contains rows -- a latent bug that was masked while the
            # Mnemosyne.remember(extract=True) wrapper never populated the
            # table (see C12.a).
            row = dict(raw_row)
            confidence = row.get("confidence")
            subject = row.get("subject")
            predicate = row.get("predicate")
            obj = row.get("object")
            fact_text = obj if obj else f"{subject} {predicate} {obj}"
            results.append({
                "content": fact_text,
                "score": confidence if confidence is not None else 0.5,
                "fact_id": row["fact_id"],
                "subject": subject if subject is not None else "",
                "predicate": predicate if predicate is not None else "",
            })

        return results

    def get_episodic_stats(self, author_id: str = None, author_type: str = None,
                           channel_id: str = None) -> Dict:
        cursor = self.conn.cursor()
        where_clauses = []
        params = []
        if author_id:
            where_clauses.append("author_id = ?")
            params.append(author_id)
        if author_type:
            where_clauses.append("author_type = ?")
            params.append(author_type)
        if channel_id:
            where_clauses.append("channel_id = ?")
            params.append(channel_id)
        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        cursor.execute(f"SELECT COUNT(*) FROM episodic_memory{where_str}", params)
        total = cursor.fetchone()[0]
        cursor.execute(f"SELECT timestamp FROM episodic_memory{where_str} ORDER BY timestamp DESC LIMIT 1", params)
        last = cursor.fetchone()
        vec_count = 0
        vec_type = "none"
        if _vec_available(self.conn):
            try:
                vec_count = cursor.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
                vec_type = _effective_vec_type(self.conn)
            except Exception:
                pass
        return {"total": total, "last": last[0] if last else None, "vectors": vec_count, "vec_type": vec_type}

    # ------------------------------------------------------------------
    # Scratchpad
    # ------------------------------------------------------------------
    def scratchpad_write(self, content: str) -> str:
        pad_id = _generate_id(content)
        ts = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO scratchpad (id, content, session_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at
        """, (pad_id, content, self.session_id, ts, ts))
        self.conn.commit()
        return pad_id

    def scratchpad_read(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT id, content, created_at, updated_at
            FROM scratchpad
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT {SCRATCHPAD_MAX_ITEMS}
        """, (self.session_id,))
        return [dict(row) for row in cursor.fetchall()]

    def scratchpad_clear(self):
        self.conn.execute("DELETE FROM scratchpad WHERE session_id = ?", (self.session_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Tiered Episodic Degradation
    # ------------------------------------------------------------------
    def _extract_key_signal(self, content: str, max_chars: int = 300) -> str:
        """Extract the highest-signal sentences from content for tier 3 compression.

        Scores each sentence by entity/keyword density (proper nouns, technical
        terms, preference indicators) and keeps top-scoring sentences until the
        character budget is reached. Falls back to first-N-chars if content has
        no clear sentence boundaries.
        """
        import re
        if len(content) <= max_chars:
            return content

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) <= 1:
            # No sentence boundaries -- take first max_chars
            return content[:max_chars] + " [...]"

        # Scoring patterns
        signal_patterns = [
            (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', 3),     # Proper nouns: "GitHub Actions", "Docker Compose"
            (r'\b[A-Z]{2,}\b', 3),                            # Acronyms: "XKCD", "CI/CD", "API", "AWS"
            (r'\b(Docker|Kubernetes|AWS|GCP|Azure|Terraform|Python|Rust|Go|TypeScript|React|Next\.?js|Node\.?js|SQLite|Postgres|Redis|nginx|systemd|Linux|macOS|Windows)\b', 4),
            (r'\b(prefers?|uses?|likes?|loves?|hates?|dislikes?|wants?|needs?)\b', 2),  # Preference indicators
            (r'\b(password|token|secret|key|credential|auth|encrypt|decrypt|private)\b', 3),  # Security terms
            (r'\b(production|staging|deploy|database|backup|migration)\b', 2),  # Infra terms
            (r'\b(critical|urgent|important|breaking|incident|outage|down)\b', 3),  # Urgency
            (r'\b(always|never|every|must|should)\b', 1),  # Emphasis words
            (r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\b', 3),  # Version numbers
            (r'\b(https?://|www\.|[a-z]+\.[a-z]{2,})\b', 2),  # URLs / domains
            (r'["\'].*?["\']', 1),  # Quoted strings
        ]

        scored = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            score = 0
            # Bonus for shorter sentences (signal density)
            if len(sentence) < 120:
                score += 1
            for pattern, weight in signal_patterns:
                score += len(re.findall(pattern, sentence)) * weight
            scored.append((score, sentence))

        # Sort by score descending, keep top sentences up to max_chars
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        used = 0
        for _, sentence in scored:
            if used + len(sentence) + 1 > max_chars:
                break
            result.append(sentence)
            used += len(sentence) + 1  # +1 for space

        if not result:
            return content[:max_chars] + " [...]"

        compressed = " ".join(result)
        if len(content) > len(compressed):
            compressed += " [...]"
        return compressed

    def _refresh_episodic_embedding(self, memory_id: str, rowid: int, new_content: str):
        """Refresh dense-recall embedding stores for an episodic row whose
        content has been mutated (degraded). Without this the
        vec_episodes / memory_embeddings / binary_vector entries continue
        representing the pre-mutation content, so dense recall scores
        rows by semantics that no longer match what the row displays.
        See C18.b in the memory-contract ledger.

        - If embeddings provider is available: regenerate using the new
          content and overwrite the existing vector store entries.
        - If unavailable: invalidate (DELETE / NULL) the stale entries so
          dense recall stops returning semantically misleading hits. The
          row remains discoverable via FTS.
        """
        cursor = self.conn.cursor()

        vec_available_now = _vec_available(self.conn)

        if _embeddings.available():
            try:
                vec = _embeddings.embed([new_content])
            except Exception:
                vec = None
            if vec is not None:
                # vec_episodes is a sqlite-vec virtual table; vec0 doesn't
                # support UPDATE on the embedding column reliably, so we
                # DELETE+INSERT to refresh.
                if vec_available_now:
                    cursor.execute("DELETE FROM vec_episodes WHERE rowid = ?", (rowid,))
                    _vec_insert(self.conn, rowid, vec[0].tolist())
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                        VALUES (?, ?, ?)
                    """, (memory_id, _embeddings.serialize(vec[0]), _embeddings._DEFAULT_MODEL))

                if _mib is not None:
                    try:
                        bv = _mib(vec[0])
                        cursor.execute(
                            "UPDATE episodic_memory SET binary_vector = ? WHERE id = ?",
                            (bv, memory_id),
                        )
                    except Exception:
                        pass
                return

        # Provider unavailable (or embed() returned None). Invalidate the
        # stale entries so dense recall doesn't lie. The row keeps its
        # FTS-searchable content and remains otherwise intact. Each DELETE
        # is gated on the matching store's availability -- vec_episodes is
        # a sqlite-vec virtual table that doesn't exist when the extension
        # isn't loaded, so an unconditional DELETE there raises
        # OperationalError and the caller's broad except would silently
        # skip the memory_embeddings cleanup too.
        if vec_available_now:
            cursor.execute("DELETE FROM vec_episodes WHERE rowid = ?", (rowid,))
        cursor.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
        if _mib is not None:
            cursor.execute(
                "UPDATE episodic_memory SET binary_vector = NULL WHERE id = ?",
                (memory_id,),
            )

    def degrade_episodic(self, dry_run: bool = False) -> Dict:
        """Degrade old episodic memories through tier 1→2→3 compression.

        Tier 1 (0-TIER2_DAYS): Full detail, 1.0x recall weight
        Tier 2 (TIER2_DAYS-TIER3_DAYS): LLM-summarized, 0.5x weight
        Tier 3 (TIER3_DAYS+): Text extraction compressed, 0.25x weight

        Each tier transition that mutates content also refreshes the
        row's dense-recall embedding (or invalidates it if the embeddings
        provider is unavailable) so vec_episodes / memory_embeddings /
        binary_vector stay aligned with the displayed text. See C18.b.

        Returns summary of tier transitions performed.
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        results = {"status": "dry_run" if dry_run else "degraded",
                   "tier1_to_tier2": 0, "tier2_to_tier3": 0}

        # --- Find candidates for degradation ---
        tier2_cutoff = (now - timedelta(days=TIER2_DAYS)).isoformat()
        tier3_cutoff = (now - timedelta(days=TIER3_DAYS)).isoformat()

        # Tier 1 → Tier 2: old enough, still at tier 1.
        # rowid is selected so the embedding refresh can address vec_episodes.
        cursor.execute("""
            SELECT id, rowid, content, importance FROM episodic_memory
            WHERE tier = 1 AND created_at < ?
            ORDER BY created_at ASC LIMIT ?
        """, (tier2_cutoff, DEGRADE_BATCH_SIZE))
        tier1_rows = cursor.fetchall()

        # Tier 2 → Tier 3: very old, at tier 2
        cursor.execute("""
            SELECT id, rowid, content FROM episodic_memory
            WHERE tier = 2 AND created_at < ?
            ORDER BY created_at ASC LIMIT ?
        """, (tier3_cutoff, DEGRADE_BATCH_SIZE // 2))
        tier2_rows = cursor.fetchall()

        if dry_run:
            results["tier1_to_tier2"] = len(tier1_rows)
            results["tier2_to_tier3"] = len(tier2_rows)
            return results

        # --- Degrade tier 1 → tier 2: LLM summarization ---
        # Each row's UPDATE + embedding refresh runs inside a SAVEPOINT so
        # a refresh failure rolls back the content mutation too. Without
        # this the broad except below would swallow the refresh exception
        # while leaving the UPDATE staged in the implicit transaction,
        # which then commits at the end of degrade_episodic -- producing
        # the very content/embedding drift this fix exists to prevent
        # (caught by /review for C18.b).
        from mnemosyne.core import local_llm
        for row in tier1_rows:
            cursor.execute("SAVEPOINT degrade_row")
            try:
                compressed = row["content"]
                if local_llm.llm_available() and len(row["content"]) > 300:
                    summary = local_llm.summarize_memories([row["content"]])
                    if summary:
                        compressed = summary[:400]
                final_content = compressed[:800]
                cursor.execute(
                    "UPDATE episodic_memory SET content = ?, tier = 2, degraded_at = ? WHERE id = ?",
                    (final_content, now.isoformat(), row["id"])
                )
                # Only refresh the embedding when content actually changed.
                # If LLM was unavailable and content is unchanged the
                # existing embedding is already correct and an embed()
                # call would be wasted.
                if final_content != row["content"]:
                    self._refresh_episodic_embedding(row["id"], row["rowid"], final_content)
                cursor.execute("RELEASE degrade_row")
                results["tier1_to_tier2"] += 1
            except Exception:
                try:
                    cursor.execute("ROLLBACK TO degrade_row")
                    cursor.execute("RELEASE degrade_row")
                except Exception:
                    pass

        # --- Degrade tier 2 → tier 3: smart extraction (keep key entities) ---
        for row in tier2_rows:
            cursor.execute("SAVEPOINT degrade_row")
            try:
                content = row["content"]
                if SMART_COMPRESS and len(content) > TIER3_MAX_CHARS:
                    compressed = self._extract_key_signal(content, max_chars=TIER3_MAX_CHARS)
                else:
                    compressed = content[:TIER3_MAX_CHARS]
                    if len(content) > TIER3_MAX_CHARS:
                        compressed += " [...]"
                cursor.execute(
                    "UPDATE episodic_memory SET content = ?, tier = 3, degraded_at = ? WHERE id = ?",
                    (compressed, now.isoformat(), row["id"])
                )
                if compressed != row["content"]:
                    self._refresh_episodic_embedding(row["id"], row["rowid"], compressed)
                cursor.execute("RELEASE degrade_row")
                results["tier2_to_tier3"] += 1
            except Exception:
                try:
                    cursor.execute("ROLLBACK TO degrade_row")
                    cursor.execute("RELEASE degrade_row")
                except Exception:
                    pass

        self.conn.commit()
        return results

    def get_contaminated(self, limit: int = 50, min_importance: float = 0.0) -> List[Dict]:
        """Return potentially contaminated memories for review.

        Contaminated = veracity in ('inferred', 'tool', 'imported', 'unknown')
        -- i.e., anything not explicitly stated by the user. Sorted by
        importance descending so the highest-stakes items surface first.

        Args:
            limit: Max memories to return
            min_importance: Only return memories with importance >= this
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, source, veracity, tier, importance,
                   created_at, degraded_at, session_id
            FROM episodic_memory
            WHERE veracity IN ('inferred', 'tool', 'imported', 'unknown')
              AND importance >= ?
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """, (min_importance, limit))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Consolidation Health Check
    # ------------------------------------------------------------------
    def health(self, stale_threshold_hours: float = 24.0) -> Dict:
        """Return consolidation health status for monitoring/alerting.

        Checks:
        - last successful consolidation timestamp (from consolidation_log)
        - error count in recent attempts (last 100 log entries)
        - stale threshold alert: no consolidation in `stale_threshold_hours`

        Returns a dict with keys: ``status`` (\"healthy\" | \"stale\" | \"no_data\"),
        ``last_successful_consolidation``, ``error_count``, ``stale_hours``,
        ``details``, and ``recommendation``.
        """
        cursor = self.conn.cursor()

        # Last successful consolidation across ALL sessions (not just
        # self.session_id) so a health monitor run from an active session
        # can detect that an inactive-session's sleep_all_sessions
        # maintenance broke silently.
        cursor.execute("""
            SELECT max(created_at) AS last_consolidation
            FROM consolidation_log
            WHERE items_consolidated > 0
        """)
        row = cursor.fetchone()
        last_ts_str = row["last_consolidation"] if row and row["last_consolidation"] else None

        # Error count: look at entries with zero items_consolidated but
        # a non-empty summary_preview that suggests an attempted run.
        # Also scan sleep_all_sessions "errors" recorded via
        # summary_preview text patterns.
        cursor.execute("""
            SELECT count(*) AS err_count
            FROM consolidation_log
            WHERE created_at > datetime('now', '-7 days')
              AND (
                  items_consolidated = 0
                  AND summary_preview LIKE '%error%'
                  OR summary_preview LIKE '%fail%'
              )
        """)
        error_count = cursor.fetchone()["err_count"]

        now = datetime.now()

        # Determine status
        if last_ts_str is None:
            status = "no_data"
            stale_hours = None
            recommendation = (
                "No consolidation_log entries found with items_consolidated > 0. "
                "Either sleep() has never run, or all runs have produced zero "
                "summaries. Run sleep_all_sessions() or check logs."
            )
        else:
            last_ts = datetime.fromisoformat(last_ts_str)
            stale_hours = round((now - last_ts).total_seconds() / 3600.0, 2)
            if stale_hours > stale_threshold_hours:
                status = "stale"
                recommendation = (
                    f"Last successful consolidation was {stale_hours:.1f} hours ago "
                    f"(threshold: {stale_threshold_hours:.0f}h). "
                    "Run sleep_all_sessions() to catch up, and investigate why "
                    "scheduled consolidation stopped (e.g. LLM unreachable, "
                    "silent failures in summarize_memories, or cron/loop down)."
                )
            else:
                status = "healthy"
                recommendation = "Consolidation is within the healthy window."

        return {
            "status": status,
            "last_successful_consolidation": last_ts_str,
            "error_count": error_count,
            "stale_hours": stale_hours,
            "stale_threshold_hours": stale_threshold_hours,
            "details": {
                "stale": status == "stale",
                "consolidation_log_entries_checked": "last 7 days",
            },
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Consolidation / Sleep
    # ------------------------------------------------------------------
    def sleep(self, dry_run: bool = False) -> Dict:
        """
        Consolidate old working_memory for this session into episodic summaries.
        Uses a local lightweight LLM when available; falls back to aaak
        compression if the model is missing or inference fails.
        Returns summary of what was done.

        Post-E3 (additive): the source working_memory rows are NOT
        deleted. Instead they're marked with consolidated_at = NOW
        so the next sleep cycle skips them. Originals remain
        recallable alongside the new episodic summary.

        Note: this method intentionally remains session-scoped. Use
        sleep_all_sessions() for maintenance that consolidates eligible old
        working memories across inactive sessions.
        """
        from mnemosyne.core.aaak import encode as aaak_encode
        from mnemosyne.core import local_llm

        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)).isoformat()
        # COALESCE(session_id, 'default') so a "default"-session beam also
        # consolidates rows with literal NULL session_id (which can land
        # via imports or schema migrations). Without the COALESCE these
        # NULL-session rows are stranded -- sleep_all_sessions's GROUP BY
        # collects them as a NULL group, maps to "default" for the loop,
        # then beam.sleep("default") would query session_id = 'default'
        # and miss the NULL rows. See Codex /review note for C9.
        # consolidated_at IS NULL filters out rows already processed by
        # a prior sleep so we don't re-summarize the same originals.
        # E4.a.1: select veracity so the summary can inherit aggregated
        # source-row trust signal (instead of defaulting to 'unknown').
        cursor.execute(f"""
            SELECT id, content, source, timestamp, importance, metadata_json, scope, valid_until, veracity
            FROM working_memory
            WHERE COALESCE(session_id, 'default') = ?
              AND timestamp < ?
              AND consolidated_at IS NULL
            ORDER BY timestamp ASC
            LIMIT {SLEEP_BATCH_SIZE}
        """, (self.session_id, cutoff))
        rows = cursor.fetchall()
        if not rows:
            return {"status": "no_op", "message": "No old working memories to consolidate"}

        # Atomic claim: mark rows consolidated_at BEFORE writing the
        # episodic summary, gated on consolidated_at IS STILL NULL.
        # This serves two roles at once:
        # (1) concurrent sleep() callers -- a second process that also
        #     SELECTed the same rows finds rowcount=0 on its claim and
        #     bails before producing a duplicate summary
        # (2) crash safety -- if the process dies after the claim but
        #     before episodic INSERT, the next sleep cycle finds
        #     consolidated_at set and skips them rather than producing
        #     a duplicate. The flip side is a possible orphan claim
        #     (marker set, no summary) -- acceptable; the originals
        #     remain recallable and a manual "reclaim" can clear
        #     consolidated_at if needed.
        # The dry_run branch skips the claim entirely so it stays
        # side-effect-free.
        if not dry_run:
            now_iso = datetime.now().isoformat()
            ids_to_claim = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids_to_claim))
            cursor.execute(
                f"UPDATE working_memory SET consolidated_at = ? "
                f"WHERE id IN ({placeholders}) AND consolidated_at IS NULL",
                (now_iso, *ids_to_claim),
            )
            claimed_ids = set()
            if cursor.rowcount == len(ids_to_claim):
                # Fast path: we got all of them.
                claimed_ids = set(ids_to_claim)
            else:
                # Slow path: at least one row was claimed concurrently
                # by another sleep. Re-read which ones we actually own
                # so we only summarize those.
                cursor.execute(
                    f"SELECT id FROM working_memory "
                    f"WHERE id IN ({placeholders}) AND consolidated_at = ?",
                    (*ids_to_claim, now_iso),
                )
                claimed_ids = {r["id"] for r in cursor.fetchall()}

            if not claimed_ids:
                # Lost the race entirely.
                self.conn.commit()
                return {"status": "no_op", "message": "All eligible rows claimed by concurrent sleep"}

            # Filter rows to only those we successfully claimed.
            rows = [r for r in rows if r["id"] in claimed_ids]
            self.conn.commit()

        grouped: Dict[str, List[Dict]] = {}
        for row in rows:
            grouped.setdefault(row["source"], []).append(dict(row))

        consolidated_ids = []
        summaries_created = 0
        llm_used_count = 0
        for source, items in grouped.items():
            lines = [item["content"] for item in items]
            ids = [item["id"] for item in items]

            # Aggregate scope: if ANY item is global, the summary is global
            aggregated_scope = "session"
            aggregated_valid_until = None
            for item in items:
                if item.get("scope") == "global":
                    aggregated_scope = "global"
                if item.get("valid_until"):
                    if aggregated_valid_until is None or item["valid_until"] < aggregated_valid_until:
                        aggregated_valid_until = item["valid_until"]

            # E4.a.1: aggregate per-row veracity into the summary's
            # veracity label. Mode of sources with conservative tie-break.
            # Pre-fix the episodic INSERT omitted veracity and the row
            # took 'unknown' (0.8 multiplier) regardless of how confident
            # the sources were.
            aggregated_veracity = aggregate_veracity(
                [item.get("veracity") for item in items]
            )

            # --- Try LLM summarization (chunked to fit context) ---
            summary = None
            llm_succeeded = False
            if local_llm.llm_available():
                # --- Optional pre-compression for small local LLMs ---
                # Uses CompressionPlugin (registered in plugins.py). The env
                # var MNEMOSYNE_USE_CAVEMAN still works as a deprecated
                # fallback — see CompressionPlugin for deprecation warning.
                compression_plugin = _plugins.get_manager().get_plugin("compression")
                if compression_plugin and compression_plugin.enabled:
                    lines = compression_plugin.compress_lines(lines)

                chunks = local_llm.chunk_memories_by_budget(lines, source=source)
                if chunks:
                    if len(chunks) == 1:
                        # All memories fit in one prompt
                        summary = local_llm.summarize_memories(chunks[0], source=source)
                    else:
                        # Multi-chunk: summarize each chunk, then summarize the summaries
                        chunk_summaries = []
                        for chunk in chunks:
                            chunk_summary = local_llm.summarize_memories(chunk, source=source)
                            if chunk_summary:
                                chunk_summaries.append(chunk_summary)
                        if chunk_summaries:
                            # Second-pass: summarize the chunk summaries
                            if len(chunk_summaries) == 1:
                                summary = chunk_summaries[0]
                            else:
                                summary = local_llm.summarize_memories(
                                    chunk_summaries,
                                    source=f"{source} (consolidated)"
                                )
                                # If second-pass also overflows, concatenate
                                if not summary:
                                    summary = " | ".join(chunk_summaries)
                    if summary:
                        llm_used_count += 1
                        llm_succeeded = True

            # --- Fallback to aaak encoding ---
            if summary is None:
                logger.warning(
                    "sleep: LLM summarization failed for source=%r (items=%d, "
                    "llm_available=%s) — falling back to AAAK compression",
                    source, len(items), local_llm.llm_available(),
                )
                combined = " | ".join(lines)
                compressed = aaak_encode(combined)
                summary = f"[{source}] {compressed}"

            if not dry_run:
                # Originals are already claimed (consolidated_at set above).
                # Just write the summary. If consolidate_to_episodic raises
                # the claim survives -- the rows show as consolidated but
                # without a summary. That's preferable to a phantom-summary-
                # without-claim race the previous ordering allowed.
                self.consolidate_to_episodic(
                    summary=summary,
                    source_wm_ids=ids,
                    source="sleep_consolidation",
                    importance=0.6,
                    scope=aggregated_scope,
                    valid_until=aggregated_valid_until,
                    veracity=aggregated_veracity,
                    metadata={
                        "original_count": len(items),
                        "source": source,
                        "llm_used": llm_succeeded
                    }
                )
            consolidated_ids.extend(ids)
            summaries_created += 1

        method = "llm" if llm_used_count == summaries_created else ("llm+aaak" if llm_used_count > 0 else "aaak")
        if not dry_run:
            cursor.execute("""
                INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview)
                VALUES (?, ?, ?)
            """, (self.session_id, len(consolidated_ids), f"{summaries_created} summaries ({method}) from {len(consolidated_ids)} items"))
            self.conn.commit()

        # Run tiered degradation after consolidation
        degrade_result = self.degrade_episodic(dry_run=dry_run)

        return {
            "status": "dry_run" if dry_run else "consolidated",
            "items_consolidated": len(consolidated_ids),
            "summaries_created": summaries_created,
            "llm_used": llm_used_count,
            "method": method,
            "consolidated_ids": consolidated_ids,
            "degradation": degrade_result
        }

    def sleep_all_sessions(self, dry_run: bool = False) -> Dict:
        """
        Consolidate eligible old working memories across all sessions.

        This is the maintenance-oriented counterpart to sleep(), which remains
        scoped to self.session_id. It prevents inactive sessions from leaving
        old working_memory rows stranded after they pass the sleep cutoff.
        """
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)).isoformat()
        # Mirror sleep()'s filter: only count rows that haven't been
        # consolidated yet, so we don't redo work on every maintenance pass.
        cursor.execute("""
            SELECT session_id, COUNT(*) AS eligible
            FROM working_memory
            WHERE timestamp < ?
              AND consolidated_at IS NULL
            GROUP BY session_id
            ORDER BY MIN(timestamp) ASC
        """, (cutoff,))
        session_rows = cursor.fetchall()
        if not session_rows:
            return {
                "status": "no_op",
                "message": "No old working memories to consolidate",
                "sessions_scanned": 0,
                "sessions_consolidated": 0,
                "items_consolidated": 0,
                "summaries_created": 0,
                "llm_used": 0,
                "errors": 0,
                "session_results": [],
            }

        session_results = []
        sessions_consolidated = 0
        items_consolidated = 0
        summaries_created = 0
        llm_used = 0
        errors = []

        for row in session_rows:
            session_id = row["session_id"] if hasattr(row, "keys") else row[0]
            if session_id is None:
                session_id = "default"
            try:
                # Pass author_id/author_type so the alien-session BeamMemory
                # tags consolidated episodic rows with the caller's authorship
                # (e.g. a maintenance bot can audit-recall its own work).
                #
                # channel_id is intentionally NOT propagated. BeamMemory.__init__
                # defaults channel_id to its own session_id when None -- passing
                # self.channel_id (which may itself be the caller's defaulted
                # session_id) would tag alien rows with the caller's channel,
                # creating cross-session pollution where filter by
                # channel_id=caller surfaces alien content. Letting it default
                # to the alien session_id is the semantically correct behavior.
                # See C9 + adversarial review in the memory-contract ledger.
                beam = self if session_id == self.session_id else BeamMemory(
                    session_id=session_id,
                    db_path=self.db_path,
                    author_id=self.author_id,
                    author_type=self.author_type,
                )
                result = beam.sleep(dry_run=dry_run)
                result = dict(result)
                result["session_id"] = session_id
                result["eligible"] = row["eligible"] if hasattr(row, "keys") else row[1]
                session_results.append(result)

                if result.get("status") in ("consolidated", "dry_run"):
                    sessions_consolidated += 1
                    items_consolidated += int(result.get("items_consolidated", 0) or 0)
                    summaries_created += int(result.get("summaries_created", 0) or 0)
                    llm_used += int(result.get("llm_used", 0) or 0)
            except Exception as exc:
                logger.error(
                    "sleep_all_sessions: session %r consolidation failed: %s",
                    session_id, exc, exc_info=True,
                )
                errors.append({"session_id": session_id, "error": repr(exc)})

        # Run tiered degradation after all-sessions consolidation
        degrade_result = self.degrade_episodic(dry_run=dry_run)

        return {
            "status": "dry_run" if dry_run else ("consolidated" if items_consolidated else "no_op"),
            "sessions_scanned": len(session_rows),
            "sessions_consolidated": sessions_consolidated,
            "items_consolidated": items_consolidated,
            "summaries_created": summaries_created,
            "llm_used": llm_used,
            "errors": len(errors),
            "error_details": errors,
            "session_results": session_results,
            "degradation": degrade_result
        }

    def get_consolidation_log(self, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, session_id, items_consolidated, summary_preview, created_at
            FROM consolidation_log
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (self.session_id, limit))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------
    def export_to_dict(self) -> Dict:
        """
        Export all BEAM data to a portable dictionary.
        Includes working_memory, episodic_memory, embeddings, scratchpad,
        and consolidation_log across ALL sessions (not just current).
        """
        cursor = self.conn.cursor()
        export = {
            "mnemosyne_export": {
                "version": "1.0",
                "export_date": datetime.now().isoformat(),
                "source_db": str(self.db_path),
                "component": "beam"
            }
        }

        # Working memory (all sessions). veracity is now part of the
        # row's recall-scoring identity (post-E4 -- the multiplier
        # applies to working_memory hits), so it must survive
        # backup/restore. Without it, restored rows collapse to
        # 'unknown' and lose their per-row trust signal.
        # post-E3: consolidated_at carries the sleep marker; without it
        # on the export, restored DBs would re-summarize every already-
        # slept row on next sleep.
        # NOTE: the recall multiplier at beam.py::recall (the
        # `if r.get("tier") == "working":` block) depends on veracity
        # being in the row dict; do not drop it from this
        # SELECT without updating the multiplier path.
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance,
                   metadata_json, valid_until, superseded_by, scope,
                   recall_count, last_recalled, created_at, veracity, consolidated_at
            FROM working_memory
            ORDER BY session_id, timestamp
        """)
        export["working_memory"] = [dict(row) for row in cursor.fetchall()]

        # Episodic memory (all sessions)
        cursor.execute("""
            SELECT rowid, id, content, source, timestamp, session_id, importance,
                   metadata_json, summary_of, valid_until, superseded_by, scope,
                   recall_count, last_recalled, created_at
            FROM episodic_memory
            ORDER BY session_id, timestamp
        """)
        export["episodic_memory"] = [dict(row) for row in cursor.fetchall()]

        # Episodic embeddings from vec_episodes
        export["episodic_embeddings"] = []
        if _vec_available(self.conn):
            try:
                cursor.execute("SELECT rowid, embedding FROM vec_episodes")
                for row in cursor.fetchall():
                    emb = row["embedding"]
                    if isinstance(emb, bytes):
                        emb = list(emb)
                    elif isinstance(emb, str):
                        try:
                            emb = json.loads(emb)
                        except Exception:
                            pass
                    export["episodic_embeddings"].append({
                        "rowid": row["rowid"],
                        "embedding": emb
                    })
            except Exception:
                pass

        # Scratchpad (all sessions)
        cursor.execute("""
            SELECT id, content, session_id, created_at, updated_at
            FROM scratchpad
            ORDER BY session_id, updated_at
        """)
        export["scratchpad"] = [dict(row) for row in cursor.fetchall()]

        # Consolidation log (all sessions)
        cursor.execute("""
            SELECT id, session_id, items_consolidated, summary_preview, created_at
            FROM consolidation_log
            ORDER BY session_id, created_at
        """)
        export["consolidation_log"] = [dict(row) for row in cursor.fetchall()]

        return export

    def import_from_dict(self, data: Dict, force: bool = False) -> Dict:
        """
        Import BEAM data from a dictionary produced by export_to_dict().
        Idempotent by default: skips records whose id already exists.
        Set force=True to overwrite existing records.
        Returns import statistics.
        """
        stats = {
            "working_memory": {"inserted": 0, "skipped": 0, "overwritten": 0},
            "episodic_memory": {"inserted": 0, "skipped": 0, "overwritten": 0, "embeddings_inserted": 0},
            "scratchpad": {"inserted": 0, "updated": 0},
            "consolidation_log": {"inserted": 0},
        }
        cursor = self.conn.cursor()

        # -- Working memory --
        for item in data.get("working_memory", []):
            mid = item.get("id")
            cursor.execute("SELECT 1 FROM working_memory WHERE id = ?", (mid,))
            exists = cursor.fetchone() is not None
            if exists and not force:
                stats["working_memory"]["skipped"] += 1
                continue
            if exists and force:
                cursor.execute("DELETE FROM working_memory WHERE id = ?", (mid,))
                stats["working_memory"]["overwritten"] += 1
            else:
                stats["working_memory"]["inserted"] += 1
            # veracity preserves the per-row trust signal across
            # backup/restore. Pre-E4 1.0 exports (no key in dict) get
            # NULL, which the recall multiplier handles via the
            # 'unknown' fallback. The clamp at write time means new
            # rows always carry a canonical label.
            cursor.execute("""
                INSERT INTO working_memory
                (id, content, source, timestamp, session_id, importance, metadata_json,
                 valid_until, superseded_by, scope, recall_count, last_recalled, created_at,
                 veracity, consolidated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mid, item.get("content"), item.get("source"), item.get("timestamp"),
                item.get("session_id", "default"), item.get("importance", 0.5),
                item.get("metadata_json", "{}"), item.get("valid_until"),
                item.get("superseded_by"), item.get("scope", "session"),
                item.get("recall_count", 0), item.get("last_recalled"), item.get("created_at"),
                item.get("veracity"),
                # consolidated_at: pre-E3 exports (no key) get NULL --
                # treated as "not yet consolidated" so the next sleep
                # cycle on the importing DB processes them normally.
                item.get("consolidated_at"),
            ))
        self.conn.commit()

        # -- Episodic memory --
        # Capture sqlite-vec availability once before the loop. Reused
        # both for the cascade-cleanup below AND the episodic_embeddings
        # import section further down.
        vec_ok = _vec_available(self.conn)

        old_to_new_rowid = {}
        for item in data.get("episodic_memory", []):
            mid = item.get("id")
            cursor.execute("SELECT rowid FROM episodic_memory WHERE id = ?", (mid,))
            existing = cursor.fetchone()
            if existing and not force:
                stats["episodic_memory"]["skipped"] += 1
                old_to_new_rowid[item.get("rowid")] = existing["rowid"]
                continue
            if existing and force:
                # Cascade-cleanup vec_episodes before deleting the
                # episodic_memory row. sqlite-vec's `vec_episodes` is
                # a virtual table keyed by `episodic_memory.rowid`;
                # without this DELETE the row vanishes from
                # episodic_memory but its vector embedding stays in
                # vec_episodes forever, pointing at a rowid that
                # episodic_memory's AUTOINCREMENT will never re-issue.
                # The INSERT below assigns a new rowid via lastrowid;
                # the orphan from the deleted row would never be
                # cleaned by natural reuse. Operators with high
                # import churn would see vec_episodes grow indefinitely
                # while episodic_memory stays bounded.
                # /review (E2.a.5 Codex adversarial L6, deferred sibling
                # cleanup item) flagged this as the canonical orphan
                # site.
                if vec_ok:
                    try:
                        cursor.execute(
                            "DELETE FROM vec_episodes WHERE rowid = ?",
                            (existing["rowid"],),
                        )
                    except sqlite3.Error as cleanup_exc:
                        # Broad sqlite3.Error catch (covers
                        # OperationalError, DatabaseError,
                        # NotSupportedError, etc.) -- `working_memory`
                        # was already committed at line 3978, so
                        # propagating a non-OperationalError mid-loop
                        # would leave partial state. Best-effort
                        # cleanup: log and continue with the
                        # episodic_memory DELETE. Data integrity > orphan
                        # cleanup. /review (Claude H2 + Codex H2 on
                        # commit 1) flagged the narrow OperationalError
                        # catch as a mid-import abort risk.
                        logger.warning(
                            "import_from_dict: vec_episodes cleanup "
                            "failed for rowid=%s: %s; continuing with "
                            "episodic DELETE (orphan may remain)",
                            existing["rowid"], cleanup_exc,
                        )
                cursor.execute("DELETE FROM episodic_memory WHERE id = ?", (mid,))
                stats["episodic_memory"]["overwritten"] += 1
            else:
                stats["episodic_memory"]["inserted"] += 1
            cursor.execute("""
                INSERT INTO episodic_memory
                (id, content, source, timestamp, session_id, importance, metadata_json,
                 summary_of, valid_until, superseded_by, scope, recall_count, last_recalled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mid, item.get("content"), item.get("source"), item.get("timestamp"),
                item.get("session_id", "default"), item.get("importance", 0.5),
                item.get("metadata_json", "{}"), item.get("summary_of", ""),
                item.get("valid_until"), item.get("superseded_by"),
                item.get("scope", "session"), item.get("recall_count", 0),
                item.get("last_recalled"), item.get("created_at")
            ))
            new_rowid = cursor.lastrowid
            old_to_new_rowid[item.get("rowid")] = new_rowid
        self.conn.commit()

        # -- Episodic embeddings --
        # vec_ok was set above before the episodic_memory loop so the
        # cascade-cleanup of vec_episodes shares the same availability
        # check. Reused here.
        for emb_item in data.get("episodic_embeddings", []):
            old_rowid = emb_item.get("rowid")
            new_rowid = old_to_new_rowid.get(old_rowid)
            if not new_rowid:
                continue
            embedding = emb_item.get("embedding")
            if not embedding:
                continue
            if vec_ok:
                try:
                    _vec_insert(self.conn, new_rowid, embedding)
                    stats["episodic_memory"]["embeddings_inserted"] += 1
                except Exception:
                    pass
        if vec_ok:
            self.conn.commit()

        # -- Scratchpad --
        for item in data.get("scratchpad", []):
            pid = item.get("id")
            cursor.execute("SELECT 1 FROM scratchpad WHERE id = ?", (pid,))
            exists = cursor.fetchone() is not None
            if exists:
                cursor.execute("""
                    UPDATE scratchpad SET content=?, session_id=?, created_at=?, updated_at=?
                    WHERE id=?
                """, (item.get("content"), item.get("session_id", "default"),
                      item.get("created_at"), item.get("updated_at"), pid))
                stats["scratchpad"]["updated"] += 1
            else:
                cursor.execute("""
                    INSERT INTO scratchpad (id, content, session_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (pid, item.get("content"), item.get("session_id", "default"),
                      item.get("created_at"), item.get("updated_at")))
                stats["scratchpad"]["inserted"] += 1
        self.conn.commit()

        # -- Consolidation log --
        for item in data.get("consolidation_log", []):
            cursor.execute("""
                INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview, created_at)
                VALUES (?, ?, ?, ?)
            """, (item.get("session_id", "default"), item.get("items_consolidated", 0),
                  item.get("summary_preview", ""), item.get("created_at")))
            stats["consolidation_log"]["inserted"] += 1
        self.conn.commit()

        return stats
