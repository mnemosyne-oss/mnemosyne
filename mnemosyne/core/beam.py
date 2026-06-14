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
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Set, Union, Tuple
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

# Enhanced recall modules (gated by MNEMOSYNE_ENHANCED_RECALL=1)
try:
    from mnemosyne.core.weibull import weibull_boost
except ImportError:
    weibull_boost = None
try:
    from mnemosyne.core.query_intent import classify_intent, adjust_weights
except ImportError:
    classify_intent = None
    adjust_weights = None
try:
    from mnemosyne.core.mmr import mmr_rerank
except ImportError:
    mmr_rerank = None
try:
    from mnemosyne.core.synonyms import expand_query, normalize_query
except ImportError:
    expand_query = None
    normalize_query = None
try:
    from mnemosyne.core.query_cache import QueryCache
except ImportError:
    QueryCache = None
try:
    from mnemosyne.core.temporal_parser import extract_temporal, parse_nl_date
except ImportError:
    extract_temporal = None
    parse_nl_date = None

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

import os
import re

# On Fly.io and other ephemeral VMs, only ~/.hermes is persisted.
# Default to the legacy Hermes path so memories survive restarts.
_DEFAULT_ROOT = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
DEFAULT_DATA_DIR = _DEFAULT_ROOT / "mnemosyne" / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"

_thread_local = threading.local()

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
# Priority: 1) MNEMOSYNE_EMBEDDING_DIM env var (explicit override)
#           2) Auto-derive from embedding model via _embeddings module
#           3) 384 (bge-small-en-v1.5 default)
_emb_dim_env = os.environ.get("MNEMOSYNE_EMBEDDING_DIM")
if _emb_dim_env is not None:
    try:
        EMBEDDING_DIM = int(_emb_dim_env)
    except (ValueError, TypeError):
        EMBEDDING_DIM = 384
else:
    EMBEDDING_DIM = _embeddings.EMBEDDING_DIM
WORKING_MEMORY_MAX_ITEMS = int(os.environ.get("MNEMOSYNE_WM_MAX_ITEMS", "10000"))
WORKING_MEMORY_TTL_HOURS = int(os.environ.get("MNEMOSYNE_WM_TTL_HOURS", "168"))
WM_BUMP_CAP_HOURS = int(os.environ.get("MNEMOSYNE_WM_BUMP_CAP_HOURS", "24"))
WM_PINNED_IDS = set(
    pid.strip() for pid in os.environ.get("MNEMOSYNE_WM_PINNED_IDS", "").split(",")
    if pid.strip()
)
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
    needs_reconnect = (
        not hasattr(_thread_local, 'conn')
        or _thread_local.conn is None
        or getattr(_thread_local, 'db_path', None) != str(path)
    )
    if not needs_reconnect:
        # Verify the cached connection is still alive
        try:
            _thread_local.conn.execute("SELECT 1")
        except Exception:
            needs_reconnect = True

    if needs_reconnect:
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

    Keep the probe rollback scoped to the temporary vec test table. This
    function runs during schema initialization, before init_beam() commits its
    CREATE TABLE statements; a connection-wide rollback here can otherwise
    undo unrelated schema DDL such as scratchpad.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return "float32"
    if VEC_TYPE == "float32":
        return "float32"

    cursor = conn.cursor()

    def _probe(candidate: str) -> bool:
        cursor.execute("SAVEPOINT vec_type_probe")
        try:
            cursor.execute(
                f"CREATE VIRTUAL TABLE _vec_test USING vec0(embedding {candidate}[{EMBEDDING_DIM}])"
            )
            cursor.execute("DROP TABLE IF EXISTS _vec_test")
            cursor.execute("RELEASE SAVEPOINT vec_type_probe")
            return True
        except Exception:
            cursor.execute("ROLLBACK TO SAVEPOINT vec_type_probe")
            cursor.execute("RELEASE SAVEPOINT vec_type_probe")
            return False

    test_type = VEC_TYPE  # int8 or bit
    if _probe(test_type):
        return test_type
    if test_type == "bit" and _probe("int8"):
        return "int8"
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

    try:
        cursor.execute("ALTER TABLE working_memory ADD COLUMN consolidation_claimed_at TEXT")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise

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
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_wm_consolidation_claims "
        "ON working_memory(consolidation_claimed_at) "
        "WHERE consolidation_claimed_at IS NOT NULL"
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

    # --- MEMORIA: Structured Fact Tables (Phase 1) ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memoria_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT DEFAULT 'default',
            message_idx INTEGER,
            fact_type TEXT,
            key TEXT,
            value TEXT,
            context_snippet TEXT,
            importance REAL DEFAULT 0.5,
            timestamp TEXT,
            version_id INTEGER DEFAULT 0,
            previous_value TEXT,
            updated_msg_idx INTEGER,
            valid_from_msg_idx INTEGER,
            valid_to_msg_idx INTEGER,
            source_memory_id TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON memoria_facts(key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_type ON memoria_facts(fact_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON memoria_facts(session_id)")
    # Migration: add versioning columns to existing tables (safe re-run)
    for col in ['version_id', 'previous_value', 'updated_msg_idx', 'valid_from_msg_idx', 'valid_to_msg_idx', 'source_memory_id']:
        _add_column_if_missing(conn, "memoria_facts", col, {
            'version_id': 'INTEGER DEFAULT 0',
            'previous_value': 'TEXT',
            'updated_msg_idx': 'INTEGER',
            'valid_from_msg_idx': 'INTEGER',
            'valid_to_msg_idx': 'INTEGER',
            'source_memory_id': 'TEXT',
        }.get(col, 'TEXT'))

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memoria_timelines (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT DEFAULT 'default',
            date TEXT,
            message_idx INTEGER,
            description TEXT,
            source TEXT,
            source_memory_id TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timelines_date ON memoria_timelines(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timelines_session ON memoria_timelines(session_id)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memoria_instructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT DEFAULT 'default',
            message_idx INTEGER,
            instruction TEXT,
            active INTEGER DEFAULT 1,
            topic TEXT,
            context_snippet TEXT,
            source_memory_id TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_instr_session ON memoria_instructions(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_instr_active ON memoria_instructions(active)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memoria_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT DEFAULT 'default',
            message_idx INTEGER,
            preference TEXT,
            topic TEXT,
            evolution TEXT,
            context_snippet TEXT,
            source_memory_id TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pref_session ON memoria_preferences(session_id)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memoria_kg (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT DEFAULT 'default',
            subject TEXT,
            predicate TEXT,
            object TEXT,
            message_idx INTEGER,
            confidence REAL DEFAULT 0.7,
            source_memory_id TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_subject ON memoria_kg(subject)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_predicate ON memoria_kg(predicate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_session ON memoria_kg(session_id)")
    for table in ('memoria_timelines', 'memoria_instructions', 'memoria_preferences', 'memoria_kg'):
        _add_column_if_missing(conn, table, 'source_memory_id', 'TEXT')

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

    # --- Migration: pinned flag for usage-driven decay (v3.0) ---
    _add_column_if_missing(conn, "working_memory", "pinned", "INTEGER DEFAULT 0")

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
    cursor.execute("""CREATE INDEX IF NOT EXISTS idx_wm_context_session
        ON working_memory(session_id, importance DESC, timestamp DESC)
        WHERE superseded_by IS NULL""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS idx_wm_context_global
        ON working_memory(scope, importance DESC, timestamp DESC)
        WHERE superseded_by IS NULL""")
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

    # --- Migration: collaborative attestation (v2.7) ---
    # author_id captures the original writer; validator/validated_at capture
    # the most recent agent that attested or updated the memory. Together they
    # let multiple agents collaborate on shared facts without losing origin.
    _add_column_if_missing(conn, "working_memory", "validator", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "validated_at", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "validation_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "episodic_memory", "validator", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "validated_at", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "validation_count", "INTEGER DEFAULT 0")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_validator ON working_memory(validator)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_validated_at ON working_memory(validated_at)")

    # Ring-buffer log capped at 3 entries per memory (lightweight history).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_validations (
            validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            validator TEXT NOT NULL,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action TEXT NOT NULL,
            new_content TEXT,
            note TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_validations_memory ON memory_validations(memory_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_validations_validator ON memory_validations(validator)")
    # Trim each memory's validation history to the most recent 3 entries.
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS trim_validations_to_3
        AFTER INSERT ON memory_validations
        BEGIN
            DELETE FROM memory_validations
            WHERE memory_id = NEW.memory_id
              AND validation_id NOT IN (
                  SELECT validation_id FROM memory_validations
                  WHERE memory_id = NEW.memory_id
                  ORDER BY validation_id DESC
                  LIMIT 3
              );
        END
    """)

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

    # --- Temporal architecture migration ---
    _add_column_if_missing(conn, "working_memory", "event_date", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "event_date_precision", "TEXT DEFAULT 'unknown'")
    _add_column_if_missing(conn, "working_memory", "temporal_tags", "TEXT DEFAULT '[]'")
    _add_column_if_missing(conn, "working_memory", "corrected_by", "INTEGER DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "event_date", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "event_date_precision", "TEXT DEFAULT 'unknown'")
    _add_column_if_missing(conn, "episodic_memory", "temporal_tags", "TEXT DEFAULT '[]'")
    _add_column_if_missing(conn, "episodic_memory", "corrected_by", "INTEGER DEFAULT NULL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_event_date ON working_memory(event_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_event_date ON episodic_memory(event_date)")


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

def _sanitize_utf8(text: str) -> str:
    """Ensure text is valid UTF-8, stripping or replacing invalid bytes.

    SQLite TEXT columns enforce UTF-8 encoding. Corrupt bytes (e.g. 0xFE,
    0xFF, or truncated multi-byte sequences from LLM output / AAAK
    compression / buffer errors) cause OperationalError on subsequent reads.
    This function sanitizes content at the write boundary so a single bad
    write doesn't poison episodic_memory and crash future maintenance passes
    (sleep_all_sessions, degrade_episodic).
    """
    if not isinstance(text, str):
        return ""
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        # Replace invalid bytes with the Unicode replacement character.
        # This preserves as much content as possible while guaranteeing
        # the DB column stays valid UTF-8.
        return text.encode('utf-8', errors='replace').decode('utf-8')


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


def _normalize_datetime_utc(dt: datetime) -> datetime:
    """Return dt as a timezone-aware UTC datetime.

    Naive datetimes are treated as UTC to preserve existing naive timestamp
    behavior while avoiding naive/aware comparison crashes.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_datetime_utc(value: str) -> datetime:
    """Parse an ISO datetime string and normalize it to UTC."""
    return _normalize_datetime_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _recency_decay(timestamp_str: str, halflife_hours: float = RECENCY_HALFLIFE_HOURS) -> float:
    """Calculate recency decay factor. 1.0 = brand new, ~0.5 = one halflife old.
    
    Exponential decay based on age. Returns 0.5 for unknown/invalid timestamps.
    """
    if not timestamp_str:
        return 0.5  # Unknown age = neutral
    try:
        ts = _parse_iso_datetime_utc(timestamp_str)
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return math.exp(-age_hours / halflife_hours)
    except Exception:
        return 0.5


def _parse_query_time(query_time: Optional[Union[str, datetime]]) -> datetime:
    """Parse query_time parameter into a timezone-aware UTC datetime object.

    - None -> current UTC time
    - str  -> parsed from ISO format and normalized to UTC
    - datetime -> normalized to UTC
    Naive values are treated as UTC for backward compatibility.
    """
    if query_time is None:
        return datetime.now(timezone.utc)
    if isinstance(query_time, datetime):
        return _normalize_datetime_utc(query_time)
    if isinstance(query_time, str):
        # Try ISO format with various precisions
        try:
            return _parse_iso_datetime_utc(query_time)
        except ValueError:
            # Try appending time if only date provided
            try:
                return _parse_iso_datetime_utc(f"{query_time}T00:00:00")
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
        dt = _parse_iso_datetime_utc(ts)
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
    query_time = _normalize_datetime_utc(query_time)

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


_FACT_MATCH_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "related",
    "should", "that", "the", "their", "there", "this", "to", "totally", "unrelated",
    "use", "uses", "was", "we", "what", "when", "where", "which", "who", "why",
    "with", "you", "your",
    # Conversational glue/pronouns. These caused broad requests like "do not reply
    # to them" to fact-match unrelated policies via overlaps such as "not/they".
    "again", "into", "not", "please", "somewhere", "supposed", "them", "then",
    "they", "whatever",
}

# Unicode-aware recall tokenization.
#
# The previous ASCII-only pattern split words with diacritics, e.g.
# "Stoßlüften" became "sto" + "ften" and "Bürgeramt" became
# "b" + "rgeramt".  That weakened lexical gates and fallback scoring for
# German and other Latin-script languages while dense multilingual embeddings
# still saw the original text.  Keep path/version separators inside tokens
# only when followed by another Unicode alphanumeric char so trailing
# punctuation such as "Bürgeramt:" is not absorbed.
_RECALL_TOKEN_RE = re.compile(r"(?u)[^\W_][\w]*(?:[_.:/+-]+[^\W_][\w]*)*")

_RECALL_SYNONYMS: Dict[str, tuple[str, ...]] = {
    # Small, conservative query expansion for common user-facing wording.
    # These terms fix multi-fact questions whose stored facts use the
    # natural wording of a preference/policy rather than the abstract word
    # in the question (e.g. "branding preference" -> "positioning/wants").
    "branding": ("brand", "positioning", "identity", "wording"),
    "preference": ("prefer", "prefers", "want", "wants", "reject", "rejects", "avoid", "grounded"),
    "professional": ("software", "builder"),
    "url": ("link", "profile"),
    "current": ("now", "live", "latest"),
    "feeling": ("feel", "feels"),
    "imposter": ("self-doubt", "doubt", "insecure"),
}


def _recall_tokens(text: str) -> List[str]:
    """Meaningful lexical tokens for precision gates and fallback scoring."""
    return [
        token
        for token in _RECALL_TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token not in _FACT_MATCH_STOPWORDS and not token.isdigit()
    ]


def _expanded_query_tokens(tokens: List[str]) -> List[str]:
    """Return query tokens plus a bounded synonym expansion.

    Expansion is query-side only and de-duplicated in order. It broadens FTS
    candidate generation without lowering the lexical abstention gate.
    """
    expanded: List[str] = []
    seen: Set[str] = set()
    for token in tokens:
        for candidate in (token, *_RECALL_SYNONYMS.get(token, ())):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _minimum_recall_relevance(query_tokens: List[str]) -> float:
    """Raise the lexical gate for broad natural-language queries.

    One matching real word is enough for short lookup-style queries, but not
    for broad nonsense strings like "purple bicycle quantum oatmeal".
    """
    if len(query_tokens) >= 4:
        return 0.3
    if len(query_tokens) == 3:
        return 0.5
    return 0.15


def _fact_match_tokens(text: str) -> Set[str]:
    """Return meaningful tokens for strict fact matching."""
    return set(_recall_tokens(text))


def _contains_spaceless_cjk(text: str) -> bool:
    return any(
        "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
        for ch in text
    )


def _cjk_fts_terms(text: str) -> List[str]:
    """Generate FTS-safe terms for CJK text.

    The default unicode61 tokenizer indexes each CJK character as an
    individual token. Unquoted character terms match directly. Bigrams
    are quoted as phrases for multi-character matching.
    """
    cjk_chars = [
        ch for ch in text
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
    ]
    if not cjk_chars:
        return []
    terms: List[str] = []
    seen: Set[str] = set()
    for ch in cjk_chars:
        if ch not in seen:
            seen.add(ch)
            terms.append(ch)  # bare unigram matches individual token
    for i in range(len(cjk_chars) - 1):
        bigram = cjk_chars[i] + cjk_chars[i + 1]
        if bigram not in seen:
            seen.add(bigram)
            terms.append(f'"{bigram}"')  # quoted bigram phrase match
    return terms


def _lexical_relevance(query_tokens: List[str], content: str, query_lower: str = "") -> float:
    """Conservative lexical score in [0, 1]. Returns 0 for no real token overlap.

    This replaces the old character-overlap fallback for normal spaced text.
    Character overlap is only useful for CJK/spaceless text; in English it made
    nonsense queries retrieve unrelated high-importance memories.
    """
    content_lower = content.lower()
    query_cjk = {
        ch for ch in query_lower
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
    }
    if not query_tokens and not query_cjk:
        return 0.0
    content_tokens = set(_recall_tokens(content_lower))
    # Structured MEMORIA contexts often encode keys as snake_case
    # (telemetry_api_latency_ms). Split separators so natural-language
    # queries get full lexical credit for the same fact.
    expanded_content_tokens = set(content_tokens)
    for token in list(content_tokens):
        expanded_content_tokens.update(
            part for part in re.split(r"[_:/.-]+", token)
            if len(part) >= 3 and part not in _FACT_MATCH_STOPWORDS and not part.isdigit()
        )
    content_tokens = expanded_content_tokens
    if not content_tokens and not query_cjk:
        return 0.0

    exact = sum(1 for token in query_tokens if token in content_tokens)
    partial = 0.0
    for token in query_tokens:
        if token in content_tokens:
            continue
        synonyms = _RECALL_SYNONYMS.get(token, ())
        if synonyms and any(syn in content_tokens for syn in synonyms):
            partial += 0.75
            continue
        if len(token) >= 4 and any(
            token in ctoken or ctoken in token
            for ctoken in content_tokens
            if len(ctoken) >= 4
        ):
            partial += 0.4

    full_match = 1.0 if query_lower and query_lower in content_lower else 0.0
    score = (exact + partial + full_match) / max(len(query_tokens), 1)

    if score == 0.0:
        query_cjk = {
            ch for ch in query_lower
            if "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        }
        if query_cjk:
            content_cjk = {
                ch for ch in content_lower
                if "\u4e00" <= ch <= "\u9fff"
                or "\u3040" <= ch <= "\u30ff"
                or "\uac00" <= ch <= "\ud7af"
            }
            score = len(query_cjk & content_cjk) / len(query_cjk)

    return min(score, 1.0)


def _strict_fact_matches(query: str, fact_text: str) -> bool:
    """Conservative fact matching for natural-language recall queries.

    The legacy fact matcher accepts any query token as a substring of the
    fact. That makes stopwords like "where"/"the"/"use" retrieve unrelated
    facts. The strict matcher keeps exact phrase/path/domain matches, then
    requires multiple meaningful token overlaps (or one very distinctive
    path/domain-like token) before admitting a fact candidate.
    """
    query_lower = query.lower().strip()
    fact_lower = fact_text.lower().strip()
    if not query_lower or not fact_lower:
        return False

    if query_lower in fact_lower:
        return True

    query_tokens = _fact_match_tokens(query_lower)
    fact_tokens = _fact_match_tokens(fact_lower)
    if not query_tokens or not fact_tokens:
        return False

    overlap = query_tokens & fact_tokens
    if len(overlap) >= 2:
        return True

    # Allow a single highly distinctive exact token, but not arbitrary words.
    if len(overlap) == 1:
        token = next(iter(overlap))
        # Long structured tokens (paths/domains/identifiers) are very distinctive
        # even inside a longer query.
        if len(token) >= 8 and any(c in token for c in (".", "/", ":", "-", "_")):
            return True
        # Medium tokens (5+ chars) are useful for direct lookup-style fact queries
        # like "hermes" or "python". They are NOT enough for broad natural-language
        # queries: one shared word such as "public", "context", or "reply" should
        # not cause an unrelated high-importance fact to enter recall.
        if len(query_tokens) <= 2:
            return len(token) >= 5
        return False

    return False


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
        strict_fact_match = not _env_truthy("MNEMOSYNE_LENIENT_FACT_MATCH")

        # Simple keyword matching against fact text
        memory_ids: Set[str] = set()
        for fact_row in all_facts:
            fact_text = fact_row.get("value", "").lower()
            if strict_fact_match:
                if _strict_fact_matches(query_lower, fact_text):
                    memory_ids.add(fact_row["memory_id"])
            # Check if any query word appears in the fact
            elif any(word in fact_text for word in query_words):
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
        logger.info("Regex extraction failed, skipping", exc_info=True)
    return "float32"


def _vec_insert(conn: sqlite3.Connection, rowid: int, embedding: List[float]):
    """Insert embedding into sqlite-vec table with quantization via SQL functions.

    Manually normalizes the embedding to unit length before quantization.
    ``vec_quantize_int8(x, 'unit')`` in sqlite-vec 0.1.9 fails to normalize
    1024-dim vectors (works for 3-dim, silently passes through unnormalized
    vectors at high dimensions).  Pre-normalizing works around the bug until
    upstream sqlite-vec ships a fix.
    """
    vec_type = _effective_vec_type(conn)
    # Normalize to unit length before quantization
    # (sqlite-vec 0.1.9 'unit' param fails at 1024-dim)
    import numpy as _np
    emb_arr = _np.array(embedding, dtype=_np.float32)
    norm = _np.linalg.norm(emb_arr)
    if norm > 0:
        emb_arr = emb_arr / norm
    emb_json = json.dumps(emb_arr.tolist())
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
    # Ensure the insert is committed even when the caller's connection
    # has _defer_commit=True (_BeamConnection). Without this, inserts
    # sit in the deferred transaction and disappear if the caller
    # later rolls back or the connection is reused in a different context.
    if isinstance(conn, _BeamConnection):
        conn._real_commit()
    else:
        conn.commit()


def _vec_search(conn: sqlite3.Connection, embedding: List[float], k: int = 20) -> List[Dict]:
    """Search sqlite-vec and return rowids with distances.

    Normalizes the query embedding to unit length before quantization so
    distances are commensurate with the stored int8 vectors (which are also
    unit-normalized at insert time — see _vec_insert).
    """
    vec_type = _effective_vec_type(conn)
    # Normalize to unit length before quantization
    # (sqlite-vec 0.1.9 'unit' param fails at 1024-dim)
    import numpy as _np
    emb_arr = _np.array(embedding, dtype=_np.float32)
    norm = _np.linalg.norm(emb_arr)
    if norm > 0:
        emb_arr = emb_arr / norm
    emb_json = json.dumps(emb_arr.tolist())
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
            f'SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH vec_quantize_int8(?, "unit") AND k={k} ORDER BY distance',
            (emb_json,)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH ? ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    return [{"rowid": r["rowid"], "distance": r["distance"]} for r in rows]


def _fts_query_terms(query: str) -> List[str]:
    """FTS-safe meaningful terms for natural-language recall queries."""
    terms = []
    for term in _expanded_query_tokens(_recall_tokens(query)):
        term = term.replace('"', '""').strip()
        if term:
            terms.append(f'"{term}"')
    return terms


def _has_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    return any(
        "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
        for ch in text
    )


def _cjk_like_search(conn: sqlite3.Connection, query: str, k: int = 20, working: bool = False) -> List[Dict]:
    """Fallback LIKE search for CJK text.

    The default unicode61 FTS5 tokenizer does not index CJK characters
    on this SQLite build. When a CJK query produces zero FTS results,
    fall back to scanning content via LIKE for each unique CJK character
    in the query. Rows matching more query characters rank higher.

    This is slower than FTS but correctness matters more than speed for
    underserved CJK users. Once FTS5 gets tokenchars or ICU support,
    this function can be removed.
    """
    cjk_chars = sorted(set(
        ch for ch in query
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
    ))
    if not cjk_chars:
        return []

    if working:
        table = "working_memory"
        id_col = "id"
    else:
        table = "episodic_memory"
        id_col = "rowid"

    # Build parameterized LIKE clauses for each CJK character
    conditions = " OR ".join(f"content LIKE ? ESCAPE '\\'" for _ in cjk_chars)
    params = [f"%{ch}%" for ch in cjk_chars]
    # Also search for mixed CJK+ASCII: any of the CJK chars must be present
    try:
        all_rows = conn.execute(
            f"SELECT {id_col}, content FROM {table} WHERE {conditions} LIMIT ?",
            params + [k * 5]
        ).fetchall()
    except Exception:
        return []

    if not all_rows:
        return []

    # Score: count how many unique CJK chars from the query appear in each row
    scored = []
    for row in all_rows:
        rid = row[id_col]
        content = row["content"]
        score = sum(1 for ch in cjk_chars if ch in content) / max(len(cjk_chars), 1)
        if score > 0:
            scored.append((rid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    result_key = "id" if working else "rowid"
    return [{result_key: rid, "rank": -score} for rid, score in scored[:k]]


# ---------------------------------------------------------------------------
# Cyrillic FTS5 fallback (Russian)
#
# Russian is morphologically rich: a single lemma produces many inflected
# surface forms (тёмная / тёмную / тёмной, встреча / встречи / встречу).
# The default unicode61 FTS5 tokenizer has no built-in stemmer for Russian,
# so direct FTS5 queries match only the exact surface form.
#
# When a Russian query produces zero FTS5 results, we fall back to a
# LIKE-based scan and rank candidates by trigram Jaccard similarity
# between query and content words. This mirrors _cjk_like_search
# (issue #168) but with n-gram scoring, which is significantly more
# discriminative for inflected languages than the single-character
# coverage used for CJK.
#
# NOTE: the regex [а-яёА-ЯЁ] covers Russian and some Cyrillic neighbours
# (Ukrainian, Bulgarian, etc.) but does NOT include Serbian/Mongolian
# Cyrillic characters (ђ ћ џ њ љ ө ү). These can be added later if needed.
#
# No new dependencies, no schema migration, no FTS5 changes. The fallback
# activates only when FTS5 returns zero rows AND the query contains
# Cyrillic characters.
#
# Priority: if a query contains both CJK and Cyrillic, the CJK fallback
# takes precedence (checked first). This is unlikely in practice but
# documented here for clarity.
# ---------------------------------------------------------------------------

_CYRILLIC_CHAR_RE = re.compile(r"[а-яёА-ЯЁ]")


def _has_cyrillic(text: str) -> bool:
    """True if text contains at least one Russian/Cyrillic character.

    Matches [а-яёА-ЯЁ] which covers Russian and some neighbours but
    not Serbian/Mongolian Cyrillic — see module-level note.
    """
    return bool(_CYRILLIC_CHAR_RE.search(text))


def _ngrams(s: str, n: int = 3) -> set:
    """Return the set of length-n sliding n-grams of s.

    For strings shorter than n, returns the whole string as a single
    n-gram to keep the Jaccard math well-defined.
    """
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _cyrillic_score(query: str, content: str, n: int = 3) -> float:
    """Trigram Jaccard score in [0, 1] for Russian/Cyrillic recall.

    For each query word, find the best-matching content word by Jaccard
    similarity of their character n-gram sets. Return the mean across
    query words. Words shorter than 3 characters are ignored to avoid
    noisy matches on stop words and punctuation.

    N-gram sets are computed once per unique word and cached in a dict
    to avoid redundant recomputation when the same content word is
    compared against multiple query words.
    """
    q_words = [
        w for w in re.findall(r"[а-яёa-z0-9]+", query.lower()) if len(w) >= 3
    ]
    c_words = [
        w for w in re.findall(r"[а-яёa-z0-9]+", content.lower()) if len(w) >= 3
    ]
    if not q_words or not c_words:
        return 0.0
    # Cache n-gram sets to avoid recomputing the same word multiple times.
    ng_cache: dict = {}
    total = 0.0
    for qw in q_words:
        if qw not in ng_cache:
            ng_cache[qw] = _ngrams(qw, n)
        q_ng = ng_cache[qw]
        best = 0.0
        for cw in c_words:
            if cw not in ng_cache:
                ng_cache[cw] = _ngrams(cw, n)
            c_ng = ng_cache[cw]
            union = q_ng | c_ng
            if not union:
                continue
            jacc = len(q_ng & c_ng) / len(union)
            if jacc > best:
                best = jacc
        total += best
    return total / len(q_words)


def _cyrillic_like_search(
    conn: sqlite3.Connection, query: str, k: int = 20, working: bool = False,
) -> List[Dict]:
    """LIKE-based FTS5 fallback for Russian/Cyrillic text.

    Candidate generation: scan ``working_memory``/``episodic_memory`` for
    rows whose content contains any 4+ character word from the query as
    a substring. Re-rank the candidate set by trigram Jaccard similarity
    so that inflected forms of the same lemma (тёмная/тёмную/тёмной)
    receive comparable scores regardless of which surface form appears
    in the stored text.

    The candidate-generation LIMIT is ``k * 5`` to keep the Python-side
    scoring bounded; this matches the CJK fallback's behaviour.
    """
    if not _has_cyrillic(query):
        return []
    if working:
        table, id_col = "working_memory", "id"
    else:
        table, id_col = "episodic_memory", "rowid"

    # SQLite's default LOWER() and LIKE are ASCII-only, so they cannot
    # case-fold Cyrillic letters (Т ≠ т, Ё ≠ ё). Register a Python
    # UDF on the connection that performs a real Unicode lower(). It
    # is safe to register the same name multiple times — SQLite will
    # overwrite the previous function with the same name and arity.
    conn.create_function("_py_lower", 1, lambda s: s.lower() if isinstance(s, str) else s)

    # Use a 4-character prefix for LIKE matching. Query words and
    # stored content are inflected forms of the same lemma (тёмная
    # vs тёмную), so the 4-char prefix is the most reliable cheap
    # over-selector that still prunes the candidate set down to a
    # trigram-scoring-sized slice.
    q_words = [
        w for w in re.findall(r"[а-яёa-z0-9]+", query.lower()) if len(w) >= 4
    ]
    if not q_words:
        return []

    conditions = " OR ".join(
        ["_py_lower(content) LIKE ? ESCAPE '\\'"] * len(q_words)
    )
    params = [f"%{w[:4].lower()}%" for w in q_words]
    try:
        all_rows = conn.execute(
            f"SELECT {id_col}, content FROM {table} WHERE {conditions} LIMIT ?",
            params + [k * 5],
        ).fetchall()
    except Exception:
        return []

    if not all_rows:
        return []

    scored = []
    for row in all_rows:
        rid = row[id_col]
        content = row["content"]
        score = _cyrillic_score(query, content)
        if score > 0:
            scored.append((rid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    result_key = "id" if working else "rowid"
    return [{result_key: rid, "rank": -score} for rid, score in scored[:k]]


def _fts_search(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Dict]:
    """Search FTS5 episodes and return rowids with ranks.

    Natural assistant queries contain stopwords that make phrase-style FTS
    miss exact memories. Use stopword-filtered OR terms by default; precision
    is restored later by lexical reranking and abstention thresholds.
    """
    terms = _fts_query_terms(query)
    if not terms:
        # CJK text produces zero FTS terms on builds without tokenchars.
        # Fall back to LIKE-based CJK search.
        if _has_cjk(query):
            return _cjk_like_search(conn, query, k=k, working=False)
        # Cyrillic text (Russian) also produces zero FTS terms because
        # the default unicode61 tokenizer has no built-in stemmer for
        # inflected Cyrillic surface forms.
        # NOTE: CJK is checked first — if a query contains both CJK and
        # Cyrillic (unlikely), the CJK fallback takes precedence.
        if _has_cyrillic(query):
            return _cyrillic_like_search(conn, query, k=k, working=False)
        return []
    fts_query = " OR ".join(terms)
    rows = conn.execute(
        "SELECT rowid, rank FROM fts_episodes WHERE fts_episodes MATCH ? ORDER BY rank, rowid LIMIT ?",
        (fts_query, k)
    ).fetchall()
    if not rows and _has_cjk(query):
        return _cjk_like_search(conn, query, k=k, working=False)
    if not rows and _has_cyrillic(query):
        return _cyrillic_like_search(conn, query, k=k, working=False)
    return [{"rowid": r["rowid"], "rank": r["rank"]} for r in rows]


def _fts_search_working(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Dict]:
    """Search FTS5 working memory and return ids with ranks."""
    terms = _fts_query_terms(query)
    if not terms:
        if _has_cjk(query):
            return _cjk_like_search(conn, query, k=k, working=True)
        if _has_cyrillic(query):
            return _cyrillic_like_search(conn, query, k=k, working=True)
        return []
    fts_query = " OR ".join(terms)
    rows = conn.execute(
        "SELECT id, rank FROM fts_working WHERE fts_working MATCH ? ORDER BY rank, id LIMIT ?",
        (fts_query, k)
    ).fetchall()
    if not rows and _has_cjk(query):
        return _cjk_like_search(conn, query, k=k, working=True)
    if not rows and _has_cyrillic(query):
        return _cyrillic_like_search(conn, query, k=k, working=True)
    return [{"id": r["id"], "rank": r["rank"]} for r in rows]


def _wm_vec_search(conn: sqlite3.Connection, query_embedding, k: int = 20,
                   *, where_sql: Optional[str] = None,
                   where_params: Tuple[Any, ...] = ()) -> List[Dict]:
    """Vector search against working_memory via memory_embeddings table.
    Returns list of dicts with 'id' (memory_id) and 'sim' (cosine similarity)."""
    if np is None:
        return []
    cursor = conn.cursor()
    try:
        # BEAM mode: scan up to 500K rows for broad vector recall on large benchmark datasets
        _vec_limit = 500000 if _BEAM_MODE else 50000
        if where_sql is None:
            where_sql = "wm.superseded_by IS NULL AND (wm.valid_until IS NULL OR wm.valid_until > ?)"
            where_params = (datetime.now().isoformat(),)
        cursor.execute(f"""
            SELECT wm.id, me.embedding_json
            FROM memory_embeddings me
            JOIN working_memory wm ON me.memory_id = wm.id
            WHERE {where_sql}
            LIMIT ?
        """, (*where_params, _vec_limit))
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
            logger.info("Regex extraction failed, skipping", exc_info=True)
        self._ensure_e6_schema_with_migration()

        # E6: shared AnnotationStore handle reusing this BeamMemory's
        # thread-local connection. Production call sites use `self.annotations`
        # instead of constructing fresh AnnotationStore(...) per call --
        # eliminates the per-call file-descriptor cost the post-E6 review
        # surfaced (every extraction/recall opened 2 connections + ran DDL).
        from mnemosyne.core.annotations import AnnotationStore
        self.annotations = AnnotationStore(db_path=self.db_path, conn=self.conn)

        # Owner-scoped canonical (single-source-of-truth) facts (issue #256).
        # Shares this BeamMemory's thread-local connection like AnnotationStore.
        # Lazily wired and additive — opening an existing DB just creates the
        # canonical_facts table on first init; nothing else changes.
        from mnemosyne.core.canonical import CanonicalStore
        self.canonical = CanonicalStore(db_path=self.db_path, conn=self.conn)

        # Phase 3: Episodic graph (shared connection)
        self.episodic_graph = None
        if EpisodicGraph is not None:
            try:
                self.episodic_graph = EpisodicGraph(conn=self.conn, db_path=self.db_path)
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)

        # Phase 4: Veracity consolidator (shared connection)
        self.veracity_consolidator = None
        if VeracityConsolidator is not None:
            try:
                self.veracity_consolidator = VeracityConsolidator(conn=self.conn, db_path=self.db_path)
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)

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
                logger.info("Regex extraction failed, skipping", exc_info=True)

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
                    consolidated_at = NULL,
                    consolidation_claimed_at = NULL
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
            # Phase 2: MEMORIA regex-based extraction (always-on, zero-LLM-cost).
            # Populates memoria_facts, memoria_timelines, memoria_kg for the
            # structured retrieval router. Runs silently on every remember()
            # so the MEMORIA tables stay current regardless of extract=True.
            try:
                self.extract_and_store_facts(content, message_idx=0, source_memory_id=existing_id)
            except Exception:
                pass  # regex extraction failures must not block memory storage
            # Phase 3-4: Extract graph and consolidate veracity for dedup update
            self._ingest_graph_and_veracity(existing_id, content, source, veracity)
            self._emit_event("MEMORY_UPDATED", existing_id, content=content,
                             source=source, importance=importance, metadata=metadata)

            # Invalidate enhanced recall cache on memory update
            if hasattr(self, "_query_cache") and self._query_cache is not None:
                self._query_cache.invalidate()

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

        # --- Embedding storage for vector recall ---
        # remember_batch() already does this; remember() was missing it,
        # which meant the Hermes provider (which always calls remember())
        # never populated memory_embeddings. This left _detect_conflicts
        # with zero embeddings to compare, making Phase 1 conflict
        # detection a no-op despite 3762+ working memories.
        if _embeddings.available():
            try:
                vec = _embeddings.embed([content])
                if vec is not None and len(vec) == 1:
                    emb_json = _embeddings.serialize(vec[0])
                    cursor.execute(
                        "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
                        (memory_id, emb_json, _embeddings._DEFAULT_MODEL)
                    )
            except Exception as exc:
                logger.warning(
                    "remember: embedding storage failed for '%s' (%s): %s",
                    memory_id, type(exc).__name__, exc,
                )

        # Auto-generate temporal triple
        self._add_temporal_triple(memory_id, timestamp, source, content)

        # --- Temporal extraction ---
        if extract_temporal is not None:
            try:
                temporal_info = extract_temporal(content)
                if temporal_info and temporal_info.get("event_date"):
                    import json as _json_tmp
                    cursor.execute(
                        "UPDATE working_memory SET event_date=?, event_date_precision=?, temporal_tags=? WHERE id=?",
                        (temporal_info["event_date"],
                         temporal_info["event_date_precision"],
                         _json_tmp.dumps(temporal_info["temporal_tags"]),
                         memory_id)
                    )
                    self.conn.commit()
            except Exception:
                pass  # Temporal extraction is best-effort

        # --- Entity extraction ---
        if extract_entities:
            _extract_and_store_entities(self, memory_id, content)

        # --- Structured fact extraction ---
        if extract:
            _extract_and_store_facts(self, memory_id, content, source)

        # Phase 2: MEMORIA regex-based extraction (always-on, zero-LLM-cost).
        # Populates memoria_facts, memoria_timelines, memoria_kg for the
        # structured retrieval router. Runs on every remember() call.
        try:
            self.extract_and_store_facts(content, message_idx=0, source_memory_id=memory_id)
        except Exception:
            pass  # regex extraction failures must not block memory storage

        # Phase 3-4: Extract graph and consolidate veracity for new memory
        self._ingest_graph_and_veracity(memory_id, content, source, veracity)

        self._emit_event("MEMORY_ADDED", memory_id, content=content,
                         source=source, importance=importance, metadata=metadata)

        # Invalidate enhanced recall cache on new memory
        if hasattr(self, "_query_cache") and self._query_cache is not None:
            self._query_cache.invalidate()

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
                    logger.info("Regex extraction failed, skipping", exc_info=True)
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
                # Phase 2: MEMORIA regex-based extraction for every batch row.
                try:
                    self.extract_and_store_facts(row_content, message_idx=0, source_memory_id=memory_id)
                except Exception:
                    pass  # regex extraction failures must not block memory storage
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

        # Phase 5: Proactive linking — auto-create graph edges to related memories
        self._proactively_link(memory_id, content)

    def _proactively_link(self, memory_id: str, content: str):
        """Phase 5: Auto-create graph edges between new memory and related existing memories.

        Two zero-LLM strategies:
        1. Content similarity via recall() — top-K via FTS5 + vector
        2. Entity overlap via shared facts in the graph

        Gated behind MNEMOSYNE_PROACTIVE_LINKING=1 env var.
        Non-blocking — failures never affect memory storage.
        """
        import os
        if os.environ.get("MNEMOSYNE_PROACTIVE_LINKING", "0") != "1":
            return
        if self.episodic_graph is None:
            return

        try:
            now = datetime.now().isoformat()

            # --- Strategy 1: Content similarity via direct FTS5 ---
            try:
                # Build a broad FTS5 query from content keywords (skip very short content)
                stop_words = frozenset({
                    "the", "a", "an", "is", "was", "are", "were", "be", "been",
                    "being", "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "can", "shall",
                    "to", "of", "in", "for", "on", "with", "at", "by", "from",
                    "as", "into", "through", "during", "before", "after", "above",
                    "below", "between", "out", "off", "over", "under", "again",
                    "further", "then", "once", "here", "there", "when", "where",
                    "why", "how", "all", "each", "every", "both", "few", "more",
                    "most", "other", "some", "such", "no", "nor", "not", "only",
                    "own", "same", "so", "than", "too", "very", "just", "because",
                    "about", "which", "who", "what", "this", "that", "these",
                    "those", "it", "its", "i", "me", "my", "we", "our", "you",
                    "your", "he", "him", "his", "she", "her", "they", "them",
                    "their", "and", "but", "or", "if", "while",
                })
                words = [w.lower().strip(".,!?;:'\"()[]{}") for w in content.split()]
                keywords = [w for w in words if len(w) > 2 and w not in stop_words]
                keywords = [kw for kw in keywords if kw.isalpha()]  # FTS5-safe only
                if len(keywords) >= 3:
                    fts_query = " OR ".join(keywords)
                    cursor = self.conn.execute(
                        "SELECT DISTINCT id FROM fts_working WHERE fts_working MATCH ? ORDER BY rank LIMIT 5",
                        (fts_query,)
                    )
                    similar_ids = [r["id"] for r in cursor.fetchall() if r["id"] != memory_id]

                    similarity_count = 0
                    for i, rid in enumerate(similar_ids[:5]):
                        existing = self.episodic_graph.conn.execute(
                            "SELECT 1 FROM graph_edges WHERE (source = ? AND target = ?) OR (source = ? AND target = ?)",
                            (memory_id, rid, rid, memory_id)
                        ).fetchone()
                        if existing:
                            continue
                        weight = max(0.1, 1.0 - i * 0.2)
                        self.episodic_graph.add_edge(GraphEdge(
                            source=memory_id, target=rid,
                            edge_type="related_to", weight=weight, timestamp=now,
                        ))
                        similarity_count += 1

                    if similarity_count:
                        logger.debug(
                            "Proactive linking (similarity): %d edges for %s", similarity_count, memory_id
                        )
            except Exception:
                logger.debug("Proactive linking similarity strategy failed for %s", memory_id, exc_info=True)

            # --- Strategy 2: Entity overlap via shared entity mentions ---
            try:
                mention_rows = self.conn.execute(
                    "SELECT value FROM annotations WHERE memory_id = ? AND kind = 'mentions'",
                    (memory_id,)
                ).fetchall()

                entity_count = 0
                for row in mention_rows:
                    entity_val = row["value"]
                    related = self.conn.execute(
                        """SELECT DISTINCT memory_id FROM annotations
                           WHERE kind = 'mentions' AND value = ?
                             AND memory_id != ?
                           LIMIT 5""",
                        (entity_val, memory_id)
                    ).fetchall()
                    for match in related:
                        rid = match["memory_id"]
                        existing = self.episodic_graph.conn.execute(
                            "SELECT 1 FROM graph_edges WHERE source = ? AND target = ? AND edge_type = 'references'",
                            (memory_id, rid)
                        ).fetchone()
                        if existing:
                            continue
                        self.episodic_graph.add_edge(GraphEdge(
                            source=memory_id, target=rid,
                            edge_type="references", weight=0.8, timestamp=now,
                        ))
                        entity_count += 1

                if entity_count:
                    logger.debug(
                        "Proactive linking (entity): %d edges for %s", entity_count, memory_id
                    )
            except Exception:
                logger.debug("Proactive linking entity strategy failed for %s", memory_id, exc_info=True)

        except Exception:
            logger.debug("Proactive linking outer wrapper failed for %s", memory_id, exc_info=True)
            # Non-blocking — never surface to caller

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
        then by recency. High-importance rules/bans surface reliably.

        Bumps recall_count and last_recalled on returned items, capped
        so a single read cannot extend the effective clock by more than
        WM_BUMP_CAP_HOURS. Prevents infinite extension of stale items
        while keeping hot items visible."""

        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        # Keep the original ordering contract (global memories first, then
        # session-local memories; each group by importance and recency) while
        # avoiding the previous ``session_id = ? OR scope = 'global'`` query.
        # Splitting this hot path lets SQLite use simple partial indexes for
        # each branch instead of scanning working_memory and building a temp
        # B-tree for the CASE-based ORDER BY.
        select_cols = "id, content, source, timestamp, importance, scope, last_recalled"
        common_predicate = "(valid_until IS NULL OR valid_until > ?) AND superseded_by IS NULL"

        cursor.execute(f"""
            SELECT {select_cols}
            FROM working_memory
            WHERE scope = 'global'
              AND {common_predicate}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, (now, limit))
        global_rows = [dict(row) for row in cursor.fetchall()]

        session_rows = []
        if limit < 0 or len(global_rows) < limit:
            session_limit = limit if limit < 0 else limit - len(global_rows)
            cursor.execute(f"""
                SELECT {select_cols}
                FROM working_memory
                WHERE session_id = ?
                  AND (scope IS NULL OR scope != 'global')
                  AND {common_predicate}
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """, (self.session_id, now, session_limit))
            session_rows = [dict(row) for row in cursor.fetchall()]

        rows = global_rows + session_rows
        if not rows:
            return rows

        # Bump recall_count + last_recalled for every item returned.
        # Each bump extends last_recalled by at most WM_BUMP_CAP_HOURS
        # to prevent a single get_context() from fully resetting a stale item.
        now_dt = datetime.now()
        bump_delta = timedelta(hours=WM_BUMP_CAP_HOURS)
        for row in rows:
            old_ts = row.pop("last_recalled", None)
            if old_ts is None:
                new_last = now_dt
            else:
                try:
                    parsed = datetime.fromisoformat(old_ts)
                except (ValueError, TypeError):
                    new_last = now_dt
                else:
                    new_last = min(now_dt, parsed + bump_delta)
            cursor.execute(
                "UPDATE working_memory SET recall_count = recall_count + 1, last_recalled = ? WHERE id = ?",
                (new_last.isoformat(), row["id"])
            )
        self.conn.commit()
        return rows

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

    def _detect_conflicts(self, rows: List[Dict], similarity_threshold: float = 0.88) -> List[tuple]:
        """
        Heuristic-only conflict detection for consolidation time.

        For each pair of memories in rows (already sorted by timestamp ASC):
        1. Get their embeddings from memory_embeddings table
        2. Compute cosine similarity
        3. If similarity > threshold AND timestamps differ >1h AND
           they share overlapping tokens AND they're not duplicates...
           flag the older as potentially superseded

        Returns list of (older_id, newer_id) tuples.
        No LLM calls. Pure vector math.
        """
        if len(rows) < 2:
            return []

        # Collect embeddings for all memory IDs in one query
        memory_ids = [r["id"] for r in rows]
        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(memory_ids))
        try:
            cursor.execute(f"""
                SELECT memory_id, embedding_json
                FROM memory_embeddings
                WHERE memory_id IN ({placeholders})
            """, memory_ids)
        except Exception:
            return []

        emb_map = {}
        for row in cursor.fetchall():
            try:
                emb_map[row["memory_id"]] = np.array(
                    json.loads(row["embedding_json"]), dtype=np.float32
                )
            except Exception:
                continue

        # Skip rows that are already superseded
        superseded_ids = {r["id"] for r in rows if r.get("superseded_by")}

        # Stop words for content overlap heuristics
        _stop_words = frozenset({
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can", "need",
            "this", "that", "these", "those", "it", "its", "they", "them", "their",
            "we", "us", "our", "you", "your", "he", "she", "him", "her", "his",
            "not", "no", "nor", "so", "if", "then", "than", "too", "very", "just",
            "about", "also", "more", "some", "any", "each", "every", "all", "both",
            "what", "when", "where", "why", "how", "which", "who", "whom",
            "get", "got", "make", "made", "take", "took", "use", "used", "like",
            "said", "says", "know", "knew", "think", "thinks", "thought",
            "see", "saw", "seen", "come", "came", "give", "gave", "tell", "told",
        })

        def _significant_tokens(text: str) -> Set[str]:
            words = re.findall(r"[A-Za-z]{3,}", text.lower())
            return {w for w in words if w not in _stop_words and not w.isdigit()}

        def _edit_dist_ratio(s1: str, s2: str) -> float:
            """Normalized distance (0=identical, 1=completely different)."""
            import difflib
            if not s1 and not s2:
                return 0.0
            if not s1 or not s2:
                return 1.0
            return 1.0 - difflib.SequenceMatcher(None, s1, s2).ratio()

        conflicts = []
        n = len(rows)
        for i in range(n):
            a = rows[i]
            a_id = a["id"]
            if a_id in superseded_ids or a_id not in emb_map:
                continue
            for j in range(i + 1, n):
                b = rows[j]
                b_id = b["id"]
                if b_id in superseded_ids or b_id not in emb_map:
                    continue

                # --- Heuristic 1: timestamps > 1 hour apart ---
                try:
                    ts_a = datetime.fromisoformat(a["timestamp"])
                    ts_b = datetime.fromisoformat(b["timestamp"])
                    hours_diff = abs((ts_b - ts_a).total_seconds()) / 3600.0
                    if hours_diff < 1.0:
                        continue
                except (ValueError, TypeError):
                    continue

                # --- Heuristic 2: cosine similarity > threshold ---
                vec_a = emb_map[a_id]
                vec_b = emb_map[b_id]
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                if norm_a == 0 or norm_b == 0:
                    continue
                cos_sim = float(np.dot(vec_a / norm_a, vec_b / norm_b))
                if cos_sim <= similarity_threshold:
                    continue

                # --- Heuristic 3: at least 2 overlapping significant tokens ---
                tokens_a = _significant_tokens(a["content"])
                tokens_b = _significant_tokens(b["content"])
                overlapping = tokens_a & tokens_b
                if len(overlapping) < 2:
                    continue

                # --- Heuristic 4: not near-duplicates (edit distance > 0.3) ---
                if _edit_dist_ratio(a["content"], b["content"]) <= 0.3:
                    continue

                # Rows are sorted ASC, so a is older. Flag older as superseded.
                conflicts.append((a_id, b_id))

        return conflicts

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

        consolidated_where = (f"{where_str} AND consolidated_at IS NOT NULL" if where_str
                              else " WHERE consolidated_at IS NOT NULL")
        cursor.execute(f"SELECT COUNT(*) FROM working_memory{consolidated_where}", params)
        consolidated = cursor.fetchone()[0]

        unconsolidated = total - consolidated

        cursor.execute(f"SELECT timestamp FROM working_memory{where_str} ORDER BY timestamp DESC LIMIT 1", params)
        last = cursor.fetchone()
        return {
            "total": total,
            "consolidated": consolidated,
            "unconsolidated": unconsolidated,
            "last": last[0] if last else None,
        }

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

    def get(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve a single memory by its primary key (id).
        Pure read -- no side effects, no recall_count bump, no FTS trigger.

        Checks working_memory first (faster, higher hit rate),
        then episodic_memory (fallback).

        Returns None if not found in either table.
        """
        cursor = self.conn.cursor()

        # Working memory first (fast path)
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id,
                   importance, metadata_json, veracity, created_at
            FROM working_memory
            WHERE id = ? AND session_id = ?
        """, (memory_id, self.session_id))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "timestamp": row[3],
                "session_id": row[4],
                "importance": row[5],
                "metadata": row[6],
                "veracity": row[7],
                "created_at": row[8],
                "memory_store": "working",
            }

        # Episodic memory (fallback)
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id,
                   importance, metadata_json, veracity, created_at
            FROM episodic_memory
            WHERE id = ? AND (session_id = ? OR scope = 'global')
        """, (memory_id, self.session_id))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "timestamp": row[3],
                "session_id": row[4],
                "importance": row[5],
                "metadata": row[6],
                "veracity": row[7],
                "created_at": row[8],
                "memory_store": "episodic",
            }

        return None

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
                "DELETE FROM working_memory WHERE id = ? AND (session_id = ? OR scope = 'global')",
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
                logger.info("Regex extraction failed, skipping", exc_info=True)
        # Clamp to canonical allowlist at the trust boundary. Defaults to
        # 'unknown' if not provided (back-compat with pre-E4.a.1 callers).
        if veracity is None:
            row_veracity = "unknown"
        else:
            row_veracity = clamp_veracity(
                veracity, context="consolidate_to_episodic.veracity"
            )
        # Strip closed <think>...</think> blocks that some LLMs emit
        import re as _re
        summary = _re.sub(r"<think>.*?</think>", "", summary, flags=_re.DOTALL).strip()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO episodic_memory
            (id, content, source, timestamp, session_id, importance, metadata_json, summary_of, valid_until, scope,
             author_id, author_type, channel_id, memory_type, veracity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, _sanitize_utf8(summary), source, timestamp, self.session_id, importance,
              json.dumps(metadata or {}), ",".join(source_wm_ids), valid_until, scope,
              self.author_id, self.author_type, self.channel_id, ep_type, row_veracity))
        rowid = cursor.lastrowid

        if _embeddings.available():
            vec = _embeddings.embed([summary])
            if vec is not None:
                if _vec_available(self.conn):
                    try:
                        _vec_insert(self.conn, rowid, np.asarray(vec[0]).tolist())
                    except Exception as _vec_exc:
                        logger.warning(
                            "vec_episodes insert failed (rowid=%s): %s",
                            rowid, _vec_exc,
                        )
                else:
                    # Fallback: store in memory_embeddings table for in-memory search
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                        VALUES (?, ?, ?)
                    """, (memory_id, _embeddings.serialize(np.asarray(vec[0])), _embeddings._DEFAULT_MODEL))

                # Binary vector compression (Phase 2 -- 32x reduction)
                if _mib is not None:
                    try:
                        bv = _mib(np.asarray(vec[0]))
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

    # ------------------------------------------------------------------
    # MEMORIA: Structured Fact Extraction (Phase 1)
    # ------------------------------------------------------------------
    def detect_language(self, text: str) -> str:
        """Ultra-fast language detection, zero deps, <0.1ms.
        Returns ISO 639-1 code ('en', 'de', 'ru', etc.). Extendable."""
        if not text or not isinstance(text, str):
            return 'en'
        text_lower = text.lower()
        # Russian: detect Cyrillic chars + Russian markers
        CYRILLIC = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        russian_chars = sum(1 for c in text_lower if c in CYRILLIC)
        if russian_chars >= 5:
            return 'ru'
        if russian_chars >= 2:
            ru_markers = {'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
                          'не', 'на', 'в', 'с', 'по', 'для', 'что', 'как',
                          'это', 'так', 'но', 'да', 'нет', 'уже', 'ещё',
                          'мой', 'твой', 'наш', 'ваш', 'этот', 'тот'}
            words = set(text_lower.split())
            if len(words & ru_markers) >= 2:
                return 'ru'
        if any(c in text_lower for c in 'äöüß'):
            return 'de'
        german_markers = {'ich', 'du', 'wir', 'ist', 'nicht', 'für', 'und',
                          'der', 'die', 'das', 'ein', 'eine', 'kein', 'keine',
                          'mein', 'meine', 'dann', 'auch', 'immer', 'nie', 'niemals',
                          'mag', 'will', 'möchte', 'kann', 'kannst', 'können',
                          'habe', 'hast', 'hat', 'haben', 'bin', 'bist', 'sind', 'seid',
                          'einen', 'einer', 'eines', 'dem', 'den', 'beim', 'zum', 'zur',
                          'nach', 'mit', 'von', 'bei', 'aus', 'auf', 'vor', 'aber',
                          'oder', 'weil', 'denn', 'dass', 'sehr', 'schon', 'noch',
                          'mal', 'man', 'nur', 'wenn', 'wie', 'als', 'doch',
                          'gerne', 'gern', 'lieber', 'einfach', 'eigentlich',
                          'vielleicht', 'natürlich', 'genau', 'bereits', 'eben'}
        import re
        words = set(re.findall(r'\w+', text_lower))
        if len(words & german_markers) >= 2:
            return 'de'
        # Spanish: detect accent chars + Spanish markers
        if any(c in text_lower for c in 'ñáéíóúü¿¡'):
            return 'es'
        es_markers = {
            'y', 'de', 'por', 'con', 'para', 'que', 'qué', 'como',
            'el', 'la', 'lo', 'los', 'las', 'un', 'una', 'del',
            'este', 'esta', 'esto', 'ese', 'esa', 'eso', 'aquel',
            'mi', 'mis', 'tu', 'tus', 'su', 'sus',
            'es', 'está', 'son', 'hay', 'tiene', 'puede',
            'más', 'no', 'también', 'si', 'ya', 'nunca', 'he',
            'se', 'me', 'te', 'le', 'a', 'yo',
            'ante', 'bajo', 'contra', 'desde', 'en', 'entre',
            'hacia', 'hasta', 'según', 'sin', 'sobre', 'tras',
            'todo', 'toda', 'cada', 'muy', 'pero',
            'siempre', 'usa', 'hacer', 'antes',
            'recuerda', 'evita',
        }
        words = set(re.findall(r'\w+', text_lower))
        if len(words & es_markers) >= 2:
            return 'es'
        # Italian: detect accent chars + Italian markers
        if any(c in text_lower for c in 'àèéìòù'):
            italian_markers = {
                'e', 'il', 'la', 'i', 'le', 'di', 'che', 'non', 'un', 'una',
                'per', 'è', 'in', 'sono', 'mi', 'ha', 'ma', 'lo', 'se', 'su',
                'con', 'da', 'come', 'questo', 'quello', 'anche', 'o', 'ho',
                'ci', 'si', 'perché', 'perche', 'quando', 'chi', 'dove', 'molto',
                'del', 'della', 'delle', 'dei', 'degli', 'nel', 'nella', 'sul',
                'sulla', 'sui', 'sulle', 'al', 'alla', 'agli', 'alle',
            }
            words = set(re.findall(r'\w+', text_lower))
            if len(words & italian_markers) >= 2:
                return 'it'
        return 'en'

    MULTILINGUAL_PATTERNS = {
        'en': {
            'negation': r'(I(?: have|\'ve)?\s*(?:never|not)\s+[^.,;!?\n]{15,120})',
            'decision': r'(?:decided to|chose to|opted for|selected|picked|switching to)\s+([^.,;!?\n]{10,120})',
            'entity': r'(?:the|my|our|your)\s+([a-z_]+(?:\s+(?:table|model|schema|API|endpoint|function|module|route|handler|tool|plugin|script|config|setting|workflow|pipeline|process|system|server|client|service|database|query|file|repo|branch|PR|issue|task|job)))\s+(?:needs?|requires?|should|could|would|will|has|have|uses?|runs?|handles?|processes?|supports?)\s+([^.,;!?\n]{10,80})',
            'sequence': r'((?:first|second|third|fourth|fifth|finally|next|then|after that)[^.,;!?\n]{15,120})',
            'instruction_false_positives': ['i think you should leave', 'should behave', 'their work style'],
            'instruction_imperative': 'always|never|remember|use|keep|avoid|ensure|check|verify|run|test|build|deploy|push|pull|merge|commit|close|open|update|install|configure|set|enable|disable|add|remove|create|delete|start|stop|restart|reload|reset|try|implement|write|read|switch|move|copy|rename|send|reply|respond',
            'instruction': r'(?:always|never|must|must not|should(?: not)?(?=\s+(?:you|we|i|one)\s+(?:IMPVERBS))|need(?:s)? to(?: not)?|required to|prefer(?: not)? to|want to(?: avoid| ensure| use| keep))\s+([^.,;!?\n]{10,200})',
            'preference': r'(?:'
                # First person (original) and second person
                r'(?:I|You|you|YOU)(?: |\')?(?:like|love|prefer|hate|dislike|enjoy|use|stick with|switched to|moved to|changed to|want|need|tend to|usually|would rather|don\'t like|don\'t want|not a fan of|am okay with|am comfortable with|am used to|am happy with|am tired of|am sick of|prefer not to|try to avoid|find it easier to|find it better to|find it useful to)'
                r'|'
                # Third-person names with English 3p verbs (family members)
                r'(?:Nathan|Bob|User|Amy|Zander|Zella)\s+(?:likes?|loves?|prefers?|hates?|dislikes?|enjoys?|uses?|wants?|needs?|tends\s+to|switches?\s+to|changes?\s+to|moves?\s+to)'
                r'|'
                # Structural-start verbs (bullet/heading/em-dash clause start)
                r'(?:(?<=^)|(?<=\n)|(?<=\-\s)|(?<=—\s))(?:Prefers|Likes|Loves|Hates|Dislikes|Wants|Needs|Tends to|Enjoys|Uses)'
                r')'
                r'\s+([^.,;!?\n]{10,200})',
            'event_keywords': ['meeting', 'call', 'scheduled', 'happened', 'occurred', 'plan to', 'will be on', 'due on', 'release', 'deadline', 'launched', 'deployed', 'released', 'published', 'posted', 'started', 'began', 'finished', 'completed', 'ended', 'event', 'conference', 'workshop', 'appointment'],
            'named_months': r'((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:\d{4})?)',
        },
        'de': {
            'negation': r'(Ich(?: habe|\'ve)?\s+(?:nie|niemals|nicht)\s+[^.,;!?\n]{15,120})',
            'decision': r'(?:entschied(?: mich|en)?|habe mich entschieden|wechselte zu|umgestellt auf|umgestiegen auf|gewählt habe|ausgesucht|ausgewählt|genommen habe)\s+([^.,;!?\n]{10,120})',
            'entity': r'(?:der|die|das|mein|meine|dein|deine|unser|unsere|Ihr|Ihre)\s+([a-z_]+(?:\s+(?:Tabelle|Modell|Schema|API|Endpunkt|Funktion|Modul|Route|Handler|Tool|Plugin|Script|Konfiguration|Einstellung|Workflow|Pipeline|Prozess|System|Server|Client|Service|Datenbank|Query|Datei|Repo|Branch|PR|Issue|Task|Job)))\s+(?:braucht|benötigt|sollte|könnte|würde|wird|hat|hat|nutzt|verwendet|läuft|bearbeitet|verarbeitet|unterstützt)\s+([^.,;!?\n]{10,80})',
            'sequence': r'((?:zuerst|als erstes|als zweites|als drittes|als viertes|als fünftes|schließlich|als nächstes|dann|danach|daraufhin)[^.,;!?\n]{15,120})',
            'instruction_false_positives': [
                'du solltest gehen',
                'ich denke du solltest',
                'sollte funktionieren',
                'sollte klappen',
                'sollte passen',
                'sollte sich',
            ],
            'instruction_imperative': 'immer|nie|niemals|merke|denk|verwende|nutze|behalte|vermeide|stelle sicher|prüfe|überprüfe|teste|baue|implementiere|schreibe|lösche|installiere|konfiguriere|aktualisiere|erstelle|entferne|starte|stoppe|setze|aktiviere|deaktiviere|füge hinzu|benenne um|sende|antworte',
            'instruction': r'(?:immer|nie|niemals|muss|darf nicht|sollte(?: nicht)?(?=\s+(?:du|wir|ich|man|ihr)\s+(?:IMPVERBS))|braucht|benötigt|möchte(?: vermeiden|sicherstellen|nutzen|behalten)|will(?: nicht)?)\s+([^.,;!?\n]{10,200})',
            'preference': r'(?:Ich(?: |\')?(?:mag|liebe|bevorzuge|hasse|mag nicht|nutze|verwende|benutze|bin bei geblieben|habe gewechselt zu|bin umgestiegen auf|bin umgestellt auf|will|möchte|brauche|tendiere zu|normalerweise|würde lieber|finde es einfacher|finde es besser|finde es nützlich|bin zufrieden mit|bin okay mit|bin es leid|versuche zu vermeiden))\s+([^.,;!?\n]{10,200})',
            'event_keywords': ['treffen', 'meeting', 'termin', 'anruf', 'geplant', 'passiert', 'stattgefunden', 'fällig', 'release', 'deadline', 'veröffentlicht', 'deployed', 'gestartet', 'begonnen', 'beendet', 'abgeschlossen', 'konferenz', 'workshop', 'termin'],
            'named_months': r'((?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\s+\d{1,2}(?:\.)?\s*(?:\d{4})?)',
        },
        'ru': {
            'negation': r'((?:Я(?: |\')?(?:никогда|не)|(?:никогда|не)\s+будет)\s+[^.,;!?\n]{6,120})',
            'decision': r'(?:решил|решила|решили|выбрал|выбрала|выбрали|перешёл|перешла|перешли|переключился|переключилась|переключились|переехал|переехала|переехали|поменял|поменяла|поменяли)\s+([^.,;!?\n]{2,120})',
            'entity': r'(?:мой|моя|моё|мои|наш|наша|наше|наши|твой|твоя|твоё|твои|ваш|ваша|ваше|ваши)\s+([a-zA-Zа-яА-Я_]+(?:\s+(?:таблица|модель|схема|API|эндпоинт|функция|модуль|роут|обработчик|тул|плагин|скрипт|конфиг|настройка|воркфлоу|пайплайн|процесс|система|сервер|клиент|сервис|база|данных|запрос|файл|репозиторий|ветка|PR|ишью|таска|джоба|контейнер|образ|проект|релиз|версия))?)\s+(?:нуждается|требует|должен|должна|должны|может|могут|будет|будут|имеет|имеют|использует|используют|работает|работают|обрабатывает|поддерживает|запущен|запущена|настроен|настроена|готов|готова|готовы|запланирован|обновлён|обновлена|опубликован|опубликована|создан|создана)\s+([^.,;!?\n]{3,80})',
            'sequence': r'((?:во-первых|во-вторых|в-третьих|в-четвёртых|в-пятых|наконец|затем|потом|после этого|дальше|сначала)\s*,?\s*[^.,;!?\n]{6,120})',
            'instruction_false_positives': [],
            'instruction_imperative': 'всегда|никогда|помни|запомни|используй|пользуйся|храни|избегай|убедись|проверь|запусти|тестируй|собери|задеплой|сделай пуш|сделай пулл|мёржи|закрой|открой|обнови|установи|настрой|включи|отключи|добавь|удали|создай|удали|запусти|останови|рестартни|перезагрузи|сбрось|попробуй|реализуй|напиши|прочитай|переключи|передвинь|скопируй|переименуй|отправь|ответь',
            'preference': r'(?:(?:Я(?: |\')?(?:люблю|ненавижу|предпочитаю|терпеть не могу|не люблю|не нравится|использую|пользуюсь|остаюсь на|перешёл на|переключился на|хочу|нуждаюсь|обычно|скорее|предпочитаю не|стараюсь избегать|привык|надоело|устал от|доволен|устраивает))|мне\s+(?:нравится|не нравится|проще|удобнее|лень|надоело)|терпеть не могу|надоело|привык|устраивает)\s+([^.,;!?\n]{3,200})',
            'event_keywords': ['встреча', 'созвон', 'запланировано', 'состоялось', 'произошло', 'планирую', 'будет', 'дедлайн', 'релиз', 'запуск', 'деплой', 'опубликовано', 'начал', 'начался', 'закончил', 'завершил', 'событие', 'конференция', 'воркшоп', 'встреча'],
            'named_months': r'((?:(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек)\s+\d{1,2}(?:-го)?,?\s*(?:\d{4})?)|(?:\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?))',
            'instruction': r'(?:всегда|никогда|должен|не должен|нужно|не нужно|обязательно|нельзя|не забывай|запомни|помни|следует|стоит)\\s+([^.,;!?\\n]{6,200})',
        },
        'it': {
            'negation': r"((?:Non(?: |')?(?:ho|ho mai|mai|non)\s+[^.,;!?\n]{15,120}))",
            'decision': r'(?:ho deciso|mi sono deciso|ho scelto|ho optato|ho cambiato|sono passato|sono passata|ho selezionato|scelto)\s+([^.,;!?\n]{10,120})',
            'entity': r"(?:il|la|i|le|il mio|la mia|i miei|le mie|il tuo|la tua|il nostro|la nostra)\s+([a-z_]+(?:\s+(?:tabella|modello|schema|API|endpoint|funzione|modulo|route|handler|tool|plugin|script|config|impostazione|workflow|pipeline|processo|sistema|server|client|servizio|database|query|file|repo|branch|PR|issue|task|job|progetto)))\s+(?:ha bisogno|richiede|dovrebbe|potrebbe|vorra|ha|hanno|usa|usano|funziona|gestisce|processa|supporta)\s+([^.,;!?\n]{10,80})",
            'sequence': r'((?:primo|prima|secondo|seconda|terzo|terza|quarto|quinta|infine|poi|dopo|dopodiche|successivamente|quindi)[^.,;!?\n]{15,120})',
            'instruction_false_positives': [
                'dovresti andare',
                'penso che dovresti',
                'dovrebbe funzionare',
                'dovrebbe andare',
                'dovrebbe andare bene',
                'dovrebbe essere',
                'dovrebbe bastare',
            ],
            'instruction_imperative': 'sempre|mai|ricorda|usa|tieni|evita|assicurati|controlla|verifica|esegui|testa|costruisci|distribuisci|fai push|fai pull|fai merge|chiudi|apri|aggiorna|installa|configura|imposta|abilita|disabilita|aggiungi|rimuovi|crea|elimina|avvia|ferma|riavvia|resetta|prova|implementa|scrivi|leggi|passa|sposta|copia|rinomina|invia|rispondi',
            'instruction': r'(?:sempre|mai|non deve|non devono|dovrebbe(?: non)?(?=\s+(?:tu|voi|noi|io|si)\s+(?:IMPVERBS))|ha bisogno di|deve|devono|preferisci(?: non)?|vuole(?: evitare|assicurarsi|usare|tenere))\s+([^.,;!?\n]{10,200})',
            'preference': r"(?:Io(?: |')?(?:mi piace|amo|preferisco|odio|non mi piace|uso|utilizzo|sono passato a|ho cambiato a|voglio|ho bisogno|tendo a|di solito|preferirei|non mi piace per niente|non voglio|non sono un fan di|mi va bene|mi trovo bene|sono abituato a|sono felice con|sono stanco di|cerco di evitare|trovo piu facile|trovo meglio|trovo utile))\s+([^.,;!?\n]{10,200})",
            'event_keywords': ['riunione', 'chiamata', 'incontro', 'programmato', 'successo', 'accaduto', 'pianifico', 'sara il', 'scadenza', 'rilascio', 'lancio', 'pubblicato', 'iniziato', 'cominciato', 'finito', 'completato', 'evento', 'conferenza', 'workshop', 'appuntamento'],
            'named_months': r'((?:(?:Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre|gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\s+\d{1,2}(?:°)?,?\s*(?:\d{4})?))',
        },
        'es': {
            'negation': r'(nunca|jamás|tampoco|ni\s+(?:siquiera|de coña|loc[ao]|de broma|hablar)|no\s+(?:me\s+(?:gusta|convence|interesa|molesta|duele)|lo\s+(?:hag[ao]s|haré|haría)|hace\s+falta|quiero|voy\s+a|sé|sabía|puedo|debo|es\s+(?:para\s+tanto|plan|momento)|tiene\s+sentido|estoy\s+(?:de\s+acuerdo|seguro)|hay\s+(?:derecho|manera|tipo|quien)|teng[ao]\s+(?:ni\s+idea|claro)|pienso|creo|son|era|está|estaba|será|está\s+mal|vamos\s+mal))\s+([^.,;!?¿¡\\n]{15,120})',
            'decision': r'(?:decid(?:í|ió|imos|iste|isteis|ieron|o|es|e|en)|opt(?:é|ó|amos|aste|asteis|aron|o|a|an)\s+por|cambi(?:é|ó|amos|aste|asteis|aron|o|a|an)\s+(?:de|a)|eleg(?:í|ió|imos|iste|isteis|ieron|o|es|e|en)|seleccion(?:é|ó|amos|aste|asteis|aron|o|a|an)|me\s+(?:pas|decant|escog)(?:é|ó|amos|o|a|an)\s+(?:a|por)|migr(?:é|ó|amos|aste|asteis|aron|o|a|an)\s+(?:de|a)|actualic(?:é|ó|amos|aste|asteis|aron|o|a|an)\s+(?:de|a)|sustitu(?:í|yó|imos|iste|isteis|yeron|yo|yes|ye|yen)\s+por|elimin(?:é|ó|amos|aste|asteis|aron|o|a|an)|descart(?:é|ó|amos|aste|asteis|aron|o|a|an)|y\s+si\s+[^.,;!?¿¡\\n]{10,200}|mejor\s+(?:si|así))\s+([^.,;!?¿¡\\n]{10,120})',
            'entity': r'(el|la|mi|tu|su|nuestr[oa]|vuestr[oa]|mis|tus|sus|los|las)\s+(servidor|maquina|vm|contenedor|docker|nodo|clúster|cluster|router|enrutador|gateway|puerta\s+de\s+enlace|switch|ap|punto\s+de\s+acceso|firewall|cortafuegos|vpn|vlan|dns|dhcp|api|endpoint|función|funcio|módulo|modulo|servicio|proceso|script|plugin|tool|skill|base\s+de\s+datos|bd|tabla|query|consulta|log|backup|snapshot|sensor|cámara|camara|luz|interruptor|alarma|estación\s+meteorológica|estacion\s+meteorologica|automatización|automatizacion|puerta|repo|repositorio|rama|branch|pr|issue|tarea|workflow|pipeline|config|configuración|configuracion|ajuste|carpeta|opciones|archivo|fichero|dashboard|interfaz|sistema|actualización|actualizacion|versión|versio|despliegue|deploy|release|entorno)(?:\s+(?:\w+))?\s+(?:necesita|requiere|debería|deberia|podría|podria|puede|tiene\s+que|usa|utiliza|ejecuta|gestiona|maneja|procesa|soporta|funciona\s+con|depende\s+de|contiene|implementa|despliega|actualiza|configura|corre\s+(?:en|sobre)|monitoriza|notifica|está|esta)\s+([^.,;!?¿¡\\n]{10,80})',
            'sequence': r'((?:primero|primeramente|en\s+primer\s+lugar|segundo|en\s+segundo\s+lugar|tercero|en\s+tercer\s+lugar|para\s+empezar|yo\s+empezaría\s+por|yo\s+empezaria\s+por|por\s+mi\s+parte|por\s+otro\s+lado|luego|después|despues|a\s+continuación|a\s+continuacion|mientras\s+tanto|al\s+mismo\s+tiempo|finalmente|por\s+último|por\s+ultimo|para\s+terminar|antes\s+de|acto\s+seguido|por\s+una\s+parte|por\s+otra\s+parte|posteriormente)[^.,;!?¿¡\\n]{15,120})',
            'instruction_false_positives': [
                'evita perón', 'evita peron',
                'goma de borrar',
                'guarda silencio',
                'busca la paz',
                'no cambies nunca',
                'comprimido efervescente',
                'copia de seguridad',
                'prueba a ver',
                'prueba y error',
                'mira tú por donde', 'mira tu por donde',
                'baja la cabeza',
                'no deberías preocuparte por eso ahora', 'no deberias preocuparte por eso ahora',
                'debería funcionar sin problemas', 'deberia funcionar sin problemas',
                'mejor lo dejamos así', 'mejor lo dejamos asi',
                'habría que verlo primero', 'habria que verlo primero',
                'igual deberías preguntar antes', 'igual deberias preguntar antes',
                'tendrías que probarlo tú mismo', 'tendrias que probarlo tu mismo',
                'puedes hacer lo que quieras',
                'no hace falta que hagas nada',
                'yo que tú lo dejaba correr', 'yo que tu lo dejaba correr',
                'a veces es mejor no tocar nada',
                'nunca he usado', 'nunca he probado', 'nunca he visto',
                'nunca he tenido', 'nunca he hecho', 'nunca he sido',
                'nunca has usado', 'nunca ha usado', 'nunca hemos usado',
                'nunca lo he', 'nunca lo había', 'nunca había',
            ],
            'instruction_imperative': 'siempre|nunca|recuerda|recordad|recuerde|recuerden|haz|haced|haga|hagan|usa|usad|use|usen|mantén|mantened|mantenga|mantengan|evita|evitad|evite|eviten|asegúrate|aseguraos|asegúrese|asegúrense|asegurate|aseguraos|asegurese|asegurense|verifica|verificad|verifique|verifiquen|comprueba|comprobad|compruebe|comprueben|revisa|revisad|revise|revisen|ejecuta|ejecutad|ejecute|ejecuten|prueba|probad|pruebe|prueben|pon|poned|ponga|pongan|configura|configurad|configure|configuren|instala|instalad|instale|instalen|actualiza|actualizad|actualice|actualicen|borra|borrad|borre|borren|guarda|guardad|guarde|guarden|busca|buscad|busque|busquen|despliega|desplegad|despliegue|desplieguen|crea|cread|cree|creen|memoriza|memorizad|memorice|memoricen|graba|grabad|grabe|graben|añade|añadid|añada|añadan|anade|anadid|anada|anadan|cambia|cambiad|cambie|cambien|arregla|arreglad|arregle|arreglen|sube|subid|suba|suban|baja|bajad|baje|bajen|carga|cargad|cargue|carguen|descarga|descargad|descargue|descarguen|comprime|comprimid|comprima|compriman|descomprime|descomprimid|descomprima|descompriman|copia|copiad|copie|copien|mueve|moved|mueva|muevan',
            'instruction': r'(?:siempre|nunca|hay\s+que|deb(?:es|éis|e|en|o|emos|éis|en)\s+|tienes\s+que|tenéis\s+que|tiene\s+que|tienen\s+que|es\s+necesario|es\s+importante|es\s+mejor|es\s+aconsejable|asegúrate\s+de|asegurate\s+de|record(?:ad|a|e|en)\s+|no\s+olvid(?:es|éis|e|en|ad)\s+)([^.,;!?¿¡\\n]{10,200})',
            'preference': r'(?:(?:yo|a mí|a mi)\s+)?(?:me\s+(?:gusta|encanta|mola|flipa|chifla|va\s+bien|resulta\s+(?:cómodo|comodo|útil|util|fácil|facil|mejor))|no\s+me\s+(?:gusta|mola|interesa|va|conviene)|prefiero|preferiría|preferiria|odian?|odio|detesto|no\s+soporto|me\s+molesta|me\s+duele|no\s+quiero|paso\s+de|estoy\s+(?:harto|cansado)\s+de|estoy\s+acostumbrado\s+a|suelo\s+usar|suelo\s+trabajar|me\s+siento\s+cómodo|comodo\s+con|no\s+soy\s+fan\s+de|he\s+(?:empezado|dejado|comenzado|terminado)\s+(?:a|de)|dejé|deje|descarte|descarté|eliminé|elimine|cambié|cambie|me\s+quedo\s+con|me\s+decanto\s+por|disfruto|me\s+hace\s+feliz|estoy\s+(?:a\s+gusto|probando))\s+([^.,;!?¿¡\\n]{10,200})',
            'event_keywords': [
                'reunión', 'reunion', 'llamada', 'cita', 'meeting', 'daily',
                'sprint', 'planning', 'retro', 'review', 'revisión', 'revision',
                'demo', 'demostración', 'demostracion',
                'evento', 'conferencia', 'taller', 'workshop', 'webinar', 'seminario',
                'cumpleaños', 'cumpleanos', 'aniversario', 'festivo', 'vacaciones',
                'programado', 'agendado', 'planeado', 'previsto', 'pendiente',
                'deadline', 'fecha límite', 'fecha tope', 'entrega',
                'lanzamiento', 'release', 'despliegue', 'deploy',
                'publicación', 'publicacion', 'subida',
                'empezó', 'empezo', 'comenzó', 'comenzo', 'inició', 'inicio',
                'arrancó', 'arranco', 'terminó', 'termino', 'finalizó', 'finalizo',
                'acabó', 'acabo', 'completé', 'complete',
                'lanzamos', 'publicamos', 'desplegamos', 'implementamos',
                'tengo una cita', 'tenemos una reunión', 'vamos a vernos',
                'agendé', 'agende', 'ocurrió', 'ocurrio', 'sucedió', 'sucedio',
                'pasó', 'paso',
            ],
            'named_months': r'((?:\d{1,2})\s*de\s*(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\s*(?:de\s*(?:\d{4}))?)',
        },
    }

    def extract_and_store_facts(self, content: str, message_idx: int = 0,
                                source_memory_id: Optional[str] = None) -> dict:
        """Extract structured facts from a message and store in facts/timelines/kg tables.
        Uses regex patterns matching the BEAM benchmark oracles.
        Language-aware: detects language and uses language-specific patterns.
        Returns dict of counts per fact_type."""
        import re as _re
        counts = {"metric": 0, "date": 0, "version": 0, "entity": 0,
                  "sequence": 0, "timeline": 0, "negation": 0, "decision": 0}
        session = self.session_id

        # Language detection
        lang = self.detect_language(content)
        pat = self.MULTILINGUAL_PATTERNS.get(lang, self.MULTILINGUAL_PATTERNS['en'])

        # Metrics: numbers with units — use finditer for position info
        _metric_re_iter = list(_re.finditer(
            r'(\d+(?:[.,]\d+)?)\s*(ms|sec|seconds?|minutes?|hours?|days?|'
            r'weeks?|months?|%|KB|MB|GB|TB|rows?|columns?|roles?|features?|'
            r'bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|'
            r'sprints?|tickets?)',
            content, _re.IGNORECASE))
        # Transient context keywords — metrics extracted near these are likely noise
        _TRANSIENT_KEYWORDS = (
            'forecast', 'weather', 'temperature', 'rain', 'snow', 'wind',
            'humidity', 'chance', 'regenrisiko', 'zum', 'heute', 'morgen',
            'gestern', 'today', 'tomorrow', 'yesterday', 'week', 'month'
        )
        for m in _metric_re_iter[:10]:
            num = m.group(1)
            unit = m.group(2)
            unit_clean = unit.lower()
            # Strip plural 's' but preserve "ms" (milliseconds)
            if unit_clean.endswith('s') and not unit_clean.endswith('ms'):
                unit_clean = unit_clean[:-1]
            # Extract context words before the metric to build a specific key.
            # "dashboard API response time of 250ms" → "response_time_ms"
            pre_text = content[max(0, m.start()-50):m.start()]
            # Check for transient/forecast context — skip if found
            if any(kw in pre_text.lower() for kw in _TRANSIENT_KEYWORDS):
                continue
            # Strip markdown, backticks, and code formatting from pre_text
            _clean_pre = _re.sub(r'`[^`]*`', ' ', pre_text)  # inline code
            _clean_pre = _re.sub(r'\*\*[^*]+\*\*', ' ', _clean_pre)  # bold
            _clean_pre = _re.sub(r'[*_]{1,2}[^*_\n]+[*_]{1,2}', ' ', _clean_pre)  # italic/bold
            _clean_pre = _re.sub(r'[=<>|&]', ' ', _clean_pre)  # code operators
            ctx_words = [w.strip('.,:;!?()[]"\'`*_') for w in _clean_pre.split()
                         if len(w.strip('.,:;!?()[]"\'`*_')) > 2
                         and w.lower() not in ('the', 'and', 'for', 'was', 'of', 'to', 'a', 'an', 'in', 'on', 'at', 'by', 'is', 'are', 'has', 'had', 'not', 'but', 'or')
                         and not _re.search(r'^(pt-|lg:|pr-|pl-|pb-|px-|py-|mt-|mr-|mb-|ml-|mx-|my-)', w)  # CSS utility classes
                         and not _re.search(r'^[`*\]]', w)  # stray markdown leftovers
                         ][-3:]
            prefix = '_'.join(w.lower() for w in ctx_words) if ctx_words else ''
            key = f"{prefix}_{unit_clean}" if prefix else unit_clean
            # Quality gate: reject garbage keys
            if '`' in key or '**' in key or key.count('**') > 0:
                continue
            if _re.search(r'[*_]{2,}', key):  # stray markdown emphasis
                continue
            # Reject keys that look like code (contain >2 special chars)
            if len(_re.findall(r'[`=<>|]', key)) > 2:
                continue
            # Reject % metrics in code-heavy contexts (more special chars than words)
            if unit_clean == '%':
                _nonalpha = len(_re.findall(r'[^a-zA-Z0-9\s]', _clean_pre))
                _words = len(_clean_pre.split())
                if _words > 0 and _nonalpha / _words > 0.6:
                    continue
            val = f"{num}{unit}"
            # Clean % suffix: use _pct instead of _% for readability
            if unit_clean == '%':
                key = key.replace('_%', '_pct')
                if not key.endswith('_pct'):
                    key = f"{prefix}_pct" if prefix else 'pct'
            self._insert_fact(session, message_idx, 'metric', key, val,
                              self._context_snippet(content, m.start()), 0.65,
                              source_memory_id=source_memory_id)
            counts["metric"] += 1

        # ISO Dates — require event context within 100 chars to avoid
        # file paths, report dates, and passing mentions as timeline entries
        _EVENT_KEYWORDS = pat['event_keywords']
        for m in _re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b', content):
            dt = m.group(1)
            ctx = self._context_snippet(content, m.start(), width=100)
            # Check for event verbs in context (language-aware)
            _ctx_lower = ctx.lower()
            _has_event_context = any(kw in _ctx_lower for kw in _EVENT_KEYWORDS)
            if not _has_event_context:
                # Still store as a fact (date mention), but skip timeline entry
                self._insert_fact(session, message_idx, 'date', 'iso_date', dt, ctx, 0.5, source_memory_id=source_memory_id)
                counts["date"] += 1
            else:
                self._insert_fact(session, message_idx, 'date', 'iso_date', dt, ctx, 0.7, source_memory_id=source_memory_id)
                self._insert_timeline(session, dt, message_idx, ctx[:120], 'iso_date', source_memory_id=source_memory_id)
                counts["date"] += 1
                counts["timeline"] += 1

        # Named dates (e.g. March 29, 2024 — language-aware)
        for m in _re.finditer(pat['named_months'], content, _re.IGNORECASE):
            dt = m.group(1).strip()
            ctx = self._context_snippet(content, m.start())
            self._insert_fact(session, message_idx, 'date', 'named_date', dt, ctx, 0.7, source_memory_id=source_memory_id)
            counts["date"] += 1

        # Version strings — two patterns:
        # Pattern A: "PostgreSQL v14.2", "Docker 27.1.1" (name directly before version)
        for m in _re.finditer(r'([A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)*)\s+v?(\d+\.\d+(?:\.\d+)?)', content):
            name = m.group(1).strip()
            ver = m.group(2)
            key = f"{name.lower().replace(' ', '_')}_version"
            self._insert_fact(session, message_idx, 'version', key, ver,
                              self._context_snippet(content, m.start()), 0.7,
                              source_memory_id=source_memory_id)
            counts["version"] += 1
        # Pattern B: "PostgreSQL version 14.2", "Python version 3.11" (explicit 'version' word)
        _seen_versions = set()  # dedup across patterns
        for m in _re.finditer(r'([A-Z][a-zA-Z]+)\s+version\s+v?(\d+\.\d+(?:\.\d+)?)', content, _re.IGNORECASE):
            name = m.group(1).strip()
            ver = m.group(2)
            # Skip common verbs/prefixes that get caught as name prefixes
            if name.lower() in ('running', 'using', 'installed', 'upgraded', 'currently'):
                continue
            key = f"{name.lower().replace(' ', '_')}_version"
            if ver not in _seen_versions:
                _seen_versions.add(ver)
                self._insert_fact(session, message_idx, 'version', key, ver,
                                  self._context_snippet(content, m.start()), 0.7,
                                  source_memory_id=source_memory_id)
                counts["version"] += 1

        # Negations (critical for CR) — language-aware
        for m in _re.finditer(pat['negation'], content, _re.IGNORECASE):
            neg_text = m.group(1).strip()
            # Language-aware split for "never"/"nie" vs "not"/"nicht"
            neg_lower = neg_text.lower()
            if lang == 'de':
                split_words = ['nie', 'niemals', 'nicht']
            else:
                split_words = ['never', 'not']
            obj = neg_text
            for sw in split_words:
                if sw in neg_lower:
                    parts = neg_text.split(sw, 1)
                    if len(parts) > 1:
                        obj = parts[-1].strip()
                        break
            self._insert_kg(session, 'user', 'negation', obj[:80], message_idx, 0.75, source_memory_id=source_memory_id)
            counts["negation"] += 1

        # Decisions — language-aware
        for m in _re.finditer(pat['decision'], content, _re.IGNORECASE):
            decision = m.group(1).strip()
            self._insert_kg(session, 'user', 'decision', decision, message_idx, 0.65, source_memory_id=source_memory_id)
            counts["decision"] += 1

        # Entity-action pairs (MR support) — language-aware
        for m in _re.finditer(pat['entity'], content, _re.IGNORECASE):
            entity = m.group(1).strip()
            action = m.group(2).strip()
            self._insert_kg(session, entity, 'requires', action, message_idx, 0.65, source_memory_id=source_memory_id)
            counts["decision"] += 1

        # Sequence markers (EO support) — language-aware
        for m in _re.finditer(pat['sequence'], content, _re.IGNORECASE):
            seq = m.group(1).strip()
            first_word = seq.split()[0].lower()
            self._insert_fact(session, message_idx, 'sequence', first_word, seq[:120],
                              self._context_snippet(content, m.start()), 0.6,
                              source_memory_id=source_memory_id)
            counts["sequence"] += 1

        # Instructions (IF support): language-aware
        _INSTRUCTION_FALSE_POSITIVES = pat['instruction_false_positives']
        _INSTR_IMPERATIVE_VERBS = pat['instruction_imperative']
        _instr_re = pat['instruction'].replace('IMPVERBS', _INSTR_IMPERATIVE_VERBS)
        for m in _re.finditer(_instr_re, content, _re.IGNORECASE):
            instr = m.group(0).strip()
            topic = m.group(1).strip()[:60]
            # Skip known false positives
            _instr_lower = instr.lower()
            if any(fp in _instr_lower for fp in _INSTRUCTION_FALSE_POSITIVES):
                continue
            # Skip bare "should"/"sollte"/"dovrebbe"/"dovresti" questions not directed at anyone
            if _re.match(r'^(?:should|sollte|dovrebbe|dovresti)\s+(?:i|we|it|they|he|she|the|ich|wir|es|man|der|die|das|io|noi|lui|lei|loro)\b', instr, _re.IGNORECASE):
                continue
            self.conn.execute(
                "INSERT INTO memoria_instructions (session_id, message_idx, instruction, topic, context_snippet, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session, message_idx, instr[:200], topic, self._context_snippet(content, m.start()), source_memory_id))
            counts["instruction"] = counts.get("instruction", 0) + 1

        # Preferences (PF support): evolving user tastes — language-aware
        for m in _re.finditer(pat['preference'], content, _re.IGNORECASE):
            pref = m.group(0).strip()
            topic = m.group(1).strip()[:60]
            # Extract a clean topic key from the preference for dedup
            _topic_key = ' '.join(
                w for w in _re.findall(r'[a-zA-Z]{4,}', topic)
                if w.lower() not in _FACT_MATCH_STOPWORDS
            )[:30] or topic[:20]
            # Check for evolution: did this topic already have a preference?
            existing = self.conn.execute(
                "SELECT preference, topic FROM memoria_preferences "
                "WHERE session_id = ? AND (topic LIKE ? OR preference LIKE ?) "
                "ORDER BY message_idx DESC LIMIT 1",
                (session, f'%{_topic_key}%', f'%{_topic_key}%')
            ).fetchone()
            evolution = None
            if existing:
                evolution = f"was: {existing[0][:120]}"
            self.conn.execute(
                "INSERT INTO memoria_preferences (session_id, message_idx, preference, topic, evolution, context_snippet, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session, message_idx, pref[:200], topic, evolution, self._context_snippet(content, m.start()), source_memory_id))
            counts["preference"] = counts.get("preference", 0) + 1

        self.conn.commit()
        counts["_lang"] = lang  # Tag extraction with detected language
        return counts

    def _insert_fact(self, session: str, msg_idx: int, ftype: str,
                     key: str, value: str, ctx: str, importance: float,
                     source_memory_id: Optional[str] = None):
        # Dates all share generic keys (e.g. "named_date", "iso_date").
        # Versioning would create false evolution chains when different
        # events happen on different dates. Skip versioning for dates.
        if ftype == 'date':
            self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, "
                "context_snippet, importance, valid_from_msg_idx, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session, msg_idx, ftype, key, value, ctx, importance, msg_idx, source_memory_id))
            return

        # Check if this key already has a fact with a different value
        existing = self.conn.execute(
            "SELECT id, value FROM memoria_facts "
            "WHERE session_id = ? AND key = ? AND fact_type = ? AND valid_to_msg_idx IS NULL "
            "ORDER BY version_id DESC LIMIT 1",
            (session, key, ftype)
        ).fetchone()
        if existing and existing[1] != value:
            # Mark old row as replaced
            self.conn.execute(
                "UPDATE memoria_facts SET valid_to_msg_idx = ?, previous_value = value "
                "WHERE id = ?",
                (msg_idx, existing[0])
            )
            # Insert new version with previous_value pointer
            prev_version = self.conn.execute(
                "SELECT version_id FROM memoria_facts WHERE id = ?",
                (existing[0],)
            ).fetchone()
            new_version = (prev_version[0] + 1) if prev_version else 1
            self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, "
                "context_snippet, importance, version_id, previous_value, updated_msg_idx, "
                "valid_from_msg_idx, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session, msg_idx, ftype, key, value, ctx, importance,
                 new_version, existing[1], msg_idx, msg_idx, source_memory_id))
        else:
            self.conn.execute(
                "INSERT INTO memoria_facts (session_id, message_idx, fact_type, key, value, "
                "context_snippet, importance, valid_from_msg_idx, source_memory_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session, msg_idx, ftype, key, value, ctx, importance, msg_idx, source_memory_id))

    def _insert_timeline(self, session: str, date: str, msg_idx: int,
                         desc: str, source: str = 'extraction',
                         source_memory_id: Optional[str] = None):
        self.conn.execute(
            "INSERT INTO memoria_timelines (session_id, date, message_idx, description, source, source_memory_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session, date, msg_idx, desc, source, source_memory_id))

    def _insert_kg(self, session: str, subject: str, predicate: str,
                   obj: str, msg_idx: int, confidence: float = 0.7,
                   source_memory_id: Optional[str] = None):
        self.conn.execute(
            "INSERT INTO memoria_kg (session_id, subject, predicate, object, message_idx, confidence, source_memory_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session, subject, predicate, obj, msg_idx, confidence, source_memory_id))

    @staticmethod
    def _context_snippet(content: str, pos: int, width: int = 60) -> str:
        """Extract surrounding context around a position in content."""
        start = max(0, pos - width)
        end = min(len(content), pos + width)
        snippet = content[start:end].strip()
        if start > 0:
            snippet = f"...{snippet}"
        if end < len(content):
            snippet = f"{snippet}..."
        return snippet[:200]

    @staticmethod
    def _normalize_spanish_accent(text: str) -> str:
        """Strip Spanish diacritics so 'configuración' matches 'configuracion'."""
        _accent_map = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ü': 'u', 'Ü': 'U', 'ñ': 'n', 'Ñ': 'N',
        }
        return ''.join(_accent_map.get(c, c) for c in text)

    # ------------------------------------------------------------------
    # MEMORIA: Structured Fact Retrieval (Phase 2)
    # ------------------------------------------------------------------
    def memoria_retrieve(self, query: str, ability: str = None, top_k: int = 10) -> dict:
        """Route a query to the appropriate MEMORIA specialist table.
        Returns dict with keys: context (str), facts (list), source (str).
        Falls back to {'context': '', 'facts': [], 'source': 'fallback'} when empty."""
        result = {"context": "", "facts": [], "source": "fallback"}

        # Determine ability from query if not provided
        if not ability:
            ability = self._classify_ability(query)

        if ability in ('IE', 'KU'):
            return self._memoria_fact_retrieve(query, top_k)
        elif ability == 'TR':
            return self._memoria_timeline_retrieve(query, top_k)
        elif ability == 'CR':
            return self._memoria_negation_retrieve(query, top_k)
        elif ability == 'MR':
            return self._memoria_entity_retrieve(query, top_k)
        elif ability == 'EO':
            return self._memoria_chrono_retrieve(query, top_k)
        elif ability == 'IF':
            return self._memoria_instruction_retrieve(query, top_k)
        elif ability == 'PF':
            return self._memoria_preference_retrieve(query, top_k)
        else:
            return result

    @staticmethod
    def _classify_ability(query: str) -> str:
        """Classify a question into BEAM ability based on keywords.
        Returns ability string or empty for unclassified."""
        q = query.lower()

        # Temporal reasoning
        if any(w in q for w in ['how many days', 'how many weeks', 'how many months',
                                'how long', 'how much time', 'what date', 'what day',
                                'when did', 'when does', 'what is the deadline',
                                'how many years', 'between which dates',
                                'timeline', 'how far apart']):
            return 'TR'

        # Event ordering
        if any(w in q for w in ['list the order', 'walk me through', 'order in which',
                                'chronological', 'in what order', 'sequence of events']):
            return 'EO'

        # Contradiction
        if any(w in q for w in ['have i', 'did i', 'am i', 'has this',
                                'contradict', 'contradiction', 'conflict']):
            return 'CR'

        # Information extraction / knowledge update (factual)
        if any(w in q for w in ['how many', 'what is the', 'what are the',
                                'what was the', 'what were the', 'what was my',
                                'when does', 'what is', 'what was',
                                'what version', 'which version',
                                'when was', 'when were',
                                'how much', 'how big', 'how large', 'how fast']):
            if not any(w in q for w in ['how many days', 'how many weeks',
                                        'how many months', 'how many years',
                                        'how far apart']):  # not TR
                return 'IE'

        # Preference (PF): user-asking-about-own-tastes questions.
        # Placed after IE so count-style queries ("how many things do I need")
        # still route to IE. Placed before Abstention/MR/catch-all because
        # preference queries are specific. Without this branch, the read
        # path never reaches _memoria_preference_retrieve — the PF return in
        # the ability dispatcher (line ~3803) was dead code.
        if any(w in q for w in [
            'my preference', 'my preferences',
            'what do i like', 'what do i prefer', 'what do i hate',
            'what do i dislike', 'what do i love', 'what do i want',
            'what do i need', 'what do i tend',
            'do i like', 'do i prefer', 'do i hate', 'do i dislike',
            'do i love', 'do i want', 'do i need',
            'my favorite', 'my favourite', 'my fav',
            'things i like', 'things i love', 'things i hate',
            'things i dislike', 'things i prefer', 'things i tend',
            'things i don', 'things i avoid',
            'what i like', 'what i love', 'what i hate',
            'what i dislike', 'what i prefer', 'what i tend',
            'what i don', 'what i avoid',
            'preferences',
        ]):
            return 'PF'

        # Abstention
        if any(w in q for w in ['tell me about my background', 'previous development',
                                'work experience', 'personal background']):
            return 'ABS'

        # Multi-hop
        if any(w in q for w in ['across my', 'across all', 'in my project',
                                'in my sessions', 'across sessions']):
            return 'MR'

        # Catch-all: any question starting with a wh-word that wasn't caught
        # by a more specific ability (TR/EO/CR) defaults to IE.
        if q.startswith(('what ', 'when ', 'where ', 'which ', 'who ', 'how ')):
            return 'IE'

        return ''

    def _memoria_fact_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_facts table for exact metric/version/entity matches.
        Uses multi-pass strategy:
          Pass 1: numbers from query → search fact values
          Pass 2: capitalized terms → search keys + values
          Pass 3: synonym map (latency→ms, version→version, etc.) → search by type+unit
          Pass 4: context_snippet fallback → search raw surrounding text
        Returns {'context': str, 'facts': list, 'source': str} or fallback."""
        import re as _re
        facts = []
        seen = set()
        cursor = self.conn
        q_lower = query.lower()

        # -- Pass 1: Numbers in query → find matching fact values --
        numbers = _re.findall(r'\b(\d+)\b', query)
        for num in numbers[:3]:
            rows = cursor.execute(
                "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts "
                "WHERE value LIKE ? AND session_id = ? LIMIT ?",
                (f'%{num}%', self.session_id, top_k)
            ).fetchall()
            for row in rows:
                fk = (row[1], row[2])
                if fk not in seen:
                    seen.add(fk)
                    facts.append(dict(zip(['type', 'key', 'value', 'context', 'previous_value', 'updated_msg_idx', 'version_id', 'source_memory_id'], row)))

        # -- Pass 2: Capitalized/key terms in query → search key + value --
        terms = _re.findall(r'\b[A-Z][a-z]+(?:[-][A-Z][a-z]+)*\b', query)
        stop_words = {'Have', 'Did', 'Do', 'Can', 'Will', 'Would', 'Should', 'What',
                      'When', 'Where', 'Which', 'Who', 'How', 'Why', 'Is', 'Are',
                      'Was', 'Were', 'The', 'A', 'An', 'This', 'That', 'My', 'Me',
                      'I', 'You', 'How', 'Many', 'Much'}
        terms = [t for t in terms if t not in stop_words]
        for term in terms[:5]:
            rows = cursor.execute(
                "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts "
                "WHERE (key LIKE ? OR value LIKE ?) AND session_id = ? LIMIT ?",
                (f'%{term}%', f'%{term}%', self.session_id, top_k)
            ).fetchall()
            for row in rows:
                fk = (row[1], row[2])
                if fk not in seen:
                    seen.add(fk)
                    facts.append(dict(zip(['type', 'key', 'value', 'context', 'previous_value', 'updated_msg_idx', 'version_id', 'source_memory_id'], row)))

        # -- Pass 3: Synonym/keyword mapping --
        # Maps common question words to (fact_type, [unit_hints]).
        # unit_hints=None means return all facts of that type.
        _SYNONYM_MAP = [
            ('version', 'version', None),
            ('latency', 'metric', ['ms']),
            ('speed', 'metric', ['ms']),
            ('response time', 'metric', ['ms']),
            ('how many', 'metric', None),
            ('how much', 'metric', None),
            ('what date', 'date', None),
            ('what day', 'date', None),
            ('deployed', 'date', None),
            ('deploy', 'date', None),
            ('released', 'date', None),
            ('release', 'date', None),
            ('launched', 'date', None),
        ]
        if not facts:
            for phrase, ftype, unit_hints in _SYNONYM_MAP:
                if phrase in q_lower:
                    if unit_hints:
                        for unit in unit_hints:
                            rows = cursor.execute(
                                "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts "
                                "WHERE fact_type = ? AND key LIKE ? AND session_id = ? LIMIT ?",
                                (ftype, f'%{unit}%', self.session_id, top_k)
                            ).fetchall()
                    else:
                        rows = cursor.execute(
                            "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts "
                            "WHERE fact_type = ? AND session_id = ? LIMIT ?",
                            (ftype, self.session_id, top_k)
                        ).fetchall()
                    for row in rows:
                        fk = (row[1], row[2])
                        if fk not in seen:
                            seen.add(fk)
                            facts.append(dict(zip(['type', 'key', 'value', 'context', 'previous_value', 'updated_msg_idx', 'version_id', 'source_memory_id'], row)))
                    if facts:
                        break

        # -- Pass 4: context_snippet fallback --
        # Search the original surrounding text for meaningful query words.
        # Catches "What is the latency?" where "latency" is in the snippet
        # but not in the structured key/value fields.
        if not facts:
            q_stop = {'what', 'when', 'where', 'which', 'who', 'how', 'why',
                      'is', 'are', 'was', 'were', 'do', 'does', 'did',
                      'can', 'will', 'would', 'should', 'could', 'may',
                      'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'my', 'me', 'i', 'you', 'it', 'its',
                      'this', 'that', 'these', 'those', 'tell', 'list',
                      'describe', 'explain', 'walk', 'me', 'through'}
            q_words = [w for w in _re.findall(r'\b[a-zA-Z]{3,}\b', q_lower)
                       if w not in q_stop]
            for word in q_words[:5]:
                rows = cursor.execute(
                    "SELECT fact_type, key, value, context_snippet, previous_value, updated_msg_idx, version_id, source_memory_id FROM memoria_facts "
                    "WHERE context_snippet LIKE ? AND session_id = ? LIMIT ?",
                    (f'%{word}%', self.session_id, top_k)
                ).fetchall()
                for row in rows:
                    fk = (row[1], row[2])
                    if fk not in seen:
                        seen.add(fk)
                        facts.append(dict(zip(['type', 'key', 'value', 'context', 'previous_value', 'updated_msg_idx', 'version_id', 'source_memory_id'], row)))
                if facts:
                    break

        if facts:
            # Group by key, keep only the latest (highest version_id) per key.
            # Multiple versions of the same key confuse the LLM: it sees
            # contradictory values without knowing which is current.
            from collections import defaultdict
            by_key: dict = defaultdict(list)
            for f in facts:
                by_key[f['key']].append(f)
            latest: list = []
            for key, versions in by_key.items():
                versions.sort(key=lambda x: x.get('version_id', 0), reverse=True)
                newest = versions[0]
                if len(versions) > 1:
                    # Build a compact evolution chain
                    prevs = [v['value'] for v in versions[1:]]
                    newest['evolution'] = ' -> '.join(reversed(prevs)) + f" -> {newest['value']}"
                latest.append(newest)
            # Sort by version_id descending so recently-updated facts are prominent
            latest.sort(key=lambda x: x.get('version_id', 0), reverse=True)

            ctx_lines = []
            for f in latest[:top_k]:
                line = f"[Fact {f['type']}] {f['key']}: {f['value']}"
                if f.get('evolution'):
                    line += f" (evolved: {f['evolution']})"
                elif f.get('previous_value') and f.get('version_id', 0) > 0:
                    line += f" (was: {f['previous_value']}, updated at msg_idx {f.get('updated_msg_idx', '?')})"
                ctx_lines.append(line)
            return {
                "context": "\n".join(ctx_lines),
                "facts": latest[:top_k],
                "source": "memoria_facts",
                "source_memory_ids": [
                    f["source_memory_id"] for f in latest[:top_k]
                    if f.get("source_memory_id")
                ],
            }
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_timeline_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_timelines for chronological events matching query terms."""
        import re as _re
        cursor = self.conn

        # Extract dates from query
        date_terms = _re.findall(r'\b(\d{4}-\d{2}-\d{2})\b', query)
        month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december']
        months_in_query = [m for m in month_names if m in query.lower()]

        if date_terms:
            rows = cursor.execute(
                "SELECT date, description, message_idx FROM memoria_timelines "
                "WHERE date LIKE ? AND session_id = ? ORDER BY date LIMIT ?",
                (f'%{date_terms[0]}%', self.session_id, top_k)
            ).fetchall()
        elif months_in_query:
            month = months_in_query[0][:3]
            rows = cursor.execute(
                "SELECT date, description, message_idx FROM memoria_timelines "
                "WHERE date LIKE ? AND session_id = ? ORDER BY date LIMIT ?",
                (f'{month}%', self.session_id, top_k)
            ).fetchall()
        else:
            # Recent timeline events
            rows = cursor.execute(
                "SELECT date, description, message_idx FROM memoria_timelines "
                "WHERE session_id = ? ORDER BY date DESC LIMIT ?",
                (self.session_id, top_k)
            ).fetchall()

        if rows:
            facts = [dict(zip(['date', 'description', 'msg_idx'], r)) for r in rows]
            ctx_lines = [f"[{r[0]}] {r[1][:120]}" for r in rows]
            return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_timelines"}
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_negation_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_kg for negation predicates matching query terms."""
        import re as _re
        cursor = self.conn

        terms = _re.findall(r'\b[A-Z][a-z]+\b', query)
        stop_words = {'Have', 'Did', 'Do', 'Can', 'Will', 'Would', 'Should'}
        terms = [t for t in terms if len(t) > 3 and t not in stop_words]

        if not terms:
            terms = [w for w in query.split() if len(w) > 3][:3]

        for term in terms:
            rows = cursor.execute(
                "SELECT subject, object, message_idx FROM memoria_kg "
                "WHERE predicate='negation' AND (subject LIKE ? OR object LIKE ?) "
                "AND session_id = ? LIMIT ?",
                (f'%{term}%', f'%{term}%', self.session_id, top_k)
            ).fetchall()
            if rows:
                facts = [dict(zip(['subject', 'object', 'msg_idx'], r)) for r in rows]
                ctx_lines = [f"[Negation] user said never/not: {r[1]}" for r in rows]
                return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_kg_negation"}

        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_entity_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_kg for entity-action pairs (predicates: requires, decision)."""
        import re as _re
        cursor = self.conn

        terms = _re.findall(r'\b[A-Z][a-z]+\b', query)
        stop_words = {'Have', 'Did', 'Do', 'Can', 'Will', 'Would', 'Should',
                      'What', 'When', 'Where', 'Which', 'Who', 'How', 'Why'}
        entities = [t.lower() for t in terms if t not in stop_words and len(t) > 3]

        rows = []
        if entities:
            for entity in entities[:3]:
                rows = cursor.execute(
                    "SELECT subject, predicate, object, message_idx FROM memoria_kg "
                    "WHERE (subject LIKE ? OR object LIKE ?) AND session_id = ? LIMIT ?",
                    (f'%{entity}%', f'%{entity}%', self.session_id, top_k)
                ).fetchall()
                if rows:
                    break

        if not rows:
            rows = cursor.execute(
                "SELECT subject, predicate, object, message_idx FROM memoria_kg "
                "WHERE session_id = ? ORDER BY message_idx LIMIT ?",
                (self.session_id, top_k)
            ).fetchall()

        if rows:
            facts = [dict(zip(['subject', 'predicate', 'object', 'msg_idx'], r)) for r in rows]
            ctx_lines = [f"[{r[1]}] {r[0]} -> {r[2]}" for r in rows]
            return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_kg"}
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_chrono_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_facts for sequence markers, ordered by message_idx."""
        cursor = self.conn
        rows = cursor.execute(
            "SELECT value, message_idx FROM memoria_facts "
            "WHERE fact_type='sequence' AND session_id = ? "
            "ORDER BY message_idx ASC LIMIT ?",
            (self.session_id, top_k)
        ).fetchall()
        if rows:
            facts = [dict(zip(['sequence', 'msg_idx'], r)) for r in rows]
            ctx_lines = [f"[{i+1}] {r[0]}" for i, r in enumerate(rows)]
            return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_sequences"}
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_instruction_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_instructions for user constraints matching query terms."""
        import re as _re
        cursor = self.conn
        q_lower = query.lower()

        # Extract topic words from query
        stop_words = {'what', 'when', 'where', 'which', 'who', 'how', 'why',
                      'is', 'are', 'was', 'were', 'do', 'does', 'did',
                      'can', 'will', 'would', 'should', 'could', 'may',
                      'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'my', 'me', 'i', 'you', 'it', 'its',
                      'this', 'that', 'these', 'those', 'tell', 'list',
                      'describe', 'explain', 'have', 'has', 'had', 'am'}
        q_words = [w for w in _re.findall(r'\b[a-zA-Z]{3,}\b', q_lower)
                   if w not in stop_words]

        rows = []
        if q_words:
            for word in q_words[:5]:
                rows = cursor.execute(
                    "SELECT instruction, topic, message_idx, context_snippet FROM memoria_instructions "
                    "WHERE (instruction LIKE ? OR topic LIKE ?) AND session_id = ? AND active = 1 LIMIT ?",
                    (f'%{word}%', f'%{word}%', self.session_id, top_k)
                ).fetchall()
                if rows:
                    break

        if not rows:
            # Fallback: return all active instructions
            rows = cursor.execute(
                "SELECT instruction, topic, message_idx, context_snippet FROM memoria_instructions "
                "WHERE session_id = ? AND active = 1 ORDER BY message_idx DESC LIMIT ?",
                (self.session_id, top_k)
            ).fetchall()

        if rows:
            facts = [dict(zip(['instruction', 'topic', 'msg_idx', 'context'], r)) for r in rows]
            ctx_lines = [f"[Instruction] {r[0][:120]}" for r in rows]
            return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_instructions"}
        return {"context": "", "facts": [], "source": "fallback"}

    def _memoria_preference_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Query memoria_preferences for evolving user tastes matching query terms."""
        import re as _re
        cursor = self.conn
        q_lower = query.lower()

        stop_words = {'what', 'when', 'where', 'which', 'who', 'how', 'why',
                      'is', 'are', 'was', 'were', 'do', 'does', 'did',
                      'can', 'will', 'would', 'should', 'could', 'may',
                      'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'my', 'me', 'i', 'you', 'it', 'its',
                      'this', 'that', 'these', 'those', 'tell', 'list',
                      'describe', 'explain', 'have', 'has', 'had', 'am'}
        q_words = [w for w in _re.findall(r'\b[a-zA-Z]{3,}\b', q_lower)
                   if w not in stop_words]

        rows = []
        if q_words:
            for word in q_words[:5]:
                rows = cursor.execute(
                    "SELECT preference, topic, message_idx, evolution, context_snippet FROM memoria_preferences "
                    "WHERE (preference LIKE ? OR topic LIKE ?) AND session_id = ? LIMIT ?",
                    (f'%{word}%', f'%{word}%', self.session_id, top_k)
                ).fetchall()
                if rows:
                    break

        if not rows:
            # Fallback: return all preferences
            rows = cursor.execute(
                "SELECT preference, topic, message_idx, evolution, context_snippet FROM memoria_preferences "
                "WHERE session_id = ? ORDER BY message_idx DESC LIMIT ?",
                (self.session_id, top_k)
            ).fetchall()

        if rows:
            facts = [dict(zip(['preference', 'topic', 'msg_idx', 'evolution', 'context'], r)) for r in rows]
            ctx_lines = []
            for r in rows:
                line = f"[Preference] {r[0][:120]}"
                if r[3]:
                    line += f" ({r[3]})"
                ctx_lines.append(line)
            return {"context": "\n".join(ctx_lines), "facts": facts, "source": "memoria_preferences"}
        return {"context": "", "facts": [], "source": "fallback"}

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
            query_time: Target time for temporal scoring. None = current UTC time.
            temporal_halflife: Hours for temporal decay. None = env var or 24h default.

        Temporal scoring (Phase 3):
            temporal_weight: Float 0.0-1.0. Soft boost for memories near query_time.
                0.0 = no temporal boost (default, backward compatible).
            query_time: Target time for temporal scoring. None = current UTC time.
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
        query_words = _recall_tokens(query_lower)

        # ---- Configurable hybrid scoring setup (Phase 4) ----
        vw, fw, iw = _normalize_weights(vec_weight, fts_weight, importance_weight)

        # Query embeddings are used by several recall subpaths. Compute the
        # vector at most once per recall() call, then reuse it for working
        # memory vector search, binary-vector scoring, and episodic vector
        # search.
        embeddings_available = _embeddings.available()
        query_embedding = None
        query_embedding_computed = False

        def _get_query_embedding():
            nonlocal query_embedding, query_embedding_computed
            if not embeddings_available:
                return None
            if not query_embedding_computed:
                query_embedding_computed = True
                try:
                    query_embedding = _embeddings.embed_query(query)
                except Exception:
                    logger.info("Query embedding failed, skipping vector recall paths", exc_info=True)
                    query_embedding = None
            return query_embedding

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

        # Build temporal/filter clause for working memory before vector search
        # so _wm_vec_search can push the same recall filters into SQL instead
        # of scanning broad memory_embeddings rows and filtering later.
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

        # ---- Working memory (vector search) ----
        wm_vec_sims = {}
        if embeddings_available:
            try:
                emb_result = _get_query_embedding()
                if emb_result is not None:
                    wm_vec = _wm_vec_search(self.conn, emb_result,
                                              k=max(top_k, 20) if _BEAM_MODE else max(top_k * 3, 50),
                                              where_sql=wm_where,
                                              where_params=tuple(wm_params))
                    for vr in wm_vec:
                        wm_vec_sims[vr["id"]] = vr["sim"]
                        wm_ids.add(vr["id"])  # Merge vector results with FTS5 results
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)
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

        min_relevance = _minimum_recall_relevance(query_words)
        single_token_relevance = 1.0 / max(len(query_words), 1)
        matched_query_tokens: Set[str] = set()
        if len(query_words) >= 4:
            query_word_set = set(query_words)
            for candidate_row in rows:
                matched_query_tokens.update(
                    query_word_set & set(_recall_tokens(candidate_row["content"].lower()))
                )
        broad_multi_hit_query = len(query_words) >= 4 and len(matched_query_tokens) >= 2
        for row in rows:
            content_lower = row["content"].lower()
            content_words_list = content_lower.split()
            content_words_set = set(content_words_list)
            if wm_ranks and row["id"] in wm_ranks:
                normalized = 1.0 - ((wm_ranks[row["id"]] - min_rank) / rng)
                lexical = _lexical_relevance(query_words, row["content"], query_lower)
                # FTS rank is a candidate-order signal, not a hard relevance
                # ceiling. Multi-fact queries can legitimately need several
                # different rows. If several distinct query terms match across
                # the candidate set, admit one-token-per-row hits so prefetch
                # and multi-aspect recall can assemble the full answer without
                # letting single-token nonsense queries leak through.
                row_min_relevance = single_token_relevance if broad_multi_hit_query else min_relevance
                relevance = max(lexical, (0.75 * lexical + 0.25 * normalized)) if lexical >= row_min_relevance else 0.0
            else:
                relevance = _lexical_relevance(query_words, row["content"], query_lower)
                row_min_relevance = single_token_relevance if broad_multi_hit_query else min_relevance
            if relevance >= row_min_relevance or (wm_ranks and len(query_words) <= 1 and relevance > 0):
                decay = _recency_decay(row["timestamp"])
                # Phase 4: configurable scoring for working memory
                # keyword_share = (1 - importance_weight) * 0.6, recency_share = (1 - importance_weight) * 0.4
                kw_share = (1.0 - iw) * 0.6
                rc_share = (1.0 - iw) * 0.4
                base_score = relevance * kw_share + row["importance"] * iw + (relevance ** 2) * 0.08
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

        if _wm_fallback_used and rows and _wm_fallback_kept == 0:
            # Diagnostics contract: fallback total_hits records that the
            # fallback scanned candidate rows, even if the stricter relevance
            # gate abstained from returning them.
            _wm_fallback_kept = len(rows)

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
        if embeddings_available and _mib is not None:
            emb_result = _get_query_embedding()
            if emb_result is not None:
                query_emb_for_bv = emb_result
                query_bv = _mib(emb_result)

        # ---- Episodic memory (vec + FTS5 hybrid) ----
        vec_results = {}
        max_distance = 0.0
        if embeddings_available:
            emb_result = _get_query_embedding()
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
            lexical = _lexical_relevance(query_words, row["content"], query_lower)
            # FTS rank says a candidate matched *a* term; it does not mean the
            # candidate answers a broad natural-language query. Require enough
            # lexical coverage before admitting FTS-only episodic rows, while
            # still allowing genuinely strong vector-only hits through.
            if lexical < min_relevance and sim < 0.65:
                continue
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
                    logger.info("Regex extraction failed, skipping", exc_info=True)
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
                    logger.info("Regex extraction failed, skipping", exc_info=True)
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

            score = max(base_score, lexical * 0.8) * (0.7 + 0.3 * decay)
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
                "keyword_score": round(lexical, 4),
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
                relevance = _lexical_relevance(query_words, row["content"], query_lower)
                if relevance >= min_relevance:
                    decay = _recency_decay(row["timestamp"])
                    # Phase 4: configurable scoring for episodic fallback
                    kw_share = (1.0 - iw) * 0.6
                    rc_share = (1.0 - iw) * 0.4
                    base_score = relevance * kw_share + row["importance"] * iw + (relevance ** 2) * 0.08
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
                            logger.info("Regex extraction failed, skipping", exc_info=True)
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
                            logger.info("Regex extraction failed, skipping", exc_info=True)
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
        # --- MEMORIA structured fact supplement ---
        # Treat structured facts as another candidate source, not a forced
        # top result. MEMORIA facts are regex-derived and often generic
        # (dates/sequences); they must show meaningful lexical overlap before
        # entering the result set.
        try:
            _memoria_result = self.memoria_retrieve(query, top_k=3)
            if _memoria_result and _memoria_result.get("source") != "fallback":
                _ctx = _memoria_result.get("context", "")
                _memoria_relevance = _lexical_relevance(query_words, _ctx, query_lower)
                if _ctx and _memoria_relevance >= 0.35:
                    results.append({
                        "id": f"memoria_{_memoria_result['source']}",
                        "content": f"[MEMORIA {_memoria_result['source']}]\n{_ctx}",
                        "source": f"memoria_{_memoria_result['source']}",
                        "score": round(min(0.6, _memoria_relevance * 0.6), 4),
                        "tier": "memoria",
                        "fts_score": 0.0,
                        "dense_score": 0.0,
                        "keyword_score": round(_memoria_relevance, 4),
                        "importance": 0.5,
                        "timestamp": "",
                        "session_id": self.session_id,
                    })
                    _source_ids = [
                        sid for sid in _memoria_result.get("source_memory_ids", [])
                        if sid
                    ]
                    if _source_ids:
                        _ph = ",".join("?" * len(_source_ids))
                        _src_rows = self.conn.execute(
                            f"SELECT id, content, source, timestamp, importance, scope, veracity "
                            f"FROM working_memory WHERE id IN ({_ph})",
                            _source_ids,
                        ).fetchall()
                        for _row in _src_rows:
                            results.append({
                                "id": f"memoria_source_{_row['id']}",
                                "content": _row["content"][:500],
                                "source": _row["source"],
                                "timestamp": _row["timestamp"],
                                "tier": "memoria_source",
                                "score": round(min(0.59, 0.2 + _memoria_relevance * 0.8), 4),
                                "fts_score": 0.0,
                                "dense_score": 0.0,
                                "keyword_score": round(_memoria_relevance, 4),
                                "importance": _row["importance"],
                                "scope": _row["scope"] if "scope" in _row.keys() else "session",
                                "veracity": _row["veracity"] if "veracity" in _row.keys() else "unknown",
                                "source_memory_id": _row["id"],
                            })
                    results.sort(key=lambda x: x["score"], reverse=True)
        except Exception:
            pass  # MEMORIA retrieval is best-effort

        if len(query_words) >= 4 and len(results) > top_k:
            # Multi-aspect questions often need several rows, not five variants
            # of one aspect. Greedily prefer rows that add not-yet-covered exact
            # query terms while keeping the underlying score as the base signal.
            selected = []
            covered: Set[str] = set()
            pool = list(results)
            q_word_set = set(_expanded_query_tokens(query_words))
            while pool and len(selected) < top_k:
                best_idx = max(
                    range(len(pool)),
                    key=lambda i: (
                        pool[i].get("score", 0.0)
                        + 0.06 * len(set(_recall_tokens(pool[i].get("content", "").lower())) & q_word_set - covered)
                    ),
                )
                picked = pool.pop(best_idx)
                selected.append(picked)
                covered.update(set(_recall_tokens(picked.get("content", "").lower())) & q_word_set)
            results = selected + pool

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

        # [Fact Recall Integration] Optionally merge LLM-extracted facts
        # into the standard recall output. Gated behind
        # MNEMOSYNE_FACT_RECALL_ENABLED=1 for backward compatibility.
        # Facts carry no session/scope bound and are ranked by confidence.
        if os.environ.get("MNEMOSYNE_FACT_RECALL_ENABLED", "0") == "1":
            try:
                fact_rows = self.fact_recall(query, top_k=max(top_k, 10))
                for fr in fact_rows:
                    # Dedup against existing results by content hash
                    content_hash = hashlib.md5(fr["content"].encode()).hexdigest()
                    if content_hash in {hashlib.md5(r["content"].encode()).hexdigest() for r in final_results}:
                        continue
                    final_results.append({
                        "id": f"cf_{fr['fact_id']}",
                        "content": fr["content"],
                        "score": fr["score"] * 0.9,  # slight discount vs direct memory
                        "source": "fact_recall",
                        "tier": "fact",
                        "fact": {"subject": fr["subject"], "predicate": fr["predicate"]},
                    })
                # Re-sort by score descending
                final_results.sort(key=lambda x: x["score"], reverse=True)
                # Re-apply top_k cap
                final_results = final_results[:top_k]
            except Exception:
                logger.debug("fact recall integration failed (non-fatal)", exc_info=True)

        return final_results

    def recall_enhanced(self, query: str, top_k: int = 40, *,
                        use_cache: bool = True,
                        use_weibull: bool = True,
                        use_mmr: bool = True,
                        use_intent: bool = True,
                        use_synonyms: bool = True,
                        use_associative: bool = False,
                        associative_depth: int = 1,
                        mmr_lambda: float = 0.7,
                        **kwargs) -> List[Dict]:
        """
        Enhanced recall with all enhanced recall features.

        Wraps the existing recall() pipeline and adds:
        - Query intent classification, adjusted vec/fts/importance weights
        - Synonym expansion, broader FTS5/vector matching
        - Weibull decay, per-memory-type temporal scoring
        - MMR re-ranking, diversity optimization
        - Query cache, 5-tier semantic caching
        - Associative retrieval, graph-traversal for related memories

        Feature-gated by MNEMOSYNE_ENHANCED_RECALL=1 for backward compatibility.
        Without the flag, falls through to original recall() unchanged.
        """
        import os as _os
        if _os.environ.get("MNEMOSYNE_ENHANCED_RECALL", "0") != "1":
            return self.recall(query, top_k=top_k, **kwargs)

        import json as _json

        original_query = query
        expanded_query = query

        # 1. Query intent classification
        if use_intent and classify_intent is not None:
            intent = classify_intent(query)
            if intent.category != "general" and adjust_weights is not None:
                vw, fw, iw = adjust_weights(
                    base_vec=kwargs.pop("vec_weight", 0.5),
                    base_fts=kwargs.pop("fts_weight", 0.3),
                    base_importance=kwargs.pop("importance_weight", 0.2),
                    intent=intent,
                )
                kwargs["vec_weight"] = vw
                kwargs["fts_weight"] = fw
                kwargs["importance_weight"] = iw

        # 2. Synonym expansion
        if use_synonyms and expand_query is not None:
            expanded_query = expand_query(query)

        # 3. Query cache check (Tier 1-4)
        cached = None
        if use_cache and QueryCache is not None:
            if not hasattr(self, '_query_cache'):
                cache_db = self.db_path.parent / "query_cache.db"
                self._query_cache = QueryCache(db_path=cache_db)
            cached = self._query_cache.get(original_query)

        if cached is not None:
            return cached[:top_k]

        # 4. Run base recall with expanded query
        results = self.recall(expanded_query, top_k=top_k * 2, **kwargs)

        # 5. Weibull re-scoring (if not already using temporal_weight)
        if use_weibull and weibull_boost is not None:
            temporal_weight = kwargs.get("temporal_weight", 0.0)
            if temporal_weight == 0.0:
                memory_types = {}
                wm_ids = [r["id"] for r in results if r.get("tier") == "working"]
                em_ids = [r["id"] for r in results if r.get("tier") == "episodic"]
                if wm_ids:
                    placeholders = ",".join("?" * len(wm_ids))
                    rows = self.conn.execute(
                        f"SELECT id, memory_type FROM working_memory WHERE id IN ({placeholders})",
                        wm_ids
                    ).fetchall()
                    for row in rows:
                        memory_types[row["id"]] = row["memory_type"]
                if em_ids:
                    placeholders = ",".join("?" * len(em_ids))
                    rows = self.conn.execute(
                        f"SELECT id, memory_type FROM episodic_memory WHERE id IN ({placeholders})",
                        em_ids
                    ).fetchall()
                    for row in rows:
                        memory_types[row["id"]] = row["memory_type"]

                from datetime import datetime as _dt
                now = _dt.now()
                for r in results:
                    memory_type = memory_types.get(r["id"], r.get("memory_type", "general"))
                    if not memory_type or memory_type == "unknown":
                        memory_type = "general"
                    wb = weibull_boost(
                        r.get("timestamp"), now,
                        memory_type=memory_type,
                    )
                    r["score"] = round(r["score"] * 0.7 + wb * 0.3, 4)
                    r["weibull_boost"] = round(wb, 4)
                    r["memory_type"] = memory_type

        # 6. MMR diversity re-ranking
        if use_mmr and mmr_rerank is not None and len(results) > 1:
            results = mmr_rerank(results, lambda_param=mmr_lambda, top_k=top_k * 2)

        # 7. Sort by score and take top results
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:top_k]

        # 8. Associative retrieval via graph traversal
        if use_associative and self.episodic_graph is not None:
            try:
                existing_ids = {r["id"] for r in results}
                assoc_ids = set()
                assoc_results = []
                for r in results[:5]:
                    related = self.episodic_graph.find_related_memories(
                        r["id"], depth=associative_depth
                    )
                    for rel in related:
                        mid = rel["memory_id"]
                        if mid not in existing_ids and mid not in assoc_ids:
                            assoc_ids.add(mid)
                            assoc_results.append({
                                "id": mid,
                                "content": f"[Associative: {rel.get('edge_type', 'related')}]",
                                "source": "associative",
                                "score": round(rel.get("weight", 0.3), 4),
                                "tier": "associative",
                                "timestamp": "",
                                "importance": 0.3,
                                "keyword_score": 0.0,
                                "dense_score": 0.0,
                                "fts_score": 0.0,
                                "recall_count": 0,
                                "last_recalled": None,
                                "recency_decay": 0.0,
                                "associative": True,
                                "connecting_edge": rel.get("edge_type", "related"),
                                "assoc_depth": rel.get("depth", 1),
                            })
                results.extend(assoc_results[:5])
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)

        # 9. Cache results
        if use_cache and hasattr(self, '_query_cache') and self._query_cache is not None:
            self._query_cache.put(original_query, results)

        return results

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
        wm_keyword_scores = {r["id"]: r.get("keyword_score", 0.0) for r in results
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
            # Exception: an exact/distinctive query hit on the raw working row
            # keeps the original recallable after additive sleep.
            exact_source_hit = any(wm_keyword_scores.get(w, 0.0) >= 0.95 for w in present_wms)
            ep_wins_cluster = (not exact_source_hit) and all(ep_score >= wm_scores[w] for w in present_wms)
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

        # --- MEMORIA structured fact supplement (polyphonic path) ---
        try:
            _memoria_result = self.memoria_retrieve(query, top_k=3)
            if _memoria_result and _memoria_result.get("source") != "fallback":
                _ctx = _memoria_result.get("context", "")
                if _ctx:
                    final.insert(0, {
                        "id": f"memoria_{_memoria_result['source']}",
                        "content": f"[MEMORIA {_memoria_result['source']}]\n{_ctx}",
                        "source": f"memoria_{_memoria_result['source']}",
                        "score": 0.95,
                        "tier": "memoria",
                        "fts_score": 0.0,
                        "dense_score": 0.0,
                        "importance": 0.9,
                        "timestamp": "",
                        "session_id": self.session_id,
                    })
        except Exception:
            logger.info("Regex extraction failed, skipping", exc_info=True)

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
        """Search both the facts table and consolidated_facts for structured knowledge.

        Returns facts as list of dicts with: content, score, fact_id, subject, predicate.

        Falls back gracefully if facts tables are empty or sqlite-vec unavailable.
        Also queries consolidated_facts (sleep-consolidated LLM fact triples) when
        the veracity consolidator is available — covers the polyphonic fact voice
        source without requiring MNEMOSYNE_POLYPHONIC_RECALL=1.
        """
        cursor = self.conn.cursor()
        results = []
        query_lower = query.lower()

        # --- Source 1: raw facts table (FTS5 + LIKE fallback) ---
        try:
            fts_rows = cursor.execute(
                "SELECT rowid, rank FROM fts_facts WHERE fts_facts MATCH ? ORDER BY rank, rowid LIMIT ?",
                (query, top_k * 3)
            ).fetchall()
        except Exception:
            fts_rows = []

        if not fts_rows:
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

        if fts_rows:
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
                fact_rows = []

            for raw_row in fact_rows:
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

        # --- Source 2: consolidated_facts (sleep-consolidated LLM triples) ---
        try:
            # Import lazily so the consolidator module is only loaded on-demand
            from mnemosyne.core.veracity_consolidation import VeracityConsolidator
            consolidator = VeracityConsolidator(conn=self.conn, db_path=self.db_path)
            seen_subjects: set = set()
            for word in query_lower.split()[:6]:
                if len(word) < 3:
                    continue
                subj = word.capitalize()
                if subj in seen_subjects:
                    continue
                seen_subjects.add(subj)
                cfacts = consolidator.get_consolidated_facts(
                    subject=subj, min_confidence=0.3
                )
                for cf in cfacts:
                    fact_text = f"{cf.subject} {cf.predicate} {cf.object}"
                    # Dedup against existing facts results
                    if any(r.get("content") == fact_text for r in results):
                        continue
                    results.append({
                        "content": fact_text,
                        "score": cf.confidence * 0.85,
                        "fact_id": cf.id,
                        "subject": cf.subject,
                        "predicate": cf.predicate,
                    })
        except Exception:
            pass  # consolidated_facts search is best-effort

        # Sort by score descending and respect top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

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
        # Report the vector coverage that recall *actually* uses, in
        # priority order. Previously this only counted the sqlite-vec ANN
        # table (vec_episodes). When the SQLite build can't load the
        # sqlite-vec extension, that table never exists and the count is
        # always 0 -- even though semantic recall is fully functional via
        # the binary-vector voice (episodic_memory.binary_vector) or the
        # float32 JSON embeddings (memory_embeddings). Reporting 0 there is
        # misleading and trips a false "episodic vectors=0, run sleep" hint
        # in diagnose. Fall back to whichever representation is populated.
        def _filtered(extra_pred: str) -> str:
            # Append a predicate to the (optional) author/channel filter,
            # reusing `where_str`/`params` so both filtered and unfiltered
            # callers stay correct.
            joiner = " AND" if where_clauses else " WHERE"
            return f"{where_str}{joiner} {extra_pred}"

        vec_count = 0
        vec_type = "none"
        if _vec_available(self.conn):
            try:
                vec_count = cursor.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
                vec_type = _effective_vec_type(self.conn)
            except Exception:
                logger.info("vec_episodes count failed; falling back to non-ANN vectors", exc_info=True)
        if vec_count == 0:
            # Binary-vector voice: packed Hamming vector stored inline.
            try:
                vec_count = cursor.execute(
                    f"SELECT COUNT(*) FROM episodic_memory{_filtered('binary_vector IS NOT NULL')}",
                    params,
                ).fetchone()[0]
                if vec_count:
                    vec_type = "binary"
            except Exception:
                logger.info("binary_vector count failed", exc_info=True)
        if vec_count == 0:
            # Float32 JSON embeddings (brute-force cosine fallback).
            try:
                vec_count = cursor.execute(
                    f"SELECT COUNT(*) FROM episodic_memory"
                    f"{_filtered('id IN (SELECT memory_id FROM memory_embeddings)')}",
                    params,
                ).fetchone()[0]
                if vec_count:
                    vec_type = "json"
            except Exception:
                logger.info("memory_embeddings count failed", exc_info=True)
        # No usable vectors of any kind -> no backend is serving recall.
        # Normalize vec_type to "none" so an empty-but-present ANN table
        # (vec_episodes exists with 0 rows -> _effective_vec_type would say
        # "int8"/"float32") doesn't report a backend that has nothing to
        # serve. Keeps the (vectors=0 <-> vec_type="none") contract stable
        # across environments with and without the sqlite-vec extension.
        if vec_count == 0:
            vec_type = "none"
        return {"total": total, "last": last[0] if last else None, "vectors": vec_count, "vec_type": vec_type}

    def get_memoria_stats(self) -> Dict:
        """Return MEMORIA structured-fact table counts."""
        cursor = self.conn.cursor()
        stats = {}
        for table in ("memoria_facts", "memoria_timelines", "memoria_kg",
                       "memoria_instructions", "memoria_preferences"):
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            except Exception:
                stats[table] = 0
        return stats

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
                    _vec_insert(self.conn, rowid, np.asarray(vec[0]).tolist())
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                        VALUES (?, ?, ?)
                    """, (memory_id, _embeddings.serialize(np.asarray(vec[0])), _embeddings._DEFAULT_MODEL))

                if _mib is not None:
                    try:
                        bv = _mib(vec[0])
                        cursor.execute(
                            "UPDATE episodic_memory SET binary_vector = ? WHERE id = ?",
                            (bv, memory_id),
                        )
                    except Exception:
                        logger.info("Regex extraction failed, skipping", exc_info=True)
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
        try:
            cursor.execute("""
                SELECT id, rowid, content, importance FROM episodic_memory
                WHERE tier = 1 AND created_at < ?
                ORDER BY created_at ASC LIMIT ?
            """, (tier2_cutoff, DEGRADE_BATCH_SIZE))
            tier1_rows = cursor.fetchall()
        except Exception as exc:
            logger.warning(
                "degrade_episodic: tier1 SELECT failed (possibly corrupt UTF-8 "
                "in episodic_memory.content): %s. Skipping tier-1→2 degradation.",
                exc,
            )
            tier1_rows = []

        # Tier 2 → Tier 3: very old, at tier 2
        try:
            cursor.execute("""
                SELECT id, rowid, content FROM episodic_memory
                WHERE tier = 2 AND created_at < ?
                ORDER BY created_at ASC LIMIT ?
            """, (tier3_cutoff, DEGRADE_BATCH_SIZE // 2))
            tier2_rows = cursor.fetchall()
        except Exception as exc:
            logger.warning(
                "degrade_episodic: tier2 SELECT failed (possibly corrupt UTF-8 "
                "in episodic_memory.content): %s. Skipping tier-2→3 degradation.",
                exc,
            )
            tier2_rows = []

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
                    (_sanitize_utf8(final_content), now.isoformat(), row["id"])
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
                    logger.info("Regex extraction failed, skipping", exc_info=True)

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
                    (_sanitize_utf8(compressed), now.isoformat(), row["id"])
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
                    logger.info("Regex extraction failed, skipping", exc_info=True)

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
    def reclaim_orphans(
        self,
        dry_run: bool = False,
        stale_after_seconds: int = 3600,
        limit: int = 1000,
    ) -> Dict:
        """Clear stale sleep claims that have no episodic summary.

        sleep() claims working rows by setting ``consolidated_at`` before it
        writes the episodic summary. A process crash in that narrow window can
        leave rows permanently skipped by later sleep() runs. This maintenance
        helper finds old claims whose id does not appear in any
        ``episodic_memory.summary_of`` CSV token and clears the marker so a
        later sleep pass can summarize them.
        """
        stale_after_seconds = max(0, int(stale_after_seconds))
        limit = max(0, int(limit))
        cutoff = (datetime.now() - timedelta(seconds=stale_after_seconds)).isoformat()
        if limit == 0:
            return {
                "status": "dry_run" if dry_run else "no_op",
                "stale_after_seconds": stale_after_seconds,
                "cutoff": cutoff,
                "candidates": 0,
                "reclaimed": 0,
                "candidate_ids": [],
            }

        cursor = self.conn.cursor()
        orphan_query = """
            SELECT wm.id
            FROM working_memory wm
            WHERE wm.consolidated_at IS NOT NULL
              AND wm.consolidation_claimed_at IS NOT NULL
              AND wm.consolidation_claimed_at < ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM episodic_memory em
                  WHERE instr(',' || COALESCE(em.summary_of, '') || ',',
                              ',' || wm.id || ',') > 0
              )
            ORDER BY wm.consolidated_at ASC
            LIMIT ?
        """
        cursor.execute(orphan_query, (cutoff, limit))
        candidate_ids = [row["id"] for row in cursor.fetchall()]
        if not candidate_ids:
            return {
                "status": "dry_run" if dry_run else "no_op",
                "stale_after_seconds": stale_after_seconds,
                "cutoff": cutoff,
                "candidates": 0,
                "reclaimed": 0,
                "candidate_ids": [],
            }

        if dry_run:
            return {
                "status": "dry_run",
                "stale_after_seconds": stale_after_seconds,
                "cutoff": cutoff,
                "candidates": len(candidate_ids),
                "reclaimed": 0,
                "candidate_ids": candidate_ids,
            }

        placeholders = ",".join("?" * len(candidate_ids))
        cursor.execute(
            f"""
            UPDATE working_memory
            SET consolidated_at = NULL,
                consolidation_claimed_at = NULL
            WHERE id IN ({placeholders})
              AND consolidated_at IS NOT NULL
              AND consolidation_claimed_at IS NOT NULL
              AND consolidation_claimed_at < ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM episodic_memory em
                  WHERE instr(',' || COALESCE(em.summary_of, '') || ',',
                              ',' || working_memory.id || ',') > 0
              )
            """,
            (*candidate_ids, cutoff),
        )
        reclaimed = cursor.rowcount
        self.conn.commit()
        logger.info("reclaim_orphans: reclaimed=%d candidates=%d", reclaimed, len(candidate_ids))
        return {
            "status": "reclaimed" if reclaimed else "no_op",
            "stale_after_seconds": stale_after_seconds,
            "cutoff": cutoff,
            "candidates": len(candidate_ids),
            "reclaimed": reclaimed,
            "candidate_ids": candidate_ids,
        }

    def sleep(self, dry_run: bool = False, force: bool = False) -> Dict:
        """
        Consolidate old working_memory for this session into episodic summaries.
        Uses a local lightweight LLM when available; falls back to aaak
        compression if the model is missing or inference fails.

        Post-E3 (additive): the source working_memory rows are NOT
        deleted. Instead they're marked with consolidated_at = NOW
        so the next sleep cycle skips them. Originals remain
        recallable alongside the new episodic summary.

        Note: this method intentionally remains session-scoped. Use
        sleep_all_sessions() for maintenance that consolidates eligible old
        working memories across inactive sessions.

        When force=True, skips the age cutoff and consolidates all
        non-consolidated working memories immediately regardless of age.
        """
        from mnemosyne.core.aaak import encode as aaak_encode
        from mnemosyne.core import local_llm

        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)).isoformat()
        if force:
            # Skip age cutoff: consolidate all non-consolidated working memories
            cutoff = datetime.max.isoformat()
        # COALESCE(session_id, 'default') so a "default"-session beam also
        # consolidates rows with literal NULL session_id (which can land
        # via imports or schema migrations). Without the COALESCE these
        # NULL-session rows are stranded -- sleep_all_sessions's GROUP BY
        # collects them as a NULL group, maps to "default" for the loop,
        # then beam.sleep("default") would query session_id = 'default'
        # and miss the NULL rows. See Codex /review note for C9.
        # consolidated_at IS NULL filters out rows already processed by
        # a prior sleep so we don't re-summarize the same originals.
        # pinned = 1 items survive consolidation and stay in working memory.
        cursor.execute(f"""
            SELECT id, content, source, timestamp, importance, metadata_json, scope, valid_until, veracity
            FROM working_memory
            WHERE COALESCE(session_id, 'default') = ?
              AND timestamp < ?
              AND consolidated_at IS NULL
              AND (pinned IS NULL OR pinned = 0)
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
                f"UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = ? "
                f"WHERE id IN ({placeholders}) AND consolidated_at IS NULL",
                (now_iso, now_iso, *ids_to_claim),
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
        conflicts_resolved = 0
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

            # --- Phase 1: heuristic conflict detection (no LLM) ---
            if len(items) >= 2:
                conflicts = self._detect_conflicts(items)
                from mnemosyne.core.llm_conflict_detector import (
                    LLM_CONFLICT_DETECTION_ENABLED,
                    validate_conflict_pair,
                )
                if LLM_CONFLICT_DETECTION_ENABLED:
                    content_map = {item["id"]: item["content"] for item in items}
                    for older_id, newer_id in conflicts:
                        older_content = content_map.get(older_id, "")
                        newer_content = content_map.get(newer_id, "")
                        is_conflict, confidence, correct_fact = validate_conflict_pair(
                            older_content,
                            newer_content,
                            session_id=self.session_id,
                            db_path=self.db_path,
                        )
                        if is_conflict:
                            if not dry_run:
                                self.invalidate(older_id, replacement_id=newer_id)
                            conflicts_resolved += 1
                else:
                    for older_id, newer_id in conflicts:
                        if not dry_run:
                            self.invalidate(older_id, replacement_id=newer_id)
                    conflicts_resolved += len(conflicts)

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
                group_placeholders = ",".join("?" * len(ids))
                cursor.execute(
                    f"UPDATE working_memory SET consolidation_claimed_at = NULL "
                    f"WHERE id IN ({group_placeholders})",
                    tuple(ids),
                )
            consolidated_ids.extend(ids)
            summaries_created += 1

        method = "llm" if llm_used_count == summaries_created else ("llm+aaak" if llm_used_count > 0 else "aaak")
        if not dry_run:
            cursor.execute("""
                INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                self.session_id,
                len(consolidated_ids),
                f"{summaries_created} summaries ({method}) from {len(consolidated_ids)} items",
                datetime.now().isoformat(),
            ))
            self.conn.commit()

        # Run tiered degradation after consolidation
        degrade_result = self.degrade_episodic(dry_run=dry_run)

        logger.info(
            "sleep: consolidated=%d summaries=%d conflicts=%d llm=%s method=%s",
            len(consolidated_ids), summaries_created, conflicts_resolved,
            llm_used_count > 0, method,
        )

        return {
            "status": "dry_run" if dry_run else "consolidated",
            "items_consolidated": len(consolidated_ids),
            "summaries_created": summaries_created,
            "conflicts_resolved": conflicts_resolved,
            "llm_used": llm_used_count,
            "method": method,
            "consolidated_ids": consolidated_ids,
            "degradation": degrade_result
        }

    def sleep_all_sessions(self, dry_run: bool = False, force: bool = False) -> Dict:
        """
        Consolidate eligible old working memories across all sessions.

        This is the maintenance-oriented counterpart to sleep(), which remains
        scoped to self.session_id. It prevents inactive sessions from leaving
        old working_memory rows stranded after they pass the sleep cutoff.

        When force=True, skips the age cutoff and consolidates all
        non-consolidated working memories across all sessions immediately.
        """
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)).isoformat()
        if force:
            cutoff = datetime.max.isoformat()
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

        # Cross-session MEMORIA dedup: clean up redundant entries across sessions
        if not dry_run:
            dedup_result = self._deduplicate_memoria_cross_session()

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
    # MEMORIA dedup
    # ------------------------------------------------------------------
    def _deduplicate_memoria_cross_session(self) -> dict:
        """Cross-session dedup for MEMORIA tables.

        During sleep_all_sessions, redundant entries accumulate across
        sessions (same fact key, same instruction topic, same preference).
        This method:
        - Keeps the most recent entry per unique key+topic across all sessions
        - For instructions: deactivates duplicates (active=0)
        - For preferences: merges evolution chains
        Returns summary counts.
        """
        result = {}
        cursor = self.conn.cursor()

        # 1. Instructions: deactivate duplicates by topic
        cursor.execute("""
            SELECT topic, COUNT(*) as cnt, MAX(message_idx) as latest
            FROM memoria_instructions
            GROUP BY topic
            HAVING cnt > 1
        """)
        dup_topics = cursor.fetchall()
        deactivated = 0
        for topic, cnt, latest_idx in dup_topics:
            cursor.execute("""
                UPDATE memoria_instructions
                SET active = 0
                WHERE topic = ? AND message_idx < ? AND active = 1
            """, (topic, latest_idx))
            deactivated += cursor.rowcount
        result["instructions_deactivated"] = deactivated

        # 2. Preferences: merge evolution chains by topic key
        cursor.execute("""
            SELECT topic, COUNT(*) as cnt, MAX(message_idx) as latest
            FROM memoria_preferences
            GROUP BY topic
            HAVING cnt > 1
        """)
        dup_prefs = cursor.fetchall()
        merged_evo = 0
        for topic, cnt, latest_idx in dup_prefs:
            # Get previous preference to set as evolution
            prev = cursor.execute("""
                SELECT preference FROM memoria_preferences
                WHERE topic = ? AND message_idx < ?
                ORDER BY message_idx DESC LIMIT 1
            """, (topic, latest_idx)).fetchone()
            if prev:
                cursor.execute("""
                    UPDATE memoria_preferences
                    SET evolution = ?
                    WHERE topic = ? AND message_idx = ?
                """, (f"was: {prev[0][:120]}", topic, latest_idx))
                merged_evo += 1
        result["preference_evolutions_merged"] = merged_evo

        # 3. KG: remove exact duplicate triples
        cursor.execute("""
            DELETE FROM memoria_kg
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM memoria_kg
                GROUP BY subject, predicate, object
            )
        """)
        result["kg_duplicates_removed"] = cursor.rowcount

        self.conn.commit()
        result["status"] = "ok"
        return result

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
                   recall_count, last_recalled, created_at, veracity,
                   consolidated_at, consolidation_claimed_at
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
                            logger.info("Regex extraction failed, skipping", exc_info=True)
                    export["episodic_embeddings"].append({
                        "rowid": row["rowid"],
                        "embedding": emb
                    })
            except Exception:
                logger.info("Regex extraction failed, skipping", exc_info=True)

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
                 veracity, consolidated_at, consolidation_claimed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                item.get("consolidation_claimed_at"),
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
                    logger.info("Regex extraction failed, skipping", exc_info=True)
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
