#!/usr/bin/env python3
"""
Auto-generate docs/api/ files from live code.

Usage:
    python3 scripts/generate-docs.py

Writes canonical copies to docs/api/ inside the mnemosyne repo.
Also writes to the website sibling repo (../mnemosyne-docs/src/) if present.
All website writes are optional — canonical copies are always written.
"""
from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------
# Tool schema definitions (25 real tools — verified against
# hermes_memory_provider/__init__.py::ALL_TOOL_SCHEMAS and
# mnemosyne/mcp_tools.py::_TOOL_HANDLERS, v3.6.0)
# ---------------------------------------------------------------
ALL_TOOL_SCHEMAS = [
    {"name": "mnemosyne_remember", "description": "Store a durable memory", "params": {"content": "string", "importance": "float=0.5", "source": "string=user", "scope": "string=session", "valid_until": "string=", "extract_entities": "bool=false", "extract": "bool=false", "metadata": "dict={}", "veracity": "string=unknown"}},
    {"name": "mnemosyne_recall", "description": "Search memories by vector+FTS hybrid ranking", "params": {"query": "string", "limit": "int=5", "temporal_weight": "float=0.0", "query_time": "string=", "temporal_halflife": "float=24", "vec_weight": "float=null", "fts_weight": "float=null", "importance_weight": "float=null"}},
    {"name": "mnemosyne_forget", "description": "Permanently delete a memory by ID", "params": {"memory_id": "string"}},
    {"name": "mnemosyne_get", "description": "Retrieve a single memory by ID (no search)", "params": {"memory_id": "string"}},
    {"name": "mnemosyne_update", "description": "Update content or importance of an existing memory", "params": {"memory_id": "string", "content": "string=", "importance": "float="}},
    {"name": "mnemosyne_validate", "description": "Attest, update, or invalidate a memory (collaborative ownership)", "params": {"memory_id": "string", "action": "enum[attest,update,invalidate,delete]", "validator": "string=", "new_content": "string=", "note": "string=", "bank": "enum[private,surface]=private"}},
    {"name": "mnemosyne_invalidate", "description": "Mark a memory as expired/superseded", "params": {"memory_id": "string", "replacement_id": "string="}},
    {"name": "mnemosyne_import", "description": "Import memories from JSON file or provider (Hindsight, Mem0)", "params": {"input_path": "string=", "provider": "string=", "api_key": "string=", "user_id": "string=", "agent_id": "string=", "base_url": "string=", "dry_run": "bool=false", "channel_id": "string=", "force": "bool=false"}},
    {"name": "mnemosyne_export", "description": "Export all memories to a JSON file", "params": {"output_path": "string"}},
    {"name": "mnemosyne_diagnose", "description": "PII-safe diagnostics: deps, DB state, vector readiness, vec_working coverage", "params": {"repair_vec_working": "bool=false", "dry_run": "bool=false"}},
    {"name": "mnemosyne_stats", "description": "Memory statistics: working count, episodic count, BEAM tiers", "params": {}},
    {"name": "mnemosyne_sleep", "description": "Run consolidation cycle (compress old working memories)", "params": {"all_sessions": "bool=false", "dry_run": "bool=false"}},
    {"name": "mnemosyne_triple_add", "description": "Add a fact triple to the knowledge graph", "params": {"subject": "string", "predicate": "string", "object": "string", "valid_from": "string="}},
    {"name": "mnemosyne_triple_query", "description": "Query the temporal knowledge graph", "params": {"subject": "string=", "predicate": "string=", "object": "string="}},
    {"name": "mnemosyne_graph_link", "description": "Declare a semantic edge between two memories", "params": {"source_id": "string", "target_id": "string", "relationship": "string", "weight": "float=0.5"}},
    {"name": "mnemosyne_graph_query", "description": "Multi-hop BFS traversal from a seed memory", "params": {"seed_memory_id": "string", "max_hops": "int=2", "edge_type": "string=", "min_weight": "float=0.0"}},
    {"name": "mnemosyne_shared_remember", "description": "Store compact cross-agent surface memory", "params": {"content": "string", "kind": "string=meta", "importance": "float=0.8", "veracity": "string=unknown", "metadata": "dict"}},
    {"name": "mnemosyne_shared_recall", "description": "Search only the shared surface DB", "params": {"query": "string", "limit": "int=5"}},
    {"name": "mnemosyne_shared_forget", "description": "Delete one working shared-surface memory by ID", "params": {"memory_id": "string"}},
    {"name": "mnemosyne_shared_stats", "description": "Return shared surface DB path and counts", "params": {}},
    {"name": "mnemosyne_scratchpad_write", "description": "Write a temporary note to the scratchpad", "params": {"content": "string"}},
    {"name": "mnemosyne_scratchpad_read", "description": "Read the scratchpad entries", "params": {}},
    {"name": "mnemosyne_scratchpad_clear", "description": "Clear all scratchpad entries", "params": {}},
    {"name": "mnemosyne_remember_canonical", "description": "Store an owner-scoped canonical fact (single source of truth)", "params": {"content": "string", "importance": "float=0.5", "veracity": "string=unknown", "source": "string=user", "valid_until": "string="}},
    {"name": "mnemosyne_recall_canonical", "description": "Recall canonical facts by slot, category, or substring", "params": {"slot": "string=", "category": "string=", "substring": "string=", "limit": "int=5", "include_history": "bool=false"}},
]

# NOTE: 25 MCP tools with real handler implementations in mcp_tools.py.
# mnemosyne_end was removed — it had no handler, no schema in the provider,
# and would raise ValueError("Unknown tool") if called.
# mnemosyne_triple_end exists in Hermes provider schemas but has no MCP handler;
# it is NOT included here since it would fail at MCP runtime.

# ---------------------------------------------------------------
# Config schema (env vars — verified against actual os.environ.get
# calls in beam.py, the hermes_memory_provider, and integrations, v3.6.0)
# ---------------------------------------------------------------
CONFIG_ENTRIES = [
    # ── Storage & Paths ──
    {"key": "MNEMOSYNE_DATA_DIR", "env": "MNEMOSYNE_DATA_DIR", "default": "~/.hermes/mnemosyne/data", "desc": "Directory for database, logs, models, and stats"},
    {"key": "MNEMOSYNE_HOME", "env": "MNEMOSYNE_HOME", "default": "~/.hermes/mnemosyne", "desc": "Override home directory for all Mnemosyne data"},
    {"key": "MNEMOSYNE_SHARED_DB_PATH", "env": "MNEMOSYNE_SHARED_DB_PATH", "default": "data/shared/mnemosyne.db", "desc": "SQLite path for shared surface memory DB"},
    {"key": "MNEMOSYNE_BLOB_DIR", "env": "MNEMOSYNE_BLOB_DIR", "default": "", "desc": "Directory for blob storage (content sanitizer output)"},
    {"key": "MNEMOSYNE_AUTO_MIGRATE", "env": "MNEMOSYNE_AUTO_MIGRATE", "default": "1", "desc": "Auto-migrate DB schema on startup (set to 0 to disable)"},

    # ── Working Memory ──
    {"key": "MNEMOSYNE_WM_MAX_ITEMS", "env": "MNEMOSYNE_WM_MAX_ITEMS", "default": "10000", "desc": "Maximum items in working memory before eviction"},
    {"key": "MNEMOSYNE_WM_TTL_HOURS", "env": "MNEMOSYNE_WM_TTL_HOURS", "default": "24", "desc": "Hours before working memory entries expire"},

    # ── Episodic & Recall ──
    {"key": "MNEMOSYNE_EP_LIMIT", "env": "MNEMOSYNE_EP_LIMIT", "default": "50000", "desc": "Max episodic memories returned per recall"},
    {"key": "MNEMOSYNE_SP_MAX", "env": "MNEMOSYNE_SP_MAX", "default": "1000", "desc": "Maximum scratchpad entries"},
    {"key": "MNEMOSYNE_RECENCY_HALFLIFE", "env": "MNEMOSYNE_RECENCY_HALFLIFE", "default": "168", "desc": "Recency decay halflife in hours (default: 1 week)"},
    {"key": "MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS", "env": "MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS", "default": "24", "desc": "Temporal voice halflife for time-weighted recall scoring"},

    # ── Sleep & Consolidation ──
    {"key": "MNEMOSYNE_SLEEP_BATCH", "env": "MNEMOSYNE_SLEEP_BATCH", "default": "5000", "desc": "Batch size for sleep consolidation"},
    {"key": "MNEMOSYNE_AUTO_SLEEP_ENABLED", "env": "MNEMOSYNE_AUTO_SLEEP_ENABLED", "default": "false", "desc": "Enable automatic sleep consolidation (Hermes provider, default off)"},
    {"key": "MNEMOSYNE_SESSION_END_TIMEOUT", "env": "MNEMOSYNE_SESSION_END_TIMEOUT", "default": "15", "desc": "Max seconds for session-end sleep (Hermes provider)"},
    {"key": "MNEMOSYNE_AUTO_SLEEP_TIMEOUT", "env": "MNEMOSYNE_AUTO_SLEEP_TIMEOUT", "default": "5", "desc": "Max seconds for auto-sleep cycle"},
    {"key": "MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT", "env": "MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT", "default": "2", "desc": "Max seconds to drain LLM queue on shutdown"},
    {"key": "MNEMOSYNE_SLEEP_PROMPT", "env": "MNEMOSYNE_SLEEP_PROMPT", "default": "", "desc": "Custom prompt for sleep LLM consolidation"},

    # ── Tiered Degradation (BEAM) ──
    {"key": "MNEMOSYNE_TIER2_DAYS", "env": "MNEMOSYNE_TIER2_DAYS", "default": "30", "desc": "Days before memories enter Tier 2 (compressed)"},
    {"key": "MNEMOSYNE_TIER3_DAYS", "env": "MNEMOSYNE_TIER3_DAYS", "default": "180", "desc": "Days before memories enter Tier 3 (summary only)"},
    {"key": "MNEMOSYNE_TIER1_WEIGHT", "env": "MNEMOSYNE_TIER1_WEIGHT", "default": "1.0", "desc": "Scoring weight for Tier 1 (fresh) memories"},
    {"key": "MNEMOSYNE_TIER2_WEIGHT", "env": "MNEMOSYNE_TIER2_WEIGHT", "default": "0.5", "desc": "Scoring weight for Tier 2 (compressed) memories"},
    {"key": "MNEMOSYNE_TIER3_WEIGHT", "env": "MNEMOSYNE_TIER3_WEIGHT", "default": "0.25", "desc": "Scoring weight for Tier 3 (summary) memories"},
    {"key": "MNEMOSYNE_DEGRADE_BATCH", "env": "MNEMOSYNE_DEGRADE_BATCH", "default": "100", "desc": "Batch size for tiered degradation pass"},
    {"key": "MNEMOSYNE_SMART_COMPRESS", "env": "MNEMOSYNE_SMART_COMPRESS", "default": "true", "desc": "Use LLM for smart tiered compression instead of truncation"},
    {"key": "MNEMOSYNE_TIER3_MAX_CHARS", "env": "MNEMOSYNE_TIER3_MAX_CHARS", "default": "300", "desc": "Max character length for Tier 3 summaries"},

    # ── Veracity Weights ──
    {"key": "MNEMOSYNE_STATED_WEIGHT", "env": "MNEMOSYNE_STATED_WEIGHT", "default": "1.0", "desc": "Veracity weight for stated (user-asserted) memories"},
    {"key": "MNEMOSYNE_INFERRED_WEIGHT", "env": "MNEMOSYNE_INFERRED_WEIGHT", "default": "0.7", "desc": "Veracity weight for inferred (LLM-extracted) memories"},
    {"key": "MNEMOSYNE_TOOL_WEIGHT", "env": "MNEMOSYNE_TOOL_WEIGHT", "default": "0.5", "desc": "Veracity weight for tool-returned memories"},
    {"key": "MNEMOSYNE_IMPORTED_WEIGHT", "env": "MNEMOSYNE_IMPORTED_WEIGHT", "default": "0.6", "desc": "Veracity weight for externally imported memories"},
    {"key": "MNEMOSYNE_UNKNOWN_WEIGHT", "env": "MNEMOSYNE_UNKNOWN_WEIGHT", "default": "0.8", "desc": "Veracity weight for memories with unknown source"},

    # ── Vector & Embeddings ──
    {"key": "MNEMOSYNE_VEC_TYPE", "env": "MNEMOSYNE_VEC_TYPE", "default": "int8", "desc": "Vector storage format (int8, float32, float16, binary)"},
    {"key": "MNEMOSYNE_EMBEDDING_MODEL", "env": "MNEMOSYNE_EMBEDDING_MODEL", "default": "BAAI/bge-small-en-v1.5", "desc": "fastembed model for vector embeddings"},
    {"key": "MNEMOSYNE_EMBEDDING_DIM", "env": "MNEMOSYNE_EMBEDDING_DIM", "default": "384", "desc": "Override embedding vector dimension"},
    {"key": "MNEMOSYNE_EMBEDDING_API_KEY", "env": "MNEMOSYNE_EMBEDDING_API_KEY", "default": "", "desc": "API key for cloud embedding provider"},
    {"key": "MNEMOSYNE_EMBEDDING_API_URL", "env": "MNEMOSYNE_EMBEDDING_API_URL", "default": "https://openrouter.ai/api/v1", "desc": "API endpoint for cloud embeddings"},
    {"key": "MNEMOSYNE_NO_EMBEDDINGS", "env": "MNEMOSYNE_NO_EMBEDDINGS", "default": "false", "desc": "Disable dense vector retrieval entirely"},
    {"key": "MNEMOSYNE_EMBEDDINGS_VIA_API", "env": "MNEMOSYNE_EMBEDDINGS_VIA_API", "default": "false", "desc": "Force cloud API mode for embeddings"},
    {"key": "MNEMOSYNE_EMBEDDING_FALLBACK_MODEL", "env": "MNEMOSYNE_EMBEDDING_FALLBACK_MODEL", "default": "", "desc": "Local fastembed model for API fallback (v3.6.0)"},

    # ── Hybrid Scoring ──
    {"key": "MNEMOSYNE_VEC_WEIGHT", "env": "MNEMOSYNE_VEC_WEIGHT", "default": "0.5", "desc": "Vector similarity weight in hybrid ranking"},
    {"key": "MNEMOSYNE_FTS_WEIGHT", "env": "MNEMOSYNE_FTS_WEIGHT", "default": "0.3", "desc": "Full-text search weight in hybrid ranking"},
    {"key": "MNEMOSYNE_IMPORTANCE_WEIGHT", "env": "MNEMOSYNE_IMPORTANCE_WEIGHT", "default": "0.2", "desc": "Importance score weight in hybrid ranking"},

    # ── LLM Backends ──
    {"key": "MNEMOSYNE_LLM_ENABLED", "env": "MNEMOSYNE_LLM_ENABLED", "default": "true", "desc": "Enable LLM summarization during sleep consolidation"},
    {"key": "MNEMOSYNE_LLM_BASE_URL", "env": "MNEMOSYNE_LLM_BASE_URL", "default": "", "desc": "OpenAI-compatible API base URL for remote LLM"},
    {"key": "MNEMOSYNE_LLM_API_KEY", "env": "MNEMOSYNE_LLM_API_KEY", "default": "", "desc": "API key for remote LLM endpoint"},
    {"key": "MNEMOSYNE_LLM_MODEL", "env": "MNEMOSYNE_LLM_MODEL", "default": "", "desc": "Model identifier for remote LLM calls"},
    {"key": "MNEMOSYNE_LLM_MAX_TOKENS", "env": "MNEMOSYNE_LLM_MAX_TOKENS", "default": "2048", "desc": "Max output tokens per LLM summary"},
    {"key": "MNEMOSYNE_LLM_N_CTX", "env": "MNEMOSYNE_LLM_N_CTX", "default": "2048", "desc": "Context window size for local LLM"},
    {"key": "MNEMOSYNE_LLM_N_THREADS", "env": "MNEMOSYNE_LLM_N_THREADS", "default": "4", "desc": "CPU threads for local LLM inference"},
    {"key": "MNEMOSYNE_LLM_REPO", "env": "MNEMOSYNE_LLM_REPO", "default": "openbmb/MiniCPM5-1B-GGUF", "desc": "HuggingFace repo for GGUF model"},
    {"key": "MNEMOSYNE_LLM_FILE", "env": "MNEMOSYNE_LLM_FILE", "default": "MiniCPM5-1B-Q4_K_M.gguf", "desc": "GGUF filename for local LLM"},
    {"key": "MNEMOSYNE_LLM_FALLBACK_BASE_URL", "env": "MNEMOSYNE_LLM_FALLBACK_BASE_URL", "default": "", "desc": "Fallback API URL when primary remote LLM fails"},
    {"key": "MNEMOSYNE_LLM_FALLBACK_API_KEY", "env": "MNEMOSYNE_LLM_FALLBACK_API_KEY", "default": "", "desc": "API key for fallback LLM endpoint"},
    {"key": "MNEMOSYNE_LLM_FALLBACK_MODELS", "env": "MNEMOSYNE_LLM_FALLBACK_MODELS", "default": "", "desc": "Comma-separated fallback model names to try"},
    {"key": "MNEMOSYNE_FORCE_LOCAL", "env": "MNEMOSYNE_FORCE_LOCAL", "default": "false", "desc": "Skip remote LLM and use local model directly"},
    {"key": "MNEMOSYNE_LLM_CONFLICT_DETECTION", "env": "MNEMOSYNE_LLM_CONFLICT_DETECTION", "default": "false", "desc": "Enable LLM-based conflict detection during sleep"},
    {"key": "MNEMOSYNE_CONFLICT_LLM_BASE_URL", "env": "MNEMOSYNE_CONFLICT_LLM_BASE_URL", "default": "", "desc": "API base URL for conflict detection LLM"},
    {"key": "MNEMOSYNE_CONFLICT_LLM_API_KEY", "env": "MNEMOSYNE_CONFLICT_LLM_API_KEY", "default": "", "desc": "API key for conflict detection LLM"},
    {"key": "MNEMOSYNE_CONFLICT_LLM_MODEL", "env": "MNEMOSYNE_CONFLICT_LLM_MODEL", "default": "", "desc": "Model for conflict detection LLM calls"},
    {"key": "MNEMOSYNE_HOST_LLM_ENABLED", "env": "MNEMOSYNE_HOST_LLM_ENABLED", "default": "false", "desc": "Route consolidation through host-provided LLM adapter"},
    {"key": "MNEMOSYNE_HOST_LLM_MODEL", "env": "MNEMOSYNE_HOST_LLM_MODEL", "default": "", "desc": "Model override for host LLM adapter"},
    {"key": "MNEMOSYNE_HOST_LLM_PROVIDER", "env": "MNEMOSYNE_HOST_LLM_PROVIDER", "default": "", "desc": "Provider override for host LLM adapter (e.g. openai-codex)"},
    {"key": "MNEMOSYNE_HOST_LLM_N_CTX", "env": "MNEMOSYNE_HOST_LLM_N_CTX", "default": "32000", "desc": "Context window budget when using host LLM adapter"},
    {"key": "MNEMOSYNE_EXTRACTION_MODEL", "env": "MNEMOSYNE_EXTRACTION_MODEL", "default": "google/gemini-2.5-flash", "desc": "Model for entity/fact extraction via OpenRouter"},
    {"key": "MNEMOSYNE_EXTRACTION_PROMPT", "env": "MNEMOSYNE_EXTRACTION_PROMPT", "default": "", "desc": "Custom prompt for entity/fact extraction"},

    # ── Feature Flags & A/B Toggles ──
    {"key": "MNEMOSYNE_BEAM_OPTIMIZATIONS", "env": "MNEMOSYNE_BEAM_OPTIMIZATIONS", "default": "false", "desc": "Enable BEAM benchmark optimizations (feature flag)"},
    {"key": "MNEMOSYNE_ENHANCED_RECALL", "env": "MNEMOSYNE_ENHANCED_RECALL", "default": "0", "desc": "Enable enhanced recall pipeline (fact+graph+episodic fusion)"},
    {"key": "MNEMOSYNE_FACT_RECALL_ENABLED", "env": "MNEMOSYNE_FACT_RECALL_ENABLED", "default": "0", "desc": "Enable fact-based recall (backward compat alias)"},
    {"key": "MNEMOSYNE_POLYPHONIC_RECALL", "env": "MNEMOSYNE_POLYPHONIC_RECALL", "default": "0", "desc": "Enable polyphonic recall engine (multi-voice fusion)"},
    {"key": "MNEMOSYNE_PROACTIVE_LINKING", "env": "MNEMOSYNE_PROACTIVE_LINKING", "default": "0", "desc": "Enable proactive cross-memory linking on insertion"},
    {"key": "MNEMOSYNE_GRAPH_BONUS", "env": "MNEMOSYNE_GRAPH_BONUS", "default": "1", "desc": "A/B toggle: graph traversal bonus in recall scoring"},
    {"key": "MNEMOSYNE_FACT_BONUS", "env": "MNEMOSYNE_FACT_BONUS", "default": "1", "desc": "A/B toggle: fact match bonus in recall scoring"},
    {"key": "MNEMOSYNE_BINARY_BONUS", "env": "MNEMOSYNE_BINARY_BONUS", "default": "1", "desc": "A/B toggle: binary vector bonus in recall scoring"},
    {"key": "MNEMOSYNE_LENIENT_FACT_MATCH", "env": "MNEMOSYNE_LENIENT_FACT_MATCH", "default": "false", "desc": "Use substring instead of exact match for fact recall"},
    {"key": "MNEMOSYNE_VERACITY_MULTIPLIER", "env": "MNEMOSYNE_VERACITY_MULTIPLIER", "default": "1", "desc": "A/B toggle: apply veracity multiplier to recall scores"},
    {"key": "MNEMOSYNE_CROSS_TIER_DEDUP", "env": "MNEMOSYNE_CROSS_TIER_DEDUP", "default": "1", "desc": "A/B toggle: cross-tier deduplication in BEAM recall"},
    {"key": "MNEMOSYNE_VOICE_VECTOR", "env": "MNEMOSYNE_VOICE_VECTOR", "default": "1", "desc": "A/B toggle: polyphonic vector voice (Phase 3d)"},
    {"key": "MNEMOSYNE_VOICE_GRAPH", "env": "MNEMOSYNE_VOICE_GRAPH", "default": "1", "desc": "A/B toggle: polyphonic graph voice (Phase 3b)"},
    {"key": "MNEMOSYNE_VOICE_FACT", "env": "MNEMOSYNE_VOICE_FACT", "default": "1", "desc": "A/B toggle: polyphonic fact voice (Phase 3a)"},
    {"key": "MNEMOSYNE_VOICE_TEMPORAL", "env": "MNEMOSYNE_VOICE_TEMPORAL", "default": "1", "desc": "A/B toggle: polyphonic temporal voice (Phase 3c)"},
    {"key": "MNEMOSYNE_BEAM_MODE", "env": "MNEMOSYNE_BEAM_MODE", "default": "false", "desc": "Enable BEAM mode (polyphonic recall engine extension)"},
    {"key": "MNEMOSYNE_USE_CAVEMAN", "env": "MNEMOSYNE_USE_CAVEMAN", "default": "false", "desc": "Use caveman/AAAK encoding fallback for consolidation"},

    # ── Hermes Provider ──
    {"key": "MNEMOSYNE_SYNC_ROLES", "env": "MNEMOSYNE_SYNC_ROLES", "default": "user,assistant", "desc": "Conversation roles to sync into memory"},
    {"key": "MNEMOSYNE_SKIP_CONTEXTS", "env": "MNEMOSYNE_SKIP_CONTEXTS", "default": "cron,flush,subagent,background,skill_loop", "desc": "Comma-separated context names to skip"},
    {"key": "MNEMOSYNE_SYNC_TURN_USER_LIMIT", "env": "MNEMOSYNE_SYNC_TURN_USER_LIMIT", "default": "500", "desc": "Max chars of user content synced per turn (0=no limit)"},
    {"key": "MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "env": "MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "default": "800", "desc": "Max chars of assistant content synced per turn (0=no limit)"},
    {"key": "MNEMOSYNE_PREFETCH_CONTENT_CHARS", "env": "MNEMOSYNE_PREFETCH_CONTENT_CHARS", "default": "0", "desc": "Truncate prefetched content to N chars (0=no truncation)"},
    {"key": "MNEMOSYNE_PREFETCH_PROFILE", "env": "MNEMOSYNE_PREFETCH_PROFILE", "default": "general", "desc": "Prefetch profile name (general, coding, etc.)"},

    # ── MCP & Identity ──
    {"key": "MNEMOSYNE_MCP_TOKEN", "env": "MNEMOSYNE_MCP_TOKEN", "default": "", "desc": "Bearer token for MCP server auth (required for remote deployment)"},
    {"key": "MNEMOSYNE_MCP_BANK", "env": "MNEMOSYNE_MCP_BANK", "default": "default", "desc": "Default MCP bank name for tool operations"},
    {"key": "MNEMOSYNE_AUTHOR_ID", "env": "MNEMOSYNE_AUTHOR_ID", "default": "", "desc": "Identifier for the author/agent creating memories"},
    {"key": "MNEMOSYNE_AUTHOR_TYPE", "env": "MNEMOSYNE_AUTHOR_TYPE", "default": "", "desc": "Type of author (user, assistant, system, etc.)"},
    {"key": "MNEMOSYNE_CHANNEL_ID", "env": "MNEMOSYNE_CHANNEL_ID", "default": "", "desc": "Channel/session identifier for memory scoping"},
    {"key": "MNEMOSYNE_DEFAULT_OWNER", "env": "MNEMOSYNE_DEFAULT_OWNER", "default": "default", "desc": "Default owner for canonical facts and shared memory"},

    # ── SHMR (Semantic Hierarchical Memory Reorganization) ──
    {"key": "MNEMOSYNE_SHMR_BATCH_SIZE", "env": "MNEMOSYNE_SHMR_BATCH_SIZE", "default": "50", "desc": "Max memories per SHMR reorganization batch"},
    {"key": "MNEMOSYNE_SHMR_MAX_ITERATIONS", "env": "MNEMOSYNE_SHMR_MAX_ITERATIONS", "default": "3", "desc": "Max SHMR clustering iterations per batch"},
    {"key": "MNEMOSYNE_SHMR_SIMILARITY_THRESHOLD", "env": "MNEMOSYNE_SHMR_SIMILARITY_THRESHOLD", "default": "0.70", "desc": "Cosine similarity threshold for memory clustering"},
    {"key": "MNEMOSYNE_SHMR_HARMONY_THRESHOLD", "env": "MNEMOSYNE_SHMR_HARMONY_THRESHOLD", "default": "0.60", "desc": "Harmony threshold for cluster merging"},
    {"key": "MNEMOSYNE_SHMR_MODEL", "env": "MNEMOSYNE_SHMR_MODEL", "default": "", "desc": "LLM model for SHMR summarization (empty = use default)"},
    {"key": "MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE", "env": "MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE", "default": "2", "desc": "Minimum memories to form a SHMR cluster"},
    {"key": "MNEMOSYNE_SHMR_TEMPERATURE", "env": "MNEMOSYNE_SHMR_TEMPERATURE", "default": "0.2", "desc": "LLM temperature for SHMR summarization"},
]

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _version() -> str:
    import re
    init_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mnemosyne", "__init__.py")
    if os.path.exists(init_path):
        with open(init_path) as f:
            m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read())
            if m:
                return m.group(1)
    return "3.4.0"  # fallback

def _write_tool_schema_mdx(tools, version):
    lines = [
        "---",
        f'title: "MCP Tool Schema"',
        f"version: {version}",
        "tool_count: {}".format(len(tools)),
        'generated_at: "auto"',
        "---",
        "",
        f"# MCP Tool Schema (v{version})",
        "",
        f"Mnemosyne exposes **{len(tools)} MCP tools** for memory management, retrieval, and diagnostics.",
        "",
        "---",
        "",
    ]
    for i, t in enumerate(tools, 1):
        params = t.get("params", {})
        lines.append(f"### {i}. `{t['name']}`")
        lines.append(t['description'])
        lines.append("")
        if params:
            lines.append("| Parameter | Type | Required |")
            lines.append("|-----------|------|----------|")
            for pname, pdef in params.items():
                ptype, required = pdef, "yes"
                if "=" in pdef:
                    ptype, default = pdef.split("=", 1)
                    required = "no (default: {})".format(default)
                lines.append("| `{}` | `{}` | {} |".format(pname, ptype, required))
            lines.append("")
        else:
            lines.append("*No parameters*")
            lines.append("")
    return "\n".join(lines)

def _write_config_mdx(entries, version):
    lines = [
        "---",
        f'title: "Configuration"',
        f"version: {version}",
        'generated_at: "auto"',
        "---",
        "",
        "# Configuration",
        "",
        "Mnemosyne is configured entirely through environment variables. No config files, no YAML, no JSON.",
        "",
        "---",
        "",
        "| Variable | Default | Description |",
        "|----------|---------|-------------|",
    ]
    for e in entries:
        default = e.get("default", "")
        if default:
            default = "`{}`".format(default)
        else:
            default = "*(required)*"
        lines.append("| `{}` | {} | {} |".format(e["key"], default, e["desc"]))
    return "\n".join(lines)

def _inject_config_table(page_path, table_html):
    """Inject a generated config table into an existing page.mdx (for website only)."""
    if not os.path.exists(page_path):
        print("  ⚠️  config page not found at {} — skipping injection".format(page_path))
        return
    with open(page_path, 'r') as f:
        content = f.read()
    start_marker = "{/* GENERATED_CONFIG_TABLE */}"
    end_marker = "{/* /GENERATED_CONFIG_TABLE */}"
    new_block = start_marker + "\n" + table_html + "\n" + end_marker

    # Strip ALL existing GENERATED_CONFIG_TABLE blocks (handle duplicates, handle HTML and MDX comment markers)
    for start_v, end_v in [
        ("<!-- GENERATED_CONFIG_TABLE -->", "<!-- /GENERATED_CONFIG_TABLE -->"),
        ("{/* GENERATED_CONFIG_TABLE */}", "{/* /GENERATED_CONFIG_TABLE */}"),
    ]:
        while start_v in content and end_v in content:
            start_idx = content.index(start_v)
            end_idx = content.index(end_v, start_idx) + len(end_v)
            content = content[:start_idx] + content[end_idx:].lstrip('\n')

    # Insert a single clean block
    content = content.rstrip() + "\n\n" + new_block + "\n"
    with open(page_path, 'w') as f:
        f.write(content)

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    version = _version()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Canonical copies: always write to docs/api/ inside the mnemosyne repo
    canonical = os.path.join(repo_root, "docs", "api")
    os.makedirs(canonical, exist_ok=True)

    # Tool schema
    schema_mdx = _write_tool_schema_mdx(ALL_TOOL_SCHEMAS, version)
    with open(os.path.join(canonical, "tool-schema.mdx"), "w") as f:
        f.write(schema_mdx)
    print("✓ tool-schema.mdx ({} tools) → docs/api/".format(len(ALL_TOOL_SCHEMAS)))

    # Config
    config_mdx = _write_config_mdx(CONFIG_ENTRIES, version)
    with open(os.path.join(canonical, "configuration.mdx"), "w") as f:
        f.write(config_mdx)
    print("✓ configuration.mdx ({} keys) → docs/api/".format(len(CONFIG_ENTRIES)))

    # Website sibling repo (optional — gracefully skip if missing)
    docs_sibling = os.path.normpath(os.path.join(repo_root, "..", "mnemosyne-docs", "src"))
    
    if os.path.isdir(docs_sibling):
        # Tool schema
        www_tool = os.path.join(docs_sibling, "app/(docs)", "api", "tool-schema", "page.mdx")
        os.makedirs(os.path.dirname(www_tool), exist_ok=True)
        with open(www_tool, "w") as f:
            f.write(schema_mdx)
        print("✓ tool-schema page → website sibling")

        # Config table injection
        www_config = os.path.join(docs_sibling, "app/(docs)", "getting-started", "configuration", "page.mdx")
        if os.path.isfile(www_config):
            _inject_config_table(www_config, "\n".join(
                "| `{}` | {} | {} |".format(e["key"], e.get("default", ""), e["desc"])
                for e in CONFIG_ENTRIES
            ))
            print("✓ config table → website sibling")
        else:
            print("⚠️  website config page not found — skip")
    else:
        print("⚠️  website sibling not found — skip (CI runner ok)")

    print("")
    print("Done. Canonical docs written to docs/api/")

if __name__ == "__main__":
    main()
