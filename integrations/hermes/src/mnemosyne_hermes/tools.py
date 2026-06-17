"""Tool schemas exposed by the Mnemosyne memory provider.

28 tools: remember, recall, shared_remember, shared_recall, shared_forget,
shared_stats, sleep, stats, invalidate, validate, get, triple_add, triple_query,
triple_end, remember_canonical, recall_canonical, scratchpad_write, scratchpad_read,
scratchpad_clear, export, update, forget, import, diagnose, graph_query, graph_link,
sync_push, sync_pull, sync_status.
"""

REMEMBER_SCHEMA = {
    "name": "mnemosyne_remember",
    "description": (
        "Store a durable memory in Mnemosyne. Use for ANY fact, preference, "
        "identity, insight, or context that should persist across sessions. Higher importance "
        "(0.0-1.0) surfaces the memory more often. Use scope='global' for user-level "
        "facts; scope='session' for conversation-specific context. Use valid_until "
        "(ISO date YYYY-MM-DD) for time-bound facts. Use extract_entities=True to "
        "extract named entities for fuzzy recall (e.g. 'Abdias' and 'Abdias J.' will match). "
        "Use extract=True to also pull subject-predicate-object fact triples via LLM "
        "for fact-aware recall. Use veracity to tag confidence: 'stated' for direct "
        "user assertions, 'tool' for deterministic tool output, 'inferred' for derived "
        "guesses; 'unknown' (default) gets no recall boost."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "importance": {"type": "number", "description": "Importance 0.0-1.0. Default 0.5.", "default": 0.5},
            "source": {"type": "string", "description": "Source tag: preference, fact, insight, identity, task, etc.", "default": "user"},
            "scope": {"type": "string", "description": "'session' (default) or 'global'.", "default": "session"},
            "valid_until": {"type": "string", "description": "Optional expiry date YYYY-MM-DD.", "default": ""},
            "extract_entities": {"type": "boolean", "description": "Extract named entities for fuzzy recall. Default False.", "default": False},
            "extract": {"type": "boolean", "description": "Extract subject-predicate-object fact triples via LLM for fact-aware recall. Default False.", "default": False},
            "metadata": {"type": "object", "description": "Optional dict of additional fields (source_doc, tags, page, etc.). Default empty.", "default": {}},
            "veracity": {"type": "string", "description": "Confidence label: 'stated' | 'inferred' | 'tool' | 'imported' | 'unknown'. Default 'unknown'.", "default": "unknown"},
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "mnemosyne_recall",
    "description": (
        "Search Mnemosyne for relevant memories. Uses hybrid ranking: by default "
        "50% vector similarity + 30% FTS5 text rank + 20% importance + optional "
        "temporal boost. Tune the per-query weights via vec_weight, fts_weight, "
        "importance_weight (omit to use environment defaults). Returns ranked results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language query."},
            "limit": {"type": "integer", "description": "Max results. Default 5.", "default": 5},
            "temporal_weight": {
                "type": "number",
                "description": "How much to boost recent memories (0.0 = ignore time, 0.2 = mild recency bias, 0.5 = strong recency bias). Default 0.0.",
                "default": 0.0,
            },
            "query_time": {
                "type": "string",
                "description": "ISO timestamp to treat as 'now' for temporal scoring. Default is current time.",
                "default": "",
            },
            "temporal_halflife": {
                "type": "number",
                "description": "Hours until temporal boost decays by half. Default 24. Lower = faster decay.",
                "default": 24,
            },
            "vec_weight": {
                "type": "number",
                "description": "Vector similarity weight in hybrid scoring. Omit (or pass null) to use MNEMOSYNE_VEC_WEIGHT env var or built-in default 0.5.",
            },
            "fts_weight": {
                "type": "number",
                "description": "Full-text search weight in hybrid scoring. Omit (or pass null) to use MNEMOSYNE_FTS_WEIGHT env var or built-in default 0.3.",
            },
            "importance_weight": {
                "type": "number",
                "description": "Importance score weight in hybrid scoring. Omit (or pass null) to use MNEMOSYNE_IMPORTANCE_WEIGHT env var or built-in default 0.2.",
            },
        },
        "required": ["query"],
    },
}

SHARED_REMEMBER_SCHEMA = {
    "name": "mnemosyne_shared_remember",
    "description": (
        "Store compact cross-agent surface memory in a dedicated shared Mnemosyne DB. "
        "Use only for stable user/system/workflow metadata or general preferences. "
        "Normal mnemosyne_remember writes stay private."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Surface memory content to store."},
            "kind": {"type": "string", "description": "meta | preference | correction | identity", "default": "meta"},
            "importance": {"type": "number", "description": "Importance 0.0-1.0. Default 0.8.", "default": 0.8},
            "veracity": {"type": "string", "description": "stated | inferred | tool | imported | unknown", "default": "unknown"},
            "metadata": {"type": "object", "description": "Optional metadata object.", "default": {}},
        },
        "required": ["content"],
    },
}

SHARED_RECALL_SCHEMA = {
    "name": "mnemosyne_shared_recall",
    "description": "Search only the shared Mnemosyne surface DB.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
}

SHARED_FORGET_SCHEMA = {
    "name": "mnemosyne_shared_forget",
    "description": "Delete one working shared-surface memory by exact ID.",
    "parameters": {
        "type": "object",
        "properties": {"memory_id": {"type": "string"}},
        "required": ["memory_id"],
    },
}

SHARED_STATS_SCHEMA = {
    "name": "mnemosyne_shared_stats",
    "description": "Return shared surface DB path and counts.",
    "parameters": {"type": "object", "properties": {}},
}

SLEEP_SCHEMA = {
    "name": "mnemosyne_sleep",
    "description": (
        "Run the Mnemosyne consolidation cycle. Compresses old working memories "
        "into episodic summaries. Call after long sessions or when memory feels stale. "
        "Set all_sessions=true to consolidate eligible old working memories across inactive sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "all_sessions": {
                "type": "boolean",
                "description": "If true, consolidate eligible old working memories across all sessions instead of only the current session.",
                "default": False,
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, report what would be consolidated without writing changes.",
                "default": False,
            },
            "force": {
                "type": "boolean",
                "description": "If true, skip the age threshold and consolidate all non-consolidated working memories immediately.",
                "default": False,
            },
        },
    },
}

STATS_SCHEMA = {
    "name": "mnemosyne_stats",
    "description": "Return Mnemosyne memory statistics: working count, episodic count, BEAM tiers.",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}

INVALIDATE_SCHEMA = {
    "name": "mnemosyne_invalidate",
    "description": (
        "Mark a memory as expired or superseded. Provide memory_id from recall results. "
        "Optionally provide replacement_id to chain old to new."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of memory to invalidate."},
            "replacement_id": {"type": "string", "description": "Optional new memory that replaces this one.", "default": ""},
        },
        "required": ["memory_id"],
    },
}

VALIDATE_SCHEMA = {
    "name": "mnemosyne_validate",
    "description": (
        "Attest, update, or invalidate a memory the caller did not necessarily author. "
        "Supports collaborative ownership: any agent can validate any memory in either "
        "the private bank or the shared surface. The original author is preserved; "
        "validator + validated_at are updated to record the most recent attester. "
        "A 3-entry ring buffer keeps lightweight history. "
        "Actions: 'attest' (confirm correctness), 'update' (replace content), "
        "'invalidate' (mark superseded), 'delete' (remove)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of memory to validate."},
            "action": {
                "type": "string",
                "enum": ["attest", "update", "invalidate", "delete"],
                "description": "What kind of validation to record.",
            },
            "validator": {
                "type": "string",
                "description": "Agent identifier performing the validation. Defaults to the caller's agent_identity if not set.",
                "default": "",
            },
            "new_content": {
                "type": "string",
                "description": "New content (only used with action='update').",
                "default": "",
            },
            "note": {
                "type": "string",
                "description": "Optional reason or evidence for this validation.",
                "default": "",
            },
            "bank": {
                "type": "string",
                "enum": ["private", "surface"],
                "description": "Which bank holds the memory. Default 'private'.",
                "default": "private",
            },
        },
        "required": ["memory_id", "action"],
    },
}

GET_SCHEMA = {
    "name": "mnemosyne_get",
    "description": (
        "Retrieve a single memory by its primary key. Pure read, no side effects. "
        "No semantic search. Returns the exact memory with the given ID or None. "
        "Use this when you already know the memory ID from a previous recall response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The memory ID to retrieve."},
        },
        "required": ["memory_id"],
    },
}

TRIPLE_ADD_SCHEMA = {
    "name": "mnemosyne_triple_add",
    "description": (
        "Add a temporal fact triple (subject, predicate, object) to the knowledge graph. "
        "Example: ('user', 'prefers', 'neovim'). Use for structured relationships."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "valid_from": {"type": "string", "description": "ISO date YYYY-MM-DD", "default": ""},
        },
        "required": ["subject", "predicate", "object"],
    },
}

TRIPLE_QUERY_SCHEMA = {
    "name": "mnemosyne_triple_query",
    "description": "Query the temporal knowledge graph for facts matching subject/predicate/object patterns.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "default": ""},
            "predicate": {"type": "string", "default": ""},
            "object": {"type": "string", "default": ""},
        },
    },
}

REMEMBER_CANONICAL_SCHEMA = {
    "name": "mnemosyne_remember_canonical",
    "description": (
        "Store a CANONICAL (single-source-of-truth) self-fact for the current "
        "profile. Each (category, name) slot holds exactly one current value: "
        "restating the same body is a no-op, and a new body supersedes the old "
        "one (kept as history). Use for stable identity cards — name, voice, "
        "stable preferences, relationships — that must not contradict themselves "
        "over time. Scoped privately to this profile. For relational facts use "
        "mnemosyne_triple_add; for episodic recall use mnemosyne_remember."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Slot group, e.g. 'identity', 'voice', 'preference'"},
            "name": {"type": "string", "description": "Slot key within the category, e.g. 'name', 'pronouns'"},
            "body": {"type": "string", "description": "The authoritative free-text value for this slot"},
            "source": {"type": "string", "description": "Optional provenance label", "default": ""},
            "confidence": {"type": "number", "description": "Optional 0..1 confidence", "default": 1.0},
        },
        "required": ["category", "name", "body"],
    },
}

RECALL_CANONICAL_SCHEMA = {
    "name": "mnemosyne_recall_canonical",
    "description": (
        "Read CANONICAL self-facts for the current profile. With category+name: "
        "return the single authoritative value for that slot. With category "
        "only: list that category's slots. With query: substring-search the "
        "profile's canonical values. With nothing: list all canonical slots. "
        "Set include_history=true to also return superseded versions of a slot."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "default": ""},
            "name": {"type": "string", "default": ""},
            "query": {"type": "string", "description": "Substring search across the profile's canonical values", "default": ""},
            "include_history": {"type": "boolean", "description": "Include superseded versions (requires category+name)", "default": False},
            "limit": {"type": "integer", "description": "Max results for query/list modes", "default": 10},
        },
    },
}

SCRATCHPAD_WRITE_SCHEMA = {
    "name": "mnemosyne_scratchpad_write",
    "description": "Write a temporary note to the Mnemosyne scratchpad.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["content"],
    },
}

SCRATCHPAD_READ_SCHEMA = {
    "name": "mnemosyne_scratchpad_read",
    "description": "Read the Mnemosyne scratchpad entries.",
    "parameters": {"type": "object", "properties": {}},
}

SCRATCHPAD_CLEAR_SCHEMA = {
    "name": "mnemosyne_scratchpad_clear",
    "description": "Clear all entries from the Mnemosyne scratchpad.",
    "parameters": {"type": "object", "properties": {}},
}

EXPORT_SCHEMA = {
    "name": "mnemosyne_export",
    "description": "Export all Mnemosyne memories to a JSON file for backup or migration.",
    "parameters": {
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "description": "File path to write the export JSON (e.g., /tmp/mnemosyne_backup.json)",
            },
        },
        "required": ["output_path"],
    },
}

UPDATE_SCHEMA = {
    "name": "mnemosyne_update",
    "description": "Update the content or importance of an existing memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of the memory to update"},
            "content": {"type": "string", "description": "New content for the memory (optional)"},
            "importance": {"type": "number", "description": "New importance from 0.0 to 1.0 (optional)"},
        },
        "required": ["memory_id"],
    },
}

FORGET_SCHEMA = {
    "name": "mnemosyne_forget",
    "description": "Permanently delete a memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of the memory to delete"},
        },
        "required": ["memory_id"],
    },
}

IMPORT_SCHEMA = {
    "name": "mnemosyne_import",
    "description": "Import Mnemosyne memories from a JSON file or another memory provider (Hindsight, Mem0). Idempotent by default.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "File path to read the export JSON from (for file imports)",
            },
            "provider": {
                "type": "string",
                "description": "Provider to import from: 'hindsight', 'mem0'. Requires api_key.",
            },
            "api_key": {
                "type": "string",
                "description": "API key for the source provider (can also be set via env var)",
            },
            "user_id": {
                "type": "string",
                "description": "Filter imported memories by user ID (provider-specific)",
            },
            "agent_id": {
                "type": "string",
                "description": "Filter imported memories by agent ID (provider-specific)",
            },
            "base_url": {
                "type": "string",
                "description": "Base URL for self-hosted provider instances",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, validate and transform but don't write any memories",
                "default": False,
            },
            "channel_id": {
                "type": "string",
                "description": "Channel to assign imported memories to",
            },
            "force": {
                "type": "boolean",
                "description": "If true, overwrite existing records instead of skipping",
                "default": False,
            },
        },
    },
}

DIAGNOSE_SCHEMA = {
    "name": "mnemosyne_diagnose",
    "description": "Run PII-safe diagnostics on Mnemosyne installation. Checks dependencies, database state, vector search readiness, and optional vec_working migration coverage. Never includes memory content or API keys.",
    "parameters": {
        "type": "object",
        "properties": {
            "repair_vec_working": {
                "type": "boolean",
                "description": "If true, idempotently backfill missing vec_working rows from memory_embeddings.",
                "default": False,
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true with repair_vec_working, report what would be repaired without writing.",
                "default": False,
            },
        },
    },
}

GRAPH_QUERY_SCHEMA = {
    "name": "mnemosyne_graph_query",
    "description": "Traverse the memory graph to find memories related to a seed memory. Uses multi-hop BFS through graph_edges with optional edge_type and min_weight filtering.",
    "parameters": {
        "type": "object",
        "properties": {
            "seed_memory_id": {
                "type": "string",
                "description": "Memory ID to start traversal from",
            },
            "max_hops": {
                "type": "integer",
                "description": "Maximum traversal depth (default: 2)",
                "default": 2,
            },
            "edge_type": {
                "type": "string",
                "description": "Filter by edge type (empty = all types, e.g. 'ctx', 'rel', 'syn', 'references', 'caused', 'supersedes')",
                "default": "",
            },
            "min_weight": {
                "type": "number",
                "description": "Minimum edge weight threshold (0.0 to 1.0, default: 0.0 = no filter)",
                "default": 0.0,
            },
        },
        "required": ["seed_memory_id"],
    },
}

GRAPH_LINK_SCHEMA = {
    "name": "mnemosyne_graph_link",
    "description": "Declare a semantic edge between two memories in the graph. Use this to explicitly link related memories so graph traversal finds them.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "description": "Source memory ID",
            },
            "target_id": {
                "type": "string",
                "description": "Target memory ID",
            },
            "relationship": {
                "type": "string",
                "description": "Relationship label (e.g. 'references', 'caused', 'supersedes', 'related_to')",
            },
            "weight": {
                "type": "number",
                "description": "Edge weight from 0.0 to 1.0 (default: 0.5)",
                "default": 0.5,
            },
        },
        "required": ["source_id", "target_id", "relationship"],
    },
}

# Sync tool schemas (v0.2.0 — bidirectional sync with optional encryption)
SYNC_PUSH_SCHEMA = {
    "name": "mnemosyne_sync_push",
    "description": (
        "Push local memory changes to a remote Mnemosyne sync server. "
        "Only events created since the last sync are sent. Requires a "
        "configured remote sync server (configured via config.yaml or "
        "MNEMOSYNE_SYNC_REMOTE env var)."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

SYNC_PULL_SCHEMA = {
    "name": "mnemosyne_sync_pull",
    "description": (
        "Pull remote memory changes from the configured Mnemosyne sync server. "
        "Applies incoming events locally with timestamp + importance conflict "
        "resolution."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

SYNC_STATUS_SCHEMA = {
    "name": "mnemosyne_sync_status",
    "description": (
        "Show Mnemosyne sync status: device ID, last cursor, event count, "
        "remote URL, and encryption state."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

ALL_TOOL_SCHEMAS = [
    REMEMBER_SCHEMA, RECALL_SCHEMA, SHARED_REMEMBER_SCHEMA, SHARED_RECALL_SCHEMA,
    SHARED_FORGET_SCHEMA, SHARED_STATS_SCHEMA, SLEEP_SCHEMA, STATS_SCHEMA,
    INVALIDATE_SCHEMA, VALIDATE_SCHEMA, GET_SCHEMA, TRIPLE_ADD_SCHEMA, TRIPLE_QUERY_SCHEMA,
    REMEMBER_CANONICAL_SCHEMA, RECALL_CANONICAL_SCHEMA,
    SCRATCHPAD_WRITE_SCHEMA, SCRATCHPAD_READ_SCHEMA, SCRATCHPAD_CLEAR_SCHEMA,
    EXPORT_SCHEMA, UPDATE_SCHEMA, FORGET_SCHEMA, IMPORT_SCHEMA, DIAGNOSE_SCHEMA,
    GRAPH_QUERY_SCHEMA, GRAPH_LINK_SCHEMA,
    SYNC_PUSH_SCHEMA, SYNC_PULL_SCHEMA, SYNC_STATUS_SCHEMA,
]
