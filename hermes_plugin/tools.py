"""
Mnemosyne Plugin Tools for Hermes

Tool implementations that wrap Mnemosyne core functionality.
"""

import json

from hermes_plugin import _get_memory, _get_triples

# Tool Schemas (for Hermes tool registration)
REMEMBER_SCHEMA = {
    "name": "mnemosyne_remember",
    "description": "Store a memory in Mnemosyne local database. Use for important facts, preferences, or context to remember later.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember"
            },
            "importance": {
                "type": "number",
                "description": "Importance from 0.0 to 1.0 (0.9+ for critical facts)"
            },
            "source": {
                "type": "string",
                "description": "Source of the memory (preference, fact, identity, conversation, etc.)"
            },
            "valid_until": {
                "type": "string",
                "description": "ISO timestamp when this memory expires (optional)"
            },
            "scope": {
                "type": "string",
                "description": "'session' (default) or 'global' to make visible across all sessions",
                "enum": ["session", "global"]
            },
            "extract_entities": {
                "type": "boolean",
                "description": "If true, extract named entities from content and link them for fuzzy recall (e.g. 'Abdias' and 'Abdias J.' will match)",
                "default": False
            },
            "extract": {
                "type": "boolean",
                "description": "If true, extract structured facts from content using LLM and store as triples for fact-aware recall",
                "default": False
            },
            "author_id": {
                "type": "string",
                "description": "Who stored this memory (e.g., 'abdias'). Auto-set from session if not provided."
            },
            "author_type": {
                "type": "string",
                "description": "Type: 'human', 'agent', or 'system'."
            },
            "channel_id": {
                "type": "string",
                "description": "Channel/group this belongs to (e.g., 'fluxspeak-team')."
            },
            "bank": {
                "type": "string",
                "description": "Memory bank to store in. Use to isolate memories into separate banks (e.g., 'work', 'personal'). Defaults to 'default'.",
                "default": "default"
            }
        },
        "required": ["content"]
    }
}

RECALL_SCHEMA = {
    "name": "mnemosyne_recall",
    "description": "Search memories in Mnemosyne. Uses hybrid vector + full-text search across working and episodic memory. Supports temporal weighting to boost recent memories and per-query scoring weight overrides.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            },
            "temporal_weight": {
                "type": "number",
                "description": "How much to boost recent memories (0.0 = ignore time, 0.2 = mild recency bias, 0.5 = strong recency bias). Default 0.0 for backward compatibility.",
                "default": 0.0
            },
            "query_time": {
                "type": "string",
                "description": "ISO timestamp to treat as 'now' for temporal scoring (e.g., '2026-04-28T12:00:00'). Default is current time.",
                "default": None
            },
            "temporal_halflife": {
                "type": "number",
                "description": "Hours until temporal boost decays by half. Default 24. Lower = faster decay.",
                "default": 24,
            },
            "vec_weight": {
                "type": "number",
                "description": "Vector similarity weight in hybrid scoring. Omit (or pass null) to use MNEMOSYNE_VEC_WEIGHT env var or built-in default 0.5."
            },
            "fts_weight": {
                "type": "number",
                "description": "Full-text search weight in hybrid scoring. Omit (or pass null) to use MNEMOSYNE_FTS_WEIGHT env var or built-in default 0.3."
            },
            "importance_weight": {
                "type": "number",
                "description": "Importance score weight in hybrid scoring. Omit (or pass null) to use MNEMOSYNE_IMPORTANCE_WEIGHT env var or built-in default 0.2."
            },
            "author_id": {
                "type": "string",
                "description": "Filter by author (e.g., 'abdias'). Only recalls memories by this author."
            },
            "author_type": {
                "type": "string",
                "description": "Filter by author type: 'human', 'agent', or 'system'."
            },
            "channel_id": {
                "type": "string",
                "description": "Filter by channel/group (e.g., 'fluxspeak-team')."
            },
            "bank": {
                "type": "string",
                "description": "Memory bank to search in. Use to search isolated banks (e.g., 'work', 'personal'). Defaults to 'default'.",
                "default": "default"
            }
        },
        "required": ["query"],
    },
}

STATS_SCHEMA = {
    "name": "mnemosyne_stats",
    "description": "Get Mnemosyne memory statistics including BEAM tiers",
    "parameters": {
        "type": "object",
        "properties": {
            "bank": {
                "type": "string",
                "description": "Memory bank to get stats for (e.g., 'work', 'personal'). Defaults to 'default'.",
                "default": "default"
            }
        }
    }
}

TRIPLE_ADD_SCHEMA = {
    "name": "mnemosyne_triple_add",
    "description": "Add a temporal triple to the knowledge graph. Example: (Maya, assigned_to, auth-migration, valid_from=2026-01-15)",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Entity the fact is about"},
            "predicate": {"type": "string", "description": "Relationship or property"},
            "object": {"type": "string", "description": "Value or target entity"},
            "valid_from": {"type": "string", "description": "Date when fact became true (YYYY-MM-DD)"},
            "source": {"type": "string", "description": "Origin of the fact"},
            "confidence": {"type": "number", "description": "Confidence from 0.0 to 1.0"}
        },
        "required": ["subject", "predicate", "object"]
    }
}

TRIPLE_QUERY_SCHEMA = {
    "name": "mnemosyne_triple_query",
    "description": "Query temporal triples. Use as_of to ask what was true at a specific date.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "as_of": {"type": "string", "description": "Date to query historical truth (YYYY-MM-DD)"}
        }
    }
}

SLEEP_SCHEMA = {
    "name": "mnemosyne_sleep",
    "description": "Run the Mnemosyne sleep/consolidation cycle. Old working memories are summarized and moved to episodic memory. Set all_sessions=true to include inactive sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "If true, preview what would be consolidated without making changes",
                "default": False
            },
            "all_sessions": {
                "type": "boolean",
                "description": "If true, consolidate eligible old working memories across all sessions instead of only the current session",
                "default": False
            },
            "bank": {
                "type": "string",
                "description": "Memory bank to consolidate (e.g., 'work', 'personal'). Defaults to 'default'.",
                "default": "default"
            }
        }
    }
}

INVALIDATE_SCHEMA = {
    "name": "mnemosyne_invalidate",
    "description": "Mark a memory as expired or superseded. Use when a fact is no longer true or has been replaced.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the memory to invalidate"
            },
            "replacement_id": {
                "type": "string",
                "description": "Optional ID of the memory that replaces this one"
            }
        },
        "required": ["memory_id"]
    }
}

GET_SCHEMA = {
    "name": "mnemosyne_get",
    "description": "Retrieve a single memory by its primary key. Pure read, no side effects, no semantic search. Returns the exact memory with the given ID or None if not found. Use this when you already know the memory ID and want to read its content directly.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to retrieve"
            }
        },
        "required": ["memory_id"]
    }
}

SCRATCHPAD_WRITE_SCHEMA = {
    "name": "mnemosyne_scratchpad_write",
    "description": "Write a temporary note to the Mnemosyne scratchpad.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to write"
            },
            "bank": {
                "type": "string",
                "description": "Memory bank to write scratchpad to (e.g., 'work', 'personal'). Defaults to 'default'.",
                "default": "default"
            }
        },
        "required": ["content"]
    }
}

SCRATCHPAD_READ_SCHEMA = {
    "name": "mnemosyne_scratchpad_read",
    "description": "Read the Mnemosyne scratchpad entries.",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}

SCRATCHPAD_CLEAR_SCHEMA = {
    "name": "mnemosyne_scratchpad_clear",
    "description": "Clear all entries from the Mnemosyne scratchpad.",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}

EXPORT_SCHEMA = {
    "name": "mnemosyne_export",
    "description": "Export all Mnemosyne memories to a JSON file for backup or migration to another machine.",
    "parameters": {
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "description": "File path to write the export JSON (e.g., /tmp/mnemosyne_backup.json)"
            }
        },
        "required": ["output_path"]
    }
}

UPDATE_SCHEMA = {
    "name": "mnemosyne_update",
    "description": "Update the content or importance of an existing memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the memory to update"
            },
            "content": {
                "type": "string",
                "description": "New content for the memory (optional)"
            },
            "importance": {
                "type": "number",
                "description": "New importance from 0.0 to 1.0 (optional)"
            }
        },
        "required": ["memory_id"]
    }
}

FORGET_SCHEMA = {
    "name": "mnemosyne_forget",
    "description": "Permanently delete a memory by ID from working and legacy memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the memory to delete"
            }
        },
        "required": ["memory_id"]
    }
}

IMPORT_SCHEMA = {
    "name": "mnemosyne_import",
    "description": "Import Mnemosyne memories from a JSON file or another memory provider (Mem0, etc.). Idempotent by default.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "File path to read the export JSON from (for file imports)"
            },
            "provider": {
                "type": "string",
                "description": "Provider to import from: 'mem0'. Requires api_key. Use --list-providers to see supported providers."
            },
            "api_key": {
                "type": "string",
                "description": "API key for the source provider (can also be set via env var: MEM0_API_KEY)"
            },
            "user_id": {
                "type": "string",
                "description": "Filter imported memories by user ID (provider-specific)"
            },
            "agent_id": {
                "type": "string",
                "description": "Filter imported memories by agent ID (provider-specific)"
            },
            "base_url": {
                "type": "string",
                "description": "Base URL for self-hosted provider instances"
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, validate and transform but don't write any memories",
                "default": False
            },
            "channel_id": {
                "type": "string",
                "description": "Channel to assign imported memories to"
            },
            "force": {
                "type": "boolean",
                "description": "If true, overwrite existing records instead of skipping",
                "default": False
            }
        },
        "required": []
    }
}

DIAGNOSE_SCHEMA = {
    "name": "mnemosyne_diagnose",
    "description": "Run PII-safe diagnostics on Mnemosyne installation. Checks dependencies, database state, and vector search readiness. Writes a JSONL log file that can be shared for troubleshooting. Never includes memory content or API keys.",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}


# Tool Handlers
def mnemosyne_remember(args: dict, **kwargs) -> str:
    """Store a memory"""
    try:
        content = args.get("content", "").strip()
        importance = args.get("importance", 0.5)
        source = args.get("source", "conversation")
        valid_until = args.get("valid_until")
        scope = args.get("scope", "session")
        extract_entities = args.get("extract_entities", False)

        if not content:
            return json.dumps({"error": "Content is required"})

        extract = args.get("extract", False)
        bank = args.get("bank", "default")

        mem = _get_memory(bank=bank)
        memory_id = mem.remember(
            content, source=source, importance=importance,
            valid_until=valid_until, scope=scope,
            extract_entities=extract_entities,
            extract=extract
        )

        return json.dumps({
            "status": "stored",
            "id": memory_id,
            "scope": scope,
            "valid_until": valid_until,
            "extract_entities": extract_entities,
            "extract": extract,
            "content_preview": content[:80] + "..." if len(content) > 80 else content
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_recall(args: dict, **kwargs) -> str:
    """Search memories"""
    try:
        query = args.get("query", "").strip()
        top_k = args.get("top_k", 5)
        temporal_weight = args.get("temporal_weight", 0.0)
        query_time = args.get("query_time")
        temporal_halflife_hours = args.get("temporal_halflife", 24)

        if not query:
            return json.dumps({"error": "Query is required"})

        # Forward configurable scoring weights ONLY when caller supplied them.
        # mem.recall treats None as "fall back to env var or default" via
        # _normalize_weights; passing 0.0 / 0.5 / etc. when the caller didn't
        # ask for tuning would override that resolution and break
        # MNEMOSYNE_*_WEIGHT env-var deployments. See issue #45.
        recall_kwargs = {
            "top_k": top_k,
            "temporal_weight": temporal_weight,
            "query_time": query_time,
            "temporal_halflife": temporal_halflife_hours,
        }
        for weight_key in ("vec_weight", "fts_weight", "importance_weight"):
            if weight_key in args:
                recall_kwargs[weight_key] = args[weight_key]

        mem = _get_memory(bank=args.get("bank", "default"))
        results = mem.recall(query, **recall_kwargs)

        return json.dumps({
            "query": query,
            "results_count": len(results),
            "temporal_weight": temporal_weight,
            "query_time": query_time,
            "results": results
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_stats(args: dict, **kwargs) -> str:
    """Get memory statistics"""
    try:
        mem = _get_memory(bank=args.get("bank", "default"))
        stats = mem.get_stats()

        return json.dumps(stats)

    except Exception as e:
        return json.dumps({"error": str(e)})


from mnemosyne.core.annotations import ANNOTATION_KINDS as _ANNOTATION_KINDS


def _is_annotation_predicate(predicate) -> bool:
    """Membership check that safely handles non-string predicates.

    Returns False for None, lists, dicts, or any non-hashable value the
    caller might supply through the MCP boundary. Whitespace and case
    differences are NOT normalized — agents that mean `mentions` must
    spell it `mentions`. A future enhancement could accept case variants
    and emit a warning.
    """
    return isinstance(predicate, str) and predicate in _ANNOTATION_KINDS


def mnemosyne_triple_add(args: dict, **kwargs) -> str:
    """Add a temporal triple, or an annotation if the predicate is annotation-flavored.

    Post-E6 routing: predicates in `ANNOTATION_KINDS` (mentions, fact,
    occurred_on, has_source) write to `AnnotationStore` instead of
    `TripleStore`. Without this routing, an agent calling this tool with
    e.g. predicate="mentions" would silently invalidate prior annotation
    rows sharing the (subject, predicate) key — the same silent-destruction
    bug E6 fixed for the production extraction path. Current-truth
    predicates (anything else) still route to `TripleStore.add()`.

    Note for callers: `valid_from` is ignored when the predicate routes
    to annotations. The annotations table has no valid_from / valid_until
    columns — multi-valued annotations don't supersede each other. For
    `occurred_on` specifically, the date should be passed in `object`.
    """
    try:
        predicate = args["predicate"]
        if _is_annotation_predicate(predicate):
            mem = _get_memory()
            # Reuse BeamMemory's cached AnnotationStore handle (shared
            # connection, no per-call file-descriptor cost).
            store = getattr(mem.beam, "annotations", None)
            if store is None:
                # Defensive fallback for older BeamMemory builds that
                # don't expose the cached handle yet.
                from mnemosyne.core.annotations import AnnotationStore
                store = AnnotationStore(db_path=mem.db_path, conn=mem.beam.conn)
            row_id = store.add(
                memory_id=args["subject"],
                kind=predicate,
                value=args["object"],
                source=args.get("source", "conversation"),
                confidence=args.get("confidence", 1.0),
            )
            return json.dumps({
                "status": "added",
                "annotation_id": row_id,
                "store": "annotations",
            })

        # Current-truth temporal fact — TripleStore is the right home.
        kg = _get_triples()
        triple_id = kg.add(
            subject=args["subject"],
            predicate=predicate,
            object=args["object"],
            valid_from=args.get("valid_from"),
            source=args.get("source", "conversation"),
            confidence=args.get("confidence", 1.0)
        )
        return json.dumps({
            "status": "added",
            "triple_id": triple_id,
            "store": "triples",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_triple_query(args: dict, **kwargs) -> str:
    """Query temporal triples or annotations.

    Post-E6.a routing mirrors `mnemosyne_triple_add`: predicates in
    `ANNOTATION_KINDS` query `AnnotationStore`, others query `TripleStore`.
    Without this, agents would write via the tool, the data would land in
    annotations, but read-back through this same tool would return [] —
    a silent split-brain between write and read paths.

    The `as_of` arg only applies to TripleStore queries (current-truth
    semantics with validity windows). Annotation queries ignore it
    because the annotations table has no time-bounded validity.
    """
    try:
        predicate = args.get("predicate")
        if _is_annotation_predicate(predicate):
            mem = _get_memory()
            store = getattr(mem.beam, "annotations", None)
            if store is None:
                from mnemosyne.core.annotations import AnnotationStore
                store = AnnotationStore(db_path=mem.db_path, conn=mem.beam.conn)
            results = store.query_by_kind(
                kind=predicate,
                value=args.get("object"),
                memory_id=args.get("subject"),
            )
            return json.dumps({
                "results_count": len(results),
                "results": results,
                "store": "annotations",
            })

        # Current-truth: TripleStore with optional as_of filter.
        kg = _get_triples()
        results = kg.query(
            subject=args.get("subject"),
            predicate=predicate,
            object=args.get("object"),
            as_of=args.get("as_of")
        )
        return json.dumps({
            "results_count": len(results),
            "results": results,
            "store": "triples",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_sleep(args: dict, **kwargs) -> str:
    """Run consolidation sleep cycle"""
    try:
        dry_run = args.get("dry_run", False)
        all_sessions = args.get("all_sessions", False)
        bank = args.get("bank", "default")
        mem = _get_memory(bank=bank)
        if all_sessions and hasattr(mem, "sleep_all_sessions"):
            result = mem.sleep_all_sessions(dry_run=dry_run)
        else:
            result = mem.sleep(dry_run=dry_run)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_scratchpad_write(args: dict, **kwargs) -> str:
    """Write to scratchpad"""
    try:
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "Content is required"})
        bank = args.get("bank", "default")
        mem = _get_memory(bank=bank)
        pad_id = mem.scratchpad_write(content)
        return json.dumps({"status": "written", "id": pad_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_scratchpad_read(args: dict, **kwargs) -> str:
    """Read scratchpad"""
    try:
        mem = _get_memory(bank=args.get("bank", "default"))
        entries = mem.scratchpad_read()
        return json.dumps({"entries_count": len(entries), "entries": entries})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_scratchpad_clear(args: dict, **kwargs) -> str:
    """Clear scratchpad"""
    try:
        mem = _get_memory(bank=args.get("bank", "default"))
        mem.scratchpad_clear()
        return json.dumps({"status": "cleared"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_invalidate(args: dict, **kwargs) -> str:
    """Invalidate a memory"""
    try:
        memory_id = args.get("memory_id", "").strip()
        replacement_id = args.get("replacement_id")
        if not memory_id:
            return json.dumps({"error": "memory_id is required"})

        mem = _get_memory()
        ok = mem.invalidate(memory_id, replacement_id=replacement_id)
        return json.dumps({
            "status": "invalidated" if ok else "not_found",
            "memory_id": memory_id,
            "replacement_id": replacement_id
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_get(args: dict, **kwargs) -> str:
    """Retrieve a single memory by its primary key"""
    try:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return json.dumps({"error": "memory_id is required"})

        mem = _get_memory()
        result = mem.get(memory_id)
        if result is None:
            return json.dumps({"status": "not_found", "memory_id": memory_id})
        return json.dumps({"status": "ok", "memory": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_export(args: dict, **kwargs) -> str:
    """Export all memories to a JSON file"""
    try:
        output_path = args.get("output_path", "").strip()
        if not output_path:
            return json.dumps({"error": "output_path is required"})

        mem = _get_memory()
        result = mem.export_to_file(output_path)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_update(args: dict, **kwargs) -> str:
    """Update an existing memory by ID"""
    try:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return json.dumps({"error": "memory_id is required"})

        content = args.get("content")
        importance = args.get("importance")

        mem = _get_memory()
        ok = mem.update(memory_id, content=content, importance=importance)
        return json.dumps({
            "status": "updated" if ok else "not_found",
            "memory_id": memory_id
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_forget(args: dict, **kwargs) -> str:
    """Permanently delete a memory by ID"""
    try:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return json.dumps({"error": "memory_id is required"})

        mem = _get_memory()
        ok = mem.forget(memory_id)
        return json.dumps({
            "status": "deleted" if ok else "not_found",
            "memory_id": memory_id
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_import(args: dict, **kwargs) -> str:
    """Import memories from a JSON file or another memory provider."""
    try:
        provider = args.get("provider", "").strip().lower()
        input_path = args.get("input_path", "").strip()
        dry_run = args.get("dry_run", False)
        channel_id = args.get("channel_id")
        force = args.get("force", False)

        mem = _get_memory()

        # Cross-provider import
        if provider:
            api_key = args.get("api_key", "").strip()
            user_id = args.get("user_id", "").strip() or None
            agent_id = args.get("agent_id", "").strip() or None
            base_url = args.get("base_url", "").strip() or None

            if not api_key:
                import os
                env_key = f"{provider.upper()}_API_KEY"
                api_key = os.environ.get(env_key, "")
            if not api_key:
                return json.dumps({
                    "error": f"api_key required for {provider} import. Set {provider.upper()}_API_KEY env var or pass api_key parameter."
                })

            from mnemosyne.core.importers import import_from_provider
            result = import_from_provider(
                provider, mem,
                api_key=api_key,
                user_id=user_id,
                agent_id=agent_id,
                base_url=base_url,
                dry_run=dry_run,
                channel_id=channel_id,
            )
            return json.dumps(result.to_dict())

        # File import
        if not input_path:
            return json.dumps({
                "error": "Either input_path (for file import) or provider (for cross-provider import) is required"
            })

        stats = mem.import_from_file(input_path, force=force)
        return json.dumps({
            "status": "imported",
            "stats": stats
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_diagnose(args: dict, **kwargs) -> str:
    """Run PII-safe diagnostics and return summary"""
    try:
        from mnemosyne.diagnose import run_diagnostics
        result = run_diagnostics()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
