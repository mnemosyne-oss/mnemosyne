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
                "description": "Source of the memory (preference, fact, conversation, etc.)"
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
            }
        },
        "required": ["content"]
    }
}

RECALL_SCHEMA = {
    "name": "mnemosyne_recall",
    "description": "Search memories in Mnemosyne. Uses hybrid vector + full-text search across working and episodic memory.",
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
            }
        },
        "required": ["query"]
    }
}

STATS_SCHEMA = {
    "name": "mnemosyne_stats",
    "description": "Get Mnemosyne memory statistics including BEAM tiers",
    "parameters": {
        "type": "object",
        "properties": {}
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
    "description": "Run the Mnemosyne sleep/consolidation cycle. Old working memories are summarized and moved to episodic memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "If true, preview what would be consolidated without making changes",
                "default": False
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

SCRATCHPAD_WRITE_SCHEMA = {
    "name": "mnemosyne_scratchpad_write",
    "description": "Write a temporary note to the Mnemosyne scratchpad.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to write"
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
    "description": "Import Mnemosyne memories from a JSON file. Idempotent by default.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "File path to read the export JSON from"
            },
            "force": {
                "type": "boolean",
                "description": "If true, overwrite existing records instead of skipping",
                "default": False
            }
        },
        "required": ["input_path"]
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

        mem = _get_memory()
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

        if not query:
            return json.dumps({"error": "Query is required"})

        mem = _get_memory()
        results = mem.recall(query, top_k=top_k)

        return json.dumps({
            "query": query,
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_stats(args: dict, **kwargs) -> str:
    """Get memory statistics"""
    try:
        mem = _get_memory()
        stats = mem.get_stats()

        return json.dumps(stats)

    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_triple_add(args: dict, **kwargs) -> str:
    """Add a temporal triple"""
    try:
        kg = _get_triples()
        triple_id = kg.add(
            subject=args["subject"],
            predicate=args["predicate"],
            object=args["object"],
            valid_from=args.get("valid_from"),
            source=args.get("source", "conversation"),
            confidence=args.get("confidence", 1.0)
        )
        return json.dumps({"status": "added", "triple_id": triple_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_triple_query(args: dict, **kwargs) -> str:
    """Query temporal triples"""
    try:
        kg = _get_triples()
        results = kg.query(
            subject=args.get("subject"),
            predicate=args.get("predicate"),
            object=args.get("object"),
            as_of=args.get("as_of")
        )
        return json.dumps({"results_count": len(results), "results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_sleep(args: dict, **kwargs) -> str:
    """Run consolidation sleep cycle"""
    try:
        dry_run = args.get("dry_run", False)
        mem = _get_memory()
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
        mem = _get_memory()
        pad_id = mem.scratchpad_write(content)
        return json.dumps({"status": "written", "id": pad_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_scratchpad_read(args: dict, **kwargs) -> str:
    """Read scratchpad"""
    try:
        mem = _get_memory()
        entries = mem.scratchpad_read()
        return json.dumps({"entries_count": len(entries), "entries": entries})
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_scratchpad_clear(args: dict, **kwargs) -> str:
    """Clear scratchpad"""
    try:
        mem = _get_memory()
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
    """Import memories from a JSON file"""
    try:
        input_path = args.get("input_path", "").strip()
        force = args.get("force", False)
        if not input_path:
            return json.dumps({"error": "input_path is required"})

        mem = _get_memory()
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
