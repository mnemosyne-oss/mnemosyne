"""Tool schemas for the L3 persona layer (v3.10.0).

4 tools: promote, demote, list, reinforce.
All schema names follow mnemosyne_persona_* convention used elsewhere
in the provider. Schemas are JSON-schema-compatible for the Hermes
tool dispatch surface.
"""

PERSONA_PROMOTE_SCHEMA = {
    "name": "mnemosyne_persona_promote",
    "description": (
        "Promote a working or episodic memory into the L3 persona tier. "
        "Persona facts are always auto-injected into the system prompt regardless "
        "of semantic relevance. Tier values: 'permanent' (never evicted, requires "
        "explicit demotion), 'long_term' (default; reinforcement-driven decay), "
        "'working' (transient). Returns the new persona id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the source memory in working_memory or episodic_memory.",
            },
            "tier": {
                "type": "string",
                "enum": ["permanent", "long_term", "working"],
                "default": "long_term",
                "description": "Retention tier for the promoted persona fact.",
            },
            "reason": {
                "type": "string",
                "default": "",
                "description": "Optional human-readable reason for the promotion (logged for audit).",
            },
        },
        "required": ["memory_id"],
    },
}

PERSONA_DEMOTE_SCHEMA = {
    "name": "mnemosyne_persona_demote",
    "description": (
        "Move a persona fact back to memoria_preferences (acting as a tombstone). "
        "Use when a previously-promoted rule no longer applies or was a one-off. "
        "Returns the demoted persona id and the new preference record location."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "persona_id": {
                "type": "integer",
                "description": "ID of the persona row to demote.",
            },
            "reason": {
                "type": "string",
                "default": "",
                "description": "Optional reason for the demotion.",
            },
        },
        "required": ["persona_id"],
    },
}

PERSONA_LIST_SCHEMA = {
    "name": "mnemosyne_persona_list",
    "description": (
        "List L3 persona facts, optionally filtered by tier and/or topic. "
        "Returns personas ordered by tier (permanent first), then by reinforcement "
        "count descending (most-used first)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "enum": ["permanent", "long_term", "working"],
                "description": "Optional tier filter.",
            },
            "topic": {
                "type": "string",
                "description": "Optional topic filter (exact match).",
            },
        },
    },
}

PERSONA_REINFORCE_SCHEMA = {
    "name": "mnemosyne_persona_reinforce",
    "description": (
        "Bump the reinforcement_count and last_reinforced_at on a persona fact. "
        "Use when the persona rule was just applied -- signals 'this rule is in "
        "active use' to the decay logic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "persona_id": {
                "type": "integer",
                "description": "ID of the persona row to reinforce.",
            },
        },
        "required": ["persona_id"],
    },
}
