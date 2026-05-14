"""Mnemosyne Memory Provider for Hermes.

Deploy to Hermes via:
    ln -s /path/to/mnemosyne/hermes_memory_provider ~/.hermes/plugins/mnemosyne

Then set in ~/.hermes/config.yaml:
    memory:
      provider: mnemosyne

This gives Mnemosyne first-class MemoryProvider integration (system prompt
injection, pre-turn prefetch, post-turn sync, tool dispatch) while remaining
a standalone plugin deployed through the plugin system.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure mnemosyne core is importable from this directory
_mnemosyne_root = Path(__file__).resolve().parent.parent
if str(_mnemosyne_root) not in sys.path:
    sys.path.insert(0, str(_mnemosyne_root))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# C13: provider-active flag for memory-context double-injection prevention.
# ---------------------------------------------------------------------------
# When Hermes loads BOTH the MemoryProvider (canonical surface) AND the
# legacy hermes_plugin (composed by the provider's register() at line ~828,
# or independently discovered when a plugin.yaml is found), TWO pre-turn
# memory-injection paths fire on every LLM call:
#   1. MnemosyneMemoryProvider.prefetch() renders a `## Mnemosyne Context`
#      block.
#   2. hermes_plugin._on_pre_llm_call() renders a `MNEMOSYNE CONTEXT /
#      MNEMOSYNE RECALL` block.
# Both run their own beam.recall() and write to the system prompt, doubling
# the per-turn token cost and confusing the agent with duplicated context.
#
# Fix: when at least one MemoryProvider instance is the active surface (its
# initialize() ran successfully in a non-skip context), the plugin's
# _on_pre_llm_call() defers via the ``_provider_active`` flag below. The
# flag is the boolean view of an instance refcount so:
#   - Multiple provider instances coexisting in one process all keep the
#     flag True until ALL of them shut down (codex review #3 -- a single
#     bool can't represent multi-instance lifecycle).
#   - Skip-context re-init of an already-active instance DEACTIVATES it
#     (codex review #2 -- otherwise a primary->subagent re-init silences
#     the plugin for the subagent session, breaking legacy plugin behavior
#     for skip contexts).
#   - Init FAILURE keeps the flag at whatever it was -- if init fails,
#     this instance never activated, so the plugin path remains available
#     as the legacy fallback (codex review #1 -- without C27 merged here,
#     the provider's system_prompt_block returns "" on init failure;
#     suppressing the plugin too would leave a failed install completely
#     invisible).
_provider_active: bool = False
_active_provider_count: int = 0

# ---------------------------------------------------------------------------
# Lazy imports — fail gracefully if mnemosyne core is missing
# ---------------------------------------------------------------------------

def _get_beam_class():
    from mnemosyne.core.beam import BeamMemory
    return BeamMemory


def _get_triple_module():
    from mnemosyne.core.triples import add_triple, query_triples
    return add_triple, query_triples


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

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
        "Optionally provide replacement_id to chain old → new."
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
    "description": "Run PII-safe diagnostics on Mnemosyne installation. Checks dependencies, database state, and vector search readiness. Never includes memory content or API keys.",
    "parameters": {"type": "object", "properties": {}},
}

ALL_TOOL_SCHEMAS = [
    REMEMBER_SCHEMA, RECALL_SCHEMA, SLEEP_SCHEMA, STATS_SCHEMA,
    INVALIDATE_SCHEMA, TRIPLE_ADD_SCHEMA, TRIPLE_QUERY_SCHEMA,
    SCRATCHPAD_WRITE_SCHEMA, SCRATCHPAD_READ_SCHEMA, SCRATCHPAD_CLEAR_SCHEMA,
    EXPORT_SCHEMA, UPDATE_SCHEMA, FORGET_SCHEMA, IMPORT_SCHEMA, DIAGNOSE_SCHEMA,
]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    # Graceful fallback if ABC not available (shouldn't happen in practice)
    MemoryProvider = object  # type: ignore


class MnemosyneMemoryProvider(MemoryProvider):
    """Mnemosyne native memory — local SQLite with vector + FTS5 hybrid search."""

    # How long on_session_end will wait for sleep/consolidation to finish before
    # giving up and letting the daemon thread continue in the background. Tests
    # may shorten this to keep the suite fast.
    SESSION_END_SLEEP_TIMEOUT_SECONDS = 15

    def __init__(self):
        self._beam: Optional[Any] = None
        # C27: capture init exception so downstream methods can surface it
        # instead of silently no-op'ing. `_beam is None AND _init_error is None`
        # means a deliberate skip (subagent/cron/skill_loop context, or pre-init);
        # `_beam is None AND _init_error is not None` means a real failure that
        # users and operators need to see.
        self._init_error: Optional[BaseException] = None
        self._session_id = "hermes_default"
        self._hermes_home = ""
        self._platform = "cli"
        self._agent_context = "primary"
        self._turn_count = 0
        self._auto_sleep_threshold = 50
        self._auto_sleep_enabled = os.environ.get("MNEMOSYNE_AUTO_SLEEP_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
        self._ignore_patterns: List[str] = []  # Regex patterns to filter from memory
        self._skip_contexts = {"cron", "flush", "subagent", "background", "skill_loop"}  # Agent contexts to skip
        # Profile memory isolation: when enabled, each Hermes profile gets its own
        # Mnemosyne bank (separate SQLite DB). Default OFF for backward compatibility.
        self._profile_isolation_enabled = False
        # Tracked so shutdown() can wait briefly for in-flight consolidation
        # before clearing the host LLM backend, preventing the post-timeout
        # daemon thread from racing with unregister and falling through to
        # MNEMOSYNE_LLM_BASE_URL.
        self._session_end_thread: Optional[threading.Thread] = None
        # C13: per-instance tracking of whether THIS provider contributed
        # to the module-level _active_provider_count. Lets each instance
        # increment exactly once on activate and decrement exactly once on
        # deactivate, even across re-init cycles, without producing a
        # negative count when shutdown is called on a never-activated
        # instance.
        self._is_active_in_module: bool = False

    def _activate_in_module(self) -> None:
        """Bump the module-level active-provider count exactly once per
        instance lifecycle. Called when this instance transitions into
        the active state (non-skip-context initialize completed)."""
        global _active_provider_count, _provider_active
        if not self._is_active_in_module:
            self._is_active_in_module = True
            _active_provider_count += 1
            _provider_active = True

    def _deactivate_in_module(self) -> None:
        """Drop this instance from the module-level active-provider
        count. Idempotent -- a never-activated instance is a no-op.
        ``_provider_active`` stays True as long as ANY other instance is
        still active (multi-instance refcount semantics)."""
        global _active_provider_count, _provider_active
        if self._is_active_in_module:
            self._is_active_in_module = False
            _active_provider_count = max(0, _active_provider_count - 1)
            _provider_active = (_active_provider_count > 0)

    def _init_error_reason(self) -> str:
        """Return a human-readable failure reason for tool responses.

        Truncates the exception message to 200 chars so a verbose SQLite
        error (or similar) can't bloat downstream tool-call payloads.
        Collapses whitespace (including embedded newlines) into single
        spaces so the message can't break the system-prompt structure or
        look like multi-line instructions to the LLM -- defense in depth
        against an exception whose ``str()`` includes user-controllable
        text (e.g. a filesystem path supplied via MNEMOSYNE_DATA_DIR).
        Returns a generic string when init was never attempted (e.g. a
        subagent-context session that legitimately skipped initialize()).
        """
        if self._init_error is None:
            return "Mnemosyne not initialized"
        msg = str(self._init_error)
        # Collapse all whitespace (\n, \r, \t, runs of spaces) into a
        # single space. Codex finding #3: a multi-line exception text or
        # one containing tab-separated instruction-like content could
        # otherwise reach the LLM as structured input.
        import re
        msg = re.sub(r"\s+", " ", msg).strip()
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return f"{type(self._init_error).__name__}: {msg}"

    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        """Check if Mnemosyne core is importable. No network calls."""
        try:
            _get_beam_class()
            return True
        except Exception:
            return False

    def _apply_provider_config(self, kwargs: Dict[str, Any]) -> None:
        """Apply provider-specific config from Hermes kwargs or config.yaml.

        Precedence: kwargs > config.yaml > env var > hardcoded defaults.
        """
        # auto_sleep: prefer kwargs, then config.yaml, then env var
        auto_sleep = kwargs.get("auto_sleep")
        if auto_sleep is None:
            auto_sleep = self._read_config_key("auto_sleep")
        if auto_sleep is not None:
            self._auto_sleep_enabled = bool(auto_sleep)
        # env var is already applied in __init__, so it is the base default

        # sleep_threshold: prefer kwargs, then config.yaml, then default 50
        sleep_threshold = kwargs.get("sleep_threshold")
        if sleep_threshold is None:
            sleep_threshold = self._read_config_key("sleep_threshold")
        if sleep_threshold is not None:
            try:
                self._auto_sleep_threshold = int(sleep_threshold)
            except (TypeError, ValueError):
                logger.warning("Mnemosyne: invalid sleep_threshold=%r, keeping %d",
                               sleep_threshold, self._auto_sleep_threshold)

        # vector_type: pass through to BeamMemory if supported, log if not yet wired
        vector_type = kwargs.get("vector_type") or self._read_config_key("vector_type")
        if vector_type and vector_type not in ("float32", "int8", "bit"):
            logger.warning("Mnemosyne: unknown vector_type=%r, ignoring", vector_type)

        # ignore_patterns: list of regex patterns to filter from memory storage
        patterns = kwargs.get("ignore_patterns") or self._read_config_key("ignore_patterns")
        if patterns:
            if isinstance(patterns, str):
                patterns = [p.strip() for p in patterns.replace(",", "\n").split("\n") if p.strip()]
            elif isinstance(patterns, list):
                patterns = [str(p).strip() for p in patterns if str(p).strip()]
            self._ignore_patterns = patterns

        # profile_isolation: separate DB per Hermes profile (bank-based).
        # Default OFF. When enabled, each profile derives its own Mnemosyne bank.
        profile_isolation = kwargs.get("profile_isolation")
        if profile_isolation is None:
            profile_isolation = self._read_config_key("profile_isolation")
        if profile_isolation is not None:
            self._profile_isolation_enabled = bool(profile_isolation)

    def _should_filter(self, content: str) -> bool:
        """Check if content matches any ignore pattern. Returns True if it should be skipped."""
        if not self._ignore_patterns:
            return False
        import re
        for pattern in self._ignore_patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            except re.error:
                logger.debug("Mnemosyne: invalid ignore pattern %r, skipping", pattern)
        return False

    def _read_config_key(self, key: str) -> Any:
        """Read a single key from memory.mnemosyne in config.yaml."""
        try:
            import yaml, os
            config_path = os.path.join(self._hermes_home, "config.yaml") if self._hermes_home else ""
            if not config_path or not os.path.exists(config_path):
                return None
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            return config.get("memory", {}).get("mnemosyne", {}).get(key)
        except Exception:
            return None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "auto_sleep", "description": "Auto-run sleep() when working memory exceeds threshold. Set true to enable. Backward-compatible with MNEMOSYNE_AUTO_SLEEP_ENABLED env var.", "default": False},
            {"key": "sleep_threshold", "description": "Working memory count before auto-sleep triggers", "default": 50},
            {"key": "vector_type", "description": "Vector storage type (note: not yet wired to BeamMemory at runtime; reserved for future use)", "choices": ["float32", "int8", "bit"], "default": "int8"},
            {"key": "ignore_patterns", "description": "Regex patterns to filter from memory storage (one per line in config, or comma-separated). Memories matching any pattern are skipped.", "default": []},
            {"key": "profile_isolation", "description": "Enable per-profile memory isolation via Mnemosyne banks. Each Hermes profile gets its own SQLite database under mnemosyne/data/banks/<profile>/. Default false for backward compatibility.", "default": False},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist provider-specific config values."""
        try:
            import yaml, os
            config_path = os.path.join(hermes_home, "config.yaml") if hermes_home else ""
            if not config_path or not os.path.exists(config_path):
                return
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            memory_cfg = config.setdefault("memory", {}).setdefault("mnemosyne", {})
            memory_cfg.update(values)
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            logger.debug("Mnemosyne: could not persist config values", exc_info=True)

    import re
    _BANK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

    @staticmethod
    def _sanitize_bank_name(raw: str) -> str:
        """Sanitize a raw string into a valid bank name.

        Bank names become directory names. Rules:
        - Only [a-z0-9_-], max 64 chars
        - Must start with alphanumeric
        - Reject .. and / for path traversal safety
        - Fallback to 'default' if raw is empty or un-sanitizable
        """
        if not raw:
            return "default"
        # Lowercase and replace spaces/separators with underscore
        sanitized = raw.lower().strip()
        # Replace any disallowed characters with underscore
        sanitized = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in sanitized
        )
        # Collapse consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        # Strip leading/trailing underscores/hyphens
        sanitized = sanitized.strip("_-")
        # Ensure starts with alphanumeric
        if not sanitized or not sanitized[0].isalnum():
            sanitized = "b_" + sanitized if sanitized else "default"
        # Truncate to 64 chars
        if len(sanitized) > 64:
            sanitized = sanitized[:64].rstrip("_-")
        # Reject path traversal
        if ".." in sanitized or "/" in sanitized:
            return "default"
        return sanitized or "default"

    def _resolve_profile_bank(self) -> str:
        """Derive a bank name from the active Hermes profile.

        Precedence:
        1. agent_identity (explicit profile name from Hermes)
        2. hermes_home basename (derived from profile directory)
        3. Fallback to 'default' (backward-compatible shared DB)
        """
        # Try agent_identity first (most reliable)
        identity = getattr(self, "_agent_identity", None) or ""
        if identity and identity.lower() not in ("primary", "default", "none", ""):
            bank = self._sanitize_bank_name(identity)
            if bank != "default":
                return bank

        # Fall back to hermes_home basename
        hermes_home = getattr(self, "_hermes_home", "") or ""
        if hermes_home:
            from pathlib import Path
            basename = Path(hermes_home).name
            if basename and basename.lower() not in (".hermes", "hermes", "default", ""):
                bank = self._sanitize_bank_name(basename)
                if bank != "default":
                    return bank

        return "default"

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize Mnemosyne beam for this session."""
        # C27: clear stale state from any prior init attempt so a re-init
        # returns the provider to a clean slate. _beam reset is critical
        # for the primary->skip-context re-init case (codex review finding
        # #1): without it, a previously-initialized primary session that
        # later re-initialized into a subagent context would leave the old
        # _beam active, causing system_prompt_block() to report "Active"
        # and handle_tool_call() to silently write into the wrong session.
        # _init_error reset complements this for the failure-recovery case.
        self._beam = None
        self._init_error = None

        self._agent_context = kwargs.get("agent_context", "primary")
        self._platform = kwargs.get("platform", "cli")
        self._hermes_home = kwargs.get("hermes_home", "")
        self._agent_identity = kwargs.get("agent_identity", None) or ""

        # Apply provider-specific config from kwargs (Hermes-passed) or config.yaml fallback
        self._apply_provider_config(kwargs)

        if self._agent_context in self._skip_contexts:
            logger.debug("Mnemosyne skipped: non-primary context=%s", self._agent_context)
            # C13: a skip-context re-init must DEACTIVATE the instance if
            # it was previously active in this process. Without this, a
            # primary -> subagent re-init keeps _provider_active=True and
            # silences the legacy plugin's pre_llm_call for the subagent
            # session -- which the plugin used to handle (it has no
            # skip-context check of its own). Preserving legacy behavior
            # for the plugin in skip contexts is the smaller blast radius
            # vs. silently dropping memory injection for those sessions.
            self._deactivate_in_module()
            return

        self._session_id = f"hermes_{session_id}"

        try:
            if self._profile_isolation_enabled:
                # Route through Mnemosyne(bank=...) so BankManager handles
                # directory creation, canonical path resolution, and isolates
                # memories per Hermes profile.
                bank_name = self._resolve_profile_bank()
                from mnemosyne import Mnemosyne
                mem = Mnemosyne(
                    session_id=self._session_id,
                    bank=bank_name,
                    channel_id=kwargs.get("channel_id", ""),
                )
                self._beam = mem.beam
                logger.info(
                    "Mnemosyne initialized (profile isolation ON): session=%s, bank=%s, db=%s",
                    self._session_id, bank_name, mem.db_path,
                )
            else:
                BeamMemory = _get_beam_class()
                self._beam = BeamMemory(session_id=self._session_id)
                logger.info("Mnemosyne initialized: session=%s", self._session_id)
        except Exception as e:
            # C27: capture the exception so system_prompt_block() can render a
            # visible "UNAVAILABLE" banner every turn and handle_tool_call()
            # can return a structured `memory_unavailable` response. Without
            # this, an operator misconfiguration (corrupt DB, missing extras,
            # permissions, schema mismatch) silently masquerades as "the agent
            # doesn't remember anything" with no signal to the user.
            logger.warning("Mnemosyne init failed: %s", e)
            self._beam = None
            self._init_error = e

        # C13: activate AFTER the BeamMemory init result is known. If
        # init succeeded (_beam is set) the provider is the live memory
        # surface and the plugin path should defer. If init FAILED the
        # provider can't serve prefetch() / handle_tool_call() either,
        # so leaving the plugin's pre_llm_call enabled preserves a
        # legacy fallback that at least keeps the agent's memory
        # surface functional rather than silently breaking both paths.
        # Once C27 (provider-init-error-visible) merges, this fallback
        # becomes redundant -- but until then it's the conservative
        # choice (codex review #1).
        if self._beam is not None:
            self._activate_in_module()

        # Register the Hermes auxiliary LLM backend so Mnemosyne can route
        # consolidation and fact extraction through Hermes' authenticated
        # provider (e.g., openai-codex via OAuth) when the user opts in via
        # MNEMOSYNE_HOST_LLM_ENABLED=true. Registration alone does not
        # change Mnemosyne behavior; failure here must not break the provider.
        try:
            from hermes_memory_provider.hermes_llm_adapter import register_hermes_host_llm
            if register_hermes_host_llm():
                logger.info("Mnemosyne registered Hermes auxiliary LLM backend for memory operations")
        except Exception as exc:
            logger.debug("Mnemosyne could not register Hermes auxiliary LLM backend: %s", exc)

    def system_prompt_block(self) -> str:
        if self._beam:
            # Merge resolution (PR #106 + C27): keep PR #106's description
            # update that adds "identity" to the recognized memory kinds
            # (matches the auto-capture for identity-significant feelings
            # added in that PR), and keep C27's three-branch structure
            # (working / init-failed-visible / skip-context-silent).
            return (
                "# Mnemosyne Memory\n"
                "Active (native local memory). Use mnemosyne_remember to store ANY "
                "durable fact, preference, identity, or insight. Use mnemosyne_recall to search. "
                "The legacy memory tool is deprecated for durable storage — Mnemosyne is primary."
            )
        # C27: when init failed (as opposed to a deliberate skip-context),
        # surface the failure in the system prompt so the agent -- and through
        # it the user -- can see that memory is unavailable rather than
        # silently behaving as if nothing was stored. The skip-context case
        # still returns "" because that is the documented contract for
        # cron/subagent/skill_loop sessions.
        if self._init_error is not None:
            return (
                "# Mnemosyne Memory\n"
                f"⚠️ UNAVAILABLE: {self._init_error_reason()}\n"
                "Memory operations will fail this session. Resolve the underlying issue "
                "(check ~/.hermes/logs/agent.log for the WARNING) and restart Hermes to retry."
            )
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context via Mnemosyne hybrid search with temporal weighting.
        
        Only includes memories above a relevance threshold to prevent context pollution
        from low-quality matches. Scoped to the user's author_id when available."""
        if not self._beam or self._agent_context in self._skip_contexts:
            return ""
        try:
            import os
            author_id = self._beam.author_id or os.environ.get("MNEMOSYNE_AUTHOR_ID")
            results = self._beam.recall(
                query, top_k=8, 
                temporal_weight=0.2, temporal_halflife=48,
                author_id=author_id,
            )
            if not results:
                return ""
            # Filter out low-relevance results to prevent context pollution
            # Only include memories with score above threshold or high importance
            MIN_SCORE_THRESHOLD = 0.15
            MIN_IMPORTANCE_THRESHOLD = 0.5
            filtered = [
                r for r in results
                if r.get("score", 0) >= MIN_SCORE_THRESHOLD
                or r.get("importance", 0) >= MIN_IMPORTANCE_THRESHOLD
            ]
            if not filtered:
                return ""
            lines = ["## Mnemosyne Context"]
            for r in filtered:
                content = r.get("content", "")[:200]
                if len(r.get("content", "")) > 200:
                    content += "..."
                ts = r.get("timestamp", "")[:16] if r.get("timestamp") else ""
                imp = r.get("importance", 0.0)
                trust = r.get("trust_tier", "STATED")
                trust_tag = f" [{trust}]" if trust != "STATED" else ""
                lines.append(f"  [{ts}] (importance {imp:.2f}){trust_tag} {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Mnemosyne prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist the turn to Mnemosyne episodic memory."""
        if not self._beam or self._agent_context in self._skip_contexts:
            return
        try:
            if user_content and len(user_content) > 5 and not self._should_filter(user_content):
                self._beam.remember(
                    content=f"[USER] {user_content[:500]}",
                    source="conversation",
                    importance=0.3,
                    extract_entities=True,
                )
                # Check for identity-significant signals in user content
                self._capture_identity_signals(user_content)
            if assistant_content and len(assistant_content) > 10 and not self._should_filter(assistant_content):
                self._beam.remember(
                    content=f"[ASSISTANT] {assistant_content[:800]}",
                    source="conversation",
                    importance=0.2,
                    extract_entities=True,
                )
            self._turn_count += 1
            if self._auto_sleep_enabled and self._turn_count % 10 == 0:
                self._maybe_auto_sleep()
        except Exception as e:
            logger.debug("Mnemosyne sync_turn failed: %s", e)

    # Identity-significant expressions the user may voice about themselves or
    # their relationship to their work. When a match is found, the memory is
    # saved with source="identity" and higher importance so it survives
    # consolidation and remains recallable across sessions.
    _IDENTITY_SIGNALS: List[str] = [
        "feeling like",
        "imposter",
        "impostor",
        "barely know",
        "don't know my own",
        "don't even know how",
        "want them to feel",
        "i'm proud",
        "i feel like a",
        "i don't know how to",
    ]

    def _capture_identity_signals(self, user_content: str) -> None:
        content_lower = user_content.lower()
        for signal in self._IDENTITY_SIGNALS:
            if signal in content_lower:
                # Save identity memory with high importance for durable recall
                self._beam.remember(
                    content=f"[IDENTITY] {user_content[:400]}",
                    source="identity",
                    importance=0.85,
                    scope="global",
                    veracity="stated",
                )
                break  # One identity memory per turn

    def _maybe_auto_sleep(self) -> None:
        try:
            stats = self._beam.get_working_stats()
            working = stats.get("total", 0)
            if working > self._auto_sleep_threshold:
                logger.info("Mnemosyne auto-sleep: working=%d > threshold=%d", working, self._auto_sleep_threshold)
                sleep_fn = self._beam.sleep_all_sessions if hasattr(self._beam, "sleep_all_sessions") else self._beam.sleep
                sleep_thread = threading.Thread(target=sleep_fn, daemon=True)
                sleep_thread.start()
                sleep_thread.join(timeout=5)
                if sleep_thread.is_alive():
                    logger.warning("Mnemosyne auto-sleep timed out after 5s — consolidation deferred")
        except Exception:
            pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas — static, do not depend on initialization state."""
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._beam:
            # C27: structured response carries the actual failure reason
            # instead of a generic "not initialized" string. Status field
            # is parseable by tool consumers; `reason` is human-readable for
            # the agent to relay to the user. The `error` field is kept
            # alongside `status` so callers using the prior "if 'error' in
            # payload" pattern (codex review finding #4) don't silently
            # misclassify unavailable as success.
            reason = self._init_error_reason()
            return json.dumps({
                "status": "memory_unavailable",
                "tool": tool_name,
                "reason": reason,
                "error": f"Mnemosyne unavailable: {reason}",
            })
        try:
            if tool_name == "mnemosyne_remember":
                return self._handle_remember(args)
            elif tool_name == "mnemosyne_recall":
                return self._handle_recall(args)
            elif tool_name == "mnemosyne_sleep":
                return self._handle_sleep(args)
            elif tool_name == "mnemosyne_stats":
                return self._handle_stats(args)
            elif tool_name == "mnemosyne_invalidate":
                return self._handle_invalidate(args)
            elif tool_name == "mnemosyne_triple_add":
                return self._handle_triple_add(args)
            elif tool_name == "mnemosyne_triple_query":
                return self._handle_triple_query(args)
            elif tool_name == "mnemosyne_scratchpad_write":
                return self._handle_scratchpad_write(args)
            elif tool_name == "mnemosyne_scratchpad_read":
                return self._handle_scratchpad_read(args)
            elif tool_name == "mnemosyne_scratchpad_clear":
                return self._handle_scratchpad_clear(args)
            elif tool_name == "mnemosyne_export":
                return self._handle_export(args)
            elif tool_name == "mnemosyne_update":
                return self._handle_update(args)
            elif tool_name == "mnemosyne_forget":
                return self._handle_forget(args)
            elif tool_name == "mnemosyne_import":
                return self._handle_import(args)
            elif tool_name == "mnemosyne_diagnose":
                return self._handle_diagnose(args)
            else:
                return json.dumps({"error": f"Unknown Mnemosyne tool: {tool_name}"})
        except Exception as e:
            logger.error("Mnemosyne tool %s failed: %s", tool_name, e)
            return json.dumps({"error": f"Mnemosyne tool '{tool_name}' failed: {e}"})

    def _handle_remember(self, args: Dict[str, Any]) -> str:
        # Import at call-site so the provider module loads even when
        # the optional veracity_consolidation chain isn't on path
        # (BeamMemory ships a fallback). At call-time the import is
        # always satisfied because BeamMemory is already constructed.
        from mnemosyne.core.veracity_consolidation import clamp_veracity

        content = args.get("content", "")
        importance = float(args.get("importance", 0.5))
        source = args.get("source", "user")
        scope = args.get("scope", "session")
        valid_until = args.get("valid_until", None) or None
        extract_entities = bool(args.get("extract_entities", False))
        extract = bool(args.get("extract", False))
        metadata = args.get("metadata") or None
        # Trust-boundary clamp — see VERACITY_ALLOWED in
        # mnemosyne/core/veracity_consolidation.py for the canonical set.
        veracity = clamp_veracity(
            args.get("veracity"), context="mnemosyne_remember"
        )
        if not content:
            return json.dumps({"error": "content is required"})
        memory_id = self._beam.remember(
            content=content,
            importance=importance,
            source=source,
            scope=scope,
            valid_until=valid_until,
            extract_entities=extract_entities,
            extract=extract,
            metadata=metadata,
            veracity=veracity,
        )
        return json.dumps({
            "status": "stored",
            "memory_id": memory_id,
            "content_preview": content[:100],
            "extract_entities": extract_entities,
            "extract": extract,
            "metadata": metadata,
            "veracity": veracity,
        })

    def _handle_recall(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        top_k = int(args.get("limit", 5))
        temporal_weight = float(args.get("temporal_weight", 0.0))
        query_time = args.get("query_time") or None
        temporal_halflife_hours = float(args.get("temporal_halflife", 24))
        if not query:
            return json.dumps({"error": "query is required"})

        # Forward configurable scoring weights ONLY when the caller actually
        # supplied them. beam.recall treats None as "fall back to env var or
        # default" via _normalize_weights; passing 0.0 / 0.5 / etc. when the
        # caller didn't ask for tuning would override that resolution and
        # break MNEMOSYNE_*_WEIGHT env-var deployments. See issue #45.
        recall_kwargs: Dict[str, Any] = {
            "top_k": top_k,
            "temporal_weight": temporal_weight,
            "query_time": query_time,
            "temporal_halflife": temporal_halflife_hours,
        }
        for weight_key in ("vec_weight", "fts_weight", "importance_weight"):
            if weight_key in args:
                recall_kwargs[weight_key] = args[weight_key]

        results = self._beam.recall(query, **recall_kwargs)
        return json.dumps({"query": query, "count": len(results), "temporal_weight": temporal_weight, "results": results})

    def _handle_sleep(self, args: Dict[str, Any]) -> str:
        dry_run = bool(args.get("dry_run", False))
        all_sessions = bool(args.get("all_sessions", False))
        if all_sessions and hasattr(self._beam, "sleep_all_sessions"):
            result = self._beam.sleep_all_sessions(dry_run=dry_run)
        else:
            result = self._beam.sleep(dry_run=dry_run)
        working = self._beam.get_working_stats()
        episodic = self._beam.get_episodic_stats()
        return json.dumps({"status": result.get("status", "consolidated"), "result": result, "working": working, "episodic": episodic})

    def _handle_stats(self, args: Dict[str, Any]) -> str:
        working = self._beam.get_working_stats()
        episodic = self._beam.get_episodic_stats()
        return json.dumps({"provider": "mnemosyne", "session_id": self._session_id, "working": working, "episodic": episodic})

    def _handle_invalidate(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "")
        replacement_id = args.get("replacement_id", None) or None
        if not memory_id:
            return json.dumps({"error": "memory_id is required"})
        self._beam.invalidate(memory_id, replacement_id=replacement_id if replacement_id else None)
        return json.dumps({"status": "invalidated", "memory_id": memory_id})

    def _handle_triple_add(self, args: Dict[str, Any]) -> str:
        subject = args.get("subject", "")
        predicate = args.get("predicate", "")
        obj = args.get("object", "")
        valid_from = args.get("valid_from", None) or None
        if not all([subject, predicate, obj]):
            return json.dumps({"error": "subject, predicate, and object are required"})
        add_triple, _ = _get_triple_module()
        triple_id = add_triple(subject, predicate, obj, valid_from=valid_from,
                               db_path=self._beam.db_path)
        return json.dumps({"status": "stored", "triple_id": triple_id})

    def _handle_triple_query(self, args: Dict[str, Any]) -> str:
        subject = args.get("subject", "") or None
        predicate = args.get("predicate", "") or None
        obj = args.get("object", "") or None
        _, query_triples = _get_triple_module()
        results = query_triples(subject=subject, predicate=predicate, object=obj,
                                db_path=self._beam.db_path)
        return json.dumps({"count": len(results), "results": results})

    def _handle_scratchpad_write(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "Content is required"})
        pad_id = self._beam.scratchpad_write(content)
        return json.dumps({"status": "written", "id": pad_id})

    def _handle_scratchpad_read(self, args: Dict[str, Any]) -> str:
        entries = self._beam.scratchpad_read()
        return json.dumps({"entries_count": len(entries), "entries": entries})

    def _handle_scratchpad_clear(self, args: Dict[str, Any]) -> str:
        self._beam.scratchpad_clear()
        return json.dumps({"status": "cleared"})

    def _handle_export(self, args: Dict[str, Any]) -> str:
        output_path = args.get("output_path", "").strip()
        if not output_path:
            return json.dumps({"error": "output_path is required"})
        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne(session_id=self._session_id, db_path=self._beam.db_path)
        result = mem.export_to_file(output_path)
        return json.dumps(result)

    def _handle_update(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return json.dumps({"error": "memory_id is required"})
        content = args.get("content")
        importance = args.get("importance")
        ok = self._beam.update_working(memory_id, content=content, importance=importance)
        return json.dumps({
            "status": "updated" if ok else "not_found",
            "memory_id": memory_id,
        })

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return json.dumps({"error": "memory_id is required"})
        ok = self._beam.forget_working(memory_id)
        return json.dumps({
            "status": "deleted" if ok else "not_found",
            "memory_id": memory_id,
        })

    def _handle_import(self, args: Dict[str, Any]) -> str:
        provider = (args.get("provider") or "").strip().lower()
        input_path = args.get("input_path", "").strip()
        dry_run = bool(args.get("dry_run", False))
        force = bool(args.get("force", False))

        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne(session_id=self._session_id, db_path=self._beam.db_path)

        if provider:
            api_key = args.get("api_key", "").strip()
            user_id = args.get("user_id", "").strip() or None
            agent_id = args.get("agent_id", "").strip() or None
            base_url = args.get("base_url", "").strip() or None
            channel_id = args.get("channel_id")

            if not api_key:
                import os
                env_key = f"{provider.upper()}_API_KEY"
                api_key = os.environ.get(env_key, "")
            if not api_key:
                return json.dumps({
                    "error": f"api_key required for {provider} import. "
                             f"Set {provider.upper()}_API_KEY env var or pass api_key parameter.",
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

        if not input_path:
            return json.dumps({
                "error": "Either input_path (for file import) or provider "
                         "(for cross-provider import) is required",
            })
        stats = mem.import_from_file(input_path, force=force)
        return json.dumps({"status": "imported", "stats": stats})

    def _handle_diagnose(self, args: Dict[str, Any]) -> str:
        from mnemosyne.diagnose import run_diagnostics
        result = run_diagnostics()
        return json.dumps(result, indent=2, default=str)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = turn_number

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # Bound the consolidation call so a slow LLM (e.g., a Hermes-routed
        # network call) cannot block Hermes shutdown indefinitely. Mirrors
        # the daemon-thread pattern already used by _maybe_auto_sleep above:
        # the thread keeps running in the background if it overruns, but the
        # main shutdown path is freed after the join timeout.
        if not self._beam:
            return
        try:
            logger.info("Mnemosyne session end — running consolidation")
            timeout = self.SESSION_END_SLEEP_TIMEOUT_SECONDS
            beam = self._beam

            def _sleep_with_logging():
                # Wrap the target so exceptions get logged at the same
                # severity the previous synchronous version used, instead
                # of bubbling out as an uncaught daemon-thread traceback.
                try:
                    beam.sleep()
                except Exception as inner:
                    logger.debug("Mnemosyne session-end sleep failed: %s", inner)

            sleep_thread = threading.Thread(target=_sleep_with_logging, daemon=True)
            self._session_end_thread = sleep_thread
            sleep_thread.start()
            sleep_thread.join(timeout=timeout)
            if sleep_thread.is_alive():
                logger.warning(
                    "Mnemosyne session-end sleep timed out after %ss — consolidation deferred",
                    timeout,
                )
        except Exception as e:
            logger.debug("Mnemosyne session-end sleep failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._beam or action not in ("add", "replace"):
            return
        try:
            scope = "global" if target == "user" else "session"
            self._beam.remember(
                content=content,
                source=f"builtin_memory_{target}",
                importance=0.7 if target == "user" else 0.5,
                scope=scope,
            )
        except Exception as e:
            logger.debug("Mnemosyne mirror write failed: %s", e)

    # How long shutdown() will wait for an in-flight session_end consolidation
    # to finish before clearing the host backend. Bounded so shutdown is never
    # held up indefinitely; just long enough to close the race window where
    # the daemon thread's post-join host call could see a None backend and
    # fall through to MNEMOSYNE_LLM_BASE_URL (violating the host-skips-remote
    # contract). Tests may shorten this to keep the suite fast.
    SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 2

    def shutdown(self) -> None:
        # If session_end's daemon thread is still consolidating when shutdown
        # arrives, briefly wait for it. Otherwise clearing the host backend
        # next would race with the in-flight summarize/extract call and a
        # post-timeout "host attempted" decision could degrade to remote URL
        # despite A3.
        thread = self._session_end_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.SHUTDOWN_DRAIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                logger.debug(
                    "Mnemosyne shutdown: session-end thread still running after %ss; "
                    "proceeding (daemon thread will be reaped on process exit)",
                    self.SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                )
        self._session_end_thread = None

        # Symmetric with initialize(): clear the Hermes host LLM backend so a
        # process that later uses Mnemosyne outside Hermes does not retain a
        # stale reference into agent.auxiliary_client.
        try:
            from hermes_memory_provider.hermes_llm_adapter import unregister_hermes_host_llm
            unregister_hermes_host_llm()
        except Exception as exc:
            logger.debug("Mnemosyne could not unregister Hermes auxiliary LLM backend: %s", exc)
        self._beam = None

        # C13: decrement this instance's contribution to the module-level
        # active-provider count. ``_provider_active`` stays True if other
        # provider instances are still active in the process (codex
        # review #3 -- a single shared bool can't represent multi-
        # instance lifecycle).
        self._deactivate_in_module()


# ---------------------------------------------------------------------------
# Plugin registration (used when loaded via plugins.memory discovery)
# ---------------------------------------------------------------------------

def register_memory_provider(ctx):
    """Called by Hermes memory provider discovery system."""
    provider = MnemosyneMemoryProvider()
    ctx.register_memory_provider(provider)


# ---------------------------------------------------------------------------
# Plugin registration (used when loaded via Hermes plugin system)
# ---------------------------------------------------------------------------

def register(ctx):
    """Called by Hermes plugin loader to register CLI commands and tools."""
    from .cli import register_cli, mnemosyne_command
    ctx.register_cli_command(
        name="mnemosyne",
        help="Manage Mnemosyne local memory",
        description="Inspect, consolidate, and manage Mnemosyne native memory.",
        setup_fn=register_cli,
        handler_fn=mnemosyne_command,
    )

    # Also register tools and hooks from hermes_plugin (sibling directory).
    # This way a single symlink to hermes_memory_provider/ gives us the
    # full Mnemosyne experience: CLI + tools + hooks.
    try:
        _repo_root = str(Path(__file__).resolve().parent.parent)
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from hermes_plugin import register as _plugin_register
        _plugin_register(ctx)
    except Exception:
        pass  # Graceful degradation — CLI still works without plugin tools
