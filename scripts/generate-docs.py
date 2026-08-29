#!/usr/bin/env python3
"""
Auto-generate docs/api/ files from live code.

Usage:
    python3 scripts/generate-docs.py
    python3 scripts/generate-docs.py --check      # report drift, write nothing

Writes canonical copies to docs/api/ inside the mnemosyne repo.
Also writes to the docs sibling repo (../mnemosyne-docs) if present.
All sibling writes are optional; canonical copies are always written.

DESIGN: PARSE, DO NOT IMPORT
----------------------------
Everything factual is read out of the source tree with `ast`, never by
importing mnemosyne. Two reasons, both load-bearing:

  1. The CI `docs-check` job runs a bare python3 with NO pip install
     (see .github/workflows/ci.yml). `import mnemosyne` would fail on
     PyYAML, so an import-based generator could only ever work locally.
  2. Importing pulls in sqlite, config singletons, and optionally
     fastembed. Docs generation must not depend on a working runtime.

WHAT IS DERIVED VS WHAT IS AUTHORED
-----------------------------------
Derived from code (cannot drift):
  - the tool list, names, descriptions, and parameters  <- mnemosyne/tool_schemas.py
  - which tools are reachable over MCP                  <- mnemosyne/mcp_tools.py::_TOOL_HANDLERS
  - the config key list, env var names, and defaults    <- mnemosyne/core/config.py
  - which keys need a restart                           <- config.py::REQUIRES_RESTART
  - the version                                         <- mnemosyne/__init__.py

Authored here (editorial prose only):
  - CONFIG_DESCRIPTIONS, one line per config key.

The split matters. Before this rewrite the generator hardcoded the tool
list and the config defaults as literals "verified against v3.6.0", while
reading the version dynamically. The result was docs stamped with a
current version carrying a two-year-old payload: 25 tools when the code
had 37, and roughly a dozen wrong defaults. Facts now come from the code
and only prose is hand-held, and DESCRIPTION drift is a hard error below.
"""
from __future__ import annotations

import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------
# AST extraction helpers
# ---------------------------------------------------------------
def _parse(rel_path: str) -> ast.Module:
    with open(os.path.join(REPO_ROOT, rel_path), encoding="utf-8") as f:
        return ast.parse(f.read(), filename=rel_path)


def _module_literal(rel_path: str, names: set) -> dict:
    """Return {name: python_value} for module-level literal assignments."""
    out = {}
    for node in _parse(rel_path).body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id in names and node.value is not None:
                try:
                    out[tgt.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
    return out


def _dict_keys_of_assignment(rel_path: str, var_name: str) -> list:
    """
    String keys of a module-level dict whose VALUES are not literals.

    _TOOL_HANDLERS maps tool names to function objects, so literal_eval
    cannot touch it. We only need the keys.
    """
    for node in _parse(rel_path).body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id == var_name and isinstance(value, ast.Dict):
                return [k.value for k in value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return []


def _version() -> str:
    v = _module_literal("mnemosyne/__init__.py", {"__version__"}).get("__version__")
    if not v:
        sys.exit("FATAL: could not read __version__ from mnemosyne/__init__.py")
    return v


# ---------------------------------------------------------------
# Tool schemas, derived
# ---------------------------------------------------------------
def _collect_tools() -> list:
    """
    Every *_SCHEMA dict in tool_schemas.py, annotated with MCP reachability.

    Returns a list of dicts: name, description, params (ordered),
    required (set), mcp (bool).
    """
    handlers = set(_dict_keys_of_assignment("mnemosyne/mcp_tools.py", "_TOOL_HANDLERS"))
    if not handlers:
        sys.exit("FATAL: could not parse _TOOL_HANDLERS from mnemosyne/mcp_tools.py")

    # Every module-level *_SCHEMA dict, keyed by its variable name.
    by_var, tree = {}, _parse("mnemosyne/tool_schemas.py")
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id.endswith("_SCHEMA")):
            continue
        try:
            schema = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        if isinstance(schema, dict) and "name" in schema:
            by_var[tgt.id] = schema

    # ALL_TOOL_SCHEMAS is the advertised surface: mcp_tools.TOOLS is built
    # from it, so a schema that is defined but absent from this list is not
    # exposed over MCP at all. Documenting definitions rather than exports
    # would overstate the tool count.
    exported_vars = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        tgt = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id == "ALL_TOOL_SCHEMAS":
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                exported_vars = [e.id for e in value.elts if isinstance(e, ast.Name)]

    if not exported_vars:
        sys.exit("FATAL: could not parse ALL_TOOL_SCHEMAS from mnemosyne/tool_schemas.py")

    orphans = sorted(
        by_var[v]["name"] for v in by_var if v not in set(exported_vars)
    )
    if orphans:
        print("  note: schemas defined but absent from ALL_TOOL_SCHEMAS, "
              "so not advertised over MCP: " + ", ".join(orphans))

    tools, seen = [], set()
    for var in exported_vars:
        schema = by_var.get(var)
        if schema is None:
            continue
        name = schema["name"]
        if name in seen:
            continue
        seen.add(name)

        params = schema.get("parameters") or {}
        props = params.get("properties") or {}
        tools.append({
            "name": name,
            "description": (schema.get("description") or "").strip(),
            "props": props,
            "required": set(params.get("required") or []),
            "mcp": name in handlers,
        })

    if not tools:
        sys.exit("FATAL: ALL_TOOL_SCHEMAS resolved to no schemas")
    tools.sort(key=lambda t: (not t["mcp"], t["name"]))
    return tools


def _type_of(spec: dict) -> str:
    """Render a JSON Schema property as a short type string."""
    if not isinstance(spec, dict):
        return "any"
    if "enum" in spec:
        return "enum[" + ", ".join(str(v) for v in spec["enum"]) + "]"
    t = spec.get("type")
    if isinstance(t, list):
        return " \\| ".join(str(x) for x in t)
    if t == "array":
        item = spec.get("items") or {}
        inner = _type_of(item) if item else "any"
        return f"array<{inner}>"
    if t:
        return str(t)
    if "anyOf" in spec:
        return " \\| ".join(_type_of(s) for s in spec["anyOf"])
    return "any"


# ---------------------------------------------------------------
# Config, derived
# ---------------------------------------------------------------
def _collect_config() -> tuple:
    got = _module_literal(
        "mnemosyne/core/config.py",
        {"ENV_VAR_MAP", "DEFAULTS", "REQUIRES_RESTART"},
    )
    env_map = got.get("ENV_VAR_MAP")
    defaults = got.get("DEFAULTS")
    restart = got.get("REQUIRES_RESTART")
    if not env_map or not defaults or restart is None:
        sys.exit("FATAL: could not parse ENV_VAR_MAP / DEFAULTS / REQUIRES_RESTART "
                 "from mnemosyne/core/config.py")
    return env_map, defaults, set(restart)


# Editorial prose, one line per config key. Keys and defaults come from
# code; only these sentences are hand-written. A key here that no longer
# exists in ENV_VAR_MAP, or an ENV_VAR_MAP key missing here, is a hard
# error in main() so this table cannot silently rot.
CONFIG_DESCRIPTIONS = {
    # Paths
    "data_dir": "Root directory for all Mnemosyne data. Overrides the `$HERMES_HOME` default.",
    "home": "Base Hermes home directory used to derive default paths.",
    "db_path": "Explicit path to the primary SQLite database file.",
    "backup_dir": "Directory for `mnemosyne backups` snapshots.",
    "blob_dir": "Directory for content-addressed blob storage.",
    "shared_db_path": "Path to the shared cross-agent surface database.",

    # Embeddings
    "embedding_model": "Embedding model name. Local fastembed model, or an API model when routed to a provider.",
    "embedding_dim": "Override the embedding dimension. Must match the dimension stored in the existing vec tables.",
    "embedding_api_key": "API key for a remote embedding provider. Falls back to `OPENAI_API_KEY`.",
    "embedding_api_url": "Base URL of an OpenAI-compatible embedding endpoint.",
    "embeddings_via_api": "Force API mode for embeddings instead of inferring it from the model name.",
    "no_embeddings": "Disable dense retrieval entirely. Recall degrades to lexical FTS5.",
    "skip_embeddings": "Alias for `no_embeddings`.",
    "embeddings_off": "Alias for `no_embeddings`.",
    "fastembed_cache_dir": "Cache directory for downloaded fastembed ONNX models.",

    # Vectors
    "vec_type": "sqlite-vec element type: `int8`, `float32`, or `bit`.",

    # Hybrid scoring
    "vec_weight": "Vector similarity weight in hybrid ranking. Normalized with the next two to sum to 1.0.",
    "fts_weight": "FTS5 rank weight in hybrid ranking.",
    "importance_weight": "Importance weight in hybrid ranking.",
    "temporal_halflife_hours": "Hours until the temporal recall boost decays by half.",
    "recency_halflife": "Hours until the recency decay factor halves.",
    "recall_extra_stopwords": "Extra comma or space separated stopwords for lexical recall.",
    "cross_session": "Allow recall to return memories from other sessions.",

    # Recall engines
    "polyphonic_recall": "Enable the polyphonic multi-voice recall engine with RRF fusion.",
    "query_intent": "Classify query intent and adjust scoring weights accordingly.",
    "fact_recall_enabled": "Enable structured fact matching during recall.",
    "enhanced_recall": "Enable the enhanced recall pipeline with fact, graph, and episodic fusion.",
    "proactive_linking": "Create cross-memory graph edges on insertion.",
    "lenient_fact_match": "Match facts by substring instead of exact equality.",
    "recall_diagnostics": "Collect per-stage recall diagnostics. See `explain=True`.",

    # Working memory
    "wm_max_items": "Maximum rows retained in `working_memory` before eviction.",
    "wm_ttl_hours": "Age at which a working memory row becomes eligible for consolidation or trim.",
    "wm_bump_cap_hours": "Maximum window in which a recall can refresh a row's recency.",
    "wm_pinned_ids": "Comma separated memory IDs that are never evicted.",

    # Episodic and consolidation
    "ep_limit": "Maximum rows retained in `episodic_memory`.",
    "sleep_batch": "Working memory rows processed per sleep cycle.",
    "sp_max": "Maximum scratchpad entries retained.",

    # Tier degradation
    "tier2_days": "Age in days at which an episodic memory degrades to tier 2.",
    "tier3_days": "Age in days at which an episodic memory degrades to tier 3.",
    "tier1_weight": "Recall score multiplier for tier 1 (fresh) episodic memories.",
    "tier2_weight": "Recall score multiplier for tier 2 memories.",
    "tier3_weight": "Recall score multiplier for tier 3 memories.",
    "smart_compress": "Use LLM summarization rather than truncation when degrading a tier.",
    "tier3_max_chars": "Character budget for tier 3 compressed content.",
    "degrade_batch": "Rows degraded per maintenance pass.",

    # LLM
    "llm_enabled": "Enable LLM summarization during sleep consolidation.",
    "llm_max_tokens": "Maximum output tokens per LLM summary.",
    "llm_n_threads": "CPU threads for local GGUF inference.",
    "llm_n_ctx": "Context window for local GGUF inference.",
    "llm_repo": "Hugging Face repo holding the local GGUF model.",
    "llm_file": "GGUF filename within the repo.",
    "llm_base_url": "Base URL of an OpenAI-compatible chat endpoint.",
    "llm_api_key": "API key for the remote chat endpoint.",
    "llm_model": "Model identifier for remote chat calls.",
    "llm_timeout": "Per-request timeout in seconds for LLM calls.",
    "llm_fallback_models": "Comma separated models to try if the primary fails.",
    "llm_fallback_base_url": "Base URL for the fallback chat endpoint.",
    "llm_fallback_api_key": "API key for the fallback chat endpoint.",
    "force_local": "Skip the remote chain and use the local GGUF model directly.",
    "sleep_prompt": "Override the consolidation summarization prompt.",
    "host_llm_enabled": "Route LLM work through a host-registered backend. See `core/llm_backends.py`.",
    "host_llm_provider": "Provider override passed to the host backend.",
    "host_llm_model": "Model override passed to the host backend.",
    "host_llm_n_ctx": "Context budget assumed for the host backend.",

    # Modality providers (RFC 0002)
    "modality_enabled": "Allow outbound calls to describe non-text media. Off by default; nothing is sent until this is set.",
    "modality_base_url": "Base URL of an OpenAI-compatible vision endpoint. Any provider that speaks the protocol works.",
    "modality_api_key": "API key for the modality endpoint.",
    "modality_vision_model": "Model used to describe images and documents.",
    "modality_video_model": "Model used to describe video.",
    "modality_audio_model": "Model used to transcribe or describe audio.",
    "modality_timeout": "Timeout in seconds for a modality describe call.",
    "modality_max_moments": "Maximum moments retained per media asset.",

    # Conflict detection
    "llm_conflict_detection": "Enable LLM-based contradiction detection during sleep.",
    "conflict_llm_base_url": "Base URL for the conflict detection endpoint.",
    "conflict_llm_api_key": "API key for the conflict detection endpoint.",
    "conflict_llm_model": "Model for conflict detection calls.",

    # Sync
    "sync_remote": "Remote sync server URL.",
    "sync_host": "Bind address for `mnemosyne sync-serve`.",
    "sync_port": "Bind port for `mnemosyne sync-serve`.",
    "sync_key": "Passphrase used to derive the client-side encryption key.",
    "sync_encrypt": "Encrypt sync payloads client-side with XChaCha20-Poly1305.",
    "sync_roles": "Conversation roles synced into memory. Defaults to user turns only.",

    # Auto sleep and reflection
    "auto_sleep_enabled": "Run consolidation automatically on a background daemon.",
    "reflect_disabled_for_cron": "Skip reflection when running in a cron context.",
    "reflect_max_calls_per_session": "Maximum reflection passes per session.",
    "skip_contexts": "Comma separated host context names that should not write memories.",

    # Prefetch and turn limits
    "prefetch_content_chars": "Truncate prefetched memory content to this many characters. `0` disables truncation.",
    "sync_turn_user_limit": "Maximum user turns captured per sync pass.",
    "sync_turn_assistant_limit": "Maximum assistant turns captured per sync pass.",

    # Persona (L3)
    "persona_enabled": "Inject L3 persona facts into the system prompt.",
    "persona_token_cap": "Token budget for injected persona facts.",
    "persona_interval": "Turns between persona refreshes.",
    "persona_daily_sync_hour": "Local hour at which the daily persona sync runs.",

    # Sleep-time model refresh
    "sleep_model_refresh_enabled": "Let sleep propose updates to canonical facts.",
    "sleep_model_refresh_auto_apply": "Apply high-confidence proposals without review.",
    "sleep_model_refresh_categories": "Comma separated canonical categories eligible for refresh.",
    "sleep_model_refresh_max_tokens": "Token budget for a refresh proposal.",
    "sleep_model_refresh_temperature": "Sampling temperature for refresh proposals.",
    "sleep_model_refresh_auto_apply_min_confidence": "Minimum confidence to auto-apply a proposal.",
    "sleep_model_refresh_min_evidence": "Minimum supporting memories before proposing a change.",
    "sleep_model_refresh_conflict_min_confidence": "Minimum confidence to auto-apply a proposal that contradicts a current fact.",
    "sleep_model_refresh_conflict_min_evidence": "Minimum supporting memories for a contradicting change.",

    # SHMR
    "shmr_batch_size": "Memories per SHMR harmonization batch.",
    "shmr_max_iterations": "Maximum clustering iterations per SHMR batch.",
    "shmr_similarity_threshold": "Cosine similarity required to cluster two memories.",
    "shmr_harmony_threshold": "Harmony score required to merge a cluster into a belief.",
    "shmr_model": "LLM model for SHMR summarization. Empty uses the default chain.",
    "shmr_min_cluster_size": "Minimum memories required to form a cluster.",
    "shmr_temperature": "Sampling temperature for SHMR summarization.",

    # Misc
    "auto_migrate": "Run packaged migrations automatically on database open.",
    "default_scope": "Default scope for new memories: `session` or `global`.",
    "default_owner": "Default owner ID for canonical facts and shared memory.",
    "ignore_patterns": "Comma separated regexes; matching content is never stored.",
    "write_classifier": "Write classifier mode controlling which content is stored.",
}

# Keys whose effective default is set by a module-level constant that reads
# os.environ directly, bypassing MnemosyneConfig. For these, the value in
# DEFAULTS is not what the runtime uses.
#
# This list is DERIVED, not maintained by hand: _scan_effective_defaults()
# below finds every `os.environ.get("MNEMOSYNE_...", "literal")` in the
# package and compares it against DEFAULTS. Hand-maintaining it was how the
# rest of this file rotted, and there are currently ~20 real divergences,
# which is far too many to track manually.
#
# Only prose lives here, keyed by config key, and only where the plain
# "module reads it directly" note is not explanation enough.
EFFECTIVE_DEFAULT_PROSE = {
    "vec_weight": "Recall weights resolve as `config.yaml > MNEMOSYNE_*_WEIGHT > defaults` at request time.",
    "fts_weight": "Recall weights resolve as `config.yaml > MNEMOSYNE_*_WEIGHT > defaults` at request time.",
    "importance_weight": "Recall weights resolve as `config.yaml > MNEMOSYNE_*_WEIGHT > defaults` at request time.",
    "llm_enabled": "Note the direction: `config.py` declares this off while the module defaults it on.",
}

# These keys are resolved by _resolve_recall_weights() at request time through
# MnemosyneConfig, so their environment-variable reads are fallback inputs,
# not module-level configuration bypasses.
RUNTIME_CONFIG_RESOLVED_KEYS = {
    "vec_weight",
    "fts_weight",
    "importance_weight",
}


def _scan_effective_defaults(env_map: dict, defaults: dict) -> dict:
    """Find keys whose runtime default differs from the declared one.

    Returns {config_key: (effective_value, source_file)}.

    Matches module-level `os.environ.get("MNEMOSYNE_X", "literal")`. Values
    that differ only in spelling (`true` vs `1`, `0.7` vs `0.70`, an empty
    declared default) are not divergences and are filtered out.
    """
    import glob
    import re

    pattern = re.compile(
        r'os\.environ\.get\(\s*["\'](MNEMOSYNE_[A-Z0-9_]+)["\']\s*,\s*'
        r'(?:"([^"]*)"|\'([^\']*)\')\s*\)'
    )
    env_to_key = {v: k for k, v in env_map.items()}

    found = {}
    root = os.path.join(REPO_ROOT, "mnemosyne")
    for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        for m in pattern.finditer(text):
            envvar = m.group(1)
            value = m.group(2) if m.group(2) is not None else m.group(3)
            key = env_to_key.get(envvar)
            if key is None or key in found or key in RUNTIME_CONFIG_RESOLVED_KEYS:
                continue
            found[key] = (value, os.path.relpath(path, REPO_ROOT))

    def _same(declared, effective: str) -> bool:
        if declared is None:
            return False
        if isinstance(declared, bool):
            return effective.strip().lower() in (
                ("true", "1", "yes", "on") if declared else ("false", "0", "no", "off")
            )
        d = str(declared).strip()
        e = effective.strip()
        if d == e:
            return True
        try:
            return float(d) == float(e)
        except (TypeError, ValueError):
            return False

    out = {}
    for key, (value, src) in found.items():
        declared = defaults.get(key)
        # An empty declared default means "unset", which the module fallback
        # fills in. That is the documented design, not a contradiction.
        if declared in ("", None):
            continue
        # An empty module fallback is also "unset": the module reads the env
        # var, gets nothing, and applies its own logic downstream. There is
        # no competing default to report.
        if value.strip() == "":
            continue
        if not _same(declared, value):
            out[key] = (value, src)
    return out

# Env vars read directly via os.environ, bypassing MnemosyneConfig. These
# are NOT in ENV_VAR_MAP, so they cannot be set in config.yaml and are
# env-only. Documented separately and honestly.
ENV_ONLY_DESCRIPTIONS = {
    "MNEMOSYNE_AUTHOR_ID": "Identifier recorded as the author of new memories.",
    "MNEMOSYNE_AUTHOR_TYPE": "Author kind: user, assistant, system.",
    "MNEMOSYNE_AUTO_SLEEP_TIMEOUT": "Seconds the Hermes provider waits for a background sleep to finish.",
    "MNEMOSYNE_BANK": "Default memory bank for CLI operations.",
    "MNEMOSYNE_BEAM_MODE": "Enable BEAM mode extensions in the polyphonic engine.",
    "MNEMOSYNE_BEAM_OPTIMIZATIONS": "Enable BEAM benchmark optimizations.",
    "MNEMOSYNE_BINARY_BONUS": "A/B toggle for the binary vector bonus in episodic scoring.",
    "MNEMOSYNE_BUSY_TIMEOUT_MS": "SQLite `busy_timeout` in milliseconds.",
    "MNEMOSYNE_CHANNEL_ID": "Channel or session identifier for memory scoping.",
    "MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED": "Include already-consolidated rows in assembled context.",
    "MNEMOSYNE_CROSS_TIER_DEDUP": "A/B toggle for deduplication across working and episodic results.",
    "MNEMOSYNE_EMBEDDING_DOC_PREFIX": "Prefix prepended to documents before embedding. Applied verbatim.",
    "MNEMOSYNE_EMBEDDING_QUERY_PREFIX": "Prefix prepended to queries before embedding. Applied verbatim.",
    "MNEMOSYNE_EMBEDDING_THREADS": "Thread count for local ONNX embedding inference.",
    "MNEMOSYNE_EXTRACTION_MODEL": "Model used by the cloud extraction client.",
    "MNEMOSYNE_EXTRACTION_PROMPT": "Override the fact extraction prompt.",
    "MNEMOSYNE_FACT_BONUS": "A/B toggle for the fact match bonus in episodic scoring.",
    "MNEMOSYNE_GRAPH_BONUS": "A/B toggle for the graph traversal bonus in episodic scoring.",
    "MNEMOSYNE_HOST_LLM_TIMEOUT": "Timeout in seconds for host LLM backend calls.",
    "MNEMOSYNE_IMPORTED_WEIGHT": "Veracity multiplier for imported memories.",
    "MNEMOSYNE_INFERRED_WEIGHT": "Veracity multiplier for inferred memories.",
    "MNEMOSYNE_LEXICAL_GATE_MIN": "Override the lexical admission gate (float 0.0–1.0, clamped; non-finite/invalid values fall back to the historical query-length thresholds 0.15/0.5/0.3). `0.0` admits purely-vector candidates (recall-first, at a precision cost). Read on every recall call; retrieval stays local (sqlite-vec + JSON/NumPy embeddings, or lexical-only FTS5 when embeddings are unavailable).",
    "MNEMOSYNE_LLM_EXTRA_BODY": "JSON object merged last into every chat-completions request to the primary remote endpoint, for provider-specific keys the OpenAI shape has no name for (a thinking-mode toggle, provider routing). Unset, blank, invalid JSON or a non-object value means nothing is merged, and the reserved keys `messages`, `model` and `stream` are dropped from an otherwise valid object so the escape hatch cannot silently change the request the logs describe. Read at import.",
    "MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY": "Same as `MNEMOSYNE_LLM_EXTRA_BODY`, for requests to the fallback endpoint (`MNEMOSYNE_LLM_FALLBACK_MODELS`). Read at import.",
    "MNEMOSYNE_MCP_ALLOWED_HOSTS": "Comma-separated Host header values allowed for a non-loopback MCP Streamable HTTP bind (exact names or `name:*` patterns covering any port). A non-loopback bind exposes the selected local SQLite-backed memory bank to network clients, so this variable and `MNEMOSYNE_MCP_TOKEN` together form the security boundary. Required to start a non-loopback server; requests presenting any other Host are rejected with HTTP 421.",
    "MNEMOSYNE_MCP_ALLOWED_ORIGINS": "Optional comma-separated browser Origins allowed for a non-loopback MCP Streamable HTTP bind. Requests with no Origin header (CLI/SDK clients) always pass; any Origin not listed is rejected with HTTP 403. Origins only narrow the exposure boundary; the token and Host policy above are always required.",
    "MNEMOSYNE_MCP_BANK": "Memory bank used by the MCP server.",
    "MNEMOSYNE_MCP_TOKEN": "Bearer token for MCP HTTP (SSE and Streamable HTTP) auth. Required for any non-loopback bind, which exposes the selected local SQLite-backed memory bank to network clients.",
    "MNEMOSYNE_MODEL_CACHE_DIR": "Directory holding the local GGUF consolidation model. Unset or blank keeps `~/.hermes/mnemosyne/models`. Read at import; an explicitly set path is authoritative, so an unusable one fails rather than falling back to the default.",
    "MNEMOSYNE_PERSONA_FILE": "Path to an external persona facts file.",
    "MNEMOSYNE_PREFETCH_MODEL_SLOT_LIMIT": "Maximum canonical slots prefetched per turn.",
    "MNEMOSYNE_PREFETCH_MODEL_SLOT_MIN_OVERLAP": "Minimum token overlap for a canonical slot to count as relevant.",
    "MNEMOSYNE_PREFETCH_PROFILE": "Prefetch profile name, for example `general` or `coding`.",
    "MNEMOSYNE_SESSION_END_TIMEOUT": "Seconds allowed for end-of-session memory writes.",
    "MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT": "Seconds allowed to drain pending writes on shutdown.",
    "MNEMOSYNE_STATED_WEIGHT": "Veracity multiplier for directly stated memories.",
    "MNEMOSYNE_SYNC_KEY_SOURCE": "Where the sync passphrase comes from: `keyring` or `prompt`.",
    "MNEMOSYNE_SYNC_MODE": "Sync mode selector used by the Hermes sync adapter.",
    "MNEMOSYNE_SYNC_TOKEN": "Bearer token presented to the sync server.",
    "MNEMOSYNE_SYNC_TURN_SLOW_THRESHOLD": "Milliseconds after which a turn sync is logged as slow.",
    "MNEMOSYNE_TOOL_WEIGHT": "Veracity multiplier for tool-produced memories.",
    "MNEMOSYNE_UNKNOWN_WEIGHT": "Veracity multiplier for memories of unknown provenance.",
    "MNEMOSYNE_USE_CAVEMAN": "Use the AAAK keyword compression fallback for consolidation.",
    "MNEMOSYNE_VERACITY_MULTIPLIER": "A/B toggle for applying veracity multipliers at all.",
    "MNEMOSYNE_VOICE_FACT": "Set to `0` to disable the polyphonic fact voice.",
    "MNEMOSYNE_VOICE_GRAPH": "Set to `0` to disable the polyphonic graph voice.",
    "MNEMOSYNE_VOICE_TEMPORAL": "Set to `0` to disable the polyphonic temporal voice.",
    "MNEMOSYNE_VOICE_VECTOR": "Set to `0` to disable the polyphonic vector voice.",
}


# ---------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------
def _fmt_default(value) -> str:
    if value is None:
        return "*(none)*"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if value == "":
        return "*(unset)*"
    return "`{}`".format(value)


def _render_tool_schema(tools, version: str) -> str:
    mcp = [t for t in tools if t["mcp"]]
    plugin_only = [t for t in tools if not t["mcp"]]

    lines = [
        "---",
        'title: "MCP Tool Schema"',
        f"version: {version}",
        f"tool_count: {len(tools)}",
        f"mcp_tool_count: {len(mcp)}",
        f"plugin_only_tool_count: {len(plugin_only)}",
        'generated_at: "auto"',
        "---",
        "",
        f"# MCP Tool Schema (v{version})",
        "",
        "> **Generated file. Do not edit by hand.**",
        "> Source: `mnemosyne/tool_schemas.py` and `mnemosyne/mcp_tools.py`.",
        "> Regenerate with `python3 scripts/generate-docs.py`.",
        "",
        f"Mnemosyne declares **{len(tools)} tools**. Of those, **{len(mcp)} are callable over MCP** "
        f"(stdio, SSE and Streamable HTTP), and **{len(plugin_only)} are implemented only in the Hermes provider** "
        "and are not reachable through the MCP server.",
        "",
        "The split is real and worth respecting: calling a plugin-only tool over MCP raises "
        "`Unknown tool`. The counts above are derived from "
        "`mnemosyne/mcp_tools.py::_TOOL_HANDLERS`, so they cannot drift from the code.",
        "",
        "---",
        "",
        f"## Tools available over MCP ({len(mcp)})",
        "",
    ]
    lines += _render_tool_list(mcp)

    if plugin_only:
        lines += [
            "---",
            "",
            f"## Hermes plugin only, NOT available over MCP ({len(plugin_only)})",
            "",
            "These schemas exist in `mnemosyne/tool_schemas.py` and are implemented by the "
            "Hermes provider adapters under `hermes_memory_provider/`. They have no entry in "
            "`_TOOL_HANDLERS`, so an MCP client that calls them gets an error.",
            "",
        ]
        lines += _render_tool_list(plugin_only)

    return "\n".join(lines).rstrip() + "\n"


def _render_tool_list(tools) -> list:
    lines = []
    for i, t in enumerate(tools, 1):
        lines.append(f"### {i}. `{t['name']}`")
        lines.append("")
        if t["description"]:
            lines.append(t["description"])
            lines.append("")
        if t["props"]:
            lines.append("| Parameter | Type | Required | Default |")
            lines.append("|-----------|------|----------|---------|")
            for pname, spec in t["props"].items():
                spec = spec if isinstance(spec, dict) else {}
                required = "yes" if pname in t["required"] else "no"
                default = _fmt_default(spec["default"]) if "default" in spec else "*(none)*"
                lines.append("| `{}` | `{}` | {} | {} |".format(
                    pname, _type_of(spec), required, default))
            lines.append("")
        else:
            lines.append("*No parameters.*")
            lines.append("")
    return lines


def _render_config(env_map, defaults, restart, version: str, effective=None) -> str:
    effective = effective or {}
    lines = [
        "---",
        'title: "Configuration"',
        f"version: {version}",
        f"config_key_count: {len(env_map)}",
        f"env_only_count: {len(ENV_ONLY_DESCRIPTIONS)}",
        'generated_at: "auto"',
        "---",
        "",
        "# Configuration",
        "",
        "> **Generated file. Do not edit by hand.**",
        "> Source: `mnemosyne/core/config.py` (`ENV_VAR_MAP`, `DEFAULTS`, `REQUIRES_RESTART`).",
        "> Regenerate with `python3 scripts/generate-docs.py`.",
        "",
        "Mnemosyne reads configuration from a YAML file and from environment variables.",
        "Precedence is **`config.yaml` > environment variable > built-in default**.",
        "",
        "`config.yaml` lives at `$MNEMOSYNE_DATA_DIR/config.yaml`, falling back to",
        "`$HERMES_HOME/mnemosyne/config.yaml` and then `~/.hermes/mnemosyne/config.yaml`.",
        "Nested YAML works, and leaf keys resolve too, so `memory.mnemosyne.wm_max_items`",
        "and `wm_max_items` are equivalent.",
        "",
        "Keys marked **restart** are read once at startup. Changing them at runtime warns",
        "and does not take effect until the process restarts.",
        "",
        f"There are **{len(env_map)} configuration keys**. A further",
        f"**{len(ENV_ONLY_DESCRIPTIONS)} environment variables** are read directly from the",
        "environment and are **not** settable in `config.yaml`; they are listed separately below.",
        "",
        "---",
        "",
        f"## Configuration keys ({len(env_map)})",
        "",
        "| Config key | Environment variable | Default | Restart | Description |",
        "|------------|----------------------|---------|---------|-------------|",
    ]
    for key in sorted(env_map):
        declared = _fmt_default(defaults.get(key))
        if key in effective:
            shown = f"`{effective[key][0]}` [^{key}] (declared {declared})"
        else:
            shown = declared
        lines.append("| `{}` | `{}` | {} | {} | {} |".format(
            key,
            env_map[key],
            shown,
            "**yes**" if key in restart else "no",
            CONFIG_DESCRIPTIONS.get(key, ""),
        ))

    if effective:
        lines += [
            "",
            f"### Keys whose effective default bypasses `config.py` ({len(effective)})",
            "",
            "For these keys a module-level constant reads the environment variable "
            "directly with its own fallback, so the value in `DEFAULTS` is not what "
            "the runtime uses and a `config.yaml` entry alone does not reach the "
            "module. Treat the environment variable as authoritative.",
            "",
            "This list is derived by scanning the package for "
            "`os.environ.get(\"MNEMOSYNE_...\", default)` and diffing against "
            "`DEFAULTS`, so it cannot fall out of date.",
            "",
        ]
        for key in sorted(effective):
            value, src = effective[key]
            prose = EFFECTIVE_DEFAULT_PROSE.get(key, "")
            tail = f" {prose}" if prose else ""
            lines.append(
                f"[^{key}]: `{key}` -- effective default `{value}`, set in "
                f"`{src}`, not the `{_fmt_default(defaults.get(key))}` declared in "
                f"`config.py`.{tail}"
            )

    lines += [
        "",
        "---",
        "",
        f"## Environment-only variables ({len(ENV_ONLY_DESCRIPTIONS)})",
        "",
        "These are read with `os.environ` at their point of use and bypass",
        "`MnemosyneConfig` entirely. They cannot be set in `config.yaml`.",
        "",
        "| Environment variable | Description |",
        "|----------------------|-------------|",
    ]
    for name in sorted(ENV_ONLY_DESCRIPTIONS):
        lines.append("| `{}` | {} |".format(name, ENV_ONLY_DESCRIPTIONS[name]))

    lines += [
        "",
        "---",
        "",
        "## Recall scoring weights",
        "",
        "`vec_weight`, `fts_weight`, and `importance_weight` are normalized to sum to 1.0",
        "at query time. They are the highest-leverage tuning knobs in the system.",
        "",
        "They resolve at request time as `config.yaml > MNEMOSYNE_*_WEIGHT > defaults`.",
        "A config reload applies to the next request, and enhanced-recall cache entries",
        "are isolated by the effective weight snapshot.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _render_config_table_block(env_map, defaults, restart) -> str:
    """A standalone markdown table for injection into a hand-written page."""
    rows = [
        "| Config key | Environment variable | Default | Restart | Description |",
        "|------------|----------------------|---------|---------|-------------|",
    ]
    for key in sorted(env_map):
        rows.append("| `{}` | `{}` | {} | {} | {} |".format(
            key,
            env_map[key],
            _fmt_default(defaults.get(key)),
            "**yes**" if key in restart else "no",
            CONFIG_DESCRIPTIONS.get(key, ""),
        ))
    return "\n".join(rows)


START_MARKER = "{/* GENERATED_CONFIG_TABLE */}"
END_MARKER = "{/* /GENERATED_CONFIG_TABLE */}"


def _inject_config_table(page_path: str, table_md: str) -> bool:
    """
    Replace the generated block IN PLACE, preserving its position.

    The previous implementation stripped the markers and appended the block
    to the end of the file, which pushed a headerless table below the page's
    closing section. Position is now preserved, and the emitted block is a
    complete table with a header row.
    """
    if not os.path.isfile(page_path):
        print(f"  skip: config page not found at {page_path}")
        return False

    with open(page_path, encoding="utf-8") as f:
        content = f.read()

    block = f"{START_MARKER}\n{table_md}\n{END_MARKER}"

    # Strip every existing block first, in both marker dialects, then insert
    # one clean block at the position the first one occupied. Doing this in a
    # single pass matters: the replacement text itself contains the markers,
    # so a naive replace-in-a-while-loop re-matches its own output and deletes
    # the block it just wrote.
    insert_at = None
    for start_v, end_v in (
        ("<!-- GENERATED_CONFIG_TABLE -->", "<!-- /GENERATED_CONFIG_TABLE -->"),
        (START_MARKER, END_MARKER),
    ):
        while start_v in content and end_v in content:
            i = content.index(start_v)
            j = content.index(end_v, i) + len(end_v)
            if insert_at is None or i < insert_at:
                insert_at = i
            content = content[:i] + content[j:]

    replaced = insert_at is not None
    if replaced:
        content = content[:insert_at] + block + content[insert_at:]

    if not replaced:
        print(f"  skip: no GENERATED_CONFIG_TABLE markers in {page_path}")
        return False

    with open(page_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def _validate_descriptions(env_map) -> None:
    """Prose must cover exactly the keys the code declares."""
    documented = set(CONFIG_DESCRIPTIONS)
    actual = set(env_map)
    missing = sorted(actual - documented)
    extra = sorted(documented - actual)
    if missing or extra:
        print("FATAL: CONFIG_DESCRIPTIONS is out of sync with config.py::ENV_VAR_MAP")
        for k in missing:
            print(f"  missing description for new key: {k}")
        for k in extra:
            print(f"  description for removed key:     {k}")
        print("")
        print("Add or remove entries in CONFIG_DESCRIPTIONS in scripts/generate-docs.py.")
        sys.exit(1)

    stale_notes = sorted(set(EFFECTIVE_DEFAULT_PROSE) - actual)
    if stale_notes:
        print("FATAL: EFFECTIVE_DEFAULT_PROSE references keys that no longer exist:")
        for k in stale_notes:
            print(f"  {k}")
        sys.exit(1)


def main() -> None:
    check_only = "--check" in sys.argv

    version = _version()
    tools = _collect_tools()
    env_map, defaults, restart = _collect_config()
    _validate_descriptions(env_map)
    effective = _scan_effective_defaults(env_map, defaults)

    mcp_count = sum(1 for t in tools if t["mcp"])
    schema_mdx = _render_tool_schema(tools, version)
    config_mdx = _render_config(env_map, defaults, restart, version, effective)

    if check_only:
        print(f"v{version} | {len(tools)} tools ({mcp_count} over MCP) | "
              f"{len(env_map)} config keys | {len(ENV_ONLY_DESCRIPTIONS)} env-only | "
              f"{len(effective)} effective-default divergences")
        return

    canonical = os.path.join(REPO_ROOT, "docs", "api")
    os.makedirs(canonical, exist_ok=True)

    with open(os.path.join(canonical, "tool-schema.mdx"), "w", encoding="utf-8") as f:
        f.write(schema_mdx)
    print(f"wrote docs/api/tool-schema.mdx  ({len(tools)} tools, {mcp_count} over MCP)")

    with open(os.path.join(canonical, "configuration.mdx"), "w", encoding="utf-8") as f:
        f.write(config_mdx)
    print(f"wrote docs/api/configuration.mdx ({len(env_map)} config keys, "
          f"{len(ENV_ONLY_DESCRIPTIONS)} env-only)")

    # Docs sibling repo. content/ is the source of truth; the src/app route
    # re-exports it, so only content/ is written here.
    sibling = os.path.normpath(os.path.join(REPO_ROOT, "..", "mnemosyne-docs"))
    if not os.path.isdir(sibling):
        print("skip: ../mnemosyne-docs not checked out (CI runner ok)")
    else:
        tool_page = os.path.join(sibling, "content", "api", "tool-schema.mdx")
        if os.path.isdir(os.path.dirname(tool_page)):
            with open(tool_page, "w", encoding="utf-8") as f:
                f.write(schema_mdx)
            print("wrote ../mnemosyne-docs/content/api/tool-schema.mdx")

        table = _render_config_table_block(env_map, defaults, restart)
        for rel in (
            os.path.join("content", "getting-started", "configuration.mdx"),
            os.path.join("src", "app", "(docs)", "getting-started", "configuration", "page.mdx"),
        ):
            path = os.path.join(sibling, rel)
            if _inject_config_table(path, table):
                print(f"injected config table -> ../mnemosyne-docs/{rel}")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
