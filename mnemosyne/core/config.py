"""
Central config reader for Mnemosyne.

Single source of truth with precedence: config.yaml > env vars > hardcoded defaults.
Mirrors the Hermes Agent config pattern.

Without a config.yaml file, behavior is identical to today (env vars only).
The config.yaml is purely additive — it overrides env vars, which override defaults.

Usage:
    from mnemosyne.core.config import get_config

    config = get_config()
    wm_max = config.get("wm_max_items", default=10000)
    config.set("wm_max_items", 5000)
    config.reload()
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config schema — every known key with metadata
# ---------------------------------------------------------------------------

# Keys that require a process restart to take effect.
# Changing them via config.yaml at runtime will warn but not apply.
REQUIRES_RESTART: Set[str] = {
    "data_dir",
    "db_path",
    "home",
    "shared_db_path",
    "backup_dir",
    "blob_dir",
    "embedding_model",
    "embedding_dim",
    "embedding_api_url",
    "fastembed_cache_dir",
    "vec_type",
    "llm_repo",
    "llm_file",
    "author_id",
    "author_type",
    "channel_id",
    "mcp_bank",
    "default_owner",
    "sync_host",
    "sync_port",
    "sync_remote",
}

# Mapping from config key (snake_case, no MNEMOSYNE_ prefix) to env var name.
# This is the canonical bridge between config.yaml keys and env vars.
ENV_VAR_MAP: Dict[str, str] = {
    # Paths
    "data_dir": "MNEMOSYNE_DATA_DIR",
    "home": "MNEMOSYNE_HOME",
    "db_path": "MNEMOSYNE_DB_PATH",
    "backup_dir": "MNEMOSYNE_BACKUP_DIR",
    "blob_dir": "MNEMOSYNE_BLOB_DIR",
    "shared_db_path": "MNEMOSYNE_SHARED_DB_PATH",
    # Embeddings
    "embedding_model": "MNEMOSYNE_EMBEDDING_MODEL",
    "embedding_dim": "MNEMOSYNE_EMBEDDING_DIM",
    "embedding_api_key": "MNEMOSYNE_EMBEDDING_API_KEY",
    "embedding_api_url": "MNEMOSYNE_EMBEDDING_API_URL",
    "embeddings_via_api": "MNEMOSYNE_EMBEDDINGS_VIA_API",
    "no_embeddings": "MNEMOSYNE_NO_EMBEDDINGS",
    "skip_embeddings": "MNEMOSYNE_SKIP_EMBEDDINGS",
    "embeddings_off": "MNEMOSYNE_EMBEDDINGS_OFF",
    "fastembed_cache_dir": "MNEMOSYNE_FASTEMBED_CACHE_DIR",
    "vec_type": "MNEMOSYNE_VEC_TYPE",
    "vec_weight": "MNEMOSYNE_VEC_WEIGHT",
    # Recall
    "fts_weight": "MNEMOSYNE_FTS_WEIGHT",
    "importance_weight": "MNEMOSYNE_IMPORTANCE_WEIGHT",
    "temporal_halflife_hours": "MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS",
    "recency_halflife": "MNEMOSYNE_RECENCY_HALFLIFE",
    "recall_extra_stopwords": "MNEMOSYNE_RECALL_EXTRA_STOPWORDS",
    "cross_session": "MNEMOSYNE_CROSS_SESSION",
    "polyphonic_recall": "MNEMOSYNE_POLYPHONIC_RECALL",
    "query_intent": "MNEMOSYNE_QUERY_INTENT",
    "fact_recall_enabled": "MNEMOSYNE_FACT_RECALL_ENABLED",
    "enhanced_recall": "MNEMOSYNE_ENHANCED_RECALL",
    "proactive_linking": "MNEMOSYNE_PROACTIVE_LINKING",
    "lenient_fact_match": "MNEMOSYNE_LENIENT_FACT_MATCH",
    "recall_diagnostics": "MNEMOSYNE_RECALL_DIAGNOSTICS",
    # Tiers
    "wm_max_items": "MNEMOSYNE_WM_MAX_ITEMS",
    "wm_ttl_hours": "MNEMOSYNE_WM_TTL_HOURS",
    "wm_bump_cap_hours": "MNEMOSYNE_WM_BUMP_CAP_HOURS",
    "wm_pinned_ids": "MNEMOSYNE_WM_PINNED_IDS",
    "ep_limit": "MNEMOSYNE_EP_LIMIT",
    "sleep_batch": "MNEMOSYNE_SLEEP_BATCH",
    "sp_max": "MNEMOSYNE_SP_MAX",
    "tier2_days": "MNEMOSYNE_TIER2_DAYS",
    "tier3_days": "MNEMOSYNE_TIER3_DAYS",
    "tier1_weight": "MNEMOSYNE_TIER1_WEIGHT",
    "tier2_weight": "MNEMOSYNE_TIER2_WEIGHT",
    "tier3_weight": "MNEMOSYNE_TIER3_WEIGHT",
    # Compression
    "smart_compress": "MNEMOSYNE_SMART_COMPRESS",
    "tier3_max_chars": "MNEMOSYNE_TIER3_MAX_CHARS",
    "degrade_batch": "MNEMOSYNE_DEGRADE_BATCH",
    # LLM
    "llm_enabled": "MNEMOSYNE_LLM_ENABLED",
    "llm_max_tokens": "MNEMOSYNE_LLM_MAX_TOKENS",
    "llm_n_threads": "MNEMOSYNE_LLM_N_THREADS",
    "llm_n_ctx": "MNEMOSYNE_LLM_N_CTX",
    "llm_repo": "MNEMOSYNE_LLM_REPO",
    "llm_file": "MNEMOSYNE_LLM_FILE",
    "llm_base_url": "MNEMOSYNE_LLM_BASE_URL",
    "llm_api_key": "MNEMOSYNE_LLM_API_KEY",
    "llm_model": "MNEMOSYNE_LLM_MODEL",
    "llm_timeout": "MNEMOSYNE_LLM_TIMEOUT",
    "llm_fallback_models": "MNEMOSYNE_LLM_FALLBACK_MODELS",
    "llm_fallback_base_url": "MNEMOSYNE_LLM_FALLBACK_BASE_URL",
    "llm_fallback_api_key": "MNEMOSYNE_LLM_FALLBACK_API_KEY",
    "force_local": "MNEMOSYNE_FORCE_LOCAL",
    "sleep_prompt": "MNEMOSYNE_SLEEP_PROMPT",
    "host_llm_enabled": "MNEMOSYNE_HOST_LLM_ENABLED",
    "host_llm_provider": "MNEMOSYNE_HOST_LLM_PROVIDER",
    "host_llm_model": "MNEMOSYNE_HOST_LLM_MODEL",
    "host_llm_n_ctx": "MNEMOSYNE_HOST_LLM_N_CTX",
    # Conflict detection
    "llm_conflict_detection": "MNEMOSYNE_LLM_CONFLICT_DETECTION",
    "conflict_llm_base_url": "MNEMOSYNE_CONFLICT_LLM_BASE_URL",
    "conflict_llm_api_key": "MNEMOSYNE_CONFLICT_LLM_API_KEY",
    "conflict_llm_model": "MNEMOSYNE_CONFLICT_LLM_MODEL",
    # Sync
    "sync_remote": "MNEMOSYNE_SYNC_REMOTE",
    "sync_host": "MNEMOSYNE_SYNC_HOST",
    "sync_port": "MNEMOSYNE_SYNC_PORT",
    "sync_key": "MNEMOSYNE_SYNC_KEY",
    "sync_encrypt": "MNEMOSYNE_SYNC_ENCRYPT",
    "sync_roles": "MNEMOSYNE_SYNC_ROLES",
    # Provider
    "auto_sleep_enabled": "MNEMOSYNE_AUTO_SLEEP_ENABLED",
    "reflect_disabled_for_cron": "MNEMOSYNE_REFLECT_DISABLED_FOR_CRON",
    "reflect_max_calls_per_session": "MNEMOSYNE_REFLECT_MAX_CALLS_PER_SESSION",
    "skip_contexts": "MNEMOSYNE_SKIP_CONTEXTS",
    "prefetch_content_chars": "MNEMOSYNE_PREFETCH_CONTENT_CHARS",
    "sync_turn_user_limit": "MNEMOSYNE_SYNC_TURN_USER_LIMIT",
    "sync_turn_assistant_limit": "MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT",
    # Persona
    "persona_enabled": "MNEMOSYNE_PERSONA_ENABLED",
    "persona_token_cap": "MNEMOSYNE_PERSONA_TOKEN_CAP",
    "persona_interval": "MNEMOSYNE_PERSONA_INTERVAL",
    "persona_daily_sync_hour": "MNEMOSYNE_PERSONA_DAILY_SYNC_HOUR",
    # Model refresh
    "sleep_model_refresh_enabled": "MNEMOSYNE_SLEEP_MODEL_REFRESH_ENABLED",
    "sleep_model_refresh_auto_apply": "MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY",
    "sleep_model_refresh_categories": "MNEMOSYNE_SLEEP_MODEL_REFRESH_CATEGORIES",
    "sleep_model_refresh_max_tokens": "MNEMOSYNE_SLEEP_MODEL_REFRESH_MAX_TOKENS",
    "sleep_model_refresh_temperature": "MNEMOSYNE_SLEEP_MODEL_REFRESH_TEMPERATURE",
    "sleep_model_refresh_auto_apply_min_confidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY_MIN_CONFIDENCE",
    "sleep_model_refresh_min_evidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE",
    "sleep_model_refresh_conflict_min_confidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_CONFIDENCE",
    "sleep_model_refresh_conflict_min_evidence": "MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE",
    # SHMR
    "shmr_batch_size": "MNEMOSYNE_SHMR_BATCH_SIZE",
    "shmr_max_iterations": "MNEMOSYNE_SHMR_MAX_ITERATIONS",
    "shmr_similarity_threshold": "MNEMOSYNE_SHMR_SIMILARITY_THRESHOLD",
    "shmr_harmony_threshold": "MNEMOSYNE_SHMR_HARMONY_THRESHOLD",
    "shmr_model": "MNEMOSYNE_SHMR_MODEL",
    "shmr_min_cluster_size": "MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE",
    "shmr_temperature": "MNEMOSYNE_SHMR_TEMPERATURE",
    # Migrations
    "auto_migrate": "MNEMOSYNE_AUTO_MIGRATE",
    # MCP
    "default_scope": "MNEMOSYNE_DEFAULT_SCOPE",
    "default_owner": "MNEMOSYNE_DEFAULT_OWNER",
    # Filters
    "ignore_patterns": "MNEMOSYNE_IGNORE_PATTERNS",
    "write_classifier": "MNEMOSYNE_WRITE_CLASSIFIER",
}

# Reverse map: env var name → config key
CONFIG_KEY_MAP: Dict[str, str] = {v: k for k, v in ENV_VAR_MAP.items()}


def _default_config_path() -> Path:
    """Resolve the config.yaml path."""
    data_dir = os.environ.get("MNEMOSYNE_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "config.yaml"
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home) / "mnemosyne" / "config.yaml"
    return Path.home() / ".hermes" / "mnemosyne" / "config.yaml"


class MnemosyneConfig:
    """Central config reader with YAML + env var + defaults precedence.

    Thread-safe singleton. Call get_config() to get the shared instance.
    """

    _instance: Optional["MnemosyneConfig"] = None
    _lock = threading.Lock()

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or _default_config_path()
        self._yaml_cache: Dict[str, Any] = {}
        self._yaml_mtime: float = 0.0
        self._yaml_lock = threading.Lock()
        self._load_yaml()

    @classmethod
    def get_instance(cls) -> "MnemosyneConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for tests)."""
        with cls._lock:
            cls._instance = None

    # -------------------------------------------------------------------
    # YAML loading
    # -------------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self._config_path

    def _load_yaml(self) -> None:
        """Load config.yaml into the cache if it exists and has changed."""
        with self._yaml_lock:
            try:
                if not self._config_path.exists():
                    self._yaml_cache = {}
                    self._yaml_mtime = 0.0
                    return
                mtime = self._config_path.stat().st_mtime
                if mtime == self._yaml_mtime and self._yaml_cache:
                    return  # unchanged
                import yaml
                with open(self._config_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                # Flatten nested YAML into dot-separated keys, but most
                # Mnemosyne config is flat key: value. Support both.
                self._yaml_cache = self._flatten_yaml(data)
                self._yaml_mtime = mtime
                logger.debug("Loaded config from %s (%d keys)",
                             self._config_path, len(self._yaml_cache))
            except Exception as e:
                logger.warning("Failed to load config.yaml: %s", e)
                self._yaml_cache = {}
                self._yaml_mtime = 0.0

    def _flatten_yaml(self, data: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested YAML into dot-separated keys.

        Example: {memory: {mnemosyne: {wm_max_items: 5000}}}
        → {"memory.mnemosyne.wm_max_items": 5000}
        Also extracts the leaf key: {"wm_max_items": 5000}
        """
        flat = {}
        for key, val in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                flat.update(self._flatten_yaml(val, full_key))
            else:
                flat[full_key] = val
                # Also store the leaf key (last segment)
                leaf = key
                flat.setdefault(leaf, val)
        return flat

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def reload(self) -> Set[str]:
        """Re-read config.yaml. Returns set of changed keys.

        Also checks for file mtime changes to skip unnecessary reloads.
        """
        old_values = dict(self._yaml_cache)
        self._yaml_mtime = 0.0  # force reload
        self._load_yaml()

        changed = set()
        for key in set(list(old_values.keys()) + list(self._yaml_cache.keys())):
            old_val = old_values.get(key)
            new_val = self._yaml_cache.get(key)
            if old_val != new_val:
                changed.add(key)

        # Warn about requires_restart keys
        for key in changed:
            config_key = self._yaml_to_config_key(key)
            if config_key and config_key in REQUIRES_RESTART:
                logger.warning(
                    "Config key '%s' requires restart to take effect. "
                    "The new value will apply on next process start.",
                    config_key,
                )

        return changed

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value.

        Precedence: config.yaml > env var > default.

        Args:
            key: Config key (snake_case, no MNEMOSYNE_ prefix).
                 e.g. "wm_max_items", "vec_type", "llm_enabled".
            default: Fallback if not found in YAML or env.
        """
        # 1. Check YAML cache (refresh if file changed)
        self._load_yaml()
        if key in self._yaml_cache:
            return self._yaml_cache[key]

        # 2. Check env var
        env_var = ENV_VAR_MAP.get(key)
        if env_var:
            val = os.environ.get(env_var)
            if val is not None:
                return val

        # 3. Default
        return default

    def get_str(self, key: str, default: str = "") -> str:
        """Get a string config value."""
        val = self.get(key, default)
        return str(val) if val is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an int config value."""
        val = self.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float config value."""
        val = self.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean config value.

        Accepts: 1/true/yes/on (case-insensitive) as True.
        """
        val = self.get(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    def set(self, key: str, value: Any) -> None:
        """Write a config value to config.yaml.

        Creates the file if it doesn't exist.
        """
        self._load_yaml()

        # Read existing YAML
        import yaml
        existing: Dict[str, Any] = {}
        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                existing = {}

        # Set the key (flat structure for now)
        existing[key] = value

        # Ensure parent dir
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write back
        with open(self._config_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=True)

        # Refresh cache
        self._yaml_mtime = 0.0
        self._load_yaml()

        # Warn about requires_restart
        if key in REQUIRES_RESTART:
            logger.warning(
                "Config key '%s' requires restart to take effect.", key
            )

    def migrate_from_env(self) -> List[str]:
        """Export current env vars to config.yaml.

        Reads all MNEMOSYNE_* env vars and writes their values to config.yaml.
        Does not unset the env vars — they remain as lower-priority fallbacks.

        Returns list of keys that were migrated.
        """
        migrated = []
        for config_key, env_var in ENV_VAR_MAP.items():
            val = os.environ.get(env_var)
            if val is not None and val != "":
                self.set(config_key, val)
                migrated.append(config_key)

        logger.info("Migrated %d env vars to config.yaml", len(migrated))
        return migrated

    def _yaml_to_config_key(self, yaml_key: str) -> Optional[str]:
        """Map a YAML key (possibly dot-separated) to a config key."""
        # Direct match
        if yaml_key in ENV_VAR_MAP:
            return yaml_key
        # Try the last segment of a dot-separated key
        if "." in yaml_key:
            leaf = yaml_key.rsplit(".", 1)[-1]
            if leaf in ENV_VAR_MAP:
                return leaf
        return None

    def all_keys(self) -> List[str]:
        """Return all known config keys."""
        return sorted(ENV_VAR_MAP.keys())

    def dump(self) -> Dict[str, Any]:
        """Return all config values as a dict (for inspection)."""
        result = {}
        for key in self.all_keys():
            result[key] = self.get(key)
        return result


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_config() -> MnemosyneConfig:
    """Get the shared MnemosyneConfig singleton."""
    return MnemosyneConfig.get_instance()


def get(key: str, default: Any = None) -> Any:
    """Shortcut: get a config value from the shared instance."""
    return get_config().get(key, default)


def get_int(key: str, default: int = 0) -> int:
    return get_config().get_int(key, default)


def get_float(key: str, default: float = 0.0) -> float:
    return get_config().get_float(key, default)


def get_bool(key: str, default: bool = False) -> bool:
    return get_config().get_bool(key, default)


def get_str(key: str, default: str = "") -> str:
    return get_config().get_str(key, default)


def reload() -> Set[str]:
    """Shortcut: reload the shared config instance."""
    return get_config().reload()
