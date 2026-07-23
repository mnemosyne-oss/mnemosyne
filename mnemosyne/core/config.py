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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config schema — every known key with metadata
# ---------------------------------------------------------------------------

# Keys that require a process restart to take effect.
# Changing them via config.yaml at runtime will warn but not apply.
REQUIRES_RESTART: set[str] = {
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
ENV_VAR_MAP: dict[str, str] = {
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
    # LLM / consolidation
    "llm_repo": "MNEMOSYNE_LLM_REPO",
    "llm_file": "MNEMOSYNE_LLM_FILE",
    "llm_ctx": "MNEMOSYNE_LLM_CTX",
    "llm_threads": "MNEMOSYNE_LLM_THREADS",
    "llm_batch": "MNEMOSYNE_LLM_BATCH",
    "llm_temp": "MNEMOSYNE_LLM_TEMP",
    "llm_top_p": "MNEMOSYNE_LLM_TOP_P",
    "llm_top_k": "MNEMOSYNE_LLM_TOP_K",
    "llm_repeat_penalty": "MNEMOSYNE_LLM_REPEAT_PENALTY",
    "llm_n_gpu_layers": "MNEMOSYNE_LLM_N_GPU_LAYERS",
    "llm_offload_kqv": "MNEMOSYNE_LLM_OFFLOAD_KQV",
    "llm_flash_attn": "MNEMOSYNE_LLM_FLASH_ATTN",
    "llm_mmap": "MNEMOSYNE_LLM_MMAP",
    "llm_split_mode": "MNEMOSYNE_LLM_SPLIT_MODE",
    "llm_main_gpu": "MNEMOSYNE_LLM_MAIN_GPU",
    "llm_use_mlock": "MNEMOSYNE_LLM_USE_MLOCK",
    "llm_mmap_disable": "MNEMOSYNE_LLM_MMAP_DISABLE",
    "llm_extra_args": "MNEMOSYNE_LLM_EXTRA_ARGS",
    "llm_enabled": "MNEMOSYNE_LLM_ENABLED",
    "llm_base_url": "MNEMOSYNE_LLM_BASE_URL",
    "llm_api_key": "MNEMOSYNE_LLM_API_KEY",
    "llm_model": "MNEMOSYNE_LLM_MODEL",
    "shmr_max_clusters": "MNEMOSYNE_SHMR_MAX_CLUSTERS",
    "shmr_max_items_per_cluster": "MNEMOSYNE_SHMR_MAX_ITEMS_PER_CLUSTER",
    "shmr_prompt_template": "MNEMOSYNE_SHMR_PROMPT_TEMPLATE",
    "shmr_min_cluster_size": "MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE",
    "shmr_llm_timeout": "MNEMOSYNE_SHMR_LLM_TIMEOUT",
    # Working memory
    "wm_max_items": "MNEMOSYNE_WM_MAX_ITEMS",
    "wm_importance_floor": "MNEMOSYNE_WM_IMPORTANCE_FLOOR",
    "wm_max_chars": "MNEMOSYNE_WM_MAX_CHARS",
    "wm_dedup_enabled": "MNEMOSYNE_WM_DEDUP_ENABLED",
    "wm_dedup_threshold": "MNEMOSYNE_WM_DEDUP_THRESHOLD",
    "wm_dedup_window": "MNEMOSYNE_WM_DEDUP_WINDOW",
    "wm_evict_policy": "MNEMOSYNE_WM_EVICT_POLICY",
    "wm_promote_threshold": "MNEMOSYNE_WM_PROMOTE_THRESHOLD",
    "wm_temporal_decay": "MNEMOSYNE_WM_TEMPORAL_DECAY",
    # Episodic
    "ep_gist_model": "MNEMOSYNE_EP_GIST_MODEL",
    "ep_gist_prompt": "MNEMOSYNE_EP_GIST_PROMPT",
    "ep_gist_max_chars": "MNEMOSYNE_EP_GIST_MAX_CHARS",
    "ep_gist_min_chars": "MNEMOSYNE_EP_GIST_MIN_CHARS",
    "ep_embedding_batch": "MNEMOSYNE_EP_EMBEDDING_BATCH",
    "ep_gist_interval": "MNEMOSYNE_EP_GIST_INTERVAL",
    "ep_max_gists_per_session": "MNEMOSYNE_EP_MAX_GISTS_PER_SESSION",
    # Memory consolidation
    "sleep_interval": "MNEMOSYNE_SLEEP_INTERVAL",
    "sleep_max_age": "MNEMOSYNE_SLEEP_MAX_age",
    "sleep_min_items": "MNEMOSYNE_SLEEP_MIN_ITEMS",
    "sleep_force": "MNEMOSYNE_SLEEP_FORCE",
    # Hybrid recall
    "binary_bonus": "MNEMOSYNE_BINARY_BONUS",
    "binary_threshold": "MNEMOSYNE_BINARY_THRESHOLD",
    "binary_bonus_weight": "MNEMOSYNE_BINARY_BONUS_WEIGHT",
    # Sync
    "sync_roles": "MNEMOSYNE_SYNC_ROLES",
    "skip_contexts": "MNEMOSYNE_SKIP_CONTEXTS",
    "sync_remote": "MNEMOSYNE_SYNC_REMOTE",
    "sync_port": "MNEMOSYNE_SYNC_PORT",
    "sync_host": "MNEMOSYNE_SYNC_HOST",
    # Authentication
    "author_id": "MNEMOSYNE_AUTHOR_ID",
    "author_type": "MNEMOSYNE_AUTHOR_TYPE",
    "channel_id": "MNEMOSYNE_CHANNEL_ID",
    "mcp_bank": "MNEMOSYNE_MCP_BANK",
    "default_owner": "MNEMOSYNE_DEFAULT_OWNER",
    # Default scope for remember()
    "default_scope": "MNEMOSYNE_DEFAULT_SCOPE",
    # Filters
    "ignore_patterns": "MNEMOSYNE_IGNORE_PATTERNS",
    "write_classifier": "MNEMOSYNE_WRITE_CLASSIFIER",
}


CONFIG_KEY_MAP: dict[str, str] = {v: k for k, v in ENV_VAR_MAP.items()}

DEFAULTS: dict[str, Any] = {
    # Paths
    "data_dir": None,
    "home": None,
    "db_path": None,
    "backup_dir": None,
    "blob_dir": None,
    "shared_db_path": None,
    # Embeddings
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "embedding_api_key": "",
    "embedding_api_url": "",
    "embeddings_via_api": False,
    "no_embeddings": False,
    "skip_embeddings": False,
    "embeddings_off": False,
    "fastembed_cache_dir": "",
    "vec_type": "json",
    "vec_weight": 0.5,
    # Recall
    "fts_weight": 0.3,
    "importance_weight": 0.2,
    "temporal_halflife_hours": 24.0,
    # LLM / consolidation
    "llm_repo": None,
    "llm_file": None,
    "llm_ctx": 4096,
    "llm_threads": 4,
    "llm_batch": 512,
    "llm_temp": 0.3,
    "llm_top_p": 0.95,
    "llm_top_k": 40,
    "llm_repeat_penalty": 1.1,
    "llm_n_gpu_layers": -1,
    "llm_offload_kqv": True,
    "llm_flash_attn": True,
    "llm_mmap": True,
    "llm_split_mode": "none",
    "llm_main_gpu": 0,
    "llm_use_mlock": True,
    "llm_mmap_disable": False,
    "llm_extra_args": "",
    "llm_enabled": True,
    "llm_base_url": "http://localhost:11434/v1",
    "llm_api_key": "",
    "llm_model": "nemotron-3-ultra",
    "shmr_max_clusters": 8,
    "shmr_max_items_per_cluster": 25,
    "shmr_prompt_template": "",
    "shmr_min_cluster_size": 2,
    "shmr_llm_timeout": 30.0,
    # Working memory
    "wm_max_items": 10000,
    "wm_importance_floor": 0.05,
    "wm_max_chars": 12000,
    "wm_dedup_enabled": True,
    "wm_dedup_threshold": 0.95,
    "wm_dedup_window": 2000,
    "wm_evict_policy": "importance",
    "wm_promote_threshold": 0.7,
    "wm_temporal_decay": 0.01,
    # Episodic
    "ep_gist_model": "",
    "ep_gist_prompt": "",
    "ep_gist_max_chars": 500,
    "ep_gist_min_chars": 50,
    "ep_embedding_batch": 32,
    "ep_gist_interval": 300,
    "ep_max_gists_per_session": 50,
    # Memory consolidation
    "sleep_interval": 3600,
    "sleep_max_age": 86400,
    "sleep_min_items": 5,
    "sleep_force": False,
    # Hybrid recall
    "binary_bonus": True,
    "binary_threshold": 0.25,
    "binary_bonus_weight": 1.0,
    # Sync
    "sync_roles": "user,assistant",
    "skip_contexts": "",
    "sync_remote": None,
    "sync_port": 8080,
    "sync_host": "0.0.0.0",
    # Authentication
    "author_id": None,
    "author_type": None,
    "channel_id": None,
    "mcp_bank": "default",
    "default_owner": None,
    # Default scope for remember()
    "default_scope": "session",
    # Filters
    "ignore_patterns": "",
    "write_classifier": "",
}


def _default_config_path() -> Path:
    """Resolve the config.yaml path."""
    data_dir = os.environ.get("MNEMOSYNE_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "config.yaml"
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home) / "mnemosyne" / "config.yaml"
    return Path.home() / ".hermes" / "mnemosyne" / "config.yaml"


@dataclass(frozen=True)
class BeamRuntimeConfig:
    """Typed runtime settings consumed by Beam on each configuration lookup."""

    cross_session: bool


class MnemosyneConfig:
    """Central config reader with YAML + env var + defaults precedence.

    Thread-safe singleton. Call get_config() to get the shared instance.
    """

    _instance: Optional["MnemosyneConfig"] = None
    _lock = threading.Lock()

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or _default_config_path()
        self._yaml_cache: dict[str, Any] = {}
        self._yaml_mtime: float = 0.0
        self._yaml_lock = threading.Lock()

        # Auto-seed config.yaml on first access if it doesn't exist
        if not self._config_path.exists():
            self._seed()
        else:
            self._warn_legacy_provider_defaults()

        self._load_yaml()

    @classmethod
    def get_instance(cls) -> "MnemosyneConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _seed(self) -> None:
        """Create config.yaml with sensible defaults, respecting existing env vars.

        When the file doesn't exist, creates it with every known key.
        For each key: if the corresponding env var is set, that value is used.
        Otherwise the hardcoded default is used.

        This ensures that users with existing env var configurations don't
        get silently overridden by the auto-seeded defaults. The resulting
        config.yaml reflects exactly what's already running. The file
        does NOT overwrite an existing file.
        Returns without error if the file already exists.
        """
        if self._config_path.exists():
            return

        import yaml
        try:
            # Build the seed data: env var value if set, otherwise default
            seed_data: dict[str, Any] = {}
            for key, default_val in DEFAULTS.items():
                env_var = ENV_VAR_MAP.get(key)
                if env_var and env_var in os.environ:
                    env_val = os.environ[env_var]
                    # Type-coerce env vars to match the default type
                    if isinstance(default_val, bool):
                        seed_data[key] = env_val.strip().lower() in ("1", "true", "yes", "on")
                    elif isinstance(default_val, int):
                        try:
                            seed_data[key] = int(env_val)
                        except ValueError:
                            seed_data[key] = default_val
                    elif isinstance(default_val, float):
                        try:
                            seed_data[key] = float(env_val)
                        except ValueError:
                            seed_data[key] = default_val
                    else:
                        seed_data[key] = env_val
                else:
                    seed_data[key] = default_val

            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w") as f:
                f.write("# Mnemosyne config — edit freely, hot-reload with `mnemosyne config reload`\n")
                f.write("# Precedence: config.yaml > env vars > hardcoded defaults\n")
                f.write("# Values below reflect your current env vars where set, otherwise defaults.\n")
                f.write("# Run `mnemosyne config migrate` to re-export env vars at any time.\n\n")
                yaml.dump(seed_data, f, default_flow_style=False, sort_keys=True)

            env_count = sum(1 for k in seed_data if ENV_VAR_MAP.get(k, "") in os.environ)
            logger.info("Seeded config.yaml at %s (%d keys, %d from env vars)",
                         self._config_path, len(seed_data), env_count)
        except Exception as e:
            logger.warning("Failed to seed config.yaml: %s", e)

    def _warn_legacy_provider_defaults(self) -> None:
        """Warn about ambiguous 3.12.1/3.12.2 auto-seeded provider values.

        The legacy seed header does not record whether values came from the
        environment, so rewriting the file could destroy an explicit opt-in.
        Preserve it and provide deterministic commands for adopting the safer
        defaults.
        """
        try:
            text = self._config_path.read_text(encoding="utf-8")
            if "# Values below reflect your current env vars" not in text:
                return

            import yaml

            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict):
                return
            if (
                data.get("sync_roles") == "user,assistant"
                and data.get("skip_contexts") == ""
            ):
                logger.warning(
                    "Legacy provider defaults detected in %s; values may be "
                    "explicit environment choices and were not rewritten. "
                    "To adopt safe defaults, run: mnemosyne config set "
                    "sync_roles user && mnemosyne config set skip_contexts "
                    "cron,flush,subagent,background,skill_loop",
                    self._config_path,
                )
        except Exception as e:
            logger.warning("Failed to inspect legacy provider defaults: %s", e)

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

    def _flatten_yaml(self, data: dict, prefix: str = "") -> dict[str, Any]:
        """Flatten nested YAML into dot-separated keys.

        Example: {memory: {mnemosyne: {wm_max_items: 5000}}}
        → {"memory.mnemosyne.wm_max_items": 5000}
        Also extracts the leaf key: {"wm_max_items": 5000}
        """
        result: dict[str, Any] = {}
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten_yaml(value, full_key))
                # Also store the leaf key for direct access
                result[key] = value
            else:
                result[full_key] = value
                result[key] = value
        return result

    def _read_yaml(self, key: str, default: Any = None) -> Any:
        """Read from YAML cache (dot-notation or leaf key)."""
        self._load_yaml()
        return self._yaml_cache.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with precedence: YAML > env var > default."""
        # 1. YAML cache
        yaml_val = self._read_yaml(key)
        if yaml_val is not None:
            return yaml_val

        # 2. Environment variable
        env_var = ENV_VAR_MAP.get(key)
        if env_var and env_var in os.environ:
            env_val = os.environ[env_var]
            # Try to coerce to the default's type
            default_val = DEFAULTS.get(key)
            if isinstance(default_val, bool):
                return env_val.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(default_val, int):
                try:
                    return int(env_val)
                except ValueError:
                    return default_val
            if isinstance(default_val, float):
                try:
                    return float(env_val)
                except ValueError:
                    return default_val
            return env_val

        # 3. Hardcoded default
        return default if default is not None else DEFAULTS.get(key)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean config value with type coercion."""
        val = self.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(val, int):
            return val != 0
        return bool(val)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer config value with type coercion."""
        val = self.get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float config value with type coercion."""
        val = self.get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_str(self, key: str, default: str = "") -> str:
        """Get a string config value."""
        val = self.get(key, default)
        return str(val) if val is not None else ""

    def set(self, key: str, value: Any) -> None:
        """Set a config key in YAML (persisted to disk)."""
        with self._yaml_lock:
            self._load_yaml()
            self._yaml_cache[key] = value
            self._write_yaml()

    def set_many(self, items: dict[str, Any]) -> None:
        """Set multiple config keys at once."""
        with self._yaml_lock:
            self._load_yaml()
            self._yaml_cache.update(items)
            self._write_yaml()

    def _write_yaml(self) -> None:
        """Write the current YAML cache to disk."""
        import yaml
        with open(self._config_path, "w") as f:
            f.write("# Mnemosyne config — edit freely, hot-reload with `mnemosyne config reload`\n")
            f.write("# Precedence: config.yaml > env vars > hardcoded defaults\n\n")
            yaml.dump(self._yaml_cache, f, default_flow_style=False, sort_keys=True)
        self._yaml_mtime = self._config_path.stat().st_mtime

    def reload(self) -> set[str]:
        """Reload config from disk. Returns set of changed keys."""
        with self._yaml_lock:
            old_keys = set(self._yaml_cache.keys())
            self._load_yaml()
            new_keys = set(self._yaml_cache.keys())
            return old_keys.symmetric_difference(new_keys)

    def migrate_from_env(self) -> list[str]:
        """Migrate all known env vars to config.yaml.
        Returns list of keys that were set."""
        migrated: list[str] = []
        for key, env_var in ENV_VAR_MAP.items():
            if env_var in os.environ:
                self.set(key, os.environ[env_var])
                migrated.append(key)
        return migrated

    def all_keys(self) -> list[str]:
        """Return all known config keys (from schema + YAML + env)."""
        keys: set[str] = set(DEFAULTS.keys())
        keys.update(self._yaml_cache.keys())
        for env_var in ENV_VAR_MAP.values():
            if env_var in os.environ:
                keys.add(CONFIG_KEY_MAP.get(env_var, env_var))
        return sorted(keys)

    def dump(self) -> dict[str, Any]:
        """Dump all resolved config values (YAML > env > defaults)."""
        result: dict[str, Any] = {}
        for key in self.all_keys():
            result[key] = self.get(key)
        return result

    def requires_restart(self, key: str) -> bool:
        """Check if a key requires a process restart to take effect."""
        return key in REQUIRES_RESTART


def get_config(config_path: Optional[Path] = None) -> MnemosyneConfig:
    """Get the singleton MnemosyneConfig instance."""
    if config_path is not None:
        return MnemosyneConfig(config_path)
    return MnemosyneConfig.get_instance()


def reload() -> set[str]:
    """Reload the singleton config from disk. Returns changed keys."""
    config = get_config()
    return config.reload()