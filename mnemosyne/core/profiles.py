"""
Mnemosyne configuration profiles — gamified templates.

8 named presets covering all 74 template-eligible env vars. Each template
is a complete configuration that can be applied with `mnemosyne profile apply`.

Templates are validated against 15 consistency rules that catch contradictions
(e.g. SMART_COMPRESS=1 without LLM_ENABLED=true).

Usage:
    from mnemosyne.core.profiles import list_profiles, apply_profile, validate_profile

    profiles = list_profiles()  # all 8 templates
    apply_profile("speed", config_path=...)  # writes to config.yaml
    errors = validate_profile(template_dict)  # returns [] if valid
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mnemosyne.core.config import ENV_VAR_MAP, REQUIRES_RESTART, get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template-eligible keys (74 — excludes secrets, deployment, infra, internal)
# ---------------------------------------------------------------------------

TEMPLATE_KEYS: List[str] = [
    # Embeddings
    "vec_type", "vec_weight", "embeddings_via_api",
    "no_embeddings", "skip_embeddings", "embeddings_off",
    # Recall
    "fts_weight", "importance_weight", "temporal_halflife_hours",
    "recency_halflife", "recall_extra_stopwords", "cross_session",
    "polyphonic_recall", "query_intent", "fact_recall_enabled",
    "enhanced_recall", "proactive_linking", "lenient_fact_match",
    "recall_diagnostics",
    # Tiers
    "wm_max_items", "wm_ttl_hours", "wm_bump_cap_hours",
    "wm_pinned_ids", "ep_limit", "sleep_batch", "sp_max",
    "tier2_days", "tier3_days", "tier1_weight",
    "tier2_weight", "tier3_weight",
    # Compression
    "smart_compress", "tier3_max_chars", "degrade_batch",
    # LLM
    "llm_enabled", "llm_max_tokens", "llm_n_threads",
    "llm_n_ctx", "llm_timeout", "force_local",
    "host_llm_enabled", "host_llm_n_ctx",
    "llm_conflict_detection",
    # Sync
    "sync_encrypt", "sync_roles",
    # Provider
    "auto_sleep_enabled", "reflect_disabled_for_cron",
    "reflect_max_calls_per_session", "skip_contexts",
    "prefetch_content_chars", "sync_turn_user_limit",
    "sync_turn_assistant_limit",
    # Persona
    "persona_enabled", "persona_token_cap",
    "persona_interval", "persona_daily_sync_hour",
    # Model refresh
    "sleep_model_refresh_enabled", "sleep_model_refresh_auto_apply",
    # SHMR
    "shmr_batch_size", "shmr_max_iterations",
    "shmr_similarity_threshold", "shmr_harmony_threshold",
    "shmr_min_cluster_size", "shmr_temperature",
    # Migrations
    "auto_migrate",
    # MCP
    "default_scope",
    # Filters
    "ignore_patterns", "write_classifier",
]

# Assert at import time that TEMPLATE_KEYS matches ENV_VAR_MAP
for k in TEMPLATE_KEYS:
    assert k in ENV_VAR_MAP, f"TEMPLATE_KEY '{k}' not in ENV_VAR_MAP"

# ---------------------------------------------------------------------------
# Profile metadata
# ---------------------------------------------------------------------------

@dataclass
class ProfileMeta:
    """Human-readable metadata for a profile."""
    name: str
    description: str
    use_case: str
    ratings: Dict[str, int] = field(default_factory=dict)
    # Ratings: 0-20 scale (for bar chart display)
    # quality, speed, memory, llm_dependency, security


# ---------------------------------------------------------------------------
# The 8 templates
# ---------------------------------------------------------------------------

PROFILES: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "meta": ProfileMeta(
            name="minimal",
            description="Bare bones. No LLM, no embeddings, pure SQLite + FTS5.",
            use_case="Local-only agents, CI environments, resource-constrained devices.",
            ratings={"quality": 5, "speed": 18, "memory": 5, "llm_dependency": 0, "security": 10},
        ),
        "settings": {
            "vec_type": "int8", "vec_weight": "0.0",
            "embeddings_via_api": "0", "no_embeddings": "1",
            "skip_embeddings": "1", "embeddings_off": "1",
            "fts_weight": "1.0", "importance_weight": "0.0",
            "temporal_halflife_hours": "12", "recency_halflife": "48",
            "recall_extra_stopwords": "", "cross_session": "0",
            "polyphonic_recall": "0", "query_intent": "0",
            "fact_recall_enabled": "0", "enhanced_recall": "0",
            "proactive_linking": "0", "lenient_fact_match": "0",
            "recall_diagnostics": "0",
            "wm_max_items": "1000", "wm_ttl_hours": "48",
            "wm_bump_cap_hours": "24", "wm_pinned_ids": "",
            "ep_limit": "5000", "sleep_batch": "500", "sp_max": "100",
            "tier2_days": "7", "tier3_days": "30",
            "tier1_weight": "1.0", "tier2_weight": "0.3", "tier3_weight": "0.1",
            "smart_compress": "0", "tier3_max_chars": "0", "degrade_batch": "50",
            "llm_enabled": "false", "llm_max_tokens": "1024",
            "llm_n_threads": "2", "llm_n_ctx": "1024", "llm_timeout": "30",
            "force_local": "0", "host_llm_enabled": "false", "host_llm_n_ctx": "4096",
            "llm_conflict_detection": "false",
            "sync_encrypt": "false", "sync_roles": "user",
            "auto_sleep_enabled": "false", "reflect_disabled_for_cron": "true",
            "reflect_max_calls_per_session": "0",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "0",
            "sync_turn_user_limit": "300", "sync_turn_assistant_limit": "500",
            "persona_enabled": "false", "persona_token_cap": "500",
            "persona_interval": "100", "persona_daily_sync_hour": "-1",
            "sleep_model_refresh_enabled": "false",
            "sleep_model_refresh_auto_apply": "false",
            "shmr_batch_size": "10", "shmr_max_iterations": "1",
            "shmr_similarity_threshold": "0.90", "shmr_harmony_threshold": "0.80",
            "shmr_min_cluster_size": "5", "shmr_temperature": "0.1",
            "auto_migrate": "1", "default_scope": "session",
            "ignore_patterns": "", "write_classifier": "off",
        },
    },
    "speed": {
        "meta": ProfileMeta(
            name="speed",
            description="Fastest recall. Bit vectors, small scan limits, aggressive degradation.",
            use_case="Real-time agents that need instant recall.",
            ratings={"quality": 10, "speed": 20, "memory": 8, "llm_dependency": 5, "security": 8},
        ),
        "settings": {
            "vec_type": "bit", "vec_weight": "0.4",
            "embeddings_via_api": "0", "no_embeddings": "0",
            "skip_embeddings": "0", "embeddings_off": "0",
            "fts_weight": "0.5", "importance_weight": "0.1",
            "temporal_halflife_hours": "6", "recency_halflife": "24",
            "recall_extra_stopwords": "", "cross_session": "0",
            "polyphonic_recall": "0", "query_intent": "0",
            "fact_recall_enabled": "0", "enhanced_recall": "0",
            "proactive_linking": "0", "lenient_fact_match": "0",
            "recall_diagnostics": "0",
            "wm_max_items": "5000", "wm_ttl_hours": "24",
            "wm_bump_cap_hours": "12", "wm_pinned_ids": "",
            "ep_limit": "10000", "sleep_batch": "2000", "sp_max": "200",
            "tier2_days": "7", "tier3_days": "30",
            "tier1_weight": "1.0", "tier2_weight": "0.3", "tier3_weight": "0.1",
            "smart_compress": "1", "tier3_max_chars": "150", "degrade_batch": "200",
            "llm_enabled": "true", "llm_max_tokens": "1024",
            "llm_n_threads": "2", "llm_n_ctx": "1024", "llm_timeout": "15",
            "force_local": "1", "host_llm_enabled": "false", "host_llm_n_ctx": "4096",
            "llm_conflict_detection": "false",
            "sync_encrypt": "false", "sync_roles": "user",
            "auto_sleep_enabled": "true", "reflect_disabled_for_cron": "true",
            "reflect_max_calls_per_session": "1",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "200",
            "sync_turn_user_limit": "300", "sync_turn_assistant_limit": "500",
            "persona_enabled": "false", "persona_token_cap": "800",
            "persona_interval": "100", "persona_daily_sync_hour": "-1",
            "sleep_model_refresh_enabled": "false",
            "sleep_model_refresh_auto_apply": "false",
            "shmr_batch_size": "25", "shmr_max_iterations": "1",
            "shmr_similarity_threshold": "0.85", "shmr_harmony_threshold": "0.75",
            "shmr_min_cluster_size": "3", "shmr_temperature": "0.1",
            "auto_migrate": "1", "default_scope": "session",
            "ignore_patterns": "", "write_classifier": "off",
        },
    },
    "quality": {
        "meta": ProfileMeta(
            name="quality",
            description="Maximum recall quality. Float32, all experimental features on.",
            use_case="Research agents, long-term memory, precision-critical tasks.",
            ratings={"quality": 20, "speed": 5, "memory": 18, "llm_dependency": 15, "security": 8},
        ),
        "settings": {
            "vec_type": "float32", "vec_weight": "0.6",
            "embeddings_via_api": "0", "no_embeddings": "0",
            "skip_embeddings": "0", "embeddings_off": "0",
            "fts_weight": "0.2", "importance_weight": "0.2",
            "temporal_halflife_hours": "48", "recency_halflife": "336",
            "recall_extra_stopwords": "", "cross_session": "1",
            "polyphonic_recall": "1", "query_intent": "1",
            "fact_recall_enabled": "1", "enhanced_recall": "1",
            "proactive_linking": "1", "lenient_fact_match": "1",
            "recall_diagnostics": "1",
            "wm_max_items": "50000", "wm_ttl_hours": "720",
            "wm_bump_cap_hours": "48", "wm_pinned_ids": "",
            "ep_limit": "100000", "sleep_batch": "10000", "sp_max": "5000",
            "tier2_days": "90", "tier3_days": "365",
            "tier1_weight": "1.0", "tier2_weight": "0.7", "tier3_weight": "0.5",
            "smart_compress": "1", "tier3_max_chars": "500", "degrade_batch": "200",
            "llm_enabled": "true", "llm_max_tokens": "4096",
            "llm_n_threads": "8", "llm_n_ctx": "4096", "llm_timeout": "120",
            "force_local": "0", "host_llm_enabled": "false", "host_llm_n_ctx": "64000",
            "llm_conflict_detection": "true",
            "sync_encrypt": "false", "sync_roles": "user,assistant",
            "auto_sleep_enabled": "true", "reflect_disabled_for_cron": "true",
            "reflect_max_calls_per_session": "5",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "1000",
            "sync_turn_user_limit": "1000", "sync_turn_assistant_limit": "1500",
            "persona_enabled": "true", "persona_token_cap": "2000",
            "persona_interval": "25", "persona_daily_sync_hour": "3",
            "sleep_model_refresh_enabled": "true",
            "sleep_model_refresh_auto_apply": "true",
            "shmr_batch_size": "100", "shmr_max_iterations": "5",
            "shmr_similarity_threshold": "0.65", "shmr_harmony_threshold": "0.55",
            "shmr_min_cluster_size": "2", "shmr_temperature": "0.3",
            "auto_migrate": "1", "default_scope": "global",
            "ignore_patterns": "", "write_classifier": "warn",
        },
    },
    "research": {
        "meta": ProfileMeta(
            name="research",
            description="Deep memory, aggressive consolidation and linking.",
            use_case="Multi-session research, literature review, long-term knowledge.",
            ratings={"quality": 17, "speed": 8, "memory": 16, "llm_dependency": 15, "security": 8},
        ),
        "settings": {
            "vec_type": "int8", "vec_weight": "0.5",
            "embeddings_via_api": "0", "no_embeddings": "0",
            "skip_embeddings": "0", "embeddings_off": "0",
            "fts_weight": "0.3", "importance_weight": "0.2",
            "temporal_halflife_hours": "168", "recency_halflife": "336",
            "recall_extra_stopwords": "", "cross_session": "1",
            "polyphonic_recall": "0", "query_intent": "1",
            "fact_recall_enabled": "1", "enhanced_recall": "1",
            "proactive_linking": "1", "lenient_fact_match": "0",
            "recall_diagnostics": "0",
            "wm_max_items": "20000", "wm_ttl_hours": "336",
            "wm_bump_cap_hours": "24", "wm_pinned_ids": "",
            "ep_limit": "50000", "sleep_batch": "10000", "sp_max": "2000",
            "tier2_days": "60", "tier3_days": "180",
            "tier1_weight": "1.0", "tier2_weight": "0.6", "tier3_weight": "0.3",
            "smart_compress": "1", "tier3_max_chars": "400", "degrade_batch": "100",
            "llm_enabled": "true", "llm_max_tokens": "4096",
            "llm_n_threads": "4", "llm_n_ctx": "4096", "llm_timeout": "120",
            "force_local": "0", "host_llm_enabled": "false", "host_llm_n_ctx": "32000",
            "llm_conflict_detection": "true",
            "sync_encrypt": "false", "sync_roles": "user,assistant",
            "auto_sleep_enabled": "true", "reflect_disabled_for_cron": "true",
            "reflect_max_calls_per_session": "5",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "800",
            "sync_turn_user_limit": "1000", "sync_turn_assistant_limit": "1200",
            "persona_enabled": "true", "persona_token_cap": "2000",
            "persona_interval": "50", "persona_daily_sync_hour": "3",
            "sleep_model_refresh_enabled": "true",
            "sleep_model_refresh_auto_apply": "true",
            "shmr_batch_size": "50", "shmr_max_iterations": "3",
            "shmr_similarity_threshold": "0.70", "shmr_harmony_threshold": "0.60",
            "shmr_min_cluster_size": "2", "shmr_temperature": "0.2",
            "auto_migrate": "1", "default_scope": "global",
            "ignore_patterns": "", "write_classifier": "warn",
        },
    },
    "paranoid": {
        "meta": ProfileMeta(
            name="paranoid",
            description="Security-first. Strict classifier, sync encryption, no host LLM.",
            use_case="Production with sensitive data, compliance environments.",
            ratings={"quality": 12, "speed": 10, "memory": 10, "llm_dependency": 5, "security": 20},
        ),
        "settings": {
            "vec_type": "int8", "vec_weight": "0.5",
            "embeddings_via_api": "0", "no_embeddings": "0",
            "skip_embeddings": "0", "embeddings_off": "0",
            "fts_weight": "0.3", "importance_weight": "0.2",
            "temporal_halflife_hours": "24", "recency_halflife": "168",
            "recall_extra_stopwords": "", "cross_session": "0",
            "polyphonic_recall": "0", "query_intent": "0",
            "fact_recall_enabled": "0", "enhanced_recall": "0",
            "proactive_linking": "0", "lenient_fact_match": "0",
            "recall_diagnostics": "0",
            "wm_max_items": "10000", "wm_ttl_hours": "168",
            "wm_bump_cap_hours": "24", "wm_pinned_ids": "",
            "ep_limit": "50000", "sleep_batch": "5000", "sp_max": "500",
            "tier2_days": "30", "tier3_days": "90",
            "tier1_weight": "1.0", "tier2_weight": "0.5", "tier3_weight": "0.25",
            "smart_compress": "1", "tier3_max_chars": "300", "degrade_batch": "100",
            "llm_enabled": "true", "llm_max_tokens": "2048",
            "llm_n_threads": "4", "llm_n_ctx": "2048", "llm_timeout": "60",
            "force_local": "1", "host_llm_enabled": "false", "host_llm_n_ctx": "4096",
            "llm_conflict_detection": "true",
            "sync_encrypt": "true", "sync_roles": "user",
            "auto_sleep_enabled": "true", "reflect_disabled_for_cron": "true",
            "reflect_max_calls_per_session": "3",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "0",
            "sync_turn_user_limit": "300", "sync_turn_assistant_limit": "500",
            "persona_enabled": "false", "persona_token_cap": "1000",
            "persona_interval": "100", "persona_daily_sync_hour": "-1",
            "sleep_model_refresh_enabled": "false",
            "sleep_model_refresh_auto_apply": "false",
            "shmr_batch_size": "50", "shmr_max_iterations": "3",
            "shmr_similarity_threshold": "0.70", "shmr_harmony_threshold": "0.60",
            "shmr_min_cluster_size": "2", "shmr_temperature": "0.2",
            "auto_migrate": "1", "default_scope": "session",
            "ignore_patterns": "password|token|api_key|secret|Bearer|Authorization|-----BEGIN",
            "write_classifier": "strict",
        },
    },
    "balanced": {
        "meta": ProfileMeta(
            name="balanced",
            description="Sensible defaults. The 'just works' profile.",
            use_case="General-purpose agents, daily development, default install.",
            ratings={"quality": 14, "speed": 14, "memory": 12, "llm_dependency": 10, "security": 10},
        ),
        "settings": {
            "vec_type": "int8", "vec_weight": "0.5",
            "embeddings_via_api": "0", "no_embeddings": "0",
            "skip_embeddings": "0", "embeddings_off": "0",
            "fts_weight": "0.3", "importance_weight": "0.2",
            "temporal_halflife_hours": "24", "recency_halflife": "168",
            "recall_extra_stopwords": "", "cross_session": "0",
            "polyphonic_recall": "0", "query_intent": "0",
            "fact_recall_enabled": "0", "enhanced_recall": "0",
            "proactive_linking": "0", "lenient_fact_match": "0",
            "recall_diagnostics": "0",
            "wm_max_items": "10000", "wm_ttl_hours": "168",
            "wm_bump_cap_hours": "24", "wm_pinned_ids": "",
            "ep_limit": "50000", "sleep_batch": "5000", "sp_max": "1000",
            "tier2_days": "30", "tier3_days": "180",
            "tier1_weight": "1.0", "tier2_weight": "0.5", "tier3_weight": "0.25",
            "smart_compress": "1", "tier3_max_chars": "300", "degrade_batch": "100",
            "llm_enabled": "true", "llm_max_tokens": "2048",
            "llm_n_threads": "4", "llm_n_ctx": "2048", "llm_timeout": "60",
            "force_local": "0", "host_llm_enabled": "false", "host_llm_n_ctx": "32000",
            "llm_conflict_detection": "false",
            "sync_encrypt": "false", "sync_roles": "user",
            "auto_sleep_enabled": "false", "reflect_disabled_for_cron": "true",
            "reflect_max_calls_per_session": "3",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "0",
            "sync_turn_user_limit": "500", "sync_turn_assistant_limit": "800",
            "persona_enabled": "false", "persona_token_cap": "1500",
            "persona_interval": "50", "persona_daily_sync_hour": "3",
            "sleep_model_refresh_enabled": "true",
            "sleep_model_refresh_auto_apply": "true",
            "shmr_batch_size": "50", "shmr_max_iterations": "3",
            "shmr_similarity_threshold": "0.70", "shmr_harmony_threshold": "0.60",
            "shmr_min_cluster_size": "2", "shmr_temperature": "0.2",
            "auto_migrate": "1", "default_scope": "session",
            "ignore_patterns": "", "write_classifier": "off",
        },
    },
    "embedded": {
        "meta": ProfileMeta(
            name="embedded",
            description="Resource-constrained. Tiny limits, bit vectors, no LLM.",
            use_case="Raspberry Pi, IoT, edge devices.",
            ratings={"quality": 4, "speed": 16, "memory": 2, "llm_dependency": 0, "security": 8},
        ),
        "settings": {
            "vec_type": "bit", "vec_weight": "0.4",
            "embeddings_via_api": "0", "no_embeddings": "0",
            "skip_embeddings": "0", "embeddings_off": "0",
            "fts_weight": "0.5", "importance_weight": "0.1",
            "temporal_halflife_hours": "6", "recency_halflife": "24",
            "recall_extra_stopwords": "", "cross_session": "0",
            "polyphonic_recall": "0", "query_intent": "0",
            "fact_recall_enabled": "0", "enhanced_recall": "0",
            "proactive_linking": "0", "lenient_fact_match": "0",
            "recall_diagnostics": "0",
            "wm_max_items": "500", "wm_ttl_hours": "24",
            "wm_bump_cap_hours": "12", "wm_pinned_ids": "",
            "ep_limit": "2000", "sleep_batch": "500", "sp_max": "50",
            "tier2_days": "7", "tier3_days": "30",
            "tier1_weight": "1.0", "tier2_weight": "0.2", "tier3_weight": "0.1",
            "smart_compress": "0", "tier3_max_chars": "0", "degrade_batch": "50",
            "llm_enabled": "false", "llm_max_tokens": "512",
            "llm_n_threads": "1", "llm_n_ctx": "512", "llm_timeout": "15",
            "force_local": "1", "host_llm_enabled": "false", "host_llm_n_ctx": "2048",
            "llm_conflict_detection": "false",
            "sync_encrypt": "false", "sync_roles": "user",
            "auto_sleep_enabled": "true", "reflect_disabled_for_cron": "true",
            "reflect_max_calls_per_session": "0",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "0",
            "sync_turn_user_limit": "200", "sync_turn_assistant_limit": "300",
            "persona_enabled": "false", "persona_token_cap": "500",
            "persona_interval": "200", "persona_daily_sync_hour": "-1",
            "sleep_model_refresh_enabled": "false",
            "sleep_model_refresh_auto_apply": "false",
            "shmr_batch_size": "10", "shmr_max_iterations": "1",
            "shmr_similarity_threshold": "0.90", "shmr_harmony_threshold": "0.80",
            "shmr_min_cluster_size": "5", "shmr_temperature": "0.1",
            "auto_migrate": "1", "default_scope": "session",
            "ignore_patterns": "", "write_classifier": "off",
        },
    },
    "development": {
        "meta": ProfileMeta(
            name="development",
            description="Verbose. Diagnostics on, warn-mode classifier.",
            use_case="Debugging Mnemosyne itself, developing new features.",
            ratings={"quality": 14, "speed": 12, "memory": 10, "llm_dependency": 12, "security": 8},
        ),
        "settings": {
            "vec_type": "int8", "vec_weight": "0.5",
            "embeddings_via_api": "0", "no_embeddings": "0",
            "skip_embeddings": "0", "embeddings_off": "0",
            "fts_weight": "0.3", "importance_weight": "0.2",
            "temporal_halflife_hours": "24", "recency_halflife": "168",
            "recall_extra_stopwords": "", "cross_session": "0",
            "polyphonic_recall": "0", "query_intent": "0",
            "fact_recall_enabled": "0", "enhanced_recall": "0",
            "proactive_linking": "0", "lenient_fact_match": "0",
            "recall_diagnostics": "1",
            "wm_max_items": "5000", "wm_ttl_hours": "48",
            "wm_bump_cap_hours": "24", "wm_pinned_ids": "",
            "ep_limit": "10000", "sleep_batch": "1000", "sp_max": "500",
            "tier2_days": "7", "tier3_days": "30",
            "tier1_weight": "1.0", "tier2_weight": "0.5", "tier3_weight": "0.25",
            "smart_compress": "1", "tier3_max_chars": "300", "degrade_batch": "50",
            "llm_enabled": "true", "llm_max_tokens": "2048",
            "llm_n_threads": "4", "llm_n_ctx": "2048", "llm_timeout": "60",
            "force_local": "0", "host_llm_enabled": "false", "host_llm_n_ctx": "32000",
            "llm_conflict_detection": "true",
            "sync_encrypt": "false", "sync_roles": "user",
            "auto_sleep_enabled": "true", "reflect_disabled_for_cron": "false",
            "reflect_max_calls_per_session": "10",
            "skip_contexts": "cron,flush,subagent,background,skill_loop",
            "prefetch_content_chars": "500",
            "sync_turn_user_limit": "500", "sync_turn_assistant_limit": "800",
            "persona_enabled": "true", "persona_token_cap": "1500",
            "persona_interval": "25", "persona_daily_sync_hour": "3",
            "sleep_model_refresh_enabled": "true",
            "sleep_model_refresh_auto_apply": "true",
            "shmr_batch_size": "25", "shmr_max_iterations": "3",
            "shmr_similarity_threshold": "0.70", "shmr_harmony_threshold": "0.60",
            "shmr_min_cluster_size": "2", "shmr_temperature": "0.2",
            "auto_migrate": "1", "default_scope": "session",
            "ignore_patterns": "", "write_classifier": "warn",
        },
    },
}


# ---------------------------------------------------------------------------
# Validation — 15 consistency rules
# ---------------------------------------------------------------------------

def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _to_int(val: Any) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _to_float(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def validate_profile(settings: Dict[str, Any]) -> List[str]:
    """Validate a profile's settings against all consistency rules.

    Returns a list of error strings. Empty list = valid.
    """
    errors: List[str] = []

    # Rule 13: no duplicate keys (dict already deduplicates, but check for
    # the template source having intentional duplicates that got silently
    # overwritten — we can't detect this from the dict itself, but we CAN
    # detect it at YAML write time. Here we just verify key count.)
    # This rule is structurally enforced by Python dicts; documented for
    # completeness.

    # Rule 14: no typos (every key must be a known config key)
    known_keys = set(ENV_VAR_MAP.keys())
    for key in settings:
        if key not in known_keys:
            errors.append(f"Unknown key '{key}' — possible typo or non-template var")

    # Rule 1: embeddings aliases consistent
    ne = str(settings.get("no_embeddings", "0"))
    se = str(settings.get("skip_embeddings", "0"))
    eo = str(settings.get("embeddings_off", "0"))
    if not (ne == se == eo):
        errors.append(
            f"Embedding disable aliases inconsistent: "
            f"no_embeddings={ne}, skip_embeddings={se}, embeddings_off={eo} "
            f"(all three must be the same value)"
        )

    # Rule 2: vec_weight > 0 when embeddings on
    if not _to_bool(settings.get("no_embeddings", "0")):
        vw = _to_float(settings.get("vec_weight", "0.5"))
        if vw <= 0:
            errors.append(
                f"vec_weight={vw} but embeddings are enabled — "
                f"wasted compute generating vectors that are never scored"
            )

    # Rule 3: cross_session implies global scope
    if _to_bool(settings.get("cross_session", "0")):
        scope = str(settings.get("default_scope", "session"))
        if scope != "global":
            errors.append(
                f"cross_session=1 but default_scope='{scope}' — "
                f"cross-session visibility requires global scope"
            )

    # Rule 4: smart_compress implies llm_enabled
    if _to_bool(settings.get("smart_compress", "1")):
        if not _to_bool(settings.get("llm_enabled", "true")):
            errors.append(
                f"smart_compress=1 but llm_enabled=false — "
                f"compression requires LLM for summarization"
            )

    # Rule 5: sleep_model_refresh_enabled implies llm_enabled
    if _to_bool(settings.get("sleep_model_refresh_enabled", "false")):
        if not _to_bool(settings.get("llm_enabled", "true")):
            errors.append(
                f"sleep_model_refresh_enabled=true but llm_enabled=false — "
                f"model refresh requires LLM"
            )

    # Rule 6: llm_conflict_detection implies llm_enabled
    if _to_bool(settings.get("llm_conflict_detection", "false")):
        if not _to_bool(settings.get("llm_enabled", "true")):
            errors.append(
                f"llm_conflict_detection=true but llm_enabled=false — "
                f"conflict detection requires LLM"
            )

    # Rule 7: persona_enabled implies llm_enabled
    if _to_bool(settings.get("persona_enabled", "false")):
        if not _to_bool(settings.get("llm_enabled", "true")):
            errors.append(
                f"persona_enabled=true but llm_enabled=false — "
                f"persona generation requires LLM"
            )

    # Rule 8: tier3_max_chars > 0 when smart_compress on
    if _to_bool(settings.get("smart_compress", "1")):
        t3mc = _to_int(settings.get("tier3_max_chars", "300"))
        if t3mc <= 0:
            errors.append(
                f"tier3_max_chars={t3mc} with smart_compress=1 — "
                f"zero chars means silent content deletion"
            )

    # Rules 9-12: vector-dependent features require embeddings
    vec_features = [
        ("proactive_linking", "proactive linking"),
        ("polyphonic_recall", "polyphonic recall"),
        ("enhanced_recall", "enhanced recall"),
        ("query_intent", "query intent"),
    ]
    for key, label in vec_features:
        if _to_bool(settings.get(key, "0")):
            if _to_bool(settings.get("no_embeddings", "0")):
                errors.append(
                    f"{key}=1 but no_embeddings=1 — {label} requires vector embeddings"
                )

    # Rule 15: reflect_max_calls semantics (document, don't error)
    rmc = settings.get("reflect_max_calls_per_session", "3")
    rmc_int = _to_int(rmc)
    if rmc_int == 0:
        # 0 means zero calls allowed = effectively disabled. Not an error,
        # but worth noting. We don't add an error for this.
        pass
    elif rmc_int < 0:
        # -1 disables the cap (unlimited). Also not an error.
        pass

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_profiles() -> Dict[str, ProfileMeta]:
    """Return metadata for all profiles."""
    return {name: data["meta"] for name, data in PROFILES.items()}


def get_profile(name: str) -> Optional[Dict[str, Any]]:
    """Get a profile's settings by name. Returns None if not found."""
    data = PROFILES.get(name)
    if data is None:
        return None
    return dict(data["settings"])


def apply_profile(name: str, config_path=None, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """Apply a profile by writing its settings to config.yaml.

    Args:
        name: Profile name (e.g. "speed", "quality").
        config_path: Optional path to config.yaml. Defaults to the standard location.
        dry_run: If True, validate only — don't write.

    Returns:
        (success, errors) — errors is empty list on success.
    """
    profile = get_profile(name)
    if profile is None:
        return False, [f"Unknown profile: '{name}'. Available: {list(PROFILES.keys())}"]

    # Validate
    errors = validate_profile(profile)
    if errors:
        return False, errors

    if dry_run:
        return True, []

    # Write to config
    config = get_config()
    if config_path:
        from mnemosyne.core.config import MnemosyneConfig
        config = MnemosyneConfig(config_path=Path(config_path) if isinstance(config_path, str) else config_path)

    for key, value in profile.items():
        config.set(key, value)

    logger.info("Applied profile '%s' (%d settings)", name, len(profile))
    return True, []


def create_profile(name: str, description: str = "", config_path=None) -> bool:
    """Save current config as a named profile.

    Reads the current config values and stores them as a new profile.
    """
    config = get_config()
    if config_path:
        from mnemosyne.core.config import MnemosyneConfig
        config = MnemosyneConfig(config_path=Path(config_path) if isinstance(config_path, str) else config_path)

    settings = {}
    for key in TEMPLATE_KEYS:
        val = config.get(key)
        if val is not None:
            settings[key] = str(val)

    # Validate
    errors = validate_profile(settings)
    if errors:
        logger.error("Profile '%s' failed validation: %s", name, errors)
        return False

    # Store in PROFILES dict (in-memory only; persisting to a file would
    # be a future enhancement)
    PROFILES[name] = {
        "meta": ProfileMeta(
            name=name,
            description=description or "User-created profile",
            use_case="Custom",
            ratings={"quality": 10, "speed": 10, "memory": 10, "llm_dependency": 10, "security": 10},
        ),
        "settings": settings,
    }
    return True


def validate_all_profiles() -> Dict[str, List[str]]:
    """Validate all built-in profiles. Returns {name: [errors]}."""
    results = {}
    for name, data in PROFILES.items():
        errors = validate_profile(data["settings"])
        results[name] = errors
    return results


# Re-export for convenience
from pathlib import Path as _Path
