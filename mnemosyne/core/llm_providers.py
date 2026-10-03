"""
Mnemosyne remote LLM provider presets
======================================
The OpenAI-compatible remote LLM client (:mod:`mnemosyne.core.local_llm`)
accepts a base URL and model name through raw environment variables
(``MNEMOSYNE_LLM_BASE_URL`` / ``MNEMOSYNE_LLM_MODEL``). That works for any
endpoint but requires operators to memorize per-region URLs and model ids.

This module adds named provider presets so an operator can set
``MNEMOSYNE_LLM_PROVIDER`` (and optionally ``MNEMOSYNE_LLM_REGION``) and have
the client resolve the correct OpenAI-compatible base URL and a sensible
default model. Explicit ``MNEMOSYNE_LLM_BASE_URL`` / ``MNEMOSYNE_LLM_MODEL``
values always win, so existing configurations are unchanged.

Each preset records, per region, both the OpenAI-compatible base URL and the
Anthropic-compatible base URL, plus per-model metadata (context window,
per-million-token pricing, input modalities, and supported thinking modes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------
#
# Structure per provider:
#   default_region: region key used when MNEMOSYNE_LLM_REGION is unset
#   default_model:  model id used when MNEMOSYNE_LLM_MODEL is unset
#   regions:        region key -> {openai_base_url, anthropic_base_url, docs_root}
#   models:         model id  -> {context_window, pricing_usd_per_million_tokens,
#                                 input_modalities, thinking}
#
# The remote client is OpenAI-compatible, so it consumes ``openai_base_url``;
# ``anthropic_base_url`` is recorded for completeness and future use.

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "minimax": {
        "name": "MiniMax",
        "default_region": "global_en",
        "default_model": "MiniMax-M3",
        "regions": {
            "global_en": {
                "openai_base_url": "https://api.minimax.io/v1",
                "anthropic_base_url": "https://api.minimax.io/anthropic",
                "docs_root": "https://platform.minimax.io/docs",
            },
            "cn_zh": {
                "openai_base_url": "https://api.minimaxi.com/v1",
                "anthropic_base_url": "https://api.minimaxi.com/anthropic",
                "docs_root": "https://platform.minimaxi.com/docs",
            },
        },
        "models": {
            "MiniMax-M3": {
                "context_window": 1_000_000,
                "pricing_usd_per_million_tokens": {
                    "input": 0.6,
                    "output": 2.4,
                    "cache_read": 0.12,
                    "cache_write": None,
                },
                "input_modalities": ["text", "image", "video"],
                "thinking": ["adaptive", "disabled"],
            },
            "MiniMax-M2.7": {
                "context_window": 204_800,
                "pricing_usd_per_million_tokens": {
                    "input": 0.3,
                    "output": 1.2,
                    "cache_read": 0.06,
                    "cache_write": 0.375,
                },
                "input_modalities": ["text"],
                "thinking": ["always_on"],
            },
        },
    },
    # OrcaRouter is an OpenAI-compatible gateway exposing 150+ models from
    # OpenAI, Anthropic, Google, DeepSeek, Qwen and others behind a single
    # endpoint and API key. Model ids are namespaced (openai/...,
    # anthropic/..., orcarouter/auto). The virtual router ``orcarouter/auto``
    # picks the best upstream per request, so context window and pricing vary
    # by upstream and are left unknown rather than stated as fixed figures.
    "orcarouter": {
        "name": "OrcaRouter",
        "default_region": "global",
        "default_model": "orcarouter/auto",
        "regions": {
            "global": {
                "openai_base_url": "https://api.orcarouter.ai/v1",
                "anthropic_base_url": "https://api.orcarouter.ai",
                "docs_root": "https://www.orcarouter.ai",
            },
        },
        "models": {
            "orcarouter/auto": {
                "context_window": None,
                "pricing_usd_per_million_tokens": {
                    "input": None,
                    "output": None,
                    "cache_read": None,
                    "cache_write": None,
                },
                "input_modalities": ["text"],
                "thinking": ["disabled"],
            },
            "openai/gpt-4o-mini": {
                "context_window": 128_000,
                "pricing_usd_per_million_tokens": {
                    "input": None,
                    "output": None,
                    "cache_read": None,
                    "cache_write": None,
                },
                "input_modalities": ["text"],
                "thinking": ["disabled"],
            },
        },
    },
}


def _normalize(name: str) -> str:
    """Lower-case and trim a provider or region name for lookup."""
    return (name or "").strip().lower()


def get_provider_preset(name: str) -> Optional[Dict[str, Any]]:
    """Return the preset for ``name`` (case-insensitive), or None if unknown."""
    return PROVIDER_PRESETS.get(_normalize(name))


def list_providers() -> List[str]:
    """Return the list of known provider preset keys."""
    return sorted(PROVIDER_PRESETS.keys())


def resolve_region(name: str, region: str = "") -> Optional[str]:
    """Return the region key to use for ``name``.

    Falls back to the preset's ``default_region`` when ``region`` is empty or
    unknown. Returns None if the provider itself is unknown.
    """
    preset = get_provider_preset(name)
    if preset is None:
        return None
    regions = preset.get("regions", {})
    key = _normalize(region)
    if key and key in regions:
        return key
    return preset.get("default_region")


def get_base_url(name: str, region: str = "", api: str = "openai") -> Optional[str]:
    """Return the base URL for ``name`` in ``region``.

    ``api`` selects ``"openai"`` (default, OpenAI-compatible) or
    ``"anthropic"``. Returns None when the provider, region, or API family is
    unknown. Any trailing slash is stripped.
    """
    preset = get_provider_preset(name)
    if preset is None:
        return None
    region_key = resolve_region(name, region)
    region_cfg = preset.get("regions", {}).get(region_key or "", {})
    field = "anthropic_base_url" if _normalize(api) == "anthropic" else "openai_base_url"
    url = region_cfg.get(field)
    return url.rstrip("/") if isinstance(url, str) else None


def default_model(name: str) -> Optional[str]:
    """Return the preset's default model id, or None if the provider is unknown."""
    preset = get_provider_preset(name)
    if preset is None:
        return None
    return preset.get("default_model")


def list_models(name: str) -> List[str]:
    """Return the model ids configured for ``name`` (empty list if unknown)."""
    preset = get_provider_preset(name)
    if preset is None:
        return []
    return list(preset.get("models", {}).keys())


def get_model_config(name: str, model_id: str) -> Optional[Dict[str, Any]]:
    """Return the per-model metadata dict for ``model_id``, or None if absent."""
    preset = get_provider_preset(name)
    if preset is None:
        return None
    return preset.get("models", {}).get(model_id)


def resolve_provider_defaults(
    name: str, region: str = ""
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ``(openai_base_url, default_model)`` for a provider preset.

    Used by the remote LLM client to fill in a base URL and model only when
    the operator has not set them explicitly. Returns ``(None, None)`` for an
    unknown provider so callers can leave their existing values untouched.
    """
    preset = get_provider_preset(name)
    if preset is None:
        return (None, None)
    return (get_base_url(name, region, api="openai"), default_model(name))
