"""
Mnemosyne LLM-Based Conflict Detector (Phase 2)
==============================================
Two-Tier hybrid conflict validator. Gated, isolated, and cost-aware.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from mnemosyne.core.llm_backends import call_host_llm, get_host_llm_backend
from mnemosyne.core.cost_log import log_cost

logger = logging.getLogger(__name__)

# Gating environment variable (default: off)
LLM_CONFLICT_DETECTION_ENABLED = os.environ.get("MNEMOSYNE_LLM_CONFLICT_DETECTION", "false").lower() in ("1", "true", "yes")

# Configuration fallback keys (consistent with local_llm.py remote settings)
LLM_BASE_URL = os.environ.get("MNEMOSYNE_LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("MNEMOSYNE_LLM_API_KEY", "")
LLM_REMOTE_MODEL = os.environ.get("MNEMOSYNE_LLM_MODEL", "google/gemini-flash-1.5")

# Specific overrides if configured
CONFLICT_LLM_BASE_URL = os.environ.get("MNEMOSYNE_CONFLICT_LLM_BASE_URL", LLM_BASE_URL).rstrip("/")
CONFLICT_LLM_API_KEY = os.environ.get("MNEMOSYNE_CONFLICT_LLM_API_KEY", LLM_API_KEY)
CONFLICT_LLM_MODEL = os.environ.get("MNEMOSYNE_CONFLICT_LLM_MODEL", LLM_REMOTE_MODEL)

# Pricing catalog (USD per 1M tokens) for cost logging
# Default pricing fits cheap Flash tier ($0.15 input / $0.60 output)
MODEL_PRICING = {
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "meta-llama/llama-3-8b": {"input": 0.15, "output": 0.60},
    "default": {"input": 0.15, "output": 0.60}
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def _calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate call cost in USD based on model pricing."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    input_cost = (input_tokens / 1_000_000.0) * pricing["input"]
    output_cost = (output_tokens / 1_000_000.0) * pricing["output"]
    return input_cost + output_cost


def _call_conflict_llm(prompt: str) -> Optional[str]:
    """Call either host-provided LLM completion or direct remote endpoint."""
    # 1. Try host LLM backend first if enabled
    if os.environ.get("MNEMOSYNE_HOST_LLM_ENABLED", "false").lower() in ("1", "true", "yes"):
        if get_host_llm_backend() is not None:
            try:
                # Max 512 output tokens for JSON structured result
                return call_host_llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.0,
                    provider=os.environ.get("MNEMOSYNE_HOST_LLM_PROVIDER", "").strip() or None,
                    model=os.environ.get("MNEMOSYNE_HOST_LLM_MODEL", "").strip() or None,
                )
            except Exception as exc:
                logger.warning("Conflict detection: host LLM call failed (%s): %s", type(exc).__name__, exc)

    # 2. Fall back to remote endpoint
    if not CONFLICT_LLM_BASE_URL:
        logger.warning("Conflict detection: no LLM completion endpoint configured.")
        return None

    import json
    try:
        import httpx
        has_httpx = True
    except ImportError:
        has_httpx = False

    url = f"{CONFLICT_LLM_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if CONFLICT_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {CONFLICT_LLM_API_KEY}"

    payload = {
        "model": CONFLICT_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "response_format": {"type": "json_object"} if "gemini" in CONFLICT_LLM_MODEL.lower() or "gpt" in CONFLICT_LLM_MODEL.lower() else None
    }

    try:
        if has_httpx:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode())

        choices = data.get("choices", [])
        if choices and choices[0].get("message", {}).get("content"):
            return choices[0]["message"]["content"]
        return None
    except Exception as exc:
        logger.warning("Conflict detection: remote LLM completion failed (%s): %s", type(exc).__name__, exc)
        return None


def validate_conflict_pair(
    older_content: str,
    newer_content: str,
    session_id: str,
    db_path: Optional[Path] = None
) -> Tuple[bool, float, Optional[str]]:
    """
    Use an LLM to validate if older_content is contradicted/superseded by newer_content.
    
    Returns:
        Tuple[is_conflict (bool), confidence (float), correct_fact (str)]
    """
    prompt = f"""You are an advanced agentic memory consolidation engine. Your task is to analyze two memories and determine if they represent a factual contradiction or a conflict (where the newer memory corrects, updates, or overrides the older one).

Older Memory: "{older_content}"
Newer Memory: "{newer_content}"

Analyze them carefully:
- If they are about different subjects or unrelated, there is NO conflict.
- If they represent chronological updates, corrections of errors, or changed preferences (e.g. "I love apples" corrected by "Actually I prefer oranges now", or "event is May 29" vs "event is June 5"), this IS a conflict where the newer memory overrides the older one.
- If they are near-duplicates or additions that complement each other without factual contradiction, there is NO conflict.

You must respond ONLY with a valid JSON object matching this schema:
{{
  "is_conflict": true or false,
  "confidence": 0.0 to 1.0,
  "correct_fact": "The correct fact summarized",
  "reason": "Brief explanation"
}}
"""
    raw_response = _call_conflict_llm(prompt)
    if not raw_response:
        return False, 0.0, None

    # Strip markdown code blocks if any
    clean_json = raw_response.strip()
    if clean_json.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_json, re.DOTALL)
        if match:
            clean_json = match.group(1)

    try:
        data = json.loads(clean_json)
        is_conflict = bool(data.get("is_conflict", False))
        confidence = float(data.get("confidence", 0.0))
        correct_fact = data.get("correct_fact")

        # Estimate and log costs
        input_t = _estimate_tokens(prompt)
        output_t = _estimate_tokens(raw_response)
        est_cost = _calculate_cost(input_t, output_t, CONFLICT_LLM_MODEL)

        logger.info(
            "Conflict LLM validation completed. Model: %s. Conflict detected: %s. Estimated Cost: $%0.6f",
            CONFLICT_LLM_MODEL, is_conflict, est_cost
        )
        
        # Write to core cost logs
        try:
            log_cost(
                session_id=session_id,
                memory_count=2,
                token_count=input_t + output_t,
                estimated_cost_usd=est_cost,
                model=CONFLICT_LLM_MODEL,
                db_path=db_path
            )
        except Exception as log_exc:
            logger.debug("Failed to record cost to database cost_entries: %s", log_exc)

        return is_conflict, confidence, correct_fact
    except Exception as exc:
        logger.warning("Conflict detection: failed to parse LLM JSON output (%s): %s", type(exc).__name__, exc)
        return False, 0.0, None
