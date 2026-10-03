"""
Mnemosyne LLM-Based Conflict Detector (Phase 2)
==============================================
Two-Tier hybrid conflict validator. Gated, isolated, and cost-aware.
"""

import os
import re
import json
import logging
import math
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler
from typing import Optional, Tuple
from pathlib import Path

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


def conflict_endpoint_is_safe() -> bool:
    """Require a real HTTP(S) endpoint, and HTTPS when sending credentials."""
    try:
        endpoint = urlsplit(CONFLICT_LLM_BASE_URL or "")
        return bool(
            endpoint.hostname
            and endpoint.scheme in ("http", "https")
            and not endpoint.username and not endpoint.password
            and (not CONFLICT_LLM_API_KEY or endpoint.scheme == "https")
        )
    except (TypeError, ValueError):
        return False


class _NoConflictRedirects(HTTPRedirectHandler):
    """Never replay memory content or Authorization to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _call_conflict_llm_with_retry(
    prompt: str, *, deadline: Optional[float] = None,
) -> Optional[Tuple[str, Optional[int], Optional[int]]]:
    """
    Call OpenAI-compatible completions endpoint with exponential backoff retry logic.
    Returns: Tuple[content_str, prompt_tokens, completion_tokens] or None
    """
    if not conflict_endpoint_is_safe():
        logger.warning("Conflict detection: LLM endpoint missing or unsafe.")
        return None

    import time
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

    max_retries = 2
    backoff_factor = 2.0
    initial_delay = 1.0

    if deadline is not None and not math.isfinite(deadline):
        return None
    for attempt in range(max_retries + 1):
        remaining = 15.0 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            return None
        timeout = min(15.0, remaining)
        try:
            if has_httpx:
                with httpx.Client(timeout=timeout, follow_redirects=False) as client:
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
                opener = urllib.request.build_opener(_NoConflictRedirects())
                with opener.open(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())

            choices = data.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                content = choices[0]["message"]["content"]
                
                # Extract actual token counts if provided by the API
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                
                if not isinstance(content, str):
                    raise ValueError("Invalid completion content type")
                return content, prompt_tokens, completion_tokens
            
            logger.warning("Conflict detection: empty response from LLM on attempt %d", attempt + 1)
        except Exception as exc:
            logger.warning(
                "Conflict detection: LLM call failed on attempt %d (%s)",
                attempt + 1, type(exc).__name__
            )
            if attempt < max_retries:
                sleep_time = initial_delay * (backoff_factor ** attempt)
                logger.info("Conflict detection: retrying in %0.1fs...", sleep_time)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= sleep_time:
                        return None
                time.sleep(sleep_time)
            else:
                logger.error("Conflict detection: all retries exhausted. Call failed.")
    
    return None


def validate_conflict_pair(
    older_content: str,
    newer_content: str,
    session_id: str,
    db_path: Optional[Path] = None,
    *, deadline: Optional[float] = None,
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
    result = _call_conflict_llm_with_retry(prompt, deadline=deadline)
    if not result:
        return False, 0.0, None

    raw_response, actual_prompt_tokens, actual_completion_tokens = result
    if not isinstance(raw_response, str):
        return False, 0.0, None

    # Strip markdown code blocks if any
    clean_json = raw_response.strip()
    if clean_json.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_json, re.DOTALL)
        if match:
            clean_json = match.group(1)

    try:
        data = json.loads(clean_json)
        if not isinstance(data, dict) or type(data.get("is_conflict")) is not bool:
            raise ValueError("Invalid conflict verdict type")
        is_conflict = data["is_conflict"]
        confidence = data.get("confidence")
        correct_fact = data.get("correct_fact")
        if (type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or (correct_fact is not None and not isinstance(correct_fact, str))):
            raise ValueError("Invalid conflict response schema")
        confidence = float(confidence)

        # Estimate fallback tokens if actual ones are not provided by the API
        input_t = actual_prompt_tokens if actual_prompt_tokens is not None else _estimate_tokens(prompt)
        output_t = actual_completion_tokens if actual_completion_tokens is not None else _estimate_tokens(raw_response)
        
        est_cost = _calculate_cost(input_t, output_t, CONFLICT_LLM_MODEL)

        logger.info(
            "Conflict LLM validation completed. Model: %s. Conflict detected: %s. Actual usage: Input=%s, Output=%s. Cost: $%0.6f",
            CONFLICT_LLM_MODEL, is_conflict, 
            actual_prompt_tokens if actual_prompt_tokens is not None else "Estimated",
            actual_completion_tokens if actual_completion_tokens is not None else "Estimated",
            est_cost
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
            logger.debug("Failed to record conflict cost (%s)", type(log_exc).__name__)

        return is_conflict, confidence, correct_fact
    except Exception as exc:
        logger.warning("Conflict detection: failed to parse LLM JSON output (%s)", type(exc).__name__)
        return False, 0.0, None

