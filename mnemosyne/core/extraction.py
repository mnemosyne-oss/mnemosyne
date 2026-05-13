"""
Mnemosyne Structured Fact Extraction
====================================
LLM-driven fact extraction as a derived layer.
Extracts 2-5 concise factual statements from raw text.
Facts are stored as TripleStore triples, not replacements for raw text.

Uses the same LLM fallback chain as local_llm.py:

0. Host-provided LLM backend (when MNEMOSYNE_HOST_LLM_ENABLED=true and a
   backend is registered). On host attempt with no usable output, skips
   the remote URL and goes straight to local GGUF.
1. Remote OpenAI-compatible API (if MNEMOSYNE_LLM_BASE_URL set
   AND MNEMOSYNE_LLM_ENABLED is not false).
2. Local ctransformers GGUF model.
3. Skip extraction (graceful degradation).

Extraction uses temperature=0.0 (deterministic) so re-ingesting the same
content does not create near-duplicate facts in the facts table.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Reuse local_llm infrastructure
from mnemosyne.core import local_llm
from mnemosyne.core.local_llm import (
    llm_available,
    _call_remote_llm,
    _load_llm,
    _try_host_llm,
    LLM_BASE_URL,
    LLM_ENABLED,
    LLM_MAX_TOKENS,
    _clean_output,
)

# --- Config ------------------------------------------------------------------
EXTRACTION_PROMPT = os.environ.get(
    "MNEMOSYNE_EXTRACTION_PROMPT",
    "Extract 2-5 concise factual statements from the following text. "
    "Each fact should be a complete sentence describing something true about the subject. "
    "Focus on preferences, opinions, experiences, and factual claims. "
    "Return one fact per line. Do not number them. "
    "If no facts can be extracted, return 'NO_FACTS'.\n\nText: {text}\n\nFacts:"
)


def _build_extraction_prompt(text: str) -> str:
    """Build the extraction prompt with the user text inserted."""
    return EXTRACTION_PROMPT.format(text=text)


def _parse_facts(raw_output: str) -> List[str]:
    """Parse LLM output into individual facts."""
    if not raw_output or raw_output.strip().upper() == "NO_FACTS":
        return []
    
    # Split on newlines, filter empty lines
    lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
    
    # Clean up any numbering or bullet prefixes
    cleaned = []
    for line in lines:
        # Remove leading numbers/bullets: "1. fact" or "- fact" or "* fact"
        line = line.lstrip("0123456789.-* ").strip()
        if line and len(line) > 10:  # Minimum fact length
            cleaned.append(line)
    
    return cleaned[:5]  # Cap at 5 facts


def _call_local_extraction_llm(llm, prompt: str) -> str:
    """Run deterministic local extraction for the loaded local LLM backend.

    llama-cpp-python exposes ``max_tokens`` via its completion/chat APIs,
    while ctransformers exposes ``max_new_tokens`` on the direct callable.
    Using ctransformers kwargs against a llama.cpp ``Llama`` instance raises
    ``unexpected keyword argument 'max_new_tokens'`` and disables fact
    extraction on installs where llama-cpp-python is preferred.
    """
    if getattr(local_llm, "_llm_backend", None) == "llamacpp":
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=local_llm.LLM_MAX_TOKENS,
            stop=["</s>", "<|user|>"],
            temperature=0.0,
        )
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
        return ""
    return llm(
        prompt,
        max_new_tokens=local_llm.LLM_MAX_TOKENS,
        stop=["</s>", "<|user|>"],
    )


def extract_facts(text: str) -> List[str]:
    """
    Extract structured facts from raw text using LLM.

    Args:
        text: Raw memory content to extract facts from

    Returns:
        List of extracted fact strings (0-5 items). Empty list if LLM unavailable.

    Notes:
        - The host backend (Hermes auxiliary client) is consulted first when
          enabled. Temperature is fixed at 0.0 so re-ingesting the same content
          produces deterministic facts (avoids near-duplicate writes to the
          facts table).
        - When the host attempt produces no usable text, the remote URL is
          **skipped** — falls through to local GGUF, then []. This honors the
          plan's host-vs-remote precedence rule.
        - [C13.b] All tier transitions and failures are recorded to the
          process-global `ExtractionDiagnostics`. Operators query via
          `mnemosyne.extraction.get_extraction_stats()` to see why
          extraction is producing empty results.
    """
    # Lazy import to avoid a circular dependency: mnemosyne.extraction
    # re-exports diagnostics, and tests/core import extraction.py very
    # early; importing diagnostics at module load would tangle the
    # init order. After first call sys.modules caches the import.
    from mnemosyne.extraction.diagnostics import get_diagnostics, _safe_for_log as diagnostics_safe_for_log
    diag = get_diagnostics()

    if not text or not text.strip():
        # Caller passed nothing — this isn't a failure, just no work.
        # Don't record_call: this isn't really an extraction attempt.
        return []

    if not local_llm.llm_available():
        diag.record_failure(
            "local", reason="llm_unavailable_at_call_site",
        )
        diag.record_call(succeeded=False, all_empty=False)
        return []

    prompt = _build_extraction_prompt(text)

    # 0. Host backend (deterministic; temperature=0.0).
    # Reference live module values so monkeypatch on local_llm reaches us.
    #
    # /review fix: record host attempt ONLY when the host backend
    # actually ran (`attempted=True`). Pre-fix every call incremented
    # the host counter, including configurations with no host backend
    # registered — phantom attempts polluted the metric. Plus wrap
    # the call so an exception inside _try_host_llm gets attributed
    # to host instead of escaping to the outer wrapper.
    try:
        attempted, host_text = local_llm._try_host_llm(
            prompt, max_tokens=local_llm.LLM_MAX_TOKENS, temperature=0.0
        )
    except Exception as e:
        # Host adapter itself raised — count as host failure rather
        # than letting it escape to the outer wrapper where it'd be
        # misattributed to a generic tier.
        diag.record_attempt("host")
        diag.record_failure("host", exc=e, reason="host_adapter_raised")
        diag.record_call(succeeded=False)
        logger.warning(
            "extract_facts: host LLM adapter raised: %s",
            diagnostics_safe_for_log(e),
        )
        return []

    if attempted:
        diag.record_attempt("host")
        if host_text:
            facts = _parse_facts(host_text)
            if facts:
                diag.record_success("host", fact_count=len(facts))
                diag.record_call(succeeded=True)
                return facts
            diag.record_no_output("host")
        else:
            diag.record_no_output("host")
        # Host attempted but produced no facts. Skip remote per A3; try local.
        diag.record_attempt("local")
        try:
            llm = local_llm._load_llm()
        except Exception as e:
            diag.record_failure("local", exc=e, reason="load_llm_raised")
            logger.warning(
                "extract_facts: _load_llm raised: %s",
                diagnostics_safe_for_log(e),
            )
            diag.record_call(succeeded=False)
            return []
        if llm is not None:
            try:
                raw_output = _call_local_extraction_llm(llm, prompt)
                facts = _parse_facts(local_llm._clean_output(raw_output))
                if facts:
                    diag.record_success("local", fact_count=len(facts))
                    diag.record_call(succeeded=True)
                else:
                    diag.record_no_output("local")
                    diag.record_call(succeeded=False, all_empty=True)
                return facts
            except Exception as e:
                diag.record_failure("local", exc=e, reason="ctransformers_raised")
                logger.warning(
                    "extract_facts: local LLM raised on host-fallback path: %s",
                    diagnostics_safe_for_log(e),
                )
                diag.record_call(succeeded=False)
                return []
        diag.record_failure("local", reason="model_not_loaded")
        diag.record_call(succeeded=False, all_empty=True)
        return []

    # 1. Remote LLM. Pass temperature=0.0 so the C2 determinism contract
    # holds even on the standalone remote path (where extract_facts shares
    # _call_remote_llm with summarize_memories' default of 0.3).
    if local_llm.LLM_ENABLED and local_llm.LLM_BASE_URL:
        diag.record_attempt("remote")
        try:
            raw_output = local_llm._call_remote_llm(prompt, temperature=0.0)
        except Exception as e:
            diag.record_failure("remote", exc=e, reason="remote_call_raised")
            logger.warning(
                "extract_facts: remote LLM raised: %s",
                diagnostics_safe_for_log(e),
            )
            raw_output = ""
        if raw_output:
            facts = _parse_facts(local_llm._clean_output(raw_output))
            if facts:
                diag.record_success("remote", fact_count=len(facts))
                diag.record_call(succeeded=True)
                return facts
            diag.record_no_output("remote")
        else:
            diag.record_no_output("remote")

    # 2. Local LLM.
    diag.record_attempt("local")
    try:
        llm = local_llm._load_llm()
    except Exception as e:
        diag.record_failure("local", exc=e, reason="load_llm_raised")
        logger.warning(
            "extract_facts: _load_llm raised: %s",
            diagnostics_safe_for_log(e),
        )
        diag.record_call(succeeded=False)
        return []
    if llm is not None:
        try:
            raw_output = _call_local_extraction_llm(llm, prompt)
            facts = _parse_facts(local_llm._clean_output(raw_output))
            if facts:
                diag.record_success("local", fact_count=len(facts))
                diag.record_call(succeeded=True)
            else:
                diag.record_no_output("local")
                diag.record_call(succeeded=False, all_empty=True)
            return facts
        except Exception as e:
            diag.record_failure("local", exc=e, reason="ctransformers_raised")
            logger.warning(
                "extract_facts: local LLM raised: %s",
                diagnostics_safe_for_log(e),
            )
            diag.record_call(succeeded=False)
            return []

    diag.record_failure("local", reason="model_not_loaded")
    diag.record_call(succeeded=False, all_empty=True)
    return []


def extract_facts_safe(text: str) -> List[str]:
    """
    Best-effort fact extraction that never raises.
    Wrapper for extract_facts with exception handling.

    [C13.b] Outer-wrapper failures (anything `extract_facts` lets
    escape) are recorded under the synthetic `wrapper` tier with
    reason `outer_wrapper_caught`. /review caught the prior pattern
    of misattributing these to `local` — that inflated the local
    tier's failure count and misled operators triaging local-LLM
    health. The `wrapper` tier is explicitly for "tier of origin
    can't be determined" failures.
    """
    try:
        return extract_facts(text)
    except Exception as e:
        from mnemosyne.extraction.diagnostics import get_diagnostics, _safe_for_log
        diag = get_diagnostics()
        diag.record_failure(
            "wrapper", exc=e, reason="outer_wrapper_caught"
        )
        diag.record_call(succeeded=False)
        logger.warning(
            "extract_facts_safe: extract_facts() raised: %s",
            _safe_for_log(e),
        )
        return []
