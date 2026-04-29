"""
Mnemosyne Structured Fact Extraction
====================================
LLM-driven fact extraction as a derived layer.
Extracts 2-5 concise factual statements from raw text.
Facts are stored as TripleStore triples, not replacements for raw text.

Uses the same LLM fallback chain as local_llm.py:
1. Remote OpenAI-compatible API (if MNEMOSYNE_LLM_BASE_URL set)
2. Local ctransformers GGUF model
3. Skip extraction (graceful degradation)
"""

import os
from typing import List, Optional

# Reuse local_llm infrastructure
from mnemosyne.core.local_llm import (
    llm_available,
    _call_remote_llm,
    _load_llm,
    LLM_BASE_URL,
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


def extract_facts(text: str) -> List[str]:
    """
    Extract structured facts from raw text using LLM.
    
    Args:
        text: Raw memory content to extract facts from
        
    Returns:
        List of extracted fact strings (0-5 items). Empty list if LLM unavailable.
    """
    if not text or not text.strip():
        return []
    
    if not llm_available():
        return []
    
    prompt = _build_extraction_prompt(text)
    raw_output = None
    
    # --- Try remote LLM first ---
    if LLM_BASE_URL:
        raw_output = _call_remote_llm(prompt)
        if raw_output:
            facts = _parse_facts(_clean_output(raw_output))
            if facts:
                return facts
    
    # --- Fall back to local LLM ---
    llm = _load_llm()
    if llm is not None:
        try:
            raw_output = llm(prompt, max_new_tokens=LLM_MAX_TOKENS, stop=["</s>", "<|user|>"])
            facts = _parse_facts(_clean_output(raw_output))
            return facts
        except Exception:
            pass
    
    return []


def extract_facts_safe(text: str) -> List[str]:
    """
    Best-effort fact extraction that never raises.
    Wrapper for extract_facts with exception handling.
    """
    try:
        return extract_facts(text)
    except Exception:
        return []
