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
import re
from typing import List, Tuple

import collections

logger = logging.getLogger(__name__)

# Reuse local_llm infrastructure
from mnemosyne.core import local_llm

# --- Config ------------------------------------------------------------------
EXTRACTION_PROMPT_TEMPLATE = os.environ.get(
    "MNEMOSYNE_EXTRACTION_PROMPT",
    "You are an expert structured memory extractor for Mnemosyne v3.0+ MEMORIA tables.\n"
    "The user message below may be in English, German, Russian, or another language.\n"
    "First detect the language, then extract ONLY high-signal, long-term relevant items.\n"
    "Categories to extract (return valid JSON only, no extra text):\n"
    "- facts: persistent user metrics, states, knowledge, or personal data\n"
    "  (Examples: 'my name is X', 'I work at Y', 'server runs on port 8080')\n"
    "  (Russian: 'меня зовут', 'работаю в', 'использую', 'мой пароль', 'живу в')\n"
    "- instructions: rules or commands directed at me the agent\n"
    "  (Examples: 'always use tabs', 'never delete logs', 'call me boss')\n"
    "  (Russian: 'всегда', 'никогда', 'не забудь', 'нужно', 'должен', 'обязательно')\n"
    "- preferences: likes, dislikes, and their evolution\n"
    "  (Examples: 'I like dark mode', 'I prefer Python over Go')\n"
    "  (Russian: 'нравится', 'люблю', 'терпеть не могу', 'предпочитаю', 'я за')\n"
    "- timelines: real events with dates/times\n"
    "  (Examples: 'release on 2024-12-01', 'meeting next Tuesday')\n"
    "  (Russian: 'встреча', 'релиз', 'запланировано на', 'дедлайн', крайний срок)\n"
    "- kg: knowledge-graph triples in subject-predicate-object form\n"
    "  (Examples: 'user prefers vim', 'project uses fastapi')\n\n"
    "Rules:\n"
    "- Only extract persistent, non-transient content. Ignore weather, one-off chat, system text.\n"
    "- Use semantic understanding — do NOT rely on English keywords.\n"
    "- Preserve original casing and language.\n"
    "- If nothing qualifies, return empty arrays.\n\n"
    "Return JSON in this exact format:\n"
    '{"facts": [], "instructions": [], "preferences": [], "timelines": [], "kg": []}\n\n'
    "User message: {text}\n\n"
    "Extraction:"
)


def _build_extraction_prompt(text: str, detected_lang: str = 'en') -> str:
    """Build the extraction prompt with the user text and language context."""
    prompt = EXTRACTION_PROMPT_TEMPLATE.replace("{text}", text).replace("{lang}", detected_lang)
    return prompt


def _parse_facts(raw_output: str) -> List[str]:
    """Parse LLM output into individual facts.
    Handles both JSON format (new MEMORIA prompt) and line-by-line format (legacy).
    JSON format: {"facts": [...], "instructions": [...], ...}
    Legacy format: one fact per line, optionally numbered."""
    if not raw_output or raw_output.strip().upper() == "NO_FACTS":
        return []

    # Try JSON parsing first (new MEMORIA prompt format)
    import json as _json
    raw_clean = raw_output.strip()
    # Find JSON block if wrapped in backticks
    if raw_clean.startswith("```"):
        raw_clean = raw_clean.split("```\n")[-1] if "```\n" in raw_clean else raw_clean.removeprefix("```json").removesuffix("```").strip()
    if raw_clean.startswith("{"):
        try:
            parsed = _json.loads(raw_clean)
            if isinstance(parsed, dict):
                # Collect all extracted items across categories.
                # The categories overlap by design — "always use tabs" is both a
                # fact and an instruction, and a dated statement is both a fact
                # and a timeline — so a model that files one statement under
                # every category it belongs to is behaving correctly. Without a
                # dedupe here that correctness is punished: the same string is
                # returned once per category and stored once per copy.
                all_items = []
                seen = set()
                saw_supported_category = False
                for category in ('facts', 'instructions', 'preferences', 'timelines'):
                    if category not in parsed:
                        continue
                    # Presence decides, not usability. A document that parsed as
                    # a dict and names a supported category is complete, not
                    # partial, so `"facts": null` or a bare string means "no
                    # facts" rather than "try the streaming recovery regex".
                    saw_supported_category = True
                    items = parsed.get(category)
                    if isinstance(items, list):
                        for item in items:
                            if not item:
                                continue
                            text = str(item)
                            key = text.strip().casefold()
                            if not key or key in seen:
                                continue
                            seen.add(key)
                            all_items.append(text)
                # Answered by presence, not by yield: a payload that supplies a
                # supported category and no usable entries means "nothing to
                # extract", so return that. Falling through would hand the raw
                # text to the partial-JSON regex, which matches any quoted run
                # of ten or more characters — including the schema's own key
                # names, making "instructions" and "preferences" stored facts.
                # No supported category still falls through, which keeps the
                # malformed/unsupported and kg-only paths intact.
                if saw_supported_category:
                    return all_items
        except (_json.JSONDecodeError, Exception):
            pass
        # Partial JSON — try to extract from streaming output
        try:
            import re as _re
            # Match incomplete JSON and extract any complete strings in arrays
            matches = _re.findall(r'"([^"]{10,})"', raw_output)
            if matches:
                return matches[:5]
        except Exception:
            pass

    # Legacy: split on newlines, filter empty lines
    lines = [line.strip() for line in raw_output.split("\n") if line.strip()]

    # Clean up any numbering or bullet prefixes
    cleaned = []
    for line in lines:
        # Remove leading numbers/bullets: "1. fact" or "- fact" or "* fact"
        line = line.lstrip("0123456789.-* ").strip()
        if line and len(line) > 10:  # Minimum fact length
            cleaned.append(line)

    return cleaned[:5]  # Cap at 5 facts
    
    return cleaned[:5]  # Cap at 5 facts


# --- KG triple parsing / validation ------------------------------------------
#
# The extraction prompt asks the LLM for a "kg" category of subject-predicate-
# object triples, but `_parse_facts` iterates only
# ('facts', 'instructions', 'preferences', 'timelines') and silently drops
# every triple the model returns. `_parse_kg_triples` recovers that category
# from the same JSON payload; `validate_kg_triples` gates what may reach the
# TripleStore.


def _parse_kg_triples(raw_output: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples from LLM output.

    Only the well-formed JSON branch is honoured: kg items must come from the
    parsed JSON payload, each item either a "subject predicate object" string,
    a [s, p, o] 3-list, or a dict with subject/predicate/object keys. Anything
    else degrades to no triples — we never guess an SPO split out of prose,
    because a fabricated split is worse than none.
    """
    if not raw_output or raw_output.strip().upper() == "NO_FACTS":
        return []

    import json as _json
    raw_clean = raw_output.strip()
    if raw_clean.startswith("```"):
        raw_clean = (
            raw_clean.split("```\n")[-1]
            if "```\n" in raw_clean
            else raw_clean.removeprefix("```json").removesuffix("```").strip()
        )
    if not raw_clean.startswith("{"):
        return []
    try:
        parsed = _json.loads(raw_clean)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    items = parsed.get("kg", [])
    if not isinstance(items, list):
        return []

    triples: List[Tuple[str, str, str]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            if any(f is None for f in item):
                continue
            triples.append((str(item[0]), str(item[1]), str(item[2])))
        elif isinstance(item, dict):
            s, p, o = item.get("subject"), item.get("predicate"), item.get("object")
            if s and p and o:
                triples.append((str(s), str(p), str(o)))
        elif isinstance(item, str) and item.strip():
            parts = item.strip().split(None, 2)
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


# Validation caps for LLM-proposed KG triples before they reach TripleStore.
KG_MAX_SUBJECT_CHARS = 120
KG_MAX_PREDICATE_CHARS = 60
KG_MAX_OBJECT_CHARS = 300

# Hedge / conversational-filler openings that mark a proposed triple as
# chat debris rather than durable knowledge (same hit class the regex
# prototype stored as junk decisions). Trailing space-or-end match keeps
# "Wellington" etc. Includes the #848 review examples ("you said", "let me").
_KG_FILLER_PREFIXES = (
    "what ", "whats ", "what's ", "whether ", "maybe ", "perhaps ",
    "probably ", "possibly ", "i think ", "i guess ", "i mean ",
    "it seems ", "not sure ", "kinda ", "sorta ", "well ",
    "you said ", "let me ", "let me check ",
)


def _kg_normalize_field(value) -> str:
    """Coerce to str, collapse whitespace runs, strip wrapping quotes."""
    if value is None:
        return ""
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    return collapsed.strip("\"'`").strip()


def _kg_word_safe_truncate(text: str, limit: int) -> str:
    """Cut to <= limit chars without ending mid-word when avoidable.

    Returns "" when no clean cut exists (single token longer than limit);
    callers treat "" as a rejection.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    if " " in head:
        return head.rsplit(" ", 1)[0].rstrip(",;:")
    return ""


def validate_kg_triples(triples) -> List[Tuple[str, str, str]]:
    """Filter/normalize LLM-proposed (subject, predicate, object) triples.

    Gates, in order:
    - every field must be a non-empty string after whitespace normalization;
    - conversational-filler openings are rejected on subject and object;
    - length caps: subject <= 120, predicate <= 60, object <= 300 chars.
      Over-long objects are cut at a word boundary (never mid-word); a field
      with no clean cut is rejected outright;
    - predicates are lowercased with whitespace/hyphens folded to underscores
      (collapse runs, trim boundary underscores), enforcing snake_case;
    - within-batch duplicates (case-insensitive) are dropped.
    """
    validated: List[Tuple[str, str, str]] = []
    seen = set()
    for triple in triples or []:
        try:
            raw_subject, raw_predicate, raw_object = triple
        except (TypeError, ValueError):
            continue
        subject = _kg_normalize_field(raw_subject)
        predicate = _kg_normalize_field(raw_predicate)
        obj = _kg_normalize_field(raw_object)
        if not subject or not predicate or not obj:
            continue

        low_subject, low_object = subject.lower(), obj.lower()

        def _is_filler(text: str) -> bool:
            for pfx in _KG_FILLER_PREFIXES:
                stem = pfx.rstrip()
                if text == stem or text.startswith(stem + " "):
                    return True
            return False

        if _is_filler(low_subject) or _is_filler(low_object):
            continue

        subject = _kg_word_safe_truncate(subject, KG_MAX_SUBJECT_CHARS)
        predicate = _kg_word_safe_truncate(predicate, KG_MAX_PREDICATE_CHARS)
        obj = _kg_word_safe_truncate(obj, KG_MAX_OBJECT_CHARS)
        if not subject or not predicate or not obj:
            continue

        predicate = re.sub(r"[\s\-]+", "_", predicate.lower())
        predicate = re.sub(r"_+", "_", predicate).strip("_")
        if not predicate:
            continue

        key = (subject.lower(), predicate, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        validated.append((subject, predicate, obj))
    return validated


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


def _extract_facts_impl(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Core LLM tier chain shared by :func:`extract_facts` and
    :func:`extract_facts_with_triples`.

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
        return [], []

    if not local_llm.llm_available():
        diag.record_failure(
            "local", reason="llm_unavailable_at_call_site",
        )
        diag.record_call(succeeded=False, all_empty=False)
        return [], []

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
        return [], []

    if attempted:
        diag.record_attempt("host")
        if local_llm._is_invalid_reasoning_output(host_text):
            diag.record_failure("host", reason="malformed_reasoning_trace")
            diag.record_call(succeeded=False)
            return [], []
        if host_text:
            facts = _parse_facts(host_text)
            kg_triples = validate_kg_triples(_parse_kg_triples(host_text))
            if facts or kg_triples:
                diag.record_success("host", fact_count=len(facts))
                diag.record_call(succeeded=True)
                return facts, kg_triples
            diag.record_no_output("host")
        else:
            diag.record_no_output("host")
        # Host attempted but produced no usable output. Skip remote per A3;
        # try local.
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
            return [], []
        if llm is not None:
            try:
                raw_output = _call_local_extraction_llm(llm, prompt)
                cleaned_output = local_llm._clean_output(raw_output)
                if local_llm._is_invalid_reasoning_output(cleaned_output):
                    diag.record_failure(
                        "local", reason="malformed_reasoning_trace")
                    diag.record_call(succeeded=False)
                    return [], []
                facts = _parse_facts(cleaned_output)
                kg_triples = validate_kg_triples(
                    _parse_kg_triples(cleaned_output))
                if facts or kg_triples:
                    diag.record_success("local", fact_count=len(facts))
                    diag.record_call(succeeded=True)
                else:
                    diag.record_no_output("local")
                    diag.record_call(succeeded=False, all_empty=True)
                return facts, kg_triples
            except Exception as e:
                diag.record_failure("local", exc=e, reason="ctransformers_raised")
                logger.warning(
                    "extract_facts: local LLM raised on host-fallback path: %s",
                    diagnostics_safe_for_log(e),
                )
                diag.record_call(succeeded=False)
                return [], []
        diag.record_failure("local", reason="model_not_loaded")
        diag.record_call(succeeded=False, all_empty=True)
        return [], []

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
            remote_clean = local_llm._clean_output(raw_output)
            if local_llm._is_invalid_reasoning_output(remote_clean):
                diag.record_failure(
                    "remote", reason="malformed_reasoning_trace")
                diag.record_call(succeeded=False)
                return [], []
            facts = _parse_facts(remote_clean)
            kg_triples = validate_kg_triples(_parse_kg_triples(remote_clean))
            if facts or kg_triples:
                diag.record_success("remote", fact_count=len(facts))
                diag.record_call(succeeded=True)
                return facts, kg_triples
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
        return [], []
    if llm is not None:
        try:
            raw_output = _call_local_extraction_llm(llm, prompt)
            local_clean = local_llm._clean_output(raw_output)
            if local_llm._is_invalid_reasoning_output(local_clean):
                diag.record_failure("local", reason="malformed_reasoning_trace")
                diag.record_call(succeeded=False)
                return [], []
            facts = _parse_facts(local_clean)
            kg_triples = validate_kg_triples(_parse_kg_triples(local_clean))
            if facts or kg_triples:
                diag.record_success("local", fact_count=len(facts))
                diag.record_call(succeeded=True)
                return facts, kg_triples
            else:
                diag.record_no_output("local")
                diag.record_call(succeeded=False, all_empty=True)
            return facts, kg_triples
        except Exception as e:
            diag.record_failure("local", exc=e, reason="ctransformers_raised")
            logger.warning(
                "extract_facts: local LLM raised: %s",
                diagnostics_safe_for_log(e),
            )
            diag.record_call(succeeded=False)
            return [], []

    diag.record_failure("local", reason="model_not_loaded")
    diag.record_call(succeeded=False, all_empty=True)
    return [], []


def extract_facts(text: str) -> List[str]:
    """
    Extract structured fact strings from raw text using LLM.

    Thin wrapper over :func:`_extract_facts_impl` that drops the KG-triple
    half of the payload. Returns 0-5 fact strings; [] when no tier yields
    usable output.
    """
    facts, _kg_triples = _extract_facts_impl(text)
    return facts


def extract_facts_with_triples(
    text: str,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Like :func:`extract_facts`, but also returns KG triples.

    The extraction prompt asks the model for a "kg" category of
    subject-predicate-object triples alongside the flat fact categories;
    this is the only accessor that surfaces them instead of discarding.
    Triples are already passed through :func:`validate_kg_triples`.

    Returns:
        (facts, triples) — facts matches extract_facts() exactly; triples is
        a list of validated (subject, predicate, object) tuples (possibly []).
    """
    cached = _EXTRACT_CACHE.get(text)
    if cached is not None:
        return cached
    result = _extract_facts_impl(text)
    _EXTRACT_CACHE.put(text, result)
    return result


class _ExtractResultCache:
    """Tiny content-keyed memo shared by the extraction accessors.

    BeamMemory's fact path and KG-triple path both need the results of one
    extraction pass over the same content. Memoizing on exact content keeps
    a single LLM call per remember() while letting each consumer keep its
    own seam (extract_facts_safe stays independently mockable for hosts
    that only want facts).
    """

    def __init__(self, maxsize: int = 32) -> None:
        self._maxsize = maxsize
        self._store: "collections.OrderedDict[str, tuple]" = (
            collections.OrderedDict()
        )

    def get(self, text):
        result = self._store.get(text)
        if result is not None:
            self._store.move_to_end(text)
        return result

    def put(self, text, result) -> None:
        self._store[text] = result
        self._store.move_to_end(text)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)


_EXTRACT_CACHE = _ExtractResultCache()


def extract_triples_for_beam(text: str) -> List[Tuple[str, str, str]]:
    """Best-effort KG-triple accessor mirroring extract_facts_safe.

    Shares the same LLM pass as extract_facts_safe via the content cache;
    never raises. Kept as a separate seam so the facts path can stay
    exactly as upstream tests and hosts patch it.
    """
    try:
        _facts, kg_triples = extract_facts_with_triples(text)
        return kg_triples
    except Exception as e:
        from mnemosyne.extraction.diagnostics import get_diagnostics, _safe_for_log
        diag = get_diagnostics()
        diag.record_failure(
            "wrapper", exc=e, reason="outer_wrapper_caught"
        )
        diag.record_call(succeeded=False)
        logger.warning(
            "extract_triples_for_beam: extraction raised: %s",
            _safe_for_log(e),
        )
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
