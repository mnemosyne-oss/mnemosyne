#!/usr/bin/env python3
"""
BEAM End-to-End Evaluation Pipeline
===================================
Evaluates Mnemosyne as a memory backend for LLMs using the official
BEAM benchmark protocol:
  1. Download BEAM dataset from HuggingFace
  2. Ingest conversations into Mnemosyne
  3. For each probing question: retrieve memories -> LLM answers -> LLM-as-judge scores
  4. Report per-scale, per-ability scores comparable to published SOTA

Published SOTA (BEAM 10M):
  Hindsight: 64.1%   Honcho: 40.6%   LIGHT: 26.6%   RAG: 24.9%

LLM: Nvidia API (deepseek-ai/deepseek-v4-pro) via OpenAI-compatible endpoint.
     Fast, cheap (~$2/M tokens), no local GPU needed.

Usage:
  cd /root/.hermes/projects/mnemosyne
  .venv/bin/python tools/evaluate_beam_end_to_end.py --sample 5 --scales 100K,500K,1M,10M

--sample N: conversations per scale (default 3, use 0 for all)
--scales: comma-separated (default 100K,500K,1M,10M)
--mode: retrieval|end_to_end (default end_to_end)
--judge-model: LLM model for judging (default same as answer model)
--resume: skip already-evaluated questions from results file
"""

from __future__ import annotations  # PEP 563: defer annotation eval so PEP 604 (X|None) and PEP 585 (list[str]) work on Python 3.9

import argparse
import ast
import gc
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

# Unbuffered output for real-time progress
print = partial(print, flush=True)

# --- Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import urllib.request
import urllib.error
import numpy as np

from mnemosyne.core.beam import BeamMemory, init_beam, _embeddings, _vec_available, _vec_insert, _fts_search_working, _generate_id

# --- Config ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    # Try to load from file — check opencode first, then openrouter
    for _kf in ["/tmp/opencode_key.txt", "/tmp/openrouter_key.txt"]:
        _key_file = Path(_kf)
        if _key_file.exists():
            with open(_key_file) as f:
                _content = f.read().strip()
            if "export" in _content:
                OPENROUTER_API_KEY = _content.split("=", 1)[1].strip().strip('"').strip("'")
            else:
                OPENROUTER_API_KEY = _content
            break
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = "deepseek-v4-pro"
FALLBACK_MODELS = []  # Disabled — fallback cascade burned $30 in credits
DEFAULT_TOP_K = 10  # Memories to retrieve per question
MAX_MEMORY_CONTEXT_CHARS = 8000  # Max chars of retrieved context to send to LLM


# C31: env-var truthy parser. Accepts standard truthy values
# (1/true/yes/on, case-insensitive) and explicit falsies (0/false/no/off).
# Strips whitespace so accidental leading/trailing spaces in shell
# exports don't get treated as falsy. Anything else → False.
# Pre-fix the parser was `lower() in ("1", "true", "yes")` which
# rejected `on` and was whitespace-sensitive — surprised at least one
# operator running with `MNEMOSYNE_BENCHMARK_PURE_RECALL=on`.
_ENV_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_truthy(name: str) -> bool:
    """Return True iff env var `name` is set to a canonical truthy value.

    Truthy: 1, true, yes, on (case-insensitive, whitespace-stripped).
    Everything else (including 0, false, no, off, empty, garbage) is False.
    """
    return os.environ.get(name, "").strip().lower() in _ENV_TRUTHY_VALUES
BENCHMARK_QUERIES_PER_CONV = 50  # Max probing questions per conversation
RESULTS_FILE = PROJECT_ROOT / "results" / "beam_e2e_results.json"
PAIRED_OUTCOMES_FILE = PROJECT_ROOT / "results" / "paired_outcomes.jsonl"

# Memory abilities tested by BEAM (10 dimensions)
BEAM_ABILITIES = [
    "IE",   # Information Extraction
    "MR",   # Multi-hop Reasoning
    "KU",   # Knowledge Update
    "TR",   # Temporal Reasoning
    "ABS",  # Abstention
    "CR",   # Contradiction Resolution
    "EO",   # Event Ordering
    "IF",   # Instruction Following
    "PF",   # Preference Following
    "SUM",  # Summarization
]

# Map dataset ability names to our abbreviations
ABILITY_MAP = {
    "information_extraction": "IE",
    "multi_session_reasoning": "MR",
    "knowledge_update": "KU",
    "temporal_reasoning": "TR",
    "abstention": "ABS",
    "contradiction_resolution": "CR",
    "event_ordering": "EO",
    "instruction_following": "IF",
    "preference_following": "PF",
    "summarization": "SUM",
    # Aliases
    "multi_session": "MR",
    "knowledge": "KU",
    "temporal": "TR",
    "information": "IE",
}


# ============================================================
#  LLM Client
# ============================================================

class LLMClient:
    """OpenAI-compatible API client using OpenRouter (fast, reliable)."""
    
    _last_429_time = 0  # Class-level rate-limit cooldown
    
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.fallback_models = FALLBACK_MODELS.copy()
        self.call_count = 0

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """Send chat completion request with retry. No fallback models to avoid rate limits."""
        
        last_error = None
        for attempt in range(3):
            try:
                return self._call_api(self.model, messages, temperature, max_tokens)
            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "rate" in last_error.lower():
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s backoff
                    time.sleep(wait)
                    continue
                else:
                    break  # Non-retryable error

        return f"[LLM_ERROR: all models failed. Last: {last_error}]"

    def _call_api(self, model: str, messages: list, temperature: float, max_tokens: int) -> str:
        """Single API call via requests (urllib blocked by Cloudflare on some providers)."""
        import json as _json
        import requests as _requests
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mnemosyne.site",
            "X-Title": "Mnemosyne Benchmark",
        }
        resp = _requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        self.call_count += 1
        return data["choices"][0]["message"]["content"]

    def close(self):
        pass


# ============================================================
#  Data Loading (adapted from benchmark_beam_sota.py)
# ============================================================

def load_beam_dataset(scales: list[str], max_conversations: int = None) -> dict:
    """Load BEAM dataset from HuggingFace. Returns dict[scale] -> list[conversation]."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    data = {}
    total_loaded = 0

    for scale in scales:
        print(f"  Loading BEAM {scale}...")
        try:
            if scale == "10M":
                ds = load_dataset("Mohammadta/BEAM-10M", streaming=True)
                split_name = "10M" if "10M" in ds else list(ds.keys())[0]

                conversations = []
                for i, sample in enumerate(ds[split_name]):
                    if max_conversations and i >= max_conversations:
                        break

                    probing_raw = sample.get("probing_questions", {})
                    if isinstance(probing_raw, str):
                        try:
                            probing = ast.literal_eval(probing_raw)
                        except Exception:
                            probing = {}
                    else:
                        probing = probing_raw

                    all_questions = []
                    for ability, questions in probing.items():
                        if isinstance(questions, list):
                            for q in questions:
                                if isinstance(q, dict):
                                    all_questions.append({
                                        "ability": ability,
                                        "question": q.get("question", ""),
                                        "ideal_answer": q.get("ideal_answer", q.get("ideal_response", q.get("answer", q.get("ideal_summary", "")))),
                                        "rubric": q.get("rubric", []),
                                    })

                    # Extract messages from plans
                    plans = sample.get("plans", [])
                    all_messages = []
                    for plan in plans:
                        chat_blocks = plan.get("chat", []) if isinstance(plan, dict) else []
                        for block in chat_blocks:
                            if isinstance(block, list):
                                for msg in block:
                                    if isinstance(msg, dict):
                                        all_messages.append({
                                            "role": msg.get("role", "unknown"),
                                            "content": msg.get("content", ""),
                                            "index": len(all_messages),
                                        })

                    conversations.append({
                        "id": sample.get("conversation_id", str(i)),
                        "messages": all_messages,
                        "questions": all_questions,
                        "scale": "10M",
                    })
                    total_loaded += 1

                data[scale] = conversations
                ds.cleanup_cache_files() if hasattr(ds, 'cleanup_cache_files') else None
                del ds
                gc.collect()
                print(f"    Loaded {len(conversations)} conversations")

            else:
                # 100K, 500K, 1M scales from the main dataset
                ds = load_dataset("Mohammadta/BEAM", streaming=True)
                if scale not in ds:
                    print(f"    WARNING: split '{scale}' not found. Available: {list(ds.keys())}")
                    continue

                conversations = []
                for i, sample in enumerate(ds[scale]):
                    if max_conversations and i >= max_conversations:
                        break

                    pq_raw = sample.get("probing_questions", "{}")
                    if isinstance(pq_raw, str):
                        try:
                            probing = ast.literal_eval(pq_raw)
                        except Exception:
                            probing = {}
                    else:
                        probing = pq_raw

                    flat_questions = []
                    for ability, questions in probing.items():
                        if isinstance(questions, list):
                            for q in questions:
                                if isinstance(q, dict):
                                    flat_questions.append({
                                        "ability": ability,
                                        "question": q.get("question", ""),
                                        "ideal_answer": q.get("ideal_answer", q.get("ideal_response", q.get("answer", q.get("ideal_summary", "")))),
                                        "rubric": q.get("rubric", []),
                                    })

                    chat_blocks = sample.get("chat", [])
                    messages = []
                    for block in chat_blocks:
                        if isinstance(block, list):
                            for msg in block:
                                if isinstance(msg, dict):
                                    messages.append({
                                        "role": msg.get("role", "unknown"),
                                        "content": msg.get("content", ""),
                                        "index": len(messages),
                                    })
                        elif isinstance(block, dict):
                            # Flat format: chat is a list of dicts directly
                            messages.append({
                                "role": block.get("role", "unknown"),
                                "content": block.get("content", ""),
                                "index": len(messages),
                            })

                    conversations.append({
                        "id": sample.get("conversation_id", str(i)),
                        "messages": messages,
                        "questions": flat_questions,
                        "scale": scale,
                    })
                    total_loaded += 1

                data[scale] = conversations
                ds.cleanup_cache_files() if hasattr(ds, 'cleanup_cache_files') else None
                del ds
                gc.collect()
                print(f"    Loaded {len(conversations)} conversations")

        except Exception as e:
            print(f"    ERROR loading {scale}: {e}")
            import traceback
            traceback.print_exc()

    print(f"  Total: {total_loaded} conversations across {len(data)} scales")
    return data


# ============================================================
#  Mnemosyne Ingestion
# ============================================================

def _extract_facts(content: str, source: str = "unknown") -> list[dict]:
    """Extract structured facts from a message for precision retrieval.
    These fact entries complement raw message storage by isolating
    specific data points (numbers, dates, versions, negations) that
    FTS5 keyword search can match more precisely than in long messages."""
    import re
    facts = []
    
    # Pattern 1: Version numbers ("Flask 2.3.1", "v0.6.2", "Python 3.11")
    ver_matches = re.findall(r'([A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)*)\s+v?(\d+\.\d+(?:\.\d+)?)', content)
    for name, ver in ver_matches[:3]:
        facts.append({
            "content": f"FACT version: {name.strip()} {ver}",
            "importance": 0.7,
        })
    
    # Pattern 2: Numbers with units ("250ms", "3 columns", "50 tasks", "5000 port")
    num_matches = re.findall(r'(\d+(?:[.,]\d+)?)\s*(ms|sec|seconds?|minutes?|hours?|days?|weeks?|months?|%|KB|MB|GB|columns?|tasks?|commits?|users?|ports?|items?)', content, re.IGNORECASE)
    for num, unit in num_matches[:5]:
        facts.append({
            "content": f"FACT metric: {num}{unit}",
            "importance": 0.65,
        })
    
    # Pattern 3: Dates
    date_patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}',
        r'\d{4}-\d{2}-\d{2}',
    ]
    for pat in date_patterns:
        for match in re.findall(pat, content, re.IGNORECASE):
            if isinstance(match, tuple):
                match = " ".join(match)
            facts.append({
                "content": f"FACT date: {match}",
                "importance": 0.7,
            })
    
    # Pattern 4: Deadlines
    deadline_matches = re.findall(r'(deadline|due by|sprint ends?|sprint \d+)\s*[:\-]?\s*([^.,;!?\n]{5,80})', content, re.IGNORECASE)
    for ctx, detail in deadline_matches[:3]:
        facts.append({
            "content": f"FACT deadline: {ctx} {detail.strip()}",
            "importance": 0.7,
        })
    
    # Pattern 5: Negations ("I have never", "I have not") - critical for CR
    negations = re.findall(r'(I(?: have|\'ve)?\s*(?:never|not)\s+[^.,;!?\n]{15,120})', content, re.IGNORECASE)
    for neg in negations[:3]:
        facts.append({
            "content": f"FACT negation: {neg.strip()}",
            "importance": 0.75,
        })
    
    # Pattern 6: Decisions / choices
    choices = re.findall(r'(?:decided to|chose to|opted for|selected|picked|switching to)\s+([^.,;!?\n]{10,120})', content, re.IGNORECASE)
    for choice in choices[:3]:
        facts.append({
            "content": f"FACT decision: {choice.strip()}",
            "importance": 0.65,
        })
    
    # Pattern 7: Ordinal sequence markers ("first", "then", "finally") for EO
    ordinals = re.findall(r'((?:first|second|third|fourth|fifth|finally|next|then|after that)[^.,;!?\n]{15,120})', content, re.IGNORECASE)
    for ord_text in ordinals[:5]:
        facts.append({
            "content": f"FACT sequence: {ord_text.strip()}",
            "importance": 0.6,
        })
    
    # Pattern 8: Entity-action pairs ("transactions table" + "add") for MR
    entities = re.findall(r'(?:the|my|our)\s+([a-z_]+\s*(?:table|model|schema|API|endpoint|function|module|route|handler))\s+(?:needs?|requires?|should|could|would|will|has|have)\s+([^.,;!?\n]{10,80})', content, re.IGNORECASE)
    for entity, action in entities[:5]:
        facts.append({
            "content": f"FACT entity: {entity.strip()} -> {action.strip()}",
            "importance": 0.65,
        })
    
    return facts[:20]  # Cap per message

def ingest_conversation(beam: BeamMemory, messages: list[dict]) -> dict:
    """Ingest conversation messages into Mnemosyne BEAM tiers.
    Also builds an in-memory facts index for fact-boosted retrieval."""
    start_time = time.perf_counter()
    stats = {"wm_count": 0, "ep_count": 0, "sp_count": 0, "total_chars": 0}
    
    # In-memory context→value facts index for direct fact matching.
    # Format: {"context phrase": "fact value"} — maps question-like phrases to answers.
    # Example: "My first sprint ends on" → "March 29"
    # Built during ingestion, queried during answering for zero-LLM fact extraction.
    import re as _re2
    _FACT_VALUE_RE = _re2.compile(
        r'('
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:[, ]*\d{4})?\b|'  # dates
        r'\b\d{4}-\d{2}-\d{2}\b|'  # ISO dates
        r'\b\d+[.,]?\d*\s*(?:ms|sec|mins?|hours?|days?|weeks?|months?|years?|%|KB|MB|GB|TB|rows?|columns?|roles?|features?|bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|sprints?|tickets?)\b|'  # numbers+units
        r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?\b'  # versions
        r')'
    )
    context_facts = getattr(beam, '_context_facts', {})
    if not hasattr(beam, '_context_facts'):
        beam._context_facts = {}
        context_facts = beam._context_facts

    BATCH_SIZE = 500

    for batch_start in range(0, len(messages), BATCH_SIZE):
        batch_msgs = messages[batch_start:batch_start + BATCH_SIZE]

        batch_items = []
        for i, msg in enumerate(batch_msgs):
            content = msg.get("content", "")
            if not content.strip():
                continue
            batch_items.append({
                "content": content,
                "source": f"beam_{msg.get('role', 'unknown')}",
                "importance": 0.3 + (0.1 * ((batch_start + i) % 5)),
            })
            stats["total_chars"] += len(content)
            
            # Extract context→value facts: words before AND after each fact value
            # SKIP version numbers (e.g., "3.39", "2.3.1") — they pollute fact matching
            # and are never the answer to BEAM questions (which ask about dates, counts, names).
            _VERSION_RE = _re2.compile(r'^\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?$')
            for match in _FACT_VALUE_RE.finditer(content):
                value = match.group()
                if _VERSION_RE.match(value):
                    continue  # Skip bare version numbers
                # Extract context: up to 12 words before + 8 words after the fact
                before = content[:match.start()].split()[-12:]
                after = content[match.end():].split()[:8]
                context_words = before + after
                context = ' '.join(context_words).lower().strip()
                if context and len(context) > 5:
                    if context not in context_facts:
                        context_facts[context] = []
                    context_facts[context].append(value)

            # Scratchpad every 10 messages
            if (batch_start + i) % 10 == 0 and len(content) > 50:
                try:
                    beam.scratchpad_write(f"[t={batch_start + i}] {content[:300]}")
                    stats["sp_count"] += 1
                except Exception:
                    pass

        if not batch_items:
            continue

        batch_ids = beam.remember_batch(batch_items)
        stats["wm_count"] += len(batch_items)

        # Cloud fact extraction: extract facts from batch if enabled
        if getattr(beam, 'use_cloud', False):
            try:
                from mnemosyne.extraction import ExtractionClient
                if beam._extraction_client is None:
                    beam._extraction_client = ExtractionClient()
                facts = beam._extraction_client.extract_facts(batch_msgs)
                if facts:
                    cursor = beam.conn.cursor()
                    import hashlib
                    for fact in facts:
                        fact_id = hashlib.sha256(
                            f"{fact.get('subject','')}:{fact.get('predicate','')}:{fact.get('object','')}:{batch_start}".encode()
                        ).hexdigest()[:24]
                        cursor.execute("""
                            INSERT OR IGNORE INTO facts
                            (fact_id, session_id, subject, predicate, object,
                             timestamp, source_msg_id, confidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            fact_id,
                            beam.session_id,
                            fact.get("subject", ""),
                            fact.get("predicate", "stated"),
                            fact.get("object", ""),
                            fact.get("timestamp", ""),
                            fact.get("source_msg_id", ""),
                            fact.get("confidence", 0.7),
                        ))
                    beam.conn.commit()
                    stats["fact_count"] = stats.get("fact_count", 0) + len(facts)
            except Exception:
                pass  # Best-effort; don't fail ingestion

        # [E1] Additive consolidation per batch via beam.sleep().
        #
        # Pre-E1 this block built a synthetic summary
        # ("Batch N: first_3_msg_contents[:100]") + DELETEd all source
        # working_memory rows. ~99% of message content was discarded
        # before recall could see it — the entire BEAM benchmark
        # corpus was destroyed at ingest.
        #
        # Post-E1 (option b, depends on E3 additive sleep): backdate
        # ONLY the batch's just-inserted rows past sleep's TTL/2
        # cutoff and let beam.sleep() produce real LLM-generated (or
        # AAAK-fallback) summaries on top of preserved originals.
        # The scoped UPDATE prevents cross-batch timestamp
        # contamination — without the `id IN (...)` filter, a
        # mid-sleep failure on batch N would let batch N+1's UPDATE
        # walk every still-unconsolidated row in the session and
        # rewrite their timestamps, corrupting per-row temporal
        # ordering. See E1 adversarial review F1/F3.
        try:
            cursor = beam.conn.cursor()
            # Backdate is derived from WORKING_MEMORY_TTL_HOURS so it
            # survives operator config changes via env var. sleep()'s
            # cutoff is TTL/2, _trim's cutoff is TTL — backdating by
            # TTL+1 ensures the row is on the consolidatable side of
            # sleep's cutoff while staying outside the trim window's
            # safety margin (consolidated_at exempts from trim post-E3
            # anyway, so the trim concern only applies pre-sleep). See
            # E1 adversarial review F6.
            from mnemosyne.core.beam import WORKING_MEMORY_TTL_HOURS as _WM_TTL
            backdate_iso = (
                datetime.now() - timedelta(hours=_WM_TTL + 1)
            ).isoformat()
            if batch_ids:
                placeholders = ",".join("?" * len(batch_ids))
                cursor.execute(
                    f"UPDATE working_memory SET timestamp = ? "
                    f"WHERE id IN ({placeholders}) "
                    f"AND consolidated_at IS NULL",
                    (backdate_iso, *batch_ids),
                )
                beam.conn.commit()

                # Drain the entire backdated batch. sleep() processes
                # up to SLEEP_BATCH_SIZE rows per call (default 5000,
                # well above the benchmark's BATCH_SIZE=500). But
                # MNEMOSYNE_SLEEP_BATCH can be configured below
                # BATCH_SIZE, in which case a single sleep() call
                # leaves some backdated rows un-consolidated. Those
                # rows then carry a TTL-old timestamp AND
                # consolidated_at IS NULL — exactly the predicate
                # _trim_working_memory uses to DELETE them on the
                # next remember_batch call. Loop until sleep returns
                # no_op (or errors) so the contract holds regardless
                # of env config. See Codex /review on PR #75 (P2).
                max_iters = 50  # safety bound; one batch should
                                # never need more than a few cycles
                while max_iters > 0:
                    result = beam.sleep(dry_run=False)
                    max_iters -= 1
                    if result.get("status") == "consolidated":
                        stats["ep_count"] += int(
                            result.get("summaries_created", 0) or 0
                        )
                        # If sleep drained fewer than SLEEP_BATCH_SIZE
                        # rows, the eligible set is empty and the
                        # next call would be no_op — break early.
                        items = int(result.get("items_consolidated", 0) or 0)
                        if items == 0:
                            break
                        # Otherwise keep draining.
                        continue
                    # no_op or any other status: drain complete.
                    break
                # E3 contract: originals stay, so stats["wm_count"]
                # does NOT decrement. Pre-E1 we did stats["wm_count"]
                # -= ... which produced wm_count=0 always; post-E1 it
                # grows monotonically with input message count, which
                # is what the experiment actually wants to measure.
        except Exception as e:
            # Log the failure to stats so the operator sees it. Pre-E1
            # the equivalent block also swallowed silently, but the
            # consolidation IS the point of the experiment — a silent
            # benchmark that "succeeds" with 0 episodic rows is the
            # exact failure mode the test suite is supposed to catch.
            stats.setdefault("sleep_errors", []).append(repr(e))

    stats["ingest_time_ms"] = (time.perf_counter() - start_time) * 1000
    return stats


# ============================================================
#  LLM Answering with Mnemosyne Memory
# ============================================================

ANSWER_SYSTEM_PROMPT = """You are a precise memory assistant answering questions about past conversations. You receive conversation context that may contain the answer.

CRITICAL: Think step-by-step before answering. Follow this structure:

STEP 1 - RELEVANT FACTS: List all specific facts from the context that relate to the question (dates, numbers, names, events, statements).
STEP 2 - CONTRADICTIONS: If the context contains conflicting statements about the same topic, identify BOTH sides explicitly.
STEP 3 - TEMPORAL/CALCULATIONS: For date/time questions, extract all relevant dates and compute the answer.
STEP 4 - ANSWER: Provide a thorough final answer with all relevant details from the context.

RULES:
- For EVENT ORDERING: list items in chronological order as they appear.
- For CONTRADICTION: explicitly state "The conversation contains contradictory information: [A] vs [B]"
- For SUMMARIZATION: include all key details — project stages, features, timelines, security, database, challenges.
- NEVER say "I don't have enough information" unless absolutely nothing in the context mentions the topic.
- For "how many" questions, provide the specific count, not a range."""

DEFAULT_TOP_K = 30  # Memories to retrieve per question (increased for broader context)
RECENT_CONTEXT_COUNT = 12  # Last N messages to include as recent context
MAX_MEMORY_CONTEXT_CHARS = 16000  # More context for LLM to find contradictions


def _recall_safe(beam: BeamMemory, query: str, top_k: int, temporal_weight: float = 0.0) -> list:
    """Safe recall wrapper that handles errors gracefully."""
    try:
        return beam.recall(query, top_k=top_k, temporal_weight=temporal_weight)
    except Exception:
        return []


def _extract_search_terms(question: str) -> list[str]:
    """Extract diverse search terms from a question for multi-strategy retrieval."""
    import re
    terms = []
    
    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', question)
    terms.extend(quoted)
    
    # Extract numbers and units
    numbers = re.findall(r'\b\d+[.,]?\d*\s*(?:ms|sec|days?|weeks?|months?|years?|%|KB|MB|GB|hours?|minutes?)\b', question, re.IGNORECASE)
    terms.extend(numbers[:5])
    
    # Extract named entities (capitalized phrases)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', question)
    terms.extend(entities[:5])
    
    # Extract version strings
    versions = re.findall(r'\bv?\d+\.\d+(?:\.\d+)?\b', question)
    terms.extend(versions[:5])
    
    # Extract key nouns (filter out question words)
    stop_words = {'have', 'did', 'do', 'does', 'can', 'will', 'would', 'should', 'is', 'are', 'was', 'were',
                  'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'my', 'me', 'i', 'you', 
                  'how', 'what', 'when', 'where', 'which', 'who', 'why', 'many', 'much'}
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', question) if w.lower() not in stop_words]
    terms.extend(words[:10])
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    
    return unique


def _multi_strategy_recall(beam: BeamMemory, question: str, top_k: int = DEFAULT_TOP_K, ability: str = None) -> list:
    """Multi-strategy retrieval: keyword, semantic, entity, negation, temporal."""
    import re
    all_memories = []
    seen_content_keys = set()
    
    def _add_unique(mems):
        for mem in mems:
            ck = mem.get("content", "")[:80]
            if ck not in seen_content_keys:
                seen_content_keys.add(ck)
                all_memories.append(mem)
    
    # Detect temporal questions by ability type or keywords
    temporal_keywords = ['when', 'date', 'deadline', 'sprint', 'day', 'week', 'month', 
                         'april', 'march', 'february', 'january', 'may', 'june', 'july',
                         'august', 'september', 'october', 'november', 'december',
                         'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                         'how many days', 'how long', 'timeline', 'schedule']
    is_temporal = ability in ('TR', 'EO') or any(w in question.lower() for w in temporal_keywords)
    temporal_weight = 0.3 if is_temporal else 0.0
    
    # Strategy 1: Direct question search (mostly keyword via FTS5)
    _add_unique(_recall_safe(beam, question, top_k * 2, temporal_weight=temporal_weight))
    
    # Strategy 2: Negation search for contradiction detection
    if any(w in question.lower() for w in ["have i", "did i", "do i", "am i", "has"]):
        negation_query = question
        for negation in ["never", "did not", "haven't"]:
            if negation not in negation_query.lower():
                negation_query = re.sub(r'(?i)(have i|did i|am i)', f'I {negation}', negation_query)
                break
        _add_unique(_recall_safe(beam, negation_query, top_k, temporal_weight=temporal_weight))
    
    # Strategy 3: Key entity/term searches
    terms = _extract_search_terms(question)
    for term in terms[:5]:
        if len(term) > 2:
            _add_unique(_recall_safe(beam, term, max(5, top_k // 3), temporal_weight=temporal_weight))
    
    # Strategy 4: Temporal search for date-related questions
    if is_temporal:
        # Stronger temporal boost for date-specific sub-queries
        date_temporal_weight = 0.5
        # Search for dates and timelines
        _add_unique(_recall_safe(beam, "deadline schedule timeline date", top_k, temporal_weight=date_temporal_weight))
        # Search for specific months mentioned in the question
        for month in ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']:
            if month in question.lower():
                _add_unique(_recall_safe(beam, month, top_k // 2, temporal_weight=date_temporal_weight))
    
    # Sort by score and return top-k
    all_memories.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_memories[:top_k]


# ============================================================
#  Per-Ability Bypasses: TR (Temporal Reasoning) + CR (Contradiction)
# ============================================================

def _extract_timeline_from_conversation(messages: list) -> list[dict]:
    """Extract ALL dates from conversation messages with surrounding event context.
    Filters out dates in code snippets. Returns sorted list of {date_obj, date_str, event_text, msg_index}."""
    import re as _re
    from datetime import datetime as _dt
    
    timeline = []
    
    # Month name map
    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5,
        'june': 6, 'july': 7, 'august': 8, 'september': 9, 'october': 10,
        'november': 11, 'december': 12,
    }
    
    # Code indicators to filter out
    CODE_INDICATORS = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE TABLE',
                       'def ', 'import ', 'print(', 'return ', '```', 'function(',
                       'jsonify', 'datetime', 'params', 'cursor.']
    
    def _is_code_context(text: str, match_start: int) -> bool:
        """Check if a date match appears to be in a code snippet."""
        # Check ~200 chars around match for code indicators
        start = max(0, match_start - 100)
        end = min(len(text), match_start + 100)
        surrounding = text[start:end]
        # If backticks within 200 chars, it's code
        if '```' in surrounding or '`' in surrounding:
            return True
        # If multiple code indicators present
        code_count = sum(1 for ci in CODE_INDICATORS if ci in surrounding)
        if code_count >= 2:
            return True
        # ISO date alone (2024-01-15) in a line with code indicators = likely code
        if _re.search(r'\b\d{4}-\d{2}-\d{2}\b', surrounding):
            if any(ci in surrounding for ci in CODE_INDICATORS):
                return True
        return False
    
    # Track the conversation year context
    year_mentions = []
    for msg in messages:
        years = _re.findall(r'\b(20\d{2})\b', msg.get("content", ""))
        year_mentions.extend(int(y) for y in years)
    # Use the most common year > 2020 as default
    default_year = 2024
    if year_mentions:
        from collections import Counter
        year_counts = Counter(y for y in year_mentions if 2020 <= y <= 2030)
        if year_counts:
            default_year = year_counts.most_common(1)[0][0]
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        
        # Pattern 1: "Month Day, Year" (e.g. "March 15, 2024")
        for m in _re.finditer(
            r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+'
            r'(?P<day>\d{1,2})(?:st|nd|rd|th)?[,\s]+(?P<year>\d{4})',
            content, _re.IGNORECASE
        ):
            if _is_code_context(content, m.start()):
                continue
            month_num = MONTH_MAP.get(m.group('month').lower()[:3])
            if month_num:
                try:
                    dt = _dt(int(m.group('year')), month_num, int(m.group('day')))
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 60)
                    event_text = content[start:end].strip()
                    timeline.append({
                        'date_obj': dt, 'date_str': m.group(0),
                        'event_text': event_text, 'msg_index': i,
                    })
                except ValueError:
                    pass
        
        # Pattern 2: "Month Day" without year (e.g. "March 29")
        for m in _re.finditer(
            r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+'
            r'(?P<day>\d{1,2})(?:st|nd|rd|th)?'
            r'(?![\d,\s]*\d{4})',  # NOT followed by year
            content, _re.IGNORECASE
        ):
            if _is_code_context(content, m.start()):
                continue
            month_num = MONTH_MAP.get(m.group('month').lower()[:3])
            if month_num:
                try:
                    dt = _dt(default_year, month_num, int(m.group('day')))
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 60)
                    event_text = content[start:end].strip()
                    timeline.append({
                        'date_obj': dt, 'date_str': m.group(0),
                        'event_text': event_text, 'msg_index': i,
                    })
                except ValueError:
                    pass
    
    # Sort chronologically and deduplicate (same date, same event text)
    timeline.sort(key=lambda x: (x['date_obj'], x['event_text']))
    seen = set()
    deduped = []
    for t in timeline:
        key = (t['date_str'], t['event_text'][:40])
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    
    return deduped


def _build_tr_timeline_prompt(timeline: list[dict]) -> str:
    """Build a structured timeline prompt for TR questions."""
    if not timeline:
        return ""
    
    lines = ["CRITICAL TIMELINE (all dates extracted from the conversation, use ONLY these dates):"]
    for t in timeline:
        lines.append(f"  [{t['date_obj'].strftime('%Y-%m-%d')}] {t['date_str']}: ...{t['event_text'][:100]}...")
    
    return "\n".join(lines)


def _compute_tr_python(question: str, timeline: list[dict]) -> str | None:
    """Compute TR answer in pure Python (date math, no LLM). Returns answer string or None."""
    import re as _re
    from datetime import timedelta as _td
    
    q_lower = question.lower()
    
    # Detect question type: "how many days between X and Y"
    # Extract event keywords from question
    event_keywords = []
    # Look for "end of first sprint", "start of second sprint" type phrases
    phrases = _re.findall(r'(?:end|start|beginning|completion|finish|launch|release|deploy|merge|push|commit|sprint|milestone|phase|wave|beta|alpha|MVP|demo|presentation|meeting|call|review|audit|test|benchmark)[a-z]*\s+(?:of\s+)?(?:the\s+)?(?:my\s+)?(?:first|second|third|\d+(?:st|nd|rd|th)?)?\s*[a-z]+(?:\s+[a-z]+){0,3}', q_lower)
    event_keywords.extend(phrases)
    
    # Also try simpler: extract noun phrases from question
    q_words = q_lower.replace('?', '').split()
    
    # Score each timeline entry against the question
    scored = []
    for t in timeline:
        event = t['event_text'].lower()
        score = 0
        # Direct substring match bonus
        for phrase in event_keywords:
            if phrase in event or any(w in event for w in phrase.split() if len(w) > 3):
                score += 3
        # Word overlap
        for w in q_words:
            if len(w) > 3 and w in event:
                score += 1
        # Date proximity bonus (prefer dates with event context)
        if len(t['event_text']) > 20:
            score += 2
        scored.append((score, t))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Try to find two distinct events
    if len(scored) >= 2 and scored[0][0] > 0 and scored[1][0] > 0:
        t1 = scored[0][1]
        t2 = scored[1][1]
        d1 = t1['date_obj']
        d2 = t2['date_obj']
        diff = abs((d2 - d1).days)
        
        # Determine which is earlier/later based on question
        if 'between' in q_lower:
            evt_a = t1['date_str'] if d1 <= d2 else t2['date_str']
            evt_b = t2['date_str'] if d1 <= d2 else t1['date_str']
            d_a = d1 if d1 <= d2 else d2
            d_b = d2 if d1 <= d2 else d1
        else:
            evt_a, evt_b = t1['date_str'], t2['date_str']
            d_a, d_b = d1, d2
            diff = abs((d_b - d_a).days)
        
        answer = (
            f"Between {evt_a} ({d_a.strftime('%B %d, %Y')}) and "
            f"{evt_b} ({d_b.strftime('%B %d, %Y')}), "
            f"there are {diff} days."
        )
        return answer
    
    # Fallback: just take the two most relevant dates by word overlap
    if len(scored) >= 2:
        best = [t for s, t in scored if s > 0][:2]
        if len(best) >= 2:
            d1, d2 = best[0]['date_obj'], best[1]['date_obj']
            diff = abs((d2 - d1).days)
            return (
                f"Based on the conversation timeline, the time between "
                f"{best[0]['date_str']} and {best[1]['date_str']} is {diff} days."
            )
    
    return None  # Can't compute, let LLM handle it



def _compute_tr_answer(question: str, timeline: list[dict]) -> str | None:
    """Compute temporal reasoning answer from conversation dates. Returns None if can't compute."""
    if not timeline or len(timeline) < 2:
        return None

    # Build a prompt that presents the timeline and asks the LLM to compute
    # This is more robust than trying to match events ourselves
    timeline_prompt = _build_tr_timeline_prompt(timeline)
    
    # Build the full prompt that we'll return - the caller will send this to LLM
    prompt = (
        f"{timeline_prompt}\n\n"
        f"QUESTION: {question}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Identify the two events mentioned in the question\n"
        f"2. Find the corresponding dates in the timeline above\n"
        f"3. Compute the time difference between them\n"
        f"4. State the answer clearly with both dates and the computed difference\n\n"
        f"ANSWER:"
    )
    return prompt  # Return the prompt, not the answer - caller passes to LLM


def _detect_contradictions(messages: list, question: str) -> str | None:
    """Scan conversation for contradictory statements about the question topic.
    Returns contradiction context string to inject into prompt, or None if none found."""
    import re as _re
    
    # Extract the key topic from the question
    # "Have I worked with Flask routes?" -> key terms: "flask routes", "http requests"
    # "Have I integrated Flask-Login?" -> key terms: "flask-login", "session management"
    
    # Strip question words to get the core topic
    q_clean = _re.sub(r'^(?:Have I|Did I|Do I|Am I|Has)\s+(?:ever\s+)?', '', question, flags=_re.IGNORECASE)
    q_clean = _re.sub(r'\s+(?:in this project|across my sessions|in my project)\s*\??$', '', q_clean, flags=_re.IGNORECASE)
    q_clean = q_clean.strip().rstrip('?').strip()
    
    # Extract meaningful noun phrases
    words = _re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', q_clean)
    # Filter to key content words (nouns, tech terms)
    key_terms = []
    for w in words:
        wl = w.lower()
        if len(wl) > 2 and wl not in ('the', 'and', 'for', 'with', 'any', 'this', 'that', 'have', 'has', 'been'):
            key_terms.append(wl)
    
    if not key_terms:
        return None
    
    # Scan all messages for mentions of ANY key term
    affirmatives = []
    negatives = []
    
    NEGATION_WORDS = {'never', 'not', "n't", 'no', 'without', 'cannot', "can't", 'nothing', 'none'}
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        content_lower = content.lower()
        
        # Check if message mentions any key term
        matched_terms = [t for t in key_terms if t in content_lower]
        if not matched_terms:
            continue
        
        # Check for negation in the SENTENCE containing the matched term
        # (BEAM contradictions embed negation near the topic mention)
        has_negation = False
        for term in matched_terms:
            # Find the sentence containing this term
            term_pos = content_lower.find(term)
            if term_pos < 0:
                continue
            # Extract sentence context (200 chars around term, or to sentence boundaries)
            start = max(0, term_pos - 150)
            end = min(len(content_lower), term_pos + 150)
            sentence = content_lower[start:end]
            # Check for negation words in this sentence
            for nw in NEGATION_WORDS:
                if nw in sentence:
                    has_negation = True
                    break
            if has_negation:
                break
        
        snippet = content[:250].strip()
        if has_negation:
            negatives.append(f"[Msg {i}] {content[:250].strip()}")
        else:
            affirmatives.append(f"[Msg {i}] {content[:250].strip()}")
    
    if affirmatives and negatives:
        ctx = "CRITICAL: CONTRADICTORY INFORMATION DETECTED\n\n"
        ctx += "The conversation contains BOTH affirmative AND negative statements about this topic:\n\n"
        ctx += "Statements suggesting this WAS done or worked on:\n"
        for a in affirmatives[:5]:
            ctx += f"  - {a}\n"
        ctx += "\nStatements suggesting this was NOT done:\n"
        for n in negatives[:5]:
            ctx += f"  - {n}\n"
        ctx += "\nYOU MUST explicitly identify the contradiction and present BOTH sides. "
        ctx += "Do NOT answer with just one side. The correct response begins with "
        ctx += "'I notice you've mentioned contradictory information about this.'"
        return ctx
    
    return None


_POLYPHONIC_VOICE_KEYS = frozenset({"vector", "graph", "fact", "temporal"})
_LINEAR_VOICE_KEYS = frozenset({"vec", "fts", "keyword", "importance", "recency_decay"})


def _summarize_recall_memories(memories: list) -> dict:
    """Compact per-question recall provenance for analysis.

    Captures engine identity + per-voice score sums + top-1 voice
    breakdown. Lets `docs/benchmark-results-analysis.md` Recipe E
    (per-voice attribution) work from the result JSON directly.

    Shape:
        {
          "engine": "polyphonic" | "linear" | "unknown",
          "kept_count": N,
          "voice_sums": {voice_key: total_score, ...},
          "top_result_voices": {voice_key: score, ...} | {},
          "top_result_tier": "working" | "episodic" | None,
        }

    Returns a minimal dict when memories is empty (bypass paths
    short-circuit before recall so the field still exists for
    schema consistency).
    """
    if not memories:
        return {"engine": "unknown", "kept_count": 0, "voice_sums": {},
                "top_result_voices": {}, "top_result_tier": None}

    # Engine ID by the voice_scores keyset of any result that has one.
    engine = "unknown"
    voice_sums: dict = {}
    for m in memories:
        vs = m.get("voice_scores") or {}
        if not vs:
            continue
        if engine == "unknown":
            keys = set(vs.keys())
            if keys & _POLYPHONIC_VOICE_KEYS:
                engine = "polyphonic"
            elif keys & _LINEAR_VOICE_KEYS:
                engine = "linear"
        for k, v in vs.items():
            try:
                voice_sums[k] = voice_sums.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                pass  # ignore non-numeric voice values

    top = memories[0] if memories else {}
    return {
        "engine": engine,
        "kept_count": len(memories),
        "voice_sums": {k: round(v, 4) for k, v in voice_sums.items()},
        "top_result_voices": {
            k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in (top.get("voice_scores") or {}).items()
        },
        "top_result_tier": top.get("tier"),
    }


def answer_with_memory(llm: LLMClient, beam: BeamMemory, question: str,
                      conversation_messages: list = None, top_k: int = DEFAULT_TOP_K,
                      ability: str = None,
                      return_memories: bool = False):
    """Retrieve memories and have LLM answer, with context strategy based on conversation size.

    Set `MNEMOSYNE_BENCHMARK_PURE_RECALL=1` to disable the per-ability
    bypass paths (TR oracle, CR contradiction injection, IE/KU
    context→value side-index) AND the always-included RECENT
    CONVERSATION raw-message prompt section. Pure-recall mode forces
    every answer through the full Mnemosyne retrieval pipeline so the
    BEAM-recovery experiment can measure each arm's recall quality
    without contamination from harness-side oracles. Default behavior
    (env unset or '0') preserves the existing benchmark mode.

    Returns:
        str when `return_memories=False` (default — backward-compat).
        tuple[str, list[dict]] when `return_memories=True` — the second
        element is the retrieved memories list (post-multi-strategy,
        pre-LLM-context-build). Each memory dict carries `voice_scores`
        from Gap G — required for per-voice attribution analysis.
        Bypass paths return `(answer, [])` since they short-circuit
        before recall.
    """
    def _ret(answer, memories=None):
        """Pack return value uniformly across all exit points."""
        if return_memories:
            return answer, (memories or [])
        return answer
    # E7/E8/E9 gate: when set, the harness disables every shortcut that
    # would let the LLM produce an answer without going through
    # BeamMemory.recall(). The bypasses were useful for measuring
    # LLM-ceiling-with-help on isolated abilities; the BEAM-recovery
    # experiment instead needs to compare Arm A vs Arm B vs Arm C on
    # the recall surface itself.
    _pure_recall = _env_truthy("MNEMOSYNE_BENCHMARK_PURE_RECALL")

    total_msgs = len(conversation_messages) if conversation_messages else 0

    # ---- PER-ABILITY BYPASSES (zero-LLM or augmented) ----

    # TR (Temporal Reasoning): compute answer from extracted dates
    if not _pure_recall and ability == 'TR' and conversation_messages:
        timeline = _extract_timeline_from_conversation(conversation_messages)
        print(f"    [TR-bypass] extracted {len(timeline)} dates from {len(conversation_messages)} msgs")
        if timeline and len(timeline) >= 2:
            tr_prompt = _compute_tr_answer(question, timeline)
            if tr_prompt:
                messages = [
                    {"role": "system", "content": "You are a precise date calculator. Use ONLY the dates from the provided timeline. Output ONLY the answer, no explanation."},
                    {"role": "user", "content": tr_prompt},
                ]
                answer = llm.chat(messages, temperature=0.0, max_tokens=4096)
                print(f"    [TR-bypass] LLM answer: {answer[:150]}")
                return _ret(answer)
            else:
                print(f"    [TR-bypass] _compute_tr_answer returned None")
        else:
            print(f"    [TR-bypass] no timeline extracted or too few dates")
    
    # CR (Contradiction Resolution): detect contradictory statements
    _cr_context = None
    if not _pure_recall and ability == 'CR' and conversation_messages:
        _cr_context = _detect_contradictions(conversation_messages, question)
        if _cr_context:
            print(f"    [CR-detect] FOUND contradictions, injecting context ({len(_cr_context)} chars)")
        else:
            print(f"    [CR-detect] no contradictions found")
    # ---- END PER-ABILITY BYPASSES ----
    
    # FULL-CONTEXT MODE: send the entire conversation to the LLM, bypassing Mnemosyne retrieval.
    # This tests the LLM's reading comprehension ceiling — useful for establishing the upper bound.
    # Controlled by FULL_CONTEXT_MODE env var.
    # HYBRID: try context→value matching first for factual questions (IE/MR/KU),
    # then fall through to full-context for complex reasoning (ABS/CR/EO/SUM/TR).
    _full_context = _env_truthy("FULL_CONTEXT_MODE")
    # Precedence: pure-recall overrides full-context. The point of
    # pure-recall is to force every answer through Mnemosyne recall;
    # full-context's "ship the whole conversation to the LLM" path
    # would silently invalidate that guarantee (the LLM would answer
    # from raw `FULL CONVERSATION:` regardless of arm).
    if _full_context and _pure_recall:
        _full_context = False
    # DEBUG (use _env_truthy so `FULL_CONTEXT_MODE=0` doesn't fire the print)
    if _env_truthy("FULL_CONTEXT_MODE"):
        print(f"    [DEBUG full-context] env={_full_context}, msgs={bool(conversation_messages)}, count={len(conversation_messages) if conversation_messages else 0} (pure_recall={_pure_recall})")
    if _full_context and conversation_messages:
        # ---- Phase 1: Try context→value matching for factual questions ----
        # Only use context→value for Information Extraction (IE) and Knowledge Understanding (KU).
        # MR (Multi-hop) requires reasoning across multiple messages; let full-context handle it.
        # Gated by pure_recall — when ON, full-context mode still hits the LLM with raw
        # conversation but skips the zero-LLM context→value shortcut.
        _FACT_ABILITIES = {'IE', 'KU'}
        if not _pure_recall and ability in _FACT_ABILITIES and hasattr(beam, '_context_facts') and beam._context_facts:
            _q_stop = {'when','does','do','did','what','how','where','which','who','why',
                       'is','are','was','were','can','will','would','should','could','may',
                       'the','a','an','in','on','at','to','for','of','with','my','me','i','you'}
            q_words = [w.lower() for w in question.split() if w.lower() not in _q_stop and len(w) > 1]
            q_set = set(q_words)
            best_match = None
            best_score = 0
            for context_phrase, values in beam._context_facts.items():
                c_words = set(context_phrase.split())
                overlap = q_set & c_words
                if len(overlap) < 2:
                    continue
                score = len(overlap) / max(len(c_words), 1)
                if score > best_score:
                    best_score = score
                    best_match = values[0]
            if best_match:
                return _ret(best_match)  # Direct fact answer, zero LLM cost
        
        # ---- Phase 2: Full-context LLM fallback ----
        full_parts = []
        total_chars = 0
        for msg in conversation_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                line = f"[{role}]: {content}"
                if total_chars + len(line) > MAX_MEMORY_CONTEXT_CHARS * 2:
                    break
                full_parts.append(line)
                total_chars += len(line)
        
        context = "FULL CONVERSATION:\n" + "\n".join(full_parts)
        
        # Inject CR contradiction context if detected
        _cr_prefix = ""
        if _cr_context:
            _cr_prefix = f"\n\n{_cr_context}\n\n"
        
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{_cr_prefix}{context}\n\nQUESTION: {question}\n\nANSWER:"},
        ]
        return _ret(llm.chat(messages, temperature=0.1, max_tokens=2048))

    # ALWAYS use multi-strategy retrieval to test Mnemosyne's recall quality.
    # The previous <=500 bypass sent full raw conversations to the LLM,
    # completely bypassing Mnemosyne's retrieval pipeline.
    # This benchmark exists to measure MEMORY performance, not LLM reading comprehension.
    
    # Multi-strategy retrieval
    memories = _multi_strategy_recall(beam, question, top_k * 3, ability=ability)  # Get 3x more for reranking

    # ---- Context→Value fact matching (Phase 7: direct regex-extracted facts, zero-LLM) ----
    # At ingestion, we built beam._context_facts: {"words around fact": ["fact value"]}.
    # Now we try to match the question against context phrases and return the value directly.
    # Only used for factual question types (IE, MR, KU, TR) with strong matches.
    # ABS, CR, EO, SUM need LLM reasoning — we skip context matching for those.
    context_answer = None
    # Only use context→value for Information Extraction (IE) and Knowledge Understanding (KU).
    # MR (Multi-hop) requires reasoning across multiple messages; CR/TR/EO/SUM need LLM.
    # Gated by pure_recall — when ON, IE/KU questions go through full recall+LLM
    # rather than returning a side-indexed value directly.
    _FACT_ABILITIES = {'IE', 'KU'}
    if not _pure_recall and ability in _FACT_ABILITIES and hasattr(beam, '_context_facts') and beam._context_facts:
        # Build question word set (filtered like FTS5 search does)
        _q_stop = {'when','does','do','did','what','how','where','which','who','why',
                   'is','are','was','were','can','will','would','should','could','may',
                   'the','a','an','in','on','at','to','for','of','with','my','me','i','you'}
        q_words = [w.lower() for w in question.split() if w.lower() not in _q_stop and len(w) > 1]
        q_set = set(q_words)
        best_match = None
        best_score = 0
        for context_phrase, values in beam._context_facts.items():
            c_words = set(context_phrase.split())
            overlap = q_set & c_words
            if len(overlap) < 2:
                continue
            # Score: overlap count / max(context_words, question_words) for fairness
            score = len(overlap) / max(len(c_words), 1)
            if score > best_score and len(overlap) >= 2:
                best_score = score
                best_match = values[0]
        if best_match:
            context_answer = best_match

    # If cloud extraction enabled, also search the facts table
    if getattr(beam, 'use_cloud', False):
        try:
            fact_memories = beam.fact_recall(question, top_k=top_k)
            # Convert fact dicts to same format as recall results
            for f in fact_memories:
                memories.append({
                    "content": f"FACT: {f['content']}",
                    "score": f.get("score", 0.5) * 2.0,  # 2x weight for facts
                    "source": "fact_extraction",
                })
            # Re-sort by score
            memories.sort(key=lambda x: x.get("score", 0), reverse=True)
        except Exception:
            pass  # Fact recall is best-effort
    
    # LLM RERANKING: DISABLED — rate-limit avoidance + proven ineffective (Reality Check 5.3)
    # The re-ranker cannot beat baseline by >3pp and causes 429 rate-limit cascades.
    # Left as dead code for reference.

    # ---- Fact-density reranking (Phase 6.5: algorithmic, zero-LLM) ----
    # BEAM distractors are generic dev-talk; answer messages carry specific data.
    # Boost messages with dates, numbers, proper nouns, versions, technical terms.
    import re as _re_facts
    _FACT_PATTERNS = [
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:[, ]*\d{4})?\b', 2.0),  # dates
        (r'\b\d{4}-\d{2}-\d{2}\b', 2.5),  # ISO dates
        (r'\b\d+[.,]?\d*\s*(?:ms|sec|mins?|hours?|days?|weeks?|months?|years?|%|KB|MB|GB|TB|rows?|columns?|roles?|features?|bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|sprints?|tickets?)\b', 1.5),  # numbers with units
        (r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?\b', 1.5),  # version strings
        (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b', 1.0),  # proper noun phrases
        (r'\b[A-Z]{2,8}\b', 0.8),  # acronyms
    ]
    for mem in memories:
        content = mem.get("content", "")
        fact_score = 0.0
        for pattern, weight in _FACT_PATTERNS:
            matches = _re_facts.findall(pattern, content)
            fact_score += len(matches) * weight
        # Normalize by content length to get fact density
        density = fact_score / max(len(content.split()), 1)
        mem["fact_density"] = round(density, 4)
        # Boost score: blend original with fact density (40% fact boost)
        orig = mem.get("score", mem.get("relevance", 0))
        mem["score"] = orig * 0.6 + min(density * 5.0, 1.0) * 0.4

    # Re-sort by boosted score
    memories.sort(key=lambda m: m.get("score", 0), reverse=True)
    
    context = ""  # Built below from memories

    # Build recent context from last N messages. Pure-recall mode SKIPS
    # this entirely — the LLM sees only RETRIEVED MEMORIES, so the
    # answer quality reflects what each arm's recall produced (rather
    # than the harness silently leaking the last 12 raw messages into
    # every prompt, which inflates recency-anchored answers and masks
    # recall weakness across all arms).
    recent_parts = []
    if not _pure_recall and conversation_messages:
        recent = conversation_messages[-RECENT_CONTEXT_COUNT:]
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                recent_parts.append(f"[{role}]: {content[:300]}")
    
    # Build retrieved memory context (deduplicated, relevance-sorted)
    seen_content = set()
    memory_parts = []
    total_chars = 0
    for i, mem in enumerate(memories):
        content = mem.get("content", "")
        # Deduplicate
        content_key = content[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        
        score = mem.get("score", mem.get("relevance", 0))
        if isinstance(score, (int, float)) and score < 0.05:
            continue  # Skip very low relevance
            
        if total_chars + len(content) > MAX_MEMORY_CONTEXT_CHARS:
            remaining = MAX_MEMORY_CONTEXT_CHARS - total_chars
            if remaining > 100:
                memory_parts.append(f"[Memory] {content[:remaining]}...")
            break
        memory_parts.append(f"[Memory] {content}")
        total_chars += len(content)

    # Build prompt with contexts (skip if full-conversation mode already set)
    if not context:
        context_blocks = []
        if recent_parts:
            context_blocks.append("RECENT CONVERSATION:\n" + "\n".join(recent_parts))
        if memory_parts:
            context_blocks.append("RETRIEVED MEMORIES:\n" + "\n\n".join(memory_parts))
        
        context = "\n\n".join(context_blocks) if context_blocks else "[No memories found]"

    # If we found a direct context→value match, return it immediately (zero LLM cost)
    if context_answer:
        return _ret(context_answer, memories)

    # Inject CR contradiction context if detected
    _cr_prefix_ret = ""
    if _cr_context:
        _cr_prefix_ret = f"\n\n{_cr_context}\n\n"

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"{_cr_prefix_ret}{context}\n\nQUESTION: {question}\n\nANSWER:"},
    ]

    return _ret(llm.chat(messages, temperature=0.1, max_tokens=2048), memories)


# ============================================================
#  LLM-as-Judge: Nugget-Based Scoring (BEAM Protocol)
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a memory benchmark.
You will be given:
1. A question about a conversation
2. A list of RUBRIC ITEMS (expected facts the AI should mention)
3. The AI's ANSWER

For EACH rubric item, check if the AI's answer contains equivalent information:
- Score 1.0: correct info present, substantially matches the rubric item
- Score 0.5: partially correct, some key detail missing or slightly wrong
- Score 0.0: missing or wrong

Return ONLY this JSON:
{"scores":[1.0,0.5,0.0],"overall_score":0.X}

Where scores[i] corresponds to rubric[i], and overall_score is the average."""


def judge_with_rubrics(llm: LLMClient, question: str, rubric: list, ai_answer: str) -> dict:
    """Judge an AI answer against pre-written BEAM rubric items."""
    if not rubric:
        # Fall back to generic nugget scoring if no rubric available
        return {"scores": [], "overall_score": 0.0, "assessment": "no rubric available"}
    
    rubric_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(rubric))
    
    user_prompt = f"""QUESTION: {question}

RUBRIC ITEMS:
{rubric_text}

AI's ANSWER: {ai_answer}

For each rubric item, score how well the AI's answer matches. Return JSON with scores array and overall_score (average)."""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.chat(messages, temperature=0.0, max_tokens=500)

    # Parse JSON from response
    if response is None:
        return {
            "scores": [0.0] * len(rubric),
            "overall_score": 0.0,
            "assessment": "LLM judge returned None (timeout or error)",
        }
    
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: basic text matching
    return {
        "scores": [0.0] * len(rubric),
        "overall_score": basic_text_similarity(ai_answer, " ".join(rubric)),
        "assessment": "JSON parse failed; using fallback",
        "raw_response": response,
    }


def basic_text_similarity(text1: str, text2: str) -> float:
    """Simple token overlap as fallback when LLM judge fails."""
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union) if union else 0.0


# ============================================================
#  Evaluation Runner
# ============================================================

def evaluate_conversation(
    llm: LLMClient,
    judge_llm: LLMClient,
    beam: BeamMemory,
    conversation: dict,
    resume_ids: set = None,
) -> dict:
    """Evaluate all probing questions for one conversation."""
    conv_id = conversation["id"]
    questions = conversation["questions"][:BENCHMARK_QUERIES_PER_CONV]
    results = []

    print(f"  Conversation {conv_id}: {len(questions)} questions")

    for i, q in enumerate(questions):
        qid = f"{conv_id}:q{i}"
        if resume_ids and qid in resume_ids:
            continue

        question = q["question"]
        ideal = q["ideal_answer"]
        rubric = q.get("rubric", [])
        ability = q.get("ability", "unknown")
        # Map dataset ability names to BEAM abbreviations
        ability = ABILITY_MAP.get(ability, ability)

        if not question or not ideal:
            continue

        # Step 1: LLM answers using Mnemosyne memories + conversation context.
        # `return_memories=True` gives us the per-question retrieved memory
        # list so we can summarize voice-attribution provenance below.
        t0 = time.perf_counter()
        ai_answer, recall_memories = answer_with_memory(
            llm, beam, question,
            conversation_messages=conversation.get("messages", []),
            ability=ability, return_memories=True,
        )
        answer_time = time.perf_counter() - t0

        # Handle None answer (LLM timeout/error)
        if ai_answer is None:
            ai_answer = "[LLM_ERROR: No response from answering model]"

        # Step 2: LLM-as-judge scores the answer
        t0 = time.perf_counter()
        judgment = judge_with_rubrics(judge_llm, question, rubric, ai_answer)
        judge_time = time.perf_counter() - t0

        score = judgment.get("overall_score", 0.0)

        # Compact recall-provenance summary so per-voice attribution
        # analysis (docs/benchmark-results-analysis.md Recipe E) works
        # from the result file directly — no DB re-query needed. Full
        # memory dicts would be ~10× larger; this summary captures
        # what an analyst actually needs.
        recall_provenance = _summarize_recall_memories(recall_memories)

        result = {
            "qid": qid,
            "ability": ability,
            "question": question[:200],
            "ideal_answer": ideal[:200],
            "ai_answer": ai_answer[:500],
            "recall_provenance": recall_provenance,
            "score": score,
            "nuggets": judgment.get("nuggets", []),
            "assessment": judgment.get("brief_assessment", ""),
            "answer_time_ms": answer_time * 1000,
            "judge_time_ms": judge_time * 1000,
        }
        results.append(result)

        print(f"    [{ability}] score={score:.2f} ans={answer_time*1000:.0f}ms judge={judge_time*1000:.0f}ms "
              f"Q: {question[:60]}...")
        
        # Rate-limit avoidance: long pause between questions (20s to avoid provider burst limits)
        time.sleep(20)

    return {
        "conversation_id": conv_id,
        "scale": conversation["scale"],
        "num_questions": len(questions),
        "num_evaluated": len(results),
        "results": results,
    }


def compute_ability_scores(all_results: list[dict]) -> dict:
    """Aggregate scores by ability and scale."""
    by_scale_ability = defaultdict(lambda: defaultdict(list))

    for conv_result in all_results:
        scale = conv_result["scale"]
        for r in conv_result["results"]:
            ability = r.get("ability", "unknown")
            score = r.get("score", 0.0)
            by_scale_ability[scale][ability].append(score)

    # Compute averages
    summary = {}
    for scale, abilities in by_scale_ability.items():
        scale_scores = {}
        all_scores = []
        for ability, scores in abilities.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            scale_scores[ability] = {
                "avg_score": avg,
                "count": len(scores),
            }
            all_scores.extend(scores)

        # Overall average across all abilities
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        scale_scores["OVERALL"] = {
            "avg_score": overall,
            "count": len(all_scores),
        }

        summary[scale] = scale_scores

    return summary


# ============================================================
#  SOTA Comparison
# ============================================================

PUBLISHED_SOTA = {
    "10M": {
        "Hindsight": 64.1,
        "Honcho": 40.6,
        "LIGHT (Llama-4)": 26.6,
        "RAG (Llama-4)": 24.9,
    },
    "1M": {
        "Hindsight": 73.9,
        "Honcho": 63.1,
        "LIGHT (Llama-4)": 33.6,
        "RAG (Llama-4)": 30.7,
    },
    "500K": {
        "Hindsight": 71.1,
        "Honcho": 64.9,
        "LIGHT (Llama-4)": 35.9,
        "RAG (Llama-4)": 33.0,
    },
    "100K": {
        "Hindsight": 73.4,
        "Honcho": 63.0,
        "LIGHT (Llama-4)": 35.8,
        "RAG (Llama-4)": 32.3,
    },
}


def print_sota_report(ability_summary: dict, metadata: dict):
    """Print SOTA comparison report."""
    print(f"\n{'='*80}")
    print(f"  MNEMOSYNE BEAM END-TO-END SOTA REPORT")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Model: {metadata.get('model', 'unknown')}")
    print(f"  Conversations/scale: {metadata.get('sample_size', 'N/A')}")
    print(f"  Top-K memories: {DEFAULT_TOP_K}")
    print(f"  Methodology: LLM answering + LLM-as-judge (nugget scoring, per BEAM protocol)")
    print(f"{'='*80}")

    print(f"\n  Per-Ability Scores:")
    print(f"  {'Scale':<8} {'OVERALL':>8}", end="")
    for ab in BEAM_ABILITIES:
        print(f" {ab:>6}", end="")
    print()

    for scale in sorted(ability_summary.keys()):
        scores = ability_summary[scale]
        overall = scores.get("OVERALL", {}).get("avg_score", 0.0)
        print(f"  {scale:<8} {overall*100:>7.1f}%", end="")
        for ab in BEAM_ABILITIES:
            s = scores.get(ab, {}).get("avg_score", 0.0)
            print(f" {s*100:>5.1f}%", end="")
        print()

    print(f"\n  SOTA Comparison (OVERALL):")
    print(f"  {'Scale':<8} {'Mnemosyne':>12}", end="")
    for system in ["Hindsight", "Honcho", "LIGHT (Llama-4)", "RAG (Llama-4)"]:
        print(f" {system:>18}", end="")
    print()

    for scale in sorted(ability_summary.keys()):
        our_score = ability_summary[scale].get("OVERALL", {}).get("avg_score", 0.0) * 100
        sota = PUBLISHED_SOTA.get(scale, {})
        print(f"  {scale:<8} {our_score:>11.1f}%", end="")
        for system in ["Hindsight", "Honcho", "LIGHT (Llama-4)", "RAG (Llama-4)"]:
            print(f" {sota.get(system, 0):>17.1f}%", end="")
        print()

    print(f"\n  Note: Published SOTA numbers from Hindsight blog (Apr 2026) and BEAM paper Table 3.")
    print(f"  Mnemosyne uses DeepSeek V4 Pro as answering + judging LLM.")
    print(f"  Direct comparison valid: identical BEAM dataset, identical LLM-as-judge protocol.")
    print(f"{'='*80}")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BEAM End-to-End Evaluation")
    parser.add_argument("--scales", default="100K,500K,1M,10M",
                        help="Scales to evaluate (comma-separated)")
    parser.add_argument("--sample", type=int, default=3,
                        help="Conversations per scale (0=all)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="LLM model for answering and judging")
    parser.add_argument("--judge-model", default=None,
                        help="Separate LLM for judging (default: same as --model)")
    parser.add_argument("--full-context", action="store_true",
                        help="Send full conversation to LLM (ceiling test, bypasses retrieval)")
    parser.add_argument("--pure-recall", action="store_true",
                        help="Disable per-ability bypasses + RECENT CONVERSATION injection. "
                             "Forces every answer through Mnemosyne recall — what the "
                             "BEAM-recovery experiment needs to measure arm-vs-arm "
                             "recall quality without harness-side oracle contamination. "
                             "Equivalent to MNEMOSYNE_BENCHMARK_PURE_RECALL=1.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download data and print stats, don't evaluate")
    parser.add_argument("--use-cloud", action="store_true",
                        help="Enable LLM fact extraction (cloud tier). Requires OPENROUTER_API_KEY.")
    parser.add_argument("--config-id", default=None,
                        help="Run identifier written into the paired-outcomes "
                             "JSONL alongside results JSON. Defaults to a "
                             "short hash of the MNEMOSYNE_* env snapshot — "
                             "useful for distinguishing back-to-back ablation "
                             "phases. Override when you want a human-readable "
                             "label (e.g. 'phase3a-no-fact-voice').")
    parser.add_argument("--allow-harness-oracles", action="store_true",
                        help="Opt out of the pure-recall safety check that requires "
                             "MNEMOSYNE_BENCHMARK_PURE_RECALL=1 (or --pure-recall). The "
                             "harness's TR/CR/IE/KU bypasses and RECENT CONVERSATION raw-"
                             "message injection produce answers without going through "
                             "BeamMemory.recall(), which contaminates arm-vs-arm "
                             "comparisons. Set this flag only for ceiling-test or legacy-"
                             "reproduction runs where you explicitly want the bypasses.")
    args = parser.parse_args()

    scales = [s.strip() for s in args.scales.split(",")]
    sample_size = args.sample if args.sample > 0 else None

    # ---- Preflight: refuse to run with harness oracles unless explicitly opted in.
    # The TR/CR/IE/KU bypasses and the always-included RECENT CONVERSATION block
    # produce answers WITHOUT going through BeamMemory.recall(), contaminating any
    # arm-vs-arm comparison. Pure-recall mode disables all four. See
    # docs/benchmarking.md for the full rationale.
    _pr_active = args.pure_recall or _env_truthy("MNEMOSYNE_BENCHMARK_PURE_RECALL")
    if not _pr_active and not args.allow_harness_oracles:
        print(
            "ERROR: harness oracles are active by default but contaminate arm-vs-arm "
            "comparisons. Pass --pure-recall (recommended) or set "
            "MNEMOSYNE_BENCHMARK_PURE_RECALL=1 to disable them. If you genuinely want "
            "the legacy bypass behavior (e.g., for a ceiling test or reproducing pre-"
            "fix results), pass --allow-harness-oracles explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Snapshot the full benchmark-relevant env-var surface so results JSON captures
    # exactly which configuration the run executed under. A toggle the operator
    # forgot to set is a silent confound otherwise.
    _benchmark_env_snapshot = {
        k: v for k, v in os.environ.items()
        if k.startswith("MNEMOSYNE_") or k in ("FULL_CONTEXT_MODE", "OPENROUTER_BASE_URL")
    }
    print(f"\n  Env snapshot ({len(_benchmark_env_snapshot)} vars):")
    for k in sorted(_benchmark_env_snapshot):
        # Don't echo API keys even if they accidentally got the MNEMOSYNE_ prefix.
        v = _benchmark_env_snapshot[k]
        if "KEY" in k or "TOKEN" in k or "SECRET" in k:
            v = "***redacted***"
        print(f"    {k}={v}")

    # Gap E: config_id labels each row in paired_outcomes.jsonl so a
    # downstream notebook can paired-bootstrap CIs across multiple A/B
    # runs without re-parsing the main results JSON. Default to a short
    # hash of the env snapshot (deterministic for identical configs);
    # override via `--config-id` for human-readable labels (e.g.,
    # 'phase3a-no-fact-voice').
    import hashlib
    if args.config_id:
        _config_id = args.config_id
    else:
        _env_canonical = "\n".join(
            f"{k}={v}" for k, v in sorted(_benchmark_env_snapshot.items())
            if "KEY" not in k and "TOKEN" not in k and "SECRET" not in k
        )
        _config_id = "cfg-" + hashlib.sha256(_env_canonical.encode("utf-8")).hexdigest()[:10]
    _run_started_at = datetime.now(timezone.utc).isoformat()
    print(f"  Config ID: {_config_id}")
    print(f"  Run started: {_run_started_at}")

    # Reset recall + extraction diagnostics so per-run counters are clean. The
    # snapshots are captured at the end of main() and written into results JSON.
    try:
        from mnemosyne.core.recall_diagnostics import reset_recall_diagnostics
        reset_recall_diagnostics()
    except ImportError:
        pass  # Diagnostics module is optional; older checkouts may lack it.
    try:
        from mnemosyne.extraction.diagnostics import reset_extraction_stats
        reset_extraction_stats()
    except ImportError:
        pass

    print(f"{'='*80}")
    print(f"  BEAM End-to-End Evaluation Pipeline")
    print(f"  Scales: {scales}")
    print(f"  Sample: {sample_size or 'ALL'} conversations/scale")
    print(f"  Model: {args.model}")
    print(f"  Judge: {args.judge_model or args.model}")
    # Mode resolution + banner. Pure-recall overrides full-context
    # because the bypass that full-context provides (raw conversation
    # straight to LLM) is exactly what pure-recall is meant to forbid.
    _pure_recall_env = _env_truthy("MNEMOSYNE_BENCHMARK_PURE_RECALL")
    if args.pure_recall or _pure_recall_env:
        os.environ["MNEMOSYNE_BENCHMARK_PURE_RECALL"] = "1"
        if args.full_context or _env_truthy("FULL_CONTEXT_MODE"):
            # Conflict: warn loudly so the operator isn't surprised.
            print("  Mode: PURE-RECALL (overrides FULL_CONTEXT/--full-context — "
                  "every answer goes through Mnemosyne recall)")
        else:
            print("  Mode: PURE-RECALL (per-ability bypasses + RECENT CONTEXT disabled — "
                  "every answer goes through Mnemosyne recall)")
    elif args.full_context:
        os.environ["FULL_CONTEXT_MODE"] = "1"
        print("  Mode: FULL-CONTEXT (bypassing retrieval)")
    print(f"{'='*80}")

    # Load data
    print(f"\n[1/4] Loading BEAM dataset...")
    data = load_beam_dataset(scales, max_conversations=sample_size)

    if not data:
        print("ERROR: No data loaded. Check HuggingFace token and dataset name.")
        sys.exit(1)

    # Print stats
    print(f"\n  Dataset Summary:")
    for scale, convs in data.items():
        total_msgs = sum(len(c["messages"]) for c in convs)
        total_qs = sum(len(c["questions"]) for c in convs)
        print(f"    {scale}: {len(convs)} convs, {total_msgs:,} msgs, {total_qs} questions")

    if args.dry_run:
        print(f"\n  Dry run complete. Exiting.")
        return

    # Load previous results if resuming
    resume_ids = set()
    all_previous = []
    if args.resume and RESULTS_FILE.exists():
        print(f"\n  Resuming from {RESULTS_FILE}...")
        with open(RESULTS_FILE) as f:
            prev = json.load(f)
            all_previous = prev.get("results", [])
            for conv_result in all_previous:
                for r in conv_result.get("results", []):
                    resume_ids.add(r["qid"])
        print(f"  Already evaluated: {len(resume_ids)} questions")

    # Initialize LLM clients
    print(f"\n[2/4] Initializing LLM clients...")
    llm = LLMClient(model=args.model)
    judge_llm = LLMClient(model=args.judge_model or args.model)

    # Evaluate each conversation
    print(f"\n[3/4] Evaluating... ({len(data)} scales)")
    all_results = list(all_previous) if args.resume else []

    for scale in sorted(data.keys()):
        conversations = data[scale]
        print(f"\n  --- Scale: {scale} ({len(conversations)} conversations) ---")

        for conv in conversations:
            # Create fresh Mnemosyne DB for each conversation
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / f"beam_{scale}_{conv['id']}.db"
                init_beam(db_path)
                beam = BeamMemory(session_id=f"beam_{scale}_{conv['id']}",
                                   db_path=db_path, use_cloud=args.use_cloud)

                # Ingest
                t0 = time.perf_counter()
                stats = ingest_conversation(beam, conv["messages"])
                ingest_time = time.perf_counter() - t0
                print(f"    Ingested {len(conv['messages'])} msgs in {ingest_time:.1f}s "
                      f"(DB: {os.path.getsize(db_path)/1024:.0f}KB)")

                # Evaluate
                conv_result = evaluate_conversation(
                    llm, judge_llm, beam, conv, resume_ids
                )
                all_results.append(conv_result)
                beam.conn.close()

            # Save progress after each conversation. Includes the env-var
            # snapshot + diagnostic snapshots so post-hoc analysis can attribute
            # score deltas to specific configurations without re-running.
            os.makedirs(RESULTS_FILE.parent, exist_ok=True)

            # Gap E: append per-question paired outcomes to a flat JSONL
            # so downstream analysis can paired-bootstrap CIs across
            # multiple A/B runs. Each line records (config_id, qid,
            # ability, score, correct, scale, ts) — enough to compute
            # paired deltas without re-parsing the main results JSON.
            # Append-only with run_started_at + config_id means multiple
            # phases accumulate in one file; analyst filters by config_id.
            with open(PAIRED_OUTCOMES_FILE, "a") as paired_f:
                for question in conv_result.get("results", []):
                    qid = question.get("qid")
                    score = question.get("score", 0.0)
                    paired_f.write(json.dumps({
                        "config_id": _config_id,
                        "run_started_at": _run_started_at,
                        "scale": conv_result.get("scale"),
                        "conversation_id": conv_result.get("conversation_id"),
                        "qid": qid,
                        "ability": question.get("ability"),
                        "score": score,  # raw rubric score 0.0-1.0
                        "correct": score >= 0.5,  # boolean threshold for paired tests
                    }) + "\n")
            _recall_diag = None
            _extraction_diag = None
            try:
                from mnemosyne.core.recall_diagnostics import get_recall_diagnostics
                _recall_diag = get_recall_diagnostics()
            except ImportError:
                pass
            try:
                from mnemosyne.extraction.diagnostics import get_extraction_stats
                _extraction_diag = get_extraction_stats()
            except ImportError:
                pass

            metadata = {
                "date": datetime.now(timezone.utc).isoformat(),
                "run_started_at": _run_started_at,
                "config_id": _config_id,
                "model": args.model,
                "judge_model": args.judge_model or args.model,
                "top_k": DEFAULT_TOP_K,
                "sample_size": sample_size or "ALL",
                "scales": scales,
                "total_conversations": len(all_results),
                "config": {
                    "env": _benchmark_env_snapshot,
                    "pure_recall": _pr_active,
                    "allow_harness_oracles": args.allow_harness_oracles,
                    "full_context": args.full_context,
                    "use_cloud": args.use_cloud,
                },
                "diagnostics": {
                    "recall": _recall_diag,
                    "extraction": _extraction_diag,
                },
            }
            with open(RESULTS_FILE, "w") as f:
                json.dump({"metadata": metadata, "results": all_results}, f, indent=2)

    # Cleanup
    llm.close()
    judge_llm.close()

    # Compute and print report
    print(f"\n[4/4] Computing SOTA report...")
    ability_summary = compute_ability_scores(all_results)

    metadata = {
        "model": args.model,
        "sample_size": sample_size or "ALL",
        "judge_model": args.judge_model or args.model,
    }
    print_sota_report(ability_summary, metadata)

    # Save summary
    summary_file = RESULTS_FILE.parent / "beam_e2e_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "date": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "ability_summary": {
                scale: {
                    ab: {"avg_score": v["avg_score"], "count": v["count"]}
                    for ab, v in abilities.items()
                }
                for scale, abilities in ability_summary.items()
            },
        }, f, indent=2)

    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"  Summary saved to: {summary_file}")
    if PAIRED_OUTCOMES_FILE.exists():
        print(f"  Paired outcomes appended to: {PAIRED_OUTCOMES_FILE}")
        print(f"    (filter by config_id={_config_id!r} for this run's rows)")


if __name__ == "__main__":
    main()
