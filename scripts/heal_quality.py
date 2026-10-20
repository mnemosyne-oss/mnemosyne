#!/usr/bin/env python3
"""
Self-Healing Quality Pipeline for Mnemosyne Episodic Memory
============================================================

Detects degraded entries (bullet-format, <300 chars) and repairs them
via a 4-stage LLM-as-Judge loop: Extract → Generate → Judge → Repair.

Designed for upstream inclusion: model-agnostic, no mnemosyne internals
required beyond the existing BeamMemory API.

Usage:
    python scripts/heal_quality.py [--detect-only] [--entry-id ID] [--dry-run]
    hermes mnemosyne heal-quality [--detect-only] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sqlite3

# --- Config knobs (also overrideable via env) ---
JUDGE_THRESHOLD = int(os.environ.get("MNEMOSYNE_HEAL_JUDGE_THRESHOLD", "75"))
MAX_RETRIES = int(os.environ.get("MNEMOSYNE_HEAL_MAX_RETRIES", "3"))
MIN_SUMMARY_LEN = int(os.environ.get("MNEMOSYNE_HEAL_MIN_LEN", "300"))
MEMORY_UNIT_BUDGET = int(os.environ.get("MNEMOSYNE_HEAL_BUDGET", "4000"))
FORCE_M2_AFTER_RETRIES = int(os.environ.get("MNEMOSYNE_HEAL_ESCALATE_AFTER", "2"))
JUDGE_MODEL = os.environ.get("MNEMOSYNE_HEAL_JUDGE_MODEL", "MiniMax-M3.1-Flash")

# --- LLM backends ------------------------------------------------------------

def _call_mmx(prompt: str) -> str | None:
    """Call the configured MiniMax model via mmx-cli; return text or None."""
    import subprocess

    mmx_path = Path.home() / ".npm-global" / "bin" / "mmx"
    if not mmx_path.exists():
        return None

    try:
        result = subprocess.run(
            [str(mmx_path), "text", "chat", "--model", JUDGE_MODEL, "--message", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        # mmx returns {content: [{text: "..."}]}
        content = data.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text")
        return None
    except Exception:
        return None


def _call_mnemosyne_llm(memories: list[str], source: str) -> str | None:
    """Call mnemosyne's own summarization (local GGUF or configured remote)."""
    try:
        from mnemosyne.core.local_llm import summarize_memories as mnemo_summarize
        return mnemo_summarize(memories, source=source)
    except Exception:
        return None


# --- Core pipeline ------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """~4 chars per token for English."""
    return max(1, len(text) // 4)


def extract_memory_units(session_path: Path, budget: int = MEMORY_UNIT_BUDGET) -> list[str]:
    """Parse session JSON and split into token-budgeted memory units."""
    try:
        data = json.loads(session_path.read_text())
    except Exception:
        return []

    messages = data.get("messages", [])
    units, current = [], []
    tokens = 0

    for msg in messages[-80:]:  # last 80 messages
        role = msg.get("role", "")

        # Skip tool messages — their raw output pollutes summarization context.
        # Only user and assistant turns carry conversational signal.
        if role == "tool":
            continue

        content = str(msg.get("content", "")).strip()
        if not content:
            continue

        t = _estimate_tokens(content)
        if tokens + t > budget and current:
            units.append("\n".join(current))
            current, tokens = [], 0
        current.append(f"[{role.upper()}] {content}")
        tokens += t

    if current:
        units.append("\n".join(current))
    return units


def judge_summary(summary: str, memory_units: list[str], source: str) -> dict[str, Any]:
    """LLM-as-Judge: score summary across 4 dimensions, return structured verdict."""
    if not summary or not memory_units:
        return {
            "factual_density": 0, "format_compliance": 0,
            "length_sufficiency": 0, "grounding": 0,
            "fault": "none", "diagnosis": "Empty summary or no source memories.",
            "retry_needed": False,
        }

    prompt = f'''You are a memory quality auditor. Evaluate this summary against the source memories.

SCORING DIMENSIONS (score each 0-100):
1. FACTUAL_DENSITY — Does it contain specific facts (names, paths, versions, numbers)?
2. FORMAT_COMPLIANCE — Is it plain prose (no bullets, no dashes)?
3. LENGTH_SUFFICIENCY — Is it detailed enough (>300 chars for complex sessions)?
4. GROUNDING — Does it match what actually happened in the memories?

SUMMARY TO AUDIT:
{summary}

MEMORY UNITS (ground truth, first 5):
{chr(10).join(memory_units[:5])}

Respond with JSON only:
{{"factual_density": N, "format_compliance": N, "length_sufficiency": N, "grounding": N, "fault": "none|truncated|generic|missing_facts|wrong_format", "diagnosis": "one-sentence explanation", "retry_needed": true|false}}'''

    text = _call_mmx(prompt)
    if not text:
        # Fallback: heuristic scoring when judge call fails
        score = 100 if len(summary) >= MIN_SUMMARY_LEN else max(0, len(summary) // 3)
        return {
            "factual_density": score, "format_compliance": score,
            "length_sufficiency": score if len(summary) >= MIN_SUMMARY_LEN else 0,
            "grounding": score,
            "fault": "none", "diagnosis": "Judge call failed, used heuristic fallback.",
            "retry_needed": False,
        }

    try:
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {
            "factual_density": 50, "format_compliance": 50,
            "length_sufficiency": 50, "grounding": 50,
            "fault": "none", "diagnosis": "Judge JSON parse failed.",
            "retry_needed": False,
        }


def generate_summary(memory_units: list[str], source: str, *, force_m2: bool = False) -> str | None:
    """Generate a new summary, using M2.7 when forced or local LLM otherwise."""
    if not memory_units:
        return None

    joined = "\n---\n".join(memory_units)
    prompt = f"""Summarize into 1-3 plain prose sentences.
Preserve: names, file paths, tool names, project names, numbers, versions, decisions.
Discard: fluff, filler, meta-commentary.
Source: {source}

---MEMORY UNITS---
{joined}
---END---

Summary:"""

    # Prefer M2.7 when forced; fall back to local GGUF if mmx is unavailable
    if force_m2:
        result = _call_mmx(prompt)
        if result:
            return result
        # mmx unavailable — fall through to local LLM

    result = _call_mnemosyne_llm(memory_units, source)
    if result and len(result) >= MIN_SUMMARY_LEN:
        return result

    # Escalate to M2.7 if local model produced thin output
    result = _call_mmx(prompt)
    return result if result else None


def repair_summary(summary: str, memory_units: list[str], fault: str, source: str) -> str | None:
    """Apply fault-specific repair strategy."""
    if fault == "truncated":
        # Context was cut — retry with doubled context
        return generate_summary(memory_units * 2, source, force_m2=True)
    if fault == "generic":
        prompt = f"""Rewrite this summary to be SPECIFIC. Every noun must be named.
File paths must be real paths. Tools must be named. Numbers must be included.

Current summary: {summary}

Source context (first unit):
{memory_units[0][:1000] if memory_units else ''}

Specific rewrite:"""
        return _call_mmx(prompt)
    if fault == "missing_facts":
        # Extract and re-inject key facts
        facts = []
        for unit in memory_units:
            facts.extend(_extract_facts_from_text(unit))
        if facts:
            prompt = f"""Augment this summary by incorporating these specific facts:
{', '.join(facts[:20])}

Current: {summary}

Augmented (preserve prose, add facts inline):"""
            return _call_mmx(prompt)
        return generate_summary(memory_units, source, force_m2=True)
    if fault == "wrong_format":
        prompt = f"""Rewrite in PLAIN PROSE ONLY. No bullets, no dashes, no lists.
Convert: {summary}
Plain prose:"""
        return _call_mmx(prompt)
    # Unknown fault — full regenerate with M2.7
    return generate_summary(memory_units, source, force_m2=True)


def _extract_facts_from_text(text: str) -> list[str]:
    """Simple heuristic fact extractor: paths, URLs, version strings, numbers with context."""
    import re
    facts = []
    # File paths
    paths = re.findall(r'(?:/[\w\-./]+(?:\.\w+)?|\b[\w\-]+\/[\w\-./]+(?:\.\w+)?)', text)
    facts.extend(paths[:5])
    # Version strings
    versions = re.findall(r'\b\d+\.\d+(?:\.\d+)?\b', text)
    facts.extend([f"v{v}" for v in versions[:5]])
    # URLs
    urls = re.findall(r'https?://\S+', text)
    facts.extend(urls[:3])
    return list(dict.fromkeys(facts))  # deduplicate, preserve order


def should_retry(retry_count: int, verdict: dict, max_retries: int = MAX_RETRIES) -> bool:
    """Decide whether to retry the repair loop."""
    if retry_count >= max_retries:
        return False
    if verdict.get("fault") == "none":
        return False
    if retry_count >= FORCE_M2_AFTER_RETRIES:
        verdict["fault"] = "escalate"  # Forces M2.7 on next generate
    return True


# --- Database ----------------------------------------------------------------

def get_db_path() -> Path:
    default = Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
    return Path(os.environ.get("MNEMOSYNE_DB_PATH", default))


def get_session_path(session_id: str) -> Path | None:
    """Map mnemosyne session_id (hermes_<ts>_<hash>) to session JSON file."""
    sessions_dir = Path.home() / ".hermes" / "sessions"
    # session_id format: hermes_20260505_165757_<hash>
    # file format: session_20260505_165757_<hash>.json
    if session_id.startswith("hermes_"):
        ts_part = "_".join(session_id.split("_")[1:4])  # 20260505_165757
        remainder = "_".join(session_id.split("_")[4:])
        filename = f"session_{ts_part}_{remainder}.json"
        path = sessions_dir / filename
        if path.exists():
            return path
    # Fallback: search
    for p in sessions_dir.glob("session_*.json"):
        if session_id.replace("hermes_", "") in p.stem:
            return p
    return None


def detect_degraded_entries(conn: sqlite3.Connection) -> list[tuple]:
    """Return list of (id, content, session_id, metadata_json, len) for degraded entries."""
    cursor = conn.execute("""
        SELECT id, content, session_id, metadata_json, LENGTH(content) as len
        FROM episodic_memory
        WHERE (content LIKE '- %' AND LENGTH(content) < 150)
           OR (LENGTH(content) < 100 AND content NOT LIKE '% % %')
        ORDER BY len ASC
    """)
    return cursor.fetchall()


def update_entry(
    conn: sqlite3.Connection,
    entry_id: str,
    new_content: str,
    quality_score: float,
    verdict: dict,
    retry_count: int,
    degraded_at_was: str | None,
):
    """Update episodic_memory row with repaired content and quality metadata."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "quality_score": quality_score,
        "judge_model": JUDGE_MODEL,
        "consolidated_at": now,
        "fault_before_repair": verdict.get("fault"),
        "retry_loop_count": retry_count,
        "needs_human_review": quality_score < 60,
    }
    if degraded_at_was:
        meta["degraded_at"] = degraded_at_was

    conn.execute("""
        UPDATE episodic_memory
        SET content = ?,
            metadata_json = ?,
            degraded_at = NULL
        WHERE id = ?
    """, (new_content, json.dumps(meta), entry_id))
    conn.commit()


def get_episodic_stats(conn: sqlite3.Connection) -> dict:
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN LENGTH(content) > 300 THEN 1 ELSE 0 END) as high_quality,
            SUM(CASE WHEN (content LIKE '- %' AND LENGTH(content) < 150)
                  OR (LENGTH(content) < 100 AND content NOT LIKE '% % %')
                  THEN 1 ELSE 0 END) as degraded,
            SUM(CASE WHEN degraded_at IS NOT NULL THEN 1 ELSE 0 END) as marked_degraded
        FROM episodic_memory
    """)
    row = cursor.fetchone()
    return {
        "total": row[0] or 0,
        "high_quality": row[1] or 0,
        "degraded": row[2] or 0,
        "marked_degraded": row[3] or 0,
    }


# --- Main pipeline orchestrator ----------------------------------------------

def heal_entry(
    entry_id: str,
    conn: sqlite3.Connection,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full 4-stage pipeline on one entry. Returns result dict."""
    cursor = conn.execute(
        "SELECT id, content, session_id, source, metadata_json, degraded_at FROM episodic_memory WHERE id = ?",
        (entry_id,)
    )
    row = cursor.fetchone()
    if not row:
        return {"entry_id": entry_id, "status": "not_found"}

    eid, old_content, session_id, source, meta_json, degraded_at = row
    source = source or f"session:{session_id}"
    _meta = json.loads(meta_json) if meta_json else {}

    # Stage 0: Extract memory units from session file
    session_path = get_session_path(session_id)
    if session_path:
        memory_units = extract_memory_units(session_path)
    else:
        # No session file — use the existing content as the memory unit
        memory_units = [old_content]

    # Stage 1: Generate initial summary (M2.7 for repair pipeline)
    retry_count = 0
    verdict = {"fault": "none"}

    summary = generate_summary(memory_units, source, force_m2=True)
    if not summary:
        return {"entry_id": entry_id, "status": "generate_failed", "fault": "none"}

    # Stage 2: Judge
    verdict = judge_summary(summary, memory_units, source)
    score = (verdict["factual_density"] + verdict["format_compliance"] +
             verdict["length_sufficiency"] + verdict["grounding"]) / 4

    # Closed loop: repair if needed
    while should_retry(retry_count, verdict):
        retry_count += 1
        repaired = repair_summary(summary, memory_units, verdict.get("fault", ""), source)
        if not repaired:
            break
        summary = repaired
        verdict = judge_summary(summary, memory_units, source)
        score = (verdict["factual_density"] + verdict["format_compliance"] +
                 verdict["length_sufficiency"] + verdict["grounding"]) / 4
        if verdict.get("fault") == "none":
            break

    if dry_run:
        return {
            "entry_id": entry_id,
            "status": "dry_run",
            "old_len": len(old_content),
            "new_len": len(summary),
            "quality_score": score,
            "verdict": verdict,
            "retry_count": retry_count,
        }

    update_entry(conn, eid, summary, score, verdict, retry_count, degraded_at)
    return {
        "entry_id": entry_id,
        "status": "repaired",
        "old_len": len(old_content),
        "new_len": len(summary),
        "quality_score": score,
        "verdict": verdict,
        "retry_count": retry_count,
    }


def run_heal_pipeline(detect_only: bool = False, entry_id: str | None = None, dry_run: bool = False):
    """Main entry point."""
    db_path = get_db_path()
    if not db_path.exists():
        print(f"Error: database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    if entry_id:
        result = heal_entry(entry_id, conn, dry_run=dry_run)
        print(json.dumps(result, indent=2))
        conn.close()
        return

    # Detect degraded
    degraded = detect_degraded_entries(conn)
    stats = get_episodic_stats(conn)
    print(json.dumps({"stats": stats, "degraded_count": len(degraded), "entries": [
        {"id": e[0], "len": e[4], "session_id": e[2]} for e in degraded
    ]}, indent=2))

    if detect_only or dry_run:
        conn.close()
        return

    # Heal all degraded
    repaired = []
    for e in degraded:
        eid = e[0]
        result = heal_entry(eid, conn, dry_run=False)
        repaired.append(result)
        print(f"  {'✓' if result['status'] == 'repaired' else '✗'} {eid}: {result.get('status')}")

    final_stats = get_episodic_stats(conn)
    print(json.dumps({"final_stats": final_stats, "repaired": repaired}, indent=2))
    conn.close()


# --- CLI -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mnemosyne self-healing quality pipeline")
    parser.add_argument("--detect-only", action="store_true", help="Only detect degraded entries")
    parser.add_argument("--entry-id", type=str, help="Heal a specific entry by ID")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    args = parser.parse_args()

    run_heal_pipeline(
        detect_only=args.detect_only,
        entry_id=args.entry_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()