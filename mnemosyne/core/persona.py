"""
L3 Persona extractor + file generator (v3.10.0).

Rule-based extraction from already-classified memories. Zero LLM calls in
the default path. Optional LLM consolidation is a separate concern.

Trigger conditions (matches Hy-Memory PersonaTrigger pattern):
1. Explicit request (request_persona_update flag)
2. Cold start (no persona.md exists yet)
3. Recovery (persona.md body is empty/missing)
4. Threshold reached (memories_since_last_persona >= interval)
5. Daily sync (default 03:00 UTC, opt-in)
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Extraction rules -- NO LLM calls in default path.
PROMOTION_SOURCES = frozenset({"preference", "persona", "stated"})
MIN_IMPORTANCE_FOR_PROMOTION = 0.7

# Default trigger interval for threshold-based regeneration.
DEFAULT_INTERVAL = int(os.environ.get("MNEMOSYNE_PERSONA_INTERVAL", "50"))
# Daily sync hour (UTC). 0-23. -1 to disable.
DEFAULT_DAILY_SYNC_HOUR = int(os.environ.get("MNEMOSYNE_PERSONA_DAILY_SYNC_HOUR", "3"))
# Token cap for persona.md.
DEFAULT_TOKEN_CAP = int(os.environ.get("MNEMOSYNE_PERSONA_TOKEN_CAP", "1500"))

DEFAULT_PERSONA_FILE = Path(
    os.environ.get(
        "MNEMOSYNE_PERSONA_FILE",
        str(Path.home() / ".hermes" / "memory" / "persona.md"),
    )
)


class PersonaExtractor:
    """Rule-based persona candidate extractor.

    Reads from working_memory and episodic_memory, applies deterministic
    rules to filter candidates, deduplicates by topic, and returns a
    normalized list ready for either direct insertion into memoria_persona
    or for rendering to persona.md.
    """

    def __init__(
        self,
        beam,
        sources: Optional[frozenset] = None,
        min_importance: float = MIN_IMPORTANCE_FOR_PROMOTION,
    ):
        self._beam = beam
        self._sources = sources or PROMOTION_SOURCES
        self._min_importance = min_importance

    def extract_candidates(
        self,
        session_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Pull preference/persona/stated memories from WM + episodic.

        Returns normalized list of dicts with keys:
          content, topic, importance, source, source_memory_id
        """
        results: List[Dict[str, Any]] = []
        # Working memory -- no 'topic' column, source_memory_id maps to id.
        sql = (
            "SELECT id, content, source, importance "
            "FROM working_memory "
            "WHERE source IN ({}) AND importance >= ?"
        ).format(",".join("?" * len(self._sources)))
        params: List[Any] = list(self._sources) + [self._min_importance]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(limit)
        rows = self._beam.conn.execute(sql, params).fetchall()
        for r in rows:
            results.append(self._normalize_wm(r, tier="working"))
        # Episodic memory (same shape)
        sql2 = (
            "SELECT id, content, source, importance "
            "FROM episodic_memory "
            "WHERE source IN ({}) AND importance >= ?"
        ).format(",".join("?" * len(self._sources)))
        params2: List[Any] = list(self._sources) + [self._min_importance]
        if session_id is not None:
            sql2 += " AND session_id = ?"
            params2.append(session_id)
        sql2 += " ORDER BY importance DESC LIMIT ?"
        params2.append(limit)
        rows2 = self._beam.conn.execute(sql2, params2).fetchall()
        for r in rows2:
            # Skip if already represented by WM (same content)
            if any(c["content"] == self._safe_content(r) for c in results):
                continue
            results.append(self._normalize_em(r, tier="long_term"))
        return results

    def _safe_content(self, row) -> str:
        return (row[1] or "").strip()

    def _normalize_wm(self, row, tier: str) -> Dict[str, Any]:
        # WM rows don't carry topic; derive from memoria_timelines if present,
        # else "general".
        topic = self._derive_topic(row[0])
        return {
            "content": (row[1] or "").strip(),
            "source": row[2] or "preference",
            "importance": float(row[3] or 0.5),
            "topic": topic,
            "source_memory_id": row[0],
            "tier": tier,
        }

    def _normalize_em(self, row, tier: str) -> Dict[str, Any]:
        topic = self._derive_topic(row[0])
        return {
            "content": (row[1] or "").strip(),
            "source": row[2] or "preference",
            "importance": float(row[3] or 0.5),
            "topic": topic,
            "source_memory_id": row[0],
            "tier": tier,
        }

    def _derive_topic(self, memory_id: str) -> str:
        """Best-effort topic lookup from memoria_timelines; fall back to general."""
        try:
            r = self._beam.conn.execute(
                "SELECT topic FROM memoria_timelines WHERE source_memory_id = ? LIMIT 1",
                (memory_id,),
            ).fetchone()
            if r is not None and r[0]:
                return str(r[0]).strip() or "general"
        except Exception:
            pass
        return "general"

    def deduplicate_by_topic(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep the highest-importance candidate per topic.

        Deterministic: same input -> same output.
        """
        by_topic: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            t = c["topic"]
            existing = by_topic.get(t)
            if existing is None or c["importance"] > existing["importance"]:
                by_topic[t] = c
        # Sort by importance DESC for stable output
        return sorted(by_topic.values(), key=lambda x: (-x["importance"], x["topic"]))

    def filter_already_promoted(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Drop candidates that already exist in memoria_persona.

        Match by source_memory_id when available, fall back to content.
        """
        existing_ids = {
            row[0]
            for row in self._beam.conn.execute(
                "SELECT source_memory_id FROM memoria_persona WHERE source_memory_id IS NOT NULL"
            ).fetchall()
            if row[0]
        }
        existing_content = {
            row[0].strip()
            for row in self._beam.conn.execute(
                "SELECT content FROM memoria_persona"
            ).fetchall()
            if row[0]
        }
        out = []
        for c in candidates:
            if c["source_memory_id"] in existing_ids:
                continue
            if c["content"] in existing_content:
                continue
            out.append(c)
        return out


class PersonaTriggers:
    """5 trigger conditions (matches Hy-Memory PersonaTrigger)."""

    def __init__(
        self,
        beam,
        persona_file: Path = DEFAULT_PERSONA_FILE,
        interval: int = DEFAULT_INTERVAL,
        daily_sync_hour: int = DEFAULT_DAILY_SYNC_HOUR,
    ):
        self._beam = beam
        self._persona_file = Path(persona_file)
        self._interval = interval
        self._daily_sync_hour = daily_sync_hour

    def should_generate(self, explicit_request: bool = False) -> Dict[str, Any]:
        """Return {should: bool, reason: str}."""
        if explicit_request:
            return {"should": True, "reason": "explicit request"}
        if self._cold_start():
            return {"should": True, "reason": "cold start (no persona.md yet)"}
        if self._recovery_needed():
            return {"should": True, "reason": "recovery (persona.md body missing/empty)"}
        if self._threshold_reached():
            return {"should": True, "reason": f"threshold reached (>{self._interval} new memories)"}
        if self._daily_sync_window():
            return {"should": True, "reason": f"daily sync window (hour={self._daily_sync_hour} UTC)"}
        return {"should": False, "reason": "no trigger condition met"}

    def _cold_start(self) -> bool:
        return not self._persona_file.exists()

    def _recovery_needed(self) -> bool:
        if not self._persona_file.exists():
            return False  # handled by cold_start
        try:
            content = self._persona_file.read_text().strip()
        except Exception:
            return True
        return len(content) == 0

    def _threshold_reached(self) -> bool:
        """Count memories created since last persona regeneration."""
        # Use last_reinforced_at on memoria_persona as the watermark. Skip
        # created_at to avoid mixing semantics (created_at = row insert time,
        # last_reinforced_at = actual rule usage time -- only the latter is a
        # meaningful "since last regeneration" marker).
        last = self._beam.conn.execute(
            "SELECT MAX(last_reinforced_at) FROM memoria_persona"
        ).fetchone()
        last_dt = last[0] if last and last[0] else None
        if last_dt is None:
            return False
        # Count new WM memories since last_dt
        try:
            row = self._beam.conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE created_at > ?",
                (str(last_dt),),
            ).fetchone()
        except Exception:
            return False
        return (row[0] if row else 0) >= self._interval

    def _daily_sync_window(self) -> bool:
        if self._daily_sync_hour < 0 or self._daily_sync_hour > 23:
            return False
        return datetime.utcnow().hour == self._daily_sync_hour


def render_persona_markdown(
    candidates: List[Dict[str, Any]],
    token_cap: int = DEFAULT_TOKEN_CAP,
) -> str:
    """Render candidate list to Markdown. Sort: tier, importance, reinforcement.

    Token cap is approximate (counts words * 1.3 ~= tokens, conservative).
    """
    if not candidates:
        return ""
    # Group by topic
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_topic.setdefault(c["topic"], []).append(c)
    lines = ["# Persona", ""]
    # Topic sections sorted alphabetically for stable output
    for topic in sorted(by_topic.keys()):
        items = by_topic[topic]
        # Sort items within topic by importance DESC
        items.sort(key=lambda x: -x["importance"])
        lines.append(f"## {topic}")
        lines.append("")
        for item in items:
            imp = item.get("importance", 0.5)
            content = item["content"]
            tag = ""
            if item.get("tier") == "permanent":
                tag = " (permanent)"
            elif item.get("tier") == "long_term":
                tag = " (long-term)"
            lines.append(f"- {content} [importance: {imp:.2f}]{tag}")
        lines.append("")
    out = "\n".join(lines).rstrip() + "\n"
    # Token cap (approximate)
    approx_tokens = int(len(out.split()) * 1.3)
    if approx_tokens > token_cap:
        # Truncate by topic sections until under cap
        truncated = lines[:1] + [""]
        approx = int(len(" ".join(truncated).split()) * 1.3)
        # Iterate topics in priority order (first added = highest importance)
        topic_pri = []
        seen = set()
        for c in candidates:
            if c["topic"] in seen:
                continue
            seen.add(c["topic"])
            topic_pri.append((c["topic"], max(
                -x["importance"] for x in candidates if x["topic"] == c["topic"]
            )))
        topic_pri.sort(key=lambda x: x[1])
        for topic, _ in topic_pri:
            section = [f"## {topic}", ""]
            items = [c for c in candidates if c["topic"] == topic]
            items.sort(key=lambda x: -x["importance"])
            for item in items:
                imp = item.get("importance", 0.5)
                content = item["content"]
                section.append(f"- {content} [importance: {imp:.2f}]")
            section_text = "\n".join(section)
            new_approx = approx + int(len(section_text.split()) * 1.3)
            if new_approx > token_cap:
                break
            truncated.append(section_text)
            truncated.append("")
            approx = new_approx
        out = "\n".join(truncated).rstrip() + "\n"
    return out


def write_persona_file(
    candidates: List[Dict[str, Any]],
    persona_file: Path = DEFAULT_PERSONA_FILE,
    token_cap: int = DEFAULT_TOKEN_CAP,
) -> Path:
    """Atomic write persona.md. Returns the path."""
    persona_file = Path(persona_file)
    persona_file.parent.mkdir(parents=True, exist_ok=True)
    content = render_persona_markdown(candidates, token_cap=token_cap)
    # Atomic write: tmp + rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(persona_file.parent), prefix=".persona.md.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, persona_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
    return persona_file


def read_persona_file(persona_file: Path = DEFAULT_PERSONA_FILE) -> str:
    """Read current persona.md content. Empty string if file missing."""
    p = Path(persona_file)
    if not p.exists():
        return ""
    try:
        return p.read_text()
    except Exception:
        return ""
