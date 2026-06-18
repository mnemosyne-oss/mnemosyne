"""
Persona adapter for the hermes_memory_provider package (v3.10.0).

Exposes the L3 persona tier as 4 tools:
- mnemosyne_persona_promote(memory_id, tier, reason)
- mnemosyne_persona_demote(persona_id, reason)
- mnemosyne_persona_list(tier, topic)
- mnemosyne_persona_reinforce(persona_id)

All operations hit the memoria_persona table directly. No LLM dependency
in the default path (rule-based extraction is a separate concern).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


VALID_TIERS = ("permanent", "long_term", "working")


class PersonaAdapter:
    """Standalone persona adapter wrapping memoria_persona CRUD."""

    def __init__(self, beam_instance=None, config: Optional[Dict[str, Any]] = None):
        self._beam = beam_instance
        self._config = config or {}
        self._local_beam = None
        if self._beam is None:
            try:
                from mnemosyne.core.beam import BeamMemory
                self._local_beam = BeamMemory(session_id="persona-adapter")
            except Exception as exc:
                logger.debug("PersonaAdapter could not lazy-init BeamMemory: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._beam is not None or self._local_beam is not None

    def _conn(self):
        return (self._beam or self._local_beam).conn

    def handle_tool_call(self, tool_name: str, args: dict) -> str:
        if not self.is_ready:
            return json.dumps({
                "status": "error",
                "error": "PersonaAdapter not initialized (no beam instance).",
            })
        try:
            if tool_name == "mnemosyne_persona_promote":
                return self._promote(**args)
            elif tool_name == "mnemosyne_persona_demote":
                return self._demote(**args)
            elif tool_name == "mnemosyne_persona_list":
                return self._list(**args)
            elif tool_name == "mnemosyne_persona_reinforce":
                return self._reinforce(**args)
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc)})
        return json.dumps({"status": "error", "error": f"Unknown tool: {tool_name}"})

    def _promote(self, memory_id: str = "", tier: str = "long_term",
                 reason: str = "") -> str:
        if tier not in VALID_TIERS:
            return json.dumps({
                "status": "error",
                "error": f"Invalid tier {tier!r}. Must be one of: {VALID_TIERS}",
            })
        # Look up the source memory to extract content. Working memory has no
        # 'topic' column -- topic is derived from memoria_timelines where
        # available, otherwise we fall back to a generic label.
        content = ""
        topic = ""
        try:
            row = self._conn().execute(
                "SELECT content FROM working_memory WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                row = self._conn().execute(
                    "SELECT content FROM episodic_memory WHERE id = ?",
                    (memory_id,),
                ).fetchone()
            if row is not None:
                content = row[0] or ""
            # Try to derive a topic from memoria_timelines (best effort).
            tl = self._conn().execute(
                "SELECT topic FROM memoria_timelines WHERE source_memory_id = ? LIMIT 1",
                (memory_id,),
            ).fetchone()
            if tl is not None:
                topic = tl[0] or ""
        except Exception:
            pass
        if not content:
            return json.dumps({
                "status": "error",
                "error": f"Source memory {memory_id!r} not found.",
            })
        if not topic:
            topic = "general"
        cur = self._conn().execute(
            "INSERT INTO memoria_persona (tier, topic, content, source_memory_id, promotion_reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (tier, topic, content, memory_id, reason),
        )
        persona_id = cur.lastrowid
        return json.dumps({
            "status": "ok",
            "persona_id": persona_id,
            "tier": tier,
            "topic": topic,
        })

    def _demote(self, persona_id: int = 0, reason: str = "") -> str:
        # Demotion = delete from persona tier (caller decides what to do next).
        # We keep a record by writing into memoria_preferences as a tombstone.
        cur = self._conn().execute(
            "SELECT topic, content, tier FROM memoria_persona WHERE id = ?",
            (persona_id,),
        ).fetchone()
        if cur is None:
            return json.dumps({
                "status": "error",
                "error": f"Persona id {persona_id} not found.",
            })
        topic, content, tier = cur
        self._conn().execute(
            "INSERT INTO memoria_preferences (preference, topic, evolution, context_snippet, source_memory_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                f"[demoted from {tier}] {content}",
                topic,
                f"demoted: {reason}" if reason else "demoted",
                "",
                f"persona:{persona_id}",
            ),
        )
        self._conn().execute(
            "DELETE FROM memoria_persona WHERE id = ?",
            (persona_id,),
        )
        return json.dumps({
            "status": "ok",
            "persona_id": persona_id,
            "demoted_to": "memoria_preferences",
        })

    def _list(self, tier: Optional[str] = None, topic: Optional[str] = None) -> str:
        sql = (
            "SELECT id, tier, topic, content, confidence, reinforcement_count, last_reinforced_at "
            "FROM memoria_persona"
        )
        clauses = []
        params: List[Any] = []
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if topic is not None:
            clauses.append("topic = ?")
            params.append(topic)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Custom tier ordering: permanent > long_term > working. Alphabetical
        # sort would put 'long_term' before 'permanent', which is wrong for
        # always-on injection priority.
        sql += (
            " ORDER BY CASE tier "
            "WHEN 'permanent' THEN 0 "
            "WHEN 'long_term' THEN 1 "
            "WHEN 'working' THEN 2 "
            "ELSE 3 END, "
            "reinforcement_count DESC, id ASC"
        )
        rows = self._conn().execute(sql, params).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "tier": r[1],
                "topic": r[2],
                "content": r[3],
                "confidence": r[4],
                "reinforcement_count": r[5],
                "last_reinforced_at": r[6],
            })
        return json.dumps({"status": "ok", "count": len(out), "personas": out})

    def _reinforce(self, persona_id: int = 0) -> str:
        cur = self._conn().execute(
            "UPDATE memoria_persona SET reinforcement_count = reinforcement_count + 1, "
            "last_reinforced_at = CURRENT_TIMESTAMP WHERE id = ?",
            (persona_id,),
        )
        if cur.rowcount == 0:
            return json.dumps({
                "status": "error",
                "error": f"Persona id {persona_id} not found.",
            })
        return json.dumps({
            "status": "ok",
            "persona_id": persona_id,
            "reinforced": True,
        })
