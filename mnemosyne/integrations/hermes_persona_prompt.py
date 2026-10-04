"""Shared Hermes persona prompt helpers.

Both Hermes provider surfaces (the root plugin package and the packaged
``mnemosyne_hermes`` integration) use this module so persona prompt injection
stays behaviorally identical.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PERSONA_ENABLED_ENV = "MNEMOSYNE_PERSONA_ENABLED"
PERSONA_FILE_ENV = "MNEMOSYNE_PERSONA_FILE"
PERSONA_TOKEN_CAP_ENV = "MNEMOSYNE_PERSONA_TOKEN_CAP"
PERSONA_PROMPT_HEADER = "# L3 Persona (Active Behavioral Rules)"
DEFAULT_PERSONA_TOKEN_CAP = 1500
DEFAULT_PERSONA_FILE = Path.home() / ".hermes" / "memory" / "persona.md"


def _parse_env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    raw = value.strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _parse_env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", key, value, default)
        return default


def _persona_file_from_env() -> Path:
    return Path(os.environ.get(PERSONA_FILE_ENV, str(DEFAULT_PERSONA_FILE)))


class HermesPersonaPromptMixin:
    """Mixin providing opt-in persona prompt injection for Hermes providers."""

    # Keep the existing opt-in contract; promotion alone must not enable IO.
    PERSONA_ENABLED = _parse_env_bool(PERSONA_ENABLED_ENV, False)
    PERSONA_FILE = _persona_file_from_env()
    PERSONA_TOKEN_CAP = _parse_env_int(PERSONA_TOKEN_CAP_ENV, DEFAULT_PERSONA_TOKEN_CAP)
    _persona_cache: Dict[str, Any] = {"mtime": None, "content": None}

    def _persona_from_db(self) -> str:
        """Read at most 20 rows from the provider's call-time profile-bound Beam.

        Use complete entries only. The byte budget includes the prompt header;
        one UTF-8 byte per configured token is intentionally conservative and
        also bounds CJK/unbroken strings, unlike a word-count approximation.
        No model tokenizer is available at this provider seam.
        """
        beam = getattr(self, "_beam", None)
        conn = getattr(beam, "conn", None) if beam is not None else None
        if conn is None:
            return ""
        budget = max(0, int(self.PERSONA_TOKEN_CAP))
        used = len((PERSONA_PROMPT_HEADER + "\n").encode("utf-8"))
        if budget <= used:
            return ""
        try:
            # Never open a global/default store: the provider selects the DB.
            # Bound SQL work and avoid materializing oversized entry contents.
            rows = conn.execute(
                "SELECT tier, topic, content FROM memoria_persona "
                "WHERE length(CAST(content AS BLOB)) <= ? "
                "AND tier IN ('permanent', 'long_term', 'working') "
                "AND trim(content) <> '' "
                "ORDER BY CASE tier WHEN 'permanent' THEN 0 "
                "WHEN 'long_term' THEN 1 ELSE 2 END, "
                "reinforcement_count DESC, id ASC LIMIT 20",
                (budget,),
            ).fetchall()
            lines = []
            for tier, topic, content in rows:
                text = " ".join(str(content).split())
                label = " ".join(str(topic or "general").split())
                line = f"- [{tier}] {label}: {text}"
                size = len((line + "\n").encode("utf-8"))
                if used + size > budget:
                    continue
                lines.append(line)
                used += size
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("persona_from_db read failed: %s", exc)
            return ""

    def _persona_from_file(self) -> str:
        """Read persona.md if file exists. Cached by mtime."""
        try:
            persona_file = Path(self.PERSONA_FILE)
            if not persona_file.exists():
                self._persona_cache = {"mtime": None, "content": None}
                return ""
            stat = persona_file.stat()
            mtime = (str(persona_file.resolve()), stat.st_mtime_ns, stat.st_size)
            if (
                self._persona_cache.get("mtime") == mtime
                and self._persona_cache.get("content") is not None
            ):
                return self._persona_cache["content"]
            raw = persona_file.read_text()
            if not raw.strip():
                self._persona_cache = {"mtime": mtime, "content": ""}
                return ""
            words = raw.split()
            max_words = max(0, int(self.PERSONA_TOKEN_CAP * 0.75))
            if len(words) > max_words:
                truncated = " ".join(words[:max_words])
                last_section = truncated.rfind("\n## ")
                if last_section > 0:
                    truncated = truncated[:last_section]
                raw = truncated + "\n... (truncated, see persona file for full content)"
            self._persona_cache = {"mtime": mtime, "content": raw}
            return raw
        except Exception as exc:
            logger.debug("persona_from_file read failed: %s", exc)
            return ""

    def _persona_block(self) -> str:
        """Read the compatible file path or the call-time profile-bound DB."""
        if not self.PERSONA_ENABLED:
            return ""
        # Provider tool dispatch/rebinding uses the same reentrant Beam lock.
        # A bare mixin remains usable without importing the host runtime.
        from contextlib import nullcontext
        lock_factory = getattr(self, "_ensure_beam_access_lock", None)
        with lock_factory() if lock_factory else nullcontext():
            file_content = self._persona_from_file()
            if file_content:
                return file_content
            return self._persona_from_db()

    def _with_persona_block(self, base: str) -> str:
        persona_block = self._persona_block()
        if persona_block:
            return f"{base}\n\n{PERSONA_PROMPT_HEADER}\n{persona_block}"
        return base
