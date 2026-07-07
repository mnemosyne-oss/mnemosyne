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

    # L3 persona file injection. Default OFF -- opt in via
    # MNEMOSYNE_PERSONA_ENABLED=true. When OFF, no file IO happens.
    PERSONA_ENABLED = _parse_env_bool(PERSONA_ENABLED_ENV, False)
    PERSONA_FILE = _persona_file_from_env()
    PERSONA_TOKEN_CAP = _parse_env_int(PERSONA_TOKEN_CAP_ENV, DEFAULT_PERSONA_TOKEN_CAP)
    _persona_cache: Dict[str, Any] = {"mtime": None, "content": None}

    def _persona_block(self) -> str:
        """Read persona.md if feature enabled and file exists. Cached by mtime."""
        if not self.PERSONA_ENABLED:
            return ""
        try:
            persona_file = Path(self.PERSONA_FILE)
            if not persona_file.exists():
                self._persona_cache = {"mtime": None, "content": None}
                return ""
            mtime = persona_file.stat().st_mtime
            if (
                self._persona_cache.get("mtime") == mtime
                and self._persona_cache.get("content") is not None
            ):
                return self._persona_cache["content"]
            raw = persona_file.read_text()
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
            logger.debug("persona_block read failed: %s", exc)
            return ""

    def _with_persona_block(self, base: str) -> str:
        persona_block = self._persona_block()
        if persona_block:
            return f"{base}\n\n{PERSONA_PROMPT_HEADER}\n{persona_block}"
        return base
