from __future__ import annotations

from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


class MnemosyneMemoryProvider(MemoryProvider):
    """Lightweight Hermes memory-provider shim for the Mnemosyne plugin.

    The installed Mnemosyne plugin already supplies the real tool surface and
    prompt hooks via hermes_plugin.register(). This shim exists so Hermes's
    external memory-provider system can recognize Mnemosyne as the active
    provider in config/doctor/status flows.
    """

    def __init__(self):
        self._memory = None
        self._session_id = ""

    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        try:
            from mnemosyne.core.memory import Mnemosyne  # noqa: F401
            return True
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        from mnemosyne.core.memory import Mnemosyne

        self._session_id = session_id
        self._memory = Mnemosyne(session_id=f"hermes_{session_id}")

    def system_prompt_block(self) -> str:
        return (
            "# Mnemosyne Memory\n"
            "Active local memory provider. Durable memory is stored in the local "
            "Mnemosyne SQLite/BEAM store."
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # The installed Mnemosyne plugin already registers its tools through the
        # normal plugin system. Returning an empty list avoids duplicate schemas.
        return []

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._memory or action not in {"add", "replace"} or not content:
            return
        try:
            self._memory.remember(
                content=content,
                source=f"legacy_memory:{target}",
                importance=0.6,
                metadata={
                    "bridge": "builtin-memory",
                    "target": target,
                    "action": action,
                },
            )
        except Exception:
            # Never let provider bridging break the main agent flow.
            pass

    def shutdown(self) -> None:
        self._memory = None
