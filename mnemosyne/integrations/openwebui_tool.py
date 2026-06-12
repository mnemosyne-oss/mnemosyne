"""
Mnemosyne - OpenWebUI Tool Integration
========================================
Class-based @tool with Pydantic Valves for configuration.

For OpenWebUI (class-based Valves pattern):
    1. pip install mnemosyne-memory
    2. Create bridge file in OpenWebUI's data/tools/ directory:
       echo "from openwebui_tool import MnemosyneTool as Mnemosyne" \\
         > /path/to/open-webui/data/tools/mnemosyne.py
    3. Configure in OpenWebUI Workspace settings

Standalone usage (for testing/scripting):
    from openwebui_tool import MnemosyneTool
    tool = MnemosyneTool()
    result = await tool.mnemosyne_remember("Hello world")
"""

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# Core Mnemosyne — always available
from mnemosyne.core.memory import Mnemosyne

# ── Config ──────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "MNEMOSYNE_DATA_DIR",
        os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")) + "/mnemosyne/data",
    )
)


class MnemosyneTool:
    """
    Mnemosyne memory tool for OpenWebUI.

    Provides remember, recall, forget, stats, and sleep operations
    as discoverable @tool methods.
    """

    class Valves(BaseModel):
        """Configuration rendered as settings in OpenWebUI."""

        db_path: str = Field(
            default=str(DEFAULT_DATA_DIR),
            description="Path to Mnemosyne data directory",
        )
        bank: str = Field(
            default="default",
            description="Memory bank name for conversation isolation",
        )
        top_k: int = Field(
            default=5,
            ge=1,
            le=100,
            description="Max results returned by recall",
        )
        vec_weight: Optional[float] = Field(
            default=None,
            description="Vector search weight (None = env default)",
        )
        fts_weight: Optional[float] = Field(
            default=None,
            description="Full-text search weight (None = env default)",
        )
        importance_weight: Optional[float] = Field(
            default=None,
            description="Importance boost weight (None = env default)",
        )
        show_citations: bool = Field(
            default=True,
            description="Show source citations in tool output",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True
        self._mem = None

    # ── Lazy Initializer ──────────────────────────────────────────────

    def _get_memory(self) -> Mnemosyne:
        """Lazy-init memory once Valves are populated by OpenWebUI."""
        if self._mem is not None:
            return self._mem
        db_dir = Path(self.valves.db_path)
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(db_dir / f"{self.valves.bank}.db")
        self._mem = Mnemosyne(
            session_id=self.valves.bank,
            bank=self.valves.bank,
            db_path=Path(db_path),
        )
        return self._mem

    # ── Tools ────────────────────────────────────────────────────────

    async def mnemosyne_remember(
        self,
        content: str,
        source: str = "openwebui",
        importance: float = 0.5,
    ) -> str:
        """
        Store a memory for later recall.
        The LLM uses this to persist things the user says they want to remember.
        """
        try:
            mem = self._get_memory()
            kwargs = {"importance": importance}
            memory_id = mem.remember(content, source=source, **kwargs)
            return json.dumps({"memory_id": memory_id, "status": "ok"}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "status": "error"}, indent=2)

    async def mnemosyne_recall(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Search stored memories by semantic similarity to a query.
        Use this when the user asks 'what do you know about X' or
        'recall what I said about Y'.
        """
        try:
            mem = self._get_memory()
            k = top_k if top_k is not None else self.valves.top_k
            kwargs = {"top_k": k}
            if self.valves.vec_weight is not None:
                kwargs["vec_weight"] = self.valves.vec_weight
            if self.valves.fts_weight is not None:
                kwargs["fts_weight"] = self.valves.fts_weight
            if self.valves.importance_weight is not None:
                kwargs["importance_weight"] = self.valves.importance_weight

            results = mem.recall(query, **kwargs)
            if not results:
                return json.dumps(
                    {"results_count": 0, "results": [], "status": "ok"}, indent=2
                )
            return json.dumps(
                {
                    "results_count": len(results),
                    "results": [
                        {
                            "id": r.get("memory_id") or r.get("id", ""),
                            "content": r.get("content", ""),
                            "score": round(r.get("score", 0), 3),
                            "source": r.get("source", ""),
                            "timestamp": str(r.get("timestamp", "")),
                            "importance": r.get("importance", 0),
                        }
                        for r in results
                    ],
                    "status": "ok",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e), "status": "error"}, indent=2)

    async def mnemosyne_forget(self, memory_id: str) -> str:
        """
        Delete a specific memory by its ID.
        Use when the user wants to remove or correct a stored memory.
        """
        try:
            mem = self._get_memory()
            mem.forget(memory_id)
            return json.dumps({"memory_id": memory_id, "status": "ok"}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "status": "error"}, indent=2)

    async def mnemosyne_stats(self) -> str:
        """
        Get memory statistics: total count, per-tier breakdown,
        bank name, and database size.
        """
        try:
            mem = self._get_memory()
            stats = mem.get_stats() if hasattr(mem, "get_stats") else {}
            return json.dumps(
                {
                    "bank": self.valves.bank,
                    "data_dir": str(self.valves.db_path),
                    **stats,
                    "status": "ok",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e), "status": "error"}, indent=2)

    async def mnemosyne_sleep(self) -> str:
        """
        Run memory consolidation.
        Compresses working memories into long-term episodic summaries.
        """
        try:
            mem = self._get_memory()
            mem.sleep()
            return json.dumps({"status": "ok", "bank": self.valves.bank}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "status": "error"}, indent=2)
