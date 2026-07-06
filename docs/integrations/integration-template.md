# Integration Template — Add Mnemosyne to Any Platform

Adding Mnemosyne to a new AI platform takes ~100 lines of code.
The pattern is always the same:

## The Contract

Every integration needs to do three things:

1. **Connect** — Point Mnemosyne at a database path
2. **Expose** — Surface remember/recall/forget operations
3. **Configure** — Let users set db_path, bank, top_k

## Template

```python
"""
mnemosyne-{platform} — Mnemosyne integration for {Platform Name}.

Installation:
    pip install mnemosyne-memory

Usage:
    # Platform-specific setup instructions
"""

import json
import os
from pathlib import Path
from typing import Optional, Any, Dict, List

from mnemosyne.core.beam import BeamMemory


# ── 1. Config ──────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "MNEMOSYNE_DATA_DIR",
        Path.home() / ".hermes" / "mnemosyne" / "data",
    )
)


# ── 2. Adapter ─────────────────────────────────────────────────────────

class MnemosyneAdapter:
    """Mnemosyne adapter for {Platform Name}."""

    def __init__(
        self,
        db_path: str = str(DEFAULT_DATA_DIR),
        bank: str = "default",
        top_k: int = 5,
    ):
        self.db_path = db_path
        self.bank = bank
        self.top_k = top_k
        self._memory: Optional[BeamMemory] = None

    def _get_memory(self) -> BeamMemory:
        """Lazy-init memory backend."""
        if self._memory is None:
            db_dir = Path(self.db_path)
            db_dir.mkdir(parents=True, exist_ok=True)
            self._memory = BeamMemory(
                session_id=self.bank,
                db_path=str(db_dir / f"{self.bank}.db"),
            )
        return self._memory

    def remember(
        self,
        content: str,
        source: str = "{platform}",
        importance: float = 0.5,
    ) -> str:
        """Store a memory. Returns memory ID."""
        mem = self._get_memory()
        return str(mem.remember(content, source=source, importance=importance))

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity."""
        mem = self._get_memory()
        k = top_k or self.top_k
        results = mem.recall(query, top_k=k)
        return [
            {
                "id": r.get("memory_id") or r.get("id"),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
                "source": r.get("source", ""),
                "timestamp": str(r.get("timestamp", "")),
            }
            for r in (results or [])
        ]

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        mem = self._get_memory()
        mem.forget(memory_id)
        return True

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        mem = self._get_memory()
        base = {"bank": self.bank, "data_dir": self.db_path}
        if hasattr(mem, "get_stats"):
            base.update(mem.get_stats())
        return base
```

## Step 3: Add Platform-Specific Glue

Every platform has its own way of exposing tools. Here's how to find it:

| Platform | Integration Point | Example |
|----------|------------------|---------|
| OpenWebUI | `@tool` class with Valves | [openwebui-tool.md](openwebui-tool.md) |
| OpenClaw | `MemoryProvider` ABC | [openclaw.md](openclaw.md) |
| Claude Code | MCP config (`claude.json`) | [claude-code-mcp.md](claude-code-mcp.md) |
| Cursor | MCP config (`.cursor/mcp.json`) | [cursor-mcp.md](cursor-mcp.md) |
| Hermes | MCP config or plugin | [hermes-mcp.md](hermes-mcp.md) |
| Zero | Plugin manifest (tools + hooks) | [zero.md](zero.md) |
| Custom SDK | Direct Python import | Just call `adapter.remember()` |

## MCP Shortcut

If the platform supports MCP (Model Context Protocol), the integration is just a config file:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "mnemosyne",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

No code needed. If the platform doesn't support MCP, use the adapter template above.

## Checklist

When you finish an integration, verify:

- [ ] Install: `pip install mnemosyne-memory` works
- [ ] Connect: Database created at the configured path
- [ ] Remember: Storing a memory returns an ID
- [ ] Recall: Semantic search returns results
- [ ] Forget: Memory is removed
- [ ] Config: User can set db_path, bank, top_k
- [ ] Error handling: Bad inputs don't crash the platform
- [ ] No external deps: Only needs mnemosyne-memory
