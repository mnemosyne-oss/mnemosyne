---
description: "Persistent cross-session memory via Mnemosyne — store, recall, and consolidate facts, preferences, and context."
---

# Mnemosyne Memory

You have persistent memory across sessions via Mnemosyne. Use it to avoid making
the user repeat themselves and to recall past decisions, preferences, and
project context.

## When to Use

**memory_recall** — Call this FIRST when:
- The user references something from a past conversation ("remember when…", "like last time")
- You need to check if a preference or fact is already stored before asking
- Starting a new session in a project you've worked on before

**memory_remember** — Call this when:
- The user states a preference ("I prefer…", "always use…", "never do…")
- You learn a stable fact about the project, environment, or workflow
- A decision is made that should persist (architecture choice, convention adopted)
- You discover a tool quirk or workaround worth keeping

**memory_forget** — Call this when:
- A stored fact is stale, wrong, or superseded
- The user corrects something you previously stored

**memory_sleep** — Call this when:
- After a long session with many memories stored
- Memory recall feels stale or results seem outdated

**memory_stats** — Call this to check the state of the memory system.

## Principles

1. **Recall before asking.** If the user might have already told you something,
   search memory first. Only ask if recall returns nothing relevant.

2. **Store declarative facts, not instructions.**
   - ✅ "User prefers concise responses"
   - ✅ "Project uses pytest with xdist, 4 workers"
   - ❌ "Always respond concisely"
   - ❌ "Run tests with pytest -n 4"

3. **Importance matters.** User preferences and corrections = 0.8+. Project
   facts = 0.5. Auto-captured session activity = 0.3. Reserve 0.9+ for
   identity-level facts (name, role, core workflow).

4. **Don't store one-off task progress.** "Fixed bug X", "submitted PR Y",
   "Phase N done" will be stale in a week. Use memory for things that still
   matter later.

5. **The afterTool hook auto-captures file edits.** You don't need to manually
   store "I edited file X" — that happens automatically. Focus your manual
   memory_remember calls on preferences, decisions, and facts.

## MCP Server (Optional Advanced)

For the full Mnemosyne MCP surface (triples, graph traversal, scratchpad,
sync, canonical facts, persona management), add the Mnemosyne MCP server to
your Zero config:

```json
{
  "mcp": {
    "servers": {
      "mnemosyne": {
        "type": "stdio",
        "command": "mnemosyne",
        "args": ["mcp"]
      }
    }
  }
}
```

This exposes the full Mnemosyne tool surface including `mnemosyne_remember`,
`mnemosyne_recall`, `mnemosyne_sleep`, `mnemosyne_triple_add`,
`mnemosyne_graph_query`, `mnemosyne_scratchpad_*`, and many more. The plugin's
5 shell-based tools cover the common case; the MCP server is for power users
who want the full surface.
