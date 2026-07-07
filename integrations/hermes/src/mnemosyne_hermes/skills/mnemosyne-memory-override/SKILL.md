---
name: mnemosyne-memory-override
description: |
  Hard rule override that forces Mnemosyne for all durable memory storage.
  The legacy memory tool is DEPRECATED for user preferences, credentials,
  and project conventions. Use memory ONLY for ephemeral session state.
trigger: |
  Whenever you would use memory(action="add|replace|remove") for anything
  that should survive across sessions, STOP and use the Mnemosyne memory tools instead.
---

# Mnemosyne Memory Override

## CRITICAL RULE

**NEVER use the `memory` tool for durable facts.**

The legacy `memory` system is deprecated. It has a tiny 2,200 character limit
and does not use vector search. Mnemosyne is the single source of truth for
all durable user data.

## Mapping

| Old (FORBIDDEN) | New (REQUIRED) |
|-----------------|----------------|
| `memory(action="add", target="user", ...)` | `mnemosyne_remember(content=..., importance=0.9+, source="preference")` |
| `memory(action="add", target="memory", ...)` | `mnemosyne_remember(content=..., importance=0.7+, source="fact")` |
| `memory(action="replace", ...)` | Use the relevant Mnemosyne update flow: recall the current entry, then store corrected content with `mnemosyne_remember`; use an explicit update tool/flow when the integration exposes one. |
| `memory(action="remove", ...)` | Use the relevant Mnemosyne invalidation/forget flow: mark the obsolete fact invalid, supersede it with corrected content, or call the integration's forget/remove tool when available. Do not rely on passive aging for known-wrong durable data. |

## When to use legacy memory (rare)

- Ephemeral session state (current todo list, temp flags)
- When Mnemosyne is explicitly confirmed down

## Muscle Memory / Reflex Problem

You will reflexively reach for `memory` out of old habit. This is a known bug
in your training. Before EVERY memory tool call, pause and ask:

> "Is this durable? Would I want this next session?"

If yes → use Mnemosyne
If no (temp flag, todo state) → `memory` is acceptable

## Migration Cleanup

When moving data from legacy memory to Mnemosyne:
1. Save to Mnemosyne first with `mnemosyne_remember`
2. Then REMOVE the old entry from `memory` with `memory(action="remove")`
3. This prevents stale duplicates and confusion

## Enforcement

If you catch yourself typing `memory(action=` for durable data:
1. CANCEL that tool call
2. Use Mnemosyne instead
3. Set importance >= 0.7 for anything that matters later
4. If you already polluted legacy memory, remove the entry immediately
