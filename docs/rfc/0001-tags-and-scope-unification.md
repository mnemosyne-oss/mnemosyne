# RFC 0001: Tags and Scope Unification for Recall

**Status:** Draft
**Author:** achrllrogia45
**Related issue:** https://github.com/AxDSan/mnemosyne/issues/205
**Target version:** TBD

---

## Summary

`scope` currently exists on `remember()` but not on `recall()`. This creates an API symmetry gap: users can write scoped memories, but cannot explicitly ask recall to include or exclude a scope.

This RFC proposes a small unification:

1. Keep `scope` for backward compatibility.
2. Add first-class `tags` to memories.
3. Treat `scope` as one reserved tag namespace: `scope:<value>`.
4. Add tag filtering to `recall()`.
5. Preserve current default recall behavior unless a caller asks for filters.

In short:

```python
remember("Fix login bug", scope="session", tags=["project:app", "area:auth"])
recall("bug", tags=["area:auth"])
recall("bug", tags=["scope:session"], tag_mode="all")
```

This makes scope filtering possible without locking Mnemosyne into a single-purpose scope model.

---

## Problem

Issue #205 asks whether this is intended:

```python
remember("...", scope="session")
recall("...")  # no scope filter exists
```

Today, `scope` is write-time metadata and an internal retrieval signal. That may be intentional, but the API surface does not make it clear.

The practical user need is broader than just `session` vs `global`.

Example from the issue:

> "Say I'm working with code, I can have auth in a room, UI in another, etc."

That is not just a scope problem. It is a dimensional filtering problem.

Users want to separate memory by axes such as:

- `area:auth`
- `area:ui`
- `project:my-app`
- `profile:poseidon`
- `source:cli`
- `scope:global`
- `scope:session`

A single `scope` string cannot model this cleanly.

---

## Design thesis

`scope` is a special case of tags.

Tags can express all three mental models raised in the issue:

| Mental model | Tag interpretation |
|---|---|
| Scope as strict filter | `recall(query, tags=["scope:session"])` |
| Scope as ranking signal | boost matching tags without excluding non-matches |
| Scope as dimension | `area:auth`, `area:ui`, `project:x`, `profile:y` |

This means Mnemosyne does not need to choose between the three models as mutually exclusive. It can keep current behavior by default and expose strict filtering only when the caller requests it.

---

## Goals

- Make scoped recall possible.
- Preserve current default recall behavior.
- Support user-defined memory dimensions beyond `global` and `session`.
- Keep schema and API small.
- Avoid destructive data migration for existing users.
- Leave room for later ranking boosts and richer boolean filtering.

---

## Non-goals

- Replace `session_id`.
- Remove `scope` immediately.
- Add full boolean query language in the first implementation.
- Redesign vector recall or FTS recall.
- Require sqlite-vec changes.

---

## Proposed API

### remember()

Add optional `tags`:

```python
def remember(
    content: str,
    *,
    scope: str = "session",
    tags: list[str] | None = None,
    ...,
) -> str:
    ...
```

Behavior:

```text
stored_tags = normalize(tags)
stored_tags.add(f"scope:{scope}")
```

Examples:

```python
remember(
    "OAuth callback bug only affects mobile Safari",
    scope="session",
    tags=["project:shop", "area:auth", "platform:ios"],
)
```

Stored tags:

```text
scope:session
project:shop
area:auth
platform:ios
```

### recall()

Add optional tag filter:

```python
def recall(
    query: str,
    top_k: int = 40,
    *,
    scope: str | None = None,
    tags: list[str] | None = None,
    tag_mode: Literal["any", "all"] = "all",
    ...,
) -> list[dict]:
    ...
```

The default is `all`. When a caller passes multiple tags, the common
expectation is intersection, not union. For example
`recall("bug", tags=["project:shop", "area:auth"])` should return memories
that are in the shop project AND the auth area. Callers who want union pass
`tag_mode="any"` explicitly.

`tag_mode` is a closed set. An unknown value raises `ValueError` rather than
silently degrading to a default.

Initial `tag_mode` values:

```text
any  = at least one requested tag must match
all  = every requested tag must match
```

Examples:

```python
recall("OAuth bug", tags=["area:auth"])
recall("OAuth bug", tags=["project:shop", "area:auth"], tag_mode="all")
recall("anything", scope="global")
recall("anything", scope="session")
recall("anything", tags=["scope:global"])
recall("anything", tags=["scope:session"])
```

`scope=` is convenience syntax for the reserved scope tag. Passing
`scope="global"` is equivalent to adding `"scope:global"` to `tags`. If both
`scope=` and `tags=` are provided, the same conflict rule used by `remember()`
applies: matching scope tags are deduplicated and conflicting scope tags raise
`ValueError`.

### Why only any/all first

`any` and `all` cover most immediate cases and keep the first PR small.

Future modes can be added later:

```text
any_strict
all_strict
boolean expression
negative tags
```

Those should be follow-up work, not required for the first scope filtering fix.

---

## Schema

Use a denormalized `tags` column on memory tables:

```sql
ALTER TABLE working_memory ADD COLUMN tags TEXT DEFAULT '[]';
ALTER TABLE episodic_memory ADD COLUMN tags TEXT DEFAULT '[]';
```

Format:

```json
["scope:session", "project:shop", "area:auth"]
```

### Why denormalized first

A normalized schema would be cleaner for large tag analytics:

```sql
memory_tags(memory_id, tag)
```

But the first implementation should optimize for low migration risk and low code churn.

Denormalized JSON is enough for:

- storing tags
- returning tags in recall results
- exact tag filtering in Python after candidate retrieval
- future migration to a normalized table if needed

This avoids introducing joins into the recall hot path in the first version.

---

## Retrieval behavior

### Default behavior

No change:

```python
recall("auth bug")
```

This preserves current semantics, including global memory behavior and existing ranking.

### Filtered behavior

When tags are provided:

```python
recall("auth bug", tags=["area:auth"])
```

Recall should restrict results to candidate memories whose stored tags match the requested tag mode.

Tag filtering runs after the existing candidate gathering step, but the candidate pool must be widened when tags are present. Filtering a fixed `top_k` pool after retrieval can drop valid tagged memories that ranked just outside the pool, producing empty or low quality results even though matches exist.

```python
candidate_k = max(top_k * 5, 100) if tags else top_k
candidates = existing_recall_pipeline(query, top_k=candidate_k)
filtered = filter_by_tags(candidates, tags, tag_mode)
return rerank_or_trim(filtered, top_k)
```

This keeps the first implementation simple and avoids rewriting FTS/vector SQL, while avoiding the silent miss caused by filtering a small candidate pool.

A later optimization can push tag filtering down into SQL so the candidate pool is already tag constrained, removing the need to overfetch.

---

## Scope compatibility

Existing `scope` remains valid.

New writes:

```python
remember("...", scope="global")
```

Should behave as if this tag was also written:

```text
scope:global
```

New writes with custom scope:

```python
remember("...", scope="auth")
```

Should store:

```text
scope:auth
```

This supports the current API while making scope queryable.

---

## Migration

No destructive migration required. The `scope` column is preserved.

Migration behavior (required, not optional):

1. Add `tags TEXT DEFAULT '[]'` to memory tables.
2. Backfill every existing row with `scope:<scope>` when that tag is absent.
3. Keep a recall-time fallback only as a safety net for rows a migration missed.

A required, deterministic backfill avoids the inconsistent state where some rows are queryable by `scope:<value>` and others are not. Recall by scope tag should behave the same for old and new rows.

Recommended first PR:

```text
Add tags column to working_memory and episodic_memory.
Run a one-time deterministic backfill of scope:<scope> for all rows.
Keep scope column unchanged.
```

Example migration:

```python
def _backfill_scope_tags(conn, table):
    rows = conn.execute(f"SELECT id, scope, tags FROM {table}").fetchall()
    for row in rows:
        tags = _parse_tags(row["tags"])          # tolerant, see below
        scope_tag = f"scope:{(row['scope'] or 'session').strip().lower()}"
        if scope_tag not in tags:
            tags.append(scope_tag)
            conn.execute(
                f"UPDATE {table} SET tags = ? WHERE id = ?",
                (json.dumps(sorted(set(tags))), row["id"]),
            )
    conn.commit()
```

This is idempotent: re-running it does not duplicate the scope tag, because the tag is only appended when absent and stored deduplicated. It covers both `working_memory` and `episodic_memory`. Rows with `NULL`, empty, or malformed `tags` are handled by `_parse_tags` (see Implementation sketch) rather than crashing.

The `scope` column should not be removed.

---

## Result shape

Recall results always include a `tags` list. Rows with no tags return `"tags": []` rather than omitting the field, so clients can rely on the key existing.

```json
{
  "id": "...",
  "content": "OAuth callback bug only affects mobile Safari",
  "scope": "session",
  "tags": ["scope:session", "project:shop", "area:auth"]
}
```

This helps clients debug why a result matched.

---

## Tag format

Use namespaced strings:

```text
namespace:value
```

Examples:

```text
scope:session
scope:global
project:shop
area:auth
profile:poseidon
source:cli
```

Rules:

- Form: `namespace:value`. A bare string with no `:` is allowed but is treated as namespace-less (`value` only). Tools should prefer the namespaced form.
- `namespace`: lowercase ASCII letters, digits, underscore, hyphen.
- `value`: non-empty after trimming.
- For namespaced tags, split on the first `:` only. This allows values such as `source:https://example.com`.
- Namespaces are lowercased. Values preserve case. For example, `Area:Auth` normalizes to `area:Auth`, while `project:MyApp` keeps `MyApp` intact.
- Bare tags are lowercased as whole strings and matched exactly. They are allowed for convenience, but tools should prefer namespaced tags.
- Trim surrounding whitespace.
- Reject empty tags and non-string tags.
- Deduplicate; storage is a sorted unique set.
- Soft limits: tag length <= 128 chars, tags per memory <= 64. Over-limit input is rejected rather than silently truncated.

Reserved namespace and scope conflict rule:

- `scope:` is reserved. `remember(scope=...)` writes exactly one `scope:<value>` tag.
- If a caller passes a `scope:*` tag in `tags=` that conflicts with the `scope=` argument, raise `ValueError`. Example: `remember(scope="session", tags=["scope:global"])` is rejected.
- A `scope:*` tag in `tags=` that matches the `scope=` argument is allowed and deduplicated.

Why strings instead of dicts:

```python
# Prefer this
["project:shop", "area:auth"]

# Not this for v1
[{"project": "shop"}, {"area": "auth"}]
```

Strings are easier to pass through CLI tools, MCP tools, JSON, and simple Python clients.

---

## Interaction with session_id

`session_id` and `scope` should remain separate concepts.

```text
session_id = current conversation or workspace identity
scope      = existing visibility/ranking metadata
tags       = user-defined dimensions and query filters
```

A session can still be used as an isolation boundary:

```python
recall("bug")  # current session + globals, current behavior
```

Tags add cross-cutting dimensions:

```python
recall("bug", tags=["area:auth"])
```

This supports the issue author's "auth room" / "UI room" case without overloading `session_id`.

---

## Backward compatibility

This proposal is backward compatible because:

- Existing calls to `remember()` still work.
- Existing calls to `recall()` still work.
- `scope` remains stored.
- Default recall behavior is unchanged.
- Tag filtering is opt-in.

---

## Implementation sketch

### 1. Schema migration

Add helper:

```python
_add_column_if_missing(conn, "working_memory", "tags", "TEXT DEFAULT '[]'")
_add_column_if_missing(conn, "episodic_memory", "tags", "TEXT DEFAULT '[]'")
```

### 2. Tag parsing (tolerant read)

```python
def _parse_tags(raw) -> list[str]:
    """Tolerant read of a stored tags column.

    Handles NULL, empty string, malformed JSON, and non-list JSON without
    raising. Recall must never crash on a bad tags value.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [t for t in parsed if isinstance(t, str) and t]
```

### 3. Tag normalization (write path)

```python
_MAX_TAG_LEN = 128
_MAX_TAGS = 64


def _normalize_tags(tags: list[str] | None, scope: str | None = None) -> list[str]:
    out = set()
    for tag in tags or []:
        if not isinstance(tag, str):
            raise TypeError("tags must be strings")
        tag = tag.strip()
        if not tag:
            continue
        if ":" in tag:
            namespace, value = tag.split(":", 1)
            namespace = namespace.strip().lower()
            value = value.strip()
            if not namespace or not value:
                raise ValueError(f"invalid tag: {tag!r}")
            tag = f"{namespace}:{value}"
        else:
            tag = tag.lower()
        if len(tag) > _MAX_TAG_LEN:
            raise ValueError(f"tag too long (>{_MAX_TAG_LEN}): {tag[:32]}...")
        out.add(tag)

    if scope:
        scope_tag = f"scope:{scope.strip().lower()}"
        # Conflict rule: a scope:* tag that disagrees with scope= is rejected.
        conflicting = {t for t in out if t.startswith("scope:") and t != scope_tag}
        if conflicting:
            raise ValueError(
                f"scope tag conflict: scope={scope!r} but tags has {sorted(conflicting)}"
            )
        out.add(scope_tag)

    if len(out) > _MAX_TAGS:
        raise ValueError(f"too many tags (>{_MAX_TAGS}): {len(out)}")
    return sorted(out)
```

### 4. Store tags on write

```python
tags_json = json.dumps(_normalize_tags(tags, scope))
```

### 5. Filter recall results

```python
def _match_tags(memory_tags, requested, mode):
    memory_tags = set(memory_tags or [])
    requested = set(requested or [])
    if not requested:
        return True
    if mode == "any":
        return bool(memory_tags & requested)
    if mode == "all":
        return requested <= memory_tags
    raise ValueError(f"unknown tag_mode: {mode}")
```

### 6. Return tags in result dictionaries

Add `tags` to recall result payloads if the source row has a tags column.

---

## Open questions

1. Should tag filtering happen before or after vector/FTS candidate generation?

   Resolved for v1: after candidate generation, but with an overfetched candidate pool (`max(top_k * 5, 100)`) when tags are present, so valid tagged memories are not dropped. SQL pushdown is the later optimization.

2. Should tag matches boost rank even when tags are not strict filters?

   Recommended v1: no. Keep rank behavior unchanged unless tags are requested.

3. Should `scope=` remain a direct `recall()` parameter?

   Possible convenience wrapper:

   ```python
   recall("bug", scope="global")
   # equivalent to
   recall("bug", tags=["scope:global"])
   ```

   Resolved for v1: yes. Tags are the core primitive, but `scope=` directly addresses the original issue and is equivalent to adding the matching reserved scope tag.

4. Should tags be normalized to lowercase?

   Resolved for v1: namespaces are lowercased, values preserve case, and bare tags are lowercased as whole strings.

---

## Recommendation

Adopt tags as the general mechanism and treat scope as a reserved tag namespace.

This solves the immediate issue:

```python
recall("...", tags=["scope:session"])
```

It also solves the broader room/dimension use case:

```python
recall("...", tags=["area:auth"])
recall("...", tags=["area:ui"])
```

This avoids a narrow scope-only fix that would need to be redesigned later once users ask for project, area, profile, or source filters.

---

## Minimal first PR

A small first PR could include:

1. Add `tags` column to `working_memory` and `episodic_memory`.
2. Add `tags` parameter to `remember()`, with normalization and the scope conflict rule.
3. Store `scope:<scope>` automatically.
4. Run the deterministic backfill so existing rows gain `scope:<scope>`.
5. Add `scope`, `tags`, and `tag_mode` parameters to `recall()` (`tag_mode` default `all`).
6. Overfetch the candidate pool when tags are present, then filter.
7. Always return a `tags` list (`[]` when empty) in recall result dictionaries.
8. Add tests for:
   - scope tag auto-write
   - deterministic backfill of existing rows (legacy row gains `scope:<scope>`)
   - backfill idempotency (re-run does not duplicate scope tag)
   - recall by `scope:global`
   - recall by `scope:session`
   - recall by `scope="global"`
   - recall by `scope="session"`
   - recall by custom tag like `area:auth`
   - bare tag exact matching
   - `tag_mode="any"` (union)
   - `tag_mode="all"` (intersection, the default)
   - unknown `tag_mode` raises `ValueError`
   - scope conflict rejected (`scope="session"` with `tags=["scope:global"]`)
   - uppercase namespaces normalized while values preserve case
   - empty / non-string tags rejected
   - malformed stored `tags` JSON tolerated by recall (no crash)
   - filtered recall returns a valid tagged memory that ranks outside `top_k` before filtering

This keeps the patch focused while preserving a clean path to normalized tag tables, ranking boosts, and boolean tag queries later.
