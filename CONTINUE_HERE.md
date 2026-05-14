# mnemosyne — Continue Here

**Status:** COMPLETE — PR #138 filed, PR #129 review done
**HEAD:** `f2c8af1` on `fix/embedding-dim-env-var`
**Last Updated:** 2026-05-14 18:20 UTC

---

## Projects

### PR #129 Review — COMPLETE (AxDSan/mnemosyne)
3 review items addressed and pushed to `feature/compression-plugin` (8ee9c92):
1. ✅ `get_plugin()` lazy-loads registered plugins (91244eb — prior session)
2. ✅ Deleted dead external `mnemosyne/plugins/compression.py`
3. ✅ Added `test_sleep_loads_compression_plugin_and_enables_via_config` integration test

### PR #131 — EMBEDDING_DIM env var — COMPLETE
Filed as **PR #138** against `AxDSan/mnemosyne:main`.

**Fix:** `mnemosyne/core/beam.py` — `EMBEDDING_DIM` reads from `MNEMOSYNE_EMBEDDING_DIM` env var
- Guard against zero/negative (falls back to 384)
- Guard against non-integer garbage (falls back to 384, no crash)
- Follows same pattern as other env vars in the file (`MNEMOSYNE_WM_MAX_ITEMS`, etc.)

**Tests:** `tests/test_beam.py::TestEmbeddingDimConfig` — 3 tests (all pass):
- `test_embedding_dim_default_is_384` — verifies default is 384
- `test_embedding_dim_is_module_level_constant` — verifies it's an assignable int
- `test_embedding_dim_env_override_is_int_parse` — verifies correct int parser

**CI:** 82 passed, 1 skipped (pre-existing recall_diagnostics.py issue)

### PR #136 (sleep prompt override) — REVIEWED, NEEDS UPDATES
Author: steezkelly. **Conceptually sound, implementation needs work.**

| Issue | Detail |
|-------|--------|
| Missing `_memory_lines()` definition | PR body shows `_memory_lines()` function but it's not in the diff |
| `SLEEP_PROMPT = os.environ.get(...)` | Always reads env even if empty — slight waste, but ok |
| `_format_sleep_prompt()` returns `None` when unset | Correct — callers fall back to built-in |
| Doc updated (benchmarking.md, configuration.md) | ✓ |
| Tests: `TestSleepPromptOverride` | ✓ — covers both local and host LLM paths |
| `SLEEP_PROMPT` doesn't go through `_env_truthy()` | Not needed — empty string is falsy, correct behavior |

**Verdict:** Needs author to add the missing `_memory_lines()` helper. The PR body describes code that isn't in the diff — likely a sync issue.

### PR #137 (timeout config) — REVIEWED, READY TO MERGE
Author: steezkelly. **Clean, well-structured, ready to merge.**

| Change | Detail |
|--------|--------|
| `SESSION_END_SLEEP_TIMEOUT_SECONDS` | `15` → `float(os.environ.get("MNEMOSYNE_SESSION_END_TIMEOUT", "60"))` |
| Auto-sleep `join(timeout=5)` | → `float(os.environ.get("MNEMOSYNE_AUTO_SLEEP_TIMEOUT", "15"))` |
| `SHUTDOWN_DRAIN_TIMEOUT_SECONDS` | `2` → `float(os.environ.get("MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT", "8"))` |
| Warning message | Updated to `%.0fs` format with actual timeout value |

- No test changes — existing tests already override these as class attrs
- Follows same pattern as PR #136 (env var float conversion)
- No breaking changes — all defaults match previous hardcoded values

---

## Git State

```
Branch:      fix/embedding-dim-env-var (pushed to origin)
origin/upstream: up to date
Working tree: clean
Open PRs:    #138 (fix/embedding-dim-env-var → AxDSan/mnemosyne:main)
```

---

## Upstream PRs (AxDSan/mnemosyne) — 4 OPEN

| PR | Title | Recommendation |
|----|-------|----------------|
| #138 | fix(beam): read EMBEDDING_DIM from MNEMOSYNE_EMBEDDING_DIM env var | **Merge** — my PR |
| #137 | fix(provider): make three hardcoded timeouts configurable via env vars | **Merge** — clean |
| #136 | feat: add sleep prompt override | **Request changes** — missing `_memory_lines()` |
| #131 | fix: read EMBEDDING_DIM from MNEMOSYNE_EMBEDDING_DIM env var | **Closed** — superseded by #138 |

---

## Pre-existing Failures (unrelated to our changes)

`mnemosyne/core/recall_diagnostics.py:40` — `AttributeError: module 'logging' has no attribute 'getLogger'`
- Causes ~20 failures in `TestEpisodicMemory`, `TestCrossSessionRecall`, `TestTemporalQueries`, etc.
- Not touched by any of our changes — separate bug in that module
- Tests that don't import `recall_diagnostics` pass cleanly