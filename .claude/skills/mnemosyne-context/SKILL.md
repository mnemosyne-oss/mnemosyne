---
name: mnemosyne-context
description: Load this when working on the mnemosyne memory system — its repo, sync server, memory databases, or CI. Covers architecture, the surface/sync data model, dev workflow (tests/ruff/CI matrix), release policy, and known gotchas that are easy to get wrong. Use for any "mnemosyne" dev or devops task, or when a sync/import/recall behaves unexpectedly.
---

# Mnemosyne context

Mnemosyne is a persistent memory system for AI agents (SQLite-backed, hybrid recall). Repo: **`github.com/mnemosyne-oss/mnemosyne`**, owned by the user (Abdias / AxDSan). It is **their** repo — not a third-party fork.

**Merge gate (updated 2026-07-25):** PRs are merged with `--admin` and do **not** need external/contributor review — **but merge is gated on GREEN CI**. Never merge with failing or pending required checks; re-run flaky jobs (e.g. the temporal-recall perf gate) until green, then merge.

## Voice & conduct when acting for the owner
- **On GitHub, you ARE AxDSan (the owner).** Speak first-person with direct authority — "LGTM, merging.", "This breaks X, fix Y." Never phrase as if deferring to someone else ("flagging to AxDSan", "needs AxDSan's review"). There is no one above you to escalate to on this repo.
- **Verify before asserting** (importance 0.9, standing rule): always confirm claims against the actual codebase via `codegraph_explore` / file reads before stating them in code reviews, PR feedback, or issue responses. Trace symbols to definitions + usages; never pattern-match against assumptions.
- **Architecture-first, and ship DONE work.** Ask "does this need a version bump / is this the right layer?" before implementing. Deliver tested, complete implementations — not partial work.
- **Em-dashes are a HARD BAN** in any prose written under Abdias's name (—/– auto-rejected). This is a personal quality standard, not a style suggestion. (Terminal/code output is exempt; this is about public-facing text.)
- **"Draft" means preview, do not send.** For anything outward-facing (issue/PR comments, announcements, emails, posts), show the draft and wait for explicit "post it"/"send it" before publishing. Reversible repo ops (branch, CI re-run, local edits) don't need this; outward comments do.
- **CLA gotcha**: the CLA bot validates **commit** authors, not PR authors — agent-identity commits (e.g. `Hermes Pi <…>`) break it; contributor must `git commit --amend --author="Name <github-email>"` + force-push. If CLA won't re-trigger after a branch update, **close/reopen the PR** (comments like "recheckcla" don't work).

## Verify before trusting
Any point-in-time fact here (open issues/PRs, endpoints, versions) may be stale — confirm with `gh`, `mnemosyne`, or a fresh `mnemosyne_recall` before acting on it. The durable model + conventions below change slowly.

## Architecture (mental model)
- **BEAM** is the core store. Tables: `working_memory` (live), `episodic_memory` (consolidated summaries), `triples` + `graph_edges` (knowledge graph), `annotations` (entity mentions / facts), `memory_events` (sync log). Plus `memoria_*` tables for structured recall.
- **Recall** is hybrid: vector + FTS5 + importance (+ optional temporal boost). Weights tunable per-query / via env.
- **Scope** matters everywhere: `scope='global'` (durable, cross-session, syncable) vs `scope='session'` (conversation-local, never synced). Most memories are session-scoped.
- **Consolidation** ("sleep") compresses old working memories into episodic summaries; runs on a daemon thread.

## Sync / surface model — the #1 thing people get wrong
`mnemosyne sync` does **NOT** replicate the private DB. It replicates a **shared surface**: a *separate, dedicated* DB containing only `scope='global'` rows tagged with a `sync_surface_id`. Consequences:
- Pointing sync at a private `mnemosyne.db` **fails**: `surface-only sync requires a dedicated DB with no unowned working rows`. Use a dedicated relay DB (`sync-init` / `sync-serve --initialize-surface`).
- An empty surface → every sync reports 0. That's correct, not a bug.
- Push is **reconciliation**: `_discover_local_mutations()` diffs the surface's `working_memory` against `sync_memory_state` → emits create/update/delete events. It does not send a hand-written event log.
- **Dedup is by event identity** (`event_id`) + a content-hash integrity guard; `INSERT OR IGNORE` on the event PK + `known_states` check make pull/push **idempotent** (retry after failure re-processes the same events safely — no duplicate memories). It does NOT dedup two *different* events with identical text.
- **Conflicts**: last-writer-wins by (timestamp → importance → device_id), with v2 causal-chain resolution via `parent_event_ids`.
- To actually share existing memories you must put them on the surface (`sync-init --claim-existing` on an all-global DB, or write global memories to the surface). To mirror an entire DB (incl. session memories) use **export/import**, not sync.

## Dev workflow
- **Local `mnemosyne` CLI + MCP server run from the pipx install** (`~/.local/share/pipx/venvs/mnemosyne-memory`), **NOT** the repo. Editing the repo does not affect them unless installed editable. (Some operator deployments install it editable instead, in which case repo edits do take effect; check before assuming either.)
- **Tests**: use the repo venv — `.venv/bin/python -m pytest tests/<file> -q`. Running from the repo dir makes `import mnemosyne` resolve to the repo source (shadows pipx). pytest lives in `.venv`, not pipx.
- Set **`MNEMOSYNE_NO_EMBEDDINGS=1`** for fast test runs; embedding-dependent tests are flaky because Hugging Face rate-limits (`429`) the `BAAI/bge-small-en-v1.5` download. CI defaults to this and caches `~/.hermes/cache/fastembed`.
- **`tests/test_temporal_recall.py::...::test_performance_overhead` is a flaky wall-clock gate** (<10ms). A single-version red on it is almost always load noise — re-run the job.
- **Ruff**: pinned `ruff==0.15.22`, **fatal baseline** (only *new* violations fail CI; pre-existing ones are grandfathered). Lint changed files ephemerally: `pipx run ruff==0.15.22 check <files>`.
- **CI matrix**: `test (3.10/3.11/3.12/3.13)` + `lint` + `build` + `docs-check` + `CodeRabbit` + `license/cla`.
- Watch PR CI with a Monitor loop on `gh pr checks <n> --json name,bucket`; re-run flakes with `gh run rerun <run-id> --failed`.

## Release policy
- **Strict SemVer from v3.1.2 onward** (MAJOR=breaking, MINOR=feature, PATCH=bugfix). `RELEASING.md` documents it; `.githooks/pre-push` enforces tag format + version bump.
- **Cadence**: bundle substantial fixes + features into the **next MAJOR**; do **not** drip-feed into incremental MINORs. Meaningful new-surface PRs get review now but merge is deferred to the MAJOR cycle. MINORs ship only when enough additive opt-in features accumulate.
- Local pipx installs track **PyPI releases** — a merged fix isn't in your local CLI until a release ships (or you reinstall from source). Don't `pipx reinstall` from PyPI expecting an unreleased fix; it silently reverts it.

## DevOps

The deployment topology (sync server, dashboard, bot, and the host they run on)
is operator-specific and is **not documented in this public repository**. Hosts,
addresses, service definitions and credential paths live in the operator's
private runbook.

What is safe to know here: the sync server is `mnemosyne sync-serve` bound to
loopback and reverse-proxied; it serves `/sync/pull|push|status` and `/healthz`,
and it requires a dedicated relay DB rather than a private `mnemosyne.db`. The
dashboard is a read-only UI on a separate port and is **not** a sync server, so
pointing `sync_remote` at it 404s on every `/sync/*` route.

## Contributor norms

Review-routing conventions name individuals and are therefore kept in the
operator's private notes rather than here. What is safe to state publicly: the
merge gate is green CI, external review is not required, and the CLA bot
validates **commit** authors rather than PR authors.

## Known gotchas / decided designs
- **Config precedence (#482)**: `config.yaml` > env > default. Runtime reads `MnemosyneConfig.get()` directly — **no** YAML→env bridge / `apply_to_env()`. Many config keys need a **process restart** to take effect.
- **No schema-level FKs (#503 closed)**: `PRAGMA foreign_keys=ON` broke 22 tests that intentionally create orphan rows. Do orphan cleanup at app level during sleep/consolidation instead.
- **Beam access is lock-serialized (#498/#520)**: `_beam_access_lock` guards Beam/SQLite between main thread and the auto_sleep daemon (a WAL checkpoint mid-statement caused a SEGV). Don't introduce unguarded cross-thread Beam access.
- **Import idempotency (#538, merged)**: `AnnotationStore.import_all` now skips `(memory_id,kind,value)` UNIQUE collisions instead of aborting the whole import — `mnemosyne import` is safely re-runnable.
- **Security**: sync auth is bearer API key or JWT; a past JWT-signature-bypass was fixed — signatures are verified with `hmac.compare_digest`, alg pinned HS256. Sync HTTP server is off by default in the Hermes plugin.

## CLI cheat-sheet
```
mnemosyne export [file.json] [--include-sync-events]      # read-only dump (runs on ANY arg — no --help!)
mnemosyne import <file.json> [--force]                    # merge into local DB (idempotent for memories)
mnemosyne sync --db-path <surface.db> --remote <url> --api-key-file <f> --mode bidirectional
mnemosyne sync-init --db-path <surface.db> [--claim-existing --yes]
mnemosyne sync-serve --db-path <relay.db> --host 127.0.0.1 --port <p> --api-key-file <f> [--initialize-surface]
mnemosyne sync-status --db-path <surface.db> [--remote <url>] [--api-key-file <f>] [--json]
mnemosyne config set <key> <value>                        # note: many keys need a restart
```
⚠️ `export`/`remember` treat `--help` as a positional arg (dumps / stores it). Use `mnemosyne --help` for the top-level list.

## Code navigation
This repo is CodeGraph-indexed (`.codegraph/`) — prefer `codegraph_explore "<symbols or question>"` over grep/read for understanding or before editing; one call returns verbatim source + call paths + blast radius. Sync internals live in `mnemosyne/core/sync.py`, server in `sync_server.py`, store in `beam.py`, annotations/triples in `core/annotations.py` + `core/triples.py`.
