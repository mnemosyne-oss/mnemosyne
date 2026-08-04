# Updating Mnemosyne

Covers all upgrade paths: v2.7 → latest, source installs, PyPI installs,
and systems with Python's `externally-managed-environment` (PEP 668).

If you're on **v3.10.1** and want the latest (v3.11.0), jump to
[Upgrading to v3.11.0](#upgrading-to-v3110-automated-sleep-model-refresh).
Already on v3.8.0? See
[Upgrading to v3.8.0](#upgrading-to-v380-sync-vecworking-and-reindex).

Already on v3.0.0? See [Upgrading to v3.0.0](#upgrading-to-v300-memoria-architecture).

---

## Quick Reference

| What changed | User action |
|---|---|
| New PyPI release | `pip install --upgrade mnemosyne-memory` + restart Hermes |
| Source-only fix | `git pull` + restart Hermes |
| New dependency / entry point | `git pull` + `pip install -e .` + restart Hermes |
| `externally-managed-environment` (Debian/Ubuntu) | Use a venv or `pip install --break-system-packages` — see [PEP 668 section](#pep-668-externally-managed-environment-on-debian--ubuntu) |
| SQLite schema changed (wondering?) | See [How to confirm schema changes](#how-to-confirm-schema-changes) |
| E6 TripleStore split (v2.8) | Auto-migrates on first init. Backup at `{db}.pre_e6_backup` |
| MEMORIA architecture (v3.0) | Auto-creates 5 new tables on first init. No manual action needed |
| `plugin.yaml` / tool schema | Restart Hermes only |

---

## Current / Unreleased — Hermes wrapper install safety

### Updating a Hermes wrapper install

For a persistent wrapper (the Docker/read-only deployment path), keep wrapper
mode and name the Python environment that contains the selected package:

```bash
mnemosyne-hermes install --mode wrapper --python /path/to/venv/bin/python --force
```

`--force` refreshes a wrapper only after that Python can resolve its
site-packages and import `mnemosyne_hermes`; an invalid `--python` leaves the
existing wrapper and opted-in profile links in place. Editable installs in the
selected environment are supported. To intentionally replace a wrapper with
the historical symlink install, acknowledge the mode change:

```bash
mnemosyne-hermes install --mode symlink --force --migrate-wrapper-to-symlink
```

That migration stops using the wrapper's selected Python. The installer warns
before making the change; without the explicit flag it refuses the replacement.

---

## PEP 668: externally-managed-environment on Debian / Ubuntu

Debian 13 (and Ubuntu 24.04+) ship Python with PEP 668 protection.
`pip install` outside a virtualenv fails with:

```
error: externally-managed-environment
× This environment is externally managed
```

**Solution 1: Use a virtualenv (recommended)**

```bash
python3 -m venv ~/mnemosyne-venv
source ~/mnemosyne-venv/bin/activate
pip install --upgrade mnemosyne-memory
```

Make sure Hermes is configured to use this venv's Python.

**Solution 2: pipx (for CLI tools)**

```bash
pip install pipx
pipx install mnemosyne-memory
```

**Solution 3: Override (quick fix, use with caution)**

```bash
pip install --upgrade mnemosyne-memory --break-system-packages
```

This bypasses the guard. Fine for personal machines or containers.
Not recommended for shared/multi-app systems.

**Solution 4: Source install with editable mode**

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e . --break-system-packages
```

Editable mode means future `git pull` is all you need — no re-install
for most updates.

---

## Upgrading to v3.7.0 — Usage-Driven Working Memory Decay

Released 2026-06-13. Minor release with working memory decay, temporal-triple lifecycle fix, and several packaging improvements.

### What changed

- **Working memory decay** — default TTL bumped from 24h to 168h (7 days).
  `get_context()` now bumps `recall_count` and `last_recalled` on returned items,
  and each bump extends the item's lifetime by up to `MNEMOSYNE_WM_BUMP_CAP_HOURS`
  (default 24h). Pinned items (`MNEMOSYNE_WM_PINNED_IDS`) are excluded from
  consolidation entirely.
- **Sleep consolidation skips pinned items** — a new `pinned` column on
  `working_memory` tells `sleep()` to leave those memories untouched.
- **Temporal-triple lifecycle restored** — `supersede`, `valid_until`, and `end`
  operations on triples are now functional (were absent in v3.5.0/v3.6.0 despite
  appearing merged). New `end_triple()` module function and
  `mnemosyne_triple_end` tool.
- **`HERMES_HOME` resolution fixed** — env var now checked before `Path.home()`
  fallback across beam, banks, memory, and integration files.
- **Packaging cleanup** — `openclaw` removed from `[all]` extra. Python 3.9
  support dropped (3.10+).

### User action required

```bash
pip install --upgrade mnemosyne-memory
```

No migration steps needed. New columns (`pinned`, `recall_count`, `last_recalled`)
are created lazily if absent. Your `upto_24_hours` `MNEMOSYNE_WM_TTL_HOURS`
overrides are still honoured — the default just changed.

### Rollback to v3.6.0

```bash
pip install 'mnemosyne-memory==3.6.0'
```

The working-memory schema additions are additive (`ALTER TABLE ... ADD COLUMN`).
Downgrading Python code to 3.6.0 while the schema has `pinned`/`recall_count`/
`last_recalled` columns is harmless — 3.6.0 ignores unknown columns. To fully
reverse the schema change (not necessary, but available):

```bash
echo "ALTER TABLE working_memory DROP COLUMN pinned;" | sqlite3 path/to/mnemosyne.db
echo "ALTER TABLE working_memory DROP COLUMN recall_count;" | sqlite3 path/to/mnemosyne.db
echo "ALTER TABLE working_memory DROP COLUMN last_recalled;" | sqlite3 path/to/mnemosyne.db
```

---

## Upgrading to v3.8.0 — Sync, vec_working, and Reindex

Released 2026-06-15. Minor release with bidirectional memory sync, dedicated
vec_working table, reindex command, fact_recall ranking fix, and smart plugin
upgrade tooling.

### What changed

- **Bidirectional memory sync** with optional client-side encryption. Event-log-
  based delta protocol with conflict detection via causal version chains.
  Uses a stdlib-only HTTP server (no FastAPI). New CLI: `mnemosyne sync`,
  `sync-serve`, `sync-status`, `sync-generate-key`.
- **vec_working dedicated table** — working-memory vectors now live in their own
  sqlite-vec table with memory_embeddings as the compatibility fallback.
  `diagnose --repair-vec-working` reports coverage and backfills missing rows.
- **Synchronous reindex** — `mnemosyne reindex` rebuilds all vectors (working,
  episodic, facts) after embedding model or dimension change. Auto-backup.
- **fact_recall ranking** now scores by query relevance (not stored confidence),
  returning full triples as content. Opt-in via `MNEMOSYNE_FACT_RECALL_ENABLED`.
- **Smart plugin upgrade** — `mnemosyne-hermes upgrade` auto-detects install
  method (pipx / uv-tool / pip), shows version comparison, upgrades, and
  re-registers the plugin.
- **Plugin cleanup** — `mnemosyne-hermes cleanup` removes plugin, old dirs,
  and resets config. `--dry-run` safe.
- **CLI version** no longer depends on `__author__` (removed in v3.7.0).

### User action required

```bash
pip install --upgrade mnemosyne-memory
```

For the new plugin features:

```bash
pipx install "mnemosyne-hermes[all]"
```

For existing symlink installs only:

```bash
mnemosyne-hermes install --force
```

### New environment variables

| Variable | Default | What it does |
|---|---|---|
| `MNEMOSYNE_FACT_RECALL_ENABLED` | not set | Enables query-relevance-scored fact recall |
| `MNEMOSYNE_SYNC_SERVER_PORT` | 8765 | Sync server listening port |
| `MNEMOSYNE_SYNC_SERVER_KEY` | (none) | Encryption key for sync payloads |

### Schema changes

Adds `memory_events` table (sync event log) and `sync_meta` table (device
identity, cursors). Both created lazily. No destructive migrations.

### Rollback to v3.7.0

```bash
pip install 'mnemosyne-memory==3.7.0'
```

The sync tables persist but are ignored by v3.7.0 code. vec_working table
persists but v3.7.0 memory_embeddings fallback reads it as a normal table —
no collision.

---

## Upgrading to v3.9.0 — Synchronous Reindex + Diagnose Tool

Released 2026-06-18. Adds `mnemosyne reindex` for rebuilding vectors after
a model or dimension change, and `mnemosyne diagnose` for deployment health
checks.

### What changed

- New `mnemosyne reindex` command — synchronous vector rebuild across all
  memory tables (working, episodic, facts). Replaces the old incremental
  approach that could leave stale vectors.
- `mnemosyne diagnose` now reports `vec_working` migration coverage so you
  can confirm working-memory vector search is active.

### User action

```bash
pip install --upgrade mnemosyne-memory==3.9.0
```

No manual migration needed. The upgrade adds `memoria_` tables on first init.

---

## Upgrading to v3.10.0 — L3 Persona Layer

Released 2026-06-18. Adds always-on behavioral rule layer that survives past
the working-memory TTL. New `memoria_persona` table with four tiers:
permanent, long-term, working, ephemeral.

### What changed

- **L3 persona facts** (`memoria_persona` table) — behavioral rules extracted
  from conversation, persisted across sessions, injected into system prompt.
  Four confidence tiers with automatic reinforcement and decay.
- **5 new Hermes tools**: `mnemosyne_persona_list`, `mnemosyne_persona_add`,
  `mnemosyne_persona_reinforce`, `mnemosyne_persona_demote`,
  `mnemosyne_persona_remove`.
- **Auto-injection**: persona.md is appended to the system prompt when
  triggered by session-start, tool-call, recall, or periodic refresh.

### User action

```bash
pip install --upgrade mnemosyne-memory==3.10.0
```

No manual migration. Persona extraction starts automatically. To disable
persona auto-injection, set `memory.mnemosyne.persona_inject: false` in
`config.yaml`.

---

## Upgrading to v3.10.1 — Security Fix (Sync JWT Bypass)

Released 2026-06-22. **Security release** — fixes CVE GHSA-xcw4-53cc-hv32
(CVSS 9.1). The sync server's JWT verification was missing signature
validation.

### What changed

- HMAC-SHA256 signature verification added to sync server auth
- Strict `alg: HS256` allowlist (rejects `none`, RS256, etc.)
- Constant-time signature comparison via `hmac.compare_digest`

### User action

```bash
pip install --upgrade mnemosyne-memory==3.10.1
```

If you operate a sync server with network exposure, upgrade immediately.
If you cannot upgrade, restrict network access to the sync endpoint.

---

## Upgrading to v3.11.0 — Automated Sleep Model Refresh

Released 2026-06-30. Adds LLM-assisted canonical model refresh during
sleep, recall diagnostics + task progress tools, tool whitelist, wrapper
install mode, and several fixes.

### What changed

- **Automated sleep model refresh** — during `sleep()`, Mnemosyne asks the
  LLM for structured candidate updates to canonical model slots (user model,
  workflow model, project model). Validated candidates are auto-applied or
  auto-rejected by policy. New `mnemosyne_model_refresh` diagnostic tool.
- **Recall diagnostics** — `mnemosyne_recall_diagnostics` exposes per-row
  scoring breakdowns (weights, scores, signal contributions).
- **Task progress** — `mnemosyne_task_progress` tracks multi-step task state
  across sessions.
- **Tool whitelist** — restrict exposed tools via
  `memory.mnemosyne.tools` config key. Unknown names raise a clear error.
- **Wrapper install mode** — `mnemosyne-hermes install --mode wrapper`
  for read-only / Docker deployments.
- **`MNEMOSYNE_LLM_TIMEOUT`** — configurable HTTP timeout for remote LLM
  calls (default 60s).
- **`mnemosyne backup`** now works with sqlite-vec databases.
- **CLI bank-aware** under `profile_isolation` — CLI commands now read
  the correct profile bank.

### User action

```bash
pip install --upgrade mnemosyne-memory==3.11.0
```

No manual migration. Sync role default changed to `["user"]` — if you
want assistant-turn autosave, set `memory.mnemosyne.sync_roles:
["user", "assistant"]` in `config.yaml`.

## Upgrading to v3.6.0 — Canonical Facts + API Embedding Fallback

Released 2026-06-10. Minor release with canonical facts, holographic importer,
API embedding fallback chain, host LLM registration in CLI, and several fixes.

### What changed

- **CanonicalStore** — new `canonical_facts` table (lazy-created, no new dependency) giving long-running personas an identity layer where each `(owner_id, category, name)` slot holds exactly one current value. Two new tools: `mnemosyne_remember_canonical` and `mnemosyne_recall_canonical`. Total tool count: 23 → 25.
- **Holographic Memory importer** — `hermes mnemosyne import --from holographic` now operational. Reads Hermes' SQLite-based holographic memory plugin. No API key needed.
- **API embedding fallback** — `embed()` now falls through to local fastembed when the API call fails. Set `MNEMOSYNE_EMBEDDING_FALLBACK_MODEL` to choose your fallback (default: bge-small-en-v1.5). No configuration needed for the default.
- **Embeddings now unconditional** — `fastembed` + `sqlite-vec` are hard dependencies (previously opt-in via `[embeddings]` extra). If your environment blocks `pip install --upgrade mnemosyne-memory`, check system packages.
- **Hermes host LLM in CLI** — `hermes mnemosyne sleep` now properly respects `MNEMOSYNE_HOST_LLM_ENABLED=true`.
- **Per-entity identity in prefetch** — the agent always gets your stable self-descriptors without explicit identity search.

### User action required

```bash
pip install --upgrade mnemosyne-memory
```

That's it. The `canonical_facts` table is created lazily on first init — no migration script needed. The holographic importer works out of the box after upgrade.

### Rollback to v3.4.0

```bash
pip install mnemosyne-memory==3.4.0
```

Note: the canonical_facts table persists across downgrades (it's just a SQLite table; old code ignores it). Re-`pip install --upgrade` when ready.

---

---

## Upgrading to v3.1.2 — Strict Fact Matching + Entity Prefix Guard

Released 2026-05-28. Pure bug fix release — no schema changes, no new features.

### What changed

1. **Multi-token relevance scoring fixed.** Pre-v3.1.2...

1. **Strict fact matching is now the default.** The old permissive path matched any query word against any stored fact, pulling in unrelated memories with a false +20% score boost. Set `MNEMOSYNE_LENIENT_FACT_MATCH=1` to opt back in.

2. **Entity prefix guard added.** The prefix match in entity similarity now requires a minimum 30% length ratio. Short query prefixes like "her" no longer match "Hermes" at 0.828.

3. **Single-token strict matching fixed.** Queries like "hermes", "python", "react" (single 5+ char tokens) now pass the strict fact matcher. Previously required 8+ chars with structural characters.

### User action

```bash
pip install --upgrade mnemosyne-memory
hermes gateway restart
```

Zero manual migration needed. If you relied on the lenient fact matching, set:
```bash
export MNEMOSYNE_LENIENT_FACT_MATCH=1
```

### Known issues

Non-strict recall is still the default for **entity** and **fact** paths (`MNEMOSYNE_ENHANCED_RECALL=0`). Strict mode only applies to the built-in fact matcher (`_find_memories_by_fact`). The entity/fact recall paths also don't propagate `from_date`/`to_date`/`veracity` filters — tracked as a low-priority follow-up.

## Upgrading from v2.7 to v3.0.0

This is the most common jump for existing users. It covers 3 releases
worth of changes. Read the relevant sections in order:

1. **v2.7 → v2.8** — E6 TripleStore split (schema migration)
2. **v2.8 → v2.9** — MCP SDK 1.x compatibility (code only)
3. **v2.9 → v3.0** — MEMORIA architecture (new tables)

### Step-by-step

```bash
# 1. Update the package
pip install --upgrade mnemosyne-memory

# (If PEP 668 blocks you, use --break-system-packages)
# pip install --upgrade mnemosyne-memory --break-system-packages

# 2. Restart Hermes to load the new plugin/tools
hermes gateway restart

# 3. Verify
hermes mnemosyne version
# Should show: 3.0.0

hermes mnemosyne stats --global
# Check memory count is preserved

hermes tools list | grep mnemosyne
# Should show 17+ tools
```

**What happens to your data on first run:**

- v2.7 databases get auto-migrated by E6 on first BeamMemory init.
  Backup written to `{db}.pre_e6_backup`.
- v3.0 creates 5 new MEMORIA tables (`memoria_facts`,
  `memoria_timelines`, `memoria_instructions`, `memoria_preferences`,
  `memoria_kg`) via `CREATE TABLE IF NOT EXISTS`. Existing tables are
  untouched.
- All existing memories, triples, embeddings remain intact.

**If anything goes wrong:**

```bash
# Restore pre-E6 backup
cp ~/.hermes/mnemosyne/data/mnemosyne.db.pre_e6_backup \
   ~/.hermes/mnemosyne/data/mnemosyne.db

# Roll back to v2.7
pip install 'mnemosyne-memory==2.7.0'
hermes gateway restart
```

---

## Version-by-Version Details

### Upgrading to v3.0.0 (MEMORIA Architecture)

The MEMORIA release introduces structured fact extraction and retrieval.

**Schema changes (all auto-created):**

5 new tables: `memoria_facts`, `memoria_timelines`,
`memoria_instructions`, `memoria_preferences`, `memoria_kg`.

All use `CREATE TABLE IF NOT EXISTS` — zero risk to existing data.

**New environment variables:**

| Variable | Default | What it does |
|---|---|---|
| `MNEMOSYNE_STRICT_FACT_MATCH` | not set | Enables token-based conservative fact matching |
| `MNEMOSYNE_PROACTIVE_LINKING` | not set | Enables zero-LLM graph edge creation at ingest |
| `MNEMOSYNE_MEMORIA_MODEL` | `gemini-2.0-flash-lite` | LLM model used for MEMORIA extraction |

**What to verify after update:**

```bash
# Check MEMORIA tables exist
python3 -c "
import sqlite3, pathlib
db = pathlib.Path.home() / '.hermes' / 'mnemosyne' / 'data' / 'mnemosyne.db'
conn = sqlite3.connect(str(db))
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'memoria_%'\").fetchall()
print('MEMORIA tables:', [t[0] for t in tables])
conn.close()
"

# Expected output:
# MEMORIA tables: ['memoria_facts', 'memoria_timelines',
#                   'memoria_instructions', 'memoria_preferences',
#                   'memoria_kg']
```

**Rollback:**

```bash
pip install 'mnemosyne-memory==2.9.0'
hermes gateway restart
```

The MEMORIA tables remain in the database but are ignored by older code.
They are harmless. If you want them gone, export, delete DB, re-import.

---

### Upgrading to v3.1.0 (Shared Surface & Multilingual MEMORIA)

The v3.1.0 release adds shared surface memory, multilingual MEMORIA, custom embedding endpoints, and many fixes.

**New capabilities:**

- **Shared surface memory.** Cross-agent shared persistence via `mnemosyne_shared_*` tools. Each agent gets an isolated shared surface. Activate with `hermes memory` surface commands.
- **Multilingual MEMORIA.** Language auto-detection for German, Russian, and Chinese. Extraction now applies language-specific patterns based on detected input language.
- **Custom embedding endpoints.** Configure any OpenAI-compatible embedding provider via `OPENROUTER_BASE_URL` (set to your own server URL). Jina model dimensions auto-detected. Set `MNEMOSYNE_EMBEDDINGS_VIA_API=true` if you want to use OpenRouter-hosted embedding models specifically.
- **Deterministic `get(id)`.** Direct memory retrieval by ID — no vector search, no ranking. Call `mnemosyne.get(memory_id)` for exact lookup.

**New environment variables:**

| Variable | Default | What it does |
|---|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Override the embedding API provider URL |
| `MNEMOSYNE_EMBEDDINGS_VIA_API` | not set | Set to `true` to route all embedding models through the API |

**Fixes included:**

- sqlite-vec int8 search now uses `AND k=N` syntax (was silently wrong with `LIMIT`)
- Hermes plugin: all 6 tool schemas now include `bank` parameter for multi-bank operation
- sqlite-vec extension loaded before vector operations (fixes `vec_distance_cosine` crashes)
- Timezone normalization in temporal recall (fixes off-by-hour windowing)
- Working memory vectors generated and persisted on every `remember()` call
- MEMORIA regex dedup and language pattern fixes across German, Russian, Chinese
- Config string booleans properly coerced from YAML

**What to verify after update:**

```bash
pip install --upgrade mnemosyne-memory
hermes gateway restart

# Verify version
python3 -c "from mnemosyne import __version__; print(__version__)"
# Expected: 3.1.0
```

**Rollback:**

```bash
pip install 'mnemosyne-memory==3.0.0'
hermes gateway restart
```

Shared surface tables remain in the database but are ignored by v3.0.0.

---



MCP server transport updated for SDK v1.x. Code-only change — no schema
migration needed.

```bash
pip install --upgrade mnemosyne-memory
hermes gateway restart
```

Only affects you if you use the MCP server directly (not via Hermes).

---

### Upgrading to v2.8.0 (E6 TripleStore Split + CompressionPlugin)

This release splits the `triples` table into two purpose-specific tables
and introduces optional content compression.

**Critical schema change: E6 TripleStore Split**

Before v2.8, all triples lived in one `triples` table with auto-invalidation
semantics. This silently destroyed multi-valued annotations (entities,
facts) whenever a memory had more than one.

After v2.8:
- `triples` — retains current-truth facts (superseding behavior)
- `annotations` — append-only, hosts `mentions`, `fact`, `occurred_on`,
  `has_source` (multi-valued by design)

**Auto-migration (default):**

On first `BeamMemory` init, annotation-flavored rows are moved from
`triples` to `annotations`. A backup is created at `{db}.pre_e6_backup`.

```bash
pip install --upgrade mnemosyne-memory
hermes gateway restart
# Check logs for:
#   "E6: auto-migrated N annotation rows from triples -> annotations."
```

**Manual migration (explicit control):**

```bash
export MNEMOSYNE_AUTO_MIGRATE=0
hermes gateway restart
# BeamMemory logs a WARNING with pending row count

# Preview
python scripts/migrate_triplestore_split.py --dry-run

# Apply
python scripts/migrate_triplestore_split.py
```

**New optional feature: CompressionPlugin**

Disabled by default. Enable via config or env var:

```bash
export MNEMOSYNE_USE_CAVEMAN=1
```

Or in code:
```python
from mnemosyne.core.config import MnemosyneConfig
MnemosyneConfig.compression.enabled = True
```

**What to verify after update:**

```bash
# Check annotations table exists
python3 -c "
import sqlite3, pathlib
db = pathlib.Path.home() / '.hermes' / 'mnemosyne' / 'data' / 'mnemosyne.db'
conn = sqlite3.connect(str(db))
count = conn.execute('SELECT COUNT(*) FROM annotations').fetchone()[0]
print(f'Annotations table has {count} rows')
conn.close()
"
```

---

## How to Confirm Schema Changes

Wondering if an update changed the SQLite schema? Here's how to check:

### Before updating (baseline)

```bash
# Dump the current schema
python3 -c "
import sqlite3, pathlib
db = pathlib.Path.home() / '.hermes' / 'mnemosyne' / 'data' / 'mnemosyne.db'
conn = sqlite3.connect(str(db))
schema = conn.execute(\"SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name\").fetchall()
for row in schema:
    print(row[0] + ';')
conn.close()
" > ~/mnemosyne_schema_baseline.txt
```

### After updating (compare)

```bash
# Dump the new schema
python3 -c "
import sqlite3, pathlib
db = pathlib.Path.home() / '.hermes' / 'mnemosyne' / 'data' / 'mnemosyne.db'
conn = sqlite3.connect(str(db))
schema = conn.execute(\"SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name\").fetchall()
for row in schema:
    print(row[0] + ';')
conn.close()
" > ~/mnemosyne_schema_new.txt

# Compare
diff ~/mnemosyne_schema_baseline.txt ~/mnemosyne_schema_new.txt
```

New tables and columns appear as additions. Missing tables would appear
as removals. Mnemosyne uses `CREATE TABLE IF NOT EXISTS` and
`ALTER TABLE ADD COLUMN` with existence checks, so schema changes are
additive — no destructive migrations.

### Quick check: what version was my DB created by?

```bash
python3 -c "
import sqlite3, pathlib
db = pathlib.Path.home() / '.hermes' / 'mnemosyne' / 'data' / 'mnemosyne.db'
conn = sqlite3.connect(str(db))
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\").fetchall()
names = [t[0] for t in tables]
if 'memoria_facts' in names:
    print('DB schema: v3.0+ (MEMORIA)')
elif 'annotations' in names:
    print('DB schema: v2.8+ (E6 TripleStore split)')
elif 'episodic_memory' in names:
    print('DB schema: v2.0+ (BEAM)')
else:
    print('DB schema: v1.x (legacy)')
conn.close()
"
```

---

## By Install Path

### Option A: PyPI (recommended for users)

```bash
pip install --upgrade mnemosyne-memory
hermes gateway restart
```

To verify the new version:

```bash
hermes mnemosyne version
hermes mnemosyne stats --global
hermes memory status
```

**Note:** UPDATING.md is included in the sdist and wheel package, but
PyPI does not serve individual files at browsable URLs. The file is
available at the GitHub repo:

  https://github.com/AxDSan/mnemosyne/blob/main/UPDATING.md

### Option B: Source install (`pip install -e .`)

For most updates, only `git pull` is required:

```bash
cd mnemosyne
git pull
hermes gateway restart
```

**Re-run `pip install -e .` only when:**
- `setup.py` or `pyproject.toml` added new dependencies
- New `entry_points` or console scripts were added
- Package metadata changed

```bash
git pull
pip install -e ".[all,dev]"
hermes gateway restart
```

**Re-run the installer only when** `mnemosyne/install.py` or the symlink
logic changed:

```bash
git pull
python -m mnemosyne.install
hermes gateway restart
```

### Option C: Hermes MemoryProvider only (deploy script)

This path symlinks `~/.hermes/plugins/mnemosyne` directly into the repo:

```bash
cd mnemosyne
git pull
hermes gateway restart
```

No `pip install` needed — nothing is installed into a Python environment.

---

## Database Migrations

Mnemosyne uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF
NOT EXISTS`, so most schema changes upgrade automatically.

Run a migration script only when:
- The CHANGELOG explicitly mentions a database schema change
- You are upgrading from a pre-2.0 version
- You see errors about missing columns or tables

### Available migration scripts

| Script | What it does |
|---|---|
| `scripts/migrate_from_legacy.py` | Migrates from v1.x ephemeral databases to the canonical v2+ path. Idempotent. |
| `scripts/migrate_triplestore_split.py` | Manual E6 migration (v2.8). Only needed if you set `MNEMOSYNE_AUTO_MIGRATE=0`. Idempotent. |

```bash
# Preview first
python scripts/migrate_triplestore_split.py --dry-run

# Apply
python scripts/migrate_triplestore_split.py
```

All migration scripts are idempotent — safe to run multiple times.

---

## Rollback

### Roll back to a specific version

```bash
# Pin to a known good version
pip install 'mnemosyne-memory==2.7.0'

# Or from source
cd mnemosyne
git checkout v2.7.0
pip install -e .

# Restart Hermes
hermes gateway restart
```

### Restore a database backup

If you have a DB backup from before the update:

```bash
# E6 auto-backup
cp ~/.hermes/mnemosyne/data/mnemosyne.db.pre_e6_backup \
   ~/.hermes/mnemosyne/data/mnemosyne.db

# Or any custom backup
cp ~/backups/mnemosyne_20260101.db \
   ~/.hermes/mnemosyne/data/mnemosyne.db
```

### Export, nuke, re-import

```bash
# Export current data
hermes mnemosyne export --output ~/backup.json

# Delete the database entirely
rm ~/.hermes/mnemosyne/data/mnemosyne.db

# Start fresh with old version
pip install 'mnemosyne-memory==2.7.0'
hermes gateway restart

# Re-import
hermes mnemosyne import --input ~/backup.json
```

---

## Verifying an Update

```bash
# Version check
hermes mnemosyne version

# Stats (memories preserved?)
hermes mnemosyne stats --global

# Tools registered?
hermes tools list | grep mnemosyne

# Memory available?
hermes memory status

# Schema version (for the curious)
python3 -c "
import sqlite3, pathlib
db = pathlib.Path.home() / '.hermes' / 'mnemosyne' / 'data' / 'mnemosyne.db'
conn = sqlite3.connect(str(db))
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\").fetchall()
print(f'{len(tables)} tables: {[t[0] for t in tables]}')
conn.close()
"
```

---

## Troubleshooting

### "Command not found" after update

Entry points are registered at install time, not at runtime.
Re-run the install:

```bash
pip install -e .
```

### "No module named mnemosyne" after update

Your virtual environment may have been deactivated or the editable
install broke. Re-install:

```bash
pip install -e .
```

### Plugin changes not taking effect

Hermes caches plugins at startup. You **must** restart:

```bash
hermes gateway restart
```

### "externally-managed-environment" errors

You're on Debian 13+ / Ubuntu 24.04+ (PEP 668). See the
[PEP 668 section](#pep-668-externally-managed-environment-on-debian--ubuntu).

### Database errors after schema change

If you see errors about missing columns or tables, run a migration:

```bash
# Try auto-repair by restarting
hermes gateway restart

# If that fails, run the legacy migration
python scripts/migrate_from_legacy.py

# If errors persist, export, delete, re-import
hermes mnemosyne export --output ~/backup.json
rm ~/.hermes/mnemosyne/data/mnemosyne.db
hermes mnemosyne import --input ~/backup.json
```

### "UPDATING.md" URL on PyPI returns 404

PyPI does not serve individual package files at browsable URLs.
The correct URL for the latest version is:

  https://github.com/AxDSan/mnemosyne/blob/main/UPDATING.md

The file IS included in the sdist and wheel — `pip show -f
mnemosyne-memory` will confirm it ships.

### Memory count dropped after update

The E6 migration moves annotation rows from `triples` to a new
`annotations` table. This does not delete memories. Check:

```bash
hermes mnemosyne stats --global
```

If counts look wrong, check the E6 migration log:

```bash
grep -i "auto-migrated\|E6" ~/.hermes/logs/gateway.log
```
