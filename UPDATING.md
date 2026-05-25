# Updating Mnemosyne

Covers all upgrade paths: v2.7 → latest, source installs, PyPI installs,
and systems with Python's `externally-managed-environment` (PEP 668).

If you're on **v2.7** and want the latest (v3.0.0), jump to
[Upgrading from v2.7 to v3.0.0](#upgrading-from-v27-to-v300).

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

### Upgrading to v2.9.0 (MCP SDK Compatibility)

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
