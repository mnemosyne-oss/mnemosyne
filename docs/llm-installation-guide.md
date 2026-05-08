# Mnemosyne Installation Guide for LLM Agents

> **Target audience:** AI agents (LLMs) that need to install, configure, and verify Mnemosyne for a user.
> **When to use:** A user asks you to install Mnemosyne, set up memory for their AI agent, or integrate Mnemosyne with Hermes.

---

## Quick Decision: Which Install Path?

| User has... | Use |
|---|---|
| Hermes Agent already installed | **Path A: One-liner deploy** (fastest, 1 command) |
| Hermes Agent + wants PyPI package | **Path B: pip install + register** |
| No Hermes, just wants the library | **Path C: pip install (standalone)** |
| Wants to contribute or develop | **Path D: Source install** |

---

## Path A: One-Liner Deploy (Hermes MemoryProvider)

The fastest way to integrate Mnemosyne as Hermes's memory backend. Creates a symlink — no pip needed, no venv needed.

```bash
curl -sSL https://raw.githubusercontent.com/AxDSan/mnemosyne/main/deploy_hermes_provider.sh | bash
```

**What this does:**
1. Symlinks `hermes_memory_provider/` → `~/.hermes/plugins/mnemosyne/`
2. Tells you to set `memory.provider: mnemosyne` in config

**After running, configure Hermes:**

```bash
hermes config set memory.provider mnemosyne
```

**Or edit `~/.hermes/config.yaml` directly:**

```yaml
memory:
  provider: mnemosyne
```

**IMPORTANT:** Also add `mnemosyne` to `plugins.enabled` in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - mnemosyne
```

**Verify:**

```bash
hermes gateway restart
hermes memory status
hermes mnemosyne stats
```

---

## Path B: pip install + Register with Hermes

Install the PyPI package, then register it as Hermes's memory provider.

### Step 1: Install the package

```bash
pip install mnemosyne-memory
```

With embeddings (recommended — enables vector search):

```bash
pip install mnemosyne-memory[embeddings]
```

With ALL optional features:

```bash
pip install mnemosyne-memory[all]
```

**Ubuntu 24.04 / Debian 12 PEP 668 workaround:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mnemosyne-memory[all]
```

### Step 2: Register with Hermes

```bash
python -m mnemosyne.install
```

This creates `~/.hermes/plugins/mnemosyne/` and sets `memory.provider: mnemosyne` in config.

### Step 3: Verify

```bash
hermes gateway restart
hermes memory status        # Should show: Provider: mnemosyne
hermes mnemosyne stats      # Working + episodic counts
hermes tools list | grep mnemosyne
```

---

## Path C: Standalone (no Hermes)

Just the library, usable from any Python code.

```bash
pip install mnemosyne-memory[all]
```

**Usage:**

```python
from mnemosyne import remember, recall, get_stats

remember("User prefers dark mode", importance=0.9, source="preference")
results = recall("interface preferences")
print(results)
```

---

## Path D: From Source (Development)

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e ".[all,dev]"
```

Then register with Hermes:

```bash
python -m mnemosyne.install
hermes gateway restart
```

---

## Post-Install: Verify Everything Works

Run these checks in order. Stop if any fails.

### 1. Provider is registered

```bash
hermes memory status
```

Expected: `Provider: mnemosyne` with `is_available: true`

### 2. Tools are loaded

```bash
hermes tools list | grep mnemosyne
```

Expected: 14 tools (remember, recall, stats, sleep, triple_add, triple_query, scratchpad_write, scratchpad_read, scratchpad_clear, invalidate, export, update, forget, import)

### 3. Memory operations work

```bash
hermes mnemosyne stats
```

Expected: Working and episodic memory counts (numbers, even if 0).

### 4. Store and recall a test memory

```bash
python3 -c "
from mnemosyne import remember, recall
mid = remember('TEST: install verification', importance=0.5, source='test')
print(f'Stored: {mid}')
results = recall('install verification')
print(f'Found: {len(results)} results')
"
```

---

## Configuration Reference

### Required config

In `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mnemosyne

plugins:
  enabled:
    - mnemosyne
```

### Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_VEC_TYPE` | `float32` | Vector compression: `int8` (4x smaller) or `bit` (32x smaller) |
| `MNEMOSYNE_LOG_TOOLS` | `0` | Set to `1` to auto-log tool calls as memories |
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data/` | Custom data directory |

---

## Updating

```bash
# PyPI users
pip install --upgrade mnemosyne-memory
hermes gateway restart

# Source users
cd mnemosyne && git pull
hermes gateway restart

# Re-run pip install -e only if setup.py or pyproject.toml changed
pip install -e ".[all,dev]"
```

---

## Uninstalling

```bash
python -m mnemosyne.install --uninstall
hermes config set memory.provider null
hermes gateway restart
```

To also remove the plugin from config, delete `mnemosyne` from `plugins.enabled` in `~/.hermes/config.yaml`.

---

## Troubleshooting for Agents

### "Provider not found" after install

```bash
# Check the symlink exists
ls -la ~/.hermes/plugins/mnemosyne

# If missing, recreate it
python -m mnemosyne.install
```

### "No module named mnemosyne"

The package isn't installed in Hermes's Python environment. Either:
- Activate the correct venv and reinstall
- Use Path A (symlink deploy) instead — it doesn't need pip

### Tools not showing up

```bash
# Check plugins are loaded
hermes plugins list

# If mnemosyne isn't listed, check config.yaml plugins.enabled
grep -A5 "plugins:" ~/.hermes/config.yaml

# Restart gateway after any config change
hermes gateway restart
```

### FTS5 / vector search errors

```bash
# Install embeddings support
pip install fastembed>=0.3.0

# Verify
python3 -c "from fastembed import TextEmbedding; print('OK')"

# Restart
hermes gateway restart
```

### Session ID mismatch warning

If the `hermes_plugin` hook uses a different session ID than the MemoryProvider, memories stored in one path won't surface in the other. The fix is in the hook code — ensure `_on_pre_llm_call` uses `f"hermes_{session_id}"` as the session ID, matching the MemoryProvider convention. This is already fixed in the latest source.

---

## Agent-Specific Notes

### After installing Mnemosyne for a user

1. Tell the user to restart Hermes: `hermes gateway restart`
2. Verify with: `hermes memory status`
3. The user should notice memory persistence across sessions immediately
4. If the user has existing legacy memory (from the built-in provider), those memories are mirrored to Mnemosyne on write — they won't be lost

### When to use mnemosyne_remember vs the legacy memory tool

- **ALWAYS use `mnemosyne_remember`** for durable facts, preferences, and insights
- The legacy `memory` tool is deprecated for durable storage
- Mnemosyne supports importance scoring (0.0-1.0), global scope, expiry dates, and entity extraction — features the legacy tool doesn't have

### Memory survives gateway restarts, machine reboots, and Fly.io VM recycles

By default, the main database lives at `~/.hermes/mnemosyne/data/mnemosyne.db`; named banks live under `~/.hermes/mnemosyne/data/banks/<name>/`. No Docker, no PostgreSQL, no required network calls.
