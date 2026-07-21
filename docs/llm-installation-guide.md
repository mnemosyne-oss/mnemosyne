# Mnemosyne Installation Guide for LLM Agents

> **Target audience:** AI agents (LLMs) that need to install, configure, and verify Mnemosyne for a user.
> **When to use:** A user asks you to install Mnemosyne, set up memory for their AI agent, or integrate Mnemosyne with Hermes.

---

## Quick Decision: Which Install Path?

| User has... | Use |
|---|---|
| Hermes Agent in a persistent Docker/image deployment | **Path A: persistent side venv + wrapper** — install, configure, then restart |
| Hermes Agent installed locally | **Path B: pip install + register** |
| No Hermes, just wants the library | **Path C: pip install (standalone)** |
| Wants to contribute or develop | **Path D: Source install** |

---

## Path A: Hermes provider install

For a persistent Docker/image install, use a persistent side venv and the wrapper installer. It keeps the plugin directory independent from a rebuildable Hermes venv. Set `HERMES_HOME` to the directory that contains the active Hermes `config.yaml`; `/opt/data` below is the Docker/image example and must be replaced for other deployments:

```bash
export HERMES_HOME=/opt/data
VENV="$HERMES_HOME/.mnemosyne/venv"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install 'mnemosyne-memory[embeddings]' mnemosyne-hermes
"$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python"
hermes config set memory.provider mnemosyne
hermes gateway restart
```

The wrapper installer registers the provider plugin in `$HERMES_HOME/plugins`. Verify the active profile rather than adding a separate `plugins.enabled` entry:

```bash
"$VENV/bin/mnemosyne-hermes" status
hermes memory status
```

These status commands report local provider state only; they do not prove database access, tool execution, or a successful memory round trip. Use the stats and store/recall checks in [Post-Install: Verify Everything Works](#post-install-verify-everything-works) for functional validation.

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
mnemosyne stats             # Working + episodic counts
hermes tools list | grep mnemosyne
```

`hermes memory status` reports local provider state only; it does not prove database access, tool execution, or a successful memory round trip. Confirm functional behavior with `mnemosyne stats` and the store/recall check in [Post-Install: Verify Everything Works](#post-install-verify-everything-works).

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

For Path A, re-export `VENV=/path/to/venv` to the same side venv passed to the wrapper install when verifying from a new shell, then invoke the core CLI as `"$VENV/bin/mnemosyne"` below. Other install paths assume `mnemosyne` is on `PATH`.

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

Expected: Mnemosyne provider tools are listed. The exact tool surface evolves with the installed package versions, so do not use a fixed tool count as the health check.

### 3. Memory operations work

For Path A:

```bash
"$VENV/bin/mnemosyne" stats
```

For other install paths:

```bash
mnemosyne stats
```

Expected: Working and episodic memory counts (numbers, even if 0).

### 4. Store and recall a test memory

```bash
# Path A: use the persistent side venv. Other paths: replace with python3.
"$VENV/bin/python" -c "
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
```

Both Path A's wrapper installer and Path B's `python -m mnemosyne.install` register the plugin; do not add a separate `plugins.enabled` entry for either of those installer paths.

### Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_VEC_TYPE` | `float32` | Vector compression: `int8` (4x smaller) or `bit` (32x smaller) |
| `MNEMOSYNE_LOG_TOOLS` | `0` | Set to `1` to auto-log tool calls as memories |
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data/` | Custom data directory |

---

## Updating

```bash
# Path A: persistent side venv + wrapper
export HERMES_HOME=/opt/data  # Replace with the active Hermes home
"$HERMES_HOME/.mnemosyne/venv/bin/python" -m pip install --upgrade 'mnemosyne-memory[embeddings]' mnemosyne-hermes
hermes gateway restart

# Path B: direct PyPI install
pip install --upgrade mnemosyne-memory
hermes gateway restart

# Path D: source install
cd mnemosyne && git pull
pip install -e ".[all,dev]"
hermes gateway restart
```

---

## Uninstalling

Set `HERMES_HOME` to the active Hermes home before removing a Path A wrapper install.

### Path A: persistent side venv + wrapper

```bash
export HERMES_HOME=/opt/data  # Replace with the active Hermes home
hermes memory off  # Disable the external provider; built-in memory remains active
hermes gateway restart  # Run from a shell outside the gateway process
"$HERMES_HOME/.mnemosyne/venv/bin/mnemosyne-hermes" uninstall
rm -rf "$HERMES_HOME/.mnemosyne/venv"  # Only if this venv was created by Path A
```

### Path B or Path D: pip/source install

```bash
hermes memory off
hermes gateway restart  # Run from a shell outside the gateway process
python -m mnemosyne.install --uninstall
```

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
- Use Path A's persistent side venv + wrapper so the package does not need to live in Hermes's rebuildable environment

### Tools not showing up

```bash
# Check plugins are loaded
hermes plugins list

# For Path A, after setting HERMES_HOME as shown above:
"$HERMES_HOME/.mnemosyne/venv/bin/mnemosyne-hermes" status
hermes memory status

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
