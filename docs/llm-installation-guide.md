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

For a persistent Docker/image install, use a persistent side venv and the wrapper installer. It keeps the plugin directory independent from a rebuildable Hermes venv. The side venv must use the same Python **major/minor** as the running Hermes gateway; do not substitute `python3` from `PATH` or run bare `uv venv`.

Set `HERMES_HOME` to the directory that contains the active Hermes `config.yaml`; `/opt/data` below is the Docker/image example and must be replaced for other deployments. Before creating the side venv, set `HERMES_PYTHON` to the virtual-environment interpreter used by the **actual running Hermes gateway**, verified with the deployment operator or runtime configuration. Keep the venv interpreter path, not its resolved base-Python symlink target. If ownership cannot be determined, **stop**; an executable or a version probe alone does not establish ownership.

The standalone installer's bounded discovery supports validated launcher-based installations and recorded staged runtimes; it does not simply trust an executable sibling of `hermes`. A recorded but unusable staged runtime must not fall back to the checkout's anchor venv. For wrappers, shims, containers, or an otherwise uncertain layout, obtain the verified interpreter from the deployment rather than reproducing discovery with a sibling lookup. The explicit `--python` used below bypasses discovery and selects the side environment the wrapper imports. See [`_find_hermes_python`](../integrations/hermes/src/mnemosyne_hermes/install.py).

```bash
: "${HERMES_PYTHON:?Set this to the deployment-verified Python that starts the running Hermes gateway}"
if [ ! -f "$HERMES_PYTHON" ] || [ ! -x "$HERMES_PYTHON" ]; then
  printf 'Hermes Python is not an executable file: %s\n' "$HERMES_PYTHON" >&2
  exit 1
fi
"$HERMES_PYTHON" -c 'import sys; raise SystemExit(sys.prefix == sys.base_prefix)' || {
  printf 'Hermes Python is not a virtual-environment runtime: %s\n' "$HERMES_PYTHON" >&2
  exit 1
}
"$HERMES_PYTHON" --version || exit 1
```

Then create the side venv with that exact interpreter. Stop on the first failure;
do not register or restart a partially installed provider:

```bash
set -e
export HERMES_HOME=/opt/data
VENV="$HERMES_HOME/.mnemosyne/venv"
if [ -e "$VENV" ] || [ -L "$VENV" ]; then
  printf 'Refusing to create side venv at existing path: %s\n' "$VENV" >&2
  exit 1
fi
"$HERMES_PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip --version
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install 'mnemosyne-memory[embeddings]' mnemosyne-hermes
"$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python"
hermes config set memory.provider mnemosyne
```

If venv creation or the pip probe fails, stop. After preserving any partial target and choosing a new, nonexistent persistent path for `VENV`, a seeded fallback is `uv venv --seed --python "$HERMES_PYTHON" "$VENV"`. Use the same verified interpreter, rerun the pip probe and install commands only after success, and never silently overwrite an existing environment. For the native Windows equivalent, see [Windows persistent side-venv wrapper](hermes-integration.md#windows-persistent-side-venv-wrapper).

For Docker or Compose, restart the actual deployment service using its deployment tooling. Do not substitute `hermes gateway restart` inside the container.

The wrapper installer registers the provider plugin in `$HERMES_HOME/plugins`. Verify the active profile rather than adding a separate `plugins.enabled` entry:

```bash
"$VENV/bin/mnemosyne-hermes" status
hermes memory status
```

These status commands report local provider state only; they do not prove database access, tool execution, or a successful memory round trip. Use the stats and store/recall checks in [Post-Install: Verify Everything Works](#post-install-verify-everything-works) for functional validation.

---

## Path B: pip install + Register with Hermes

Install the core package, then register it with the local Hermes installation.
The compatibility installer now delegates to the standalone provider, so also
install `mnemosyne-hermes` in the activated environment before Step 2.

This provider requires the core embeddings extra even if the core was initially
installed without extras. Unlike Path A, the local route may bootstrap packages
into Hermes' mutable runtime; it does not preserve them across its replacement.

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
pip install mnemosyne-hermes
python -m mnemosyne.install
```

The core command migrates recognized legacy links, delegates to the standalone
provider, sets `memory.provider: mnemosyne`, and verifies status. It fails if the
standalone provider cannot be found; it does not recreate the old
`hermes_memory_provider` link. The standalone installer defaults to symlink mode
on Linux/macOS/WSL and wrapper mode on native Windows. For explicit interpreter
selection, use `mnemosyne-hermes install --python` with the verified Hermes
runtime, as described in the [canonical local install guide](hermes-integration.md#step-2-link-the-plugin-in-a-local-mutable-environment).
See also the [core compatibility entry point](../mnemosyne/install.py).

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
git clone https://github.com/mnemosyne-oss/mnemosyne.git
cd mnemosyne
pip install -e ".[all,dev]"
pip install -e integrations/hermes
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
VENV="$HERMES_HOME/.mnemosyne/venv"
"$VENV/bin/python" -m pip install --upgrade 'mnemosyne-memory[embeddings]' mnemosyne-hermes
"$VENV/bin/mnemosyne-hermes" install --mode wrapper --force --python "$VENV/bin/python"
# For Docker or Compose, restart the actual deployment service with its deployment tooling.
```

The forced install regenerates the wrapper plugin files and does not modify the Mnemosyne database. Use `--force` only after confirming the target is the Mnemosyne link or wrapper you intend to replace. If Hermes' Python major/minor changed, create a new matching side venv using the verified runtime before refreshing the wrapper; do not keep using incompatible packages.

```bash
# Path B: direct PyPI install
pip install --upgrade mnemosyne-memory mnemosyne-hermes
python -m mnemosyne.install
hermes gateway restart

# Path D: source install
cd mnemosyne && git pull
pip install -e ".[all,dev]"
pip install -e integrations/hermes
python -m mnemosyne.install
hermes gateway restart
```

---

## Uninstalling

Set `HERMES_HOME` to the active Hermes home before removing a Path A wrapper install.

### Path A: persistent side venv + wrapper

```bash
export HERMES_HOME=/opt/data  # Replace with the active Hermes home
hermes memory off  # Disable the external provider; built-in memory remains active
"$HERMES_HOME/.mnemosyne/venv/bin/mnemosyne-hermes" uninstall
rm -rf "$HERMES_HOME/.mnemosyne/venv"  # Only if this venv was created by Path A
```

For Docker or Compose, restart the actual deployment service with its deployment tooling after removing the wrapper.

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
# Check the plugin target (symlink or wrapper directory)
ls -la ~/.hermes/plugins/mnemosyne

# Local Path B/D only: delegate registration to the standalone provider
python -m mnemosyne.install
```

For Path A, use the verified compatible side venv and wrapper install/status
commands above, not the local compatibility entry point.

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
- The legacy `memory` tool is deprecated for durable storage of bulk facts; small always-injected memory files (binding rules, time-sensitive notes, short pointers) remain a supported complementary tier — this deprecation concerns bulk fact storage, which should go through `mnemosyne_remember`
- Mnemosyne supports importance scoring (0.0-1.0), global scope, expiry dates, and entity extraction — features the legacy tool doesn't have

### Memory survives gateway restarts, machine reboots, and Fly.io VM recycles

By default, the main database lives at `~/.hermes/mnemosyne/data/mnemosyne.db`; named banks live under `~/.hermes/mnemosyne/data/banks/<name>/`. No Docker, no PostgreSQL, no required network calls.

## Optional self-echo suppression

Self-echo suppression is **off by default**. To opt in, set
`MNEMOSYNE_SELF_ECHO_ENABLED=1` in the environment of the process running Hermes,
then start a new provider instance (normally by restarting that process). Unset
it or set it to `0` to disable the feature. This option applies to both Hermes
provider packages; it does not change explicit memory-tool recall.

The integration uses Hermes' existing `on_pre_compress(messages, **kwargs)`
callback automatically; users should not call it manually. Merely enabling the
flag is not sufficient: until the provider has actually observed the callback,
automatic recall remains ordinary recall. Hosts without the callback, or cores
without the optional ledger capability, continue ordinary capture and recall.

Every callback releases **all** previous exclusions, even if compression keeps
some text, does nothing, or fails. Newly captured, unchanged provider-owned rows
can be suppressed only when the sync transcript proves they follow the observed
boundary. Missing or ambiguous transcript evidence means ordinary recall, not
dropped memories. Imported/legacy rows are not retroactively marked or rewritten.
Python 3.10 is supported with a conservative SQLite parameter budget; exceeding
that budget or losing proof also means ordinary recall.

This is best-effort duplicate reduction, **not** exact live-context tracking or
a durable v2 checkpoint guarantee. A provider restart discards suppression state.
An already-returned per-turn prefetch string cannot be rewritten by the plugin;
released memories become available on the next user-turn prefetch. No Hermes
host modification or database migration is required.
