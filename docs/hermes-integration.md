# Hermes Integration

Mnemosyne is designed as a native memory backend for the [Hermes Agent Framework](https://github.com/NousResearch/hermes-agent). It implements the Hermes `MemoryProvider` interface and registers as a plugin.

> **This is the canonical Hermes setup guide.** The README links here for full instructions.

## Install Profile Comparison

| Profile | When to use | RAM | Key tradeoff |
|---------|-------------|-----|-------------|
| `mnemosyne-memory` (core) | Low-resource (Raspberry Pi, 1 GB VPS), or when using a remote embedding API | ~50 MB | No local embeddings. Point `MNEMOSYNE_EMBEDDING_API_URL` to an external endpoint. |
| `mnemosyne-memory[embeddings]` | Mid-range systems with local embedding support | ~800 MB | Adds `fastembed` for local vector generation. Best for single-user desktop agents. |
| `mnemosyne-memory[all]` | Full-featured — local embeddings + local LLM consolidation | ~1.5 GB | Adds `sentence-transformers` + local LLM deps (`ctransformers`). Maximum capability. |
| `mnemosyne-hermes` | Hermes Agent users — always pair with one of the above | Same as base | Wraps core library with plugin manifest + entry points. Run `hermes config set memory.provider mnemosyne` after install. |

**Hardware guidance:** Core alone runs on a Raspberry Pi 4 (4 GB) with ~300 MB free for LLM. `[embeddings]` needs at least 2 GB free RAM. `[all]` recommends 8 GB+.

## Setup

### Step 1: Install

**pip (recommended):**

```bash
pip install mnemosyne-hermes
```

**Debian / Trixie users:** newer Debian releases block bare pip installs. Use a venv:

```bash
python3 -m venv ~/.hermes/hermes-agent/venv
source ~/.hermes/hermes-agent/venv/bin/activate
pip install mnemosyne-hermes
```

**Or from source:**

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e "integrations/hermes[dev]"
```

> **Docker users:** Inside the official Hermes Docker container, the Hermes home directory is `/opt/data/` (the mounted volume), not `~/.hermes/`. For image-based or read-only Hermes venv installs, prefer the persistent wrapper mode so the plugin discovery directory survives image rebuilds:
>
> ```bash
> # Inside the container, pointing at a side/persistent venv that has mnemosyne-hermes installed
> export HERMES_HOME=/opt/data
> /path/to/venv/bin/mnemosyne-hermes install --mode wrapper --python /path/to/venv/bin/python
> /path/to/venv/bin/mnemosyne-hermes status
> hermes config set memory.provider mnemosyne
> hermes gateway restart
> ```
>
> The default `install` mode still creates the historical plugin symlink. Wrapper mode creates a real directory under `$HERMES_HOME/plugins/mnemosyne/` and imports `mnemosyne_hermes` from the selected Python environment. Skip the manual link and activation steps below; the installer handles plugin registration. Verify the active provider after restarting.

### Step 2: Link the plugin

Hermes discovers plugins by scanning a folder on disk, not by reading pip's metadata. Link the installed package into the plugins directory so Hermes can find it:

```bash
# Auto-detect the installed package path and symlink it
mkdir -p ~/.hermes/plugins/mnemosyne
ln -sfn "$(~/.hermes/hermes-agent/venv/bin/python -c 'import pathlib, mnemosyne_hermes; print(pathlib.Path(mnemosyne_hermes.__file__).resolve().parent)')"/* ~/.hermes/plugins/mnemosyne/
```

If you installed in a custom venv (e.g. `~/.hermes-venv`), replace `~/.hermes/hermes-agent/venv/bin/python` with the Python binary inside that venv.

### Step 3: Activate

```bash
hermes config set memory.provider mnemosyne
```

### Step 4: Verify the active provider

Do **not** use `hermes tools disable memory`: that disables the memory toolset, including provider tools. In current Hermes versions, built-in memory and an external provider are separate mechanisms; `hermes memory off` disables the external provider only. Keep existing built-in memory as a rollback/reference point during a transition.

Start a new session or restart the gateway, then verify the active Hermes profile. `hermes memory status` reports local provider registration/state; it is not a connectivity or end-to-end write test:

```bash
hermes memory status
hermes tools list
```

### Step 5: Verify

The commands below assume `mnemosyne` is on `PATH`. For persistent wrapper mode, invoke the core CLI through the side venv (for example, `/path/to/venv/bin/mnemosyne`) or activate that venv first.

```bash
hermes memory status       # Should show "Provider: mnemosyne"
mnemosyne stats            # Working + episodic memory counts
```

## Health checks and repair

Use `mnemosyne doctor` for a bounded, read-only report on one bank/database. It never writes to the inspected database and rejects an output path that would overwrite it; the example writes report files to the current directory, so choose explicit output paths when needed:

```bash
mnemosyne doctor --bank default \
  --format both \
  --json-out mnemosyne-doctor.json \
  --markdown-out mnemosyne-doctor.md
```

`mnemosyne repair` is intentionally narrow and report-gated, not a global cleanup command. Review the Doctor report, select only the candidate you intend to act on, and run a dry run first:

```bash
mnemosyne repair \
  --report mnemosyne-doctor.json \
  --select working_memory:<ID> \
  --dry-run
```

Only add `--apply` after reviewing the report and dry-run output. Repair requires both the report and an explicit selection; do not use it as a substitute for an ownership, retention, or delete-behavior decision.

## How It Works

Mnemosyne hooks into the Hermes agent lifecycle:

| Hook | Behavior |
|---|---|
| `pre_llm_call` | Injects relevant working memory context into the prompt |
| `on_session_start` | Initializes session-scoped memory state |
| `post_tool_call` | Captures tool results as memories (if configured) |

### Tool discovery

The provider tool inventory is version-specific. Confirm the active provider with `hermes memory status`, then inspect the runtime tool surface:

```bash
hermes tools list | grep mnemosyne_
```

The provider exposes memory, knowledge-graph, multi-agent-surface, working-note, and operational tools; use the runtime list rather than a fixed documentation inventory.

## CLI Commands

```bash
mnemosyne stats                         # Show memory statistics
mnemosyne sleep                         # Run consolidation
mnemosyne export backup.json            # Export memories
mnemosyne import backup.json            # Import memories
mnemosyne import-hindsight hindsight-export.json hermes
mnemosyne doctor --bank default --format both
mnemosyne repair --report mnemosyne-doctor.json --select working_memory:<ID> --dry-run
```

## Data Location

By default, data is stored under:

```
~/.hermes/mnemosyne/
├── data/
│   ├── mnemosyne.db              # Main SQLite database (BEAM + legacy)
│   ├── triples.db                # Used by standalone TripleStore()
│   └── banks/<name>/mnemosyne.db # Named memory banks
└── ...
```

This path is chosen because Hermes already persists `~/.hermes/` across sessions (including on ephemeral VMs like Fly.io).

## Auxiliary LLM routing (Codex / OAuth providers)

By default Mnemosyne uses its own LLM config (`MNEMOSYNE_LLM_BASE_URL` /
`MNEMOSYNE_LLM_API_KEY`) or a local GGUF for sleep/consolidation and fact
extraction. Hermes users with OAuth-backed providers like `openai-codex` can
opt into routing those calls through Hermes' authenticated auxiliary client
instead — no extra credentials required.

Set `MNEMOSYNE_HOST_LLM_ENABLED=true` to enable. See
[hermes-llm-integration.md](hermes-llm-integration.md) for the full behavior
model, configuration reference, and session-shutdown semantics.

## Optional MCP Server

For integration with MCP-compatible clients:

```bash
mnemosyne mcp                          # stdio transport
mnemosyne mcp --transport sse --port 8080  # SSE transport
```

Mnemosyne does not currently expose a standalone REST API server.

## Uninstall

### Persistent wrapper / Docker-image install

```bash
export HERMES_HOME=/opt/data  # Replace with the non-default Hermes home used at install time
VENV=/path/to/venv             # The same side venv passed to the wrapper install
hermes memory off  # Disable the external provider; built-in memory remains active
hermes gateway restart  # Run from a shell outside the gateway process
"$VENV/bin/mnemosyne-hermes" uninstall
"$VENV/bin/python" -m pip uninstall mnemosyne-hermes
```

`mnemosyne-hermes uninstall` removes the plugin registration at `$HERMES_HOME/plugins/mnemosyne`.

### Activated local environment

```bash
hermes memory off
hermes gateway restart  # Run from a shell outside the gateway process
mnemosyne-hermes uninstall
pip uninstall mnemosyne-hermes
```
