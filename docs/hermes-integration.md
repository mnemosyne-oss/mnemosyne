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

> **Docker users: use persistent side-venv wrapper mode.** This is the canonical Docker installation. Inside the official Hermes container, the mounted Hermes home is `/opt/data/`, not `~/.hermes/`. Keep the side venv on that mounted volume so both it and the wrapper survive image rebuilds:
>
> ```bash
> # Run inside the container once. Choose a persistent path on the mounted volume.
> VENV=/opt/data/venvs/mnemosyne
> python3 -m venv "$VENV"
> "$VENV/bin/python" -m pip install --upgrade mnemosyne-hermes
>
> export HERMES_HOME=/opt/data
> export MNEMOSYNE_DATA_DIR=/opt/data
> "$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python"
> "$VENV/bin/mnemosyne-hermes" status
> hermes config set memory.provider mnemosyne
> hermes gateway restart
> ```
>
> Wrapper mode creates a real directory at `$HERMES_HOME/plugins/mnemosyne/`. Its bootstrap adds the selected side venv's site-packages to `sys.path` before importing `mnemosyne_hermes`. Do not use the manual symlink instructions below for this installation.
>
> **Multiple profiles, one side venv:** after `status` confirms the base wrapper, profiles that use the same persistent side venv may each symlink their `plugins/mnemosyne` directory to that tested base wrapper. If a profile's `config.yaml` already selects `memory.provider: mnemosyne` before the base wrapper install, the installer creates the link automatically. It skips an existing profile target unless run with `--force`.
>
> To add a profile manually, refuse to overwrite an existing plugin target:
>
> ```bash
> # Each profile must select memory.provider: mnemosyne in its own Hermes config.
> PROFILE=/opt/data/profiles/work
> TARGET="$PROFILE/plugins/mnemosyne"
> mkdir -p "$PROFILE/plugins"
> if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then
>   printf 'Refusing to replace existing plugin target: %s\n' "$TARGET" >&2
>   exit 1
> fi
> ln -s /opt/data/plugins/mnemosyne "$TARGET"
> ```
>
> A profile-local wrapper is an equivalent runtime alternative when independent plugin directories are preferred. Use it instead of the base-wrapper link for that profile:
>
> ```bash
> HERMES_HOME=/opt/data/profiles/work "$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python"
> HERMES_HOME=/opt/data/profiles/work "$VENV/bin/mnemosyne-hermes" status
> ```
>
> For an existing target, use `--force` only after confirming it is the Mnemosyne link or wrapper you intend to replace. Do not force-replace an unknown plugin target.
>
> Do not mix the two strategies for one profile. A later base-home `install --force` can replace a profile-local wrapper with a link to the base wrapper.
>
> **Never symlink a profile directly to `site-packages/mnemosyne_hermes`.** That bypasses the wrapper's `sys.path` bootstrap and can leave fresh Hermes profiles unable to import the provider.

### Step 2: Link the plugin in a local mutable environment

For a non-Docker local installation, the supported installer default is symlink mode:

```bash
mnemosyne-hermes install
```

This creates the historical plugin symlink. Use wrapper mode above for Docker and persistent side venv deployments.

#### Legacy manual symlink fallback

Hermes discovers directory plugins by scanning a folder on disk, not by reading pip metadata. Use the following manual fallback only for a local, mutable venv that is not using wrapper mode and has no existing plugin target. If you previously ran `mnemosyne-hermes install`, run `mnemosyne-hermes uninstall` first. Do not use this block as a troubleshooting step on top of an installer-managed symlink:

```bash
TARGET="$HOME/.hermes/plugins/mnemosyne"
if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then
  printf 'Refusing to replace existing plugin target: %s\n' "$TARGET" >&2
  exit 1
fi
PKG="$(~/.hermes/hermes-agent/venv/bin/python -c 'import pathlib, mnemosyne_hermes; print(pathlib.Path(mnemosyne_hermes.__file__).resolve().parent)')"
if [ ! -d "$PKG" ]; then
  printf 'Could not locate mnemosyne_hermes in the selected venv\n' >&2
  exit 1
fi
mkdir -p "$TARGET"
# Symlink the installed package contents into the manual plugin directory.
ln -s "$PKG"/* "$TARGET/"
```

If you installed in a custom venv (for example, `~/.hermes-venv`), replace `~/.hermes/hermes-agent/venv/bin/python` with the Python binary inside that venv. Do not combine this manual mode with a wrapper directory, and do not use it to link Docker profiles to the side venv's `site-packages` package.

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
# For every profile-local wrapper, repeat the uninstall first, with that profile's Hermes home.
HERMES_HOME=/opt/data/profiles/work "$VENV/bin/mnemosyne-hermes" uninstall
"$VENV/bin/python" -m pip uninstall mnemosyne-hermes
```

`mnemosyne-hermes uninstall` removes the plugin registration at `$HERMES_HOME/plugins/mnemosyne`. Remove every profile-local wrapper before uninstalling the side-venv package.

### Activated local environment

```bash
hermes memory off
hermes gateway restart  # Run from a shell outside the gateway process
mnemosyne-hermes uninstall
pip uninstall mnemosyne-hermes
```
