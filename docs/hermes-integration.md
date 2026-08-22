# Hermes Integration

Mnemosyne is designed as a native memory backend for the [Hermes Agent Framework](https://github.com/NousResearch/hermes-agent). It implements the Hermes `MemoryProvider` interface and registers as a plugin.

> **This is the canonical Hermes setup guide.** The README links here for full instructions.

## Install Profile Comparison

| Profile | When to use | RAM | Key tradeoff |
|---------|-------------|-----|-------------|
| `mnemosyne-memory` (core) | Low-resource (Raspberry Pi, 1 GB VPS), or when using a remote embedding API | ~50 MB | No local embeddings. Point `MNEMOSYNE_EMBEDDING_API_URL` to an external endpoint, and set `MNEMOSYNE_EMBEDDING_DIM` for models not in the built-in table (else direct startup fails loudly). |
| `mnemosyne-memory[embeddings]` | Mid-range systems with local embedding support | ~800 MB | Adds `fastembed` for local vector generation. Best for single-user desktop agents. |
| `mnemosyne-memory[all]` | Full-featured — local embeddings + local LLM consolidation | ~1.5 GB | Adds `sentence-transformers` + local LLM deps (`ctransformers`). Maximum capability. |
| `mnemosyne-hermes` | Hermes Agent users | Includes `[embeddings]` | Wraps core library with plugin manifest + entry points and requires `mnemosyne-memory[embeddings]`. For a wrapper install, use its default embeddings dependency or add `mnemosyne-memory[all]`; `core` alone is unavailable. Run `hermes config set memory.provider mnemosyne` after install. |

> **Fail-loud is surface-specific.** With an unknown embedding model and no `MNEMOSYNE_EMBEDDING_DIM`, a **direct core or MCP-provider** process imports `embeddings` eagerly and exits at import with an actionable error. The **`mnemosyne-hermes` wrapper** imports core lazily and captures init failures, so the provider reports unavailable and affected tools return an error reason instead of the agent process exiting.

**Hardware guidance:** Core alone runs on a Raspberry Pi 4 (4 GB) with ~300 MB free for LLM, but it is not a valid `mnemosyne-hermes` wrapper profile. `[embeddings]` needs at least 2 GB free RAM. `[all]` recommends 8 GB+.

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

> **Docker users: use persistent side-venv wrapper mode.** This is the canonical Docker installation. Inside the official Hermes container, the mounted Hermes home is `/opt/data/`, not `~/.hermes/`. Keep the side venv on that mounted volume so both it and the wrapper survive image rebuilds. The side venv must use the same Python **major/minor** as the running Hermes gateway; do not create it with an unrelated `python3` from `PATH`.
>
> For a launcher-based Hermes installation, first derive its runtime interpreter from the resolved `hermes` launcher. This bounded launcher-sibling probe covers only that installation shape: it checks the launcher's sibling `python`, then `python3`. It is not a reproduction of the installer's broader internal discovery. If it cannot find a sibling, **stop** and determine the real gateway interpreter from the deployment; do not substitute the current-shell Python or guess another environment.
>
> ```bash
> HERMES_BIN="$(command -v hermes)" || {
>   printf 'Could not find the Hermes launcher on PATH\n' >&2
>   exit 1
> }
> HERMES_BIN="$(readlink -f "$HERMES_BIN")" || {
>   printf 'Could not resolve the Hermes launcher\n' >&2
>   exit 1
> }
> if [ ! -f "$HERMES_BIN" ] || [ ! -x "$HERMES_BIN" ]; then
>   printf 'Resolved Hermes launcher is not a regular executable file: %s\n' "$HERMES_BIN" >&2
>   exit 1
> fi
> HERMES_BIN_DIR="$(dirname "$HERMES_BIN")"
> if [ -f "$HERMES_BIN_DIR/python" ]; then
>   HERMES_PYTHON="$HERMES_BIN_DIR/python"
> elif [ -f "$HERMES_BIN_DIR/python3" ]; then
>   HERMES_PYTHON="$HERMES_BIN_DIR/python3"
> else
>   printf 'Could not find Hermes Python beside %s\n' "$HERMES_BIN" >&2
>   exit 1
> fi
> if [ ! -x "$HERMES_PYTHON" ]; then
>   printf 'Hermes Python is not executable: %s\n' "$HERMES_PYTHON" >&2
>   exit 1
> fi
> "$HERMES_PYTHON" --version || {
>   printf 'Hermes Python failed its version probe: %s\n' "$HERMES_PYTHON" >&2
>   exit 1
> }
> ```
>
> Then run the installation once inside the container. Wrapper mode requires `mnemosyne-hermes`, which itself requires `mnemosyne-memory[embeddings]`; select `embeddings` for the standard provider or `all` to add local-LLM extras. `core` is unavailable for wrapper installs because it cannot remove that required embeddings dependency.
>
> ```bash
> set -e
> # Choose a persistent path on the mounted volume.
> VENV=/opt/data/venvs/mnemosyne
> MNEMOSYNE_PROFILE="${MNEMOSYNE_PROFILE:-embeddings}"  # embeddings (default) or all.
> case "$MNEMOSYNE_PROFILE" in
>   embeddings) MNEMOSYNE_REQUIREMENT="mnemosyne-memory[embeddings]" ;;
>   all) MNEMOSYNE_REQUIREMENT="mnemosyne-memory[all]" ;;
>   core)
>     printf 'MNEMOSYNE_PROFILE=core is unavailable for mnemosyne-hermes wrapper installs: mnemosyne-hermes requires mnemosyne-memory[embeddings]. Use embeddings or all.\n' >&2
>     exit 1
>     ;;
>   *)
>     printf 'Unsupported MNEMOSYNE_PROFILE: %s (expected embeddings or all)\n' "$MNEMOSYNE_PROFILE" >&2
>     exit 1
>     ;;
> esac
> "$HERMES_PYTHON" -m venv "$VENV"
> "$VENV/bin/python" -m pip install --upgrade "$MNEMOSYNE_REQUIREMENT" mnemosyne-hermes
>
> export HERMES_HOME=/opt/data
> export MNEMOSYNE_DATA_DIR=/opt/data
> "$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python"
> "$VENV/bin/mnemosyne-hermes" status
> hermes config set memory.provider mnemosyne
> ```
>
> Restart the actual Hermes container or Compose service using its deployment tooling; do not substitute `hermes gateway restart` for that deployment restart. Once the service is running, validate the default profile from inside it:
>
> ```bash
> hermes memory status
> ```
>
> Wrapper mode creates a real directory at `$HERMES_HOME/plugins/mnemosyne/`. Its bootstrap adds the selected side venv's site-packages to `sys.path` before importing `mnemosyne_hermes`. Do not use the manual symlink instructions below for this installation.
>
> **Repair a persistent wrapper compatibility failure.** Existing wrappers whose selected venv records a different or unreadable Python major/minor now fail before activating that venv. **Before running the recovery commands, including in a fresh shell, rerun the launcher-based discovery block above in that same shell.** It sets and version-probes `HERMES_PYTHON`; do not substitute the current-shell Python. Before recovery, explicitly set `MNEMOSYNE_PROFILE` to the value used by the existing wrapper: `embeddings` or `all`. This is a documentation-local selector, not a runtime setting the recovery commands can infer. If the gateway reports that runtime-compatibility error, create a new, confirmed-empty persistent side-venv with `"$HERMES_PYTHON" -m venv` as above, install `mnemosyne-hermes` there with that same wrapper requirement, then force-refresh the wrapper with that new interpreter:
>
> ```bash
> set -e
> if [ -z "${HERMES_PYTHON:-}" ] || [ ! -f "$HERMES_PYTHON" ] || [ ! -x "$HERMES_PYTHON" ]; then
>   printf 'HERMES_PYTHON is unset or not an executable file; rerun the launcher-based discovery block in this shell before recovery.\n' >&2
>   exit 1
> fi
> export HERMES_HOME=/opt/data
> export MNEMOSYNE_DATA_DIR=/opt/data
> VENV=/opt/data/venvs/mnemosyne-compatible  # New dedicated venv; do not overwrite an unknown path.
> if [ -e "$VENV" ] || [ -L "$VENV" ]; then
>   printf 'Refusing to create recovery venv at existing path: %s\n' "$VENV" >&2
>   exit 1
> fi
> if [ -z "${MNEMOSYNE_PROFILE:-}" ]; then
>   printf 'MNEMOSYNE_PROFILE is required for recovery. Set it to the existing wrapper profile: embeddings or all.\n' >&2
>   exit 1
> fi
> case "$MNEMOSYNE_PROFILE" in
>   embeddings) MNEMOSYNE_REQUIREMENT="mnemosyne-memory[embeddings]" ;;
>   all) MNEMOSYNE_REQUIREMENT="mnemosyne-memory[all]" ;;
>   core)
>     printf 'MNEMOSYNE_PROFILE=core is unavailable for mnemosyne-hermes wrapper installs: mnemosyne-hermes requires mnemosyne-memory[embeddings]. Use embeddings or all.\n' >&2
>     exit 1
>     ;;
>   *)
>     printf 'Unsupported MNEMOSYNE_PROFILE: %s (expected embeddings or all)\n' "$MNEMOSYNE_PROFILE" >&2
>     exit 1
>     ;;
> esac
> "$HERMES_PYTHON" -m venv "$VENV"
> "$VENV/bin/python" -m pip install --upgrade "$MNEMOSYNE_REQUIREMENT" mnemosyne-hermes
> "$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python" --force
> "$VENV/bin/mnemosyne-hermes" status
> ```
>
> `--force` replaces the existing Mnemosyne plugin target, so use it only after confirming that target is the Mnemosyne link or wrapper you intend to replace. Restart the actual container or Compose service using its deployment tooling, then run `hermes memory status` inside that service for the default profile. For an existing profile-local wrapper, run the same refresh and wrapper `status` commands with that profile's `HERMES_HOME`; do not force-replace an unknown plugin target.
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
> set -e
> if [ -z "${VENV:-}" ] || [ ! -x "${VENV:-}/bin/python" ] || [ ! -x "${VENV:-}/bin/mnemosyne-hermes" ]; then
>   printf 'Set VENV to the tested compatible side venv with executable bin/python and bin/mnemosyne-hermes.\n' >&2
>   exit 1
> fi
> HERMES_HOME=/opt/data/profiles/work "$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python"
> HERMES_HOME=/opt/data/profiles/work "$VENV/bin/mnemosyne-hermes" status
> ```
>
> Keep the profile scope through restart and validation: in the example above, replace `<name>` below with `work` (the final path component of the `HERMES_HOME` used for wrapper refresh/status). For Docker or Compose, restart the actual deployment service with its deployment tooling and then, inside that service, run:
>
> ```bash
> hermes --profile <name> memory status
> ```
>
> For a local installed gateway (not Docker/Compose deployment recovery), run:
>
> ```bash
> hermes --profile <name> gateway restart
> hermes --profile <name> memory status
> ```
>
> For an existing target, use `--force` only after confirming it is the Mnemosyne link or wrapper you intend to replace. Do not force-replace an unknown plugin target.
>
> Do not mix the two strategies for one profile. A later base-home `install --force` can replace a profile-local wrapper with a link to the base wrapper.
>
> **Never symlink a profile directly to `site-packages/mnemosyne_hermes`.** That bypasses the wrapper's `sys.path` bootstrap and can leave fresh Hermes profiles unable to import the provider.

### Native Windows local install and recovery

This section is for a **native Windows** Hermes installation, not Docker, WSL, or
another user's Hermes home. Run it as the Windows user and in the Hermes profile
that will use Mnemosyne. Do not copy a path from another account: select that
Hermes installation's Python as `<hermes-python>`. If discovery cannot identify
it, the explicit `--python` form below is authoritative.

`mnemosyne-hermes` requires the standard local semantic-search dependency set,
`mnemosyne-memory[embeddings]`. A Hermes-managed virtual environment can lack
`pip`; check it first. In a fresh PowerShell, set the selected Hermes home and
interpreter. `$HermesHome` must be the active user's intended Hermes home or
profile home; it prevents this install from using another profile. Use its
adjacent Scripts directory for the installer:

```powershell
$HermesHome = '<selected-hermes-home-or-profile-home>'
$env:HERMES_HOME = $HermesHome
$HermesPython = '<hermes-python>'
$MnemosyneHermes = Join-Path (Split-Path -Parent $HermesPython) 'mnemosyne-hermes.exe'

& $HermesPython -m pip --version
if ($LASTEXITCODE -ne 0) {
    uv pip install --python $HermesPython "mnemosyne-memory[embeddings]" mnemosyne-hermes
} else {
    & $HermesPython -m pip install "mnemosyne-memory[embeddings]" mnemosyne-hermes
}

# Symlink is the normal local install mode. --python is the safe fallback when
# discovery is not applicable to this Hermes layout.
& $MnemosyneHermes install --python $HermesPython --no-profile-links
```

`[embeddings]` is the standard local semantic-search profile. `mnemosyne-memory[all]`
also adds local-LLM consolidation dependencies. On Windows, its local LLM
dependencies can require compatible wheels (including `llama-cpp-python`) or a
native build toolchain such as MSVC, NMake, and Visual Studio Build Tools; do
not assume they will build universally. Start with `[embeddings]` unless local
LLM consolidation is required.

The installer discovers native Windows `Scripts/python.exe` layouts when using
implicit discovery. An explicit `--python` remains the authoritative, safer
choice for an unusual layout.

#### Windows WinError 1314 recovery

If the normal symlink install fails specifically with **WinError 1314**, the
current process lacks effective symbolic-link privilege. Enable Windows Developer
Mode or use an account that has that effective privilege; administrator-group
membership alone does not guarantee it. The installer intentionally does not
switch modes automatically, and this guide does not use junctions as a workaround.

A wrapper retry avoids the plugin symlink and records the selected interpreter.
Use the same selected Hermes Python:

```powershell
& $MnemosyneHermes install --mode wrapper --python $HermesPython --import-timeout 60 --no-profile-links
```

`--import-timeout` is the wrapper validation timeout and currently defaults to
60 seconds; the explicit value above makes that retry policy visible. Wrapper
mode is appropriate for a persistent side virtual environment in deployments
where that side environment is retained across Hermes updates. It does **not**
make a direct install into a Hermes-managed virtual environment survive a Hermes
venv rebuild or replacement. After changing a wrapper, restart or refresh the
actual local Hermes process before checking provider state.

#### Enable and verify

If Hermes reports the Mnemosyne plugin as disabled, enable it without requesting
any built-in-tool override, then select it as the memory provider:

```powershell
hermes plugins enable mnemosyne --no-allow-tool-override
hermes config set memory.provider mnemosyne
```

Restart or start a new local Hermes session, then verify the installer and active
provider in the same user/profile scope:

```powershell
& $MnemosyneHermes status
hermes memory status
```

The current documented Hermes CLI contract does not provide a `hermes plugins
doctor mnemosyne` command, so do not substitute an unverified plugin-doctor
command. Likewise, this guide does not prescribe a direct store/recall/delete
smoke command: use the provider's runtime tools only after the two status checks
succeed.

| Symptom | Bounded recovery |
|---|---|
| `No module named pip` from `<hermes-python>` | Use `uv pip install --python "<hermes-python>" "mnemosyne-memory[embeddings]" mnemosyne-hermes`, then repeat the normal installer command. |
| `[all]` fails while installing local-LLM dependencies | Keep `[embeddings]` for local semantic search, or resolve compatible wheels/build-toolchain requirements before retrying `[all]`. |
| WinError 1314 during symlink install | Enable Developer Mode/effective symlink privilege, or retry wrapper mode with the selected Hermes Python. Do not use a junction or expect an automatic mode switch. |
| Wrapper validation times out | `--import-timeout` defaults to 60 and affects installer validation only. A larger positive finite value can retry that install probe, but `mnemosyne-hermes status` always validates with its fixed 60-second policy; investigate the selected interpreter if status still fails. |
| Hermes update or venv replacement leaves a missing/stale provider | Reinstall into the replacement Hermes interpreter for a symlink install. For a persistent-side-venv wrapper, refresh the wrapper against its retained selected interpreter, then restart Hermes and rerun both status commands. |

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

### Preflight a direct JSON file import

Before importing a JSON file directly through the Hermes provider, run:

```bash
hermes mnemosyne import --input <backup.json> --dry-run
```

This dry run validates the file and reports the import counts using a disposable database clone. It writes neither the active database nor provider audit data. Run the same command without `--dry-run` to perform the side-effecting import.

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
mnemosyne mcp --transport streamable-http --port 8080  # native MCP http transport
```

The HTTP transports bind to loopback (`127.0.0.1`) by default and need no
token there. A non-loopback bind exposes the selected local SQLite-backed
memory bank to network clients, so it requires `MNEMOSYNE_MCP_TOKEN`; the
`streamable-http` transport also requires `MNEMOSYNE_MCP_ALLOWED_HOSTS`, with
`MNEMOSYNE_MCP_ALLOWED_ORIGINS` optionally restricting browser origins.

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
