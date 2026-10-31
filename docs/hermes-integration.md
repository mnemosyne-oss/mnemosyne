# Hermes Integration

Mnemosyne is designed as a native memory backend for the [Hermes Agent Framework](https://github.com/NousResearch/hermes-agent). It implements the Hermes `MemoryProvider` interface and registers as a plugin.

> **This is the canonical Hermes setup guide.** The README links here for full instructions.

> **Desktop configuration compatibility:** current Hermes declared provider
> schemas persist non-secret fields to provider-specific JSON (or Honcho's host
> store); they cannot target `memory.mnemosyne` in `config.yaml`. Mnemosyne does
> not declare that desktop surface because doing so would create a second,
> ignored config store. Use `hermes memory setup` or `hermes config set
> memory.mnemosyne.<key> <value>` until Hermes ships a config-backed provider
> schema storage contract. This is an upstream release gate, not a Mnemosyne
> runtime limitation.

## Mnemosyne-owned Hermes home contract

For installer and status operations, Mnemosyne uses an explicitly supplied
`--hermes-home` (or API path) when present; otherwise it uses a non-empty
`HERMES_HOME`, falling back to the expanded `~/.hermes`. The resulting active
Hermes home is the base for these Mnemosyne-owned paths:

- `$HERMES_HOME/plugins/mnemosyne/` is the base plugin or wrapper directory. In
  wrapper mode it contains `mnemosyne-wrapper.json`, whose `python` and
  `site_packages` fields record the selected Python executable and
  site-packages directory.
- Where the installer manages profile-link preference, it stores that choice in
  `$HERMES_HOME/plugins/.mnemosyne-profile-links.json`.
- The bundled skill is installed at
  `$HERMES_HOME/skills/memory/mnemosyne-memory-override/SKILL.md`; its
  installer-managed SHA-256 checksum is stored beside it as `SKILL.md.sha256`.

A wrapper's selected side environment is intended to live outside the
replaceable Hermes runtime. This is a Mnemosyne deployment layout, not a claim
that Hermes itself enforces or preserves the side environment, plugin
directory, manifest, preference file, or skill. See
[Persistent side-venv wrapper mode](#step-1-install) for
setup and recovery instructions.

Mnemosyne CI can verify its own path construction, manifest contents, plugin
discovery, selected-home propagation, and simulated managed-venv replacement
using temporary fixtures. It cannot verify Hermes Desktop updates, update
behavior in tagged Hermes releases, uninstall or provider-removal behavior,
profile deletion, or experimental package-manager semantics; those require
validation in the corresponding Hermes distribution and release.

## Install Profile Comparison

| Profile | When to use | RAM | Key tradeoff |
|---------|-------------|-----|-------------|
| `mnemosyne-memory` (core) | Low-resource (Raspberry Pi, 1 GB VPS), or when using a remote embedding API | ~50 MB | No local embeddings. Point `MNEMOSYNE_EMBEDDING_API_URL` to an external endpoint, and set `MNEMOSYNE_EMBEDDING_DIM` for models not in the built-in table (else direct startup fails loudly). |
| `mnemosyne-memory[embeddings]` | Mid-range systems with local embedding support | ~800 MB | Adds `fastembed` for local vector generation. Best for single-user desktop agents. |
| `mnemosyne-memory[all]` | Full-featured — local embeddings + local LLM consolidation | ~1.5 GB | Adds `sentence-transformers` + local LLM deps (`ctransformers`). Maximum capability. |
| `mnemosyne-hermes` | Hermes Agent users | Includes `[embeddings]` | Wraps core library with plugin manifest + entry points and requires `mnemosyne-memory[embeddings]`. For a wrapper install, use its default embeddings dependency or add `mnemosyne-memory[all]`; `core` alone is unavailable. Run `hermes config set memory.provider mnemosyne` after install. |

> **Fail-loud is surface-specific.** With an unknown embedding model and no `MNEMOSYNE_EMBEDDING_DIM`, a **direct core or MCP-provider** process imports `embeddings` eagerly and exits at import with an actionable error. The **`mnemosyne-hermes` wrapper** imports core lazily and captures init failures, so the provider reports unavailable and affected tools return an error reason instead of the agent process exiting.

> **Privacy note on remote embedding endpoints.** Embeddings go to a remote API whenever `MNEMOSYNE_EMBEDDING_API_URL` points at a custom (non-OpenRouter) endpoint, the model name is API-shaped (`openai/*`, `text-embedding*`), or `MNEMOSYNE_EMBEDDINGS_VIA_API` is truthy; on the OpenRouter default (or an OpenRouter URL) the last two are what route, and with no URL set the OpenRouter default is used. That service receives the text of your memories and of your recall queries (working-memory content, summaries, annotations, and search queries) for vectorization. For privacy-sensitive or local-first deployments prefer a local-embedding profile (`[embeddings]` or `[all]`); use a remote endpoint only when you accept that the embedding provider sees your content.

**Hardware guidance:** Core alone runs on a Raspberry Pi 4 (4 GB) with ~300 MB free for LLM, but it is not a valid `mnemosyne-hermes` wrapper profile. `[embeddings]` needs at least 2 GB free RAM. `[all]` recommends 8 GB+.

## Setup

### Recall query bodies and channel envelopes

Both Hermes providers sanitize prefetch and explicit `mnemosyne_recall` queries
without changing captured text or stored memories. They recognize one serialized
recent-channel envelope only when the query starts exactly with
`[Recent channel messages]\n`, followed by nonempty history and a
`\n\n[New message]\n` separator. Backfill is removed only when there is
**exactly one** such separator. The remaining suffix then receives the usual
leading speaker-stamp removal.

Multiple separators make the boundary ambiguous: relay history can contain
literal markers, and the current body can contain the same markers or nested
envelopes. Choosing the first can retain old messages; choosing the last can
truncate the current message. Both providers therefore retain the entire query
in this case, including history. This avoids deleting current-message text but
does **not** guarantee history-free recall. Even a single complete envelope can
be ordinary quoted text; the header is a caller convention, not proof of origin.
Callers needing reliable separation must pass the current body separately from
channel history, or agree on an escaped or length-delimited upstream format.

A marker or sender stamp alone does not identify an envelope. Unwrapped bodies,
incomplete envelopes, and other channel-context formats retain their text,
apart from the existing leading-stamp handling. This is a narrow text format
contract, not sender authentication: callers must pass an ordinary body unless
they intend the exact leading format above to be interpreted as an envelope.

### Step 1: Install

**Choose an install mode first.** The default writes a symbolic link into the
Hermes home and relies on that link remaining available. Persistent wrapper mode
instead keeps Mnemosyne's Python dependencies in a side venv outside the Hermes
runtime and installs a real plugin directory. Pick wrapper mode when you need
independence from replacement of Hermes' own Python environment, or where
symbolic links are privileged operations:

| Your Hermes | Mode | Why |
|---|---|---|
| Linux or macOS, pip or source install | default (symlink) | The Hermes venv is yours and persists. |
| Docker image | **wrapper** | An image update may replace the venv. |
| Desktop binary installer | **wrapper** | An update may replace the bundled Python environment. |
| Native Windows | **wrapper** | The native default is the persistent wrapper install; a symbolic link needs Developer Mode or an elevated shell, so `WinError 1314` appears only when explicitly requesting `--mode symlink`. |
| WSL | default (symlink) | Behaves like Linux. |

The three wrapper rows use the same mechanism to keep Mnemosyne independent
when something outside your control replaces or restricts the Hermes runtime.
Only the paths differ. See
[Persistent side-venv wrapper mode](#step-1-install) below.

**pip (recommended):**

```bash
pip install mnemosyne-hermes
```

> **Which Mnemosyne core does the plugin need?** `mnemosyne-hermes` 0.7.3 requires
> `mnemosyne-memory>=4.0.0b3`, and no stable 4.0 core exists yet, so a stable-only
> install (prereleases disabled) cannot resolve it. Choose a channel:
>
> - **4.0 beta:** allow prereleases and pin the pair:
>   `pip install --pre 'mnemosyne-memory[embeddings]==4.0.0b3' 'mnemosyne-hermes==0.7.3'`
> - **Stable:** stay on the last audited stable pair:
>   `pip install 'mnemosyne-memory[embeddings]==3.15.1' 'mnemosyne-hermes==0.7.1'`
>
> Do not use `mnemosyne-hermes` 0.7.2. Its declared floor allows a core that lacks
> the APIs it imports, so it installs cleanly and then fails when the provider
> initializes. Once 4.0.0 is released as stable, the plain command above works again.

**Debian / Trixie users:** newer Debian releases block bare pip installs. Use a venv:

```bash
python3 -m venv ~/.hermes/hermes-agent/venv
source ~/.hermes/hermes-agent/venv/bin/activate
pip install mnemosyne-hermes
```

**Or from source:**

```bash
git clone https://github.com/mnemosyne-oss/mnemosyne.git
cd mnemosyne
pip install -e "integrations/hermes[dev]"
```

<a id="persistent-side-venv-wrapper-mode"></a>
> **Persistent side-venv wrapper mode.** Use this wherever the Hermes Python
> environment is rebuilt outside your control (Docker images, Desktop binary
> installers) or where symbolic links require privileges (native Windows). The
> side venv and the plugin directory live outside the replaceable runtime, so a
> rebuilt Hermes venv does not take Mnemosyne with it, and no symbolic link is
> created at all.
>
> The instructions below are written for Docker because its paths are the least
> familiar; the same commands apply elsewhere with the Hermes home and
> interpreter for your installation. For native Windows, use the PowerShell form
> in [Native Windows local install and recovery](#native-windows-local-install-and-recovery).
>
> **Docker.** Inside the official Hermes container, the mounted Hermes home is `/opt/data/`, not `~/.hermes/`. Keep the side venv on that mounted volume so both it and the wrapper survive image rebuilds. The side venv must use the same Python **major/minor** as the running Hermes gateway; do not create it with an unrelated `python3` from `PATH`.
>
> Before creating the side venv, set `HERMES_PYTHON` to the virtual-environment interpreter used by the **actual running Hermes gateway**, verified with the deployment operator or runtime configuration. Keep the venv interpreter path, not its resolved base-Python symlink target. If ownership cannot be determined, **stop**; an executable or a version probe alone does not establish ownership.
>
> The standalone installer's bounded discovery supports validated launcher-based installations and recorded staged runtimes; it does not simply trust an executable sibling of `hermes`. A recorded but unusable staged runtime must not fall back to the checkout's anchor venv. For wrappers, shims, containers, or an otherwise uncertain layout, obtain the verified interpreter from the deployment rather than reproducing discovery with a sibling lookup. The explicit `--python` used below bypasses discovery and selects the side environment the wrapper imports. See [`_find_hermes_python`](../integrations/hermes/src/mnemosyne_hermes/install.py).
>
> ```bash
> : "${HERMES_PYTHON:?Set this to the deployment-verified Python that starts the running Hermes gateway}"
> if [ ! -f "$HERMES_PYTHON" ] || [ ! -x "$HERMES_PYTHON" ]; then
>   printf 'Hermes Python is not an executable file: %s\n' "$HERMES_PYTHON" >&2
>   exit 1
> fi
> "$HERMES_PYTHON" -c 'import sys; raise SystemExit(sys.prefix == sys.base_prefix)' || {
>   printf 'Hermes Python is not a virtual-environment runtime: %s\n' "$HERMES_PYTHON" >&2
>   exit 1
> }
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
> if [ -e "$VENV" ] || [ -L "$VENV" ]; then
>   printf 'Refusing to create side venv at existing path: %s\n' "$VENV" >&2
>   exit 1
> fi
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
> "$VENV/bin/python" -m pip --version
> "$VENV/bin/python" -m pip install --upgrade "$MNEMOSYNE_REQUIREMENT" mnemosyne-hermes
>
> export HERMES_HOME=/opt/data
> export MNEMOSYNE_DATA_DIR=/opt/data
> "$VENV/bin/mnemosyne-hermes" install --mode wrapper --python "$VENV/bin/python"
> "$VENV/bin/mnemosyne-hermes" status
> hermes config set memory.provider mnemosyne
> ```
>
> If venv creation or the pip probe fails, stop. After preserving any partial target and choosing a new, nonexistent persistent path for `VENV`, a seeded fallback is `uv venv --seed --python "$HERMES_PYTHON" "$VENV"`. Use the same verified interpreter, rerun the pip probe and install commands only after success, and never run bare `uv venv` or silently overwrite an existing environment.
>
> Restart the actual Hermes container or Compose service using its deployment tooling; do not substitute `hermes gateway restart` for that deployment restart. Once the service is running, validate the default profile from inside it:
>
> ```bash
> hermes memory status
> ```
>
> Wrapper mode creates a real directory at `$HERMES_HOME/plugins/mnemosyne/`. Its bootstrap adds the selected side venv's site-packages to `sys.path` before importing `mnemosyne_hermes`. Do not use the manual symlink instructions below for this installation.
>
> **Repair a persistent wrapper compatibility failure.** Existing wrappers whose selected venv records a different or unreadable Python major/minor now fail before activating that venv. **Before running the recovery commands, including in a fresh shell, obtain the deployment-verified `HERMES_PYTHON` and rerun its validation block above in that same shell.** Do not substitute the current-shell Python or the checkout's anchor venv for a staged runtime. Before recovery, explicitly set `MNEMOSYNE_PROFILE` to the value used by the existing wrapper: `embeddings` or `all`. This is a documentation-local selector, not a runtime setting the recovery commands can infer. If the gateway reports that runtime-compatibility error, create a new, confirmed-empty persistent side-venv with `"$HERMES_PYTHON" -m venv` as above, install `mnemosyne-hermes` there with that same wrapper requirement, then force-refresh the wrapper with that new interpreter:
>
> ```bash
> set -e
> if [ -z "${HERMES_PYTHON:-}" ] || [ ! -f "$HERMES_PYTHON" ] || [ ! -x "$HERMES_PYTHON" ]; then
>   printf 'HERMES_PYTHON is unset or not an executable file; rerun the deployment-verified interpreter validation in this shell before recovery.\n' >&2
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

> **Prefer persistent wrapper mode on native Windows.** The native default is
> the persistent wrapper install, which creates a real plugin directory and
> never needs elevated privileges. A symbolic link, requested explicitly with
> `--mode symlink`, is what Windows permits only with Developer Mode enabled or
> an elevated shell; without one of those it fails with `WinError 1314`, which
> is why explicit symlink installs are reported as working for some people and
> not others. Tracked in
> [#857](https://github.com/mnemosyne-oss/mnemosyne/issues/857).
>
> Advice circulating in the community suggests linking a Mnemosyne **source
> checkout** into the Hermes home. That target is wrong even when the link
> succeeds: the only correct target is the installed package directory inside the
> Hermes venv's `Lib\site-packages\mnemosyne_hermes`. Wrapper mode avoids the
> question entirely.

This section is for a **native Windows** Hermes installation, not Docker, WSL, or
another user's Hermes home. Run it as the Windows user and in the Hermes profile
that will use Mnemosyne. Do not copy a path from another account: select that
Hermes installation's actual runtime Python as `<hermes-python>`, verified with
the deployment operator or runtime configuration. The explicit `--python` form
below is authoritative, not proof that an arbitrary interpreter belongs to Hermes.

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

# Native Windows defaults to persistent wrapper mode (symlink needs an explicit
# --mode and elevated privileges). --python is the safe fallback when discovery
# is not applicable to this Hermes layout.
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

If an explicit `--mode symlink` install fails specifically with **WinError
1314**, the current process lacks effective symbolic-link privilege. Enable
Windows Developer Mode or use an account that has that effective privilege;
administrator-group membership alone does not guarantee it. The installer
intentionally does not switch modes automatically, and this guide does not use
junctions as a workaround.

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
venv rebuild or replacement.

#### Windows persistent side-venv wrapper

Create a **new, empty** persistent side venv using the verified `$HermesPython`
selected above. Do not run bare `uv venv`: it can select a different Python
major/minor. Choose a new path or preserve a rollback copy before retiring an old
environment; these commands refuse to replace an existing side venv.

```powershell
$ErrorActionPreference = 'Stop'
$SideVenv = Join-Path $HermesHome '.mnemosyne\venv'
if (Test-Path -LiteralPath $SideVenv) {
    throw "Refusing to replace existing side venv: $SideVenv"
}
uv venv --seed --python $HermesPython $SideVenv
if ($LASTEXITCODE -ne 0) {
    throw "uv could not create the side venv: $SideVenv"
}
$SidePython = Join-Path $SideVenv 'Scripts\python.exe'
$SideInstaller = Join-Path $SideVenv 'Scripts\mnemosyne-hermes.exe'
& $SidePython -m pip install "mnemosyne-memory[embeddings]" mnemosyne-hermes
if ($LASTEXITCODE -ne 0) {
    throw 'Mnemosyne package installation failed in the side venv'
}
& $SideInstaller install --mode wrapper --python $SidePython --no-profile-links
if ($LASTEXITCODE -ne 0) {
    throw 'Mnemosyne wrapper installation failed'
}
```

`--seed` supplies pip in the side venv. If `uv` is unavailable, stop and arrange
it or another venv-creation method that explicitly uses `$HermesPython`, not an
unrelated Python on `PATH`. After changing a wrapper, restart or refresh the
actual local Hermes process. Use the side-venv installer status block in the
verification below for this route.

#### Enable and verify

If Hermes reports the Mnemosyne plugin as disabled, enable it without requesting
any built-in-tool override, then select it as the memory provider:

```powershell
hermes plugins enable hermes-mnemosyne --no-allow-tool-override
hermes config set memory.provider mnemosyne
```

Restart or start a new local Hermes session, then verify the active provider in
the same user/profile scope (common to both installation routes):

```powershell
hermes memory status
```

Then run only the installer status block for the route you used, in the same
PowerShell scope where its variable was initialized above.

**Direct install into the Hermes environment (including the WinError 1314
wrapper retry):** use `$MnemosyneHermes`, initialized beside `$HermesPython`:

```powershell
& $MnemosyneHermes status
```

**Persistent side-venv wrapper:** use `$SideInstaller`, initialized inside
`$SideVenv` alongside `$SidePython`, not the direct-install executable:

```powershell
& $SideInstaller status
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
| WinError 1314 during an explicit `--mode symlink` install | Enable Developer Mode/effective symlink privilege, or retry wrapper mode (the native default) with the selected Hermes Python. Do not use a junction or expect an automatic mode switch. |
| Wrapper validation times out | `--import-timeout` defaults to 60 and affects installer validation only. A larger positive finite value can retry that install probe, but `mnemosyne-hermes status` always validates with its fixed 60-second policy; investigate the selected interpreter if status still fails. |
| Hermes update or venv replacement leaves a missing/stale provider | Reinstall into the replacement Hermes interpreter for a symlink install. For a persistent-side-venv wrapper, refresh the wrapper against its retained selected interpreter, then restart Hermes and rerun both status commands. |

### Step 2: Link the plugin in a local mutable environment

For a non-Docker local installation on Linux, macOS, or WSL, the supported installer default is symlink mode (native Windows defaults to persistent wrapper mode):

```bash
mnemosyne-hermes install
```

This creates the historical plugin symlink. Use wrapper mode above for Docker and persistent side venv deployments.

#### Recover a legacy per-file plugin directory

Do not link individual package files. An upgrade can add a relative import without adding its link to an existing directory. The supported installer above links the whole package directory. Package-only upgrades do not repair a per-file directory.

This recovery is only for an existing directory of per-file links into the **same** installed `mnemosyne_hermes` package, in an explicitly selected Hermes home. First determine the actual active `HERMES_HOME` and gateway Python using the installation route above; a PM-managed environment or shared install needs its own owner-approved package update before this step. Do not infer either path from the shell, a build venv or a profile name. Stop if the provider is a directory symlink, a wrapper, a foreign/custom plugin, or has profile-local plugin targets; handle those separately without deleting them. Never use `install --force` on this legacy directory: it can remove its contents.

Export both variables to the verified paths, then run the following block once. The subshell stops on any failed check, install or status command without changing the caller's shell options. It checks ownership of every per-file link and moves the entire old directory to a backup **outside** the plugin discovery tree before invoking the installer. Regular custom files remain in that backup, not in the new package link. An existing backup blocks the operation.

Before running the block, verify that `HERMES_PYTHON` imports the exact installed `mnemosyne_hermes` package that owns the legacy per-file links. A `mnemosyne-hermes` executable on `PATH`, or a provider installed only in a separate pipx environment, does not satisfy this prerequisite. Do not substitute the CLI interpreter without verifying ownership of those links.

```bash
(
set -e
: "${HERMES_HOME:?select the actual Hermes home first}"
: "${HERMES_PYTHON:?select the gateway Python first}"
"$HERMES_PYTHON" - <<'PY'
import os
from pathlib import Path
import mnemosyne_hermes

home = Path(os.environ["HERMES_HOME"]).expanduser().resolve(strict=True)
package = Path(mnemosyne_hermes.__file__).resolve().parent
target = home / "plugins" / "mnemosyne"
backup = home / "mnemosyne-legacy-plugin-backup"
if target.parent.is_symlink() or target.is_symlink() or not target.is_dir():
    raise SystemExit("Not a local per-file directory; no changes made")
if (home / "plugins" / "hermes-mnemosyne").exists() or (home / "plugins" / "hermes-mnemosyne").is_symlink():
    raise SystemExit("Old plugin path also exists; inspect it first")
if backup.exists() or backup.is_symlink():
    raise SystemExit("Backup already exists; inspect it first")
profiles = home / "profiles"
if profiles.is_dir() and any(
    (profile / "plugins" / "mnemosyne").exists()
    or (profile / "plugins" / "mnemosyne").is_symlink()
    for profile in profiles.iterdir() if profile.is_dir()
):
    raise SystemExit("Profile plugin targets exist; inspect their ownership first")
init = target / "__init__.py"
if not init.is_symlink() or init.resolve(strict=True) != package / "__init__.py":
    raise SystemExit("Plugin initializer does not belong to this package")
for child in target.iterdir():
    if child.is_symlink():
        link = Path(os.readlink(child))
        if not link.is_absolute():
            link = child.parent / link
        if link.absolute() != package / child.name:
            raise SystemExit(f"Foreign per-file link: {child}")
target.rename(backup)
print(f"Legacy directory retained at {backup}")
PY
"$HERMES_PYTHON" -m mnemosyne_hermes.install --hermes-home "$HERMES_HOME" install --mode symlink --python "$HERMES_PYTHON" --no-profile-links
"$HERMES_PYTHON" -m mnemosyne_hermes.install --hermes-home "$HERMES_HOME" status
hermes memory status
)
```

If the installer or status fails, keep the backup and investigate the selected interpreter and home; do not rerun the move or remove the backup. Restart the active Hermes process after both statuses succeed, then verify capture, remember and recall in an isolated test session. This procedure does not establish survival across a native macOS PM update or repair cron initialization.

#### Migrating from the legacy `mnemosyne-install` route

`mnemosyne-install` and `mnemosyne-uninstall` from the core `mnemosyne-memory`
package remain available, but they are now a compatibility entry point rather
than a second installer. They no longer create the historical
`~/.hermes/plugins/mnemosyne -> hermes_memory_provider/` symlink (#651); they
delegate to the standalone provider and verify the result:

```bash
mnemosyne-install             # migrate any legacy link, then delegate an install
mnemosyne-install --status    # verify the provider, the plugin directory, and the config
mnemosyne-install --migrate   # remove legacy links only
mnemosyne-install --dry-run   # show what would change without changing it
```

A legacy install is detected by the resolved target of the plugin link, so a
link into `mnemosyne_hermes` — including the manual fallback above — is left
alone. A real directory is reported and preserved, never deleted. When the
standalone provider is not importable in the current Python, not the
`mnemosyne-hermes` console script, and not installed in Hermes' own venv, the
command fails with the install commands instead of recreating the obsolete
link. `--status` exits non-zero on any of those conditions, so it is usable as
a check in scripts.

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

`mnemosyne repair` is Linux-only, intentionally narrow and report-gated, not a global cleanup command. Review the Doctor report, select only the candidate you intend to act on, and run a dry run first (example below).

For repair, use a disposable **consistent offline copy** on verified native Linux
storage. Copying only the main file of a live WAL database does not guarantee a
consistent copy: committed pages may still be in WAL. Do not delete WAL/SHM files
or silently checkpoint a live bank to bypass a refusal. Trigger semantics and
WAL/snapshot policy remain separate decisions (#1040 D2/D3).

Apply requires a private staging directory whose requested POSIX mode is actually
0700, on the same device as the authorized database, and working
`linkat(AT_EMPTY_PATH)` FD linking. The resulting regular file must retain the
authorized `(st_dev, st_ino)` identity and have at least two links. These checks
remain fail-closed; there is no chmod or alternate-binding fallback.

In the #1040 reporter's WSL setup, default 9p/drvfs automounts without `metadata`,
`umask` or `dmask` returned stage mode 0777 and were correctly refused. The distro's
own ext4 storage passed binding (the reporter's `$HOME` was an example, not a
promise that every home directory is native ext4). Metadata-enabled drvfs and
device identity across a WSL restart remain untested.

Binding diagnostics identify the failed gate or operation and symbolic/numeric
errno when available, without including paths or raw OS error text. A linkat
`ENOENT`, `EPERM` or `EACCES` is an observed syscall result, not proof of capability
denial; `EIO` is an I/O failure, not proof of an unsupported filesystem. An
unavailable operation may have no errno. Do not infer a capability or filesystem
cause from these results alone.

**Dry-run does not bind the database or create link capability probes.** It plans
selected actions without certifying apply's mode, device or FD-link compatibility.
Alternate binding remains a separate security/storage decision (#1040 D4).

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

For a database containing sqlite-vec `vec0` tables, install the optional
`embeddings` extra (or compatible `sqlite-vec`) in the **same runtime** for
Doctor and repair, then generate a fresh report on the offline copy. Without
it Doctor can report `present_but_unloadable`, but the incomplete schema
fingerprint cannot authorize repair, including `expire`. Repair loads sqlite-vec
on planning and bound write connections and disables further extension loading
before checking the schema or selecting rows. This does not relax the separate
trigger, WAL/sidecar, or filesystem-binding restrictions (#1040 D2–D4).

### Preflight a direct JSON file import

Before importing a JSON file directly through the Hermes provider, run:

```bash
hermes mnemosyne import --input <backup.json> --dry-run
```

This dry run validates the file and reports the import counts using a disposable database clone. It writes neither the active database nor provider audit data. Run the same command without `--dry-run` to perform the side-effecting import.

## Multiple Profiles, Agent Teams, and Group Chats

Mnemosyne is a memory provider. It does not run agents, route messages, or
decide which agent answers in a chat room. Those are Hermes concerns:
profiles, the messaging gateway (Telegram, SimpleX, and other platforms), and
agent dispatch are configured in Hermes. See the
[Hermes Agent documentation](https://github.com/NousResearch/hermes-agent) for
profile, gateway, and per-platform setup.

What Mnemosyne controls is where each active provider instance stores and
looks up memories. These memories live in local SQLite files; Mnemosyne needs
no external database or network service. Network access happens only for
configured sync or remote services, and for the one-time download of the local
embedding model on first use.

| Concern | Owned by | Mnemosyne setting |
|---|---|---|
| Hermes profiles, gateway routing, gateway session keys, which agent replies in a room | Hermes | — |
| Private database path | Mnemosyne | Depends on the provider path; see **Data directory** below |
| Named bank per profile | Mnemosyne | `memory.mnemosyne.profile_isolation` (default `false`) |
| Session vs. cross-session visibility inside a bank | Mnemosyne | `memory.mnemosyne.default_scope` (`session` default, or `global`); `cross_session` / `MNEMOSYNE_CROSS_SESSION` (default `false`) removes the session filter on recall |
| Cross-agent sharing | Mnemosyne | `mnemosyne_shared_*` tools, `shared_surface_path`, `shared_surface_read` |

- **Data directory.** There is no single resolver for every consumer. When
  Hermes supplies `hermes_home` to the installable provider
  (`mnemosyne-hermes`), the provider builds an explicit private path under
  `<hermes_home>/mnemosyne/data/`, so profiles with separate Hermes homes
  already select separate stores, while profiles that share one home share
  one store unless `profile_isolation` is enabled and their resolved bank names
  differ. Without a supplied home, and for the Mnemosyne Core library, CLI, and
  the root `hermes_memory_provider`, the environment decides:
  `MNEMOSYNE_DATA_DIR`, else `$HERMES_HOME/mnemosyne/data`, else
  `~/.hermes/mnemosyne/data`. Run `mnemosyne_diagnose` in each profile to see
  which database it actually uses.
- **Bank.** With `profile_isolation: false`, each provider instance uses the
  `default` bank of its data directory. With `profile_isolation: true`, the
  provider selects a bank named after the Hermes agent identity or, failing
  that, the basename of the profile's Hermes home, stored under
  `data/banks/<name>/mnemosyne.db`. The base home (`~/.hermes`, `hermes`,
  `default`) still maps to the `default` bank.
- **Before enabling `profile_isolation`.** Separate profile homes or data
  directories can already give each profile a separate store, so you may not
  need it; it matters mainly when several profiles share one Hermes home. Enabling it only selects a different bank: existing memories are
  **not** migrated from the current bank, so a profile can start with empty
  memory even though nothing was lost. It is a way to choose a bank, not a
  general ownership or privacy repair for memories already written.
- **Session.** When Hermes supplies a gateway session key, the provider uses
  it as the session; otherwise it uses the Hermes session id. `session`-scoped
  memories are recalled within that session by default; `global`-scoped
  memories surface across all sessions of the same bank. This is default
  recall behavior, not an unconditional boundary: with `cross_session=true`
  (or `MNEMOSYNE_CROSS_SESSION=1`), recall drops the session filter and can
  return `session`-scoped memories from other sessions of the same bank, so
  separate session keys (for example, per participant) do not guarantee
  isolation under that override.
- **Shared surface.** A separate database for memories that several profiles
  are meant to share, written and read through the `mnemosyne_shared_*` tools.
  `shared_surface_read: true` also merges surface results into
  `mnemosyne_recall`. The default path differs by provider: the installable
  provider uses `~/.mnemosyne/data/shared/mnemosyne.db`, while the root
  `hermes_memory_provider` uses `data/shared/mnemosyne.db` relative to its own
  code location. Profiles that should share a surface should each set the same
  absolute `shared_surface_path`.

Two setups are commonly confused:

- **Several people talking to one assistant** (one Hermes profile in a group
  chat): there is one provider instance and one bank. Gateway session keys
  come from Hermes: with its default `group_sessions_per_user: true`,
  non-threaded group chats get a per-participant session key when a
  participant ID is available, so `session`-scoped memories can be separate
  per participant within that one bank. Threads, `group_sessions_per_user:
  false`, or a missing participant ID change the session key. `global`-scoped
  memories are always shared across the whole bank.
- **Several profile agents in one room** (an agent team): each agent is its
  own Hermes profile with its own `memory.provider: mnemosyne` config, and so
  its own store when the profile homes differ (see **Data directory**). Use
  `profile_isolation` only if you want named banks, and use the shared surface
  for what the team should share. Being in the same chat room does not by
  itself share or separate memory.

Installing or linking one Mnemosyne plugin for several profiles (see
[Multiple profiles, one side venv](#step-1-install)) shares
code only. It does not define shared memory; the settings above do. Confirm
the active provider in each profile with `hermes --profile <name> memory
status`; the `mnemosyne_diagnose` tool reports that profile's active database
path and whether profile isolation is enabled.

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
token there unless `MNEMOSYNE_MCP_TOKENS` is set. `MNEMOSYNE_MCP_TOKENS`
takes precedence; either it or `MNEMOSYNE_MCP_TOKEN` supplies HTTP
authentication. For multi-agent deployments, set a JSON name-to-secret mapping
(placeholders shown):

```bash
export MNEMOSYNE_MCP_TOKENS='{"agent-a":"replace-with-agent-a-secret","agent-b":"replace-with-agent-b-secret"}'
```

The matched token name becomes the authoritative memory author identity;
conflicting client-supplied `author_id` values are rejected. Startup is rejected
when the value is blank or whitespace, malformed JSON, a non-object or empty
object, or contains non-string or blank names/secrets, duplicate names (including
names equal after trimming), or duplicate secrets. See the
[CLI reference](cli-reference.md#multi-agent-tokens-per-agent-identity) and
[configuration reference](api/configuration.mdx) for the canonical contract.

A non-loopback bind exposes the selected local SQLite-backed memory bank to
network clients, so it requires authentication; the `streamable-http` transport
additionally requires `MNEMOSYNE_MCP_ALLOWED_HOSTS`, with
`MNEMOSYNE_MCP_ALLOWED_ORIGINS` optionally restricting browser origins. Bearer
tokens on a non-loopback HTTP bind require TLS termination in front of the
server, using a reverse proxy or secure tunnel.

Mnemosyne does not currently expose a standalone REST API server.

## Uninstall

### Persistent wrapper install (Docker, Desktop, POSIX shells)

```bash
export HERMES_HOME=/opt/data  # Replace with the non-default Hermes home used at install time
VENV=/path/to/venv             # The same side venv passed to the wrapper install
hermes memory off  # Disable the external provider; built-in memory remains active
"$VENV/bin/mnemosyne-hermes" uninstall
# For every profile-local wrapper, repeat the uninstall first, with that profile's Hermes home.
HERMES_HOME=/opt/data/profiles/work "$VENV/bin/mnemosyne-hermes" uninstall
"$VENV/bin/python" -m pip uninstall mnemosyne-hermes
```

For Docker or Compose, restart the actual deployment service with its deployment tooling after removing the wrapper. For a local installation, restart or refresh the actual local Hermes process.

`mnemosyne-hermes uninstall` removes the plugin registration at `$HERMES_HOME/plugins/mnemosyne`. Remove every profile-local wrapper before uninstalling the side-venv package.

### Persistent wrapper install (native Windows)

Same sequence, different shell and layout: a native Windows virtual environment
keeps its executables in `Scripts\` rather than `bin/`, so the POSIX block above
does not run in PowerShell.

```powershell
$env:HERMES_HOME = "C:\ProgramData\hermes"   # Replace with the non-default Hermes home used at install time
$Venv = "C:\path\to\venv"                   # The same side venv passed to the wrapper install
hermes memory off        # Disable the external provider; built-in memory remains active
hermes gateway restart   # Run from a shell outside the gateway process
& "$Venv\Scripts\mnemosyne-hermes.exe" uninstall
# For every profile-local wrapper, repeat the uninstall first, with that profile's Hermes home.
$env:HERMES_HOME = "C:\ProgramData\hermes\profiles\work"
& "$Venv\Scripts\mnemosyne-hermes.exe" uninstall
& "$Venv\Scripts\python.exe" -m pip uninstall mnemosyne-hermes
```

WSL is a POSIX layout, so use the block above rather than this one.

### Activated local environment

```bash
hermes memory off
hermes gateway restart  # Run from a shell outside the gateway process
mnemosyne-hermes uninstall
pip uninstall mnemosyne-hermes
```

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

Suppression filters matching working-memory candidates before ranking and
fusion; it does not hide every representation of a captured memory. Consolidated
episodic rows remain eligible for recall when their source working-memory row is
excluded from automatic prompt context. Normal episodic eligibility rules still
apply.

Python 3.10 is supported with a conservative SQLite parameter budget; exceeding
that budget or losing proof also means ordinary recall.

The suppression ledger itself performs no network calls. Configured remote
embedding services or synchronization can still process memory content
externally; enabling this feature does not make those operations local-only.

This is best-effort duplicate reduction, **not** exact live-context tracking or
a durable v2 checkpoint guarantee. A provider restart discards suppression state.
An already-returned per-turn prefetch string cannot be rewritten by the plugin;
released memories become available on the next user-turn prefetch. No Hermes
host modification or database migration is required.
