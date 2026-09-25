<div align="center">

<img src="https://raw.githubusercontent.com/mnemosyne-oss/mnemosyne/main/assets/mnemosyne.jpg" alt="Mnemosyne" width="40%">

# Mnemosyne for Hermes Agent

*Local-first memory provider for Hermes Agent. 40 tools. Zero cloud. Zero latency.*

[![PyPI](https://img.shields.io/pypi/v/mnemosyne-hermes.svg)](https://pypi.org/project/mnemosyne-hermes/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mnemosyne-oss/mnemosyne/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/mnemosyne-oss/mnemosyne.svg?style=social)](https://github.com/mnemosyne-oss/mnemosyne)

</div>

**Mnemosyne** gives Hermes Agent a local-first memory layer that captures conversation, tool calls, execution paths, decisions, outcomes, and corrections. Then surfaces it all with intent-aware hybrid recall. SQLite on your machine. No cloud. No API keys. No latency.

---

## The Problem

Agent workflows lose context across sessions. A few ways this bites:

- Prior decisions and constraints vanish between sessions
- Tool-call context isn't preserved in raw transcripts
- Failures get repeated because nobody remembered the fix
- Project context gets buried in long chat history
- Cross-session memory becomes noise without structure
- Conversation-only memory misses the execution path: what the agent *did*, not just what was *said*

## What Mnemosyne Changes

Mnemosyne adds structured, local-first, agent-native memory to Hermes through the `mnemosyne` memory provider.

It gives Hermes:

- **Automatic capture**: every turn, in the background, after the response is sent. Conversation, decisions, tool calls, outcomes.
- **Hybrid search**: vector similarity + FTS5 full-text + importance scoring. All tunable per-query. Bias toward recency, relevance, or both.
- **Episodic consolidation**: `mnemosyne_sleep` compresses old working memories into long-term summaries so the working set stays small and recall stays sharp.
- **Knowledge graph**: subject-predicate-object triples with BFS graph traversal. Link memories semantically.
- **Multi-agent validation**: agents can attest, update, or invalidate each other's memories with provenance tracking.
- **Shared surface**: compact cross-agent metadata for multi-agent workflows.
- **Zero cloud**: SQLite on your machine. No network calls. No API keys. No quota limits.

When using Mnemosyne, disable Hermes' built-in MEMORY.md/USER.md system to avoid duplication. Do NOT use `hermes tools disable memory` — that also kills all 40 Mnemosyne-registered tools (the memory toolset gates both built-in AND provider injection at `agent_init.py:1163-1172`).

Edit `~/.hermes/config.yaml`:

```yaml
memory:
  memory_enabled: false
user_profile_enabled: false
```

Mnemosyne handles everything: capture, recall, consolidation, knowledge graph, multi-agent validation. The built-in MEMORY.md/USER.md system is redundant and just burns tokens. One provider. One memory layer.

## How It Works

Mnemosyne runs on three stages:

### 1. Capture

After Hermes completes a turn, the Mnemosyne provider stores the full interaction (user message, assistant response, tool calls, and available execution context) in a local SQLite database with vector embeddings.

Each memory gets tagged with importance (0.0-1.0), scope (session or global), veracity (stated, inferred, tool, or imported), and optional metadata. Memories can carry expiration dates and named entities for fuzzy recall.

This is how Hermes builds memory from what it says *and* what it does.

### 2. Recall

Recall is intentional. Agents decide when to recall, what scope, and how many results. There is no automatic retrieval dumping context into every prompt.

When `mnemosyne_recall` fires, Mnemosyne runs hybrid search: vector similarity finds semantic matches, FTS5 full-text finds keyword matches, and importance scoring boosts what matters. All three weights are tunable per-query so you can bias toward recency, relevance, or both.

Returned context can include prior decisions, constraints, failure modes, project patterns, and execution outcomes: without stuffing irrelevant history into the prompt.

### 3. Consolidate

`mnemosyne_sleep` compresses old working memories into episodic summaries. Think of it as a nightly cleanup that knows what to keep and what to summarize. The working set stays small. Recall stays sharp. Long-running agents don't drown in their own history.

### Session lifecycle

The provider follows Hermes session changes without requiring a provider restart. After
`/new`, `/resume`, `/branch`, undo, or context compression, the active `BeamMemory`
session is rebound before the next memory operation, keeping writes and tool calls
attributed to the current conversation.

Non-empty session IDs supplied to per-turn prefetch and sync calls scope that
individual operation; empty values preserve the active session. A configured
`gateway_session_key` remains the stable scope across both paths and across
session changes, so a branch or compression switch does not adopt the child
session ID.

A configured skip context (`subagent`, `cron`, `flush`, `background`, or
`skill_loop` by default) intentionally receives no private Beam. If an existing
primary provider instance is re-initialized under one of those contexts, it must
clear the live Beam to prevent writes into the wrong session. That transition
emits a warning, returns `reason_code="reset_by_reinit"` from memory tools, and
shows an `UNAVAILABLE` prompt notice. Re-initialize the provider in a primary
context to recover. A provider that starts directly in a skip context remains
silent and returns `reason_code="skipped_context"`.

Provider lifecycle hooks are fail-soft. Database or disk failures during
prefetch, turn sync, session-end or automatic consolidation, and wrapper or
audit cleanup are logged and suppressed so Hermes can continue without a
lifecycle exception surfacing to the user.

---

## Quickstart

**Prerequisites:** Hermes Agent (pipx install), Python 3.10+, no API keys needed.

### Install

```bash
pipx install mnemosyne-hermes
hermes config set memory.provider mnemosyne
hermes memory status
```

To install only at the selected Hermes home without linking opted-in child
profiles, use:

```bash
mnemosyne-hermes install --no-profile-links
```

The default continues to link opted-in child profiles for backward compatibility.

That's it. The entry point in `mnemosyne-hermes` registers with Hermes' memory
provider discovery system (`hermes_agent.memory_providers`). No symlinks. No
directory copying. No plugin.yaml gymnastics. The provider surfaces
automatically on the next Hermes start.

Verify:

```bash
hermes memory status
# Should show:
#   Provider:  mnemosyne
#   Plugin:    installed ✓
```

#### Selecting Hermes' interpreter

`mnemosyne-hermes install` finds Hermes' Python by following the `hermes` launcher
on PATH, through symlinks and through the `exec` handoff of a shell-wrapper
launcher, and accepts the interpreter sitting beside it only when that directory
is a real virtual environment. It then checks the known install roots
(`$HERMES_HOME/hermes-agent`, `~/hermes-agent`, `/opt/hermes/hermes-agent`,
`/usr/local/lib/hermes-agent`, `/usr/lib/hermes-agent`), which are held to the
same bar.

Wrapper resolution is deliberately bounded: it reads a limited prefix of the
launcher, follows a limited number of hops, and understands a fixed set of forms
(`exec /path/to/hermes`, a relative or bare target, and `env`/`VAR=val` prefixes).
Anything outside that, such as `exec env -u VAR ...` or a launcher that starts
Hermes without `exec`, is not followed, and discovery moves on to the install
roots rather than guessing. Layouts that end there need `--python`.

On the default path, if none of those yields a virtual environment the install
stops rather than guessing. An interpreter that merely sits next to the launcher
is usually a Homebrew or system Python, and installing `mnemosyne-hermes[all]`
into it leaves Hermes' own environment untouched while reporting success. A
Hermes installed outside a virtual environment therefore needs `--python`.

`--no-bootstrap` is the deliberate exception. It already means "do not install
anything into Hermes' venv", so there is no wrong-interpreter install left to
prevent. When discovery finds nothing there, the installer says so, skips
dependency validation, and continues. The plugin is linked, but nothing has
confirmed that Hermes can import it, so a `--no-bootstrap` install is not a
validated one. Verify it separately:

```bash
hermes memory status
```

Pass `--python` to skip discovery entirely when your layout is unusual or the
wrong interpreter is being picked:

```bash
mnemosyne-hermes install --python /path/to/hermes/venv/bin/python
mnemosyne-hermes install --dry-run          # shows which interpreter would be used
```

`--python` is authoritative in both install modes: it overrides discovery for a
symlink install, and selects the site-packages a `--mode wrapper` install imports
from.

#### Wrapper validation timeout

For a wrapper install, `--import-timeout SECONDS` sets the timeout for both
selected-`--python` validation probes. It defaults to 60 seconds. Use it when
that selected interpreter needs longer to resolve its site-packages or import
`mnemosyne_hermes`. `SECONDS` must be positive and finite: zero, negative,
`NaN`, and infinite values are rejected.

```bash
mnemosyne-hermes install --mode wrapper --python /path/to/hermes/venv/bin/python --import-timeout 90
mnemosyne-hermes install --mode wrapper --python /path/to/hermes/venv/bin/python --no-bootstrap --import-timeout 90
```

The second form still validates the selected wrapper interpreter; `--no-bootstrap`
only skips automatic package installation. `mnemosyne-hermes status` intentionally
uses its fixed 60-second validation policy, independent of an install-time override.

The installer also deploys the bundled `mnemosyne-memory-override` skill to
`$HERMES_HOME/skills/memory/mnemosyne-memory-override/SKILL.md`. The skill is a
behavioral guardrail that nudges agents away from the legacy `memory` tool for
durable facts. Existing user-edited copies are skipped by default; `install
--force` overwrites the skill only after writing `SKILL.md.bak` next to it.

### How it works

When you install via pipx, the package registers two entry points:

- `hermes_agent.memory_providers` - discovered by Hermes' `discover_memory_providers()`
  so `hermes memory status` sees it as an installed plugin
- `hermes_agent.plugins` - discovered by Hermes' `PluginManager` for tool and
  CLI registration

Hermes checks pipx-installed packages alongside directory-based plugins. No
extra setup steps needed.

### Development install

```bash
git clone https://github.com/mnemosyne-oss/mnemosyne.git
cd mnemosyne
pip install -e .
pipx install -e integrations/hermes   # replaces hook with editable path
```

## Configuration

No required config. Everything defaults to `~/.mnemosyne/`. Optional overrides:

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_HOME` | `~/.mnemosyne` | Storage directory |
| `MNEMOSYNE_VEC_WEIGHT` | `0.5` | Vector similarity weight in hybrid recall |
| `MNEMOSYNE_FTS_WEIGHT` | `0.3` | Full-text search weight |
| `MNEMOSYNE_IMPORTANCE_WEIGHT` | `0.2` | Importance score weight |
| `MNEMOSYNE_AUTO_SLEEP_ENABLED` | `false` | Auto-consolidate after N turns |
| `MNEMOSYNE_AUTO_SLEEP_THRESHOLD` | `50` | Turns between auto-consolidation |
| `MNEMOSYNE_PROFILE_ISOLATION` | `false` | Separate DB per Hermes profile |
| `MNEMOSYNE_SYNC_TURN_USER_LIMIT` | `500` | User content truncation in `sync_turn()` (`0` = no limit) |
| `MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT` | `800` | Assistant content truncation in `sync_turn()` (`0` = no limit) |
| `MNEMOSYNE_FACT_RECALL_ENABLED` | `false` | Merge LLM-extracted facts into standard recall |
| `MNEMOSYNE_IGNORE_PATTERNS` | _(empty)_ | Newline-separated regular expressions; matching writes are rejected before persistence |
| `MNEMOSYNE_SYNC_ROLES` | `user` | Comma-separated roles autosaved by `sync_turn()` (`user`, `assistant`; empty disables conversation autosave) |
| `MNEMOSYNE_WRITE_CLASSIFIER` | `off` | Write admission classifier: `off`, `warn`, or `strict` |
| `MNEMOSYNE_PREFETCH_CONTENT_CHARS` | `0` | Per-memory prefetch content cap (`0` = full content) |
| `MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS` | `2` | Shared non-generic terms required for automatic prefetch injection |
| `MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE` | `0.30` | Minimum fraction of non-generic query terms covered by a prefetched memory. Exempt for a canonical slot whose entire body is one CJK bigram that also appears in the query, so a short topical fact is reachable from a contextual sentence; separators around that bigram (`。`, `／`, `｡`, quotes, brackets) are removed before the check, so `部署／` qualifies while `部署／計画` still counts as two units. Automatic prefetch also runs the same run-local U+3005 iteration-mark predicate as explicit canonical recall before scoring, so a candidate that clears these thresholds can still be excluded when its iteration-mark evidence does not match a CJK run. Automatic prefetch still applies the rarity cap below, and explicit canonical recall needs no coverage fraction at all |
| `MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY` | `1` | Maximum canonical document frequency that permits a one-token match (`0` disables the exception). A whole-body CJK bigram match counts as a one-token match, so it stays subject to this cap |
| `MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS` | path-specific built-in canonical set | Complete replacement for the canonical generic-token set used by automatic and explicit canonical lookup; does not affect working/episodic prefetch |
| `MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS` | _(empty)_ | Extra owner/deployment terms added to automatic canonical prefetch only |
| `MNEMOSYNE_DEFAULT_SCOPE` | `session` | Default scope for remember (`global` enables cross-session immediate recall) |

Or in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mnemosyne
  mnemosyne:
    auto_sleep: true
    sleep_threshold: 30
    sync_roles: [user]  # user | assistant; [] disables conversation autosave
    ignore_patterns:
      - "^\\s*\\$\\s*pip\\s"
    write_classifier: "off"  # off | warn | strict
```

`sync_roles` accepts a comma-separated string (for example, `user,assistant`) or
an `initialize(...)`/YAML list, tuple, or set containing `user` and/or
`assistant`. A string that looks like a YAML list, such as `"['user',
'assistant']"`, is not parsed as YAML; it is invalid. Empty strings and empty
containers silently disable conversation autosave. A non-empty value containing
no valid roles disables autosave and logs one warning; unknown entries are
silently dropped when at least one valid role remains. Resolution precedence is
`initialize(...)` keyword argument > Hermes `memory.mnemosyne.sync_roles` >
Mnemosyne `sync_roles` config > `MNEMOSYNE_SYNC_ROLES` > default `user`.

For `ignore_patterns` and `write_classifier`, an explicit `initialize(...)`
keyword argument takes precedence. Without that override, resolution is
Hermes `config.yaml` `memory.mnemosyne.*` > core `config.yaml` > environment
variable > default.
`MNEMOSYNE_IGNORE_PATTERNS` is newline-separated and defaults to empty (no
patterns). `MNEMOSYNE_WRITE_CLASSIFIER` controls admission for explicit writes
and autosaved turns: `off` still applies `ignore_patterns`; `warn` runs the
noise/secret classifier but stores classified content with warnings; and
`strict` rejects content classified as noise or secret-like. Unset, blank, or
invalid classifier values fall back to `off` (invalid values also log a warning).

## Tools

40 tools. All surfaced through Hermes' tool system.

**Core memory:** `remember`, `recall`, `sleep`, `stats`, `get`, `update`, `forget`, `invalidate`, `validate`

**Knowledge graph:** `triple_add`, `triple_query`, `graph_query`, `graph_link`

**Multi-agent:** `shared_remember`, `shared_recall`, `shared_forget`, `shared_stats`

**Working notes:** `scratchpad_write`, `scratchpad_read`, `scratchpad_clear`

**Ops:** `export`, `import`, `diagnose`

## Test the Memory Loop

Ask Hermes to do a multi-step task:

> "Investigate why the payment sync test is failing and fix it."

Hermes inspects files, runs commands, identifies a failing fixture, makes a decision, applies a fix, and observes the result.

After the turn, Mnemosyne stores the full execution path: the failure, the decision, the fix, the outcome, and the pattern.

In a new session:

> "A similar payment sync test is failing again. Check prior fixes before changing anything."

Hermes calls `mnemosyne_recall`, finds the relevant prior failure and fix, and doesn't repeat the mistake.

## Fail-Soft By Design

If Mnemosyne's database is unavailable or disk is full, the provider logs the error and Hermes continues answering. Memory is additive: it never blocks the user.

Memory issues are logged but never surface as user-facing errors.

## Contributing

We welcome contributions. See the [Contributing Guidelines](https://github.com/mnemosyne-oss/mnemosyne/blob/main/CONTRIBUTING.md) for code style, standards, and submitting pull requests.

To build from source:

```bash
git clone https://github.com/mnemosyne-oss/mnemosyne.git
cd mnemosyne

pip install -e .
pip install -e integrations/hermes
```

## Support

- [Documentation](https://github.com/mnemosyne-oss/mnemosyne#readme)
- [Discord](https://discord.gg/nousresearch)
- [Issues](https://github.com/mnemosyne-oss/mnemosyne/issues)

## License

MIT

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
