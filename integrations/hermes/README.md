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

## Read-only installation ownership status

`PYTHONDONTWRITEBYTECODE=1 mnemosyne-hermes --hermes-home /selected/home status`
reads installation metadata in that explicit home, not the calling Python's
runtime. It does not run Python/PM, import Core/provider code, activate `.pth`
files, open stores, sync, repair, adopt, create directories, or contact a network.
Interpreter startup is outside this command-logic guarantee; use a trusted
interpreter and disable bytecode as above.

Illustrative output (not a current installation measurement):

    Ownership: pm_managed
    Package metadata consistency: adoption complete (METADATA ONLY)
    Runtime/provider function/hooks/autosave: NOT CHECKED

`pm_managed` requires recorded PM inputs, candidate requirements, plugin binding,
and selected Core/provider package metadata to agree. Chosen extras come from
recorded requirements, never inferred from installed libraries. `legacy` means a
package symlink with verified critical RECORD metadata, not a runtime-health
claim. `external` is a legitimate explicit wrapper binding, not an instruction to
adopt it. `incomplete` and `unknown` explain absent/conflicting/unsupported data;
a candidate directory alone is not PM evidence. Missing plugins remain distinct.

The explicitly selected home is canonicalized once: home/ancestor aliases are
allowed, but metadata symlink components inside that canonical home are refused.
An outside-home plugin symlink is reported as a present binding only, with unknown
ownership: its destination is not inspected, even to distinguish missing from
existing packages. This is not proof of an installed or working runtime. Skill
metadata errors (including unsafe paths or later read failures) are reported
separately and do not erase verified plugin/PM facts. Plugin/PM metadata drift
still prevents a complete ownership verdict.

The supported reference is Hermes `f4ac0bea1ea50bba62886fb0b7e308e4edee8fd3`:
facts schema 1, selected per-install generation, manifest-only `plugin-deps`
workspace members and exact Core/provider requirements. The selected `uv.lock`
must contain exactly one matching package/version record for each of Core and
provider; empty, conflicting or ambiguous locks cannot report completeness.
The PM stamp is recorded but not recomputed/validated: this is a package-metadata
comparison, not proof that PM completed a transaction or a dependency resolver.
Other schemas, Python
without `tomllib`, metadata links, outside-home package trees, missing/ambiguous
distributions and unsupported versions remain unknown. Critical METADATA/WHEEL/
package-init hashes are checked, not every installed file or imported function.
Repeated reads and binding checks detect ordinary changes, not an atomic
transaction or protection against adversarial filesystem races. No runtime,
hook, autosave, update or release acceptance is implied. This enhancement is
separate from publication of the already merged #1217 fix.

## Native integration platform support

Android and Emscripten are excluded from the native Hermes integration. Source
entry points fail closed before provider/core imports or runtime activation;
static discovery is not an availability guarantee. See [platform boundaries and
release gates](PLATFORM_SUPPORT.md). A guard-bearing standalone release and a
catalog floor requiring that actually published version remain shipping gates;
this source change does not retrofit published `0.7.6` or existing wrappers.

## Diagnostic connection provenance

`mnemosyne_diagnose` reports orphan counts for the active provider's database in
`active_provider_orphan_diagnostics`. Its `foreign_keys_enabled` integer describes
SQLite's connection-local setting on the separate diagnostic connection, **not**
the active BEAM connection. The additive `foreign_keys_connection` field is
`"separate_diagnostic_connection"`; existing count fields and values are unchanged.
A value of `0` here does not establish disabled enforcement on BEAM or database
corruption. Diagnostics do not change either connection's foreign-key setting.

## How It Works

Mnemosyne runs on three stages:

### 1. Capture

After Hermes completes a turn, the Mnemosyne provider stores the full interaction (user message, assistant response, tool calls, and available execution context) in a local SQLite database with vector embeddings.

Each memory gets tagged with importance (0.0-1.0), scope (session or global), veracity (stated, inferred, tool, or imported), and optional metadata. Memories can carry expiration dates and named entities for fuzzy recall.

This is how Hermes builds memory from what it says *and* what it does.

### 2. Recall

Explicit `mnemosyne_recall` is intentional — agents decide when to recall, what scope, and how many results. In addition, the provider performs **automatic pre-turn prefetch**: before an LLM call it injects relevant memories into the prompt as a `## Mnemosyne Context` block. Prefetch is deliberately conservative and is not "automatic retrieval dumping context into every prompt." For ordinary working/episodic memories, it applies a lexical gate — a memory must share multiple distinctive non-generic terms with the query and meet a minimum query-coverage fraction — before silent injection. Linear recall results must additionally clear per-signal signal and score floors, which favors distilled facts and preferences over raw transcripts. Under `MNEMOSYNE_POLYPHONIC_RECALL=1`, results ranked by the polyphonic engine (RRF over vector/graph/fact/temporal voices) pass the lexical gate without the linear score floors, so raw `[USER]` conversation transcripts surface in prefetch too.

Automatic **canonical prefetch** has a separate admission and ranking path. Its default generic-token set excludes a bounded German function-word vocabulary, including `für`, `muss`, and the forms `neue`, `neuen`, `neuer`, `neuem`, and `neues`. These matches do not count as distinctive evidence. This is not universal language detection or compound matching: names made entirely of excluded words, such as `Oder` or `Die Neue`, can be omitted from automatic context. Explicit canonical recall/search remains available for those names under its existing defaults.

Admitted canonical facts rank by **owner-local IDF-weighted distinctive query coverage**: terms occurring in many of that owner's canonical facts contribute less evidence, but frequent technical terms remain eligible. Other owners' facts do not affect these weights. Canonical facts covering at least `MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE` of the distinctive query terms (default `0.30`, measured without IDF weighting for this priority check) precede ordinary results. Weaker admitted canonical facts compete with ordinary results by adjusted score. Priority and semantic deduplication are applied before the three-candidate canonical limit, so weaker rows with higher IDF scores cannot exclude every qualifying fact before the merge, and near duplicates do not consume slots needed by distinct facts. Within each priority group, adjusted-score ordering is retained. A rare single-token canonical match can be admitted subject to the rarity/coverage rules below. Its admission coverage counts distinctive query terms only, so excluded generic words cannot carry it over the threshold. Multiple distinctive matches can be admitted below the coverage threshold without receiving priority.

Prefetch then deduplicates and returns up to five memories. Canonical priority preserves a qualifying fact's provenance when a similar ordinary memory has a higher adjusted score. A longer ordinary memory containing a short canonical fact can still retain its additional instructions; exact/near duplicates are removed. Ordinary-to-ordinary containment and explicit-recall deduplication retain their existing behavior.

To tune canonical matching, `MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS` completely replaces the generic-token set for both automatic canonical prefetch and explicit canonical lookup; their built-in defaults differ. `MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS` adds deployment-specific exclusions only to automatic canonical prefetch. Neither changes ordinary working/episodic prefetch. See the configuration table below for the overlap, coverage and rare-token controls.

When `mnemosyne_recall` fires, Mnemosyne also runs hybrid search: vector similarity finds semantic matches, FTS5 full-text finds keyword matches, and importance scoring boosts what matters. All three weights are tunable per-query so you can bias toward recency, relevance, or both.

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

### Machine notice filtering

`sync_turn()` uses a deterministic, best-effort text filter to detect and drop
machine-generated execution envelopes injected by Hermes as user-role turns
(background process completions, cron preambles, MCP reloads, async delegations,
and context compaction envelopes).

When an envelope is rejected, `sync_turn()` skips autosave for the user turn,
the paired assistant turn, identity-signal extraction (`_capture_identity_signals`),
and verbatim ledger capture (`_verbatim_ledger.capture`). Explicit memory-tool
writes (`remember`, `shared_remember`) and skip-context lifecycle policies remain
unaffected. The filter handles human inquiries and trailing prompts conservatively;
it is a best-effort text boundary, not a message-origin guarantee.

Provider lifecycle hooks are fail-soft. Database or disk failures during
prefetch, turn sync, session-end or automatic consolidation, and wrapper or
audit cleanup are logged and suppressed so Hermes can continue without a
lifecycle exception surfacing to the user.

The auto-sleep and session-end workers open their own `BeamMemory` on a
short-lived thread and release that thread's cached SQLite handle in `finally`,
so cleanup also runs when the sleep pass fails. The release goes through
`mnemosyne.core.connection_release.close_thread_connection`: it never imports
`mnemosyne.core.memory` (whose import-time `init_db()` would open a default
connection and displace the worker's handle), it is fail-soft like the rest of
the lifecycle, and a handle whose `close()` raised stays cached so a later call
can retry. The module first shipped in Core `4.0.0b6`; the standalone `0.7.7` package
requires Core `4.0.0` final. Legacy pairs retained through explicit
adoption can predate that module; their cleanup keeps the fail-soft fallback.

---

## Standalone release 0.7.7

This release includes the explicit legacy PM adoption path from #1217. The
minimum Core is `4.0.0` final, and every optional Core extra uses the same floor,
so a fresh install never pairs this provider with a beta.
Adoption retains the verified prior package pair and does not upgrade it to Core 4.
Release-artifact acceptance and the #1169 gate remain separate from preparation.

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

On Hermes 0.21 and later a checkout's own `venv` is only the macOS TCC anchor. The provider runs from a staged
runtime under `$HERMES_HOME/installs/<key>/environments/<generation>/venv`, which Hermes records in
`installs/<key>/facts.json`. For the launcher's checkout and for each install root above, discovery reads that
record (`<key>` is derived from the checkout path, so `installs/` is never scanned) and returns the staged
interpreter instead of the anchor. A checkout with no committed runtime keeps the venv-based behavior. A record
that exists but cannot be used (unreadable, outside that install's `environments/`, or without an executable
interpreter) makes discovery stop with a warning naming it, rather than fall back to the anchor. `--python`
still overrides all of this. A staged generation is replaceable by a Hermes update, so a wrapper installed
against it prints the persistence warning described under wrapper mode.

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

#### Explicit legacy PM adoption (0.7.7 candidate; not yet published)

This opt-in flow is for a verified existing legacy package-symlink installation
whose exact core/provider pair should become owned by Hermes' package manager
(PM). It requires a supporting Hermes PM installation and an installer from this
checkout or the prepared 0.7.7 candidate containing `--adopt-pm`. Published
`mnemosyne-hermes` **0.7.6 does not contain this option**. This is not a replacement
for normal installation or Docker/persistent side-venv wrapper setup.

The public PM integration was verified against Hermes revision
`f4ac0bea1ea50bba62886fb0b7e308e4edee8fd3`: it exposes
`pm.plugin_inputs.Candidates` and callable `pm.sync_venv`, accepts explicit plugin
candidates, and consumes their `python_dependencies`. The verified legacy pair
was core 3.15.1/provider 0.5.0, including real resolver-conflict and repair checks
with resource-monitoring warnings. This is a tested reference, not an invented
released-version floor or an all-version/all-platform guarantee. A metadata-only
dry-run does not import PM or certify these apply-time prerequisites.

Use the supporting installer's standalone tool environment for the command below;
it is distinct from the **prior virtual environment** passed to `--python`.
Do not upgrade the prior environment to obtain the installer. Here
`/path/to/installer-venv/bin/mnemosyne-hermes` means the console command installed
from this checkout or the 0.7.7 candidate, not the old provider's command.

First inspect the read-only plan, using the selected Hermes home's global argument
before `install` and the explicit prior VENV's Python:

```bash
/path/to/installer-venv/bin/mnemosyne-hermes --hermes-home /path/to/hermes-home install --adopt-pm --python /path/to/prior-venv/bin/python --core-extras embeddings --provider-extras none --dry-run
```

The extras above are **operator-chosen examples**, not autodetection: both
`--core-extras` and `--provider-extras` require an explicit choice when the prior
selection cannot be established. Confirm your actual selections first; use `none`
for no extras or comma-separated names for selected extras. Merely installed
libraries or advertised available extras do not establish what was selected.
The plan reads installed metadata without importing the prior provider/core or
activating its environment, and does not change data, models, config or bindings.
It is not a guarantee that dependency resolution or apply will succeed.

Only after reviewing the plan, consent to PM ownership with the same command
without `--dry-run`:

```bash
/path/to/installer-venv/bin/mnemosyne-hermes --hermes-home /path/to/hermes-home install --adopt-pm --python /path/to/prior-venv/bin/python --core-extras embeddings --provider-extras none
```

Adoption preserves the verified exact installed core/provider versions and chosen
extras; it does not select latest, bootstrap a beta, or upgrade to Core 4. Plan
any version upgrade separately and explicitly. Unknown or modified legacy targets
are refused. Externally owned persistent side-venv/directory wrappers remain
untouched, even with `--force`; this flow does not convert them.

Adoption also refuses legacy child-profile links that point directly at the prior
package, including opted-in profiles; it will not leave those OLD bindings behind.
Before retrying, inspect each affected link and its ownership. Deliberately
re-point an appropriate owned profile link to the root
`<hermes-home>/plugins/mnemosyne` binding, or explicitly remove only that owned
profile link if that is your intended configuration. No profile-link removal is
automatic, and `--force` does not bypass this refusal.

A genuine resolver conflict **before PM selection commit** leaves config, facts,
the selected application generation and plugin/profile bindings unchanged. PM
commit and the later local binding swap are **not one crash-atomic transaction**.
An ambiguous admission error may occur after commit: selection may have changed,
the candidate declaration is retained, and owned legacy bindings remain unchanged
on that admission-error path. Inspect the reported PM state before taking another
action; do not infer a global rollback. A subsequent local binding failure is
reported as partial state, not success. Owned candidate/cache files can persist
even after a resolver conflict. Restart Hermes after successful adoption.

#### Wrapper validation timeout

For a wrapper install, `--import-timeout SECONDS` sets the timeout for both
selected-`--python` validation probes. It defaults to 60 seconds. Use it when
that selected interpreter needs longer to resolve its site-packages or import
`mnemosyne_hermes`. `SECONDS` must be positive and finite: zero, negative,
`NaN`, and infinite values are rejected.

First create a persistent Mnemosyne side venv outside Hermes' replaceable core/PM
venv, using a Python with the same major/minor version as Hermes. Install a
compatible released `mnemosyne-hermes`/core pair into that side venv: the
integration requires `mnemosyne-memory[embeddings]`; choose
`mnemosyne-memory[all]` only if its additional local-LLM dependencies are needed.
Follow the [persistent side-venv wrapper setup](../../docs/hermes-integration.md#persistent-side-venv-wrapper-mode)
for the package/profile selection and installation steps.

```bash
mnemosyne-hermes install --mode wrapper --python /path/to/persistent/mnemosyne-venv/bin/python --import-timeout 90
mnemosyne-hermes install --mode wrapper --python /path/to/persistent/mnemosyne-venv/bin/python --no-bootstrap --import-timeout 90
```

The second form still validates the selected wrapper interpreter; `--no-bootstrap`
only skips automatic package installation. `mnemosyne-hermes status` intentionally
uses its fixed 60-second validation policy, independent of an install-time override.

The installer also deploys the bundled `mnemosyne-memory-override` skill to
`$HERMES_HOME/skills/memory/mnemosyne-memory-override/SKILL.md`, so its
description appears in Hermes' skills index. The skill is a stricter, optional
policy that routes all durable storage to Mnemosyne, and it applies only when the
user consciously selects it: installation, the skill showing up in the index, or
an ordinary memory task is not that selection. By default, bulk durable facts go
to Mnemosyne and small standing rules and context anchors may stay in the
always-injected memory tier. Existing user-edited copies are skipped by default;
`install --force` overwrites the skill only after writing `SKILL.md.bak` next to it.

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
| `MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE` | `0.30` | Minimum query coverage for ordinary working/episodic prefetch, and unweighted distinctive-query coverage required for canonical priority. Canonical multi-token admission can remain below this threshold and compete by adjusted score. For canonical single-token admission, coverage is exempt for a slot whose entire body is one CJK bigram that also appears in the query, so a short topical fact is reachable from a contextual sentence; separators around that bigram (`。`, `／`, `｡`, quotes, brackets) are removed before the check, so `部署／` qualifies while `部署／計画` still counts as two units. Automatic prefetch also runs the same run-local U+3005 iteration-mark predicate as explicit canonical recall before scoring, so a candidate that clears these thresholds can still be excluded when its iteration-mark evidence does not match a CJK run. Automatic prefetch still applies the rarity cap below, and explicit canonical recall needs no coverage fraction at all |
| `MNEMOSYNE_POLYPHONIC_RECALL` | `0` | Route recall through the polyphonic engine (RRF over vector/graph/fact/temporal voices); prefetch then admits raw transcripts past the lexical gate without the linear score floors |
| `MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY` | `1` | Maximum canonical document frequency that permits a one-token match (`0` disables the exception). A whole-body CJK bigram match counts as a one-token match, so it stays subject to this cap |
| `MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS` | path-specific built-in canonical set | Complete replacement for the canonical generic-token set used by automatic and explicit canonical lookup; does not affect working/episodic prefetch |
| `MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS` | _(empty)_ | Extra owner/deployment terms added to automatic canonical prefetch only |
| `MNEMOSYNE_DEFAULT_SCOPE` | `session` | Default scope for remember (`global` enables cross-session immediate recall) |
| `MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION` | `0` | Enables execution of `mnemosyne_resolve_conflicts` for global-scope memories. It does **not** run automatically. `dry_run` defaults to `False` (apply); set `True` to preview without superseding memories or emitting an apply audit. Default-false `llm_eval` keeps previews deterministic; explicit `dry_run=True,llm_eval=True` can incur LLM costs and write cost records. Automatic resolution during sleep is not yet enabled. |
| `MNEMOSYNE_CROSS_SESSION_MIN_GAP_HOURS` | `0` | Minimum hour-gap between cross-session candidate pairs before they are flagged for contradiction (relaxed to 0 for independent threads) |
| `MNEMOSYNE_CROSS_SESSION_MAX_CANDIDATES` | `10000` | Per-bank row cap for the cross-session candidate scan (bounds the pairwise comparison and embedding load); truncation is reported in the tool result |
| `MNEMOSYNE_CROSS_SESSION_MAX_LLM_VALIDATIONS` | `20` | Per-run cap on pair validations during apply or an explicit LLM preview. A validation may retry its transport; this is not an exact HTTP-request or dollar cap. The tool reports `llm_cap_reached` when the cap was hit (alongside `candidates_truncated` for the row cap), so operators can tell when candidate pairs remain pending for a later explicit run |

Both the standalone `mnemosyne_hermes` provider and the bundled
`hermes_memory_provider` copy expose the same `dry_run`/`llm_eval` resolver
arguments during migration; this is not a recommendation to install the bundled
copy. The existing core `BeamMemory.resolve_cross_session_conflicts` signature
supports both arguments. The current minimum core/provider pair is core 4.0.0
and provider 0.7.7; no fallback retry against an incompatible core signature is performed.

An explicit LLM preview overrides `MNEMOSYNE_LLM_CONFLICT_DETECTION` for that
preview only, never the default-off cross-session gate. Apply still honors the
LLM detection flag and retains the existing heuristic behavior when it is off.
Apply and evaluated preview reserve **one reflection invocation**, not one slot
per pair or a dollar budget. Exhausted quota or a cron-disabled reflection
context returns `skipped` before core execution. This corrects the standalone
provider's former evaluated-preview bypass. Missing/false `llm_eval` previews
remain deterministic, with no LLM calls, cost records or quota reservation even
under exhausted/cron guards. Reservations are conservative and not refunded
when the resolver is disabled, a verdict vetoes the pair, parsing fails or the
transport fails. A missing resolver method returns `unavailable` without
reservation. Cost logging occurs only after a valid verdict; absent cost records
after a failed call do not prove a real endpoint charged nothing.

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

42 tools. All surfaced through Hermes' tool system.

**Core memory:** `remember`, `recall`, `sleep`, `stats`, `get`, `update`, `forget`, `invalidate`, `validate`, `resolve_conflicts`

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

Lifecycle failures (prefetch, turn sync, session-end, cleanup) are logged and never surface as user-facing errors. The one deliberate exception is canonical-write ownership: when ownership cannot be established because the active profile mismatches or no active profile can be resolved, a canonical remember or forget — as well as mutating `mnemosyne_task_progress` actions (`set` and `clear`, while `get` and `list` reads remain ungated) — returns a structured refusal (`canonical_owner_mismatch` / `canonical_profile_unavailable`) to the tool caller instead of writing an unowned row. A refusal means nothing was written; it never raises into the lifecycle.

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
