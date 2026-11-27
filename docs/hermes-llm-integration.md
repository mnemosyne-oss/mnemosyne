# Hermes Auxiliary LLM Integration

When Mnemosyne runs as a Hermes memory provider, it can optionally route its
LLM-backed memory operations — both consolidation (sleep) **and** structured
fact extraction — through Hermes' authenticated auxiliary client. This lets a
Hermes user reuse their configured provider (including OAuth-backed providers
such as `openai-codex`) without giving Mnemosyne its own credentials.

## Why

Mnemosyne's standalone LLM path expects an OpenAI-compatible URL plus an API
key (`MNEMOSYNE_LLM_BASE_URL` / `MNEMOSYNE_LLM_API_KEY`). That cannot reach
OAuth/session-backed providers like ChatGPT/Codex. Hermes already authenticates
those providers through `agent.auxiliary_client.call_llm(task="compression", ...)`,
so the cleanest fix is for Mnemosyne to delegate to that helper when it is
running inside Hermes — without dragging Hermes' auth into Mnemosyne core.

## Behavior

The host backend is **disabled by default** to preserve existing standalone
behavior after upgrading. To opt in:

```bash
export MNEMOSYNE_HOST_LLM_ENABLED=true
```

When enabled and a host backend is registered (which happens automatically
when Mnemosyne is loaded as a Hermes memory provider):

```text
0. Host backend (Hermes auxiliary client).
   - On success: return the host text.
   - On failure (errors, empty response, no extractable content):
     skip MNEMOSYNE_LLM_BASE_URL entirely. Fall to the local GGUF path,
     then return None / [].
1. Remote OpenAI-compatible API, if MNEMOSYNE_LLM_BASE_URL is set and
   no host call was attempted.
2. Local llama-cpp-python / ctransformers GGUF (MiniCPM5-1B by default).
3. Return None (consolidation) or [] (extraction) — caller falls back to
   the existing non-LLM path.
```

The "skip remote on host failure" rule prevents Mnemosyne from accidentally
routing memory content to a stale `MNEMOSYNE_LLM_BASE_URL` the user forgot
to clear after switching to Hermes.

When `HOST_LLM_ENABLED=true` but no backend is registered (e.g., the env var
is set in a non-Hermes process), Mnemosyne treats the host as "not attempted"
and proceeds with the existing remote/local fallback chain.

## Configuration

```bash
# Required: opt in to the host backend.
MNEMOSYNE_HOST_LLM_ENABLED=true

# Optional: override the host default compression provider/model for
# Mnemosyne calls. Leave unset to inherit Hermes' auxiliary.compression
# resolution. These are NOT credentials — Hermes still owns auth, OAuth
# refresh, and transport.
MNEMOSYNE_HOST_LLM_PROVIDER=openai-codex
MNEMOSYNE_HOST_LLM_MODEL=gpt-5.1-mini

# Optional: prompt context budget when the host is the chosen path.
# Default 32000. The existing MNEMOSYNE_LLM_N_CTX (default 2048) is
# calibrated for the local GGUF model and is far too small for typical Codex/GPT
# context windows — using it as the host budget produces wastefully many
# small chunks and lossy multi-chunk summaries.
MNEMOSYNE_HOST_LLM_N_CTX=32000

# Existing global gate. When false, ALL LLM-backed memory operations
# are disabled, including the host path.
MNEMOSYNE_LLM_ENABLED=true
```

To control the default host model without Mnemosyne-specific overrides,
configure Hermes itself:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  compression:
    provider: auto       # default; uses main provider/model first
    model: ""            # empty inherits Hermes behavior
    timeout: 15          # per attempt; Hermes may retry internally
```

The `timeout` value is **per-attempt**. Hermes can retry internally for
auth refresh, payment fallback, or provider fallback, so the total
wall-clock can exceed the configured timeout on cold start.

## Codex/ChatGPT subscriptions

For OAuth-backed providers like `openai-codex`, **do not** point
`MNEMOSYNE_LLM_BASE_URL` at `https://chatgpt.com/backend-api/codex`. That
endpoint is not an OpenAI-compatible API-key endpoint; the host backend is
the right path. Configure the provider through your normal Hermes login
(`hermes login` / `hermes config`) and let Mnemosyne route through Hermes.

## Fact-extraction determinism

Fact extraction uses `temperature=0.0` so re-ingesting the same content
produces the same facts. This avoids near-duplicate writes to the facts
table when the same conversation is processed twice. Consolidation continues
to use `temperature=0.3` — paraphrasing variance is acceptable there.

## Session shutdown

Mnemosyne's `on_session_end()` hook runs sleep/consolidation in a daemon
thread with a 15-second join timeout. If consolidation cannot finish in time
(e.g., a slow host LLM call), the join returns and Hermes shutdown proceeds
unblocked; the daemon thread continues in the background and is reaped when
the process exits. A warning is logged when the timeout fires:

```text
WARNING  Mnemosyne session-end sleep timed out after 15s — consolidation deferred
```

This protects Hermes from getting stuck on a slow LLM provider during
session shutdown without losing the chance for consolidation to complete on
faster paths.

## Scheduled consolidation from the standalone CLI

A standalone `mnemosyne sleep` process can use a configured OpenAI-compatible
endpoint or a local GGUF model. The Hermes auxiliary backend requires a
registered provider in the same process. For a scheduled CLI job, give the job
the target bank's environment and configure its own endpoint. For example:

```bash
export MNEMOSYNE_LLM_ENABLED=true
export MNEMOSYNE_LLM_BASE_URL=http://localhost:8080/v1
export MNEMOSYNE_LLM_MODEL=my-model
mnemosyne sleep --all-sessions
```

Set `MNEMOSYNE_LLM_API_KEY` in the job's environment if the endpoint requires
one. The command processes rows eligible under the normal age cutoff. Use
`--force` when you intend to include newer eligible rows as well. Check the
completed run's `llm_used` count and per-session `method` to see how many
summaries used an LLM. With no usable LLM the command can still succeed using
AAAK and logs a fallback warning. A `--dry-run` previews eligible work
without invoking an LLM. Its method and usage counts do not predict the actual
backend.

For consolidation inside Hermes, set `host_llm_enabled: true` in
Mnemosyne's `config.yaml` or export `MNEMOSYNE_HOST_LLM_ENABLED=true` before
starting Hermes. Mnemosyne reads this setting at startup. When both are
present, the YAML value takes precedence.

## For other agents

Any host that wants to expose its authenticated LLM to Mnemosyne can register
its own backend through the same tiny interface:

```python
from mnemosyne.core.llm_backends import LLMBackend, set_host_llm_backend

class MyAgentBackend:
    name = "my-agent"

    def complete(self, prompt, *, max_tokens, temperature, timeout,
                 provider=None, model=None):
        # Route through your own authenticated client and return text-or-None.
        ...

set_host_llm_backend(MyAgentBackend())
```

This mirrors the pattern Hermes uses today and avoids per-agent forks of
Mnemosyne core.


### Consolidation output completeness

Consolidation removes only recognized summary headers or an exact known
instruction echo. Factual sentences beginning with words such as “The following
memories confirm …” remain intact, including prices, dates and factual bullets.
This sanitation is confined to summary calls and does not filter fact-extraction
output or change provider prefetch admission.

An empty sanitized host response tries the existing local fallback, without
sending the memories to a separate remote URL after a host attempt. If any
input was omitted by chunk budgeting, any chunk remains unusable, or the final
merge of chunk summaries fails, the summary API returns no LLM summary. The same completeness checks apply before and after the sleep path's own
chunking. The caller then uses its existing complete-input fallback rather
than publishing one successful chunk as the summary of the whole input. No retry budget or
backend-routing policy is added by this correction.
