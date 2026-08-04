# L3 Persona Tier

**Status:** partially wired. Read [What is not wired](#what-is-not-wired) before relying on this.

The persona tier holds a small set of durable behavioural facts about the user. It is a store, not a prompt channel: promotion writes a row to the `memoria_persona` table, while [prompt injection](#prompt-injection) reads the opt-in `persona.md` file and never queries the table. Promoting a fact changes nothing about your prompts until something regenerates that file.

The intended role is a set of facts that are always present rather than competing for recall, and that role has a cost: anything that reaches the injected file consumes prompt budget on every turn, so the tier is meant to stay small.

---

## Storage

Table `memoria_persona`, created in `init_beam` (`mnemosyne/core/beam.py:988`):

| Column | Type |
|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` |
| `session_id` | `TEXT DEFAULT 'default'` |
| `tier` | `TEXT NOT NULL CHECK(tier IN ('permanent','long_term','working'))` |
| `topic` | `TEXT NOT NULL` |
| `content` | `TEXT NOT NULL` |
| `confidence` | `REAL DEFAULT 0.7` |
| `source_memory_id` | `TEXT` (soft reference, no FK) |
| `created_at` | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` |
| `last_reinforced_at` | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` |
| `reinforcement_count` | `INTEGER DEFAULT 0` |
| `promotion_reason` | `TEXT` |

Indexed on `(session_id, tier)` and `(tier, topic)`.

## Tiers

Three values are accepted: `permanent`, `long_term` (the default on promote), and `working`.

**Tier affects `mnemosyne_persona_list` ordering only.** Listing sorts `permanent` before `long_term` before `working`, then by `reinforcement_count` descending. Nothing else consults the tier: the prompt path reads `persona.md` and never queries this table.

Tier does **not** cause retention or expiry. There is no code anywhere that deletes, ages out, or reduces the confidence of a `memoria_persona` row. The only writes are insert on promote, delete on demote, and the reinforcement counter update. Descriptions elsewhere in the codebase that call `permanent` "never evicted" or `long_term` "reinforcement-driven decay" describe an intended design, not current behaviour: nothing is evicted, because no eviction exists.

## Operations

Implemented by `PersonaAdapter` in `hermes_memory_provider/persona_adapter.py`, duplicated at `integrations/hermes/src/mnemosyne_hermes/persona_adapter.py`. All methods return JSON strings and never raise.

| Operation | Behaviour |
|---|---|
| **promote** | Reads content from `working_memory`, falling back to `episodic_memory`. Derives `topic` from `memoria_timelines` via `source_memory_id`, else `"general"`. Errors if the content is empty. Default tier `long_term`. |
| **demote** | Not a plain delete. Inserts a tombstone into `memoria_preferences` recording `[demoted from <tier>] <content>` and the reason, then deletes the `memoria_persona` row. Both in one transaction. |
| **list** | Optional `tier` and `topic` filters. Returns rows in tier order (`permanent` first), then by `reinforcement_count` descending. |
| **reinforce** | `reinforcement_count += 1` and `last_reinforced_at = CURRENT_TIMESTAMP`. Returns an error if the id does not exist. |

Reinforcement feeds two consumers: the list ordering above, and the regeneration trigger, which uses `MAX(last_reinforced_at)` as its watermark.

## MCP tools

`mnemosyne_persona_promote`, `mnemosyne_persona_demote`, `mnemosyne_persona_list`, `mnemosyne_persona_reinforce`.

> **These four are advertised over MCP but are not callable over MCP.** Their schemas are in `ALL_TOOL_SCHEMAS`, so an MCP client sees them in `list_tools`, but there is no entry in `_TOOL_HANDLERS`, so calling one returns `Unknown tool`. They work only through the Hermes plugin, which routes any `mnemosyne_persona_*` prefix to the adapter.

There is no CLI surface for any of them.

## Prompt injection

`HermesPersonaPromptMixin` in `mnemosyne/integrations/hermes_persona_prompt.py` appends a block headed `# L3 Persona (Active Behavioral Rules)` to the system prompt.

The injected content comes from a **markdown file on disk**, not from the `memoria_persona` table. The file is read at prompt-build time and cached on its mtime, so a warm prompt does no file IO. When persona is disabled the function returns early with no file access at all. Any error reading the file degrades to an empty block.

Injection applies a word-based cap: `token_cap * 0.75` words. On overflow it truncates and then trims back to the last `## ` section boundary so a section is never cut mid-way, appending a note that the content was truncated.

## Configuration

All five variables are read with `os.environ.get` at **module import time**. Setting them afterwards has no effect.

| Variable | Default | Controls |
|---|---|---|
| `MNEMOSYNE_PERSONA_ENABLED` | `false` | Master switch for prompt injection |
| `MNEMOSYNE_PERSONA_FILE` | `~/.hermes/memory/persona.md` | The markdown file written and injected |
| `MNEMOSYNE_PERSONA_TOKEN_CAP` | `1500` | Prompt budget for the injected block |
| `MNEMOSYNE_PERSONA_INTERVAL` | `50` | New working memories before the regeneration trigger fires |
| `MNEMOSYNE_PERSONA_DAILY_SYNC_HOUR` | `3` | UTC hour for the daily trigger. Any value outside 0-23 disables it |

> **`config.yaml` does not reach the persona code.** Four of these keys exist in `ENV_VAR_MAP`, but neither the persona module nor the prompt mixin consults `MnemosyneConfig`, so only the environment variables take effect. The defaults declared in `config.py` also disagree with the real ones above (it declares `persona_enabled: true`, `persona_token_cap: 500`, `persona_interval: 10`). Trust this table and the environment. `MNEMOSYNE_PERSONA_FILE` has no config key at all.

## What is not wired

Being explicit, because the surrounding docstrings are optimistic:

1. **Nothing calls `mnemosyne/core/persona.py`.** `PersonaExtractor`, `PersonaTriggers`, `render_persona_markdown`, and `write_persona_file` have no callers outside the test suite. There is no scheduler, no sleep hook, and no CLI command that regenerates `persona.md`. The five trigger conditions and the daily-sync hour are library plumbing that a caller must drive.
2. **The file the injector reads must be produced by you** or by your own code calling `write_persona_file`. Promoting a fact into `memoria_persona` does not update `persona.md`, so promotion alone changes nothing about your prompts.
3. **No decay or eviction**, as described under [Tiers](#tiers).
4. The render-side and injection-side token caps use **different algorithms** (word count times 1.3 versus a 0.75 word multiplier) and disagree slightly on what fits.

If you want the tier working end to end today, the shape is: promote facts through the Hermes tools, then call `write_persona_file` yourself on whatever schedule you want, with `MNEMOSYNE_PERSONA_ENABLED=1` and `MNEMOSYNE_PERSONA_FILE` pointing at the same path.

## See also

- [Architecture](architecture.md) for where the tier sits relative to BEAM
- [Generated configuration reference](api/configuration.mdx)
- `docs/roadmap-layered-agent-memory.md` for the L0-L4 model this tier is named after
