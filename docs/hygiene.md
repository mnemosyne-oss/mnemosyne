# Memory Hygiene

Agent memory accumulates noise: terminal output pasted into a conversation, heartbeat pings, stack traces, one-word acknowledgements, and occasionally a leaked secret. Hygiene finds those rows and lets you delete, archive, or flag them.

The whole subsystem is deterministic. No LLM, no embeddings, no network. It is regex and arithmetic, so it runs anywhere and produces the same answer twice.

Auditing is **read-only** and always safe. Cleaning requires explicit confirmation.

---

## Noise scoring

`_score_noise` in `mnemosyne/core/hygiene.py` returns a score in `0.0`-`1.0`, where higher means more likely noise, plus the list of reasons that fired.

The score is **not additive**. Each rule raises the score to at least its own value, so one strong signal is enough and several weak ones do not compound into a false positive.

| Signal | Score | Reason |
|---|---|---|
| Empty or whitespace-only content | `1.0` | `empty_content` |
| A detected secret | `0.9` | `secret_detected:<labels>` |
| Terminal or package-manager output (`Collecting `, `npm warn`, `drwx`, `-rw-r--r--`, a whole `total 48` line, a line-start `Installing collected packages`, ...) | `0.85` | `terminal_output` |
| A stack trace (`Traceback`, `  File "`) | `0.85` | `stack_trace` |
| Matches one of the 24 built-in noise patterns | `0.8` | `noise_pattern_match` |
| Trivial keyword under 15 characters (`ok`, `done`, `ping`, `heartbeat`, ...) | `0.7` | `trivial_keyword` |
| Source is `heartbeat`, `cron`, `debug`, or `terminal` | `0.7` | `noisy_source:<source>` |
| Long unpunctuated dump: 30+ lines, 1000+ chars, few sentence breaks | `0.65` | `likely_dump` |
| Importance below `0.2` | `0.5` | `low_importance` |

One rule works in the other direction. If the content contains a value keyword (`prefer`, `always remember`, `never`, `project`, `convention`, `decision`, `architecture`, ...) the score is **clamped to at most `0.3`** so genuinely useful memories are not swept up by a coarse pattern. That clamp is skipped when a secret was detected, because a secret outranks usefulness.

Secret detection covers ten labelled patterns: API key prefixes, AWS access keys, GitHub and Slack tokens, Google API keys, JWTs, secret and env assignments, private key blocks, and connection strings with embedded credentials.

### Suggested action

| Condition | Suggestion |
|---|---|
| Any secret detected | `flag` |
| Score at or above `0.8` | `delete` |
| Score at or above `0.5` | `archive` |
| Otherwise | `keep` |

**Secrets are never suggested for automatic deletion.** They are flagged for a human, because deleting a memory containing a leaked credential destroys the evidence you need to rotate it.

Note also that a row containing a secret is always included in audit output regardless of `--min-score`.

## The three actions

| Action | What it does | Reversible |
|---|---|---|
| **delete** | `DELETE FROM <table> WHERE id = ?` | **No.** Only a 200-character preview survives, in the audit log |
| **archive** | Sets `importance = 0` and records `_archived`, `_archived_at`, `_archive_reason`, and `_original_importance` in metadata. Content is untouched | **Yes**, via `hygiene restore` |
| **flag** | Metadata only: `_hygiene_flagged` and the reason. No importance or content change | Nothing undoes it, but nothing was changed |

Archive is the interesting one. The row stays fully intact and searchable but drops out of importance-weighted ranking, so it stops polluting recall without being destroyed. `_original_importance` is what makes restoring exact.

`restore` reads the **current** row's metadata rather than the audit log snapshot, pops `_original_importance` back into place, and strips the archive markers. Rows archived before `_original_importance` was preserved restore to `0.5`.

Every candidate produces one `hygiene_audit_log` row, including ones left alone, which are logged as `kept`.

### `hygiene_audit_log`

`id`, `memory_id`, `table_name`, `action` (`deleted` / `archived` / `flagged` / `kept`), `reason` (JSON array), `noise_score`, `secret_flags` (JSON array of labels, not values), `original_content_preview` (200 chars), `original_metadata` (pre-action JSON), `timestamp`, `session_id`.

The table is created lazily on the first real clean. `session_id` is always NULL by design.

## CLI

```bash
# Read-only. Always safe.
mnemosyne hygiene audit [--limit N] [--offset N] [--all [--batch-size N]]
                        [--min-score F] [--json]

# Audit-log and noise summary
mnemosyne hygiene status [--limit N] [--json]

# Act on candidates. Dry run unless --confirm.
mnemosyne hygiene clean --action delete|archive|flag [--confirm] [--dry-run] <candidates.json>

# Undo archives
mnemosyne hygiene restore [--limit N]
```

Defaults: `audit` uses `--limit 200` and `--min-score 0.3`, and prints the top 20 candidates in human mode. `clean` is a **dry run by default**; `--confirm` both confirms and disables the dry run.

Two sharp edges worth knowing:

- `--confirm --dry-run` in that order stays a dry run, because `--dry-run` is applied second.
- `--action` is read positionally without validation, so a typo becomes an unrecognized action and every candidate silently falls through to `kept`. Check the printed counts.

A typical session:

```bash
mnemosyne hygiene audit --min-score 0.5 --json > candidates.json
# read it, decide
mnemosyne hygiene clean --action archive candidates.json            # dry run
mnemosyne hygiene clean --action archive --confirm candidates.json  # apply
```

Archive first, delete later. Archiving is reversible and gets the noise out of ranking immediately, which is almost always what you actually wanted.

> The CLI has **no `--bank` flag**. All four subcommands operate on the default database at `$MNEMOSYNE_DATA_DIR/mnemosyne.db`. For other banks, use the MCP tools, which are bank-aware.

## MCP tools

**`mnemosyne_hygiene_audit`** is read-only. Parameters: `limit` (200), `offset` (0), `scan_all` (false), `batch_size` (1000), `min_score` (0.3), `tables`, `bank` (`default`). Nothing is required. Opens the database read-only, so it cannot modify anything.

**`mnemosyne_hygiene_clean`** requires `candidates_json`. Also takes `action` (`delete` / `archive` / `flag` / `keep`, default `keep`), `confirm` (false), and `bank`.

The confirm gate is `dry_run = not confirm`. There is no independent `dry_run` parameter over MCP, so `confirm=false` is always a dry run and `confirm=true` always applies. The underlying function double-gates: it returns before touching the database on a dry run, and refuses to proceed if it somehow receives a non-dry-run without confirmation.

Candidates are validated strictly. Note that **`scratchpad` can be audited but not cleaned**: only `working_memory`, `memories`, and `episodic_memory` are accepted as clean targets.

## Scanned tables

By default `working_memory`, `memories`, and `episodic_memory`. Missing tables are skipped rather than erroring. `scratchpad` can be added explicitly for auditing.

The scan reads only `id`, `content`, `source`, `timestamp`, `session_id`, and `importance`. Metadata is deliberately excluded so a read-only health scan never loads it.

## Prevention

Cleaning up after the fact is the second-best option. Two settings stop noise being stored at all:

| Variable | Effect |
|---|---|
| `MNEMOSYNE_WRITE_CLASSIFIER` | `strict` rejects noisy writes, `warn` logs them, `off` disables the gate |
| `MNEMOSYNE_IGNORE_PATTERNS` | Comma-separated regexes; matching content is never stored |

The `paranoid` config profile sets both, including a pattern list covering passwords, tokens, and key blocks. See [Configuration profiles](profiles.md).

## See also

- [Configuration profiles](profiles.md)
- [Generated configuration reference](api/configuration.mdx)
- `docs/rfc/noise-remediation-rnd.md` for the R&D behind this subsystem and the remaining gaps
