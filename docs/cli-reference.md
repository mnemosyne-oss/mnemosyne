# CLI Reference

Every command registered in `mnemosyne/cli.py`. The built-in `mnemosyne --help` covers a subset and omits the aliases, so this is the complete list.

```bash
mnemosyne --help          # note: use the flag, not `mnemosyne help <cmd>`
```

> **Watch out:** `export` and `remember` treat `--help` as a positional argument. `mnemosyne remember --help` stores the literal string `--help` as a memory. Always use `mnemosyne --help`.

---

## Memories

| Command | Usage |
|---|---|
| `store`, `remember` | `store <content> [source] [importance]` |
| `recall`, `search` | `recall <query> [top_k]` |
| `update`, `edit` | `update <id> <content> [importance]` |
| `delete`, `forget` | `delete <id>` |
| `stats` | Working and episodic counts, BEAM tier breakdown |
| `sleep`, `consolidate` | `sleep [--force] [--all-sessions] [--dry-run]` |

The three flags on `sleep` are parsed but undocumented in the built-in help. `--force` skips the age threshold, `--all-sessions` consolidates across inactive sessions, `--dry-run` reports without writing.

## Health and repair

| Command | Usage |
|---|---|
| `diagnose` | `diagnose [--fix] [--dry-run] [--repair-vec-working]`. PII-safe. `--fix` installs missing dependencies |
| `doctor` | `doctor [--db PATH \| --bank NAME] [--format json\|markdown\|both]`. Bounded, read-only health report |
| `repair` | `repair --report REPORT.json --select working_memory:ID [--apply]`. Applies one narrow doctor-gated fix |
| `verify` | `verify [db_path] [--quick]`. Integrity check |
| `reindex` | `reindex [--model NAME] [--dry-run] [--yes] [--no-backup]`. Re-embeds everything and rebuilds the sqlite-vec tables |

`reindex` is the recovery path for a vector dimension mismatch. It is synchronous, backs up first unless told otherwise, and prompts unless `--yes`. Its `--dry-run` option prints a rebuild plan without writing.

For automation, do not treat a non-zero exit from a non-dry-run `mnemosyne reindex` as success: it means the vector rebuild did not complete. Likewise, non-dry-run `mnemosyne diagnose --repair-vec-working` exits non-zero unless the requested repair reaches `repaired`; its `--dry-run` mode reports what it would repair without writing.

`doctor` and `repair` are the only commands that do not create the data directory as a side effect.

## Backup and restore

| Command | Usage |
|---|---|
| `backup` | `backup [output_dir]`. Compressed snapshot |
| `restore` | `restore <backup.db.gz>` |
| `backups` | `backups [backup_dir]`. List available snapshots |

## Import and export

| Command | Usage |
|---|---|
| `export` | `export [--include-sync-events] [file.json]` |
| `import` | `import <file.json>` |
| `import-hindsight` | `import-hindsight <file\|url> [bank]` |

Import is idempotent: annotation collisions are skipped rather than aborting the run, so re-running is safe.

## Banks

| Command | Usage |
|---|---|
| `bank` | `bank list\|create\|delete [name]` |
| `migrate` | `migrate [--bank <name>]`. Adds newer schema tables to an older bank |

`bank list` hides the virtual `default` bank until its database file exists.

## Sync

| Command | Usage |
|---|---|
| `sync-init` | `sync-init --db-path <path> [--claim-existing --yes]`. Prepares a dedicated shared-surface database |
| `sync` | `sync --db-path <path> --remote <url> [--mode push\|pull\|bidirectional]` |
| `sync-serve`, `sync-server` | `sync-serve --db-path <path> [--port 8765] [--host 127.0.0.1] [--api-key\|--api-key-file] [--jwt-secret\|--jwt-secret-file] [--tls-cert --tls-key]` |
| `sync-status` | `sync-status --db-path <path> [--remote <url>] [--json]` |
| `sync-generate-key` | Prints a fresh encryption key |

Both `--remote` and `--db-path` are **required** on `sync`. The subcommands are hyphenated top-level commands, not `sync <subcommand>`; `mnemosyne sync serve` is not a command.

Point `--db-path` at a dedicated shared-surface database, never a private one. See [Mnemosyne Sync](sync/index.md).

## Maintenance

| Command | Usage |
|---|---|
| `hygiene` | `hygiene audit\|status\|clean\|restore`. See [Memory Hygiene](hygiene.md) |
| `profile` | `profile list\|apply\|show\|create`. See [Configuration Profiles](profiles.md) |
| `config` | `config reload\|get\|set\|migrate` |

The built-in help lists only `hygiene audit|clean`; `status` and `restore` exist too.

## Servers

| Command | Usage |
|---|---|
| `mcp` | `mcp [--transport stdio\|sse\|streamable-http\|http] [--host 127.0.0.1] [--port 8080] [--path /mcp] [--json-response] [--env-file FILE] [--bank NAME]`. Starts the MCP server |

stdio is the default transport. `sse` and `streamable-http` are HTTP transports; a non-loopback bind requires `MNEMOSYNE_MCP_TOKEN`. `streamable-http` (alias `http`) is the native MCP Streamable HTTP transport: clients POST JSON-RPC straight to `--path` (default `/mcp`) with no separate `/messages` route to proxy. Add `--json-response` to force JSON-only responses instead of the default SSE-upgrade streaming. A non-loopback `streamable-http` bind also requires `MNEMOSYNE_MCP_ALLOWED_HOSTS` (see below); `sse` requires only the token.

### Streamable HTTP Host/Origin policy

The Streamable HTTP transport applies a Host/Origin policy on **non-loopback**
binds (DNS-rebinding protection). Loopback binds (`127.0.0.1`, `localhost`,
`::1`) keep the SDK's built-in defaults and ignore these variables.

Streamable HTTP serves the existing local Mnemosyne/SQLite store — no external
database is involved. Binding non-loopback exposes the selected local memory
bank to network clients, so treat the token and the Host/Origin gates below as
the boundary between the local store and the network.

- `MNEMOSYNE_MCP_ALLOWED_HOSTS` — **required** to start a non-loopback server.
  Comma-separated `Host` header values clients will present. Each value is an
  exact name or a `name:*` pattern covering any port. Any request whose `Host`
  is not listed is rejected with HTTP 421.
- `MNEMOSYNE_MCP_ALLOWED_ORIGINS` — **optional**. Comma-separated browser
  `Origin` values to allow. Requests with **no** `Origin` header always pass;
  any `Origin` not listed is rejected with HTTP 403.

**Single value vs. list.** Both variables accept one value or several,
comma-separated (whitespace is trimmed, empty entries ignored):

```bash
# single
export MNEMOSYNE_MCP_ALLOWED_HOSTS="mnemosyne.k.example.com:*"
# list
export MNEMOSYNE_MCP_ALLOWED_HOSTS="mnemosyne.k.example.com:*, mnemosyne.example.org"
export MNEMOSYNE_MCP_ALLOWED_ORIGINS="https://inspector.example.com, https://app.example.com"
```

**SDK / CLI clients** (curl, MCP SDKs, Claude Code, etc.) send no `Origin`
header, so they are unaffected by `MNEMOSYNE_MCP_ALLOWED_ORIGINS`. They only
need their `Host` listed. Include the port wildcard (`name:*`) because clients
and load balancers frequently send `host:port`.

**Browser clients** (e.g. MCP Inspector) send an `Origin` header, so in
addition to a matching `Host` you must add the browser's origin to
`MNEMOSYNE_MCP_ALLOWED_ORIGINS`, otherwise they get HTTP 403. Note the SDK does
**not** support a bare `*` wildcard — list each origin explicitly.

**Reverse proxies.** The `Host` header the server sees is whatever the proxy
forwards (nginx `proxy_set_header Host $host` passes the original hostname).
If multiple public hostnames or ports route to the same server, list each one;
the same applies to `Origin` when browser clients arrive via different hosts.
Bare `*` is never a valid entry.

Example for a deployment behind an nginx ingress on one hostname:

```bash
MNEMOSYNE_MCP_TOKEN=<token> \
MNEMOSYNE_MCP_ALLOWED_HOSTS="mnemosyne.k.example.com:*" \
mnemosyne mcp --transport streamable-http --host 0.0.0.0 --port 8080
```

## Aliases

`remember`=`store`, `search`=`recall`, `edit`=`update`, `forget`=`delete`, `consolidate`=`sleep`, `sync-server`=`sync-serve`.

## See also

- [Getting Started](getting-started.md)
- [API Reference](api-reference.md) for the Python API
- [Generated configuration reference](api/configuration.mdx)
