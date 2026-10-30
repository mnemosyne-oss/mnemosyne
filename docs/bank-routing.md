# Per-Chat Bank Routing (Hermes integration)

One database per conversation partner instead of one per profile.

The Mnemosyne Hermes adapter can route every turn into a separate *bank* — a
separate SQLite database — based on the **platform identity** of the turn (its
session key / chat ID). Two people in two chats do not share a memory store;
a group, a CLI run and a cron job each get their own.

Routing is **off by default**. It is a deliberate act: a map file must exist
and must map somebody. An operator who never creates the file gets exactly the
behaviour of a provider without this feature — same database, same recall, no
quarantine directory, nothing created on disk.

---

## Enabling it

Create the map file and point the provider at it.

| Location | Path |
|---|---|
| Default | `<data dir>/banks_map.json` |
| Explicit | wherever `MNEMOSYNE_BANKS_MAP` points |

The data directory is `MNEMOSYNE_DATA_DIR`, or `<hermes_home>/mnemosyne/data`
when that variable is unset.

```json
{
  "_default": "vela",
  "_unknown": "quarantine",
  "agent:main:telegram:dm:111": "alice",
  "agent:main:telegram:dm:222": "bob",
  "agent:main:telegram:group:-1004301699476": "shared"
}
```

The file is read per turn (with an mtime cache), so a change takes effect
without a restart. A file that cannot be parsed is **not** treated as "routing
is off" — see *Fail-closed rules* below.

### Key format

```
agent:<profile>:<platform>:<chat_type>[:<chat_id>]
```

| Part | Meaning |
|---|---|
| `agent` | The gateway namespace. Only `agent:<profile>` keys are chat identities. |
| `<profile>` | Hermes profile, e.g. `main`. |
| `<platform>` | e.g. `telegram`. |
| `<chat_type>` | `dm`, `group`, `channel` or `thread`. |
| `<chat_id>` | The chat identifier; optional, so a whole type can be mapped. |

A key that is **not** a chat identity — `cron:nightly`, an internal run, a
state ID — is not routed as an "unknown chat". It belongs to nobody and nobody
else reads it, so it gets `_default`.

### How a key is matched

The **longest** matching prefix wins, and the match stops at a `:` boundary:

| Key | Turn key | Result |
|---|---|---|
| `agent:main:telegram:dm:111` | `agent:main:telegram:dm:111` | that bank |
| `agent:main:telegram:dm:111` | `agent:main:telegram:dm:1112` | **no match** — `dm:1112` is a different chat |
| `agent:main:telegram:dm:111` | `agent:main:telegram:dm:111:thread:7` | that bank (the key is a component prefix) |
| `agent:main:telegram:group:` | `agent:main:telegram:group:-1004301699476` | that bank (all groups) |

### Meta keys

| Key | Used for | Default |
|---|---|---|
| `_default` | Turns without a chat identity (CLI, cron, internal runs) | `quarantine` |
| `_unknown` | Chat identities that no key matches | `quarantine` |

Keys starting with `_` are meta keys and are not chat mappings. A map that
contains **only** meta keys leaves the provider exactly as it is: routing is
off, nothing is rebound, no bank directory is created.

---

## Fail-closed rules

The point of the feature is that private memories stay private. Every ambiguous
case resolves away from the private profile storage:

| Situation | Behaviour |
|---|---|
| Chat identity known, not mapped | `_unknown` bank (default `quarantine`) |
| Map file exists but cannot be read or parsed | `_unknown` bank (default `quarantine`) — **never** the private profile storage |
| Map file missing, or contains only meta keys | Routing off; the provider behaves as before the feature |
| Bank name with a path separator or `..` | Error — the bank directory is never left |
| Provider without its own Beam (skipped contexts) | Stays unavailable (`memory_unavailable`); routing does not create a bank for it |

The last row matters for subagents and background runs: a provider that the
runtime deliberately skipped does not get a bank behind the runtime's back.

---

## Storage

Each bank is its own database:

```
<data dir>/banks/<bank>/mnemosyne.db
```

The private profile database (`<hermes_home>/mnemosyne/data/mnemosyne.db`) is
never used as a fallback while routing is active. Bank names are a
configuration boundary: a name that would leave `<data dir>/banks` is rejected.

Each bank carries its own session scope. When the active session changes, the
cached bank instance is rebound to the new session; an explicitly configured
channel is preserved, an implicit one follows the session. Session-end
consolidation and the automatic sleep run against the bank that was written.

---

## Verifying it

Two places say which bank a turn used:

- The **audit log** (next to the active database) records a `bank_bound` event
  with the bank name, the reason and the database path.
- The provider log line:

  ```
  Mnemosyne bank bound: bank=alice reason=map:agent:main:telegram:dm:111 db=…
  ```

The reason strings are stable and worth knowing:

| Reason | Meaning |
|---|---|
| `map:<pattern>` | The turn matched a key. |
| `default:kein-session-key` | No chat identity in this turn. |
| `default:kein-chat-kontext` | The session key is not a chat identity. |
| `unknown:fail-closed` | A chat identity that no key matches. |
| `map:unreadable:fail-closed` | The map file could not be parsed. |
| `disabled:no-map` | Routing is off. |

---

## Turning it off

Delete the map file, or point `MNEMOSYNE_BANKS_MAP` at a path that does not
exist. The next turn is routed as before the feature. Bank databases stay on
disk — remove `<data dir>/banks/` by hand if you want them gone.

---

## Related

- [`docs/configuration.md`](configuration.md) — all environment variables
- [`integrations/hermes/README.md`](../integrations/hermes/README.md) — the
  Hermes adapter itself
- `docs/hermes-integration.md` — the integration as a whole
