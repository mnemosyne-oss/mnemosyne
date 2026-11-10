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
| Map file exists but cannot be read or parsed | Always the literal `quarantine` bank, regardless of `_unknown` — **never** the private profile storage |
| Map file exists but cannot be inspected at all (permission error, I/O error) | Same as above: the literal `quarantine` bank. Only a genuinely missing map switches the routing off |
| A mapped chat identity whose bank name is empty or unusable | The literal `quarantine` bank. A broken entry is a broken configuration, not a reason to fall back to the private profile storage |
| Map file missing, or contains only meta keys | Routing off; the provider behaves as before the feature |
| Bank name with a path separator or `..` | Error — the bank directory is never left |
| Provider without its own Beam (skipped contexts) | Stays unavailable (`memory_unavailable`); routing does not create a bank for it |

The `_unknown` bank applies **only** to chat identities that no key matches.
A configured `_unknown` never receives a broken configuration: a map that
cannot be read, or an entry that names no usable bank, always lands in the
literal `quarantine` bank.

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
channel is preserved, an implicit one follows the session.

### Which bank a lifecycle operation touches

Session-end consolidation, the automatic sleep and the `mnemosyne_sleep` tool
run against the bank of the **session that triggered them** — not against the
bank of the chat that wrote last. After a session switch back, the route is
resolved from the current session key again, so the lifecycle work follows the
session it belongs to.

### Connections and their owners

Every bank is a separate SQLite database, and the core caches one connection
per (thread, database path). Two consequences matter in operation:

- A bank that was bound in a worker thread owns a handle that the thread
  running the shutdown cannot see in its own cache. The shutdown releases the
  handle itself, not the cache of the calling thread.
- Two providers that open the same bank in the same process share one
  connection object. A handle is therefore closed only when its **last** owner
  lets go of it — one provider's shutdown never breaks another's live
  instance.

A re-initialization (`initialize()`) reuses the cached bank, so the same
storage path is still served by exactly one connection.

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
| `map:<pattern>:invalid-bank:fail-closed` | The matched key names no usable bank. Fail-closed: the turn goes to `quarantine`. |
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

Deployment implications:

- Enabling routing does not move or partition existing profile memories.
  They stay in the profile database.
- Turning routing off restores access to those shared profile memories.
- Bank memories stay in their own databases and are not merged back.
- A missing map turns routing off. A map that exists but cannot be parsed
  routes to `quarantine`.

Removing the map is therefore a **fail-open** action — the profile storage
becomes reachable again — while a broken map is fail-closed.

---

## Related

- [`docs/configuration.md`](configuration.md) — all environment variables
- [`integrations/hermes/README.md`](../integrations/hermes/README.md) — the
  Hermes adapter itself
- `docs/hermes-integration.md` — the integration as a whole
