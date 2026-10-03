# Migrating to Mnemosyne 4.0

4.0 carries one breaking change. Most installations are unaffected and can
upgrade without doing anything. Read the first section to find out which group
you are in before changing any configuration.

## Am I affected?

You are affected only if **both** of these are true:

1. You point `MNEMOSYNE_EMBEDDING_API_URL` at a custom embedding endpoint, and
2. the model you use is not in Mnemosyne's built-in model table, and you have
   not set `MNEMOSYNE_EMBEDDING_DIM`.

You are **not** affected if you use the default embedding model, use any model
in the built-in table (`BAAI/bge-small-en-v1.5`, `BAAI/bge-m3` and the other
listed models), already set `MNEMOSYNE_EMBEDDING_DIM`, or run with embeddings
disabled (`MNEMOSYNE_NO_EMBEDDINGS=1`).

## What changed

Before 4.0, an unrecognised embedding model silently resolved to 384
dimensions, which is `bge-small`'s dimension. A sqlite-vec table is
dimensioned at creation time, so that guess was written permanently into any
fresh database. Anyone running a different model, for example
`mxbai-embed-large` at 1024 dimensions through a custom endpoint, ended up
with a store whose vector index could never match its embeddings. Vector
search returned wrong results with no error anywhere.

From 4.0 the dimension is resolved in this order:

1. An explicit `MNEMOSYNE_EMBEDDING_DIM`, which must be a positive integer.
2. The built-in model table.
3. Otherwise `ValueError`, raised at import.

An unknown model with no explicit dimension now fails at startup instead of
corrupting a store quietly.

## What you will see

Direct core and MCP-provider startup exits at import with an error naming both
the model and the variable to set. The `mnemosyne-hermes` wrapper catches the
error and reports the provider as unavailable rather than exiting the host.

## The fix

Set the dimension your model actually produces:

```bash
export MNEMOSYNE_EMBEDDING_DIM=1024   # use your model's real dimension
```

Blank or empty `MNEMOSYNE_EMBEDDING_DIM` and `MNEMOSYNE_EMBEDDING_MODEL`
values, which are common in Docker Compose files and `.env` files, are treated
as unset rather than as invalid values, so you do not need to remove empty
declarations.

## If your store was created under the old fallback

This is the case that needs care. If a database was created while the silent
384 fallback was in effect, its vector tables are already dimensioned at 384
even though your model emits something else. Setting the correct dimension will
now trip the existing dimension-mismatch guard on startup.

That guard is not a corruption report. Your memories are intact and recall
falls back to keyword search until the index is rebuilt. You have two options:

* **Keep the existing vectors.** Relaunch with `MNEMOSYNE_EMBEDDING_DIM` set to
  the dimension already in the database, along with the model that matches it.
* **Re-embed at the correct dimension.** Run
  `MNEMOSYNE_EMBEDDING_DIM=<N> mnemosyne reindex`, which backs the store up
  before rebuilding.

Run `mnemosyne doctor` afterwards to confirm. It reports the resolved
`embeddings_dim` alongside `embeddings_model`, so you can verify the
resolution without reading a traceback.

Do not treat the override alone as a one-step fix for an existing store. It
changes what the process expects, not what the database contains.

## Standalone Hermes package

`mnemosyne-hermes` versions separately on its own `0.x` line and is not part of
this major bump. It does carry one behaviour change worth knowing about if you
upgrade both at once: a symlink install now fails closed when no validated
interpreter is found, naming `--python`, where it previously proceeded.
`--no-bootstrap` continues without dependency validation, since it installs
nothing into Hermes' environment.

## Related issues

* #518, #521: fail loud on unknown embedding model
* #666: `bge-m3` alias resolves its 1024-dimensional vectors
