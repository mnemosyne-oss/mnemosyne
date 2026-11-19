# Batch preview

`mnemosyne_batch` with `dry_run: true` inspects the selected bank and session
without running mutations, embeddings, approvals, pending writes or audit hooks.
The response labels this mode `validation: target_eligibility_snapshot`.

For existing eligible targets it returns `would_update`, `would_delete` or
`would_invalidate`. Missing, invisible and action-ineligible targets all return
`not_found`, without disclosing another session's contents or ownership. A
`remember` operation returns `would_store`; this is not a content-policy or
approval guarantee. The target database must already exist and have a compatible
schema. An unavailable database or failed lookup returns `preview_unavailable`
without creating, initializing, migrating or repairing the store.

Eligibility follows the action adapter. For example, an episodic record can be
forgotten or invalidated but is not a working-memory update target. The MCP
wrapper additionally supports its existing owner-only legacy update path.
Cross-session recall settings do not broaden mutation permissions. Supersession
requires a distinct, visible and active replacement and an active target.

Every operation sees one current read snapshot. Two forget operations for the
same existing record may both return `would_delete`: the preview does not apply
the first operation to predict the second. Concurrent changes or write-policy
checks can still make a later apply fail. This is not optimistic concurrency
control or a reservation of the target.

The Python helper `dry_run_batch(normalized)` remains a compatible **syntax-only**
preview. It takes no store and cannot establish that any target exists. Call
`snapshot_preview_batch(normalized, target=BatchPreviewTarget(conn, session_id,
adapter="beam"))` for target eligibility on an already selected connection. The
`wrapper` adapter matches MCP's wrapper mutation semantics. Neither mode commits
or rolls back a caller-owned transaction.

Providers using an older Core without target-aware preview return
`preview_unavailable` for tool previews; they do not disguise syntax-only results
as a checked target. Existing apply support remains available.

Opening an active WAL store with SQLite's read-only mode may update reader
coordination marks in `-shm`. If the last existing connection closes between
preflight and the read-only open, SQLite can remove and recreate coordination
sidecars. This normal SQLite read coordination is permitted. The preview never
issues data mutations or initializes/migrates storage. MCP opens a read-only
SQLite connection; Hermes providers read through their existing connection.
Neither path writes database contents or committed WAL data. Concurrent writers
and checkpoints can change those files independently, so this is not a promise
of byte-for-byte preservation during concurrent database activity.

MCP conservatively rejects persistent WAL stores whose required sidecars are
already missing at preflight. That check does not lock their lifetime. It does
not use an immutable view that would ignore live WAL records.
