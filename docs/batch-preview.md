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
coordination marks in its existing `-shm` file. This SQLite read coordination is
permitted; database and WAL contents, configuration and other files remain
unchanged. The preview does not initialize storage or create required sidecars.
MCP rejects persistent WAL databases whose required sidecars are missing rather
than creating them or using an immutable view that misses live WAL records.
