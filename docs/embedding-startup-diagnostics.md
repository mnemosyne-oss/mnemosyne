# Embedding startup diagnostics

At INFO level, Hermes provider startup records one embedding configuration
message per process. Both provider copies share the Core announcement guard.
It reports the loaded model, dimension and credential-presence boolean, the
current routing backend and endpoint origin, and the config-file locator and
presence observed when Core embeddings was imported.

The locator names the selected path convention, for example
`MNEMOSYNE_DATA_DIR/config.yaml`; it does not expose an absolute home path.
It does not claim that every setting came from that file or distinguish seeded
values from intentional overrides. Existing configuration precedence and its
model/dimension exceptions are unchanged.

Model, dimension and credentials are loaded at import time. Editing their
environment variables later does not reload them. Backend routing and endpoint
resolution remain call-time decisions, so these can disagree with a late
model/environment edit. Restart the actual provider process after deliberately
correcting its configuration.

The endpoint is limited to scheme, hostname and optional port. Userinfo, path,
query, fragment and credential values are omitted. No memory text is logged.
The message does not load a model, infer a vector, contact an endpoint, open
SQLite, or create configuration. This no-network guarantee applies to the
diagnostic message. SQLite and local lexical retrieval do not need network
access; normal API-backed embedding operations can contact their configured
endpoint. It describes configuration, not readiness or
reachability. Existing unavailable-engine warnings and transport refusals
remain separate and retain their warning budgets.

Older supported Core versions without this optional diagnostic helper continue
starting normally. A diagnostic failure does not prevent initialization or
suppress the existing unavailable-engine warning.
