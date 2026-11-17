# Runtime diagnostic scopes

Runtime checks describe the interpreter that performed them. They do not infer
another interpreter from `HERMES_HOME`, and a role label is not authentication.

| Caller | Local runtime scope |
| --- | --- |
| Neutral Python API / Doctor adapter | `calling_interpreter` |
| Core CLI and provider CLI wrappers | `cli_interpreter` |
| In-process Hermes provider, either implementation | `provider_runtime` |
| MCP diagnostic handler | `mcp_server` |

An explicit `mnemosyne diagnose --python PATH` still probes that interpreter in
an isolated subprocess. Its checks are always `target_interpreter`, even when the
caller has another role. An unavailable or incomplete target stays unavailable;
there is no fallback to the caller's dependency results. `--fix` still uses and
verifies the selected interpreter. Database checks remain local to the caller.

The pure collector keeps its optional positional interpreter argument and adds
keyword-only `scope`. Local role values are validated; `target_interpreter` is
reserved for explicit target probes. Each runtime check and raw payload carries
its measured scope and the executable is identified only by its basename.
Doctor preserves known role labels in JSON and Markdown. Missing or unknown
labels from older producers become `unscoped` at its report boundary.

Providers detect whether the Core diagnostic function explicitly accepts the
role keyword. With an older Core they keep the existing call signature but label
the returned runtime information `unscoped`. They do not retry an internal
`TypeError` as though it were a signature mismatch.
Logs written by the older Core are not rewritten; the response identifies
their role provenance separately as `runtime_log_scope: legacy_unverified`.
