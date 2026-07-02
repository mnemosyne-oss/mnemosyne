# Benchmarking and Testing Infrastructure

**v3.0.0 (MEMORIA).** See [beam-benchmark.md](beam-benchmark.md) for published results and the quick-start guide for running BEAM on your own hardware.

This document is the maintainer reference for the levers that affect benchmark results: env vars, recall modes, diagnostic instrumentation, and the methodology for running rigorous A/B tests. It exists because Mnemosyne has multiple recall paths (linear + polyphonic, plus MEMORIA structured retrieval), per-tool toggles, and harness modes that aren't relevant to normal usage but matter a great deal when measuring per-component contribution to scores.

**Audience:** maintainers and contributors running benchmarks against the Mnemosyne recall stack. For end-users who want to run BEAM on their hardware, start with [beam-benchmark.md](beam-benchmark.md).

---

## Why a separate doc

Past benchmark results (see [beam-benchmark.md](beam-benchmark.md)) were collected before several silent-failure surfaces were closed (May 2026, PRs #80-#91). Those results are not credible evidence for any specific tool's contribution to total score, because the prior pipeline had:

- Harness-side oracles that answered TR/CR/IE/KU questions without going through `BeamMemory.recall()` (PR #90).
- Last 12 raw conversation messages always prepended to every answer prompt -- recency-anchored answers succeeded regardless of recall quality (PR #90).
- Cross-tier `(summary, source)` duplicates ranked side-by-side under the linear scorer but collapsed under polyphonic's diversity rerank, confounding arm-vs-arm comparison (PR #88).
- Veracity destroyed at consolidation -- every post-`sleep()` episodic row scored at the 0.8 `unknown` multiplier regardless of source-row veracity (PR #89).
- `remember_batch` silently swallowing partial embedding failures, biasing the vector voice toward early-ingested rows (PR #89).
- Benchmark adapter writing template summaries and destroying source rows; the corpus recall actually saw was ~500 rows of "Batch N: …" stubs (PR #75 → E1).

This doc encodes what those fixes opened up: the ability to run **per-tool A/B tests** with a single variable changed per run. Use this document as the reference when designing or executing a benchmark; the [BEAM-recovery experiment plan](experiments/2026-05-12-beam-recovery-arms-abc.md) is a concrete application.

---

## Setup

These prerequisites are benchmark-only. They are **not** required to run Mnemosyne under normal use.

### Python dependencies

```bash
# Already in pyproject as optional groups:
pip install 'mnemosyne-memory[embeddings]'    # fastembed -- vector voice + dense recall
pip install 'mnemosyne-memory[llm]'           # llama-cpp-python -- sleep summarization (else AAAK fallback)

# Benchmark-only -- NOT in pyproject:
pip install datasets                           # HuggingFace BEAM dataset loader
pip install sqlite-vec                         # ANN backend for vec_episodes virtual table
pip install numpy                              # benchmark harness requires it unconditionally
```

The benchmark harness (`_benchmarks/evaluate_beam_end_to_end.py`) imports `numpy` and `datasets` unconditionally. Neither is declared as an installable extra of the package. Track these in your local venv setup or via `requirements-benchmark.txt`.

### Environment variables required for any benchmark run

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | LLM that answers benchmark questions (or read from `/tmp/openrouter_key.txt`) |
| `MNEMOSYNE_BENCHMARK_PURE_RECALL=1` | yes (recommended) | Disables harness-side oracles; forces every answer through Mnemosyne recall. See [Pure-recall mode](#pure-recall-mode). |
| `HF_TOKEN` | only if BEAM gets gated | Currently public at `Mohammadta/BEAM`; HF token only needed if access changes |
| `OPENROUTER_BASE_URL` | optional | Defaults to `https://openrouter.ai/api/v1` |

### Resource budget (per phase)

| Resource | 100K scale (3 conversations) | 250K scale (3 conversations) |
|---|---|---|
| Wall clock | ~15-25 min | ~60-90 min |
| Peak RSS | ~2-4 GB | ~4-8 GB |
| Disk for DB | ~500 MB | ~2-4 GB |
| LLM API spend | ~$0.50-$2 | ~$5-$15 |

v3.0.0 ingestion with full MEMORIA extraction: 188 messages in ~36 seconds (host extraction, no external API calls). Embedding adds per-message overhead; `fastembed` recommended for production runs.

API spend is dominated by per-question answer LLM calls. Caching identical queries can lower this; quantify on the first phase before committing to a long run.

---

## Environment variable reference

Every env var that affects recall ranking or benchmark behavior. Pin them across phases of an A/B run or the comparisons are meaningless. The harness should snapshot all of these into the results JSON (see [Recording](#recording-per-run)).

### Test mode / harness

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_BENCHMARK_PURE_RECALL` | unset (`0`) | When truthy (`1`/`true`/`yes`/`on`), disables four harness bypass paths: TR timeline oracle, CR contradiction injection, IE/KU `_context_facts` side-index, "RECENT CONVERSATION" raw-message prompt section. Forces every answer through `BeamMemory.recall()`. **Required for credible arm-vs-arm comparison.** Also exposed as `--pure-recall` CLI flag. |
| `FULL_CONTEXT_MODE` | unset (`0`) | Sends the entire conversation to the answer LLM, bypassing retrieval. Useful for measuring the "LLM ceiling without recall" upper bound. Overridden by `MNEMOSYNE_BENCHMARK_PURE_RECALL` if both are set. Also exposed as `--full-context`. |

Both parsers accept `1`/`true`/`yes`/`on` (case-insensitive, whitespace-stripped). Anything else is falsy.

### Recall path selection

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_POLYPHONIC_RECALL` | unset (`0`) | When truthy, routes `BeamMemory.recall()` through `PolyphonicRecallEngine` (RRF fusion across vector / graph / fact / temporal voices + diversity rerank). When unset, uses the linear scorer with inline `graph_bonus` / `fact_bonus` / `binary_bonus` terms. Read at recall time, not init time -- can be toggled per-call by changing the env. |

**Linear vs polyphonic implement related signals via DIFFERENT mechanisms.** The linear path's `graph_bonus` is an edge-count LIKE-match on `graph_edges`. The polyphonic engine's `_graph_voice` does entity extraction + `find_facts_by_subject`. They have different failure modes; do not assume an ablation on one engine carries over to the other.

### Linear-path scoring weights

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_VEC_WEIGHT` | `0.5` | Weight of vector similarity in the linear hybrid score. Normalized to sum to 1.0 with FTS + importance. |
| `MNEMOSYNE_FTS_WEIGHT` | `0.4` | Weight of FTS5 score in the linear hybrid score. |
| `MNEMOSYNE_IMPORTANCE_WEIGHT` | `0.1` | Weight of stored `importance` in the linear hybrid score. |
| `MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS` | (caller arg) | Temporal-boost half-life for the linear path's `_temporal_boost`. Only active when caller passes `temporal_weight > 0`. |

These three (`VEC` / `FTS` / `IMPORTANCE`) interact -- changing one alters the relative weight of the others. Treat as a triple to pin together.

### Veracity multipliers

Applied in both linear (post-FTS+vec scoring) and polyphonic (post-RRF) paths. Affect ranking; do not affect ingest.

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_STATED_WEIGHT` | `1.0` | Multiplier for rows tagged `veracity='stated'` |
| `MNEMOSYNE_INFERRED_WEIGHT` | `0.7` | Multiplier for `veracity='inferred'` |
| `MNEMOSYNE_TOOL_WEIGHT` | `0.5` | Multiplier for `veracity='tool'` |
| `MNEMOSYNE_IMPORTED_WEIGHT` | `0.6` | Multiplier for `veracity='imported'` |
| `MNEMOSYNE_UNKNOWN_WEIGHT` | `0.8` | Multiplier for `veracity='unknown'` (the schema default) |

**Drift caveat:** these env vars override the recall multiplier in `beam.py`, but the consolidator's Bayesian compounding in `veracity_consolidation.py` does NOT honor env overrides -- it reads `VERACITY_WEIGHTS` directly. Setting `MNEMOSYNE_STATED_WEIGHT=0.9` breaks the invariant "consolidated-as-N also ranks at N." `beam.py` module-load emits a single WARNING when any of these env vars are set, surfacing the drift risk.

If you want to disable the multiplier entirely for an A/B baseline, set `MNEMOSYNE_VERACITY_MULTIPLIER=0` -- see [A/B ablation toggles](#ab-ablation-toggles).

### Episodic tier degradation

Applied to episodic results based on their `tier` column (1, 2, 3 -- set by `degrade_episodic`).

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_TIER1_WEIGHT` | `1.0` | Hot tier multiplier (rows < `TIER2_DAYS` old) |
| `MNEMOSYNE_TIER2_WEIGHT` | `0.5` | Mid tier multiplier |
| `MNEMOSYNE_TIER3_WEIGHT` | `0.25` | Cold tier multiplier (rows > `TIER3_DAYS` old) |
| `MNEMOSYNE_TIER2_DAYS` | `30` | Threshold for tier 1→2 transition |
| `MNEMOSYNE_TIER3_DAYS` | `180` | Threshold for tier 2→3 transition |

For benchmark runs on synthetic short-time-span data, tier degradation typically doesn't fire and all weights collapse to `TIER1_WEIGHT`. Pin them to `1.0` if you want zero tier effect.

### Vector backend

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_VEC_TYPE` | `int8` | Storage format for vector embeddings: `float32` (full precision, 1536 bytes/vec), `int8` (384 bytes/vec, default), `bit` (48 bytes/vec, binary-quantized). Changes candidate set for `_vec_search` and ranking quality. |

`bit` mode trades recall quality for storage; expect lower scores on semantic-heavy questions.

### Scan breadth / FTS semantics

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_BEAM_OPTIMIZATIONS` | unset (`0`) | When truthy, switches FTS5 to OR-semantics, raises vector-scan limit, and always includes vector results. Designed for benchmark-scale recall over large corpora. Distinct from `MNEMOSYNE_BENCHMARK_PURE_RECALL` -- the latter is harness-side, this is recall-side. |

If you don't enable this for benchmarks ≥100K, expect FTS-driven recall to miss substring partial matches that the benchmark questions expect.

### Working memory / sleep

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_WM_TTL_HOURS` | `24` | Working memory rows older than this get pulled into `sleep()`. The benchmark harness backdates timestamps to ensure rows are eligible. |
| `MNEMOSYNE_SLEEP_BATCH` | `5000` | Max rows pulled per `sleep()` invocation. Larger batches reduce sleep overhead; smaller batches reduce peak memory during summarization. |
| `MNEMOSYNE_SLEEP_PROMPT` | *(built-in)* | Optional consolidation prompt override. Pin this across multilingual benchmark arms because it changes episodic summary language/content. |
| `MNEMOSYNE_LLM_ENABLED` | `true` | When `false`, `sleep()` skips local LLM summarization and falls back to AAAK encoding. Useful for benchmark runs that want deterministic summaries without per-row LLM latency. |

### MEMORIA Fact Engine (v3.0.0+)

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_STRICT_FACT_MATCH` | unset (`0`) | When truthy, fact retrieval uses conservative token-based matching (`wysie`, #143). Filters stopwords, requires multi-token overlap or distinctive structural markers. Reduces false positives at the cost of lower recall on partial matches. |
| `MNEMOSYNE_PROACTIVE_LINKING` | `0` | When set to `1`, creates zero-LLM graph edges at ingestion via content similarity (FTS5) and entity overlap strategies. Adds ~5% ingestion overhead but enables graph-traversal recall without external LLM calls. |
| `MNEMOSYNE_AUTO_MIGRATE` | `1` | When `0`, disables automatic creation of MEMORIA tables (`memoria_facts`, `memoria_timelines`, `memoria_kg`, `memoria_instructions`, `memoria_preferences`). For operators who want explicit control over schema migration. |

---

## Pure-recall mode

`MNEMOSYNE_BENCHMARK_PURE_RECALL=1` (or `--pure-recall`) is the single most important flag for any A/B benchmark. It disables four hardcoded paths in the harness that produce answers *without* going through `BeamMemory.recall()`:

1. **TR (Temporal Reasoning) oracle** -- pre-fix, TR questions extracted a timeline from raw `conversation_messages` and answered via an LLM-with-dates prompt, returning before any recall. Pure-recall lets TR questions flow through normal recall.
2. **CR (Contradiction Resolution) injection** -- pre-fix, raw-message contradiction detection injected a "you mentioned contradictory things" hint into the answer prompt. Pure-recall disables this hint; the arm must surface the contradiction via recall.
3. **IE/KU `_context_facts` side-index** -- pre-fix, the harness built a regex-keyed phrase-to-value index at ingest from raw messages, then matched questions against it and returned the value directly. Pure-recall disables the lookup.
4. **"RECENT CONVERSATION" injection** -- pre-fix, the last 12 raw conversation messages were always prepended to every answer prompt. Recency-anchored answers succeeded regardless of recall quality. Pure-recall strips the section; only retrieved memories reach the LLM.

Without these gates, any "Arm B beats Arm A by 5pp" claim is suspect -- the harness might have produced identical answers across arms for the questions hitting the bypass paths.

The flag was added in May 2026. Default behavior (env unset) preserves the legacy benchmark mode for backward compatibility, but new runs targeting per-tool A/B claims should always set it.

---

## Diagnostic instrumentation

Two diagnostic modules emit per-run snapshots that should be captured into results JSON.

### Recall diagnostics

```python
from mnemosyne.core.recall_diagnostics import (
    get_recall_diagnostics,      # returns Dict snapshot
    reset_recall_diagnostics,    # clears process-global counters
)
```

`get_recall_diagnostics()` returns a JSON-serializable dict with:

| Field | Meaning |
|---|---|
| `wm_fts_kept` | Working-memory rows kept via FTS5 |
| `wm_vec_kept` | Working-memory rows kept via vector search |
| `wm_fallback_kept` | Working-memory rows kept via substring fallback |
| `em_fts_kept` | Episodic rows kept via FTS5 |
| `em_vec_kept` | Episodic rows kept via vector search |
| `em_fallback_kept` | Episodic rows kept via substring fallback |
| `total_kept` | Sum across all tiers |
| `truly_empty_calls` | Calls where every tier produced zero kept rows |
| `fallback_rate` | `(wm_fallback + em_fallback) / total_kept` |

High `fallback_rate` (>30%) on a benchmark run is a red flag -- it means most recall is coming from the weak-signal substring scan, not FTS or vector. Investigate before trusting the score.

Call `reset_recall_diagnostics()` at the start of each phase to keep counters clean per-run.

### Extraction diagnostics

```python
from mnemosyne.extraction.diagnostics import get_extraction_stats
```

Returns per-tier (host / remote / local / cloud / wrapper) extraction call counts plus bounded error samples (10 samples per tier, 200 chars per message). Surfaces silent failures in the fact-extraction pipeline.

---

## Test sequence template

The general shape of a credible A/B benchmark:

1. **Preflight** -- assert pure-recall is active, snapshot every `MNEMOSYNE_*` env var, log recall path per call.
2. **Phase 0 (baseline floor)** -- minimum configuration: linear scorer, all veracity weights = 1.0, no enrichment.
3. **+ one variable** per subsequent phase. Run on a 100K-message slice (~20 min); save full 250K runs for confirming the top two configurations from the small-scale screen.
4. **Record** per-question paired outcomes plus the diagnostic snapshots. Compute bootstrap CIs on per-ability score deltas.
5. **Falsification criterion:** for "tool X contributes near-zero" claims, the 95% CI must exclude ±2pp before treating the prediction as confirmed.

See the [BEAM-recovery experiment plan](experiments/2026-05-12-beam-recovery-arms-abc.md) for a concrete instantiation with 10 phases and 8 theses.

### Preflight checklist

Every benchmark run should:

- [ ] Confirm pure-recall mode is active (`MNEMOSYNE_BENCHMARK_PURE_RECALL=1` or `--pure-recall`).
- [ ] Set `OPENROUTER_API_KEY` (or its fallback file).
- [ ] Pin all ranking weights (`MNEMOSYNE_VEC_WEIGHT`, `FTS_WEIGHT`, `IMPORTANCE_WEIGHT`, `TEMPORAL_HALFLIFE_HOURS`, `VEC_TYPE`, `BEAM_OPTIMIZATIONS`) identically across phases.
- [ ] Reset diagnostics before each phase.
- [ ] Verify the recall path (linear vs polyphonic) matches what the phase intends.

A small Python helper or shell wrapper that asserts all of these and refuses to run if any are missing is cheap insurance against accidentally-invalid runs.

---

## Recording per run

For the **output-file schemas, analysis recipes (paired bootstrap CIs, per-voice attribution, fallback-rate sanity checks), and an AI-assistant-friendly summary**, see [benchmark-results-analysis.md](benchmark-results-analysis.md). Quick reference:

Capture into `results/beam_e2e_results.json` (or equivalent):

- **Per-ability score** -- TR / CR / IE / KU / MR / ABS / EO / SUM, plus total.
- **Run config** -- phase name, sample size, scale, all `MNEMOSYNE_*` + `FULL_CONTEXT_MODE` env vars at start.
- **Recall diagnostics** -- full `get_recall_diagnostics()` snapshot at run end (or aggregated per-call).
- **Extraction diagnostics** -- full `get_extraction_stats()` snapshot.
- **Latency** -- p50 / p95 / p99 per-question recall + answer roundtrip.
- **Storage** -- final row counts in `working_memory`, `episodic_memory`, `memory_embeddings`, `vec_episodes`, `annotations`, `consolidated_facts`.
- **Peak RSS** during ingest and recall phases separately.

For statistical reporting, the harness emits a flat `paired_outcomes.jsonl` alongside `beam_e2e_results.json`. Each row carries `{config_id, run_started_at, scale, conversation_id, qid, ability, score, correct}`. `config_id` is either supplied via `--config-id` (recommended for human-readable phase labels like `phase3a-no-fact-voice`) or auto-derived as a SHA-256 hash of the `MNEMOSYNE_*` env snapshot. `correct` is `score >= 0.5` (treating partial-credit as correct; analyst can re-threshold off raw `score`).

Append-only: multiple A/B-phase runs accumulate in one file. Filter by `config_id` to isolate a single phase. The diagnostic snapshots are also emitted into `beam_e2e_results.json` under `metadata.diagnostics`.

**Voice-score parity:** post-recall result dicts from both engines now carry a `voice_scores: dict` field for uniform analysis. Keys differ between engines (linear uses `{vec, fts, keyword, importance, recency_decay}`; polyphonic uses `{vector, graph, fact, temporal}`), so engine identity is the dict's keyset, not the field's presence.

---

## A/B ablation toggles

Nine env-var toggles allow disabling specific components for experimental ablation. **Each defaults to ON** (production behavior preserved); set the env var to `0`/`false`/`no`/`off` (case-insensitive, whitespace-stripped) to disable. Truthy/garbage/unset values keep the feature enabled -- these are opt-out flags, not opt-in.

The toggle helper is `mnemosyne.core.beam._env_disabled(name)` (and a mirror in `polyphonic_recall.py`). For the experiment plan's phase-by-phase usage of each toggle, see [`experiments/2026-05-12-beam-recovery-arms-abc.md`](experiments/2026-05-12-beam-recovery-arms-abc.md).

| Env var | Disables when set falsy | Affects |
|---|---|---|
| `MNEMOSYNE_VOICE_VECTOR` | Polyphonic vector voice (Phase 3d ablation) | `polyphonic_recall._vector_voice` early-return |
| `MNEMOSYNE_VOICE_GRAPH` | Polyphonic graph voice (Phase 3b) | `polyphonic_recall._graph_voice` early-return |
| `MNEMOSYNE_VOICE_FACT` | Polyphonic fact voice (Phase 3a) | `polyphonic_recall._fact_voice` early-return |
| `MNEMOSYNE_VOICE_TEMPORAL` | Polyphonic temporal voice (Phase 3c) | `polyphonic_recall._temporal_voice` early-return |
| `MNEMOSYNE_GRAPH_BONUS` | Linear-path graph-edge bonus | `beam.py` ep main loop + fallback |
| `MNEMOSYNE_FACT_BONUS` | Linear-path fact-table bonus | `beam.py` ep main loop + fallback |
| `MNEMOSYNE_BINARY_BONUS` | Linear-path binary-vector Hamming bonus | `beam.py` ep main loop |
| `MNEMOSYNE_VERACITY_MULTIPLIER` | Veracity multiplier (Phase 0/1) | `beam.py` linear + polyphonic |
| `MNEMOSYNE_CROSS_TIER_DEDUP` | `_dedup_cross_tier_summary_links` (Phase 4) | `beam.py` linear + polyphonic |

Example:

```bash
# Phase 3a: disable the polyphonic fact voice to measure its contribution
MNEMOSYNE_VOICE_FACT=0 \
MNEMOSYNE_POLYPHONIC_RECALL=1 \
MNEMOSYNE_BENCHMARK_PURE_RECALL=1 \
python _benchmarks/evaluate_beam_end_to_end.py --scales 100K --sample 3
```

Pin each toggle in your `.env` or shell environment across an A/B run. The harness preflight snapshots all `MNEMOSYNE_*` env vars into the results JSON under `metadata.config.env` so you can verify the configuration retroactively.

---

## How this doc evolves

When a new env var is added that affects recall ranking or benchmark behavior, update the [Environment variable reference](#environment-variable-reference) table here. When a new diagnostic counter ships, add it to [Diagnostic instrumentation](#diagnostic-instrumentation). When a new experiment runs, add a dated artifact under `docs/experiments/`.

Past experiment artifacts:
- [v3.0.0 published results — 65.2% at 100K with MEMORIA](beam-benchmark.md)
- [2026-05-12 — BEAM-recovery Arms A/B/C](experiments/2026-05-12-beam-recovery-arms-abc.md)
- (older runs documented in previous beam-benchmark versions — note that those pre-date the May 2026 fix bundle and aren't credible for per-tool claims)
