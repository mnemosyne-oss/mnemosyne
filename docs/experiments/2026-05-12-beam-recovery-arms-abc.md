# BEAM-Recovery Experiment -- Test Sequence and Theses

**Date:** 2026-05-12 | **Status:** Plan, pre-execution. References [docs/benchmarking.md](../benchmarking.md) for env-var definitions and methodology.

This is a dated experiment artifact. For the reusable test-infrastructure reference (env vars, diagnostic instrumentation, pure-recall mode, recording template), see [benchmarking.md](../benchmarking.md). This file describes the specific 10-phase ablation sequence planned for the post-May-2026 fix bundle and the theses each phase tests.

---

## Why this exists

The goal is to measure, per tool, what each component of the Mnemosyne recall stack contributes to BEAM scores -- not to demonstrate Mnemosyne wins. PRs #80-#91 were filed specifically to enable that per-tool isolation, and **stats collected before those land are not credible evidence for or against any specific tool**, because the prior pipeline had silent failure modes and harness-side oracles that made it impossible to attribute deltas to a single change.

The proposed sequence below is designed to commit each tool to a measurable A/B test rather than a "polyphonic vs linear" bundle comparison.

---

## Hard prerequisites before any phase runs

### Code prerequisites -- PRs to merge first

**This plan is only valid against a checkout that has these PRs merged:** #80 (polyphonic vector voice rewire to `memory_embeddings` + sqlite-vec), #82 (`remember_batch` enrichment parity), #88 (cross-tier dedup), #89 (veracity preservation through consolidation), #90 (pure-recall harness gate), #91 (env-parser + telemetry). On current `main` (pre-merge) several plan claims are factually wrong -- `_dedup_cross_tier_summary_links` doesn't exist, `_recall_polyphonic` still uses `BinaryVectorStore`, the harness has no `--pure-recall` flag. If those PRs haven't merged when you run this, stop and merge them first.

### Python / system dependencies

```bash
# Core (already in pyproject as optional groups):
pip install 'mnemosyne-memory[embeddings]'   # fastembed (≥0.3.0) -- recall vector voice
pip install 'mnemosyne-memory[llm]'          # llama-cpp-python -- sleep summarization (else AAAK fallback)

# Benchmark-only -- NOT in pyproject yet:
pip install datasets                          # HuggingFace BEAM dataset loader (_benchmarks/evaluate_beam_end_to_end.py:183)
pip install sqlite-vec                        # vec_episodes ANN backend (linear path + polyphonic post-#80)
pip install numpy                             # _benchmarks/evaluate_beam_end_to_end.py:54 import is unconditional
```

`numpy` and `datasets` are runtime requirements of the benchmark harness but not declared as installable extras of the package -- install them in the same venv. Recommend tracking these in a `requirements-benchmark.txt` (see Implementation Gaps §6 below).

### External services / API keys

| Variable | Required for | Source |
|---|---|---|
| `OPENROUTER_API_KEY` | LLM that answers benchmark questions | OpenRouter account |
| `HF_TOKEN` (optional) | If BEAM dataset turns gated; currently public at `Mohammadta/BEAM` | HuggingFace |
| `OPENROUTER_BASE_URL` (optional) | API base URL override; default `https://openrouter.ai/api/v1` | env-only |

The harness falls back to reading `OPENROUTER_API_KEY` from `/tmp/opencode_key.txt` or `/tmp/openrouter_key.txt` if the env isn't set (`_benchmarks/evaluate_beam_end_to_end.py:60-71`).

### Compute / resource budget

| Resource | 100K phase | 250K phase |
|---|---|---|
| Wall clock | ~20-30 min per run | ~60-90 min per run |
| Peak RSS | ~2-4 GB (fastembed + LLM) | ~4-8 GB |
| Disk for DB | ~500 MB per conversation | ~2-4 GB per conversation |
| LLM API spend | ~$0.50-$2 per phase | ~$5-$15 per phase |

API spend is dominated by per-question answer LLM calls (BEAM has up to 50 questions per conversation per scale). Cache hits via deduplication of identical queries can lower this; quantify on first phase before committing to all 10.

**Preflight (run once per session, then per phase):**

1. **Assert pure-recall mode is active.** The harness should refuse to run if `MNEMOSYNE_BENCHMARK_PURE_RECALL` is unset. The bypasses don't produce identical answers across arms; running without the gate silently invalidates every comparison.
2. **Dump every `MNEMOSYNE_*` env var into results JSON.** A toggle the operator forgot is a confound in disguise.
3. **Pin all ranking knobs.** `MNEMOSYNE_VEC_WEIGHT`, `MNEMOSYNE_FTS_WEIGHT`, `MNEMOSYNE_IMPORTANCE_WEIGHT`, `MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS`, `MNEMOSYNE_VEC_TYPE`, `MNEMOSYNE_BEAM_OPTIMIZATIONS`, `MNEMOSYNE_TIER{1,2,3}_WEIGHT`, `MNEMOSYNE_*_WEIGHT` veracity values must be set identically across phases -- pin them in a `.env` file or assert in the runner.
4. **Sanity-check the active recall path.** Log whether each call went through linear or polyphonic; the env var is read at call time, not init time, so accidental flips mid-run are possible.
5. **Record per-result `voice_scores` and tier provenance** (from C4 + C13.b diagnostics). Without these, a "tool contributes nothing" finding is unfalsifiable.

---

## Why prior stats are suspect

- The benchmark harness answered TR/CR/IE/KU questions from a regex-extracted timeline, raw-message contradiction detection, and a `_context_facts` side-index -- bypassing `BeamMemory.recall()` entirely on those ability dimensions (PR #90).
- Every answer prompt was always prepended with the last 12 raw conversation messages, so any recency-anchored answer could succeed regardless of recall quality and arm choice (PR #90).
- Post-E3 additive sleep left both the source `working_memory` row and the episodic summary discoverable by recall, double-incrementing `recall_count` and ranking duplicates side-by-side under the linear scorer while polyphonic's diversity rerank silently collapsed them -- asymmetric dedup confounded any arm-vs-arm comparison (PR #88).
- `consolidate_to_episodic` never populated the `veracity` column on the summary row, so post-E4 per-row veracity got destroyed the moment sleep ran -- every episodic row scored at the 0.8 `unknown` multiplier (PR #89).
- `remember_batch` swallowed partial embedding failures via a bare `except Exception: pass`, silently losing entire batches of vectors at scale (PR #89).
- Pre-E1 the benchmark adapter wrote template summaries and destroyed source rows; the corpus most prior runs actually recalled against was ~500 episodic rows of "Batch N: first_3_msg_contents[:100chars]" stubs, not the 250K-message dataset (PR #75).

Each is fixed; their combined effect on prior numbers is unknowable.

---

## What's now controllable for A/B

**Key disambiguation up front:** the linear path and the polyphonic engine implement related signals through DIFFERENT mechanisms. The linear scorer applies `graph_bonus` / `fact_bonus` / `binary_bonus` as inline additions (`beam.py:~2497-2545`); the polyphonic engine runs separate voices (`polyphonic_recall.py:_graph_voice`, `_fact_voice`, `_vector_voice` which is binary-vector-driven). They have different failure modes and different ablation surfaces. Toggles must specify which engine they gate.

| Tool / axis | Engine | On/off mechanism | Landed in |
|---|---|---|---|
| Polyphonic engine vs linear scorer | both | `MNEMOSYNE_POLYPHONIC_RECALL=1` selects polyphonic | E5 (#76) |
| Polyphonic vector voice (post-#80: `memory_embeddings` + sqlite-vec ANN; pre-#80: `BinaryVectorStore`) | polyphonic | Implicit when polyphonic on; ANN gated by `_vec_available`. **Proposed:** `MNEMOSYNE_VOICE_VECTOR=0` for ablation | E5.a (#80) |
| Polyphonic graph voice (`find_gists_by_participant` / `find_facts_by_subject`) | polyphonic | **Proposed:** `MNEMOSYNE_VOICE_GRAPH=0` |  |
| Linear graph bonus (edge-count LIKE-match on `graph_edges`) | linear | **Proposed:** `MNEMOSYNE_GRAPH_BONUS=0`; **separate toggle from polyphonic graph voice** |  |
| Polyphonic fact voice (synthetic `cf_*` IDs from `consolidated_facts`) | polyphonic | **Proposed:** `MNEMOSYNE_VOICE_FACT=0` |  |
| Linear fact bonus (per-row `facts` table query) | linear | **Proposed:** `MNEMOSYNE_FACT_BONUS=0` |  |
| Polyphonic temporal voice | polyphonic | **Proposed:** `MNEMOSYNE_VOICE_TEMPORAL=0` |  |
| Linear temporal boost (`_temporal_boost` × `temporal_weight`) | linear | `temporal_weight=0` per-call arg (already exposed); needs harness flag |  |
| Linear binary-vector bonus (capped at 0.08) | linear | **Proposed:** `MNEMOSYNE_BINARY_BONUS=0` |  |
| Veracity multiplier | both | **Proposed:** `MNEMOSYNE_VERACITY_MULTIPLIER=0` -- gates the multiplier in linear + polyphonic |  |
| Tier-degradation multiplier | both | `MNEMOSYNE_TIER{1,2,3}_WEIGHT=1.0` neutralizes (existing env vars) |  |
| Cross-tier (summary, source) dedup | both | **Proposed:** `MNEMOSYNE_CROSS_TIER_DEDUP=0` (requires #88 merged) | E3.a.3 (#88) |
| Algorithmic enrichment in `remember_batch` | ingest | `extract_entities=False`, `extract=False` kwargs (rule-based always-on post-E2 is not separately toggled) | E2 (#82) |
| LLM extraction | ingest | `extract=True` kwarg (per-row cost); out of scope for Arms A-C |  |
| Pure-recall mode (no harness oracles) | harness | `--pure-recall` or `MNEMOSYNE_BENCHMARK_PURE_RECALL=1` (requires #90/#91) | #90, #91 |
| Veracity tagging at ingest | ingest | `remember_batch(items, veracity=...)` per-item or default | E4 (#74), E4.a.1 (#89) |
| Score-component weights | linear | `MNEMOSYNE_VEC_WEIGHT`, `MNEMOSYNE_FTS_WEIGHT`, `MNEMOSYNE_IMPORTANCE_WEIGHT` (existing; normalized to 1.0) |  |
| FTS5 / vec scan breadth | both | `MNEMOSYNE_BEAM_OPTIMIZATIONS=1` widens FTS OR-semantics + raises vec scan limits |  |
| Vector storage backend | both | `MNEMOSYNE_VEC_TYPE` ∈ {`float32`, `int8`, `bit`} |  |
| Recall diagnostics (per-tier hits, fallback rate) | observability | `mnemosyne.core.recall_diagnostics.get_recall_diagnostics()` | C4 (#79) |
| Extraction diagnostics (per-tier extract counts) | observability | `mnemosyne.extraction.diagnostics.get_extraction_stats()` | C13.b (#78) |

The "proposed" rows are the gap between today's code (assuming #80-#91 merged) and a clean A/B matrix. Scope is in §6.

---

## Proposed test sequence

Each phase changes exactly one variable from the prior. Run on the 100K slice unless noted; budget ~20-30 minutes per phase on a single conversation. Use `--pure-recall` on every phase -- running without it reintroduces the harness oracles and invalidates per-tool deltas.

Phases 3a-3d ablate components of the polyphonic engine specifically. They can run in parallel from a frozen DB snapshot taken at Phase 2 ingest completion, since each toggle is independent of the others. Phase 3-LIN-* exercises the analogous LINEAR-path bonus blocks separately -- those aren't redundant with 3a-3d because the linear graph/fact mechanisms are different functions on different data.

**Every phase requires:** `MNEMOSYNE_BENCHMARK_PURE_RECALL=1` (or `--pure-recall`), `OPENROUTER_API_KEY` set, and the Python deps from §1 installed. The per-phase table below lists only the variables that change from defaults.

| Phase | Name | Setup (delta from prior) | What to measure | Expectation |
|---|---|---|---|---|
| 0 | Baseline floor | Linear scorer, pure-recall, `MNEMOSYNE_VERACITY_MULTIPLIER=0` (explicit toggle, NOT data labeling -- uniform `unknown` rows still get the 0.8 multiplier), no enrichment | Per-ability score + total | The "raw working_memory + episodic via FTS + numpy-vec, no multipliers" floor |
| 1 | + veracity | Phase 0 + `MNEMOSYNE_VERACITY_MULTIPLIER=1`, rows tagged via `remember_batch` defaults | Δ vs phase 0 | Test Thesis 5 (prior: small Δ on uniform corpora; report CIs) |
| 2 | + polyphonic engine | Phase 1 + `MNEMOSYNE_POLYPHONIC_RECALL=1`, all voices on | Δ vs phase 1 | Net engine contribution -- RRF + diversity rerank vs linear scorer |
| 3a | − fact voice (polyphonic) | Phase 2 + `MNEMOSYNE_VOICE_FACT=0` | Δ vs phase 2 | Test Thesis 1: predict near-zero -- see thesis details below |
| 3b | − graph voice (polyphonic) | Phase 2 + `MNEMOSYNE_VOICE_GRAPH=0` | Δ vs phase 2 | Test Thesis 2a (polyphonic graph voice contribution) |
| 3c | − temporal voice (polyphonic) | Phase 2 + `MNEMOSYNE_VOICE_TEMPORAL=0` | Δ vs phase 2 | Question-mix dependent |
| 3d | − polyphonic vector voice | Phase 2 + `MNEMOSYNE_VOICE_VECTOR=0` | Δ vs phase 2 | **This is the polyphonic-engine analog of "no vector signal" -- gates `_vector_voice()` in `polyphonic_recall.py:113-151`, NOT the linear binary_bonus block. Vector voice is the heaviest contributor; predict large Δ.** |
| 3-LIN-bin | − linear binary bonus | Phase 1 (linear) + `MNEMOSYNE_BINARY_BONUS=0` (gates `beam.py:2527-2545`) | Δ vs phase 1 | Test Thesis 3a (linear-only binary bonus deprecation) -- predict near-zero |
| 3-LIN-graph | − linear graph bonus | Phase 1 + `MNEMOSYNE_GRAPH_BONUS=0` (gates `beam.py:2497-2521`, 2632-2656) | Δ vs phase 1 | Test Thesis 2b (linear graph-bonus contribution) |
| 3-LIN-fact | − linear fact bonus | Phase 1 + `MNEMOSYNE_FACT_BONUS=0` (gates `beam.py:2508-2521`, 2640-2655) | Δ vs phase 1 | Test Thesis 1b (linear fact-bonus contribution) |
| 4 | + cross-tier dedup off | Phase 2 + `MNEMOSYNE_CROSS_TIER_DEDUP=0` | Δ vs phase 2 | Test Thesis 4 |
| 5 | + algorithmic enrichment | Phase 2 + `extract_entities=True` at ingest | Δ vs phase 2 | Graph+fact data populated → voices have substance |
| 6 | Full 250K confirmation | Top-2 configurations from phases 0-5 + Phase 0 floor | Per-ability + total + latency | Confirms 100K findings hold at scale |

A full 2^N factorial across the voices is more rigorous but costs N²-ish runs; the single-variable ablations capture the dominant per-component contributions cheaply. If any 3a-3d delta surprises (e.g., voice X looks important when predicted dead), follow up with a paired-voice ablation (X + Y both off).

---

## Theses

Each is a **prior** with code grounding, plus the test that would prove or disprove it. **The quantitative predictions are priors, not measurement-backed claims** -- there are no pilot runs with confidence intervals behind them. Report bootstrap CIs or per-question paired deltas during execution; a "prediction confirmed" with overlapping CIs is not confirmation.

**Thesis 1a -- The polyphonic fact voice contributes effectively zero to final recall.**
`_fact_voice` (`polyphonic_recall.py:~200-217`) emits synthetic `cf_{subject}_{predicate}_{object}` IDs. `_recall_polyphonic` (`beam.py:~2935-2938`) skips any ID starting with `cf_` because `_fetch_polyphonic_row` can't map them to a real row. `_combine_voices` (`polyphonic_recall.py:~295-336`) does not join `cf_*` IDs to source memory rows -- the consolidator's `sources` list isn't threaded through. So when the fact voice "fires," its rows are filtered out before composition. Indirect contribution via RRF on memory_ids surfaced by other voices is also zero in this code, because the fact voice doesn't surface those IDs. **Prior:** Δ ≤ 0.5pp on total score, indistinguishable from noise. **Test:** Phase 3a. If true, the polyphonic fact voice is currently dead code in the recall path -- the cost is the consolidator's ingest-time Bayesian compounding, which has separate value but isn't tested here.

**Thesis 1b -- The LINEAR fact bonus is on a different mechanism and should be tested separately.**
The linear path at `beam.py:~2508-2521` per-row queries the `facts` table by `source_msg_id` and adds a capped bonus (max 0.1) when query tokens overlap with extracted fact tokens. This is NOT the same code path as the polyphonic fact voice -- it operates on `facts`, not `consolidated_facts`. **Prior:** Δ small-but-positive (1-3pp) on questions whose answer involves an extracted fact; zero on others. **Test:** Phase 3-LIN-fact.

**Thesis 2a -- The polyphonic graph voice's contribution depends heavily on entity-extraction quality.**
`_graph_voice` (`polyphonic_recall.py:~160-183`) extracts capitalized tokens from the query and calls `find_gists_by_participant` / `find_facts_by_subject`. Failure modes: case-sensitivity (lowercase entities ignored), brittle entity extraction, and the suspect `fact.id.split("_")[-1]` mapping that assumes a particular ID schema. **Prior:** uneven Δ -- 2-4pp on questions with clearly-capitalized proper-noun entities, near-zero otherwise. **Test:** Phase 3b. Stratify the analysis by whether the question contains capitalized tokens.

**Thesis 2b -- The LINEAR graph bonus rewards connectivity, not query relevance.**
`beam.py:~2497-2505` counts edges in `graph_edges` via `subject LIKE %memory_id% OR target LIKE %memory_id%`, capped at 0.08. Densely-connected rows get the bonus regardless of whether they're actually relevant to the query -- well-connected rows are surfaced more often than they should be. **Prior:** Δ near-zero on total score (the cap is small), possibly negative on questions where the densely-connected rows aren't the answer. **Test:** Phase 3-LIN-graph.

**Thesis 3a -- The LINEAR binary-vector bonus block is deprecation-eligible.**
`beam.py:~2527-2545` adds a tanh-normalized Hamming-distance bonus capped at 0.08 from the `binary_vector` column when both query and row vectors are present. The float-vector signal in `vec_results` already drives most of the ranking through `vec_sim * vw`. **Prior:** Δ < 1pp. **Test:** Phase 3-LIN-bin.

**Thesis 3b -- The POLYPHONIC vector voice is the heaviest contributor and should NOT be conflated with 3a.**
`_vector_voice` (`polyphonic_recall.py:113-151`) runs `BinaryVectorStore.search` (pre-#80) or sqlite-vec ANN over `memory_embeddings` (post-#80). Either way it's the highest-RRF-rank-weight voice and dominates the polyphonic engine's signal. Deprecating it via Phase 3d would tank polyphonic recall scores. **Prior:** Δ large-negative (5-20pp) when disabled. **Test:** Phase 3d. **Do not conclude binary-vector deprecation from Phase 3d** -- that decision is Phase 3-LIN-bin's territory, not 3d's.

**Thesis 4 -- Cross-tier dedup matters more for linear than polyphonic.**
Pre-#88 the linear path returned (summary, source) pairs ranked side-by-side. Polyphonic's diversity rerank handles this approximately via embedding similarity. **Prior:** linear regresses 3-8pp without dedup; polyphonic regresses <2pp. **Test:** Phase 4 × {Phase 0 baseline, Phase 2}. Requires #88 merged (`_dedup_cross_tier_summary_links` doesn't exist pre-merge).

**Thesis 5 -- Veracity multiplier is near-noise on uniform-veracity BEAM data.**
If every WM and EP row has the same veracity label, the multiplier becomes a global scalar that doesn't change ranking. Post-#89 the consolidator now aggregates source veracity, so summary rows can have different veracity from sources -- slightly breaking uniformity. **Prior:** Δ < 1pp on total score, possibly larger on questions answered from consolidated summaries vs raw sources. **Test:** Phase 1 vs Phase 0 with explicit multiplier toggle (NOT data labeling -- uniform rows still get the multiplier applied; only the toggle short-circuits it).

**Thesis 6 -- The RECENT CONVERSATION leak was the dominant prior driver of arm-vs-arm equivalence.**
Pre-#90 every answer prompt included the last 12 raw messages. Recency-anchored answers succeeded without needing recall -- arm choice was invisible. **Prior:** pure-recall mode shows arm-vs-arm deltas 2-5× larger than legacy mode. **Test:** Phase 2 with `--pure-recall` vs without (legacy harness behavior). Also serves as a sanity check that #90's gate is actually firing.

**Thesis 7 -- sqlite-vec ANN vs numpy exact-vec affects latency, not score (post-#80).**
The polyphonic engine over-fetches `top_k * 2`; the final RRF + multiplier + dedup pipeline returns top_k. ANN-vs-exact differences in the long tail past position ~60 get truncated. **Prior:** Δ score < 0.5pp; p95 latency divergence 5-20×. **Test:** Phase 2 with `sqlite-vec` importable vs mocked-unavailable (numpy fallback). Requires #80 merged.

**Thesis 8 -- Algorithmic enrichment captures most of what LLM extraction would.**
BEAM facts are predominantly subject-verb-object patterns the regex extractor handles. LLM extraction catches more nuance at $25-$2500 + 35-138h per 250K pass (E2.a.3 ledger note). **Prior:** ≤5pp gap between Phase 5 (algorithmic) and a separate `extract=True` LLM-extraction run. **Test:** Phase 5 vs a separate Arm D run. If gap ≤5pp, LLM extraction is cost overhead; if gap >10pp, it's a real capability.

---

## Toggles needed before full A/B

Each toggle is its own scope (not 5 LOC). LOC estimates include the env-parse code, the gate site(s), invalidation of cached `PolyphonicRecallEngine` if the toggle affects voice construction, and 2-3 regression tests per toggle.

| Toggle | Purpose | Sites | LOC (incl. tests) |
|---|---|---|---|
| `MNEMOSYNE_VOICE_VECTOR=0/1` | Phase 3d -- polyphonic vector voice | `polyphonic_recall.py:_vector_voice` early-return when off | ~25 |
| `MNEMOSYNE_VOICE_GRAPH=0/1` | Phase 3b -- polyphonic graph voice | `polyphonic_recall.py:_graph_voice` early-return | ~25 |
| `MNEMOSYNE_VOICE_FACT=0/1` | Phase 3a -- polyphonic fact voice | `polyphonic_recall.py:_fact_voice` early-return | ~25 |
| `MNEMOSYNE_VOICE_TEMPORAL=0/1` | Phase 3c -- polyphonic temporal voice | `polyphonic_recall.py:_temporal_voice` early-return | ~25 |
| `MNEMOSYNE_GRAPH_BONUS=0/1` | Phase 3-LIN-graph -- linear graph bonus | Gate `beam.py:~2497-2521` (main ep loop) AND `:~2632-2645` (fallback) | ~30 |
| `MNEMOSYNE_FACT_BONUS=0/1` | Phase 3-LIN-fact -- linear fact bonus | Gate `beam.py:~2508-2521` AND `:~2640-2655` | ~30 |
| `MNEMOSYNE_BINARY_BONUS=0/1` | Phase 3-LIN-bin -- linear binary-vector bonus | Gate `beam.py:~2527-2545` | ~20 |
| `MNEMOSYNE_VERACITY_MULTIPLIER=0/1` | Phase 0/1 -- disable veracity multiplier in both engines | Short-circuit to 1.0 in linear (`beam.py:~2700-2719`) + polyphonic (`beam.py:~3015-3017`) | ~25 |
| `MNEMOSYNE_CROSS_TIER_DEDUP=0/1` | Phase 4 -- disable E3.a.3 dedup | Short-circuit `_dedup_cross_tier_summary_links` to return input list unchanged | ~20 |

Total ~225 LOC across ~30 tests if all nine are implemented in one PR. A subset is fine if you want to prioritize: vector-voice toggle (3d) + veracity toggle (0/1) + cross-tier toggle (4) are the three most important for the experiment's central claims. The linear-bonus toggles (3-LIN-*) matter only if you want to deprecate components of the linear scorer; they're independent of the polyphonic-vs-linear comparison.

---

## Implementation gaps -- what the plan needs that does NOT yet exist

This is the complete list of functionality the plan assumes exists but doesn't (as of 2026-05-12, prior to PRs #80-#91 merging). The plan can't be executed cleanly until these gaps close. Each item is callable as its own small PR; estimates include tests.

### Gap A -- PR merges (no new code; just merge what's open) -- ✅ closed 2026-05-12

Merging PRs #80, #82, #88, #89, #90, #91 closes most of the prerequisite footprint. PRs #83, #85, #86, #87 are also open but not strictly required for the experiment to run -- they harden surrounding paths against silent failure but don't gate the test sequence itself.

### Gap B -- Nine A/B toggles (the big one -- ~225 LOC) -- ✅ closed in PR A

See §6 above. Without these, phases 0, 1, 3a-3d, 3-LIN-*, and 4 cannot be run cleanly -- they either change multiple variables at once or have no working knob. **Required to execute phases 0-5 as designed.**

### Gap C -- Harness preflight assertions (~80 LOC) -- ✅ closed in this PR

Add to `_benchmarks/evaluate_beam_end_to_end.py:main()` near argument parsing:

1. **Pure-recall guard:** `sys.exit(1)` with explanatory message if `MNEMOSYNE_BENCHMARK_PURE_RECALL` is not truthy AND `--pure-recall` was not passed AND a new `--allow-harness-oracles` opt-out wasn't passed (existing benchmark workflows might legitimately want the legacy mode for ceiling tests; require explicit opt-in to be sure).
2. **Env-var dump:** at run start, snapshot every env var matching `^MNEMOSYNE_|^FULL_CONTEXT_MODE$|^OPENROUTER_BASE_URL$` into the results JSON under `config.env`.
3. **Active recall path logger:** wrap `BeamMemory.recall` to record whether each call took the linear or polyphonic branch; emit a count in the per-conversation summary so an accidental mid-run mode flip is visible.
4. **Sentinel feature checks:** assert importable: `_dedup_cross_tier_summary_links` from `beam.py` (proves #88 merged), `aggregate_veracity` from `veracity_consolidation.py` (proves #89 merged), `_env_truthy` from the harness module itself (proves #91 merged). Fail fast with the missing-PR name.

### Gap D -- Harness diagnostic capture (~60 LOC) -- ✅ closed in this PR

Wire the harness to call `get_recall_diagnostics()` and `get_extraction_stats()` and write their JSON snapshots into `results/beam_e2e_results.json` per-conversation per-phase. Also call `reset_recall_diagnostics()` at the start of each phase to keep counters clean. The functions exist (PR #78 + #79); the harness simply doesn't use them.

### Gap E -- Per-question paired-outcome recording for CIs (~40 LOC) -- ✅ closed

Currently the harness writes per-question scores but doesn't structure them for paired statistical analysis. Add: for each (config, question) pair, record whether the answer was correct. Output a flat `paired_outcomes.jsonl` with `{config_id, question_id, ability, correct}` rows so a downstream notebook can bootstrap CIs without re-parsing the main results.

### Gap F -- `requirements-benchmark.txt` (~5 LOC) -- ✅ closed in this PR

Add a top-level file listing `numpy`, `datasets`, `sqlite-vec`, `fastembed`, optionally `llama-cpp-python` and `huggingface-hub`. Reference it from the README under "Running the BEAM benchmark." The current pyproject `[project.optional-dependencies]` groups (`llm`, `embeddings`, `mcp`, `all`) don't cover the benchmark-only deps (`numpy`, `datasets`, `sqlite-vec`).

### Gap G -- Voice-score capture in linear-path results (~20 LOC) -- ✅ closed

The polyphonic results already carry `voice_scores: dict` per result dict (`beam.py:~2969`). Linear-path results don't -- they have `dense_score`, `keyword_score`, `fts_score` separately but no unified provenance dict. For uniform post-hoc analysis ("which signal drove this row?"), add a similarly-shaped provenance dict to linear results too. Not strictly required to run the plan, but materially eases analysis. Defer if scope is tight.

### Total effort to close all gaps

- Required for plan execution (A + B + C + D): ~365 LOC + 40 tests + 6 PR merges
- Highly recommended (E + F): ~45 LOC + 1 doc file
- Nice-to-have (G): ~20 LOC

A single "experiment-readiness" PR bundling B + C + D + E + F is ~510 LOC and one /review pass. Or split into three smaller PRs:
1. Toggles (B) -- biggest, most reviewable on its own
2. Preflight + diagnostic capture (C + D + F) -- harness-only, no recall-path changes
3. Paired-outcome recording (E) -- orthogonal, easy

Tell me which order you want them in.

---

## What to record per run

For full output-file schemas + analysis recipes (paired bootstrap CIs, ability deltas, fallback-rate sanity checks), see [`../benchmark-results-analysis.md`](../benchmark-results-analysis.md). Quick reference:

Capture into `results/beam_e2e_results.json`:

- **Per-ability score** -- TR / CR / IE / KU / MR / ABS / EO / SUM, plus total.
- **Run config** -- phase number, all toggle states, polyphonic flag, sample size, scale.
- **Recall diagnostics** (`mnemosyne.core.recall_diagnostics.get_recall_diagnostics()` returns a JSON-serializable Dict snapshot; call once per recall and aggregate, OR once at run end): per-tier kept counts (`wm_fts`, `wm_vec`, `wm_fallback`, `em_fts`, `em_vec`, `em_fallback`), `fallback_rate`, `truly_empty` count. Use `reset_recall_diagnostics()` from the same module to clear counters before each phase. **The harness currently does NOT call these -- wiring this in is part of the implementation-gaps list below.**
- **Extraction diagnostics** (`mnemosyne.extraction.diagnostics.get_extraction_stats()`): per-tier extract counts + bounded error samples. Also not currently called by the harness.
- **Latency** -- p50 / p95 / p99 per-question recall + answer roundtrip.
- **Storage** -- final row counts in `working_memory`, `episodic_memory`, `memory_embeddings`, `vec_episodes`, `annotations`, `consolidated_facts`.
- **Peak RSS** during ingest phase and recall phase separately.

The diagnostics fields are the single biggest improvement to post-hoc analysis -- they tell you WHICH tier produced each kept row, which lets you attribute score deltas to specific code paths instead of guessing.

**Statistical reporting:** record per-question paired outcomes (was the answer correct in config A and config B for the same question?) and compute bootstrap CIs on the per-ability score deltas. With ~50 questions per conversation and 1-3 conversations per scale, a 2pp delta on total score is likely within noise; treat sub-3pp deltas as inconclusive until CIs separate. Where a thesis predicts "near-zero Δ," the right falsification criterion is "95% CI excludes ±2pp," not point-estimate equality.

---

## Recommended sequence of runs

Minimum credible A/B requires ~10-12 runs:

1. Phase 0 -- 100K (baseline floor, all multipliers off)
2. Phase 1 -- 100K (+ veracity multiplier)
3. Phase 2 -- 100K (+ polyphonic engine)
4. Phase 3a -- 100K (− polyphonic fact voice)
5. Phase 3b -- 100K (− polyphonic graph voice)
6. Phase 3c -- 100K (− polyphonic temporal voice)
7. Phase 3d -- 100K (− polyphonic vector voice -- expected large negative Δ; this is the polyphonic-engine analog of "no vector signal")
8. Phase 4 -- 100K (− cross-tier dedup)
9. Phase 5 -- 100K (+ algorithmic enrichment, populates entity/fact data)
10. Phase 6 -- 250K confirmation on the top-2 configs + Phase-0 floor

Phases 3a-3d can run in parallel from a Phase-2 DB snapshot since each toggle is independent of the others.

Approximate wall-clock budget: 5-6 hours total if 100K phases land in ~20 min each and the 250K run takes 60-90 min. Run a sample-1 conversation dry-run first to confirm corpus loaded, embeddings populated, and the preflight assertions fire.

**Optional linear-scorer ablations** (only if you want to deprecate components of the linear path; not required for the polyphonic-vs-linear comparison):

- Phase 3-LIN-bin -- linear binary-vector bonus off
- Phase 3-LIN-graph -- linear graph bonus off
- Phase 3-LIN-fact -- linear fact bonus off

**Optional ceiling and lower-bound bookends:**

- Phase 7 -- `--full-context` (LLM ceiling without recall). Sets the upper bound.
- Phase 8 -- Phase 6 with `MNEMOSYNE_VERACITY_MULTIPLIER=0`. Cleanest possible linear-vs-polyphonic A/B at scale.

If any thesis turns out wildly wrong (e.g., fact voice contributes 5+pp on Thesis 1a), pause and investigate before continuing -- the surprise likely indicates a code path the plan doesn't account for. The PRs ship the harness changes; the testing remains yours.

---

**Open PRs at handoff:** #80, #82, #83, #85, #86, #87, #88, #89, #90, #91. None are experiment-blocking individually, but the experiment can't run cleanly until at least #82, #88, #89, and #90 are merged. The rest improve specific tools' measurability and observability.
