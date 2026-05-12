# BEAM Benchmark — Output Files and Analysis Guide

**Audience:** anyone (human or AI assistant) reading the result files produced by `tools/evaluate_beam_end_to_end.py` and computing per-phase / per-arm comparisons.

This doc is intentionally self-contained: file paths, schemas with field types, example records, and worked analyses. An AI assistant pointed at this doc plus the result files should be able to compute paired bootstrap CIs without external context.

For the test-infrastructure overview (env vars, pure-recall mode, etc.) see [benchmarking.md](benchmarking.md). For the specific BEAM-recovery experiment plan see [experiments/2026-05-12-beam-recovery-arms-abc.md](experiments/2026-05-12-beam-recovery-arms-abc.md).

---

## Where the files live

After `tools/evaluate_beam_end_to_end.py` runs to completion, three files exist under `results/`:

```
results/
├── beam_e2e_results.json     # per-question results with full metadata + diagnostics
├── beam_e2e_summary.json     # aggregate ability scores per scale
└── paired_outcomes.jsonl     # one row per question, append-only across runs
```

The first two are overwritten per run. `paired_outcomes.jsonl` is **append-only** so multiple A/B runs accumulate. Filter by `config_id` to isolate a single phase.

---

## File 1: `beam_e2e_results.json`

Top-level JSON object: `{metadata, results}`. Schema:

```jsonc
{
  "metadata": {
    "date": "2026-05-12T14:30:00+00:00",        // run completion timestamp (UTC ISO 8601)
    "run_started_at": "2026-05-12T14:00:00+00:00", // run start timestamp
    "config_id": "phase3a-no-fact-voice",       // either --config-id arg or auto-hash
    "model": "deepseek-v4-pro",                 // answer LLM
    "judge_model": "deepseek-v4-pro",           // judge LLM (may differ from model)
    "top_k": 10,                                // recall top-k per question
    "sample_size": 3,                           // conversations per scale (or "ALL")
    "scales": ["100K", "500K"],                 // BEAM scales evaluated
    "total_conversations": 6,                   // count of conversations actually evaluated
    "config": {
      "env": {                                  // snapshot of MNEMOSYNE_* + FULL_CONTEXT_MODE at run start
        "MNEMOSYNE_POLYPHONIC_RECALL": "1",
        "MNEMOSYNE_VOICE_FACT": "0",
        "MNEMOSYNE_BENCHMARK_PURE_RECALL": "1"
        // API keys are redacted ("***redacted***")
      },
      "pure_recall": true,                      // pure-recall mode active
      "allow_harness_oracles": false,
      "full_context": false,
      "use_cloud": false
    },
    "diagnostics": {
      "recall": { /* see "recall diagnostics" below */ },
      "extraction": { /* see "extraction diagnostics" below */ }
    }
  },
  "results": [
    {
      "conversation_id": "conv_abc",
      "scale": "100K",
      "num_questions": 50,
      "num_evaluated": 50,
      "results": [
        {
          "qid": "q_001",
          "ability": "IE",                       // one of: IE, MR, KU, TR, ABS, CR, EO, IF, PF, SUM
          "question": "what is the user's favorite color?",
          "ideal_answer": "blue",
          "ai_answer": "The user's favorite color is blue.",
          "recall_provenance": {                 // see Recipe E for analysis usage
            "engine": "polyphonic",              // "polyphonic" / "linear" / "unknown"
            "kept_count": 10,
            "voice_sums": {"vector": 4.2, "graph": 1.1, "fact": 0.0, "temporal": 0.3},
            "top_result_voices": {"vector": 0.7, "graph": 0.2, "fact": 0.0, "temporal": 0.1},
            "top_result_tier": "episodic"        // tier of the top-ranked memory
          },
          "score": 1.0,                          // rubric score: 0.0, 0.5, or 1.0
          "nuggets": [...],                      // judge's nugget breakdown (may be empty)
          "assessment": "Correctly identifies blue from context.",
          "answer_time_ms": 1234.5,              // LLM answer latency
          "judge_time_ms": 567.8                 // judge latency
        }
        // ... one entry per question evaluated
      ]
    }
    // ... one entry per conversation evaluated
  ]
}
```

### `metadata.diagnostics.recall`

Recall pipeline provenance (`get_recall_diagnostics()` snapshot at run end). Tells you which recall tier produced each kept row — critical for sanity-checking that recall isn't dominated by the weak-signal substring-fallback path.

```jsonc
{
  "created_at": "2026-05-12T14:00:00+00:00",
  "snapshot_at": "2026-05-12T14:30:00+00:00",
  "totals": {
    "calls": 300,                       // total recall() invocations across the run
    "calls_using_wm_fallback": 12,      // calls that fell back to WM substring scan
    "calls_using_em_fallback": 5,
    "calls_truly_empty": 0,             // calls where every tier produced zero kept rows
    "wm_fallback_rate": 0.04,           // 0.0-1.0 — high (>0.2) = signal-poor, treat results skeptically
    "em_fallback_rate": 0.017
  },
  "by_tier": {
    "wm_fts": {"calls_with_hits": 280, "total_hits": 920},
    "wm_vec": {"calls_with_hits": 295, "total_hits": 1100},
    "wm_fallback": {"calls_with_hits": 12, "total_hits": 35},
    "em_fts": {"calls_with_hits": 250, "total_hits": 700},
    "em_vec": {"calls_with_hits": 290, "total_hits": 940},
    "em_fallback": {"calls_with_hits": 5, "total_hits": 12}
  }
}
```

### `metadata.diagnostics.extraction`

Fact-extraction pipeline provenance (`get_extraction_stats()` snapshot). Surfaces silent failures in the per-row LLM extraction tiers.

```jsonc
{
  "created_at": "...",
  "snapshot_at": "...",
  "totals": {
    "calls": 100,
    "successes": 95,
    "failures": 3,
    "empty": 2,
    "success_rate": 0.95
  },
  "by_tier": {
    "host": {"attempts": 0, "successes": 0, "no_output": 0, "failures": 0, "error_samples": []},
    "remote": {"attempts": 100, "successes": 95, "no_output": 2, "failures": 3,
               "error_samples": [{"timestamp": "...", "message": "rate limit", "exc_type": "HTTPError"}]},
    "local": {"attempts": 0, "successes": 0, "no_output": 0, "failures": 0, "error_samples": []},
    "cloud": {"attempts": 0, "successes": 0, "no_output": 0, "failures": 0, "error_samples": []},
    "wrapper": {"attempts": 0, "successes": 0, "no_output": 0, "failures": 0, "error_samples": []}
  }
}
```

---

## File 2: `beam_e2e_summary.json`

Aggregate ability scores per scale — small, summary-only.

```jsonc
{
  "date": "2026-05-12T14:30:00+00:00",
  "metadata": {
    "model": "deepseek-v4-pro",
    "sample_size": 3,
    "judge_model": "deepseek-v4-pro"
  },
  "ability_summary": {
    "100K": {
      "IE":  {"avg_score": 0.80, "count": 24},
      "MR":  {"avg_score": 0.16, "count": 12},
      "KU":  {"avg_score": 0.16, "count": 12},
      "TR":  {"avg_score": 0.29, "count": 24},
      "ABS": {"avg_score": 0.50, "count": 12},
      "CR":  {"avg_score": 0.35, "count": 24},
      "EO":  {"avg_score": 0.13, "count": 18},
      "SUM": {"avg_score": 0.42, "count": 12},
      "OVERALL": {"avg_score": 0.35, "count": 138}
    },
    "500K": { /* same shape */ }
  }
}
```

Use this for a quick at-a-glance scoreboard. For paired analysis, use `paired_outcomes.jsonl` instead.

---

## File 3: `paired_outcomes.jsonl`

One JSON object per line. **Append-only across runs.** Filter by `config_id` to isolate a phase.

```jsonc
{
  "config_id": "phase3a-no-fact-voice",       // string, equals --config-id or auto-hash
  "run_started_at": "2026-05-12T14:00:00+00:00", // UTC ISO 8601
  "scale": "100K",                             // BEAM scale tier
  "conversation_id": "conv_abc",               // BEAM conversation id
  "qid": "q_001",                              // BEAM question id (stable across runs)
  "ability": "IE",                             // ability code (see "ability codes" below)
  "score": 1.0,                                // raw rubric score: float 0.0–1.0
  "correct": true                              // bool: score >= 0.5 (partial credit counts)
}
```

Field meanings:

- `config_id` — every row in a single run shares this. Use for paired-arm comparisons. If `--config-id` not given, auto-derived as `"cfg-" + sha256(canonical_env)[:10]`.
- `qid` — stable across runs. Lets you pair "config A's score on Q" with "config B's score on Q".
- `score` — raw rubric. Use for continuous deltas + CIs.
- `correct` — bool threshold (`score >= 0.5`). Use for accuracy-style metrics. Analysts wanting stricter (e.g., only 1.0 = correct) can re-threshold off `score`.

---

## Ability codes reference

The BEAM benchmark scores 10 ability dimensions. Codes used in result files:

| Code | Name | What it measures |
|---|---|---|
| `IE` | Information Extraction | retrieve specific facts from context |
| `MR` | Multi-session Reasoning (a.k.a. Multi-hop) | connect facts across distant messages |
| `KU` | Knowledge Update | track value changes over time |
| `TR` | Temporal Reasoning | date math, ordering by time |
| `ABS` | Abstention | identify unanswerable questions |
| `CR` | Contradiction Resolution | flag conflicting info |
| `EO` | Event Ordering | sequence events chronologically |
| `IF` | Instruction Following | apply implicit instructions |
| `PF` | Preference Following | adhere to stated preferences |
| `SUM` | Summarization | synthesize across windows |

---

## Worked analyses

### A. Per-ability scores for a single run

Read `beam_e2e_summary.json.ability_summary[scale][ability].avg_score`. Done. Or:

```python
import json
data = json.load(open("results/beam_e2e_summary.json"))
for scale, abilities in data["ability_summary"].items():
    print(f"\n{scale}:")
    for ability, stats in abilities.items():
        print(f"  {ability}: {stats['avg_score']:.3f} ({stats['count']} questions)")
```

### B. Δ between two A/B configs on a single ability

Two runs were performed: one with `config_id=phase2-baseline-polyphonic`, one with `config_id=phase3a-no-fact-voice`. Compute the score delta on TR questions:

```python
import json
from statistics import mean

rows = [json.loads(line) for line in open("results/paired_outcomes.jsonl")]

def avg_for(config_id, ability):
    matching = [r["score"] for r in rows
                if r["config_id"] == config_id and r["ability"] == ability]
    return mean(matching) if matching else None

baseline = avg_for("phase2-baseline-polyphonic", "TR")
ablation = avg_for("phase3a-no-fact-voice", "TR")
print(f"Δ on TR = {ablation - baseline:+.3f}  ({len(matching)} questions)")
```

### C. Paired bootstrap CI on Δ total score

Bootstrap 5000 paired-resamples to get a 95% CI on the Δ between two configs across all questions:

```python
import json
import random
from statistics import mean

rows = [json.loads(line) for line in open("results/paired_outcomes.jsonl")]

# Group by config_id, then by qid → score
def scores_by_qid(config_id):
    return {r["qid"]: r["score"] for r in rows if r["config_id"] == config_id}

base = scores_by_qid("phase2-baseline-polyphonic")
abl = scores_by_qid("phase3a-no-fact-voice")
paired_qids = sorted(set(base) & set(abl))   # questions present in BOTH runs
paired_diffs = [abl[q] - base[q] for q in paired_qids]

n = len(paired_diffs)
print(f"Paired Δ: n={n}, point estimate={mean(paired_diffs):+.3f}")

# Bootstrap CI
rng = random.Random(42)
B = 5000
boot_means = []
for _ in range(B):
    resample = [rng.choice(paired_diffs) for _ in range(n)]
    boot_means.append(mean(resample))
boot_means.sort()
ci_lo, ci_hi = boot_means[int(0.025 * B)], boot_means[int(0.975 * B)]
print(f"  95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
print(f"  Falsifiable: {'yes (CI excludes 0)' if ci_lo > 0 or ci_hi < 0 else 'no (CI spans 0)'}")
```

### D. Fallback-rate sanity check

If `wm_fallback_rate` or `em_fallback_rate` is high (>0.2), recall is dominated by the weak-signal substring fallback and arm-vs-arm comparisons aren't credible — both arms are mostly hitting the same fallback path.

```python
import json
data = json.load(open("results/beam_e2e_results.json"))
diag = data["metadata"]["diagnostics"]["recall"]["totals"]
print(f"WM fallback rate: {diag['wm_fallback_rate']:.1%}")
print(f"EM fallback rate: {diag['em_fallback_rate']:.1%}")
if diag["wm_fallback_rate"] > 0.2 or diag["em_fallback_rate"] > 0.2:
    print("⚠ High fallback rate — arm-vs-arm comparisons may not be credible")
```

### E. Per-voice attribution (per question)

Each per-question result dict in `beam_e2e_results.json.results[i].results[j]` carries `recall_provenance` with a summary of which voices contributed. Shape:

```jsonc
{
  "recall_provenance": {
    "engine": "polyphonic",                  // or "linear" or "unknown"
    "kept_count": 10,                        // memories the LLM saw
    "voice_sums": {                          // per-voice score totals across kept results
      "vector": 4.2,
      "graph": 1.1,
      "fact": 0.0,
      "temporal": 0.3
    },
    "top_result_voices": {                   // voice_scores for the top-ranked memory
      "vector": 0.7,
      "graph": 0.2,
      "fact": 0.0,
      "temporal": 0.1
    },
    "top_result_tier": "episodic"
  }
}
```

To find what fraction of polyphonic-engine answers were vector-voice-led across a run:

```python
import json
from collections import Counter

data = json.load(open("results/beam_e2e_results.json"))
top_voice_counts = Counter()
for conv in data["results"]:
    for q in conv["results"]:
        prov = q.get("recall_provenance", {})
        if prov.get("engine") != "polyphonic":
            continue
        top_voices = prov.get("top_result_voices", {})
        if not top_voices:
            continue
        # Voice with the highest score on the top-ranked result
        leader = max(top_voices.items(), key=lambda kv: kv[1])[0]
        top_voice_counts[leader] += 1

print("Top-voice leader distribution (polyphonic engine):")
for voice, n in top_voice_counts.most_common():
    print(f"  {voice}: {n}")
```

To compute per-voice contribution by ability — e.g., "how much did the fact voice contribute to TR vs IE questions?":

```python
fact_contrib_by_ability = {}  # ability → list of fact voice_sums
for conv in data["results"]:
    for q in conv["results"]:
        prov = q.get("recall_provenance", {})
        if prov.get("engine") != "polyphonic":
            continue
        ability = q.get("ability")
        fact_contrib = prov.get("voice_sums", {}).get("fact", 0.0)
        fact_contrib_by_ability.setdefault(ability, []).append(fact_contrib)

for ability, vals in fact_contrib_by_ability.items():
    avg = sum(vals) / max(len(vals), 1)
    print(f"  {ability}: avg fact-voice sum = {avg:.3f} across {len(vals)} questions")
```

`recall_provenance.engine == "unknown"` and `kept_count == 0` indicates a bypass path (TR oracle / IE-KU context-fact match) returned the answer without going through `BeamMemory.recall()`. With `--pure-recall` active, this should only happen on full-context-mode runs.

### F. Detect a run that ran without pure-recall mode

A run without `--pure-recall` would have its results contaminated by harness oracles (see [benchmarking.md#pure-recall-mode](benchmarking.md#pure-recall-mode)). Check the metadata:

```python
import json
data = json.load(open("results/beam_e2e_results.json"))
if not data["metadata"]["config"]["pure_recall"]:
    if data["metadata"]["config"]["allow_harness_oracles"]:
        print("⚠ Run intentionally used harness oracles — not comparable to other arms")
    else:
        print("⚠ Run was not in pure-recall mode — results compromised")
```

The harness preflight refuses to start without pure-recall OR explicit `--allow-harness-oracles`, but the metadata is the post-hoc verification.

---

## For AI assistants

If you're an AI assistant reading these result files to help an operator interpret a BEAM benchmark run, here's what to know in compressed form:

1. **Three files matter.** `beam_e2e_results.json` (rich, per-question + metadata + diagnostics), `beam_e2e_summary.json` (per-ability averages, summary-only), `paired_outcomes.jsonl` (one row per question, append-only across multiple runs — filter by `config_id`).

2. **Scores are rubric, not binary.** `score` is `0.0` / `0.5` / `1.0` per question. `correct` (in the JSONL) is `score >= 0.5`. Two ways to compute "accuracy": mean of `score` (continuous), or mean of `correct` (binary at 0.5 threshold).

3. **A/B comparisons require paired questions.** Two A/B runs share `qid` values across `config_id`s. To compute Δ between configs, group by `config_id`, then inner-join on `qid`. Drop unpaired questions. Bootstrap CI on the paired differences (see Section C above).

4. **Verify the run was credible BEFORE trusting deltas.** Check three things:
   - `metadata.config.pure_recall == true` (no harness oracles)
   - `metadata.diagnostics.recall.totals.{wm,em}_fallback_rate < 0.2` (recall not dominated by weak-signal fallback)
   - `metadata.diagnostics.extraction.totals.success_rate > 0.9` (fact extraction working)

   If any of these fail, surface that to the operator before reporting score deltas.

5. **Treat 2pp as the noise floor.** With ~50 questions per conversation, only deltas with 95% CIs excluding ±2pp are credible. Sub-3pp deltas without overlapping CIs are inconclusive.

6. **The experiment plan is the authority on what each `config_id` means.** If `config_id == "phase3a-no-fact-voice"` you can look up Phase 3a in [`docs/experiments/2026-05-12-beam-recovery-arms-abc.md`](experiments/2026-05-12-beam-recovery-arms-abc.md) to find the prediction being tested. Always cross-reference; the plan documents the *intent* of each ablation.

7. **Per-ability deltas often tell a richer story than total deltas.** A toggle that drops total score by 0.5pp might drop TR by 8pp and lift IE by 7pp. Always report per-ability breakdowns alongside total.

8. **Two engines, two voice_score schemas.** If you're processing per-result voice_scores, engine-identify by the dict's keyset: polyphonic uses `{vector, graph, fact, temporal}`, linear uses `{vec, fts, keyword, importance, recency_decay}`.

---

## Where to look when a result surprises you

- **"This number looks too high."** Check `metadata.config.pure_recall`. If false, the harness oracles (TR timeline / CR injection / IE-KU `_context_facts` / RECENT CONVERSATION) likely produced answers without going through Mnemosyne recall.
- **"This number looks too low."** Check `metadata.diagnostics.recall.totals.fallback_rate`. If high, recall returned mostly fallback noise.
- **"Two runs that should be identical aren't."** Check `metadata.config.env` for both. The env snapshot includes every `MNEMOSYNE_*` env var at run start. A toggle the operator forgot to set is a silent confound.
- **"The polyphonic run scored lower than linear."** Check `voice_scores` on per-result level — possibly only 2 of 4 voices contributed because the others returned empty (e.g., temporal voice ignored a non-temporal query).
- **"My CI is huge."** Probably small sample size. Either increase `--sample` or report median+IQR alongside mean.
