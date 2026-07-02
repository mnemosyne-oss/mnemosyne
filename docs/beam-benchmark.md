# Mnemosyne BEAM Benchmark — v3.0.0

**Evaluated against the BEAM dataset (Mohammadta/BEAM on HuggingFace)**
**Date:** 2026-05-18 | **Version:** Mnemosyne 3.0.0 (MEMORIA Fact Engine) | **Model:** Llama 3.3 70B via NVIDIA API

> **v3.0.0 introduces MEMORIA** — structured fact extraction and retrieval. Temporal fact triples, smart routing by question type, recursive gap analysis, and proactive memory linking. These results replace all pre-MEMORIA benchmarks. See [benchmarking.md](benchmarking.md) for methodology and [benchmark-results-analysis.md](benchmark-results-analysis.md) for output schemas.

---

## End-to-End Results (LLM-as-Judge, Rubric Scoring)

Full BEAM protocol. Mnemosyne ingests conversations, retrieves context, LLM answers, LLM judges.

**Important:** Our run uses Llama 3.3 70B (system model) + DeepSeek V4 Flash (judge). Hindsight's published result (73.4%) uses Llama-4-Maverick as judge. Scores are not directly comparable across different judges — this is disclosed transparently.

| Scale | Mnemosyne v3 | Honcho | Hindsight | LIGHT | RAG |
|-------|-------------|--------|-----------|-------|-----|
| **100K** | **65.2%** | 63.0% | 73.4% | 35.8% | 32.3% |

Published baselines from Tavakoli et al. (ICLR 2026) and Hindsight blog (Apr 2026). Mnemosyne is competitive at 100K scale in a local-first setup. We have not yet published 10M-scale results.

| Tier | Hindsight | Honcho | LIGHT | RAG |
|-------|-----------|--------|-------|-----|
| 100K | 73.4% | 63.0% | 35.8% | 32.3% |
| 500K | 71.1% | 64.9% | 35.9% | 33.0% |
| 1M | 73.9% | 63.1% | 33.6% | 30.7% |
| 10M | 64.1% | 40.6% | 26.6% | 24.9% |

---

## Per-Ability Breakdown — 100K

| Ability | Score | Assessment |
|---------|-------|------------|
| **ABS** (Abstention) | 100.0% | Perfect. Knows when it doesn't know. |
| **IE** (Information Extraction) | 91.5% | Near-perfect fact retrieval from structured MEMORIA tables. |
| **MR** (Multi-hop Reasoning) | 87.5% | Strong. Gap analysis + recursive re-querying connects facts across turns. |
| **TR** (Temporal Reasoning) | 75.0% | Temporal triples with valid-from/to windows enable date reasoning. |
| **IF** (Instruction Following) | 62.5% | Structured instruction storage with veracity-weighted retrieval. |
| **SUM** (Summarization) | 55.6% | LLM consolidation compresses episode summaries without losing signal. |
| **PF** (Preference Following) | 54.5% | Preference facts extracted and versioned with previous-value tracking. |
| **CR** (Contradiction Resolution) | 50.0% | UNION search across episodic + structured facts catches contradictions. |
| **KU** (Knowledge Update) | 50.0% | Context-aware metric keys prevent key collisions. Fact version chains preserve history. |
| **EO** (Event Ordering) | 25.0% | Hardest ability. Strict JSON mode with negative examples reduces rambling but ordering remains difficult. |

**Overall: 65.2%**

---

## What Changed From v2.5

v2.5 scored 35.4% at 100K with Gemini 2.5 Flash. v3.0.0 scores 65.2% with Llama 3.3 70B.

| Ability | v2.5 (35.4%) | v3.0.0 (65.2%) | Delta |
|---------|-------------|----------------|-------|
| IE | 80.5% | 91.5% | +11.0 |
| MR | 16.7% | 87.5% | +70.8 |
| TR | 29.2% | 75.0% | +45.8 |
| KU | 16.7% | 50.0% | +33.3 |
| EO | 13.3% | 25.0% | +11.7 |
| CR | 35.4% | 50.0% | +14.6 |
| ABS | 50.0% | 100.0% | +50.0 |
| SUM | 41.7% | 55.6% | +13.9 |

The largest gains are in multi-hop reasoning (+70.8pp), temporal reasoning (+45.8pp), and knowledge update (+33.3pp) — the exact abilities MEMORIA's structured fact triples and gap analysis target.

---

## Ingestion Performance

Memory ingestion with full MEMORIA extraction (entity extraction, fact triples, proactive linking):

- **188 messages ingested** in 36 seconds
- **FTS5 + vector search** active throughout
- **Proactive linking** (opt-in via `MNEMOSYNE_PROACTIVE_LINKING=1`) adds ~5% overhead
- **Host extraction** — no external API calls at ingestion, fully local

---

## Run on Your Hardware

You can reproduce these results with your own LLM. The benchmark is deterministic — same dataset, same questions, same rubric.

### Quick Start

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install mnemosyne-memory[all]
pip install datasets numpy  # benchmark deps

export OPENROUTER_API_KEY="your-key"
python _benchmarks/evaluate_beam_end_to_end.py --sample 5 --scales 100K
```

### Scales

```bash
# Fast test (5 conversations, 1 scale)
--sample 5 --scales 100K

# Full SOTA run (all conversations, all scales)
--sample 0 --scales 100K,500K,1M,10M
```

### Models

Any OpenRouter model works. For reproducing published results:

```bash
python _benchmarks/evaluate_beam_end_to_end.py \
  --model "meta-llama/llama-3.3-70b-instruct" \
  --judge-model "deepseek/deepseek-v4-flash" \
  --sample 0 --scales 100K
```

### Pure Recall Mode

Measures retrieval quality only (no LLM answering). Useful for isolating Mnemosyne's recall from the LLM's intelligence:

```bash
python _benchmarks/evaluate_beam_end_to_end.py --pure-recall --sample 5 --scales 100K
```

See [benchmarking.md](benchmarking.md) for the full env-var reference, diagnostic tools, and A/B experiment methodology. See [benchmark-results-analysis.md](benchmark-results-analysis.md) for output file schemas and statistical analysis.
