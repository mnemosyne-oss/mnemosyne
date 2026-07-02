#!/bin/bash
# Multi-scale BEAM benchmark runner
# Usage: source /tmp/openrouter_key.txt && bash run_beam_all_scales.sh

cd /root/.hermes/projects/mnemosyne

MODEL="${1:-openai/gpt-4o}"
JUDGE="${2:-openai/gpt-4o}"
SAMPLE="${3:-3}"

echo "=== BEAM Multi-Scale Benchmark ==="
echo "Model: $MODEL | Judge: $JUDGE | Sample: $SAMPLE per scale"
echo "Scales: 100K 500K 1M 10M"
echo ""

for scale in 100K 500K 1M 10M; do
    echo "--- Scale: $scale ---"
    source /tmp/openrouter_key.txt
    MNEMOSYNE_EMBEDDING_MODEL=openai/text-embedding-3-large \
        .venv/bin/python _benchmarks/evaluate_beam_end_to_end.py \
        --sample "$SAMPLE" \
        --scales "$scale" \
        --model "$MODEL" \
        --judge-model "$JUDGE"
    echo ""
done

echo "=== All scales complete ==="
echo "Results: /root/.hermes/projects/mnemosyne/results/beam_e2e_results.json"
