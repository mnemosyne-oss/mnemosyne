#!/bin/bash
# Multi-scale BEAM benchmark — hybrid mode (context→value + full-context)
# Usage: source /tmp/openrouter_key.txt && bash run_hybrid_all_scales.sh [sample]

cd /root/.hermes/projects/mnemosyne

SAMPLE="${1:-3}"

echo "=== BEAM Multi-Scale Hybrid Benchmark ==="
echo "Model: openai/gpt-4.1 | Mode: FULL_CONTEXT + context→value"
echo "Sample: $SAMPLE per scale | Scales: 100K 500K 1M 10M"
echo ""

for scale in 100K 500K 1M 10M; do
    echo "--- Scale: $scale ---"
    source /tmp/openrouter_key.txt
    FULL_CONTEXT_MODE=1 \
    MNEMOSYNE_BEAM_OPTIMIZATIONS=1 \
    MNEMOSYNE_EMBEDDING_MODEL=openai/text-embedding-3-large \
        .venv/bin/python _benchmarks/evaluate_beam_end_to_end.py \
        --sample "$SAMPLE" \
        --scales "$scale" \
        --model openai/gpt-4.1 \
        --judge-model openai/gpt-4o
    echo ""
done

echo "=== All scales complete ==="
