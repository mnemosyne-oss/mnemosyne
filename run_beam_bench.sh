#!/bin/bash
# BEAM Benchmark runner
set -e
cd /root/.hermes/projects/mnemosyne

# Source API keys from hermes env
set -a
source ~/.hermes/.env
set +a

# Export the specific vars the benchmark needs
export OPENROUTER_API_KEY
export NVIDIA_API_KEY

# Pure recall mode — episodic consolidation bypass for clean benchmark numbers
export MNEMOSYNE_BENCHMARK_PURE_RECALL=1

echo "=== BEAM Benchmark Runner ==="
echo "OpenRouter key: ${OPENROUTER_API_KEY:0:20}..."
echo "Date: $(date -u)"
echo "Model: deepseek-v4-pro"
echo "Scales: $1"
echo "Sample: $2"
echo "=========================="

.venv/bin/python _benchmarks/evaluate_beam_end_to_end.py \
  --scales "$1" \
  --sample "$2" \
  --mode end_to_end \
  --top-k 10 \
  --max-context 8000 \
  2>&1
