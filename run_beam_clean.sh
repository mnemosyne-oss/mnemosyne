#!/bin/bash
# Run BEAM benchmark with clean environment - no gateway snap interference
cd /root/.hermes/projects/mnemosyne || exit 1

# Get API key from .env
OPENROUTER_KEY="$(grep '^OPENROUTER_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)"
if [ -z "$OPENROUTER_KEY" ]; then
    echo "ERROR: No OPENROUTER_API_KEY found"
    exit 1
fi

# Launch with clean env - only what we need
exec env -i \
    HOME="$HOME" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    USER="$USER" \
    TERM="$TERM" \
    OPENROUTER_API_KEY="$OPENROUTER_KEY" \
    MNEMOSYNE_BENCHMARK_PURE_RECALL=1 \
    MNEMOSYNE_LLM_ENABLED=true \
    MNEMOSYNE_LLM_BASE_URL="https://openrouter.ai/api/v1" \
    MNEMOSYNE_LLM_API_KEY="$OPENROUTER_KEY" \
    MNEMOSYNE_LLM_MODEL="deepseek/deepseek-v4-flash" \
    MNEMOSYNE_LLM_MAX_TOKENS=512 \
    .venv/bin/python -u _benchmarks/evaluate_beam_end_to_end.py \
        --scales 100K --sample 3 \
        --model "deepseek/deepseek-v4-flash" \
        --judge-model "deepseek/deepseek-v4-flash" 2>&1
