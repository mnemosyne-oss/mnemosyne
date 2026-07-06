#!/usr/bin/env bash
# memory_stats — show Mnemosyne memory statistics.
set -euo pipefail
mnemosyne stats 2>&1
