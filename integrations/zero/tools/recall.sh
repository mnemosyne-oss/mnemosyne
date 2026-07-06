#!/usr/bin/env bash
# memory_recall — semantic search of persistent memories.
# Reads JSON args on stdin: {"query": "...", "limit": 5}
set -euo pipefail

input="$(cat)"

IFS=$'\t' read -r query limit < <(
  jq -r '[.query // empty, .limit // 5] | @tsv' <<< "$input"
)

if [ -z "$query" ]; then
  echo "Error: query is required"
  exit 1
fi

mnemosyne recall "$query" "$limit" 2>&1
