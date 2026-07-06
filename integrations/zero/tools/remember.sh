#!/usr/bin/env bash
# memory_remember — store a durable memory via the mnemosyne CLI.
# Reads JSON args on stdin: {"content": "...", "source": "fact", "importance": 0.5}
set -euo pipefail

input="$(cat)"

IFS=$'\t' read -r content source_tag importance < <(
  jq -r '[.content // empty, .source // "fact", .importance // 0.5] | @tsv' <<< "$input"
)

if [ -z "$content" ]; then
  echo "Error: content is required"
  exit 1
fi

# Store the memory — mnemosyne store prints the memory ID on success.
mnemosyne store "$content" "$source_tag" "$importance" 2>&1
