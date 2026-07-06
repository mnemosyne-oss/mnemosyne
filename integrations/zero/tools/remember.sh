#!/usr/bin/env bash
# memory_remember — store a durable memory via the mnemosyne CLI.
# Reads JSON args on stdin: {"content": "...", "source": "fact", "importance": 0.5}
set -euo pipefail

input="$(cat)"

content="$(echo "$input" | jq -r '.content // empty')"
source_tag="$(echo "$input" | jq -r '.source // "fact"')"
importance="$(echo "$input" | jq -r '.importance // 0.5')"

if [ -z "$content" ]; then
  echo "Error: content is required"
  exit 1
fi

# Store the memory — mnemosyne store prints the memory ID on success.
mnemosyne store "$content" "$source_tag" "$importance" 2>&1
