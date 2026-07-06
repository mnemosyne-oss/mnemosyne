#!/usr/bin/env bash
# memory_forget — permanently delete a memory by ID.
# Reads JSON args on stdin: {"id": "..."}
set -euo pipefail

input="$(cat)"
id="$(echo "$input" | jq -r '.id // empty')"

if [ -z "$id" ]; then
  echo "Error: id is required"
  exit 1
fi

mnemosyne delete "$id" 2>&1
