#!/usr/bin/env bash
# after-tool hook — auto-captures file changes to Mnemosyne memory.
#
# Zero sends JSON on stdin with: event, tool, toolCallId, sessionId, cwd,
# status, changedFiles. This hook stores a compact memory of file-editing
# actions so the agent recalls it changed in future sessions.
#
# Only captures tools that modify files (write, edit, patch, bash with changed
# files). Read-only tools are skipped to avoid noise.
#
# All jq parsing is non-fatal: a malformed payload must never abort the hook
# before it reaches `exit 0`, because afterTool hooks are advisory and must
# never block tool execution.
set -euo pipefail

input="$(cat)"

tool="$(echo "$input" | jq -r '.tool // empty' 2>/dev/null || echo '')"
status="$(echo "$input" | jq -r '.status // empty' 2>/dev/null || echo '')"
changed_files="$(echo "$input" | jq -r '.changedFiles // [] | if length > 0 then .[] else empty end' 2>/dev/null || echo '')"
session_id="$(echo "$input" | jq -r '.sessionId // "unknown"' 2>/dev/null || echo 'unknown')"
cwd="$(echo "$input" | jq -r '.cwd // empty' 2>/dev/null || echo '')"

# Skip if the tool didn't succeed — no point remembering failures.
if [ "$status" != "ok" ] && [ "$status" != "success" ]; then
  exit 0
fi

# Skip read-only tools. Only capture tools that change files.
# Zero's write tools are named: write_file, edit_file, patch, bash.
case "$tool" in
  write_file|edit_file|patch|*write*|*edit*|*patch*)
    ;;
  *)
    # For bash, check if changedFiles is non-empty.
    if [ -z "$changed_files" ]; then
      exit 0
    fi
    ;;
esac

# Build a compact memory content string.
content="Zero session $session_id: tool '$tool' "
if [ -n "$changed_files" ]; then
  file_list="$(echo "$changed_files" | tr '\n' ',' | sed 's/,$//')"
  content+="modified files: $file_list"
else
  content+="executed (no file changes tracked)"
fi

# Add the working directory for context.
if [ -n "$cwd" ]; then
  content+=" in $cwd"
fi

# Store silently — importance 0.3 (low) since this is auto-captured context.
# Source tag "task" marks it as session activity.
mnemosyne store "$content" "task" "0.3" >/dev/null 2>&1 || true

# Exit 0 — afterTool hooks are advisory and must never block.
exit 0
