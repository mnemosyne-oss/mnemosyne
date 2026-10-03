#!/usr/bin/env python3
"""Verify that generated docs match the code they are generated from.

Usage:
    python3 scripts/verify-docs.py          # fail on drift
    python3 scripts/verify-docs.py --fix    # rewrite the files, then pass

WHY THIS CHECKS THE CANONICAL FILES
-----------------------------------
The previous implementation resolved `../mnemosyne-docs/src` and called
`sys.exit(0)` when that directory was absent. On the CI runner the sibling
repo is never checked out, so the docs-check job passed unconditionally and
verified nothing. Meanwhile the canonical `docs/api/*.mdx` in THIS repo,
the files the gate exists to protect, were never examined at all.

This version compares the generator's output against the committed
`docs/api/*.mdx` in-process, with no git and no sibling required, so the
gate does real work in CI. The sibling repo is still checked when present,
but it can only add findings, never mask them.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_generator():
    path = os.path.join(REPO, "scripts", "generate-docs.py")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: generator not found at {path}")
    spec = importlib.util.spec_from_file_location("_gendocs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _diff(label: str, expected: str, actual: str) -> bool:
    """True when they differ. Prints a bounded unified diff."""
    if expected == actual:
        return False
    print(f"  DRIFT: {label}")
    lines = list(difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=f"{label} (committed)",
        tofile=f"{label} (generated)",
        n=1,
    ))
    for line in lines[:40]:
        sys.stdout.write("    " + line if line.endswith("\n") else "    " + line + "\n")
    if len(lines) > 40:
        print(f"    ... {len(lines) - 40} more diff lines")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify generated docs match code")
    ap.add_argument("--fix", action="store_true", help="Rewrite the generated files")
    args = ap.parse_args()

    gen = _load_generator()

    version = gen._version()
    tools = gen._collect_tools()
    env_map, defaults, restart = gen._collect_config()
    gen._validate_descriptions(env_map)
    effective, conflicts = gen._scan_effective_defaults(
        env_map, defaults,
        roots=("mnemosyne", "hermes_memory_provider", "integrations"),
    )

    expected = {
        os.path.join("docs", "api", "tool-schema.mdx"): gen._render_tool_schema(tools, version),
        os.path.join("docs", "api", "configuration.mdx"): gen._render_config(
            env_map, defaults, restart, version, effective, conflicts),
    }

    mcp_count = sum(1 for t in tools if t["mcp"])
    print(f"Code reports: v{version}, {len(tools)} tools ({mcp_count} over MCP), "
          f"{len(env_map)} config keys, {len(effective)} effective-default divergences")
    print("Checking canonical docs/api/ against generated output...")

    drift = []
    for rel, want in expected.items():
        abs_path = os.path.join(REPO, rel)
        have = ""
        if os.path.isfile(abs_path):
            with open(abs_path, encoding="utf-8") as f:
                have = f.read()
        else:
            print(f"  MISSING: {rel}")
            drift.append(rel)
            continue
        if _diff(rel, want, have):
            drift.append(rel)

    if drift and args.fix:
        for rel, want in expected.items():
            abs_path = os.path.join(REPO, rel)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(want)
        print("")
        print("Rewrote: " + ", ".join(sorted(drift)))
        print("OK (--fix). Review and commit.")
        sys.exit(0)

    # Sibling docs repo: informational only, and only when present.
    sibling = os.path.normpath(os.path.join(REPO, "..", "mnemosyne-docs"))
    if os.path.isdir(sibling):
        sib_tool = os.path.join(sibling, "content", "api", "tool-schema.mdx")
        if os.path.isfile(sib_tool):
            with open(sib_tool, encoding="utf-8") as f:
                if f.read() != expected[os.path.join("docs", "api", "tool-schema.mdx")]:
                    print("  DRIFT (sibling): content/api/tool-schema.mdx")
                    drift.append("sibling:content/api/tool-schema.mdx")
        else:
            print("  note: sibling present but content/api/tool-schema.mdx missing")
    else:
        print("  note: ../mnemosyne-docs not checked out, skipping sibling check")

    print("")
    if drift:
        print("FAIL: generated docs are out of date.")
        print("Run: python3 scripts/generate-docs.py")
        sys.exit(1)

    print("OK: generated docs match the code.")
    sys.exit(0)


if __name__ == "__main__":
    main()
