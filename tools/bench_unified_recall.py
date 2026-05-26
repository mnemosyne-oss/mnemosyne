#!/usr/bin/env python3
"""Benchmark unified recall (PR: shared_surface_read flag).

Compares latency of `mnemosyne_recall` with shared_surface_read=False (current
behavior) vs True (new merged-bank behavior). Uses an isolated tmp DB so it
runs deterministically regardless of host environment.

Usage:
    python tools/bench_unified_recall.py [--memories N] [--queries Q]

Reports p50/p95 latency for both configs and the delta. The merged-bank path
adds a single shared-surface beam.recall() call plus a Python sort over the
combined list, so the expected overhead is small and bounded.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path


def _bootstrap(env_dir: Path):
    """Init a provider with private + shared surface set up."""
    os.environ["MNEMOSYNE_DATA_DIR"] = str(env_dir / "private")
    os.environ["MNEMOSYNE_HOST_LLM_ENABLED"] = "0"
    from hermes_memory_provider import MnemosyneMemoryProvider

    hermes_home = env_dir / "profile"
    hermes_home.mkdir(parents=True)

    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id="bench-session",
        hermes_home=str(hermes_home),
        agent_identity="Bench",
        shared_surface_path=str(env_dir / "shared" / "mnemosyne.db"),
    )
    return provider


def _seed(provider, n_private: int, n_surface: int) -> None:
    """Populate both banks with N rows of synthetic content."""
    topics = [
        "Tailscale VPN setup", "Postgres tuning notes", "Docker Compose workflow",
        "GitHub PR workflow", "Rust async runtime", "Python type hints",
        "FTS5 query syntax", "SQLite vacuum", "Linux file permissions",
        "OpenAI API rate limits", "ML training pipeline", "Multilingual NLP",
        "Notion API quirks", "Zsh prompt config", "Tmux key bindings",
    ]
    for i in range(n_private):
        topic = topics[i % len(topics)]
        provider.handle_tool_call("mnemosyne_remember", {
            "content": f"Private memory #{i}: detail about {topic}",
            "importance": 0.5,
            "source": "fact",
        })
    for i in range(n_surface):
        topic = topics[i % len(topics)]
        provider.handle_tool_call("mnemosyne_shared_remember", {
            "content": f"Shared note #{i}: user prefers {topic}",
            "kind": "preference",
            "importance": 0.6,
        })


def _measure(provider, query: str, *, surface_read: bool, iters: int) -> list[float]:
    provider._shared_surface_read = surface_read
    durations: list[float] = []
    # Warmup
    for _ in range(5):
        provider.handle_tool_call("mnemosyne_recall", {"query": query, "limit": 5})
    for _ in range(iters):
        t0 = time.perf_counter()
        provider.handle_tool_call("mnemosyne_recall", {"query": query, "limit": 5})
        durations.append((time.perf_counter() - t0) * 1000.0)
    return durations


def _summary(label: str, samples: list[float]) -> dict:
    samples_sorted = sorted(samples)
    p50 = samples_sorted[len(samples_sorted) // 2]
    p95 = samples_sorted[max(0, int(len(samples_sorted) * 0.95) - 1)]
    return {
        "config": label,
        "n": len(samples),
        "mean_ms": round(statistics.mean(samples), 3),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=int, default=200)
    ap.add_argument("--surface", type=int, default=50)
    ap.add_argument("--queries", type=int, default=100)
    args = ap.parse_args()

    queries = [
        "VPN preference Tailscale",
        "database tuning Postgres",
        "Rust async runtime",
        "FTS5 query syntax",
        "Docker Compose workflow",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp)
        provider = _bootstrap(env_dir)
        print(f"Seeding {args.private} private + {args.surface} shared rows...")
        _seed(provider, args.private, args.surface)

        results: list[dict] = []
        for label, surface_read in (("baseline_private_only", False),
                                     ("merged_with_surface", True)):
            print(f"Benchmarking {label}...")
            samples: list[float] = []
            iters_per_query = max(1, args.queries // len(queries))
            for q in queries:
                samples.extend(_measure(
                    provider, q,
                    surface_read=surface_read,
                    iters=iters_per_query,
                ))
            results.append(_summary(label, samples))

        baseline_p50 = results[0]["p50_ms"]
        merged_p50 = results[1]["p50_ms"]
        delta_ms = merged_p50 - baseline_p50
        delta_pct = (delta_ms / baseline_p50 * 100.0) if baseline_p50 else 0.0

        report = {
            "private_rows": args.private,
            "surface_rows": args.surface,
            "queries": args.queries,
            "results": results,
            "delta_p50_ms": round(delta_ms, 3),
            "delta_p50_pct": round(delta_pct, 2),
        }
        print()
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
