#!/usr/bin/env python3
"""Benchmark mnemosyne_validate latency.

Each `validate` call updates the live row, inserts to memory_validations,
and the trim trigger may delete an older row. This script measures the
per-call cost so reviewers can confirm the ring-buffer trigger doesn't
impose surprise overhead at scale.

Usage:
    python tools/bench_validation.py [--ops N]
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


def _seed_targets(provider, n: int) -> list[str]:
    """Pre-create N memories that will be repeatedly validated."""
    ids: list[str] = []
    for i in range(n):
        result = json.loads(provider.handle_tool_call("mnemosyne_remember", {
            "content": f"target memory {i} for validation benchmark",
            "importance": 0.5,
            "source": "fact",
        }))
        ids.append(result["memory_id"])
    return ids


def _measure(provider, target_ids: list[str], action: str, iters: int) -> list[float]:
    durations: list[float] = []
    n_targets = len(target_ids)

    def _args(i: int) -> dict:
        out = {
            "memory_id": target_ids[i % n_targets],
            "action": action,
            "validator": f"agent_{i % 4}",
        }
        # `update` requires new_content; everything else doesn't take it.
        if action == "update":
            out["new_content"] = f"updated content rev-{i} for benchmark"
        return out

    # Warmup
    for i in range(min(20, iters)):
        provider.handle_tool_call("mnemosyne_validate", _args(i))
    # Measure
    for i in range(iters):
        args = _args(i)
        t0 = time.perf_counter()
        provider.handle_tool_call("mnemosyne_validate", args)
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
    ap.add_argument("--ops", type=int, default=200)
    ap.add_argument("--targets", type=int, default=50)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        provider = _bootstrap(Path(tmp))
        target_ids = _seed_targets(provider, args.targets)

        # Compare: remember (baseline) vs each validate action.
        # remember is the reference cost; validate should be in the same ballpark.
        remember_samples: list[float] = []
        for i in range(args.ops):
            t0 = time.perf_counter()
            provider.handle_tool_call("mnemosyne_remember", {
                "content": f"remember-bench {i}",
                "importance": 0.5,
                "source": "fact",
            })
            remember_samples.append((time.perf_counter() - t0) * 1000.0)

        results = [_summary("remember_baseline", remember_samples)]
        for action in ("attest", "update", "invalidate"):
            samples = _measure(provider, target_ids, action, args.ops)
            results.append(_summary(f"validate_{action}", samples))

    print(json.dumps({
        "ops_per_config": args.ops,
        "targets": args.targets,
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
