#!/usr/bin/env python3
"""Benchmark audit-log overhead on memory mutations.

The audit log writes one row to memory_events per mutation (remember,
forget, invalidate, shared_remember, shared_forget). This script measures
the per-call latency cost so reviewers can see exactly what the overhead is.

Usage:
    python tools/bench_audit_log.py [--ops N]
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


def _measure_remember(provider, n: int) -> list[float]:
    durations: list[float] = []
    # Warmup
    for i in range(5):
        provider.handle_tool_call("mnemosyne_remember", {
            "content": f"warmup {i}", "importance": 0.5, "source": "fact",
        })
    for i in range(n):
        t0 = time.perf_counter()
        provider.handle_tool_call("mnemosyne_remember", {
            "content": f"benchmark memory row {i} for timing test",
            "importance": 0.5,
            "source": "fact",
        })
        durations.append((time.perf_counter() - t0) * 1000.0)
    return durations


def _toggle_audit(provider, enabled: bool) -> None:
    """Flip the audit log on/off without touching anything else.

    The provider exposes ``_audit``; setting it to None disables the per-event
    INSERT in ``_audit_event``. This isolates audit-log overhead from the rest
    of the remember() pipeline.
    """
    if enabled:
        if not getattr(provider, "_audit", None):
            provider._init_audit_log()
    else:
        provider._audit = None


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
    args = ap.parse_args()

    # Use ONE provider, toggle audit on/off, and interleave measurements
    # to cancel out cold-start / disk-cache asymmetry between configs.
    with tempfile.TemporaryDirectory() as tmp:
        provider = _bootstrap(Path(tmp))

        # Big warmup: pay all import / lazy-init / fts5 trigger costs upfront
        # and hit the SQLite page cache so neither config is "first".
        for i in range(50):
            provider.handle_tool_call("mnemosyne_remember", {
                "content": f"global warmup row {i} for benchmark",
                "importance": 0.5,
                "source": "fact",
            })

        on_samples: list[float] = []
        off_samples: list[float] = []
        # Alternate samples in chunks of 10 so each config sees equal access
        # patterns from disk and FTS5 trigger fanout.
        chunk = 10
        for batch in range(args.ops // chunk):
            _toggle_audit(provider, True)
            on_samples.extend(_measure_remember(provider, chunk))
            _toggle_audit(provider, False)
            off_samples.extend(_measure_remember(provider, chunk))

        results = [
            _summary("audit_off", off_samples),
            _summary("audit_on", on_samples),
        ]

    base_p50 = results[0]["p50_ms"]
    on_p50 = results[1]["p50_ms"]
    delta_ms = on_p50 - base_p50
    delta_pct = (delta_ms / base_p50 * 100.0) if base_p50 else 0.0

    report = {
        "ops_per_config": len(off_samples),
        "results": results,
        "delta_p50_ms": round(delta_ms, 3),
        "delta_p50_pct": round(delta_pct, 2),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
