#!/usr/bin/env python3
"""Advisory, offline warm linear BEAM recall benchmark (issue #901).

Run from a checkout; see docs/benchmarking.md for the workload contract.
The controller starts a fresh worker so import-time config cannot leak in.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = tuple(f"zephyr{i} calibration" for i in range(8))
SETTINGS = {
    "MNEMOSYNE_POLYPHONIC_RECALL": "0",
    "MNEMOSYNE_NO_EMBEDDINGS": "1",
    "MNEMOSYNE_LLM_ENABLED": "false",
    "MNEMOSYNE_HOST_LLM_ENABLED": "0",
    "MNEMOSYNE_RECALL_PROVENANCE": "0",
}
RECALL_KWARGS = {"temporal_weight": 0.0, "_cross_session": False}


def positive_int(value: str) -> int:
    """Parse a count or timeout, rejecting values that cannot produce a run."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    """Build the controller CLI and its internal worker-mode arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine", required=True, help="Stable local machine label (no secrets)")
    ap.add_argument("--rows-per-tier", type=positive_int, default=1000)
    ap.add_argument("--top-k", type=positive_int, default=10)
    ap.add_argument("--warmup-cycles", type=positive_int, default=3)
    ap.add_argument("--cycles-per-round", type=positive_int, default=10)
    ap.add_argument("--rounds", type=positive_int, default=5)
    ap.add_argument("--timeout", type=positive_int, default=600,
                    help="Worker time limit in seconds, including setup (default: 600)")
    ap.add_argument("--output", type=Path, help="Save JSON (stdout always contains the report)")
    ap.add_argument("--baseline", type=Path, help="Compare a compatible JSON report; no timing gate")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return ap


def summarize(rounds: list[list[int]], elapsed_ns: list[int]) -> dict:
    """Nearest-rank latency quantiles and sequential, wall-clock round QPS."""
    durations = sorted(ns / 1_000_000 for batch in rounds for ns in batch)
    qps = [len(batch) * 1_000_000_000 / elapsed for batch, elapsed in zip(rounds, elapsed_ns)]
    return {
        "calls": len(durations),
        "latency_ms": {
            "mean": statistics.mean(durations),
            "p50": durations[math.ceil(len(durations) * 0.5) - 1],
            "p95": durations[math.ceil(len(durations) * 0.95) - 1],
            "min": durations[0],
            "max": durations[-1],
        },
        "round_qps": qps,
        "median_qps": statistics.median(qps),
    }


def environment(machine: str) -> dict:
    """Record comparison metadata without copying environment variables or keys."""
    packages = {}
    for name in ("numpy", "sqlite-vec", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "machine_label": machine,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "sqlite": sqlite3.sqlite_version,
        "packages": packages,
    }


def revision() -> dict:
    """Read the checkout revision and dirty status, tolerating unavailable Git."""
    def git(*args):
        """Run a bounded metadata-only Git command in the source checkout."""
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=True, timeout=10,
        )
        return result.stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def seed(beam, rows: int) -> None:
    """Direct SQL is setup only: fixed IDs/content/timestamps, real FTS triggers."""
    for table in ("working_memory", "episodic_memory"):
        beam.conn.executemany(
            f"INSERT INTO {table} "
            "(id, content, source, timestamp, session_id, importance, veracity) "
            "VALUES (?, ?, 'benchmark', '2026-01-01T12:00:00', 'throughput', ?, 'stated')",
            (
                (f"{table}-{i:08d}",
                 f"zephyr{i % 8} calibration ledger item {i:08d} "
                 f"pressure valve inspection record for {table}",
                 0.4 + (i % 5) * 0.1)
                for i in range(rows)
            ),
        )
    beam.conn.commit()


def measure(beam, args) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Warm recall, then collect per-call samples and whole-round wall times."""
    def cycle(timings, counts):
        """Recall each fixed query once and reject empty-result measurements."""
        for query in QUERIES:
            start = time.perf_counter_ns()
            results = beam.recall(query, top_k=args.top_k, **RECALL_KWARGS)
            elapsed = time.perf_counter_ns() - start
            if not results:
                raise RuntimeError(f"Empty recall for {query!r}; refusing a misleading timing report")
            timings.append(elapsed)
            counts.append(len(results))

    for _ in range(args.warmup_cycles):
        cycle([], [])
    rounds, result_counts, elapsed_ns = [], [], []
    for _ in range(args.rounds):
        timings, counts = [], []
        start = time.perf_counter_ns()
        for _ in range(args.cycles_per_round):
            cycle(timings, counts)
        elapsed_ns.append(time.perf_counter_ns() - start)
        rounds.append(timings)
        result_counts.append(counts)
    return rounds, result_counts, elapsed_ns


def worker(args) -> dict:
    """Measure an isolated temporary corpus and return its auditable JSON report."""
    # Imports happen only after the controller has set a clean environment.
    if any(os.environ.get(key) != value for key, value in SETTINGS.items()):
        raise RuntimeError("Run the controller without --worker to isolate configuration")
    sys.path.insert(0, str(ROOT))
    from mnemosyne.core.beam import BeamMemory, close_thread_connection, _SQLITE_VEC_AVAILABLE

    db_path = Path(os.environ["MNEMOSYNE_DATA_DIR"]) / "throughput.db"
    beam = BeamMemory(session_id="throughput", db_path=db_path)
    try:
        seed(beam, args.rows_per_tier)
        probe = beam.recall(QUERIES[0], top_k=args.top_k, explain=True, **RECALL_KWARGS)
        if probe["engine"] != "linear" or not probe["results"]:
            raise RuntimeError("Expected nonempty linear recall before measurement")
        samples, counts, elapsed_ns = measure(beam, args)
        return {
            "schema_version": 1,
            "benchmark": "beam-warm-linear-v1",
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "revision": revision(),
            "environment": environment(args.machine),
            "settings": {**SETTINGS, "PYTHONHASHSEED": "0"},
            "validation": {
                "engine": probe["engine"],
                "sqlite_vec_available": _SQLITE_VEC_AVAILABLE,
                "seeded_rows": {
                    table: beam.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("working_memory", "episodic_memory")
                },
            },
            "workload": {
                "rows_per_tier": args.rows_per_tier,
                "tiers": ["working_memory", "episodic_memory"],
                "queries": list(QUERIES),
                "top_k": args.top_k,
                "recall_kwargs": RECALL_KWARGS,
                "warmup_cycles": args.warmup_cycles,
                "cycles_per_round": args.cycles_per_round,
                "rounds": args.rounds,
                "tracking": "default recall counters and diagnostics; provenance disabled",
            },
            "sqlite_pragmas": {
                key: beam.conn.execute(f"PRAGMA {key}").fetchone()[0]
                for key in ("journal_mode", "synchronous", "page_size", "cache_size")
            },
            "summary": summarize(samples, elapsed_ns),
            "samples_ns": samples,
            "round_elapsed_ns": elapsed_ns,
            "result_counts": counts,
        }
    finally:
        close_thread_connection(db_path, allow_main_thread=True)


def compare(report: dict, baseline: dict) -> dict:
    """Compute advisory deltas only when workload and environment metadata match."""
    # Commit/time intentionally differ. Host load/power state still need human control.
    keys = (
        "schema_version", "benchmark", "harness_sha256", "environment", "settings", "workload",
        "sqlite_pragmas", "validation",
    )
    different = [key for key in keys if key not in baseline or report[key] != baseline[key]]
    if different:
        return {"compatible": False, "mismatched_fields": different}
    old, new = baseline["summary"], report["summary"]
    return {
        "compatible": True,
        "baseline_revision": baseline["revision"],
        "median_qps_change_pct": (new["median_qps"] / old["median_qps"] - 1) * 100,
        "p50_latency_change_pct": (new["latency_ms"]["p50"] / old["latency_ms"]["p50"] - 1) * 100,
        "p95_latency_change_pct": (new["latency_ms"]["p95"] / old["latency_ms"]["p95"] - 1) * 100,
    }


def load_baseline(path: Path) -> dict:
    """Reject unreadable reports/invalid divisors before doing any measured work."""
    baseline = json.loads(path.read_text(encoding="utf-8"))
    try:
        if not isinstance(baseline["revision"], dict):
            raise ValueError("revision must be an object")
        summary = baseline["summary"]
        for value in (summary["median_qps"], summary["latency_ms"]["p50"],
                      summary["latency_ms"]["p95"]):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("comparison metrics must be finite positive numbers")
    except (KeyError, TypeError) as exc:
        raise ValueError("expected a benchmark report with revision and summary metrics") from exc
    return baseline


def main(argv=None) -> int:
    """Run the bounded worker and publish a report only after successful execution."""
    ap = parser()
    args = ap.parse_args(argv)
    if args.worker:
        # Keep the JSON channel usable even if an optional backend prints at import.
        with contextlib.redirect_stdout(sys.stderr):
            report = worker(args)
        print(json.dumps(report, allow_nan=False))
        return 0
    baseline = None
    if args.baseline:
        try:
            baseline = load_baseline(args.baseline)
        except (OSError, UnicodeError, ValueError) as exc:
            ap.error(f"cannot load --baseline: {exc}")
    with tempfile.TemporaryDirectory(prefix="mnemosyne-throughput-") as tmp:
        env = {key: value for key, value in os.environ.items() if not key.startswith("MNEMOSYNE_")}
        env.update(SETTINGS)
        env.update(MNEMOSYNE_DATA_DIR=tmp, PYTHONHASHSEED="0", PYTHONUTF8="1",
                   PYTHONIOENCODING="utf-8")
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", "--machine", args.machine,
            "--rows-per-tier", str(args.rows_per_tier), "--top-k", str(args.top_k),
            "--warmup-cycles", str(args.warmup_cycles),
            "--cycles-per-round", str(args.cycles_per_round), "--rounds", str(args.rounds),
        ]
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True,
                                    encoding="utf-8", timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"Benchmark worker timed out after {args.timeout} seconds; "
                  "increase --timeout for larger workloads.", file=sys.stderr)
            return 1
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode:
            return result.returncode
        report = json.loads(result.stdout)
    if baseline is not None:
        report["comparison"] = compare(report, baseline)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
