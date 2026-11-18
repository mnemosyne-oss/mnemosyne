"""
Regression test for mnemosyne-oss/mnemosyne#1147.

Two processes write to the same Mnemosyne database in lockstep.
Pre-fix, the deferred-transaction snapshot upgrade returns SQLITE_BUSY
immediately and the write is dropped. Post-fix, the write path takes
the writer lock with BEGIN IMMEDIATE and all writes land.
"""
from __future__ import annotations

import multiprocessing
import os
import sqlite3
import time
import uuid
from pathlib import Path
from queue import Empty
from typing import List

import pytest

# Each worker writes this many probes; the store must end up with 2x that.
_WRITES_PER_WORKER = 8


def _remember_worker(
    db_dir: str,
    name: str,
    start: float,
    result_queue: "multiprocessing.Queue",
) -> None:
    """Child-process worker: write one probe per second for 8 seconds."""
    os.environ["MNEMOSYNE_DATA_DIR"] = db_dir
    os.environ["MNEMOSYNE_CHANNEL_ID"] = name
    os.environ["MNEMOSYNE_NO_EMBEDDINGS"] = "1"

    # Import here so the child sees the env vars before any module init.
    from mnemosyne.mcp_tools import handle_tool_call

    # Slight stagger so the two processes do not hit the exact same millisecond.
    if name == "B":
        time.sleep(0.5)

    # Warmup creates the per-process connection and schema if needed.
    try:
        handle_tool_call(
            "mnemosyne_remember",
            {"content": f"warmup {name}", "importance": 0.1},
        )
    except Exception as exc:
        result_queue.put((name, [f"warmup-exc:{exc}"]))
        return

    statuses: List[str] = []
    for i in range(_WRITES_PER_WORKER):
        while time.time() < start + i:
            time.sleep(0.001)
        try:
            handle_tool_call(
                "mnemosyne_remember",
                {
                    "content": f"probe {name} {i} {uuid.uuid4().hex[:6]}",
                    "importance": 0.1,
                },
            )
            statuses.append("ok")
        except Exception as exc:
            statuses.append(f"EXC:{exc}")

    result_queue.put((name, statuses))


@pytest.fixture
def db_dir(tmp_path: Path) -> str:
    """Create a shared DB, seeded once in the parent process."""
    d = str(tmp_path / "concurrency")
    os.makedirs(d, exist_ok=True)

    old_data = os.environ.get("MNEMOSYNE_DATA_DIR")
    old_no_emb = os.environ.get("MNEMOSYNE_NO_EMBEDDINGS")
    try:
        os.environ["MNEMOSYNE_DATA_DIR"] = d
        os.environ["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
        from mnemosyne.mcp_tools import handle_tool_call

        handle_tool_call(
            "mnemosyne_remember",
            {"content": "seed", "importance": 0.1},
        )
    finally:
        if old_data is None:
            os.environ.pop("MNEMOSYNE_DATA_DIR", None)
        else:
            os.environ["MNEMOSYNE_DATA_DIR"] = old_data
        if old_no_emb is None:
            os.environ.pop("MNEMOSYNE_NO_EMBEDDINGS", None)
        else:
            os.environ["MNEMOSYNE_NO_EMBEDDINGS"] = old_no_emb

    return d


def test_remember_two_processes_lockstep_no_lost_writes(db_dir: str) -> None:
    """Pre-fix this test loses writes to SQLITE_BUSY; post-fix all 16 land."""
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    start = time.time() + 0.5

    proc_a = ctx.Process(target=_remember_worker, args=(db_dir, "A", start, queue))
    proc_b = ctx.Process(target=_remember_worker, args=(db_dir, "B", start, queue))
    proc_a.start()
    proc_b.start()
    for proc in (proc_a, proc_b):
        proc.join(timeout=60)

    # Reap every live worker before failing: pytest.fail raises out of the
    # test, so a worker left alive here would outlive the run.
    hung = [proc for proc in (proc_a, proc_b) if proc.is_alive()]
    for proc in hung:
        proc.terminate()
    for proc in hung:
        proc.join(timeout=10)
    if hung:
        pytest.fail(
            "workers hung and had to be terminated: "
            + ", ".join(proc.name for proc in hung)
        )

    for proc in (proc_a, proc_b):
        assert proc.exitcode == 0, (
            f"worker {proc.name} crashed with exitcode {proc.exitcode} "
            "before reporting its writes"
        )

    # Read exactly one report per worker: queue.empty() is unreliable while a
    # message is still in transit, and a crashed worker never puts one.
    results: dict[str, List[str]] = {}
    for _ in range(2):
        try:
            name, statuses = queue.get(timeout=10)
        except Empty:
            pytest.fail(
                f"only {len(results)} of 2 workers reported: {sorted(results)}"
            )
        results[name] = statuses

    for name in ("A", "B"):
        statuses = results.get(name, [])
        assert len(statuses) == _WRITES_PER_WORKER, (
            f"process {name} reported {len(statuses)} of "
            f"{_WRITES_PER_WORKER} writes: {statuses[:3]}"
        )
        failures = [s for s in statuses if not s.startswith("ok")]
        assert not failures, (
            f"process {name} raised under contention: {failures[:3]}"
        )

    db_path = Path(db_dir) / "mnemosyne.db"
    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE content LIKE 'probe %'"
        ).fetchone()[0]
    finally:
        conn.close()

    expected = 2 * _WRITES_PER_WORKER
    assert count == expected, (
        f"expected {expected} committed probe rows, got {count} -- writes lost"
    )
