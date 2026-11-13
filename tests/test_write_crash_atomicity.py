"""A hard-killed public remember must not leave a half-written dual row.

The pipe barrier pauses after the real BEAM write, before the legacy mirror;
SIGKILL prevents Python exception handling/connection cleanup from helping.
This is a process-crash test, not a power-loss or filesystem durability test.
"""

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMON = """
import json
import sys
from mnemosyne.core.memory import Mnemosyne

memory = Mnemosyne(session_id="crash-write", db_path=sys.argv[1])

def snapshot():
    # Include every column, not just counts, to protect the committed baseline.
    return json.loads(json.dumps({
        table: [list(row) for row in memory.conn.execute(
            "SELECT * FROM " + table + " ORDER BY id"
        )]
        for table in ("working_memory", "memories")
    }, default=lambda value: {"bytes": value.hex()}))
"""
_WRITER = _COMMON + """
baseline_id = memory.remember("Committed baseline: the observatory uses a blue notebook.")
assert baseline_id is not None
assert not memory.conn.in_transaction
before = snapshot()
assert all(len(rows) == 1 for rows in before.values())
original_remember = memory.beam.remember

def pause_after_beam(*args, **kwargs):
    pending_id = original_remember(*args, **kwargs)
    assert pending_id is not None and pending_id != baseline_id
    assert memory.conn.in_transaction
    assert memory.conn.execute(
        "SELECT content FROM working_memory WHERE id = ?", (pending_id,)
    ).fetchone()[0] == args[0]
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = ?", (pending_id,)
    ).fetchone()[0] == 0
    print(json.dumps({"before": before, "pending_id": pending_id}), flush=True)
    # Remain inside the public remember transaction until killed. No sleeps.
    sys.stdin.buffer.read(1)
    raise AssertionError("writer barrier unexpectedly released")

memory.beam.remember = pause_after_beam
memory.remember("In-flight observation: the telescope points toward Saturn.")
raise AssertionError("in-flight remember unexpectedly returned")
"""
_REOPEN = _COMMON + """
report = json.loads(sys.stdin.read())
assert snapshot() == report["before"], "committed baseline changed or partial write survived"
for table in ("working_memory", "memories"):
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM " + table + " WHERE id = ?", (report["pending_id"],)
    ).fetchone()[0] == 0
assert [row[0] for row in memory.conn.execute("PRAGMA integrity_check")] == ["ok"]
assert memory.conn.execute("PRAGMA foreign_key_check").fetchall() == []
assert not memory.conn.in_transaction
# Reopening must leave the actual public API writable, not merely readable.
new_id = memory.remember("After crash: a fresh observation was safely committed.")
assert new_id is not None
assert not memory.conn.in_transaction
import sqlite3
with sqlite3.connect(sys.argv[1]) as outside:
    for table in ("working_memory", "memories"):
        assert outside.execute(
            "SELECT COUNT(*) FROM " + table + " WHERE id = ?", (new_id,)
        ).fetchone()[0] == 1
memory.conn.close()
print("fresh-process reopen, baseline, rollback, integrity and subsequent write: ok")
"""


def _child_env(tmp_path):
    # No inherited deployment paths, credentials, model caches or Python hooks.
    env = {"PATH": os.defpath, "PYTHONPATH": str(_REPO_ROOT),
           "PYTHONNOUSERSITE": "1", "MNEMOSYNE_NO_EMBEDDINGS": "1"}
    for name, directory in {
        "HOME": "home", "HERMES_HOME": "hermes", "MNEMOSYNE_HOME": "mnemosyne",
        "MNEMOSYNE_DATA_DIR": "data", "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "xdg-data", "XDG_CACHE_HOME": "cache", "TMPDIR": "tmp",
        "MNEMOSYNE_FASTEMBED_CACHE_DIR": "fastembed",
        "MNEMOSYNE_MODEL_CACHE_DIR": "models", "HF_HOME": "hf",
        "HUGGINGFACE_HUB_CACHE": "hub", "TRANSFORMERS_CACHE": "transformers",
        "TORCH_HOME": "torch", "FLEET_ROOT": "fleet",
    }.items():
        path = tmp_path / directory
        path.mkdir()
        env[name] = str(path)
    return env


def _kill_at_barrier(tmp_path, script, *args):
    env = _child_env(tmp_path)
    db_path = str(tmp_path / "write-crash.db")
    with subprocess.Popen(
        [sys.executable, "-c", script, db_path, *args], cwd=_REPO_ROOT, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as writer:
        assert writer.stdin is not None
        assert writer.stdout is not None
        try:
            # Bound the entire message, not just the first readable byte.
            deadline = time.monotonic() + 30
            line = b""
            while not line.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                assert remaining > 0, "writer barrier message timed out"
                ready, _, _ = select.select([writer.stdout], [], [], remaining)
                assert ready, "writer did not reach the pipe barrier"
                chunk = os.read(writer.stdout.fileno(), 65536)
                if not chunk:
                    _, stderr = writer.communicate(timeout=5)
                    pytest.fail(f"writer exited before barrier: {stderr.decode()}")
                line += chunk
                assert len(line) <= 4 * 1024 * 1024, "oversized barrier message"
            report = json.loads(line)
            writer.kill()
            _, stderr = writer.communicate(timeout=10)
            assert writer.returncode == -signal.SIGKILL, stderr.decode()
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.communicate(timeout=10)
    return env, db_path, report


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and selectable pipes")
def test_killed_remember_preserves_committed_baseline_without_partial_write(tmp_path):
    env, db_path, report = _kill_at_barrier(tmp_path, _WRITER)
    reopened = subprocess.run(
        [sys.executable, "-c", _REOPEN, db_path], cwd=_REPO_ROOT, env=env,
        input=json.dumps(report), capture_output=True, text=True, timeout=30,
    )
    assert reopened.returncode == 0, reopened.stdout + reopened.stderr
    assert "subsequent write: ok" in reopened.stdout


_SCENARIO_COMMON = """
import json
import sqlite3
import sys
import mnemosyne.core.beam as beam_module
from mnemosyne.core.memory import Mnemosyne

operation, boundary, vector = sys.argv[2:]
vector = vector == "active"
tables = ("working_memory", "memories", "memory_embeddings")
if vector:
    tables += ("vec_working",)

def snapshot(conn):
    return json.loads(json.dumps({
        table: [list(row) for row in conn.execute(
            "SELECT " + ("rowid, embedding" if table == "vec_working" else "*") +
            " FROM " + table + " ORDER BY " +
            ("rowid" if table == "vec_working" else
             "memory_id" if table == "memory_embeddings" else "id")
        )] for table in tables
    }, default=lambda value: {"bytes": value.hex()}))

def configure_vectors(memory):
    if not vector:
        return
    import numpy as np
    import sqlite_vec
    assert beam_module._wm_vec_available(memory.conn), "real vec_working unavailable"
    assert memory.conn.execute("SELECT vec_version()").fetchone()[0]
    # Deterministic synthetic provider only; production serialization, SQL and
    # sqlite-vec virtual-table code remain untouched. Never load a model.
    def embed(contents):
        result = []
        for content in contents:
            vec = np.zeros(beam_module.EMBEDDING_DIM, dtype=np.float32)
            vec[sum(content.encode()) % len(vec)] = 1.0
            result.append(vec)
        return result
    beam_module._embeddings.available = lambda: True
    beam_module._embeddings.embed = embed
"""

_SCENARIO_WRITER = _SCENARIO_COMMON + """
memory = Mnemosyne(session_id="crash-expanded", db_path=sys.argv[1])
configure_vectors(memory)
ids = [memory.remember(text) for text in (
    "Committed observatory baseline: a blue notebook records observations.",
    "Committed telescope baseline: the brass lens belongs in the cabinet.",
    "Committed chart baseline: the star atlas belongs on the wooden shelf.",
)]
assert all(ids) and len(set(ids)) == 3
if operation.startswith("legacy-"):
    memory.conn.execute("DELETE FROM working_memory WHERE id = ?", (ids[0],))
    memory.conn.commit()
assert not memory.conn.in_transaction
before = snapshot(memory.conn)
rowids = dict(memory.conn.execute("SELECT id, rowid FROM working_memory"))
if vector:
    assert len(before["vec_working"]) == 3
    assert len(before["memory_embeddings"]) == 3
pending = []
changed_content = "Updated observatory observation: the green notebook records Jupiter."

def content(table, memory_id):
    row = memory.conn.execute(
        "SELECT content FROM " + table + " WHERE id = ?", (memory_id,)
    ).fetchone()
    return None if row is None else row[0]

def barrier():
    assert memory.conn.in_transaction
    after = snapshot(memory.conn)
    assert after != before
    if operation == "remember":
        pending[:] = [row[0] for row in memory.conn.execute(
            "SELECT id FROM working_memory WHERE id NOT IN (?, ?, ?)", ids
        )]
        assert len(pending) == 1
        assert content("working_memory", pending[0]) == content("memories", pending[0])
    elif operation == "batch":
        pending[:] = [row[0] for row in memory.conn.execute(
            "SELECT id FROM working_memory WHERE id NOT IN (?, ?, ?)", ids
        )]
        assert len(pending) == 1
        # Batch dispatches BEAM, not Mnemosyne: no invented legacy symmetry.
        assert content("memories", pending[0]) is None
        assert content("working_memory", ids[0]) == changed_content
        assert content("memories", ids[0]) != changed_content
        assert content("working_memory", ids[1]) is None
        assert content("memories", ids[1]) is None
    elif operation.endswith("update"):
        assert content("memories", ids[0]) == changed_content
        if operation == "legacy-update":
            assert content("working_memory", ids[0]) is None
        elif boundary == "legacy":
            assert content("working_memory", ids[0]) != changed_content
        else:
            assert content("working_memory", ids[0]) == changed_content
    else:
        assert content("memories", ids[0]) is None
        assert (content("working_memory", ids[0]) is not None) == (boundary == "legacy")
    if vector:
        # Prove each operation's real virtual-table mutation, not merely that
        # some vector changed somewhere in the multi-operation batch.
        assert after["vec_working"] != before["vec_working"]
        assert after["memory_embeddings"] != before["memory_embeddings"]
        old_vectors = {row[0]: row[1] for row in before["vec_working"]}
        for target_id in pending:
            rowid = memory.conn.execute(
                "SELECT rowid FROM working_memory WHERE id = ?", (target_id,)
            ).fetchone()[0]
            assert memory.conn.execute(
                "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)
            ).fetchone()[0] == 1
        if operation in ("update", "batch"):
            rowid = memory.conn.execute(
                "SELECT rowid FROM working_memory WHERE id = ?", (ids[0],)
            ).fetchone()[0]
            changed = memory.conn.execute(
                "SELECT embedding FROM vec_working WHERE rowid = ?", (rowid,)
            ).fetchone()[0]
            assert {"bytes": changed.hex()} != old_vectors[rowid]
        if operation in ("forget", "batch"):
            removed_id = ids[1] if operation == "batch" else ids[0]
            assert memory.conn.execute(
                "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowids[removed_id],)
            ).fetchone()[0] == 0
            assert memory.conn.execute(
                "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (removed_id,)
            ).fetchone()[0] == 0
            assert len(after["vec_working"]) == (3 if operation == "batch" else 2)
    print(json.dumps({"before": before, "pending": pending}), flush=True)
    sys.stdin.buffer.read(1)
    raise AssertionError("writer barrier unexpectedly released")

if boundary == "legacy":
    target = "update_working" if operation == "update" else "forget_working"
    setattr(memory.beam, target, lambda *args, **kwargs: barrier())
else:
    # Installation is after all baseline commits. This seam is reached only
    # when the owning public transaction has completed its actual mutations.
    beam_module._BeamConnection._real_commit = lambda self: barrier()

if operation == "remember":
    memory.remember("Pending observatory observation: Saturn has bright rings.")
elif operation.endswith("update"):
    memory.update(ids[0], content=changed_content)
elif operation.endswith("forget"):
    memory.forget(ids[0])
else:
    from mnemosyne.batch_tool import apply_beam_batch, validate_batch_operations
    apply_beam_batch(memory.beam, validate_batch_operations([
        {"action": "remember", "content": "Pending batch observation: Mars appears red."},
        {"action": "update", "memory_id": ids[0], "content": changed_content},
        {"action": "forget", "memory_id": ids[1]},
    ]))
raise AssertionError("mutation unexpectedly returned before kill")
"""

_SCENARIO_REOPEN = _SCENARIO_COMMON + """
report = json.loads(sys.stdin.read())
# Inspect durable state BEFORE application initialization/backfill could repair
# derived rows. This is a newly exec'd interpreter and independent connection.
with sqlite3.connect(sys.argv[1]) as raw:
    if vector:
        import sqlite_vec
        raw.enable_load_extension(True)
        sqlite_vec.load(raw)
        raw.enable_load_extension(False)
    assert snapshot(raw) == report["before"], "crash changed committed baseline"
    assert [row[0] for row in raw.execute("PRAGMA integrity_check")] == ["ok"]
    assert raw.execute("PRAGMA foreign_key_check").fetchall() == []
memory = Mnemosyne(session_id="crash-expanded", db_path=sys.argv[1])
configure_vectors(memory)
assert snapshot(memory.conn) == report["before"], "reopen changed baseline"
assert not memory.conn.in_transaction
new_id = memory.remember("After crash: the observatory safely recorded Venus.")
assert new_id is not None and new_id not in report["pending"]
assert not memory.conn.in_transaction
with sqlite3.connect(sys.argv[1]) as outside:
    if vector:
        outside.enable_load_extension(True)
        sqlite_vec.load(outside)
        outside.enable_load_extension(False)
    for table in ("working_memory", "memories"):
        assert outside.execute(
            "SELECT COUNT(*) FROM " + table + " WHERE id = ?", (new_id,)
        ).fetchone()[0] == 1
    if vector:
        rowid = outside.execute(
            "SELECT rowid FROM working_memory WHERE id = ?", (new_id,)
        ).fetchone()[0]
        assert outside.execute(
            "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)
        ).fetchone()[0] == 1
        assert outside.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (new_id,)
        ).fetchone()[0] == 1
memory.conn.close()
print("exact baseline, integrity and subsequent write: ok")
"""


def _assert_scenario(tmp_path, operation, boundary, vector):
    args = (operation, boundary, "active" if vector else "disabled")
    env, db_path, report = _kill_at_barrier(tmp_path, _SCENARIO_WRITER, *args)
    reopened = subprocess.run(
        [sys.executable, "-c", _SCENARIO_REOPEN, db_path, *args],
        cwd=_REPO_ROOT, env=env, input=json.dumps(report),
        capture_output=True, text=True, timeout=30,
    )
    assert reopened.returncode == 0, reopened.stdout + reopened.stderr
    assert "subsequent write: ok" in reopened.stdout


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and selectable pipes")
@pytest.mark.parametrize("operation,boundary", [
    ("remember", "precommit"), ("batch", "precommit"),
    ("update", "legacy"), ("update", "precommit"),
    ("forget", "legacy"), ("forget", "precommit"),
    ("legacy-update", "precommit"), ("legacy-forget", "precommit"),
])
def test_killed_mutation_restores_exact_baseline(tmp_path, operation, boundary):
    _assert_scenario(tmp_path, operation, boundary, vector=False)


_VECTOR_PREREQUISITES = """
import sqlite3
import sys

for package in ("numpy", "sqlite_vec"):
    try:
        __import__(package)
    except ModuleNotFoundError as exc:
        if exc.name != package:
            raise
        print("missing optional package: " + package)
        sys.exit(77)
import sqlite_vec

with sqlite3.connect(":memory:") as conn:
    if not hasattr(conn, "enable_load_extension"):
        print("SQLite extension loading is unavailable")
        sys.exit(77)
    try:
        conn.enable_load_extension(True)
    except sqlite3.NotSupportedError:
        print("SQLite extension loading is unsupported")
        sys.exit(77)
    except sqlite3.OperationalError as exc:
        if str(exc) != "not authorized":
            raise
        print("SQLite extension loading is not authorized")
        sys.exit(77)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    assert conn.execute("SELECT vec_version()").fetchone()[0]
    conn.execute("CREATE VIRTUAL TABLE probe USING vec0(embedding float[3])")
"""


def _require_vector_prerequisites(tmp_path):
    # Separate directories avoid _child_env's deliberately non-idempotent mkdir.
    # Use the writer's exact environment policy, not the parent's user-site.
    probe_path = tmp_path / "vector-prerequisites"
    probe_path.mkdir()
    probe = subprocess.run(
        [sys.executable, "-c", _VECTOR_PREREQUISITES], cwd=_REPO_ROOT,
        env=_child_env(probe_path), capture_output=True, text=True, timeout=30,
    )
    if probe.returncode == 77:
        pytest.skip(probe.stdout.strip())
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL and selectable pipes")
@pytest.mark.parametrize("operation", ["remember", "update", "forget", "batch"])
def test_killed_active_vector_mutation_restores_exact_baseline(tmp_path, operation):
    _require_vector_prerequisites(tmp_path)
    _assert_scenario(tmp_path, operation, "precommit", vector=True)
