"""#1107: bounded CLI batches, early validation, and real API/SQLite mapping.

Offline capsule: 65 working + 19 episodic source rows, an HTTP transport with
at most 10 inputs/request. Default 64 fails atomically; --batch-size 8 succeeds.
Only HTTP transport is replaced: dispatch, payloads and persistence are real.
"""

import io
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error

import pytest

from mnemosyne import cli
from mnemosyne.core import beam, embeddings


@pytest.fixture(autouse=True)
def isolated_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setattr(cli, "DATA_DIR", str(tmp_path / "data"))


@pytest.mark.parametrize("option, expected", [([], 64), (["--batch-size", "8"], 8)])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_forwards_batch_size(tmp_path, monkeypatch, option, expected, dry_run):
    db = tmp_path / "target.db"
    conn = sqlite3.connect(db)
    conn.close()
    calls = []

    def record(conn, *, batch_size=None, dry_run=False, progress=None):
        calls.append((batch_size, dry_run, callable(progress)))
        return {"model": "offline", "dim": embeddings.EMBEDDING_DIM}

    # Keep target resolution and BeamMemory initialization real.
    monkeypatch.setattr(beam, "reindex_vectors", record)
    cli.cmd_reindex(
        [
            "--db",
            str(db),
            "--yes",
            "--no-backup",
            *option,
            *(["--dry-run"] if dry_run else []),
        ]
    )
    assert calls == [(expected, dry_run, not dry_run)]


_INVALID_CLI = [
    (["--batch-size"], "requires a value"),
    (["--batch-size", "--yes"], "requires a value"),
    (["--batch-size", "oops"], "must be an integer"),
    (["--batch-size", "1.5"], "must be an integer"),
    (["--batch-size", "0"], "must be a positive integer"),
    (["--batch-size", "-1"], "must be a positive integer"),
]


@pytest.mark.parametrize("option, message", _INVALID_CLI)
@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_invalid_batch_precedes_target_and_model(
    monkeypatch, capsys, option, message, dry_run
):
    def forbidden(*_args):
        pytest.fail("target resolution ran before batch-size validation")

    monkeypatch.setattr(cli, "_resolve_reindex_target", forbidden)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MODEL", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_reindex(
            [
                "--model",
                "must-not-be-set",
                *(["--dry-run"] if dry_run else []),
                *option,
            ]
        )
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert "MNEMOSYNE_EMBEDDING_MODEL" not in os.environ


@pytest.mark.parametrize("option, message", _INVALID_CLI)
@pytest.mark.parametrize("target", ["default", "db", "bank"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_invalid_batch_has_no_filesystem_side_effects(
    tmp_path, option, message, target, dry_run
):
    # Fresh interpreter: no mocked predecessor can hide imports/DB/backup work.
    db = tmp_path / "target.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sentinel (value TEXT)")
    conn.execute("INSERT INTO sentinel VALUES ('unchanged')")
    conn.commit()
    conn.close()
    before = db.read_bytes()
    target_args = {
        "default": [],
        "db": ["--db", str(db)],
        "bank": ["--bank", "missing"],
    }[target]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mnemosyne.cli",
            "reindex",
            *target_args,
            "--model",
            "must-not-load",
            "--yes",
            *(["--dry-run"] if dry_run else []),
            *option,
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert db.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["target.db"]


@pytest.mark.parametrize("batch_size", [0, -1, True, False, None, "8", 8.0, [], {}])
@pytest.mark.parametrize("dry_run", [False, True])
def test_core_invalid_batch_precedes_sql(batch_size, dry_run):
    class UntouchedConnection:
        def __getattr__(self, name):
            pytest.fail(f"connection touched before validation: {name}")

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        beam.reindex_vectors(
            UntouchedConnection(), batch_size=batch_size, dry_run=dry_run
        )


class _IndexSize:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class _IntOnlySize:
    def __int__(self):
        return 8


@pytest.mark.parametrize(
    "batch_size", [_IndexSize(0), _IndexSize(-1), _IndexSize(8.0), _IntOnlySize()]
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_core_invalid_integral_protocol_precedes_sql(batch_size, dry_run):
    class UntouchedConnection:
        def __getattr__(self, name):
            pytest.fail(f"connection touched before validation: {name}")

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        beam.reindex_vectors(
            UntouchedConnection(), batch_size=batch_size, dry_run=dry_run
        )


class _Response:
    def __init__(self, vectors):
        self.body = json.dumps({"data": [{"embedding": v} for v in vectors]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def _capped_transport(monkeypatch, source_vectors):
    for flag in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "http://127.0.0.1:11435/v1")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX", "")
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MAX_CHARS", raising=False)
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")
    requests = []

    def urlopen(request, **_kwargs):
        assert request.full_url == "http://127.0.0.1:11435/v1/embeddings"
        payload = json.loads(request.data)
        assert payload["model"] == embeddings._DEFAULT_MODEL
        texts = payload["input"]
        requests.append(texts)
        if len(texts) > 10:
            raise urllib.error.HTTPError(
                request.full_url, 400, "input cap", {}, io.BytesIO()
            )
        return _Response([source_vectors[text] for text in texts])

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", urlopen)
    return requests


def _source_store(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sqlite_vec")
    memory = beam.BeamMemory(db_path=str(tmp_path / "target.db"), session_id="offline")
    conn = memory.conn
    assert beam._vec_available(conn) and beam._wm_vec_available(conn)
    assert beam._mib is not None
    vectors = {}
    for tier, count in (("working_memory", 65), ("episodic_memory", 19)):
        for index in range(count):
            text = f"{tier} anonymous source {index}"
            vector = [-1.0] * embeddings.EMBEDDING_DIM
            vector[len(vectors)] = 1.0
            vectors[text] = vector
            conn.execute(
                f"INSERT INTO {tier} (id, content, source, timestamp, session_id) "
                "VALUES (?, ?, 'test', '2026-01-01T00:00:00', 'offline')",
                (f"{tier}-{index}", text),
            )
    conn.commit()
    return memory, np, vectors


def _snapshot(db):
    import sqlite_vec

    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        result = {
            "marker": conn.execute("PRAGMA user_version").fetchone()[0],
            "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
            "json": conn.execute(
                "SELECT memory_id, embedding_json, model FROM memory_embeddings ORDER BY memory_id"
            ).fetchall(),
            "binary": conn.execute(
                "SELECT id, binary_vector FROM episodic_memory ORDER BY id"
            ).fetchall(),
        }
        for table in ("vec_working", "vec_episodes", "vec_facts"):
            result[table] = conn.execute(
                f"SELECT rowid, embedding FROM {table} ORDER BY rowid"
            ).fetchall()
        return result
    finally:
        conn.close()


def test_capped_api_default_fails_atomically_then_cli_batch8_maps_every_row(
    tmp_path, monkeypatch
):
    memory, np, vectors = _source_store(tmp_path)
    conn = memory.conn
    requests = _capped_transport(monkeypatch, vectors)
    # Seed old persisted vectors through the real API and real rebuild first.
    beam.reindex_vectors(conn, batch_size=8)
    db = tmp_path / "target.db"
    before = _snapshot(db)
    assert before["marker"] & beam._VEC_NORM_BIT
    assert len(before["vec_working"]) == 65
    assert len(before["vec_episodes"]) == 19
    requests.clear()

    with pytest.raises(RuntimeError, match="working_memory embedding batch"):
        beam.reindex_vectors(conn)
    assert [len(batch) for batch in requests] == [64]
    assert not conn.in_transaction
    assert _snapshot(db) == before

    # Replace the model's vector mapping, so stale successful seeding cannot
    # satisfy the following read-back assertions.
    for vector in vectors.values():
        vector[:] = [-value for value in vector]
    requests.clear()
    cli.cmd_reindex(["--db", str(db), "--batch-size", "8", "--yes", "--no-backup"])
    assert [len(batch) for batch in requests] == [8] * 8 + [1, 8, 8, 3]
    expected_texts = [
        row[0] for row in conn.execute("SELECT content FROM working_memory")
    ]
    expected_texts += [
        row[0] for row in conn.execute("SELECT content FROM episodic_memory")
    ]
    assert [text for batch in requests for text in batch] == expected_texts
    after = _snapshot(db)
    assert after["quick_check"] == "ok"
    assert after["marker"] == before["marker"]
    assert after["json"] != before["json"]
    assert after["binary"] != before["binary"]
    assert len(after["vec_working"]) == 65
    assert len(after["vec_episodes"]) == 19
    working = dict(conn.execute("SELECT id, content FROM working_memory"))
    assert {key: json.loads(value) for key, value, _model in after["json"]} == {
        key: vectors[text] for key, text in working.items()
    }
    for tier, table in (
        ("working_memory", "vec_working"),
        ("episodic_memory", "vec_episodes"),
    ):
        rows = conn.execute(f"SELECT rowid, content FROM {tier}").fetchall()
        assert {row[0] for row in after[table]} == {row[0] for row in rows}
        for rowid, text in rows:
            query = np.asarray(vectors[text], dtype=np.float32)
            if tier == "working_memory":
                hits = beam._wm_vec_search_sqlite(conn, query, k=1, where_sql="1=1")
                expected_id = conn.execute(
                    "SELECT id FROM working_memory WHERE rowid = ?", (rowid,)
                ).fetchone()[0]
                assert hits[0]["id"] == expected_id
            else:
                hits = beam._vec_search(conn, query.tolist(), k=1)
                assert hits[0]["rowid"] == rowid
    episodic = dict(conn.execute("SELECT id, content FROM episodic_memory"))
    assert dict(after["binary"]) == {
        key: beam._mib(np.asarray(vectors[text])) for key, text in episodic.items()
    }


@pytest.mark.parametrize("kind", ["numpy", "index"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_core_accepts_integral_protocol(tmp_path, monkeypatch, kind, dry_run):
    memory, np, vectors = _source_store(tmp_path)
    batch_size = np.int64(8) if kind == "numpy" else _IndexSize(8)
    requests = _capped_transport(monkeypatch, vectors)
    db = tmp_path / "target.db"
    before = _snapshot(db)
    plan = beam.reindex_vectors(memory.conn, batch_size=batch_size, dry_run=dry_run)
    assert (plan["working_memory"], plan["episodic_memory"]) == (65, 19)
    if kind == "index":
        assert batch_size.calls == 1
    after = _snapshot(db)
    if dry_run:
        assert requests == []
        assert after == before
    else:
        assert [len(batch) for batch in requests] == [8] * 8 + [1, 8, 8, 3]
        assert len(after["vec_working"]) == 65
        assert len(after["vec_episodes"]) == 19
        assert after["marker"] & beam._VEC_NORM_BIT
        working = dict(memory.conn.execute("SELECT id, content FROM working_memory"))
        assert {key: json.loads(value) for key, value, _model in after["json"]} == {
            key: vectors[text] for key, text in working.items()
        }
        episodic = dict(memory.conn.execute("SELECT id, content FROM episodic_memory"))
        assert dict(after["binary"]) == {
            key: beam._mib(np.asarray(vectors[text])) for key, text in episodic.items()
        }


def test_core_positive_dry_run_does_not_write(tmp_path):
    memory, _np, _vectors = _source_store(tmp_path)
    db = tmp_path / "target.db"
    before = _snapshot(db)
    plan = beam.reindex_vectors(memory.conn, batch_size=1, dry_run=True)
    assert (plan["working_memory"], plan["episodic_memory"]) == (65, 19)
    assert _snapshot(db) == before


def test_help_exposes_batch_size(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "[--batch-size N]" in result.stdout
    assert "default 64" in result.stdout
    assert not (tmp_path / "data").exists()
