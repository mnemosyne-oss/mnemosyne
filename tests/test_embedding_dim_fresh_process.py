"""Fresh-process coverage for the fail-loud embedding-dimension resolver.

The helper-level tests in ``test_embeddings_multilingual.py`` call
``_get_embedding_dim`` *after* ``mnemosyne.core.embeddings`` is already
imported, so they do not prove the actionable error fires at the
import/startup boundary -- before any ``vec0`` DDL or vector write. These
tests spawn a fresh interpreter per case so the failure is observed exactly
where an operator would see it: at process startup.

They also assert parity: direct core and both Hermes provider surfaces
(``hermes_memory_provider`` and ``mnemosyne_hermes``) surface the same
actionable configuration failure for an unknown model with no explicit
``MNEMOSYNE_EMBEDDING_DIM``.
"""
from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"

_ERROR_MARKERS = ("Unknown embedding model", "MNEMOSYNE_EMBEDDING_DIM")


def _run_fresh(
    code: str,
    tmp_path: Path,
    *,
    pythonpath: str | None = None,
    timeout: float = 120.0,
    **env_overrides: str,
):
    """Run ``code`` in a fresh interpreter with an isolated data dir.

    ``MNEMOSYNE_EMBEDDING_DIM`` is removed so the unknown-model path is
    exercised (no explicit override). The data dir points under ``tmp_path``
    so we can assert no database was written. ``timeout`` bounds the child so
    a hung interpreter fails the test instead of stalling the run.
    """
    env = os.environ.copy()
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "data")
    env["HOME"] = str(tmp_path / "home")
    env.pop("MNEMOSYNE_EMBEDDING_DIM", None)
    # Strip embedding-disable flags so the subprocess exercises the fail-loud
    # path, not the _is_disabled() 384 fallback (these are set in some CI).
    for _flag in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        env.pop(_flag, None)
    if pythonpath:
        # Prepend so the provider package resolves before any installed copy.
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=timeout,
    )


def _assert_fail_fast(result: subprocess.CompletedProcess, tmp_path: Path) -> None:
    """The process crashed at import with the actionable error, before any DB write.

    The subprocess reaches init_beam, the schema/vec0 DDL entry point that
    creates mnemosyne.db, so the no-database assertion is load-bearing: on main
    (silent 384 fallback) import succeeds and init_beam writes the database;
    only the import-time ValueError prevents it here.
    """
    assert result.returncode != 0, result.stderr
    assert any(marker in result.stderr for marker in _ERROR_MARKERS), result.stderr
    assert not list((tmp_path / "data").rglob("*.db")), "a database was written before the error"


@pytest.mark.parametrize(
    "model,extra_env",
    [
        pytest.param("some/unknown-local-model", {}, id="unknown-local"),
        pytest.param(
            "custom-endpoint-model",
            {"MNEMOSYNE_EMBEDDING_API_URL": "http://localhost:8000/v1"},
            id="custom-endpoint",
        ),
        pytest.param("openai/text-embedding-fake", {}, id="api-model"),
    ],
)
def test_unknown_model_fails_at_import_before_any_write(tmp_path, model, extra_env):
    """A fresh process with an unknown embedding model (local, custom-endpoint,
    or API) and no explicit dimension must fail at import -- before any vec0 DDL
    or vector write -- with the actionable error, not silently assume 384."""
    # Reach init_beam() so the no-database check is load-bearing (see _assert_fail_fast).
    result = _run_fresh(
        "from mnemosyne.core import beam; beam.init_beam()",
        tmp_path,
        MNEMOSYNE_EMBEDDING_MODEL=model,
        **extra_env,
    )
    _assert_fail_fast(result, tmp_path)


@pytest.mark.parametrize(
    "code,pythonpath,fail_fast",
    [
        pytest.param("from mnemosyne.core import beam", None, True, id="direct-core"),
        pytest.param("import hermes_memory_provider", str(PROJECT_ROOT), True, id="hermes_memory_provider"),
        pytest.param("import mnemosyne_hermes", str(INTEGRATION_SRC), False, id="mnemosyne_hermes"),
    ],
)
def test_unknown_model_parity_across_surfaces(tmp_path, code, pythonpath, fail_fast):
    """Direct core and both Hermes provider surfaces expose the actionable
    configuration failure for an unknown model with no explicit dim.

    Direct core and ``hermes_memory_provider`` fail fast (non-zero exit at
    import). ``mnemosyne_hermes`` graceful-degrades by design: it catches the
    configuration error, logs it with the actionable message, and disables the
    affected batch tools rather than crashing the provider. Every surface must
    expose the actionable error; the two fail-fast surfaces must also exit
    non-zero, so a future regression that swallows the error is caught.
    """
    result = _run_fresh(
        code,
        tmp_path,
        pythonpath=pythonpath,
        MNEMOSYNE_EMBEDDING_MODEL="some/unknown-model",
    )
    assert any(marker in result.stderr for marker in _ERROR_MARKERS), result.stderr
    if fail_fast:
        assert result.returncode != 0, result.stderr
    else:
        # mnemosyne_hermes graceful-degrades by design: assert it exits 0 so a
        # regression that crashes the provider outright is still caught.
        assert result.returncode == 0, result.stderr


def _fake_endpoint_code(dim: int) -> str:
    """Subprocess source that serves an OpenAI-compatible /embeddings endpoint
    returning only `dim`-dimensional vectors on an ephemeral local port, points
    MNEMOSYNE_EMBEDDING_API_URL at it, then boots mnemosyne, writes one memory
    and recalls it, printing REQUESTS / RECALL / VEC_ROWS markers for the
    parent to assert on."""
    return textwrap.dedent(
        f"""
        import json
        import math
        import os
        import sqlite3
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        REQUESTS = {{"n": 0, "log": []}}

        def _vec_for(text):
            # Input-dependent, deterministic vectors: similar texts (similar
            # character-weight sums) map to nearly-parallel vectors, different
            # texts to near-orthogonal ones, so the KNN ranking is meaningful
            # instead of identical vectors for every input.
            seed = sum((i + 1) * ord(c) for i, c in enumerate(text))
            return [math.sin(seed * 0.0001 + i * 0.01) for i in range({dim})]

        class _FakeEmbeddings(BaseHTTPRequestHandler):
            # OpenAI-compatible /embeddings endpoint serving {dim}-dim vectors only.
            def do_POST(self):
                if self.path.rstrip("/") != "/v1/embeddings":
                    self.send_error(404, f"unexpected route: {{self.path}}")
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                REQUESTS["log"].append({{"model": body.get("model"), "input": body["input"]}})
                payload = json.dumps({{
                    "data": [{{"embedding": _vec_for(t), "index": i}} for i, t in enumerate(body["input"])]
                }}).encode()
                REQUESTS["n"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _FakeEmbeddings)
        os.environ["MNEMOSYNE_EMBEDDING_API_URL"] = f"http://127.0.0.1:{{server.server_port}}/v1"
        threading.Thread(target=server.serve_forever, daemon=True).start()

        from mnemosyne.core import beam
        from mnemosyne.core.memory import recall, remember
        beam.init_beam()
        remember("dimension end to end document", source="e2e-dim")
        remember("unrelated grocery list apples", source="e2e-dim")
        print("REQUESTS", REQUESTS["n"])  # before recall: survives a recall crash
        print("WRITE_MODEL", REQUESTS["log"][0]["model"])
        print("WRITE_INPUT", json.dumps(REQUESTS["log"][0]["input"]))
        results = recall("dimension end to end document", top_k=3)
        print("REQUESTS_TOTAL", REQUESTS["n"])
        print("QUERY_MODEL", REQUESTS["log"][-1]["model"])
        print("QUERY_INPUT", REQUESTS["log"][-1]["input"][0])
        def _content(r):
            return (r.get("content") if isinstance(r, dict) else getattr(r, "content", "")) or ""

        print("RECALL", len(results))
        print("RECALL_CONTENT", json.dumps([_content(r) for r in results]))
        print("RECALL_TOP", _content(results[0]) if results else "")
        if beam._SQLITE_VEC_AVAILABLE:
            import sqlite_vec
            conn = sqlite3.connect(beam._default_db_path())
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            print("VEC_ROWS", conn.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0])
            conn.close()
        """
    )


def _fake_endpoint_recall_only_code(dim: int) -> str:
    """Same fake endpoint and boot, but only recalls: for a SECOND process
    reopening a data directory written by a previous one.

    The endpoint answers EVERY request with the vector the writer stored for
    the probe document (same seeded function), so the reader's query
    embedding is exactly the stored vector. The reader asserts the recalled
    row's dense_score: nonzero only when the KNN ran over the REOPENED
    vec_working table and matched, which FTS alone cannot produce."""
    return textwrap.dedent(
        f"""
        import json
        import math
        import os
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        def _vec_for(text):
            seed = sum((i + 1) * ord(c) for i, c in enumerate(text))
            return [math.sin(seed * 0.0001 + i * 0.01) for i in range({dim})]

        _doc_vec = _vec_for("dimension end to end document")

        class _FakeEmbeddings(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path.rstrip("/") != "/v1/embeddings":
                    self.send_error(404, f"unexpected route: {{self.path}}")
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = json.dumps({{
                    "data": [{{"embedding": _doc_vec, "index": i}} for i in range(len(body["input"]))]
                }}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _FakeEmbeddings)
        os.environ["MNEMOSYNE_EMBEDDING_API_URL"] = f"http://127.0.0.1:{{server.server_port}}/v1"
        threading.Thread(target=server.serve_forever, daemon=True).start()

        from mnemosyne.core import beam
        from mnemosyne.core.memory import recall
        beam.init_beam()
        results = recall("dimension end to end document", top_k=3)
        def _content(r):
            return (r.get("content") if isinstance(r, dict) else getattr(r, "content", "")) or ""

        print("RECALL", len(results))
        print("RECALL_CONTENT", json.dumps([_content(r) for r in results]))
        print("RECALL_TOP", _content(results[0]) if results else "")
        # dense_score is nonzero only when the KNN over the REOPENED
        # vec_working table returned the row: FTS alone would still surface
        # the document with dense_score 0.0.
        print("DENSE", results[0].get("dense_score", -1) if results else -1)
        """
    )


def test_unknown_model_with_explicit_dim_boots_end_to_end(tmp_path):
    """The success half of the #521 contract, at process scope: an unknown model
    plus an explicit positive MNEMOSYNE_EMBEDDING_DIM boots cleanly through
    init_beam(), bakes the explicit dimension into the vec0 tables, and
    completes an embedding-backed write + recall through a local fake endpoint
    that only ever serves 1024-dim vectors (the mxbai-via-custom-endpoint
    production scenario).

    Both the document embedding (write) and the query embedding (recall) must
    hit the endpoint, so a regression that stores or queries at any other
    dimension cannot pass: the endpoint produces 1024-dim vectors only, and a
    mismatched vec0 write would fail. Boot and endpoint assertions always run;
    the vec0-dimension checks run only where sqlite-vec is installed (the
    repo's importorskip convention) because the tables are not created
    without it."""
    result = _run_fresh(
        _fake_endpoint_code(1024),
        tmp_path,
        MNEMOSYNE_EMBEDDING_MODEL="mixedbread-ai/mxbai-embed-large-v1",
        MNEMOSYNE_EMBEDDING_DIM="1024",
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Write-side and query-side vectorization both went through the endpoint
    # (>=2 embedding calls total: one for the remembered content, one for the
    # query; the write-side REQUESTS marker is also checked below).
    assert int(re.search(r"REQUESTS_TOTAL (\d+)", out).group(1)) >= 2, out + result.stderr
    assert int(re.search(r"REQUESTS (\d+)", out).group(1)) >= 1, out + result.stderr
    # Both calls asked the endpoint for the configured model, and the write
    # side carried the remembered content while the query side carried the
    # query (possibly prefixed by the model's query prefix).
    assert re.search(r"^WRITE_MODEL mixedbread-ai/mxbai-embed-large-v1$", out, re.M), out
    assert re.search(r"^QUERY_MODEL mixedbread-ai/mxbai-embed-large-v1$", out, re.M), out
    write_input = re.search(r"^WRITE_INPUT (.+)$", out, re.M)
    assert write_input and "dimension end to end document" in write_input.group(1), out
    assert write_input and "grocery" not in write_input.group(1), out
    query_input = re.search(r"^QUERY_INPUT (.+)$", out, re.M)
    assert query_input and "dimension end to end document" in query_input.group(1), out
    # Recall returned the probe...
    assert re.search(r"RECALL [1-9]", out), out + result.stderr
    recall_content = re.search(r"^RECALL_CONTENT (.+)$", out, re.M)
    assert recall_content and "dimension end to end document" in recall_content.group(1), out
    # ...and ranked the matching document above the unrelated distractor.
    top = re.search(r"^RECALL_TOP (.+)$", out, re.M)
    assert top and "dimension end to end document" in top.group(1), out
    assert top and "grocery" not in top.group(1), out
    # The vec0 tables must carry the explicit dimension, not a guessed 384,
    # and the stored vector must have landed in the 1024-dim table (not the
    # float-JSON fallback). sqlite_master needs no extension load to read DDL.
    sqlite_vec = pytest.importorskip("sqlite_vec")  # noqa: F841
    conn = sqlite3.connect(tmp_path / "data" / "mnemosyne.db")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        dims = {
            name: int(re.search(r"\[(\d+)\]", sql).group(1))
            for name, sql in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('vec_episodes','vec_working')"
            )
        }
        assert dims == {"vec_episodes": 1024, "vec_working": 1024}, dims
        vec_rows = conn.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0]
        assert vec_rows >= 1, "no vector stored in the 1024-dim vec0 table"
    finally:
        conn.close()


def test_second_process_recalls_stored_memory_same_dim(tmp_path):
    """Cross-process persistence: a SECOND fresh process reopening the same
    data directory must recall the memory the first one stored, with the same
    explicit dimension. The vec0 tables and their 1024-dim vectors were
    created by the writer process; the reader proves they survive the process
    boundary and serve a vector-backed recall (same-process recall alone, or
    two readers in one process, would not)."""
    # Skip before launching either child: the DENSE assertion below requires
    # the reopened vec0 table, which does not exist without sqlite-vec.
    pytest.importorskip("sqlite_vec")
    writer = _run_fresh(
        _fake_endpoint_code(1024),
        tmp_path,
        MNEMOSYNE_EMBEDDING_MODEL="mixedbread-ai/mxbai-embed-large-v1",
        MNEMOSYNE_EMBEDDING_DIM="1024",
    )
    assert writer.returncode == 0, writer.stderr

    reader = _run_fresh(
        _fake_endpoint_recall_only_code(1024),
        tmp_path,  # same MNEMOSYNE_DATA_DIR as the writer
        MNEMOSYNE_EMBEDDING_MODEL="mixedbread-ai/mxbai-embed-large-v1",
        MNEMOSYNE_EMBEDDING_DIM="1024",
    )
    assert reader.returncode == 0, reader.stderr
    out = reader.stdout
    assert re.search(r"RECALL [1-9]", out), out + reader.stderr
    recall_content = re.search(r"^RECALL_CONTENT (.+)$", out, re.M)
    assert recall_content and "dimension end to end document" in recall_content.group(1), out
    top = re.search(r"^RECALL_TOP (.+)$", out, re.M)
    assert top and "dimension end to end document" in top.group(1), out
    assert top and "grocery" not in top.group(1), out
    # Vector-backed proof: dense_score is nonzero only when the KNN over the
    # reopened vec_working table returned the stored row (the endpoint serves
    # exactly that vector, so the match is ~1.0). An FTS-only retrieval over a
    # store that lost its vectors would report 0.0.
    dense = re.search(r"^DENSE (\S+)$", out, re.M)
    assert dense and float(dense.group(1)) > 0.9, out


def test_endpoint_serving_wrong_dim_degrades_and_serves_lexical_fallback(tmp_path):
    """The failure half of the explicit-dimension contract, post-#754: when the
    endpoint's true dimension (384) disagrees with the explicit
    MNEMOSYNE_EMBEDDING_DIM (1024), the mismatched vectors must NOT be stored
    (vec_working stays empty; the write-side insert is discarded with a logged
    warning, never corrupting the 1024-dim vec0 table) and the mismatched
    query degrades instead of crashing the process: recall() keeps serving the
    stored memories through its lexical voices while an actionable dimension
    diagnostic reaches stderr."""
    # Skip before launching the child: without sqlite-vec there is no vec0
    # table and this scenario's premise (a 1024-dim table to mismatch against)
    # is absent.
    sqlite_vec = pytest.importorskip("sqlite_vec")  # noqa: F841
    result = _run_fresh(
        _fake_endpoint_code(384),
        tmp_path,
        MNEMOSYNE_EMBEDDING_MODEL="mixedbread-ai/mxbai-embed-large-v1",
        MNEMOSYNE_EMBEDDING_DIM="1024",
    )
    # Degraded, not crashed: the process completes and the mismatch diagnostic
    # (from #754's query-side guard) reaches stderr.
    assert result.returncode == 0, "mismatched-dim query still crashes the process:\n" + result.stderr
    assert re.search(r"dimension mismatch", result.stderr, re.I), result.stderr
    # Query-side specific: the write-side warning also says "dimension
    # mismatch", so pin the guard's own wording to catch the guidance branch
    # regressing into the generic warning.
    assert re.search(r"query vector is 384-dim", result.stderr), result.stderr
    # Both the write-side and query-side embedding calls went to the endpoint.
    assert int(re.search(r"REQUESTS (\d+)", result.stdout).group(1)) >= 1, result.stdout + result.stderr
    # The lexical voices still serve the stored memories.
    out = result.stdout
    assert re.search(r"RECALL [1-9]", out), out + result.stderr
    recall_content = re.search(r"^RECALL_CONTENT (.+)$", out, re.M)
    assert recall_content and "dimension end to end document" in recall_content.group(1), out
    # And the mismatched vectors never reached the 1024-dim vec0 table.
    conn = sqlite3.connect(tmp_path / "data" / "mnemosyne.db")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        vec_rows = conn.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0]
        assert vec_rows == 0, f"{vec_rows} mismatched-dim vectors reached the 1024-dim vec0 table"
    finally:
        conn.close()


@pytest.mark.parametrize("blank", ["", "  ", "\t"], ids=["empty", "whitespace", "tab"])
def test_blank_embedding_model_env_falls_back_to_default(tmp_path, blank):
    """A blank (empty or whitespace-only) MNEMOSYNE_EMBEDDING_MODEL (routine in
    Docker Compose `- VAR=${X}` with X unset, and .env files) normalizes to the
    default model (bge-small-en-v1.5, 384-dim), not a model named empty/whitespace
    that would be unknown and raise at import under the fail-loud rule. Mirrors
    the .strip() blank handling used for MNEMOSYNE_EMBEDDING_DIM."""
    result = _run_fresh(
        "from mnemosyne.core import beam, embeddings; "
        "print(embeddings._DEFAULT_MODEL, beam.EMBEDDING_DIM)",
        tmp_path,
        MNEMOSYNE_EMBEDDING_MODEL=blank,
    )
    assert result.returncode == 0, result.stderr
    # Assert the default model too, not just its 384 dimension, so a hard-coded
    # fallback dim with a different default model cannot pass (CodeRabbit, #521).
    # Resolved in the subprocess: the parent's env is not under test here.
    assert result.stdout.strip() == "BAAI/bge-small-en-v1.5 384", result.stdout
