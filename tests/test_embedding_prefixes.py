"""Prefix env vars are applied verbatim, byte-exact, on the correct call paths.
Uses a stub HTTP server so the raw request bytes are asserted (trailing spaces!).

Covers BOTH client branches: the API path (stub HTTP server asserts raw wire
bytes) and the fastembed path (fake model object records what reaches .embed())."""
import importlib, json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

QUERY_PREFIX = "task: search result | query: "   # trailing space is load-bearing
DOC_PREFIX = "title: none | text: "

RECORDED = []

class StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        RECORDED.append(json.loads(body))
        n = len(json.loads(body)["input"])
        resp = json.dumps({"object": "list", "model": "stub",
                           "data": [{"object": "embedding", "index": i, "embedding": [0.1] * 768}
                                    for i in range(n)],
                           "usage": {"prompt_tokens": 1, "total_tokens": 1}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(resp)))
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a): pass

@pytest.fixture
def embeddings_mod(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), StubHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", url)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MODEL", "embeddinggemma-300m-q4")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DIM", "768")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", QUERY_PREFIX)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX", DOC_PREFIX)
    RECORDED.clear()
    from mnemosyne.core import embeddings
    # Reload ONLY because upstream reads the API URL/model at module import time.
    # The PREFIXES are read at call time by the patch, so no reload is ever
    # needed for prefix changes (see test_unset_prefixes_unchanged).
    importlib.reload(embeddings)
    yield embeddings
    server.shutdown()

def test_query_prefix_byte_exact(embeddings_mod):
    # No cache manipulation: the cache is keyed on the PREFIXED text, so prefix
    # changes can never serve stale entries (see patch note in Step 4).
    assert embeddings_mod.embed_query("hvor bor brukeren") is not None
    assert RECORDED[-1]["input"] == ["task: search result | query: hvor bor brukeren"]

def test_doc_prefix_on_batch(embeddings_mod):
    assert embeddings_mod.embed(["fact one", "fact two"]) is not None
    assert RECORDED[-1]["input"] == ["title: none | text: fact one",
                                     "title: none | text: fact two"]

def test_single_doc_gets_doc_prefix_not_query(embeddings_mod):
    # Regression: embed([one]) used to delegate to embed_query (query prefix on a document)
    assert embeddings_mod.embed(["only fact"]) is not None
    assert RECORDED[-1]["input"] == ["title: none | text: only fact"]

def test_unset_prefixes_unchanged(embeddings_mod, monkeypatch):
    # No reload needed: prefixes are read at call time (this test proves it), and
    # the prefixed-text cache key means "plain" cannot collide with earlier entries.
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX")
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX")
    assert embeddings_mod.embed_query("plain") is not None
    assert RECORDED[-1]["input"] == ["plain"]

def test_fastembed_path_applies_prefixes(monkeypatch):
    # The fastembed (local ONNX) branch must apply the same prefixes: only the
    # model object is doubled; the branch selection logic runs for real.
    import numpy as np
    from mnemosyne.core import embeddings as emb
    class FakeFastembedModel:
        def __init__(self): self.received = []
        def embed(self, texts):
            self.received.append(list(texts))
            return [np.ones(768, dtype=np.float32) for _ in texts]
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_URL", raising=False)   # force non-API path
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_VIA_API", raising=False)
    # CI globally disables local model loading to avoid downloads. This test
    # installs a fake already-loaded model, so it must explicitly exercise the
    # local branch rather than inherit that suite-level opt-out.
    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_SKIP_EMBEDDINGS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_OFF", raising=False)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", QUERY_PREFIX)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DOC_PREFIX", DOC_PREFIX)
    fake = FakeFastembedModel()
    monkeypatch.setattr(emb, "TextEmbedding", object())      # fastembed "available"
    monkeypatch.setattr(emb, "_embedding_model", fake)       # already-"loaded" model
    emb._embed_query_cached.cache_clear()                    # isolate from API-path tests
    assert emb.embed(["local fact"]) is not None
    assert fake.received[-1] == [DOC_PREFIX + "local fact"]
    assert emb.embed_query("local query") is not None
    assert fake.received[-1] == [QUERY_PREFIX + "local query"]
