"""Shared sanitizer contract and actual entry points in both Hermes providers."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from mnemosyne.core.query_sanitize import sanitize_prefetch_query

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('[Alice] what time does the bakery open', 'what time does the bakery open'),
    ('[Alice] [Bob] who forgot the tickets', 'who forgot the tickets'),
    ('[Alice]\tmuseum plans for Sunday', 'museum plans for Sunday'),
    ('[Jean-Luc] library plans', 'library plans'),
    ('[Mary Ann] doctor appointment', 'doctor appointment'),
    ("[O'Brien] cafe visit", 'cafe visit'),
    ("[O’Brien] cafe visit", 'cafe visit'),
    ('[Anne] cafe plans', 'cafe plans'),
    ('[Jose Maria Aznar Lopez] what did he say', 'what did he say'),
    ('[Maria de la Cruz] says hello', 'says hello'),
    ('[王 明] 你好', '你好'),
    ('[محمد أحمد] where are we meeting', 'where are we meeting'),
    ('[Mary-Jane Anne van Dijk-Visser] lunch', 'lunch'),
    ('[Untitled] live version lyrics meaning', '[Untitled] live version lyrics meaning'),
    ('[TODO] fix the login bug', '[TODO] fix the login bug'),
    ('[Note] buy milk', '[Note] buy milk'),
    ('[TODOs] remaining work items', '[TODOs] remaining work items'),
    ('[Noted] meeting follow-ups', '[Noted] meeting follow-ups'),
    ('[Annotation] margin comment', '[Annotation] margin comment'),
    ('[TODO Now] fix login', '[TODO Now] fix login'),
    ('[API Docs] how to authenticate', '[API Docs] how to authenticate'),
    ('[Release Notes] v2 launch details', '[Release Notes] v2 launch details'),
    ('[API docs](http://x) how do I auth', '[API docs](http://x) how do I auth'),
    ('[2026-05-14 12:00] meeting notes', '[2026-05-14 12:00] meeting notes'),
    ('[untitled_track] lyrics', '[untitled_track] lyrics'),
    ("[Alice]'s bakery order", "[Alice]'s bakery order"),
    ('[Alice] , what did we decide', ', what did we decide'),
    ('[Alice] - status update', '- status update'),
    ('[ＴＯＤＯ] fix login', '[ＴＯＤＯ] fix login'),
    ('[Neo1999] what did they say', '[Neo1999] what did they say'),
    ('no stamp here', 'no stamp here'),
    ('', ''),
    ('[] empty stamp is kept', '[] empty stamp is kept'),
    ('[xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx] overlong stamp kept', '[xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx] overlong stamp kept'),
    ('  ＡＢＣ\t ', '  ＡＢＣ\t '),
    (' \n[ＴＯＤＯ] fix login  ', ' \n[ＴＯＤＯ] fix login  '),
    ('［ＴＯＤＯ］ fix login', '［ＴＯＤＯ］ fix login'),
    ('[TＯDO] keep', '[TＯDO] keep'),
    ('[1/2] amount', '[1/2] amount'),
    (' \t', ' \t'),
    ('[Alice]word', '[Alice]word'),
    ('[Alice](url)', '[Alice](url)'),
    ('[Alice] [TODO]', '[TODO]'),
    ('[Ａｌｉｃｅ] ＡＢＣ  ', 'ＡＢＣ  '),
    ('［Alice］ ＡＢＣ  ', 'ＡＢＣ  '),
    ('[Álice] café\t ', 'café\t '),
    ('  [Alice]\t-5 degrees  ', '-5 degrees  '),
    ('[Alice] **bold**\n', '**bold**\n'),
    ('[Alice] [ＴＯＤＯ] fix  ', '[ＴＯＤＯ] fix  '),
    ('[Alice] [Bob]', ''),
    ('[Alice] ,!?', ''),
    ('[Alice] [１２] value', '[１２] value'),
    ('[Alice] [½] amount', '[½] amount'),
]


@pytest.mark.parametrize("query,expected", CASES)
def test_shared_sanitizer_matrix(query, expected):
    assert sanitize_prefetch_query(query) == expected


@pytest.fixture(params=("root", "packaged"))
def provider_module(request, monkeypatch):
    path = ROOT / ("hermes_memory_provider/__init__.py" if request.param == "root"
                   else "integrations/hermes/src/mnemosyne_hermes/__init__.py")
    name = "_query_sanitize_" + request.param
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == path.resolve()
    return request.param, module


class FakeBeam:
    """Instrument provider paths without recall's implicit empty handling."""
    author_id = "test-author"
    session_id = "session-a"

    def __init__(self):
        self.queries = []
        self.canonical_reads = []
        self.canonical = SimpleNamespace(list=self.list_canonical)
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE working_memory (content, importance, timestamp, source, session_id)")

    def list_canonical(self, *args, **kwargs):
        self.canonical_reads.append((args, kwargs))
        return []

    def recall(self, query, **kwargs):
        self.queries.append((query, kwargs))
        rows = [{"content": "The user usually orders spicy ramen; ramen is the user favorite order at the corner shop",
                 "timestamp": "2026-05-14T12:00:00Z", "importance": 0.6,
                 "score": 0.9, "keyword_score": 0.9, "trust_tier": "STATED"}]
        return {"results": rows, "explain": {}} if kwargs.get("explain") else rows


@pytest.fixture
def provider(provider_module, monkeypatch):
    kind, module = provider_module
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", raising=False)
    p = module.MnemosyneMemoryProvider()
    p._beam = FakeBeam()
    p._shared_surface_read = False
    if kind == "root":
        p._prefetch_profile = "general"
    yield kind, module, p
    p._beam.conn.close()


@pytest.mark.parametrize("query,expected", CASES)
def test_both_providers_delegate_to_shared_helper(provider_module, query, expected):
    _, module = provider_module
    assert module._sanitize_prefetch_query(query) == expected


def test_shared_helper_wiring(provider_module, monkeypatch):
    from mnemosyne.core import query_sanitize
    _, module = provider_module
    sentinel = object()
    monkeypatch.setattr(query_sanitize, "sanitize_prefetch_query", lambda q: sentinel)
    assert module._sanitize_prefetch_query("[Alice] topic") is sentinel


@pytest.mark.parametrize("query", ["[Alice] [Bob]", "[Ａｌｉｃｅ]", "[Alice] ,!?", "", "  \t"])
def test_prefetch_empty_skips_all_query_sources(provider, query):
    _, _, p = provider
    assert p.prefetch(query) == ""
    assert p._beam.queries == []
    assert p._beam.canonical_reads == []


def test_empty_keeps_only_query_independent_context(provider, monkeypatch):
    kind, module, p = provider
    if kind == "packaged":
        # This provider's canonical context is query-driven, not always-inject.
        assert p.prefetch("[Alice] [Bob]") == ""
        assert p._beam.canonical_reads == []
        assert p._beam.queries == []
        return
    profile = module.PrefetchProfile(name="query-test", sources=("bank", "external"))
    monkeypatch.setattr(module, "_resolve_profile", lambda _: profile)
    called = []
    p._prefetch_sources["external"] = lambda *a, **k: called.append("source")
    monkeypatch.setattr(p, "_prefetch_model_slots", lambda *a: called.append("model"))
    p._beam.conn.executemany("INSERT INTO working_memory VALUES (?, .95, '', 'identity', ?)",
                            [("Current contact identity.", "session-a"), ("Other contact identity.", "session-b")])
    output = p.prefetch("[Alice] [Bob]")
    assert "[IDENTITY] Current contact identity." in output
    assert "Other contact" not in output
    assert called == []
    assert p._beam.queries == []


@pytest.mark.parametrize("query,expected", [
    ("[Alice] what ramen does user order", "what ramen does user order"),
    ("[O’Brien] what ramen does user order", "what ramen does user order"),
    ("  what ramen does user order  ", "  what ramen does user order  "),
    ("[Alice] ＡＢＣ ramen order  ", "ＡＢＣ ramen order  "),
])
def test_prefetch_nonempty_uses_exact_query(provider, query, expected):
    _, _, p = provider
    output = p.prefetch(query)
    assert [q for q, _ in p._beam.queries] == [expected]
    assert "ramen" in output
    assert p._beam.canonical_reads


@pytest.mark.parametrize("query", ["[Alice] [Bob]", "[Alice] !?", "", " \t"])
def test_explicit_recall_empty_calls_no_sources(provider, query):
    _, _, p = provider
    p._shared_surface_read = True
    output = json.loads(p._handle_recall({"query": query}))
    assert output == {"error": "query is required"}
    assert p._beam.queries == []
    assert p._beam.canonical_reads == []


@pytest.mark.parametrize("explain", [False, True])
def test_explicit_recall_preserves_suffix_and_surface_query(provider, explain):
    _, _, p = provider
    surface = FakeBeam()
    p._shared_surface_read = True
    p._shared_surface_bank = "shared"
    p._surface_beam = surface
    try:
        output = json.loads(p._handle_recall({"query": "[Alice] ＡＢＣ ramen  ", "explain": explain}))
        assert output["query"] == "ＡＢＣ ramen  "
        assert [q for q, _ in p._beam.queries] == ["ＡＢＣ ramen  "]
        assert [q for q, _ in surface.queries] == ["ＡＢＣ ramen  "]
        assert output["count"] > 0
    finally:
        surface.conn.close()


def test_large_repeated_stamp_input():
    assert sanitize_prefetch_query("[Alice] " * 50000 + "ＡＢＣ  ") == "ＡＢＣ  "
