"""Regression tests for TripleStore.for_bank() (#548).

Bare TripleStore() always resolves to its own standalone triples.db, never to
a bank's mnemosyne.db, while the mnemosyne_triple_add/triple_query MCP tools
resolve the calling bank's mnemosyne.db. for_bank() gives library callers a
supported way to reach the same file those tools use.
"""

from mnemosyne.core.banks import BankManager
from mnemosyne.core.triples import TripleStore


def test_for_bank_default_matches_bank_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    kg = TripleStore.for_bank()
    assert kg.db_path == BankManager().get_bank_db_path("default")
    assert kg.db_path == tmp_path / "mnemosyne.db"


def test_for_bank_named_matches_bank_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    BankManager().create_bank("work")
    kg = TripleStore.for_bank("work")
    assert kg.db_path == BankManager().get_bank_db_path("work")
    assert kg.db_path == tmp_path / "banks" / "work" / "mnemosyne.db"


def test_for_bank_differs_from_standalone_default(tmp_path, monkeypatch):
    """Negative control: if for_bank() ever regressed to the standalone
    triples.db path, this must fail. Bare TripleStore() is unaffected by
    for_bank() and the two must not collide."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    bare = TripleStore()
    banked = TripleStore.for_bank()
    assert bare.db_path != banked.db_path
    assert bare.db_path.name == "triples.db"
    assert banked.db_path.name == "mnemosyne.db"


def test_triple_added_via_for_bank_is_visible_to_mcp_style_lookup(tmp_path, monkeypatch):
    """Mirrors what mcp_tools.py's _handle_triple_add/_handle_triple_query
    actually do: TripleStore(db_path=mem.beam.db_path) for a bank created via
    Mnemosyne(bank=...). Before this fix there was no supported way to reach
    that same file from the bare SDK, only from inside the MCP handlers."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    from mnemosyne.core.memory import Mnemosyne

    mem = Mnemosyne(session_id="s", bank="work")

    kg = TripleStore.for_bank("work")
    kg.add("repro_subject", "repro_predicate", "repro_object", valid_from="2026-07-25")

    mcp_style = TripleStore(db_path=mem.beam.db_path)
    results = mcp_style.query(subject="repro_subject")
    assert [r["object"] for r in results] == ["repro_object"]
