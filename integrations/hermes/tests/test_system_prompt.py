from __future__ import annotations

from mnemosyne_hermes import MnemosyneMemoryProvider


def test_system_prompt_routes_to_structured_mnemosyne_surfaces():
    provider = MnemosyneMemoryProvider()
    provider._beam = object()

    block = provider.system_prompt_block()

    assert "Mnemosyne is primary" in block
    assert "legacy memory tool is deprecated" in block
    assert "mnemosyne_recall" in block
    assert "mnemosyne_remember for ordinary facts/preferences/insights" in block
    assert "mnemosyne_remember_canonical" in block
    assert "mnemosyne_triple_add" in block
    assert "mnemosyne_graph_link/query" in block
    assert "mnemosyne_validate/invalidate/update/forget" in block
    assert "mnemosyne_scratchpad_*" in block
    assert "mnemosyne_shared_*" in block
    assert "Do not save one-off task progress" in block
