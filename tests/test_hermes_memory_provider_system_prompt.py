"""System prompt wording of the standalone Hermes provider (#835)."""

from hermes_memory_provider import MnemosyneMemoryProvider


def test_system_prompt_keeps_always_injected_memory_files_as_a_tier():
    provider = MnemosyneMemoryProvider()
    provider._beam = object()

    block = provider.system_prompt_block()

    assert "Mnemosyne is primary" in block
    assert "deprecated for bulk durable storage" in block
    assert "Small standing rules and context anchors may stay in the always-injected memory tier" in block
    assert "supported complementary tier (bounded, loaded per session)" in block
    assert "this exception takes precedence over general storage guidance" in block


def test_remember_tool_description_scopes_mnemosyne_to_bulk_storage():
    from hermes_memory_provider import REMEMBER_SCHEMA

    description = REMEMBER_SCHEMA["description"]

    assert "ANY fact" not in description
    assert "Use for bulk durable facts" in description
    assert "Small standing rules and context anchors may stay in the always-injected memory tier" in description
