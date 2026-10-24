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


def test_system_prompt_tells_agent_to_read_mnemosyne_context_first():
    """Pin the rule added in #321.

    The Mnemosyne context block is injected into the turn and the agent
    should answer from it before reaching for session_search. Without
    this standing rule, the agent defaults to retrieval tools out of
    habit and wastes a turn.
    """
    provider = MnemosyneMemoryProvider()
    provider._beam = object()

    block = provider.system_prompt_block()

    assert "## Mnemosyne Context" in block
    assert "read it before calling retrieval tools" in block
    assert "answer directly" in block
    assert "session_search" in block
    assert "missing, stale, or insufficient" in block


def test_system_prompt_keeps_always_injected_memory_files_as_a_tier():
    """Pin the qualification added in #835.

    The deprecation targets bulk durable storage. Small standing rules and
    context anchors may stay in the always-injected tier, and that exception
    outranks the general storage guidance.
    """
    provider = MnemosyneMemoryProvider()
    provider._beam = object()

    block = provider.system_prompt_block()

    assert "deprecated for bulk durable storage" in block
    assert "Small standing rules and context anchors may stay in the always-injected memory tier" in block
    assert "supported complementary tier (bounded, loaded per session)" in block
    assert "this exception takes precedence over general storage guidance" in block


def test_remember_tool_description_scopes_mnemosyne_to_bulk_storage():
    """The advertised remember tool no longer claims ANY durable fact (#835)."""
    from mnemosyne_hermes.tools import REMEMBER_SCHEMA

    description = REMEMBER_SCHEMA["description"]

    assert "ANY fact" not in description
    assert "Use for bulk durable facts" in description
    assert "Small standing rules and context anchors may stay in the always-injected memory tier" in description


def test_bundled_override_skill_requires_conscious_user_selection():
    """The strict override describes itself as user-selected, not default (#835)."""
    from mnemosyne_hermes.install import bundled_skill_text

    skill = bundled_skill_text()
    frontmatter = skill.split("---", 2)[1]

    assert "Stricter, user-selected override" in frontmatter
    assert "Apply it only when the user has\n  consciously chosen this policy" in frontmatter
    assert "Only when the user has explicitly selected the strict Mnemosyne-only memory" in frontmatter
    assert "## Activation and precedence" in skill
    assert "Under the default, small standing rules\nand context anchors stay in the always-injected tier" in skill
