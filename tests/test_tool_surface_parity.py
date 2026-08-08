"""Guards on the MCP tool surface.

Every one of these encodes a real defect found in the tree, not a
hypothetical. The counts published across three repositories had drifted to
six different numbers (6, 15, 17, 23, 25, 37) because nothing checked them,
and one tool announced in a release was never actually exposed.

These tests are deliberately structural: they compare declarations against
each other rather than asserting a magic number, so adding a tool the right
way keeps them green and adding one the wrong way does not.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent


def _plugin_yaml_tools():
    """provides_tools from plugin.yaml, parsed without requiring PyYAML."""
    text = (REPO / "hermes_memory_provider" / "plugin.yaml").read_text()
    block = text.split("provides_tools:")[1].split("provides_hooks:")[0]
    return re.findall(r"^\s*-\s*(mnemosyne_\w+)\s*$", block, re.M)


def test_every_defined_schema_is_exported():
    """A *_SCHEMA that is not in ALL_TOOL_SCHEMAS is invisible over MCP.

    mnemosyne_forget_canonical was defined and shipped in a release note but
    omitted from ALL_TOOL_SCHEMAS, so it was never advertised. If a tool is
    deliberately provider-only, add it to PROVIDER_ONLY below with a reason.
    """
    from mnemosyne import tool_schemas

    # Tools intentionally not advertised over MCP, with justification.
    PROVIDER_ONLY = {
        # Implemented in hermes_memory_provider only; no MCP handler exists.
        "mnemosyne_forget_canonical",
    }

    exported = {s["name"] for s in tool_schemas.ALL_TOOL_SCHEMAS}
    defined = {
        v["name"]
        for k, v in vars(tool_schemas).items()
        if k.endswith("_SCHEMA") and isinstance(v, dict) and "name" in v
    }
    missing = defined - exported - PROVIDER_ONLY
    assert not missing, (
        f"schema(s) defined but absent from ALL_TOOL_SCHEMAS, so not advertised "
        f"over MCP: {sorted(missing)}. Add them to ALL_TOOL_SCHEMAS, or to "
        f"PROVIDER_ONLY in this test with a reason."
    )


def test_every_mcp_handler_has_a_schema():
    """A handler with no schema can never be called: it is not advertised."""
    from mnemosyne import mcp_tools, tool_schemas

    exported = {s["name"] for s in tool_schemas.ALL_TOOL_SCHEMAS}
    handlers = set(mcp_tools._TOOL_HANDLERS)
    orphans = handlers - exported
    assert not orphans, f"handlers with no advertised schema: {sorted(orphans)}"


def test_plugin_yaml_tools_all_have_schemas():
    """plugin.yaml told Hermes about tools that must actually exist."""
    from mnemosyne import tool_schemas

    known = {
        v["name"]
        for k, v in vars(tool_schemas).items()
        if k.endswith("_SCHEMA") and isinstance(v, dict) and "name" in v
    }
    declared = _plugin_yaml_tools()
    assert declared, "could not parse provides_tools from plugin.yaml"
    unknown = [t for t in declared if t not in known]
    assert not unknown, f"plugin.yaml declares tools with no schema: {unknown}"


def test_no_tool_is_unreachable():
    """Every advertised tool must be callable somewhere.

    Either it has an MCP handler, or the Hermes provider implements it. A
    schema reachable through neither is advertised to clients that can only
    ever receive an error.
    """
    from mnemosyne import mcp_tools, tool_schemas

    handlers = set(mcp_tools._TOOL_HANDLERS)
    provider_src = ""
    for path in (REPO / "hermes_memory_provider").rglob("*.py"):
        provider_src += path.read_text()

    unreachable = [
        s["name"]
        for s in tool_schemas.ALL_TOOL_SCHEMAS
        if s["name"] not in handlers and s["name"] not in provider_src
    ]
    assert not unreachable, (
        f"advertised but callable nowhere: {unreachable}"
    )


def test_generated_docs_match_the_code():
    """The docs-check gate, run as a unit test too.

    docs-check verified nothing for a long time because it skipped when a
    sibling repo was absent, which is always true on the runner. Running the
    same comparison here means a stale generated doc fails the normal test
    job as well.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gendocs", REPO / "scripts" / "generate-docs.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    version = gen._version()
    tools = gen._collect_tools()
    env_map, defaults, restart = gen._collect_config()
    gen._validate_descriptions(env_map)
    effective = gen._scan_effective_defaults(env_map, defaults)

    expected = {
        "docs/api/tool-schema.mdx": gen._render_tool_schema(tools, version),
        "docs/api/configuration.mdx": gen._render_config(
            env_map, defaults, restart, version, effective
        ),
    }
    for rel, want in expected.items():
        path = REPO / rel
        assert path.is_file(), f"{rel} is missing; run scripts/generate-docs.py"
        assert path.read_text() == want, (
            f"{rel} is out of date. Run: python3 scripts/generate-docs.py"
        )


def test_type_renderer_escapes_union_pipes_for_markdown_tables():
    """JSON Schema unions must not add columns to generated pipe tables."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gendocs", REPO / "scripts" / "generate-docs.py"
    )
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    assert gen._type_of({"type": ["number", "null"]}) == "number \\| null"
    assert gen._type_of({"anyOf": [{"type": "number"}, {"type": "null"}]}) == (
        "number \\| null"
    )


def test_generated_mcp_docs_describe_current_transport_and_policy_contract():
    """PR #599's generated reference must name Streamable HTTP and tri-state policy."""
    tool_docs = (REPO / "docs/api/tool-schema.mdx").read_text()
    config_docs = (REPO / "docs/api/configuration.mdx").read_text()
    assert "(stdio, SSE, and Streamable HTTP)" in tool_docs
    assert "configured server policy overrides the caller" in tool_docs
    assert "MNEMOSYNE_MCP_DEFAULT_EXTRACT_ENTITIES" in config_docs
    assert "strict 1/true/yes/on or 0/false/no/off" in config_docs


@pytest.mark.parametrize("module", ["mnemosyne.mcp_server", "mnemosyne.mcp_tools"])
def test_mcp_modules_import(module):
    """Catch an incompatible mcp SDK at test time rather than at first use.

    The SDK 1.x to 2.x migration removed Server.list_tools and renamed
    Tool.inputSchema. While the dependency was unbounded, a fresh install
    could resolve an SDK the server could not start against, and nothing
    failed until a user ran `mnemosyne mcp`.
    """
    __import__(module)


def test_tool_definitions_are_constructible():
    """get_tool_definitions() must produce dicts the installed Tool accepts."""
    mcp_types = pytest.importorskip("mcp.types")
    from mnemosyne.mcp_tools import get_tool_definitions

    defs = get_tool_definitions()
    assert defs, "no tool definitions produced"
    for d in defs:
        mcp_types.Tool(**d)
