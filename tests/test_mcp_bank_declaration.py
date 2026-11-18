"""
Tests for the per-call ``bank`` argument on the MCP tool surface.

The runtime has honoured ``arguments["bank"]`` since ``_resolve_bank`` was
introduced, but almost no schema declared it, so a conforming MCP client had
no way to discover the parameter and a client validating arguments against the
advertised schema could strip it. These tests pin both halves: that the
parameter is declared everywhere it is honoured, and that it actually
partitions data.

Run with: pytest tests/test_mcp_bank_declaration.py -v
"""



from mnemosyne.mcp_tools import TOOLS, get_tool_definitions, handle_tool_call
from mnemosyne.tool_schemas import (
    ALL_TOOL_SCHEMAS,
    BANK_EXEMPT_TOOLS,
    BANK_PROPERTY,
)


def _properties(schema):
    container = schema.get("parameters") or schema.get("inputSchema") or {}
    return container.get("properties", {})


class TestBankDeclaration:
    """Every tool that honours a tenant bank must advertise it."""

    def test_non_exempt_tools_declare_bank(self):
        missing = [
            s["name"]
            for s in ALL_TOOL_SCHEMAS
            if s["name"] not in BANK_EXEMPT_TOOLS and "bank" not in _properties(s)
        ]
        assert missing == [], (
            "These tools honour arguments['bank'] at runtime but do not declare "
            f"it, so clients cannot discover it: {missing}"
        )

    def test_exempt_tools_do_not_gain_a_tenant_bank(self):
        """The shared-surface tools must not imply an isolation they lack.

        The shared surface is a single global store. Advertising a tenant bank
        there would promise partitioning that does not exist.
        """
        for name in (
            "mnemosyne_shared_remember",
            "mnemosyne_shared_recall",
            "mnemosyne_shared_forget",
            "mnemosyne_shared_stats",
        ):
            schema = next(s for s in ALL_TOOL_SCHEMAS if s["name"] == name)
            assert "bank" not in _properties(schema), (
                f"{name} operates on the global shared surface; a tenant bank "
                "there would be a false isolation guarantee"
            )

    def test_validate_declares_store_and_a_tenant_bank(self):
        """``mnemosyne_validate`` selects private/surface via ``store``.

        ``bank`` is the tenant bank, as on every other tool. The schema spells
        it out itself so the description can also name the deprecated alias
        (``bank='private'|'surface'``), which the injector must not overwrite.
        """
        schema = next(
            s for s in ALL_TOOL_SCHEMAS if s["name"] == "mnemosyne_validate"
        )
        props = _properties(schema)
        assert props["store"]["enum"] == ["private", "surface"]
        assert "enum" not in props["bank"]
        assert "Deprecated" in props["bank"]["description"]
        assert props["bank"]["description"] != BANK_PROPERTY["description"]

    def test_declaration_survives_the_mcp_tools_rebuild(self):
        """TOOLS is rebuilt from ALL_TOOL_SCHEMAS; the property must carry."""
        remember = next(t for t in TOOLS if t["name"] == "mnemosyne_remember")
        assert "bank" in remember["input_schema"]["properties"]

    def test_advertised_tools_declare_bank(self):
        """``get_tool_definitions`` is what clients actually see."""
        advertised = get_tool_definitions()
        missing = [
            t["name"]
            for t in advertised
            if t["name"] not in BANK_EXEMPT_TOOLS
            and "bank" not in t["input_schema"].get("properties", {})
        ]
        assert missing == [], f"advertised without a bank parameter: {missing}"


class TestBankIsolation:
    """The guarantee the declaration is advertising."""

    def test_two_banks_do_not_see_each_other(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        monkeypatch.delenv("MNEMOSYNE_MCP_BANK", raising=False)

        handle_tool_call(
            "mnemosyne_remember",
            {"content": "alice belongs to tenant A", "bank": "tenant_a"},
        )
        handle_tool_call(
            "mnemosyne_remember",
            {"content": "bob belongs to tenant B", "bank": "tenant_b"},
        )

        def contents(result):
            items = result.get("memories") or result.get("results") or []
            return {
                (m.get("content") if isinstance(m, dict) else str(m)) for m in items
            }

        a = contents(
            handle_tool_call(
                "mnemosyne_recall", {"query": "belongs to tenant", "bank": "tenant_a"}
            )
        )
        b = contents(
            handle_tool_call(
                "mnemosyne_recall", {"query": "belongs to tenant", "bank": "tenant_b"}
            )
        )

        assert a, "tenant_a recalled nothing; the write did not land"
        assert b, "tenant_b recalled nothing; the write did not land"
        assert a.isdisjoint(b), (
            f"banks leaked into each other: tenant_a={a} tenant_b={b}"
        )


class TestDiagnoseBank:
    """``mnemosyne_diagnose`` used to ignore the bank and report the default."""

    def test_diagnose_reports_the_requested_bank(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        monkeypatch.delenv("MNEMOSYNE_MCP_BANK", raising=False)

        result = handle_tool_call("mnemosyne_diagnose", {"bank": "tenant_a"})
        assert result.get("bank") == "tenant_a"

    def test_unspecified_bank_stays_none(self, tmp_path, monkeypatch):
        """None and "default" are not the same database.

        ``run_diagnostics(bank=None)`` uses the profile-root DB while a named
        bank uses data/banks/<name>/. Collapsing unspecified to "default" would
        silently change which database an existing caller diagnoses.
        """
        monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        monkeypatch.delenv("MNEMOSYNE_MCP_BANK", raising=False)

        result = handle_tool_call("mnemosyne_diagnose", {})
        assert result.get("bank") is None

    def test_server_default_bank_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "server_default")

        result = handle_tool_call("mnemosyne_diagnose", {})
        assert result.get("bank") == "server_default"


def test_every_declared_bank_is_read_by_the_mcp_dispatcher():
    """A schema may only advertise ``bank`` if ``mcp_tools`` dispatches it.

    The persona, sync and ``mnemosyne_triple_end`` schemas live in
    ``tool_schemas`` for the Hermes provider and are not served over MCP, so
    ``_resolve_bank()`` never runs for them. Advertising ``bank`` there would
    promise routing that does not exist, and it also breaks the Hermes
    provider parity test, which pins the provider's persona schemas to these.
    """
    from pathlib import Path

    from mnemosyne import tool_schemas as ts

    dispatcher = Path(ts.__file__).with_name("mcp_tools.py").read_text()
    offenders = [
        s["name"]
        for s in ts.ALL_TOOL_SCHEMAS
        if "bank" in _properties(s)
        and s["name"] not in BANK_EXEMPT_TOOLS
        and f'"{s["name"]}"' not in dispatcher
        and f"'{s['name']}'" not in dispatcher
    ]
    assert offenders == [], offenders

