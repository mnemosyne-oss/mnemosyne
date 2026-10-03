"""``mnemosyne_validate`` over MCP: ``store`` selects private/surface and
``bank`` is the tenant bank, with the pre-4.0 ``bank='private'|'surface'``
spelling honoured as a deprecated alias."""

import pytest

from mnemosyne import mcp_tools


@pytest.fixture
def two_banks(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_HOME", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.delenv("MNEMOSYNE_MCP_BANK", raising=False)
    a = mcp_tools._create_instance(bank="tenant-a")
    b = mcp_tools._create_instance(bank="tenant-b")
    mid_a = a.remember("alpha fact lives in tenant a")
    mid_b = b.remember("beta fact lives in tenant b")
    return mid_a, mid_b


def test_resolve_target_defaults_and_alias(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_MCP_BANK", raising=False)
    assert mcp_tools._resolve_validate_target({}) == ("private", "default", False)
    assert mcp_tools._resolve_validate_target({"bank": "tenant-x"}) == ("private", "tenant-x", False)
    assert mcp_tools._resolve_validate_target({"bank": "surface"}) == ("surface", None, True)
    assert mcp_tools._resolve_validate_target({"bank": "private"}) == ("private", "default", True)
    assert mcp_tools._resolve_validate_target({"store": "surface", "bank": "tenant-x"}) == ("surface", None, False)
    monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "srv")
    assert mcp_tools._resolve_validate_target({"store": "private"}) == ("private", "srv", False)


def test_validate_routes_to_the_requested_tenant_bank(two_banks):
    mid_a, mid_b = two_banks
    hit = mcp_tools._handle_validate({"memory_id": mid_a, "action": "attest", "bank": "tenant-a"})
    assert hit["status"] == "validation_attest"
    assert hit["store"] == "private" and hit["bank"] == "tenant-a"
    assert "deprecated" not in hit
    miss = mcp_tools._handle_validate({"memory_id": mid_a, "action": "attest", "bank": "tenant-b"})
    assert miss["error"] == "memory_not_found"
    assert miss["bank"] == "tenant-b"


def test_validate_bank_private_alias_is_honoured_and_flagged(two_banks, monkeypatch):
    mid_a, _ = two_banks
    monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "tenant-a")
    res = mcp_tools._handle_validate({"memory_id": mid_a, "action": "attest", "bank": "private"})
    assert res["status"] == "validation_attest"
    assert res["store"] == "private" and res["bank"] == "tenant-a"
    assert "5.0" in res["deprecated"]


def test_validate_unknown_store_is_rejected():
    res = mcp_tools._handle_validate({"memory_id": "x", "action": "attest", "store": "weird"})
    assert res["error"] == "unknown store: weird"
