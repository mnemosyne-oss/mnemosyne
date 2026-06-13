from __future__ import annotations

import json

from mnemosyne_hermes import MnemosyneMemoryProvider


def _provider(tmp_path, profile: str = "profile_a") -> MnemosyneMemoryProvider:
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        agent_context="primary",
        agent_identity=profile,
    )
    assert provider._beam is not None
    return provider


def _close(provider: MnemosyneMemoryProvider) -> None:
    try:
        provider._beam.conn.close()
    except Exception:
        pass


def test_active_adapter_canonical_tools_roundtrip(tmp_path):
    provider = _provider(tmp_path, profile="profile_a")
    try:
        stored = json.loads(provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {
                "category": "identity",
                "name": "name",
                "body": "My name is Profile A.",
                "source": "test",
                "confidence": 0.9,
            },
        ))

        assert stored["status"] == "created"
        assert stored["owner_id"] == "profile_a"
        assert stored["category"] == "identity"
        assert stored["name"] == "name"
        assert stored["version"] == 1

        recalled = json.loads(provider.handle_tool_call(
            "mnemosyne_recall_canonical",
            {"category": "identity", "name": "name"},
        ))

        assert recalled["mode"] == "recall"
        assert recalled["owner_id"] == "profile_a"
        assert recalled["found"] is True
        assert recalled["result"]["body"] == "My name is Profile A."
    finally:
        _close(provider)


def test_active_adapter_canonical_tools_owner_scoped(tmp_path):
    provider = _provider(tmp_path, profile="profile_a")
    try:
        provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "identity", "name": "name", "body": "I am profile A."},
        )
        provider._agent_identity = "profile_b"

        recalled = json.loads(provider.handle_tool_call(
            "mnemosyne_recall_canonical",
            {"category": "identity", "name": "name"},
        ))

        assert recalled["owner_id"] == "profile_b"
        assert recalled["found"] is False
    finally:
        _close(provider)


def test_active_adapter_canonical_tools_search_history_and_list(tmp_path):
    provider = _provider(tmp_path, profile="profile_a")
    try:
        provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "voice", "name": "register", "body": "icy and terse"},
        )
        provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "voice", "name": "register", "body": "warm and direct"},
        )

        search = json.loads(provider.handle_tool_call(
            "mnemosyne_recall_canonical", {"query": "warm"},
        ))
        assert search["mode"] == "search"
        assert search["count"] == 1
        assert search["results"][0]["body"] == "warm and direct"

        history = json.loads(provider.handle_tool_call(
            "mnemosyne_recall_canonical",
            {"category": "voice", "name": "register", "include_history": True},
        ))
        assert history["mode"] == "history"
        assert history["count"] == 2

        listed = json.loads(provider.handle_tool_call(
            "mnemosyne_recall_canonical", {"category": "voice"},
        ))
        assert listed["mode"] == "list"
        assert listed["count"] == 1
        assert listed["results"][0]["body"] == "warm and direct"
    finally:
        _close(provider)


def test_active_adapter_canonical_tools_validate_required_fields(tmp_path):
    provider = _provider(tmp_path, profile="profile_a")
    try:
        missing_body = json.loads(provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "identity", "name": "name"},
        ))
        assert missing_body == {"error": "body is required"}

        missing_slot = json.loads(provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "identity", "body": "body"},
        ))
        assert missing_slot == {"error": "category and name are required"}
    finally:
        _close(provider)


def test_active_adapter_exposes_canonical_tool_schemas():
    provider = MnemosyneMemoryProvider()
    names = {schema["name"] for schema in provider.get_tool_schemas()}

    assert "mnemosyne_remember_canonical" in names
    assert "mnemosyne_recall_canonical" in names
