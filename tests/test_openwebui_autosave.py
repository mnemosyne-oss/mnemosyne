import asyncio
from unittest.mock import Mock

from mnemosyne.integrations import auto_save_openwebui


def _memory_saver(monkeypatch):
    """Create a MemorySaver with an in-memory Mnemosyne mock."""
    monkeypatch.setattr(auto_save_openwebui, "_PROCESSED_IDS", set())
    memory = Mock()
    saver = object.__new__(auto_save_openwebui.MemorySaver)
    saver._memory = memory
    saver._bank = "openwebui"
    saver._stats = {"saved": 0, "skipped": 0, "errors": 0}
    saver.persist = Mock()
    return saver, memory


def test_health_check_uses_openwebui_health_route(monkeypatch):
    """Use Open WebUI's JSON health endpoint."""
    requested = []

    async def fake_api_get_async(url, api_key=""):
        """Record a request and return a healthy response."""
        requested.append((url, api_key))
        return {"status": True}

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)

    assert asyncio.run(auto_save_openwebui._health_check("http://openwebui.test", "key"))
    assert requested == [("http://openwebui.test/health", "key")]


def test_sync_once_reads_current_openwebui_chat_payload(monkeypatch):
    """Save overlapping history and flat-list messages only once."""
    chat_id = "chat/one"
    list_url = "http://openwebui.test/api/v1/chats/?page=1"
    detail_url = "http://openwebui.test/api/v1/chats/chat%2Fone"
    next_page_url = "http://openwebui.test/api/v1/chats/?page=2"
    responses = {
        list_url: [{"id": chat_id, "title": "Test chat"}],
        detail_url: {
            "user_id": "webui-user",
            "chat": {
                "history": {
                    "messages": {
                        "user-message": {
                            "role": "user",
                            "content": "Remember this",
                            "timestamp": 1_700_000_000,
                        }
                    },
                    "currentId": "user-message",
                },
                "messages": [
                    {
                        "id": "user-message",
                        "role": "user",
                        "content": "Remember this",
                        "timestamp": 1_700_000_000,
                    },
                    {
                        "id": "legacy-message",
                        "role": "assistant",
                        "content": "Saved too",
                        "timestamp": 1_700_000_001,
                    }
                ],
            },
        },
        next_page_url: [],
    }
    requested = []

    async def fake_api_get_async(url, api_key=""):
        """Return the matching Open WebUI fixture response."""
        requested.append((url, api_key))
        return responses[url]

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)
    engine = object.__new__(auto_save_openwebui.SyncEngine)
    engine.url = "http://openwebui.test"
    engine.api_key = "key"
    engine.saver, memory = _memory_saver(monkeypatch)

    result = asyncio.run(engine.sync_once())

    assert requested == [
        (list_url, "key"),
        (detail_url, "key"),
        (next_page_url, "key"),
    ]
    assert result["saved"] == 2
    assert result["skipped"] == 0
    assert engine.saver.get_stats()["saved"] == 2
    assert memory.remember.call_count == 2
    assert [call.kwargs["content"] for call in memory.remember.call_args_list] == [
        "Remember this",
        "Saved too",
    ]
    assert all(
        call.kwargs["metadata"]["user_id"] == "webui-user"
        for call in memory.remember.call_args_list
    )


def test_sync_once_saves_assistant_message_after_stream_completes(monkeypatch):
    """Leave unfinished assistant IDs available for a later completed sync."""
    chat_id = "chat-one"
    list_url = "http://openwebui.test/api/v1/chats/?page=1"
    detail_url = f"http://openwebui.test/api/v1/chats/{chat_id}"
    next_page_url = "http://openwebui.test/api/v1/chats/?page=2"
    details = iter(
        [
            {
                "chat": {
                    "messages": [
                        {
                            "id": "assistant-message",
                            "role": "assistant",
                            "content": "Partial answer",
                            "done": False,
                        }
                    ]
                }
            },
            {
                "chat": {
                    "messages": [
                        {
                            "id": "assistant-message",
                            "role": "assistant",
                            "content": "Complete answer",
                            "done": True,
                        }
                    ]
                }
            },
        ]
    )

    async def fake_api_get_async(url, api_key=""):
        """Return the unfinished then completed chat payloads."""
        if url == list_url:
            return [{"id": chat_id, "title": "Test chat"}]
        if url == detail_url:
            return next(details)
        if url == next_page_url:
            return []
        raise AssertionError(f"Unexpected request: {url}")

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)
    engine = object.__new__(auto_save_openwebui.SyncEngine)
    engine.url = "http://openwebui.test"
    engine.api_key = "key"
    engine.saver, memory = _memory_saver(monkeypatch)

    first_sync = asyncio.run(engine.sync_once())

    assert first_sync["saved"] == 0
    assert first_sync["skipped"] == 0
    assert "chat-one:assistant-message" not in auto_save_openwebui._PROCESSED_IDS
    assert memory.remember.call_count == 0

    second_sync = asyncio.run(engine.sync_once())

    assert second_sync["saved"] == 1
    assert second_sync["skipped"] == 0
    assert engine.saver.get_stats()["saved"] == 1
    assert memory.remember.call_count == 1
    assert memory.remember.call_args.kwargs["content"] == "Complete answer"
