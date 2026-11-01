import asyncio
from unittest.mock import Mock

import pytest

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
    list_url = (
        "http://openwebui.test/api/v1/chats/?page=1"
        "&include_pinned=true&include_folders=true"
    )
    detail_url = "http://openwebui.test/api/v1/chats/chat%2Fone"
    next_page_url = (
        "http://openwebui.test/api/v1/chats/?page=2"
        "&include_pinned=true&include_folders=true"
    )
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


@pytest.mark.parametrize(
    ("chat_data", "message_id", "content"),
    [
        (
            {
                "chat": {
                    "history": {
                        "messages": [
                            {"id": "history-list", "content": "History list"}
                        ]
                    }
                }
            },
            "history-list",
            "History list",
        ),
        (
            {"history": [{"id": "top-history", "content": "Top history"}]},
            "top-history",
            "Top history",
        ),
        (
            {"messages": [{"id": "top-messages", "content": "Top messages"}]},
            "top-messages",
            "Top messages",
        ),
    ],
)
def test_sync_once_saves_supported_message_container_shapes(
    monkeypatch, chat_data, message_id, content
):
    """Save messages from each container shape supported by extraction."""
    list_url = (
        "http://openwebui.test/api/v1/chats/?page=1"
        "&include_pinned=true&include_folders=true"
    )
    detail_url = "http://openwebui.test/api/v1/chats/chat-one"
    next_page_url = (
        "http://openwebui.test/api/v1/chats/?page=2"
        "&include_pinned=true&include_folders=true"
    )
    responses = {
        list_url: [{"id": "chat-one", "title": "Test chat"}],
        detail_url: chat_data,
        next_page_url: [],
    }

    async def fake_api_get_async(url, api_key=""):
        """Return the fixture for the requested payload shape."""
        return responses[url]

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)
    engine = object.__new__(auto_save_openwebui.SyncEngine)
    engine.url = "http://openwebui.test"
    engine.api_key = "key"
    engine.saver, memory = _memory_saver(monkeypatch)

    result = asyncio.run(engine.sync_once())

    assert result["saved"] == 1
    assert memory.remember.call_args.kwargs["content"] == content
    assert auto_save_openwebui._PROCESSED_IDS == {f"chat-one:{message_id}"}


def test_sync_once_saves_assistant_message_after_stream_completes(monkeypatch):
    """Leave unfinished assistant IDs available for a later completed sync."""
    chat_id = "chat-one"
    list_url = (
        "http://openwebui.test/api/v1/chats/?page=1"
        "&include_pinned=true&include_folders=true"
    )
    detail_url = f"http://openwebui.test/api/v1/chats/{chat_id}"
    next_page_url = (
        "http://openwebui.test/api/v1/chats/?page=2"
        "&include_pinned=true&include_folders=true"
    )
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


@pytest.mark.parametrize(
    "value",
    [
        1_700_000_000,
        1_700_000_000_000,
        "1700000000",
        "2023-11-14T22:13:20Z",
    ],
)
def test_format_message_timestamp_supports_openwebui_formats(value):
    """Normalize numeric and ISO-8601 Open WebUI timestamps."""
    assert (
        auto_save_openwebui._format_message_timestamp(value)
        == "2023-11-14T22:13:20+00:00"
    )


def test_sync_once_isolates_invalid_message_timestamps(monkeypatch):
    """Continue to later chats when one message has a malformed timestamp."""
    list_url = (
        "http://openwebui.test/api/v1/chats/?page=1"
        "&include_pinned=true&include_folders=true"
    )
    next_page_url = (
        "http://openwebui.test/api/v1/chats/?page=2"
        "&include_pinned=true&include_folders=true"
    )
    responses = {
        list_url: [
            {"id": "malformed-chat", "title": "Malformed"},
            {"id": "valid-chat", "title": "Valid"},
        ],
        "http://openwebui.test/api/v1/chats/malformed-chat": {
            "messages": [
                {
                    "id": "malformed-message",
                    "content": "Do not save",
                    "timestamp": "not-a-timestamp",
                }
            ]
        },
        "http://openwebui.test/api/v1/chats/valid-chat": {
            "messages": [
                {
                    "id": "valid-message",
                    "content": "Save later chat",
                    "timestamp": "2023-11-14T22:13:20Z",
                }
            ]
        },
        next_page_url: [],
    }

    async def fake_api_get_async(url, api_key=""):
        """Return one malformed chat followed by a valid chat."""
        return responses[url]

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)
    engine = object.__new__(auto_save_openwebui.SyncEngine)
    engine.url = "http://openwebui.test"
    engine.api_key = "key"
    engine.saver, memory = _memory_saver(monkeypatch)

    result = asyncio.run(engine.sync_once())

    assert result["saved"] == 1
    assert result["total"]["errors"] == 1
    assert memory.remember.call_count == 1
    assert memory.remember.call_args.kwargs["content"] == "Save later chat"
    assert auto_save_openwebui._PROCESSED_IDS == {"valid-chat:valid-message"}


def test_sync_once_includes_pinned_folder_and_null_id_messages(monkeypatch):
    """Request excluded chat categories and preserve message_id fallbacks."""
    list_url = (
        "http://openwebui.test/api/v1/chats/?page=1"
        "&include_pinned=true&include_folders=true"
    )
    next_page_url = (
        "http://openwebui.test/api/v1/chats/?page=2"
        "&include_pinned=true&include_folders=true"
    )
    responses = {
        list_url: [
            {"id": "pinned-chat", "title": "Pinned"},
            {"id": "folder-chat", "title": "Folder"},
        ],
        "http://openwebui.test/api/v1/chats/pinned-chat": {
            "chat": {
                "messages": [
                    {
                        "id": None,
                        "message_id": "null-id-one",
                        "role": "user",
                        "content": "First null ID",
                    },
                    {
                        "id": None,
                        "message_id": "null-id-two",
                        "role": "assistant",
                        "content": "Second null ID",
                    },
                ]
            }
        },
        "http://openwebui.test/api/v1/chats/folder-chat": {
            "chat": {
                "messages": [
                    {
                        "id": "folder-message",
                        "role": "user",
                        "content": "Folder content",
                    }
                ]
            }
        },
        next_page_url: [],
    }

    async def fake_api_get_async(url, api_key=""):
        """Return pinned, folder, and null-ID fixtures."""
        return responses[url]

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)
    engine = object.__new__(auto_save_openwebui.SyncEngine)
    engine.url = "http://openwebui.test"
    engine.api_key = "key"
    engine.saver, memory = _memory_saver(monkeypatch)

    result = asyncio.run(engine.sync_once())

    assert result["saved"] == 3
    assert result["skipped"] == 0
    assert memory.remember.call_count == 3
    assert [call.kwargs["content"] for call in memory.remember.call_args_list] == [
        "First null ID",
        "Second null ID",
        "Folder content",
    ]
    assert auto_save_openwebui._PROCESSED_IDS == {
        "pinned-chat:null-id-one",
        "pinned-chat:null-id-two",
        "folder-chat:folder-message",
    }
