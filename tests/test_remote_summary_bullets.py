"""Remote summaries preserve Markdown facts without bypassing output cleanup."""

from unittest.mock import MagicMock

import httpx
import pytest

from mnemosyne.core import local_llm


@pytest.fixture
def remote_answer(monkeypatch):
    """Stub only HTTP transport, retaining remote parsing and summary cleanup."""
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", False)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://summary.invalid/v1")
    monkeypatch.setattr(local_llm, "LLM_API_KEY", "")
    monkeypatch.setattr(local_llm, "LLM_EXTRA_BODY", {})
    monkeypatch.delenv("MNEMOSYNE_FORCE_LOCAL", raising=False)
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)

    def answer(text):
        client.post.return_value = httpx.Response(
            200, json={"choices": [{"message": {"content": text}}]}
        )
        return client.post

    return answer


@pytest.mark.parametrize("marker", ["-", "*"])
@pytest.mark.parametrize("trailing_newline", ["", "\n"])
@pytest.mark.parametrize("indent", ["", "  "])
def test_remote_summary_preserves_bulleted_facts(remote_answer, marker, trailing_newline, indent):
    facts = [
        "The user prefers dark mode in every editor.",
        "The weekly team meeting is on Tuesday at 10:00.",
        "The project database is PostgreSQL 16.",
    ]
    raw = "Key facts:\n" + "\n".join(f"{indent}{marker} {fact}" for fact in facts)
    post = remote_answer(raw + trailing_newline)

    # Input repeats are legitimate facts too, not automatically prompt echoes.
    assert local_llm.summarize_memories(facts) == raw
    post.assert_called_once()


def test_remote_summary_retains_nested_bullets_and_other_cleanup(remote_answer):
    expected = "Key facts:\n- Editors:\n  * Dark mode.\n\n- Database: PostgreSQL 16."
    post = remote_answer(
        "<think>Private reasoning.</think><|assistant|>"
        "Summarize the following memories. Preserve facts. Source: synthetic\n"
        + expected + "</s>\n<|user|>"
    )

    assert local_llm.summarize_memories(["Dark mode", "PostgreSQL 16"]) == expected
    post.assert_called_once()


@pytest.mark.parametrize("raw", [
    "<think>truncated reasoning\n- Not a fact.",
    "<think>outer<think>inner</think>outer</think>\n- Not a fact.",
    "- Not a fact.</think>",
])
def test_remote_summary_rejects_malformed_reasoning(remote_answer, raw):
    post = remote_answer(raw)

    assert local_llm.summarize_memories(["A memory."]) is None
    assert local_llm._is_invalid_reasoning_output(
        local_llm._summarize_memories(["A memory."])
    )
    assert post.call_count == 2


@pytest.mark.parametrize("path", ["local", "remote-fallback", "host-fallback"])
def test_local_summary_still_removes_bullet_echoes(monkeypatch, remote_answer, path):
    post = remote_answer("unused")
    raw = "Key facts:\n- Echoed memory one.\n* Echoed memory two.\nLocal summary."
    if path == "local":
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "")
    elif path == "remote-fallback":
        post.return_value = httpx.Response(400, request=httpx.Request("POST", "http://summary.invalid"))
    else:
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        from mnemosyne.core.llm_backends import CallableLLMBackend, set_host_llm_backend

        set_host_llm_backend(CallableLLMBackend("empty", lambda prompt, **kwargs: ""))
    local = MagicMock(return_value=raw)
    monkeypatch.setattr(local_llm, "_call_local_llm", local)

    assert local_llm.summarize_memories(["A memory."]) == "Key facts:\nLocal summary."
    local.assert_called_once()
    assert post.call_count == (1 if path == "remote-fallback" else 0)


def test_sleep_persists_all_remote_bullet_facts(remote_answer, tmp_path):
    import sqlite3

    from mnemosyne.core.beam import BeamMemory

    facts = ["The user prefers dark mode.", "The meeting is Tuesday at 10:00.", "The database is PostgreSQL 16."]
    summary = "Key facts:\n" + "\n".join(f"- {fact}" for fact in facts)
    post = remote_answer(summary + "\n")
    # Sleep also requests canonical model-slot updates; keep that separate
    # inference inert rather than feeding summary prose to its JSON parser.
    post.side_effect = [post.return_value, httpx.Response(
        200, json={"choices": [{"message": {"content": "[]"}}]}
    )]
    db_path = tmp_path / "bullets.db"
    beam = BeamMemory(session_id="remote-bullets", db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [(f"bullet-{i}", fact, "conversation", "2000-01-01T00:00:00", "remote-bullets")
             for i, fact in enumerate(facts)],
        )

    result = beam.sleep(dry_run=False)

    assert result["status"] == "consolidated"
    assert result["items_consolidated"] == len(facts)
    assert post.call_count == 2
    with sqlite3.connect(db_path) as conn:
        persisted = conn.execute("SELECT content FROM episodic_memory").fetchall()
    assert len(persisted) == 1
    for fact in facts:
        assert fact in persisted[0][0]
