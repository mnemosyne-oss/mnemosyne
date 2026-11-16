"""Consolidation must preserve facts and fail back for incomplete chunks."""
import pytest
from mnemosyne.core import local_llm as llm
from mnemosyne.core.llm_backends import CallableLLMBackend, set_host_llm_backend


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(llm,"LLM_ENABLED",True)
    monkeypatch.setattr(llm,"HOST_LLM_ENABLED",True)
    monkeypatch.setattr(llm,"HOST_LLM_N_CTX",512)
    monkeypatch.setattr(llm,"LLM_MAX_TOKENS",128)
    monkeypatch.setattr(llm,"SLEEP_PROMPT","")
    monkeypatch.setattr(llm,"LLM_BASE_URL","https://stale.example.test/v1")
    def forbidden(*args,**kwargs):
        pytest.fail("an attempted host call must not fall through to the remote URL")
    monkeypatch.setattr(llm,"_call_remote_llm",forbidden)
    monkeypatch.setattr(llm,"_call_local_llm",lambda prompt:None)
    def install(fn):
        set_host_llm_backend(CallableLLMBackend("test",fn))
    set_host_llm_backend(CallableLLMBackend("initial",lambda *a,**k:None))
    yield install
    set_host_llm_backend(None)


@pytest.mark.parametrize("text",[
    "The following memories confirm Alex paid $14.99.",
    "These memories describe Alex's holiday in 2026.",
    "The memories shared by Alex include a preference for metric units.",
])
def test_host_summary_preserves_factual_sentences(host,text):
    host(lambda *a,**k:text)
    assert llm.summarize_memories(["A stored fact"])==text


def test_empty_sanitized_host_output_uses_local_fallback(host,monkeypatch):
    host(lambda *a,**k:"Memories:")
    monkeypatch.setattr(llm,"_call_local_llm",lambda prompt:"Alex paid $14.99.")
    assert llm.summarize_memories(["Alex paid $14.99."])=="Alex paid $14.99."


def test_local_fallback_uses_same_consolidation_sanitizer(host,monkeypatch):
    host(lambda *a,**k:None)
    monkeypatch.setattr(llm,"_call_local_llm",lambda prompt:"1. Memories: Alex bought a GPU.")
    assert llm.summarize_memories(["Alex bought a GPU."])=="Alex bought a GPU."


def test_exact_distill_instruction_echo_is_removed(host):
    echo=("Distill the following memories into 1-3 concise sentences of plain fact. "
          "Preserve names, numbers, dates, decisions, preferences, and outcomes. "
          "Do not address the reader. Do not describe what you are doing. "
          "Do not restate these instructions. Do not number the sentences. "
          "Output only the distilled fact.")
    host(lambda *a,**k:echo+"\nAlex prefers metric units.")
    assert llm.summarize_memories(["Alex prefers metric units."])=="Alex prefers metric units."


def test_one_failed_chunk_does_not_publish_only_the_other_chunk(host):
    memories=["ALICE "*75,"BOB "*75]
    assert len(llm.chunk_memories_by_budget(memories))==2
    def reply(prompt,**kwargs):
        return None if "ALICE" in prompt else "Bob likes pizza."
    host(reply)
    assert llm.summarize_memories(memories) is None


def test_failed_merge_does_not_publish_only_the_first_summary(host):
    memories=["ALICE "*75,"BOB "*75]
    assert len(llm.chunk_memories_by_budget(memories))==2
    def reply(prompt,**kwargs):
        if "[chunked" in prompt:return None
        return "Alice likes tea." if "ALICE" in prompt else "Bob likes pizza."
    host(reply)
    assert llm.summarize_memories(memories) is None


def test_successful_multichunk_summary_keeps_both_facts(host):
    memories=["ALICE "*75,"BOB "*75]
    assert len(llm.chunk_memories_by_budget(memories))==2
    def reply(prompt,**kwargs):
        if "[chunked" in prompt:return "Alice likes tea. Bob likes pizza."
        return "Alice likes tea." if "ALICE" in prompt else "Bob likes pizza."
    host(reply)
    assert llm.summarize_memories(memories)=="Alice likes tea. Bob likes pizza."


def test_oversized_skipped_input_does_not_allow_a_partial_summary(host):
    host(lambda *a,**k:"Bob likes pizza.")
    assert llm.summarize_memories(["ALICE "*1000,"Bob likes pizza."]) is None


def test_host_fact_bullets_are_preserved(host):
    text="- Alex prefers metric units\n- Bedtime is 20:30"
    host(lambda *a,**k:text)
    assert llm.summarize_memories(["A fact"])==text
