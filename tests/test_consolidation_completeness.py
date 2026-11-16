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
        if "[chunked" in prompt:
            assert "Alice likes tea." in prompt
            assert "Bob likes pizza." in prompt
            return "Alice likes tea. Bob likes pizza."
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


@pytest.mark.parametrize("backend", ["direct-local", "remote"])
@pytest.mark.parametrize("case", ["header", "instruction", "fact", "empty"])
def test_summary_sanitization_uses_actual_backend_paths(monkeypatch, backend, case):
    """Both non-host branches sanitize summaries and preserve factual text."""
    import httpx
    echo=("Distill the following memories into 1-3 concise sentences of plain fact. "
          "Preserve names, numbers, dates, decisions, preferences, and outcomes. "
          "Do not address the reader. Do not describe what you are doing. "
          "Do not restate these instructions. Do not number the sentences. "
          "Output only the distilled fact.")
    fact="The following memories confirm Alex paid $14.99."
    raw={"header":"1. Memories: Alex paid $14.99.","instruction":echo+"\nAlex paid $14.99.","fact":fact,"empty":"Memories:"}[case]
    expected={"header":"Alex paid $14.99.","instruction":"Alex paid $14.99.","fact":fact,"empty":None}[case]
    monkeypatch.setattr(llm,"LLM_ENABLED",True)
    monkeypatch.setattr(llm,"HOST_LLM_ENABLED",False)
    monkeypatch.setattr(llm,"LLM_API_KEY","")
    monkeypatch.setattr(llm,"LLM_EXTRA_BODY",{})
    monkeypatch.setattr(llm,"LLM_FALLBACK_MODELS",[])
    monkeypatch.delenv("MNEMOSYNE_FORCE_LOCAL",raising=False)
    if backend=="direct-local":
        monkeypatch.setattr(llm,"LLM_BASE_URL","")
        monkeypatch.setattr(llm,"_call_local_llm",lambda prompt:raw)
    else:
        monkeypatch.setattr(llm,"LLM_BASE_URL","http://summary.example.test/v1")
        original=httpx.Client
        def respond(request):
            assert request.url.host=="summary.example.test"
            return httpx.Response(200,json={"choices":[{"message":{"content":raw}}]})
        monkeypatch.setattr(httpx,"Client",lambda **kwargs:original(transport=httpx.MockTransport(respond),**kwargs))
        monkeypatch.setattr(llm,"_call_local_llm",lambda prompt:"Local recovery includes Alex's complete fact.")
        if case=="empty":expected="Local recovery includes Alex's complete fact."
    assert llm.summarize_memories(["Alex paid $14.99."])==expected


@pytest.mark.parametrize("scenario",["oversized","failed-first","failed-last","failed-merge","success"])
def test_sleep_keeps_all_sources_when_budget_or_summary_fails(host,monkeypatch,tmp_path,scenario):
    """Real sleep persistence falls back over every source when any part is lost."""
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core import model_refresh
    monkeypatch.setattr(model_refresh,"infer_model_update_proposals",lambda items:[])
    first="ALICE_ORIGINAL " + "ALICE "*(1000 if scenario=="oversized" else 75)
    second="BOB_ORIGINAL " + "BOB "*75
    if scenario=="oversized":
        assert sum(map(len,llm.chunk_memories_by_budget([first,second])))==1
    else:
        assert len(llm.chunk_memories_by_budget([first,second]))==2
    def reply(prompt,**kwargs):
        if "(consolidated)" in prompt:
            assert "Alice likes tea." in prompt
            assert "Bob likes pizza." in prompt
            return None if scenario=="failed-merge" else "Alice likes tea. Bob likes pizza."
        if "ALICE_ORIGINAL" in prompt:
            return None if scenario=="failed-first" else "Alice likes tea."
        if "BOB_ORIGINAL" in prompt:
            return None if scenario=="failed-last" else "Bob likes pizza."
        pytest.fail("unexpected summary prompt")
    host(reply)
    beam=BeamMemory(db_path=tmp_path/"sleep-completeness.db",session_id="sleep-completeness")
    try:
        beam.conn.executemany(
            "INSERT INTO working_memory (id,content,source,timestamp,session_id) VALUES (?,?,?,?,?)",
            [("first",first,"conversation","2000-01-01T00:00:00","sleep-completeness"),
             ("second",second,"conversation","2000-01-02T00:00:00","sleep-completeness")])
        beam.conn.commit()
        result=beam.sleep(force=True)
        assert result["status"]=="consolidated"
        assert result["items_consolidated"]==2
        summaries=beam.conn.execute("SELECT content FROM episodic_memory").fetchall()
        assert len(summaries)==1
        if scenario=="success":
            assert summaries[0][0]=="Alice likes tea. Bob likes pizza."
        else:
            assert "ALICE_ORIGINAL" in summaries[0][0]
            assert "BOB_ORIGINAL" in summaries[0][0]
        rows=beam.conn.execute("SELECT consolidated_at,consolidation_claimed_at FROM working_memory WHERE id IN ('first','second')").fetchall()
        assert len(rows)==2 and all(row[0] is not None and row[1] is None for row in rows)
    finally:
        beam.conn.close()
