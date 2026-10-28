import mnemosyne.core.local_llm as local_llm


def test_strips_preamble_keeps_fact():
    text = "The following memories are concise and brief. Alex prefers metric units."
    out = local_llm._sanitize_consolidation_output(text)
    assert out == "Alex prefers metric units."


def test_strips_numbered_memories_prefix():
    text = "1. Memories: Alex bought a GPU on 2026-10-20 for EUR 640."
    out = local_llm._sanitize_consolidation_output(text)
    assert not out.startswith("1.")
    assert "Memories:" not in out
    assert "Alex bought a GPU on 2026-10-20 for EUR 640." in out


def test_meta_only_returns_empty():
    assert local_llm._sanitize_consolidation_output("Memories:") == ""


def test_short_meta_only_returns_empty():
    assert local_llm._sanitize_consolidation_output("The following memories") == ""


def test_plain_fact_untouched():
    text = "Alex's daughter's name is Mia; bedtime is 20:30."
    assert local_llm._sanitize_consolidation_output(text) == text


def test_bullets_preserved():
    text = "- Alex prefers metric units\n- Bedtime is 20:30"
    out = local_llm._sanitize_consolidation_output(text)
    assert "- Alex" in out
    assert "Bedtime is 20:30" in out


def test_preamble_line_removed_next_line_kept():
    text = "The following memories are concise and brief.\nAlex prefers metric units."
    out = local_llm._sanitize_consolidation_output(text)
    assert "The following memories" not in out
    assert out == "Alex prefers metric units."


def test_host_prompt_constrains_output():
    prompt = local_llm._build_host_prompt(["x"], source="sleep")
    assert "Output only the distilled fact." in prompt
    assert "Summarize the following memories" not in prompt
