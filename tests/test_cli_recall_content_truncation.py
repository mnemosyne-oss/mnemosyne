"""Regression: `mnemosyne recall` must not silently truncate record content.

The reporter in #685 could not tell a 150-char preview from a complete record:
the line ended in `...` with nothing saying the record was cut. Public recall
and JSON retain the backend content cap; human display can read selected stored
or MEMORIA content without changing that bounded payload contract.

These tests pin the property that matters: when the displayed content is shorter
than the stored content, the output must say so, and a full read must be available
without changing the default view.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

import pytest

LONG = (
    "Network infrastructure (three local networks):\n"
    "- 192.168.88.0/25 — main network: gw .88.1 (main Mikrotik)\n"
    "- 192.168.1.0/24 — second network: gw.itj .1.254\n"
    "- 192.168.2.0/24 — third network: gw.itj .2.253\n\n"
    "Сетевая инфраструктура (три локальные сети):\n"
    "- 192.168.88.0/25 — основная сеть: gw .88.1\n"
    "- 192.168.1.0/24 — вторая сеть: gw.itj .1.254\n"
    "- 192.168.2.0/24 — третья сеть: gw.itj .2.253"
)


def _render_recall(results: list[dict]) -> str:
    """Run the real CLI render path and capture what the user sees."""
    from mnemosyne import cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._print_recall_results("network", results)
    return buf.getvalue()


def test_long_record_is_not_truncated_at_the_display_cap():
    out = _render_recall([{"id": "a6dc2c7371307e57", "content": LONG, "score": 0.738}])
    assert "...\n" not in out.replace("192.168", "")  # no ellipsis-only body
    assert "192.168.2.0/24" in out, "the third network is dropped from the output"


def test_truncation_is_announced_when_a_preview_cap_applies():
    """With --preview the cap may apply, but the output must name what was withheld."""
    from mnemosyne import cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._print_recall_results("network", [{"id": "x", "content": "y" * 4000, "score": 0.5}], preview=True)
    out = buf.getvalue()
    assert "showing 150 of 4000" in out, out[-200:]
    assert not out.rstrip().endswith("..."), "a bare ellipsis still hides that the record was cut"


def _run_cmd_recall(monkeypatch, args):
    """Drive the real command entry point with a stubbed memory backend.

    The renderer tests above cover the content and the character counts, but
    they cannot catch a break in how `--preview` is parsed or forwarded. This
    exercises cmd_recall end to end so that wiring is covered too.
    """
    from mnemosyne import cli

    captured = {}

    class _StubMemory:
        def recall(self, query, top_k=5, explain=False):
            captured["called"] = (query, top_k, explain)
            return [{"id": "x", "content": LONG, "score": 0.5}]

    monkeypatch.setattr(cli, "_get_memory", lambda: _StubMemory())
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.cmd_recall(args)
    return buf.getvalue(), captured


def test_cmd_recall_prints_whole_record_by_default(monkeypatch):
    out, captured = _run_cmd_recall(monkeypatch, ["network"])
    assert captured["called"] == ("network", 5, False)
    assert "192.168.2.0/24" in out, "the default path must not truncate the record"


def test_cmd_recall_preview_flag_is_parsed_and_applies_the_cap(monkeypatch):
    out, _ = _run_cmd_recall(monkeypatch, ["network", "--preview"])
    assert "Content truncated: showing 150 of" in out
    assert "192.168.2.0/24" not in out


def test_cmd_recall_json_flag_is_parsed_and_stays_complete(monkeypatch):
    out, _ = _run_cmd_recall(monkeypatch, ["network", "--json"])
    payload = json.loads(out)
    assert payload["results"][0]["content"] == LONG


@pytest.fixture(params=["working", "episodic", "memoria"])
def real_recall_content(request, tmp_path, monkeypatch):
    """Exercise actual stored and generated recall content beyond the 500-char cap."""
    from mnemosyne import cli
    from mnemosyne.core.memory import Mnemosyne
    import mnemosyne.core.beam as beam_module

    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_BANK", "default")
    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 500)
    if request.node.callspec.params.get("polyphonic", False):
        # Only replace external embedding inference; keep the real voice, SQL,
        # fusion, visibility, hydration and CLI paths under test.
        import numpy as np

        vector = np.ones(384, dtype=np.float32) / np.sqrt(384)
        monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
        monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda text: vector)
        monkeypatch.setattr(beam_module._embeddings, "embed",
                            lambda texts: np.stack([vector for _ in texts]))
        monkeypatch.setenv("MNEMOSYNE_VOICE_VECTOR", "1")
    memory = Mnemosyne(session_id="cli-content", db_path=tmp_path / "cli.db")
    content = "orchid version details " * 60 + "COMPLETE-記録-END"
    mid = memory.beam.remember(content, source="manual", scope="session")
    query = "orchid version"
    if request.param == "episodic":
        memory.beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance) "
            "SELECT id, content, source, timestamp, session_id, importance "
            "FROM working_memory WHERE id=?", (mid,))
        memory.beam.conn.execute("DELETE FROM working_memory WHERE id=?", (mid,))
        memory.beam.conn.commit()
    elif request.param == "memoria":
        query = "What is the Orchid version?"
        memory.beam.conn.execute(
            "INSERT INTO memoria_facts "
            "(session_id, fact_type, key, value, version_id, source_memory_id) "
            "VALUES ('cli-content', 'version', 'Orchid', ?, 1, ?)", (content, mid))
        memory.beam.conn.commit()
        context = memory.beam.memoria_retrieve(query, top_k=3)
        content = f"[MEMORIA {context['source']}]\n{context['context']}"
        mid = "memoria_" + context["source"]
    monkeypatch.setattr(cli, "_get_memory", lambda: memory)
    return memory, query, mid, content


@pytest.mark.parametrize("mode", ["human", "preview", "json", "explain"])
def test_real_backend_cli_content_contract(real_recall_content, capsys, mode):
    """Human output can expand selected hits; public recall and JSON stay capped."""
    from mnemosyne import cli
    import mnemosyne.core.beam as beam_module

    memory, query, mid, content = real_recall_content
    before = memory.recall(query, top_k=10)
    original = next(r for r in before if r["id"] == mid)
    assert original["content"] == content[:500], "fixture must hit the real backend cap"
    displayed = cli._recall_display_results(memory, query, before)
    assert [r["id"] for r in displayed] == [r["id"] for r in before]
    assert [r["score"] for r in displayed] == [r["score"] for r in before]
    args = [query, "10"]
    if mode != "human":
        args += ["--" + mode]
    cli.cmd_recall(args)
    out = capsys.readouterr().out
    if mode == "preview":
        assert f"showing 150 of {len(content)} chars" in out
        assert "COMPLETE-記録-END" not in out
    elif mode == "json":
        row = next(r for r in json.loads(out)["results"] if r["id"] == mid)
        assert row["content"] == content[:500]
    else:
        assert content in out
    assert original["content"] == content[:500], "rendering must not mutate recall payloads"
    assert beam_module.RECALL_CONTENT_CAP == 500
    after = memory.recall(query, top_k=10)
    assert next(r for r in after if r["id"] == mid)["content"] == content[:500]


@pytest.mark.parametrize("tier", ["working", "episodic"])
def test_human_expansion_keeps_selected_tier_with_duplicate_ids(tmp_path, monkeypatch, tier):
    """A working-first get() must not replace an episodic hit with its ID alias."""
    from mnemosyne import cli
    from mnemosyne.core.memory import Mnemosyne
    import mnemosyne.core.beam as beam_module

    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 500)
    memory = Mnemosyne(session_id="reader", db_path=tmp_path / "aliases.db")
    prefix = "iris " * 100
    for table, tail in [("working_memory", "WORKING-END"), ("episodic_memory", "EPISODIC-END")]:
        memory.beam.conn.execute(
            f"INSERT INTO {table} (id, content, source, timestamp, session_id) "
            "VALUES ('same-id', ?, 'manual', '2026-01-01', 'reader')", (prefix + tail,))
    memory.beam.conn.commit()
    result = {"id": "same-id", "tier": tier, "content": prefix, "score": 0.5}
    expanded = cli._recall_display_results(memory, "iris", [result])
    tail = "WORKING-END" if tier == "working" else "EPISODIC-END"
    assert expanded[0]["content"] == prefix + tail
    assert result["content"] == prefix


@pytest.mark.parametrize("tier", ["working", "episodic"])
@pytest.mark.parametrize("scope,visible", [("session", False), ("global", True)])
def test_human_expansion_preserves_peer_visibility(tmp_path, monkeypatch, tier, scope, visible):
    """The full-text reader accepts global peers and refuses private peers in both tiers."""
    from mnemosyne import cli
    from mnemosyne.core.memory import Mnemosyne
    import mnemosyne.core.beam as beam_module

    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 500)
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
    memory = Mnemosyne(session_id="reader", db_path=tmp_path / "peers.db")
    table = "working_memory" if tier == "working" else "episodic_memory"
    prefix = "iris " * 100
    memory.beam.conn.execute(
        f"INSERT INTO {table} (id, content, source, timestamp, session_id, scope) "
        "VALUES ('peer', ?, 'manual', '2026-01-01', 'other-session', ?)",
        (prefix + "PEER-TAIL", scope))
    memory.beam.conn.commit()
    result = {"id": "peer", "tier": tier, "content": prefix, "score": 0.5}
    expanded = cli._recall_display_results(memory, "iris", [result])[0]
    assert expanded["content"] == (prefix + "PEER-TAIL" if visible else prefix)
    assert bool(expanded.get("_cli_incomplete_content")) is (not visible)


@pytest.mark.parametrize("case", ["changed", "missing", "unknown-tier"])
def test_unavailable_full_content_has_explicit_unknown_length(
        tmp_path, monkeypatch, capsys, case):
    """Never substitute changed text or claim a capped total is full."""
    from mnemosyne import cli
    from mnemosyne.core.memory import Mnemosyne
    import mnemosyne.core.beam as beam_module

    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 500)
    memory = Mnemosyne(session_id="reader", db_path=tmp_path / "scope.db")
    prefix = "iris " * 100
    body = "changed " + prefix if case == "changed" else prefix + "PRIVATE-TAIL"
    if case != "missing":
        memory.beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, scope) "
            "VALUES ('selected', ?, 'manual', '2026-01-01', ?, 'session')",
            (body, "reader"))
        memory.beam.conn.commit()
    result = {"id": "selected", "tier": "other" if case == "unknown-tier" else "working",
              "content": prefix, "score": 0.5}
    expanded = cli._recall_display_results(memory, "iris", [result])
    assert expanded[0]["content"] == prefix
    cli._print_recall_results("iris", expanded)
    out = capsys.readouterr().out
    assert "Full content unavailable; showing recall preview" in out
    assert "PRIVATE-TAIL" not in out
    cli._print_recall_results("iris", expanded, preview=True)
    preview = capsys.readouterr().out
    assert "full length unavailable" in preview
    assert "of 500" not in preview


@pytest.mark.parametrize("real_recall_content", ["working"], indirect=True)
def test_human_expansion_keeps_enhanced_cache_and_json_capped(
        real_recall_content, monkeypatch, capsys):
    """Displaying a warmed enhanced hit cannot grow later cached/public payloads."""
    from mnemosyne import cli

    memory, query, mid, content = real_recall_content
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    memory.recall(query, top_k=10)
    cli.cmd_recall([query, "10"])
    assert content in capsys.readouterr().out
    cli.cmd_recall([query, "10", "--json"])
    rows = json.loads(capsys.readouterr().out)["results"]
    assert next(r for r in rows if r["id"] == mid)["content"] == content[:500]


@pytest.mark.parametrize("real_recall_content", ["working", "episodic"], indirect=True)
@pytest.mark.parametrize("cross_session", [False, True])
@pytest.mark.parametrize("polyphonic", [False, True])
def test_real_peer_recall_expands_only_when_visible(
        real_recall_content, monkeypatch, capsys, cross_session, polyphonic):
    """Cross-session hits expand fully while disabled recall keeps peers private."""
    from mnemosyne import cli

    memory, query, mid, content = real_recall_content
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "1" if cross_session else "0")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1" if polyphonic else "0")
    for table in ("working_memory", "episodic_memory"):
        memory.beam.conn.execute(
            f"UPDATE {table} SET session_id='peer-session', scope='session' WHERE id=?",
            (mid,))
    memory.beam.conn.commit()
    before = memory.recall(query, top_k=10)
    selected = [r for r in before if r["id"] == mid]
    assert bool(selected) is cross_session
    if cross_session:
        assert selected[0]["content"] == content[:500]
        displayed = cli._recall_display_results(memory, query, before)
        assert next(r for r in displayed if r["id"] == mid)["content"] == content
        assert [(r["id"], r["score"]) for r in displayed] == [
            (r["id"], r["score"]) for r in before]
    cli.cmd_recall([query, "10"])
    human = capsys.readouterr().out
    assert (content in human) is cross_session
    cli.cmd_recall([query, "10", "--preview"])
    preview = capsys.readouterr().out
    assert (f"showing 150 of {len(content)} chars" in preview) is cross_session
    cli.cmd_recall([query, "10", "--json"])
    rows = json.loads(capsys.readouterr().out)["results"]
    selected_json = [r for r in rows if r["id"] == mid]
    assert bool(selected_json) is cross_session
    if cross_session:
        assert selected_json[0]["content"] == content[:500]
        assert selected[0]["content"] == content[:500]
        after = memory.recall(query, top_k=10)
        assert next(r for r in after if r["id"] == mid)["content"] == content[:500]


@pytest.mark.parametrize("real_recall_content", ["memoria"], indirect=True)
def test_enhanced_memoria_expands_capitalized_multi_fact_query(
        real_recall_content, monkeypatch, capsys):
    """Full multi-key aggregates survive display without growing cached JSON."""
    from mnemosyne import cli

    memory, _, _, _ = real_recall_content
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    memory.beam.conn.execute(
        "INSERT INTO memoria_facts "
        "(session_id, fact_type, key, value, version_id) "
        "VALUES ('cli-content', 'version', 'Orchid Release', ?, 1)",
        ("release notes " * 60 + "SECOND-FACT-END",))
    memory.beam.conn.commit()
    query = "What is the ORCHID version?"
    aggregate = memory.beam.memoria_retrieve(query, top_k=3)
    full = f"[MEMORIA {aggregate['source']}]\n{aggregate['context']}"
    assert "COMPLETE-記録-END" in full and "SECOND-FACT-END" in full
    mid = "memoria_" + aggregate["source"]
    before = memory.recall(query, top_k=10)
    assert next(r for r in before if r["id"] == mid)["content"] == full[:500]
    cli.cmd_recall([query, "10"])
    assert full in capsys.readouterr().out
    cli.cmd_recall([query, "10", "--preview"])
    assert f"showing 150 of {len(full)} chars" in capsys.readouterr().out
    cli.cmd_recall([query, "10", "--json"])
    rows = json.loads(capsys.readouterr().out)["results"]
    assert next(r for r in rows if r["id"] == mid)["content"] == full[:500]
    assert next(r for r in before if r["id"] == mid)["content"] == full[:500]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
