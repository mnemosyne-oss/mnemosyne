"""Offline tests at the serialized HTTP and CLI request boundaries."""

import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import urllib.request

import httpx
import pytest


@pytest.fixture
def load_client(monkeypatch):
    def load(**config):
        for key in os.environ:
            if key.startswith("MNEMOSYNE_"):
                monkeypatch.delenv(key)
        for key, value in config.items():
            monkeypatch.setenv("MNEMOSYNE_" + key, value)
        path = Path(__file__).parents[1] / "mnemosyne/core/local_llm.py"
        spec = importlib.util.spec_from_file_location("provider_transport_client", path)
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        return client

    return load


@pytest.fixture(params=["httpx", "urllib"])
def requests_sent(request, monkeypatch):
    sent = []
    response = json.dumps({"choices": [{"message": {"content": "A useful summary."}}]}).encode()
    if request.param == "httpx":
        real_client = httpx.Client

        def handle(req):
            sent.append((str(req.url), json.loads(req.content)))
            return httpx.Response(200, content=response)

        monkeypatch.setattr(
            httpx, "Client",
            lambda **kwargs: real_client(
                **kwargs, transport=httpx.MockTransport(handle), trust_env=False,
            ),
        )
    else:
        monkeypatch.setitem(sys.modules, "httpx", None)

        def urlopen(req, **kwargs):
            sent.append((req.full_url, json.loads(req.data)))
            return io.BytesIO(response)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return sent


@pytest.mark.parametrize("region,host", [
    ("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com"),
])
@pytest.mark.parametrize("model", ["", "MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"])
def test_preset_summary_reaches_http_boundary(load_client, requests_sent, region, host, model):
    client = load_client(LLM_PROVIDER="minimax", LLM_REGION=region, LLM_MODEL=model)
    assert client.summarize_memories(["User prefers concise summaries."]) == "A useful summary."
    [(url, body)] = requests_sent
    assert url == f"https://{host}/v1/chat/completions"
    assert body["model"] == (model or "MiniMax-M3.1-Flash")
    assert "User prefers concise summaries." in body["messages"][0]["content"]
    assert body["stream"] is False
    assert body["max_tokens"] == 2048
    # Omission uses the service's adaptive/max defaults for Flash.
    assert not {"thinking", "reasoning_effort", "service_tier", "reasoning_split"} & body.keys()


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_flash_explicit_effort_survives_serialization(load_client, requests_sent, effort):
    extra = {"thinking": {"type": "adaptive"}, "reasoning_effort": effort}
    client = load_client(LLM_PROVIDER="minimax", LLM_EXTRA_BODY=json.dumps(extra))
    assert client._call_remote_llm("Summarize this.") == "A useful summary."
    body = requests_sent[0][1]
    assert body["model"] == "MiniMax-M3.1-Flash"
    assert all(body[key] == value for key, value in extra.items())


@pytest.mark.parametrize("provider,model", [
    ("minimax", "MiniMax-M3"), ("minimax", "MiniMax-M2.7"),
    ("minimax", "saved-custom-model"), ("other-provider", "other-model"),
])
def test_explicit_endpoint_model_and_body_remain_authoritative(
    load_client, requests_sent, provider, model,
):
    extra = {"thinking": {"type": "disabled"}, "custom_option": "keep"}
    client = load_client(
        LLM_PROVIDER=provider, LLM_MODEL=model, LLM_BASE_URL="https://custom.invalid/v1/",
        LLM_EXTRA_BODY=json.dumps(extra),
    )
    assert client._call_remote_llm("Keep my custom configuration.") == "A useful summary."
    [(url, body)] = requests_sent
    assert url == "https://custom.invalid/v1/chat/completions"
    assert body["model"] == model
    assert all(body[key] == value for key, value in extra.items())


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
def test_multimodal_content_is_not_rewritten(load_client, requests_sent, model):
    client = load_client(LLM_PROVIDER="minimax", LLM_MODEL=model)
    content = [
        {"type": "text", "text": "Describe this."},
        {"type": "image_url", "image_url": {"url": "https://media.invalid/image.png"}},
        {"type": "video_url", "video_url": {"url": "mm_file://saved-video"}},
    ]
    text, status, error = client._call_remote_llm_with_model(content, model)
    assert (text, status, error) == ("A useful summary.", 200, None)
    assert requests_sent[0][1]["messages"][0]["content"] == content


def test_saved_fallback_uses_its_own_endpoint_and_body(load_client, monkeypatch):
    client = load_client(
        LLM_PROVIDER="minimax", LLM_FALLBACK_MODELS="MiniMax-M2.7",
        LLM_FALLBACK_BASE_URL="https://fallback.invalid/v1",
        LLM_FALLBACK_EXTRA_BODY='{"custom_option":"fallback"}',
    )
    sent = []
    real_client = httpx.Client

    def handle(req):
        sent.append((str(req.url), json.loads(req.content)))
        if len(sent) == 1:
            return httpx.Response(503, json={"error": "retry"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Recovered."}}]})

    monkeypatch.setattr(
        httpx, "Client",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle), trust_env=False),
    )
    assert client._call_remote_llm("Keep the saved fallback.") == "Recovered."
    assert sent[0][1]["model"] == "MiniMax-M3.1-Flash"
    assert sent[1][0] == "https://fallback.invalid/v1/chat/completions"
    assert sent[1][1]["model"] == "MiniMax-M2.7"
    assert sent[1][1]["custom_option"] == "fallback"
    assert "custom_option" not in sent[0][1]


@pytest.mark.parametrize("model", [None, "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
def test_heal_cli_receives_configured_model(monkeypatch, model):
    if model is None:
        monkeypatch.delenv("MNEMOSYNE_HEAL_JUDGE_MODEL", raising=False)
    else:
        monkeypatch.setenv("MNEMOSYNE_HEAL_JUDGE_MODEL", model)
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/heal_quality.py"))
    monkeypatch.setattr(Path, "exists", lambda self: True)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '{"content":[{"text":"ok"}]}')

    monkeypatch.setattr(subprocess, "run", run)
    assert script["_call_mmx"]("Judge this.") == "ok"
    assert calls[0][1:] == [
        "text", "chat", "--model", model or "MiniMax-M3.1-Flash", "--message", "Judge this.",
    ]
