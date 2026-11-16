"""Startup diagnostics describe configuration without probing or exposing it."""
import importlib
import json
import logging
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("url,endpoint", [
    ("https://user:password@example.test:8443/private-token?key=query-secret#fragment-secret", "https://example.test:8443"),
    ("http://[::1]:1234/v1", "http://[::1]:1234"),
    ("https://example.test:bad/private-secret", "<invalid-url>"),
])
def test_configuration_announcement_is_once_read_only_and_redacted(monkeypatch, caplog, url, endpoint):
    from mnemosyne.core import embeddings as e
    import sqlite3
    import urllib.request

    monkeypatch.setattr(e, "_RUNTIME_CONFIGURATION_ANNOUNCED", False, raising=False)
    monkeypatch.setattr(e, "_DEFAULT_MODEL", "BAAI/bge-m3")
    monkeypatch.setattr(e, "EMBEDDING_DIM", 1024)
    monkeypatch.setattr(e, "_OPENAI_API_KEY", "credential-secret")
    monkeypatch.setattr(e, "_configured_api_url", lambda: url)
    monkeypatch.setattr(e, "_is_disabled", lambda: False)
    def forbidden(*args, **kwargs):
        pytest.fail("configuration diagnostics must not open a DB, send a request or load a model")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(e, "available", forbidden)
    monkeypatch.setattr(e, "_ensure_fastembed", forbidden)
    with caplog.at_level(logging.INFO, logger="mnemosyne.core.embeddings"):
        e.announce_runtime_configuration()
        e.announce_runtime_configuration()
    records=[r for r in caplog.records if r.msg.startswith("Embedding runtime configuration:")]
    assert len(records)==1
    text=records[0].getMessage()
    for expected in ["model=BAAI/bge-m3", "dimension=1024", "credentials_present=True", "endpoint="+endpoint, "backend=api", "model_dim_credentials=import-time", "endpoint_resolution=call-time"]:
        assert expected in text
    for secret in ["password", "credential-secret", "private-token", "query-secret", "fragment-secret", "private-secret"]:
        assert secret not in text


def test_loaded_values_and_config_file_location_survive_late_environment_changes(tmp_path):
    config=tmp_path/"config.yaml"
    config.write_text("embedding_model: BAAI/bge-m3\nembedding_dim: 1024\nembedding_api_key: yaml-secret\nembedding_api_url: https://old.example.test/v1\n")
    env=dict(os.environ, MNEMOSYNE_DATA_DIR=str(tmp_path), MNEMOSYNE_NO_EMBEDDINGS="0", MNEMOSYNE_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5", MNEMOSYNE_EMBEDDING_DIM="384")
    script="""import logging,os,io,json
from mnemosyne.core import embeddings as e
os.environ['MNEMOSYNE_EMBEDDING_MODEL']='BAAI/bge-large-en-v1.5'
os.environ['MNEMOSYNE_EMBEDDING_DIM']='768'
stream=io.StringIO();handler=logging.StreamHandler(stream);e.logger.addHandler(handler);e.logger.setLevel(logging.INFO)
e.announce_runtime_configuration()
print(json.dumps({'log':stream.getvalue()}))
"""
    result=subprocess.run([sys.executable,"-c",script],env=env,capture_output=True,text=True,check=True)
    text=json.loads(result.stdout)["log"]
    assert "model=BAAI/bge-m3" in text and "dimension=1024" in text
    assert "config_file_at_import=MNEMOSYNE_DATA_DIR/config.yaml" in text
    assert "config_file_present_at_import=True" in text
    assert "endpoint=https://old.example.test" in text
    assert "yaml-secret" not in text
    assert config.read_text().startswith("embedding_model: BAAI/bge-m3")


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_provider_start_announces_configuration(monkeypatch,tmp_path,caplog,package):
    module=importlib.import_module(package)
    from mnemosyne.core import embeddings as e
    monkeypatch.setattr(e,"_RUNTIME_CONFIGURATION_ANNOUNCED",False,raising=False)
    monkeypatch.setattr(module,"_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED",False)
    provider=module.MnemosyneMemoryProvider()
    try:
        with caplog.at_level(logging.INFO):
            provider.initialize("startup-diagnostic",hermes_home=str(tmp_path),auto_sleep=False)
        assert provider._beam is not None
        assert sum(r.msg.startswith("Embedding runtime configuration:") for r in caplog.records)==1
    finally:
        provider.shutdown()


def test_configuration_diagnostic_failure_does_not_break_provider_start(monkeypatch,tmp_path):
    import mnemosyne_hermes as module
    from mnemosyne.core import embeddings as e
    def broken():
        raise RuntimeError("diagnostic failure")
    monkeypatch.setattr(e,"announce_runtime_configuration",broken,raising=False)
    provider=module.MnemosyneMemoryProvider()
    try:
        provider.initialize("diagnostic-failure",hermes_home=str(tmp_path),auto_sleep=False)
        assert provider._beam is not None
    finally:
        provider.shutdown()


def test_configuration_announcement_can_retry_after_logging_failure(monkeypatch,caplog):
    from mnemosyne.core import embeddings as e
    monkeypatch.setattr(e,"_RUNTIME_CONFIGURATION_ANNOUNCED",False)
    with caplog.at_level(logging.INFO,logger="mnemosyne.core.embeddings"):
        with monkeypatch.context() as m:
            def broken(*args,**kwargs):
                raise RuntimeError("handler failed")
            m.setattr(e.logger,"info",broken)
            e.announce_runtime_configuration()
        e.announce_runtime_configuration()
    assert sum(r.msg.startswith("Embedding runtime configuration:") for r in caplog.records)==1


def test_configuration_announcement_is_shared_between_concurrent_callers(monkeypatch,caplog):
    from concurrent.futures import ThreadPoolExecutor
    from mnemosyne.core import embeddings as e
    monkeypatch.setattr(e,"_RUNTIME_CONFIGURATION_ANNOUNCED",False)
    with caplog.at_level(logging.INFO,logger="mnemosyne.core.embeddings"):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _:e.announce_runtime_configuration(),range(20)))
    assert sum(r.msg.startswith("Embedding runtime configuration:") for r in caplog.records)==1


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_provider_remains_compatible_with_core_without_configuration_helper(monkeypatch,tmp_path,caplog,package):
    module=importlib.import_module(package)
    from mnemosyne.core import embeddings as e
    monkeypatch.delattr(e,"announce_runtime_configuration")
    provider=module.MnemosyneMemoryProvider()
    try:
        with caplog.at_level(logging.INFO):
            provider.initialize("older-core",hermes_home=str(tmp_path),auto_sleep=False)
        assert provider._beam is not None
        assert not any(r.msg.startswith("Embedding runtime configuration:") for r in caplog.records)
    finally:
        provider.shutdown()
