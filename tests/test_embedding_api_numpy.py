"""API embeddings need NumPy, not the optional local FastEmbed stack."""
import builtins
import io
import json
from types import SimpleNamespace

import pytest

from mnemosyne.core import embeddings


@pytest.fixture
def api(monkeypatch):
    for key in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://synthetic.invalid/v1")
    monkeypatch.setattr(embeddings, "_cfg", lambda _key, default=None: default)
    monkeypatch.setattr(embeddings, "_cfg_bool", lambda _key, default=False: default)
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", "")
    monkeypatch.setattr(embeddings, "np", None)
    monkeypatch.setattr(embeddings, "TextEmbedding", None)
    monkeypatch.setattr(embeddings, "_API_NUMPY_WARNED", False)
    monkeypatch.setattr(embeddings, "_API_NUMPY_RETRY_AT", 0.0)
    embeddings._embed_query_cached.cache_clear()
    original_import = builtins.__import__
    state = {"numpy_missing": True, "requests": [], "local_imports": []}

    def import_dependency(name, *args, **kwargs):
        if name == "fastembed":
            state["local_imports"].append(name)
            raise ModuleNotFoundError(name, name=name)
        if name == "numpy" and state["numpy_missing"]:
            raise ModuleNotFoundError("sensitive import path", name=name)
        return original_import(name, *args, **kwargs)

    def request(*args, **kwargs):
        state["requests"].append(args)
        return io.BytesIO(json.dumps({"data": [{"embedding": [0.25, 0.75]}]}).encode())

    monkeypatch.setattr(builtins, "__import__", import_dependency)
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", request)
    yield state
    embeddings._embed_query_cached.cache_clear()


@pytest.fixture
def recovery_api(request):
    """Positive vector checks need real NumPy; missing-dependency checks do not."""
    pytest.importorskip("numpy")
    return request.getfixturevalue("api")


@pytest.mark.parametrize("call", [
    lambda: embeddings.embed_query("private query"),
    lambda: embeddings.embed(["private document"]),
])
def test_missing_numpy_fails_before_sending_api_input(api, call, caplog):
    with pytest.raises(RuntimeError, match="NumPy") as error:
        call()
    assert api["requests"] == []
    assert "endpoint failed" not in str(error.value)
    assert "sensitive import path" not in str(error.value) + caplog.text
    assert "private query" not in caplog.text
    assert "private document" not in caplog.text


def test_api_availability_recovers_numpy_without_fastembed(recovery_api, caplog):
    assert embeddings.available() is False
    assert embeddings.available() is False
    assert "NumPy" in caplog.text
    recovery_api["numpy_missing"] = False
    assert embeddings.available() is True
    assert embeddings.embed_query("recovered query").tolist() == [0.25, 0.75]
    assert recovery_api["local_imports"] == []
    assert len(recovery_api["requests"]) == 1


def test_direct_api_query_recovers_after_missing_numpy(recovery_api):
    with pytest.raises(RuntimeError, match="NumPy"):
        embeddings.embed_query("same query")
    assert embeddings._embed_query_cached.cache_info().currsize == 0
    recovery_api["numpy_missing"] = False
    assert embeddings.embed_query("same query").tolist() == [0.25, 0.75]
    assert recovery_api["local_imports"] == []


def test_runtime_diagnostics_report_api_dependency_unavailable(api):
    from mnemosyne.runtime_diagnostics import collect_runtime_diagnostics

    checks = {row["check"]: row["status"] for row in collect_runtime_diagnostics()["checks"]}
    assert checks["numpy"] == "MISSING"
    assert checks["embeddings_available"] == "NO"
    assert api["requests"] == []


def test_api_numpy_broken_install_backs_off_and_recovers(recovery_api, monkeypatch, caplog):
    original_import = builtins.__import__
    clock = [100.0]
    attempts = []

    def broken_numpy(name, *args, **kwargs):
        if name == "numpy":
            attempts.append(name)
            if recovery_api["numpy_missing"]:
                raise ImportError("private native-library path")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_numpy)
    monkeypatch.setattr(embeddings, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    assert embeddings.available() is False
    assert embeddings.available() is False
    assert len(attempts) == 1
    assert "private native-library path" not in caplog.text
    recovery_api["numpy_missing"] = False
    clock[0] = 101.0
    assert embeddings.available() is True
    assert len(attempts) == 2
    assert embeddings.embed_query("repaired query").tolist() == [0.25, 0.75]
