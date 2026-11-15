"""Fail-loud contract for an unavailable embedding engine (bundled provider).

Mirrors integrations/hermes/tests/test_embedding_engine_announcement.py for
the standalone bundled copy: the engine's optional dependencies (fastembed,
sqlite_vec) were not importable in a gateway's runtime interpreter, every
embedding write was silently skipped by the ``if available():`` guard, and the
deployment ran ~30 days with dense/semantic recall off without a single log
line. The bundled provider announces the degraded mode once per process at
startup, stating the consequence and the fix, and never raises.
"""

import logging

import pytest

import hermes_memory_provider
from hermes_memory_provider import MnemosyneMemoryProvider


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch, tmp_path):
    from mnemosyne.core.config import MnemosyneConfig

    core_config_path = tmp_path / "mnemosyne-config.yaml"
    core_config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        MnemosyneConfig,
        "_instance",
        MnemosyneConfig(config_path=core_config_path),
    )


class _OkBeam:
    """Minimal BeamMemory stand-in: accepts init kwargs, does nothing."""

    author_id = None

    def __init__(self, *args, **kwargs):
        pass


def _engine_capability(monkeypatch, *, dense: bool, vec: bool) -> None:
    """Pin the capability predicates; no real dependency is required."""
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings as embeddings_module

    # Reset the once-per-process guard so the test is order-independent.
    monkeypatch.setattr(
        hermes_memory_provider, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False,
        raising=False,
    )
    monkeypatch.setattr(embeddings_module, "available", lambda: dense)
    monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", vec)
    # Clear the env opt-out aliases so the missing-dependency branch runs.
    for name in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        monkeypatch.delenv(name, raising=False)

def test_provider_announces_unavailable_embedding_engine_once(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    second = None
    kwargs = {"hermes_home": str(tmp_path), "auto_sleep": False}
    try:
        _engine_capability(monkeypatch, dense=False, vec=False)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **kwargs)

            # The consequence, the hint and both missing halves are named, once.
            assert caplog.text.count("embedding generation is unavailable") == 1
            assert "fastembed" in caplog.text
            assert "sqlite_vec" in caplog.text
            assert "keyword-only" not in caplog.text
            assert "non-vector recall remains available" in caplog.text
            assert "extra importable" in caplog.text

            # Once per process: a second provider start must not re-announce.
            second = MnemosyneMemoryProvider()
            second.initialize("sess-2", **kwargs)
            assert caplog.text.count("embedding generation is unavailable") == 1

        # Engine available -> no announcement at all.
        _engine_capability(monkeypatch, dense=True, vec=True)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess-3", **kwargs)
            assert "embedding generation is unavailable" not in caplog.text
    finally:
        provider.shutdown()
        if second is not None:
            second.shutdown()


def test_provider_reservation_released_so_later_outage_warns(monkeypatch, tmp_path, caplog):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings as embeddings_module

    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        _engine_capability(monkeypatch, dense=True, vec=True)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            # A start with the engine fully available must not consume the
            # once-per-process budget: the reservation is released again.
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert hermes_memory_provider._EMBEDDING_ENGINE_UNAVAILABLE_LOGGED is False
            assert "embedding generation is unavailable" not in caplog.text

            # Later outage (the flag is NOT reset in between): the released
            # guard must still let the warning through.
            monkeypatch.setattr(embeddings_module, "available", lambda: False)
            monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", False)
            provider.initialize("sess-2", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert caplog.text.count("embedding generation is unavailable") == 1
    finally:
        provider.shutdown()


def test_provider_announces_missing_fastembed_only(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        _engine_capability(monkeypatch, dense=False, vec=True)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "missing fastembed" in caplog.text
            assert "embedding generation is unavailable" in caplog.text
            assert "non-vector recall remains available" in caplog.text
            assert "keyword-only" not in caplog.text
            assert "sqlite_vec" not in caplog.text
            assert "extra importable" in caplog.text
    finally:
        provider.shutdown()


def test_provider_announces_missing_sqlite_vec_only(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        _engine_capability(monkeypatch, dense=True, vec=False)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "missing sqlite_vec" in caplog.text
            assert "native vector index unavailable" in caplog.text
            assert "semantic fallback may remain available" in caplog.text
            assert "embedding generation is unavailable" not in caplog.text
            assert "dense/semantic recall is disabled" not in caplog.text
            assert "keyword-only" not in caplog.text
            assert "fastembed" not in caplog.text
            assert "extra importable" in caplog.text
    finally:
        provider.shutdown()


def test_provider_reports_opt_out_without_install_hint(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        _engine_capability(monkeypatch, dense=False, vec=False)
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "opt-out environment variable" in caplog.text
            assert "keyword-only" not in caplog.text
            assert "non-vector recall remains available" in caplog.text
            assert "extra importable" not in caplog.text
            assert "missing fastembed" not in caplog.text
            assert "missing sqlite_vec" not in caplog.text
    finally:
        provider.shutdown()


def test_provider_announces_api_backend_without_credentials(monkeypatch, tmp_path, caplog):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings as embeddings_module

    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        monkeypatch.setattr(
            hermes_memory_provider, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False,
            raising=False,
        )
        monkeypatch.setattr(embeddings_module, "_DEFAULT_MODEL", "openai/text-embedding-3-small")
        monkeypatch.setattr(embeddings_module, "_OPENAI_API_KEY", "")
        monkeypatch.setattr(embeddings_module, "_FASTEMBED_AVAILABLE", True)
        monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", True)
        for name in (
            "MNEMOSYNE_NO_EMBEDDINGS",
            "MNEMOSYNE_SKIP_EMBEDDINGS",
            "MNEMOSYNE_EMBEDDINGS_OFF",
            "MNEMOSYNE_EMBEDDING_API_URL",
        ):
            monkeypatch.delenv(name, raising=False)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "embedding API backend has no usable credentials" in caplog.text
            assert "MNEMOSYNE_EMBEDDING_API_KEY" in caplog.text
            assert "missing fastembed" not in caplog.text
            assert "extra importable" not in caplog.text
    finally:
        provider.shutdown()


def test_provider_announces_api_backend_and_missing_sqlite_vec(monkeypatch, tmp_path, caplog):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings as embeddings_module

    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        monkeypatch.setattr(
            hermes_memory_provider, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False,
            raising=False,
        )
        monkeypatch.setattr(embeddings_module, "_DEFAULT_MODEL", "openai/text-embedding-3-small")
        monkeypatch.setattr(embeddings_module, "_OPENAI_API_KEY", "")
        monkeypatch.setattr(embeddings_module, "_FASTEMBED_AVAILABLE", True)
        monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", False)
        for name in (
            "MNEMOSYNE_NO_EMBEDDINGS",
            "MNEMOSYNE_SKIP_EMBEDDINGS",
            "MNEMOSYNE_EMBEDDINGS_OFF",
            "MNEMOSYNE_EMBEDDING_API_URL",
        ):
            monkeypatch.delenv(name, raising=False)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            text = caplog.text
            # Both simultaneous causes are named in the single warning.
            assert "embedding API backend has no usable credentials" in text
            assert "missing sqlite_vec" in text
            # fastembed is available in this state: it must not be named as missing.
            assert "missing fastembed" not in text
            # Both hints are present: credentials AND the dependency install.
            assert "MNEMOSYNE_EMBEDDING_API_KEY" in text
            assert "extra importable" in text
            # One warning, not two.
            assert caplog.text.count("embedding generation is unavailable") == 1
    finally:
        provider.shutdown()


def test_announcement_is_atomic_under_concurrent_calls(monkeypatch, tmp_path, caplog):
    """Concurrent calls must still emit exactly one warning."""
    import threading as _threading
    import time as _time
    from mnemosyne.core import embeddings as _embeddings

    _engine_capability(monkeypatch, dense=False, vec=True)
    # Widen the window between the guard check and the flag write so the
    # pre-fix code reliably emits one warning per thread.
    _pinned = _embeddings.available
    monkeypatch.setattr(
        _embeddings, "available", lambda: (_time.sleep(0.1), _pinned())[1]
    )

    with caplog.at_level(logging.WARNING):
        threads = [
            _threading.Thread(target=hermes_memory_provider._announce_embedding_engine_unavailable)
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Branch-agnostic: the opt-out message and the unavailable message both
    # start with this phrase, so the count is exact either way.
    assert caplog.text.count("Mnemosyne: embedding engine") == 1


def test_native_index_warning_matches_real_semantic_fallback(
    monkeypatch, tmp_path, caplog
):
    """Missing sqlite-vec must not be described as loss of a working semantic path."""
    import json

    import numpy as np
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings

    provider_module = hermes_memory_provider
    _engine_capability(monkeypatch, dense=True, vec=False)
    monkeypatch.setattr(provider_module, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False)
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr(embeddings, "embed_query", lambda text: np.array([1.0, 0.0]))
    monkeypatch.setattr(embeddings, "embed", lambda texts: np.array([[0.8, 0.6] for _ in texts]))
    memory = beam_module.BeamMemory(session_id="announcement", db_path=tmp_path / "real.db")
    try:
        mid = memory.remember("Cobalt telescope calibration", source="fact", extract_entities=False)
        memory.conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model) "
            "VALUES (?, ?, ?)", (mid, json.dumps([0.8, 0.6]), "deterministic-test")
        )
        memory.conn.commit()
        matches = beam_module._wm_vec_search(memory.conn, np.array([1.0, 0.0]))
        assert matches == [{"id": mid, "sim": pytest.approx(0.8)}]
        results = memory.recall("Cobalt telescope calibration", top_k=3)
        hit = next(row for row in results if row["id"] == mid)
        assert hit["dense_score"] == pytest.approx(0.8)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=provider_module.__name__):
            provider_module._announce_embedding_engine_unavailable()
        assert "native vector index unavailable" in caplog.text
        assert "semantic fallback may remain available" in caplog.text
        assert "keyword-only" not in caplog.text
        assert "dense/semantic recall is disabled" not in caplog.text
    finally:
        memory.conn.close()


@pytest.mark.parametrize("opt_out", [False, True], ids=["missing-generation", "opt-out"])
def test_generation_unavailable_retains_real_polyphonic_temporal_recall(
    monkeypatch, tmp_path, caplog, opt_out
):
    """The non-vector consequence must hold through public polyphonic recall."""
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings

    _engine_capability(monkeypatch, dense=False, vec=False)
    if opt_out:
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")

    def forbidden_embedding(*args, **kwargs):
        pytest.fail("generation-unavailable recall must not request embeddings")

    monkeypatch.setattr(embeddings, "embed", forbidden_embedding)
    monkeypatch.setattr(embeddings, "embed_query", forbidden_embedding)
    memory = beam_module.BeamMemory(session_id="non-vector", db_path=tmp_path / "real.db")
    try:
        mid = memory.remember(
            "Cobalt telescope calibration completed recently", source="fact",
            importance=0.9, extract_entities=False,
        )
        other = memory.remember(
            "Jade weather forecast for next season", source="fact", extract_entities=False,
        )
        # An older row must not pass the temporal voice's seven-day window.
        memory.conn.execute(
            "UPDATE working_memory SET timestamp = ? WHERE id = ?",
            ("2000-01-01T00:00:00", other),
        )
        memory.conn.commit()
        results = memory.recall("Cobalt telescope calibration recently", top_k=3)
        assert [row["id"] for row in results] == [mid]
        assert other not in [row["id"] for row in results]
        voices = results[0]["voice_scores"]
        assert voices["temporal"] > 0
        assert voices.get("vector", 0) == 0
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            hermes_memory_provider._announce_embedding_engine_unavailable()
        assert "non-vector recall remains available" in caplog.text
        assert "keyword-only" not in caplog.text
        if opt_out:
            assert "opt-out environment variable" in caplog.text
            assert "extra importable" not in caplog.text
        else:
            assert "embedding generation is unavailable" in caplog.text
    finally:
        memory.conn.close()


def test_diagnostic_exception_releases_reservation_for_retry(monkeypatch, tmp_path, caplog):
    """A failed diagnostic must not consume the later real outage announcement."""
    from mnemosyne.core import embeddings

    _engine_capability(monkeypatch, dense=False, vec=False)
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)

    def failed_detection():
        raise RuntimeError("deterministic diagnostic failure")

    monkeypatch.setattr(embeddings, "available", failed_detection)
    provider = MnemosyneMemoryProvider()
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("failed-diagnostic", hermes_home=str(tmp_path), auto_sleep=False)
            assert not hermes_memory_provider._EMBEDDING_ENGINE_UNAVAILABLE_LOGGED
            assert "Mnemosyne: embedding engine" not in caplog.text
            # Restore only the detector. Do not reset the announcement flag.
            monkeypatch.setattr(embeddings, "available", lambda: False)
            provider.initialize("later-outage", hermes_home=str(tmp_path), auto_sleep=False)
            provider.initialize("repeated-outage", hermes_home=str(tmp_path), auto_sleep=False)
            assert caplog.text.count("embedding generation is unavailable") == 1
    finally:
        provider.shutdown()



def test_config_opt_out_is_not_attributed_only_to_environment(monkeypatch, tmp_path, caplog):
    """An actual YAML opt-out must be diagnosed with all aliases unset."""
    from mnemosyne.core import config, embeddings

    _engine_capability(monkeypatch, dense=False, vec=True)
    config_path = tmp_path / "opt-out.yaml"
    config_path.write_text("no_embeddings: true\n", encoding="utf-8")
    monkeypatch.setattr(config, "_default_config_path", lambda: config_path)
    assert embeddings._is_disabled()
    with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
        hermes_memory_provider._announce_embedding_engine_unavailable()
    assert "disabled by configuration or an opt-out environment variable" in caplog.text
    assert "embedding generation is disabled" in caplog.text
    assert "extra importable" not in caplog.text
    assert "no usable credentials" not in caplog.text


@pytest.mark.parametrize("vec_available", [True, False], ids=["native-index", "no-native-index"])
@pytest.mark.parametrize("api_key, api_url", [
    ("synthetic-test-key", ""),
    ("", "https://synthetic.invalid/v1"),
], ids=["credentialed", "keyless-endpoint"])
def test_api_numpy_failure_is_not_blamed_on_credentials(
    monkeypatch, tmp_path, caplog, vec_available, api_key, api_url
):
    """Real availability with a blocked NumPy import must name the missing dependency."""
    import builtins

    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings

    for name in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_memory_provider, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False)
    monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", vec_available)
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setattr(embeddings, "_OPENAI_API_KEY", api_key)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", api_url)
    monkeypatch.setattr(embeddings, "np", None)
    monkeypatch.setattr(embeddings, "_API_NUMPY_RETRY_AT", 0.0)
    monkeypatch.setattr(embeddings, "_API_NUMPY_WARNED", False)
    original_import = builtins.__import__

    def missing_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ModuleNotFoundError("synthetic missing numpy", name=name)
        return original_import(name, *args, **kwargs)

    def forbidden_request(*args, **kwargs):
        pytest.fail("availability diagnostics must not send an API request")

    monkeypatch.setattr(builtins, "__import__", missing_numpy)
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", forbidden_request)
    assert not embeddings.available()
    with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
        hermes_memory_provider._announce_embedding_engine_unavailable()
    warnings = [r.getMessage() for r in caplog.records if r.name == "hermes_memory_provider"]
    assert len(warnings) == 1
    text = warnings[0]
    assert "NumPy is unavailable for the embedding API backend" in text
    assert "install or repair NumPy" in text
    assert "no usable credentials" not in text
    assert "MNEMOSYNE_EMBEDDING_API_KEY" not in text
    assert "missing fastembed" not in text
    assert ("missing sqlite_vec" in text) is (not vec_available)


def test_older_core_without_numpy_probe_keeps_credentials_diagnostic(monkeypatch, tmp_path, caplog):
    """A missing optional Core helper must not suppress the existing startup warning."""
    from mnemosyne.core import embeddings

    _engine_capability(monkeypatch, dense=False, vec=True)
    monkeypatch.delattr(embeddings, "_ensure_api_numpy")
    monkeypatch.setattr(embeddings, "_DEFAULT_MODEL", "openai/text-embedding-3-small")
    with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
        hermes_memory_provider._announce_embedding_engine_unavailable()
    assert "embedding API backend has no usable credentials" in caplog.text
    assert "MNEMOSYNE_EMBEDDING_API_KEY" in caplog.text
