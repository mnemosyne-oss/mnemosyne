"""Registry, gate, and degradation tests for the modality seam (RFC 0002 §3.1).

The load-bearing assertion in this file is negative: **with the flag unset,
nothing happens.** Not "an error is returned" -- nothing is called, nothing is
sent, and the caller carries on.
"""

import pytest

from mnemosyne.core import modality_backends as mb
from mnemosyne.core.modality_backends import (
    CallableModalityBackend,
    DescribedMoment,
    DescribeRequest,
    DescribeResult,
    call_modality_describe,
    clear_modality_backends,
    get_modality_backend,
    modality_enabled,
    set_modality_backend,
)


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    """Opt in the way a real operator must.

    Setting only ``MNEMOSYNE_MODALITY_ENABLED`` is not enough. ``config.yaml``
    wins over env vars, and presence beats value: once a key exists in the
    file, the variable is never consulted. A seeded config carries
    ``modality_enabled: false``, so an env-only fixture passes on a machine
    whose config predates the key and fails on a fresh one, which is exactly
    how this slipped through locally and failed in CI.

    Pointing ``MNEMOSYNE_DATA_DIR`` at a fresh directory makes the seed honour
    the variables, which is the same path a new install takes.
    """
    from mnemosyne.core.config import MnemosyneConfig
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg-enabled"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()


def _request(modality="image", uri="https://example.test/a.png", **kwargs):
    return DescribeRequest(modality=modality, uri=uri, **kwargs)


def _recording_backend(name="stub", modalities=mb.MODALITIES, result=None):
    calls = []

    def _describe(request):
        calls.append(request)
        return result if result is not None else DescribeResult(
            summary="a stub", moments=[DescribedMoment(kind="caption", text="a stub")],
            provider=name,
        )

    return CallableModalityBackend(name=name, func=_describe, modalities=modalities), calls


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_disabled_by_default():
    """Mnemosyne makes no outbound call to describe media until the operator
    has said yes, once, explicitly."""
    assert modality_enabled() is False


def test_nothing_is_called_when_disabled():
    """Not 'an error is returned' -- the backend is never invoked at all, even
    though it is registered. A host that registers eagerly must not be able to
    dial out on behalf of a user who never opted in."""
    backend, calls = _recording_backend()
    set_modality_backend(backend)
    assert call_modality_describe(_request()) is None
    assert calls == [], "the gate must be checked before the registry"


def test_the_gate_is_read_at_call_time_not_import_time(monkeypatch, tmp_path):
    """``local_llm.py:54`` reads its flag at import, which makes it
    untestable. Repeating that here would fail in the direction where the
    privacy test passes and production dials out.

    Each phase gets its own data directory. A config is seeded on first read,
    and presence in the file beats the env var from then on, so flipping the
    variable inside one directory would prove nothing.
    """
    from mnemosyne.core.config import MnemosyneConfig

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "off"))
    MnemosyneConfig.reset_instance()
    assert modality_enabled() is False

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "on"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
    MnemosyneConfig.reset_instance()
    try:
        assert modality_enabled() is True
    finally:
        MnemosyneConfig.reset_instance()


def test_an_unreadable_config_is_a_no(monkeypatch):
    """Failing closed: if the flag cannot be read, it is off."""
    def _explode(*args, **kwargs):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("mnemosyne.core.config.get_bool", _explode)
    assert modality_enabled() is False


def test_describe_runs_once_enabled(enabled):
    backend, calls = _recording_backend()
    set_modality_backend(backend)
    result = call_modality_describe(_request())
    assert result is not None and result.summary == "a stub"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_per_modality_registration_wins_over_the_default(enabled):
    default, default_calls = _recording_backend("default")
    audio, audio_calls = _recording_backend("audio-only", modalities=frozenset({"audio"}))
    set_modality_backend(default)
    set_modality_backend(audio, modalities=frozenset({"audio"}))

    call_modality_describe(_request("audio"))
    call_modality_describe(_request("image"))

    assert len(audio_calls) == 1
    assert len(default_calls) == 1
    assert audio_calls[0].modality == "audio"


def test_a_default_is_never_handed_a_modality_it_does_not_declare(enabled):
    """The failure mode this prevents is not an error -- it is an image model
    confidently captioning an mp3."""
    vision, calls = _recording_backend("vision", modalities=frozenset({"image"}))
    set_modality_backend(vision)

    assert get_modality_backend("image") is vision
    assert get_modality_backend("audio") is None
    assert call_modality_describe(_request("audio")) is None
    assert calls == []


def test_modality_lookup_is_case_and_whitespace_insensitive(enabled):
    backend, calls = _recording_backend(modalities=frozenset({"image"}))
    set_modality_backend(backend, modalities=frozenset({"IMAGE"}))
    assert get_modality_backend("  Image ") is backend


def test_clearing_the_default(enabled):
    backend, _ = _recording_backend()
    set_modality_backend(backend)
    set_modality_backend(None)
    assert get_modality_backend("image") is None


def test_clearing_specific_modalities_leaves_the_default(enabled):
    default, _ = _recording_backend("default")
    audio, _ = _recording_backend("audio", modalities=frozenset({"audio"}))
    set_modality_backend(default)
    set_modality_backend(audio, modalities=frozenset({"audio"}))

    set_modality_backend(None, modalities=frozenset({"audio"}))
    assert get_modality_backend("audio") is default


def test_clear_resets_everything(enabled):
    backend, _ = _recording_backend()
    set_modality_backend(backend)
    set_modality_backend(backend, modalities=frozenset({"audio"}))
    clear_modality_backends()
    assert get_modality_backend("image") is None
    assert get_modality_backend("audio") is None


def test_the_registry_does_not_leak_between_tests():
    """Guards the conftest reset. If this fails, an earlier test's backend is
    still armed and every privacy assertion in the suite is suspect."""
    assert get_modality_backend("image") is None
    assert mb._default is None
    assert mb._backends == {}


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def test_a_raising_backend_degrades_to_none(enabled):
    """Rung 4 of the ladder depends on this: the caller registers the asset by
    reference regardless, so a broken provider costs the user nothing."""
    def _explode(request):
        raise RuntimeError("provider on fire")

    set_modality_backend(CallableModalityBackend(name="broken", func=_explode))
    assert call_modality_describe(_request()) is None


def test_a_backend_returning_none_is_not_an_error(enabled):
    set_modality_backend(CallableModalityBackend(name="quiet", func=lambda r: None))
    assert call_modality_describe(_request()) is None


def test_no_backend_is_not_an_error(enabled):
    assert call_modality_describe(_request()) is None


def test_backend_failures_do_not_log_the_request(enabled, caplog):
    """A request carries a URI, which can be a local file path."""
    def _explode(request):
        raise RuntimeError("provider on fire")

    set_modality_backend(CallableModalityBackend(name="broken", func=_explode))
    with caplog.at_level("DEBUG"):
        call_modality_describe(_request(uri="/home/someone/private/photo.png"))
    assert "/home/someone/private" not in caplog.text


# ---------------------------------------------------------------------------
# Wire shapes
# ---------------------------------------------------------------------------

def test_fetch_is_lazy_and_optional(enabled):
    """A backend describing a public URL never reads bytes."""
    fetched = []

    def _fetch():
        fetched.append(True)
        return b"bytes"

    backend, calls = _recording_backend()
    set_modality_backend(backend)
    call_modality_describe(_request(fetch=_fetch))

    assert fetched == [], "the registry must not call fetch itself"
    assert calls[0].fetch is _fetch, "but it must pass it through"


def test_refused_is_distinct_from_failure(enabled):
    """A provider that declines will decline again; one that failed is worth
    retrying. Collapsing them loses that."""
    refusal = DescribeResult(provider="p", refused=True)
    set_modality_backend(CallableModalityBackend(name="p", func=lambda r: refusal))
    result = call_modality_describe(_request())
    assert result is not None and result.refused is True
    assert result.moments == []


def test_describe_result_defaults_are_empty_not_none():
    result = DescribeResult()
    assert (result.moments, result.warnings, result.refused) == ([], [], False)


def test_described_moment_is_not_the_storage_shape():
    """The provider wire shape must not let a backend set identity or binding
    -- those belong to the store."""
    fields = set(DescribedMoment(kind="caption", text="x").__dict__)
    assert "memory_id" not in fields
    assert "moment_id" not in fields
    assert "extra" in fields
