"""Regression coverage for #551 host-backend provider ownership."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mnemosyne.core.llm_backends import get_host_llm_backend, set_host_llm_backend


INTEGRATION_SRC = Path(__file__).resolve().parents[1] / "integrations" / "hermes" / "src"


@pytest.fixture
def provider_class():
    sys.path.insert(0, str(INTEGRATION_SRC))
    try:
        from mnemosyne_hermes import MnemosyneMemoryProvider

        yield MnemosyneMemoryProvider
    finally:
        sys.path.remove(str(INTEGRATION_SRC))


@pytest.mark.parametrize("first_to_shutdown", (0, 1))
def test_primary_peer_shutdown_keeps_host_backend_until_final_owner(
    tmp_path, provider_class, first_to_shutdown
):
    """A live primary peer retains the global host backend after its peer exits."""
    providers = [provider_class(), provider_class()]
    try:
        for index, provider in enumerate(providers):
            provider.initialize(
                f"session-{index}",
                hermes_home=str(tmp_path / f"hermes-{index}"),
                agent_context="primary",
            )
            assert provider._beam is not None

        assert get_host_llm_backend() is not None

        providers[first_to_shutdown].shutdown()
        assert get_host_llm_backend() is not None

        providers[1 - first_to_shutdown].shutdown()
        assert get_host_llm_backend() is None
    finally:
        for provider in providers:
            provider.shutdown()
        set_host_llm_backend(None)


def test_skip_context_never_releases_primary_owner(tmp_path, provider_class):
    owner = provider_class()
    skipped = provider_class()
    try:
        owner.initialize("owner", hermes_home=str(tmp_path / "owner"), agent_context="primary")
        skipped.initialize("skip", hermes_home=str(tmp_path / "skip"), agent_context="cron")

        skipped.shutdown()
        assert get_host_llm_backend() is not None

        owner.shutdown()
        assert get_host_llm_backend() is None
    finally:
        skipped.shutdown()
        owner.shutdown()
        set_host_llm_backend(None)


def test_failed_initialization_never_releases_primary_owner(tmp_path, provider_class, monkeypatch):
    owner = provider_class()
    failed = provider_class()
    module = sys.modules[provider_class.__module__]
    try:
        owner.initialize("owner", hermes_home=str(tmp_path / "owner"), agent_context="primary")
        monkeypatch.setattr(module, "_get_beam_class", lambda: _raise_init_failure)
        failed.initialize("failed", hermes_home=str(tmp_path / "failed"), agent_context="primary")
        assert failed._beam is None

        failed.shutdown()
        assert get_host_llm_backend() is not None

        owner.shutdown()
        assert get_host_llm_backend() is None
    finally:
        failed.shutdown()
        owner.shutdown()
        set_host_llm_backend(None)


def _raise_init_failure(**kwargs):
    raise RuntimeError("synthetic initialization failure")


def test_first_primary_init_failure_clears_unowned_backend(tmp_path, provider_class, monkeypatch):
    """A first failed primary init must not leave a registered unowned backend."""
    provider = provider_class()
    module = sys.modules[provider_class.__module__]
    try:
        assert module._host_llm_owner_count == 0
        assert get_host_llm_backend() is None
        monkeypatch.setattr(module, "_get_beam_class", lambda: _raise_init_failure)

        provider.initialize("failed", hermes_home=str(tmp_path / "failed"), agent_context="primary")

        assert provider._beam is None
        assert provider._owns_host_llm_backend is False
        assert module._host_llm_owner_count == 0
        assert get_host_llm_backend() is None

        provider.shutdown()
        assert provider._owns_host_llm_backend is False
        assert module._host_llm_owner_count == 0
        assert get_host_llm_backend() is None
    finally:
        provider.shutdown()
        set_host_llm_backend(None)


def test_failed_reinitialization_releases_prior_owner(tmp_path, provider_class, monkeypatch):
    provider = provider_class()
    module = sys.modules[provider_class.__module__]
    try:
        provider.initialize("healthy", hermes_home=str(tmp_path / "healthy"), agent_context="primary")
        assert get_host_llm_backend() is not None

        monkeypatch.setattr(module, "_get_beam_class", lambda: _raise_init_failure)
        provider.initialize("failed", hermes_home=str(tmp_path / "failed"), agent_context="primary")

        assert provider._beam is None
        assert get_host_llm_backend() is None
    finally:
        provider.shutdown()
        set_host_llm_backend(None)
