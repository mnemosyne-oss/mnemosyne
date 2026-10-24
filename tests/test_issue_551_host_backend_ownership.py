"""Regression coverage for #551 and #1119 host-backend provider ownership.

Both Hermes provider paths (the root ``hermes_memory_provider`` package and the
``mnemosyne_hermes`` integration) must keep the process-global host backend
until the last successfully initialized primary provider shuts down.
"""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path

import pytest

from mnemosyne.core.llm_backends import get_host_llm_backend, set_host_llm_backend


INTEGRATION_SRC = Path(__file__).resolve().parents[1] / "integrations" / "hermes" / "src"


@pytest.fixture(params=("mnemosyne_hermes", "hermes_memory_provider"))
def provider_class(request, tmp_path, monkeypatch):
    # The root provider resolves its database from MNEMOSYNE_DATA_DIR rather
    # than hermes_home, so pin it to the test directory for both paths.
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    if request.param == "hermes_memory_provider":
        module = importlib.import_module("hermes_memory_provider")
        yield module.MnemosyneMemoryProvider
        return
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


def test_double_shutdown_releases_one_lease(tmp_path, provider_class):
    """A repeated shutdown must not release a live peer's lease."""
    first = provider_class()
    peer = provider_class()
    module = sys.modules[provider_class.__module__]
    try:
        first.initialize("first", hermes_home=str(tmp_path / "first"), agent_context="primary")
        peer.initialize("peer", hermes_home=str(tmp_path / "peer"), agent_context="primary")
        assert module._host_llm_owner_count == 2

        first.shutdown()
        first.shutdown()
        assert module._host_llm_owner_count == 1
        assert get_host_llm_backend() is not None

        peer.shutdown()
        assert module._host_llm_owner_count == 0
        assert get_host_llm_backend() is None
    finally:
        first.shutdown()
        peer.shutdown()
        set_host_llm_backend(None)


def test_primary_reinitialization_keeps_a_single_lease(tmp_path, provider_class):
    provider = provider_class()
    module = sys.modules[provider_class.__module__]
    try:
        provider.initialize("one", hermes_home=str(tmp_path / "one"), agent_context="primary")
        provider.initialize("two", hermes_home=str(tmp_path / "two"), agent_context="primary")
        assert provider._owns_host_llm_backend is True
        assert module._host_llm_owner_count == 1

        provider.shutdown()
        assert module._host_llm_owner_count == 0
        assert get_host_llm_backend() is None
    finally:
        provider.shutdown()
        set_host_llm_backend(None)


def test_primary_to_skip_reinitialization_drops_lease(tmp_path, provider_class):
    owner = provider_class()
    switched = provider_class()
    module = sys.modules[provider_class.__module__]
    try:
        owner.initialize("owner", hermes_home=str(tmp_path / "owner"), agent_context="primary")
        switched.initialize("primary", hermes_home=str(tmp_path / "switched"), agent_context="primary")
        assert module._host_llm_owner_count == 2

        switched.initialize("subagent", hermes_home=str(tmp_path / "switched"), agent_context="subagent")
        assert switched._owns_host_llm_backend is False
        assert module._host_llm_owner_count == 1

        switched.shutdown()
        assert get_host_llm_backend() is not None

        owner.shutdown()
        assert get_host_llm_backend() is None
    finally:
        switched.shutdown()
        owner.shutdown()
        set_host_llm_backend(None)


def test_concurrent_peer_shutdowns_clear_once_after_final_owner(tmp_path, provider_class):
    providers = [provider_class() for _ in range(4)]
    module = sys.modules[provider_class.__module__]
    try:
        for index, provider in enumerate(providers):
            provider.initialize(
                f"session-{index}",
                hermes_home=str(tmp_path / f"hermes-{index}"),
                agent_context="primary",
            )
        assert module._host_llm_owner_count == len(providers)

        live = providers[0]
        barrier = threading.Barrier(len(providers) - 1)

        def stop(provider):
            barrier.wait(timeout=5)
            provider.shutdown()

        threads = [threading.Thread(target=stop, args=(p,)) for p in providers[1:]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

        assert module._host_llm_owner_count == 1
        assert get_host_llm_backend() is not None

        live.shutdown()
        assert module._host_llm_owner_count == 0
        assert get_host_llm_backend() is None
    finally:
        for provider in providers:
            provider.shutdown()
        set_host_llm_backend(None)


def test_acquire_waits_for_final_release_to_finish(tmp_path, provider_class, monkeypatch):
    """A new owner registering during the final clear must keep its backend."""
    adapter_name = (
        "hermes_memory_provider.hermes_llm_adapter"
        if provider_class.__module__ == "hermes_memory_provider"
        else "mnemosyne_hermes.hermes_llm_adapter"
    )
    adapter = importlib.import_module(adapter_name)
    real_unregister = adapter.unregister_hermes_host_llm
    clearing = threading.Event()
    resume = threading.Event()

    def held_unregister():
        clearing.set()
        resume.wait(timeout=5)
        real_unregister()

    leaving = provider_class()
    joining = provider_class()
    try:
        leaving.initialize("leaving", hermes_home=str(tmp_path / "leaving"), agent_context="primary")
        monkeypatch.setattr(adapter, "unregister_hermes_host_llm", held_unregister)

        stopper = threading.Thread(target=leaving.shutdown)
        stopper.start()
        assert clearing.wait(timeout=5)

        joiner = threading.Thread(target=joining._acquire_host_llm_backend_ownership)
        joiner.start()
        joiner.join(timeout=0.2)
        assert joiner.is_alive()
        assert joining._owns_host_llm_backend is False

        resume.set()
        stopper.join(timeout=5)
        joiner.join(timeout=5)
        assert not stopper.is_alive()
        assert not joiner.is_alive()

        assert joining._owns_host_llm_backend is True
        assert get_host_llm_backend() is not None
    finally:
        resume.set()
        monkeypatch.setattr(adapter, "unregister_hermes_host_llm", real_unregister)
        leaving.shutdown()
        joining.shutdown()
        set_host_llm_backend(None)
