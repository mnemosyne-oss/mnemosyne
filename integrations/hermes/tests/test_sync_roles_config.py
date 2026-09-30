from __future__ import annotations

import logging

import mnemosyne_hermes
import pytest
from mnemosyne_hermes import MnemosyneMemoryProvider

INVALID_SYNC_ROLES_WARNING = (
    "Mnemosyne: invalid sync_roles configuration; expected a comma-separated "
    "string or a list, tuple, or set containing valid roles (user, assistant). "
    "Conversation autosave remains disabled."
)


@pytest.fixture(autouse=True)
def _isolate_sync_roles_config(monkeypatch, tmp_path):
    from mnemosyne.core.config import MnemosyneConfig

    monkeypatch.delenv("MNEMOSYNE_SYNC_ROLES", raising=False)
    core_config_path = tmp_path / "mnemosyne-config.yaml"
    core_config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        MnemosyneConfig,
        "_instance",
        MnemosyneConfig(config_path=core_config_path),
    )


class _RecordingBeam:
    author_id = "test-author"

    def __init__(self, *, session_id, db_path):
        self.session_id = session_id
        self.channel_id = session_id
        self.writes = []

    def remember(self, **kwargs):
        self.writes.append(kwargs)
        return None


def _initialize_provider(monkeypatch, tmp_path, *, sync_roles=None):
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    kwargs = {
        "hermes_home": str(tmp_path),
        "agent_context": "primary",
        "auto_sleep": False,
    }
    if sync_roles is not None:
        kwargs["sync_roles"] = sync_roles
    provider.initialize("session-1", **kwargs)
    assert provider._beam is not None
    return provider


def _sync_role_warnings(caplog):
    return [record.getMessage() for record in caplog.records if "sync_roles" in record.getMessage()]


def test_stringified_list_from_provider_config_warns_and_disables_sync(
    monkeypatch, tmp_path, caplog
):
    raw_value = "['user', 'assistant']"
    (tmp_path / "config.yaml").write_text(
        f'memory:\n  provider: mnemosyne\n  mnemosyne:\n    sync_roles: "{raw_value}"\n',
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path)
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == set()
    assert provider._beam.writes == []
    warnings = _sync_role_warnings(caplog)
    assert warnings == [INVALID_SYNC_ROLES_WARNING]
    assert raw_value not in warnings[0]


def test_native_yaml_list_from_provider_config_saves_both_roles_without_warning(
    monkeypatch, tmp_path, caplog
):
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n  mnemosyne:\n    sync_roles:\n      - user\n      - assistant\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path)
    beam = provider._beam
    assert isinstance(beam, _RecordingBeam)
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == {"user", "assistant"}
    assert [write["content"].split()[0] for write in beam.writes] == [
        "[USER]",
        "[ASSISTANT]",
    ]
    assert _sync_role_warnings(caplog) == []


@pytest.mark.parametrize(
    "configured_roles",
    [
        pytest.param(" USER, assistant ", id="csv"),
        pytest.param(["USER", "assistant"], id="native-list"),
    ],
)
def test_supported_sync_role_forms_save_both_roles_without_warning(
    monkeypatch, tmp_path, caplog, configured_roles
):
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path, sync_roles=configured_roles)
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == {"user", "assistant"}
    assert [write["content"].split()[0] for write in provider._beam.writes] == [
        "[USER]",
        "[ASSISTANT]",
    ]
    assert _sync_role_warnings(caplog) == []


@pytest.mark.parametrize(
    "configured_roles",
    [
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
        pytest.param(set(), id="empty-set"),
    ],
)
def test_explicit_empty_sync_roles_disable_sync_without_warning(
    monkeypatch, tmp_path, caplog, configured_roles
):
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path, sync_roles=configured_roles)
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == set()
    assert provider._beam.writes == []
    assert _sync_role_warnings(caplog) == []


@pytest.mark.parametrize(
    "configured_roles",
    [
        pytest.param("unknown", id="unknown-only-csv"),
        pytest.param(["unknown"], id="unknown-only-list"),
        pytest.param(42, id="wrong-type-scalar"),
    ],
)
def test_invalid_nonempty_sync_roles_fail_closed_with_one_safe_warning(
    monkeypatch, tmp_path, caplog, configured_roles
):
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path, sync_roles=configured_roles)
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == set()
    assert provider._beam.writes == []
    warnings = _sync_role_warnings(caplog)
    assert warnings == [INVALID_SYNC_ROLES_WARNING]
    assert repr(configured_roles) not in warnings[0]


def test_mixed_valid_and_unknown_sync_roles_keep_allowed_role(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path, sync_roles=["USER", "unknown"])
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == {"user"}
    assert [write["content"].split()[0] for write in provider._beam.writes] == ["[USER]"]
    assert _sync_role_warnings(caplog) == []


def test_invalid_env_sync_roles_warns_once_and_writes_nothing(monkeypatch, tmp_path, caplog):
    raw_value = "['user', 'assistant'] secret-conversation-marker"
    monkeypatch.setenv("MNEMOSYNE_SYNC_ROLES", raw_value)
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path)
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == set()
    assert provider._beam.writes == []
    assert _sync_role_warnings(caplog) == [INVALID_SYNC_ROLES_WARNING]
    assert raw_value not in caplog.text
    assert "secret-conversation-marker" not in caplog.text


def test_config_override_ignores_invalid_env_without_warning(monkeypatch, tmp_path, caplog):
    raw_value = "invalid-secret-env-value"
    monkeypatch.setenv("MNEMOSYNE_SYNC_ROLES", raw_value)
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n  mnemosyne:\n    sync_roles:\n      - user\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path)
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == {"user"}
    assert [write["content"].split()[0] for write in provider._beam.writes] == ["[USER]"]
    assert _sync_role_warnings(caplog) == []
    assert raw_value not in caplog.text


def test_sync_roles_precedence_recomputes_kwargs_config_env_and_default(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("MNEMOSYNE_SYNC_ROLES", "assistant")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "memory:\n  provider: mnemosyne\n  mnemosyne:\n    sync_roles:\n      - user\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")

    provider = _initialize_provider(monkeypatch, tmp_path, sync_roles="assistant")
    assert provider._sync_roles == {"assistant"}

    provider.initialize(
        "session-2", hermes_home=str(tmp_path), agent_context="primary", auto_sleep=False
    )
    assert provider._sync_roles == {"user"}

    config_path.unlink()
    provider.initialize(
        "session-3", hermes_home=str(tmp_path), agent_context="primary", auto_sleep=False
    )
    assert provider._sync_roles == {"assistant"}

    monkeypatch.delenv("MNEMOSYNE_SYNC_ROLES")
    provider.initialize(
        "session-4", hermes_home=str(tmp_path), agent_context="primary", auto_sleep=False
    )
    assert provider._sync_roles == {"user"}
    assert _sync_role_warnings(caplog) == []


@pytest.mark.parametrize(
    "initial_override",
    [
        pytest.param("assistant", id="assistant-only"),
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-list"),
    ],
)
def test_reinitialize_without_override_does_not_retain_stale_sync_roles(
    monkeypatch, tmp_path, caplog, initial_override
):
    caplog.set_level(logging.WARNING, logger="mnemosyne_hermes")
    provider = _initialize_provider(monkeypatch, tmp_path, sync_roles=initial_override)

    provider.initialize(
        "session-2", hermes_home=str(tmp_path), agent_context="primary", auto_sleep=False
    )
    provider.sync_turn(
        "please remember this user message",
        "please remember this assistant response",
    )

    assert provider._sync_roles == {"user"}
    assert [write["content"].split()[0] for write in provider._beam.writes] == ["[USER]"]
    assert _sync_role_warnings(caplog) == []
