"""Tests for sync_roles config in sync_turn()."""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch


class TestSyncRoles:
    """Verify sync_roles controls which conversation roles are autosaved."""

    @pytest.fixture
    def provider(self):
        from hermes_memory_provider import MnemosyneMemoryProvider
        from mnemosyne.core.beam import BeamMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "mnemosyne.db")
            provider = MnemosyneMemoryProvider()
            provider._agent_context = "test"
            provider._session_id = "test-session"
            provider._skip_contexts = set()
            provider._turn_count = 0
            provider._auto_sleep_enabled = False
            provider._beam = BeamMemory(db_path=db_path)

            yield provider

            try:
                os.remove(db_path)
                for ext in ("-wal", "-shm"):
                    try:
                        os.remove(db_path + ext)
                    except OSError:
                        pass
            except OSError:
                pass

    def test_default_saves_both_roles(self, provider):
        """Default sync_roles saves both user and assistant turns."""
        provider._beam.remember = MagicMock()
        provider.sync_turn("Tell me about memory systems", "Here is what I know about memory.")

        sources = [c.kwargs.get("content", "") for c in provider._beam.remember.call_args_list]
        user_calls = [s for s in sources if s.startswith("[USER]")]
        assistant_calls = [s for s in sources if s.startswith("[ASSISTANT]")]
        assert len(user_calls) == 1
        assert len(assistant_calls) == 1

    def test_user_only(self, provider):
        """sync_roles=['user'] saves user turns, skips assistant."""
        provider._sync_roles = {"user"}
        provider._beam.remember = MagicMock()
        provider.sync_turn("Tell me about memory systems", "Here is what I know about memory.")

        sources = [c.kwargs.get("content", "") for c in provider._beam.remember.call_args_list]
        user_calls = [s for s in sources if s.startswith("[USER]")]
        assistant_calls = [s for s in sources if s.startswith("[ASSISTANT]")]
        assert len(user_calls) == 1
        assert len(assistant_calls) == 0

    def test_assistant_only(self, provider):
        """sync_roles=['assistant'] saves assistant turns, skips user."""
        provider._sync_roles = {"assistant"}
        provider._beam.remember = MagicMock()
        provider.sync_turn("Tell me about memory systems", "Here is what I know about memory.")

        sources = [c.kwargs.get("content", "") for c in provider._beam.remember.call_args_list]
        user_calls = [s for s in sources if s.startswith("[USER]")]
        assistant_calls = [s for s in sources if s.startswith("[ASSISTANT]")]
        assert len(user_calls) == 0
        assert len(assistant_calls) == 1

    def test_empty_disables_autosave(self, provider):
        """sync_roles=[] disables all conversation autosave."""
        provider._sync_roles = set()
        provider._beam.remember = MagicMock()
        provider.sync_turn("Tell me about memory systems", "Here is what I know about memory.")

        conversation_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "conversation"
        ]
        assert len(conversation_calls) == 0

    def test_empty_disables_identity_capture(self, provider):
        """sync_roles=[] also disables identity signal capture."""
        provider._sync_roles = set()
        provider._beam.remember = MagicMock()
        provider.sync_turn("I feel like an imposter at work", "That is understandable.")

        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) == 0

    def test_assistant_only_disables_identity_capture(self, provider):
        """sync_roles=['assistant'] disables identity capture (derived from user content)."""
        provider._sync_roles = {"assistant"}
        provider._beam.remember = MagicMock()
        provider.sync_turn("I feel like an imposter at work", "That is understandable.")

        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) == 0

    def test_user_only_preserves_identity_capture(self, provider):
        """sync_roles=['user'] still captures identity signals."""
        provider._sync_roles = {"user"}
        provider._beam.remember = MagicMock()
        provider.sync_turn("I feel like an imposter at work", "That is understandable.")

        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) >= 1

    def test_ignore_patterns_still_apply(self, provider):
        """sync_roles does not bypass ignore_patterns filtering."""
        provider._sync_roles = {"user", "assistant"}
        provider._ignore_patterns = [r"^Done\.?$"]
        provider._beam.remember = MagicMock()
        provider.sync_turn("Done.", "Noted.")

        conversation_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "conversation"
        ]
        user_calls = [c for c in conversation_calls if "[USER]" in c.kwargs.get("content", "")]
        assert len(user_calls) == 0

    def test_skip_contexts_overrides_sync_roles(self, provider):
        """skip_contexts still disables everything regardless of sync_roles."""
        provider._sync_roles = {"user", "assistant"}
        provider._agent_context = "cron"
        provider._skip_contexts = {"cron"}
        provider._beam.remember = MagicMock()
        provider.sync_turn("Important user message here", "Important assistant response here")

        assert provider._beam.remember.call_count == 0

    def test_turn_counter_increments_regardless(self, provider):
        """Turn counter increments even when sync_roles filters both roles."""
        provider._sync_roles = set()
        provider._beam.remember = MagicMock()
        assert provider._turn_count == 0
        provider.sync_turn("Hello there", "Hi, how can I help?")
        assert provider._turn_count == 1


class TestSyncRolesConfig:
    """Verify sync_roles config parsing in _apply_provider_config."""

    @pytest.fixture
    def provider(self):
        from hermes_memory_provider import MnemosyneMemoryProvider
        provider = MnemosyneMemoryProvider()
        provider._hermes_home = ""
        return provider

    def test_config_list(self, provider):
        provider._apply_provider_config({"sync_roles": ["user"]})
        assert provider._sync_roles == {"user"}

    def test_config_empty_list(self, provider):
        provider._apply_provider_config({"sync_roles": []})
        assert provider._sync_roles == set()

    def test_config_csv_string(self, provider):
        provider._apply_provider_config({"sync_roles": "user,assistant"})
        assert provider._sync_roles == {"user", "assistant"}

    def test_config_empty_string(self, provider):
        provider._apply_provider_config({"sync_roles": ""})
        assert provider._sync_roles == set()

    def test_unknown_roles_ignored(self, provider):
        provider._apply_provider_config({"sync_roles": ["user", "system", "tool"]})
        assert provider._sync_roles == {"user"}

    def test_not_set_preserves_default(self, provider):
        default = provider._sync_roles.copy()
        provider._apply_provider_config({})
        assert provider._sync_roles == default
        assert provider._sync_roles == {"user", "assistant"}

    def test_case_insensitive(self, provider):
        provider._apply_provider_config({"sync_roles": ["User", "ASSISTANT"]})
        assert provider._sync_roles == {"user", "assistant"}

    @pytest.mark.parametrize("value", [True, False, 42, 3.14, {"user": True}])
    def test_invalid_type_preserves_default(self, provider, value):
        provider._apply_provider_config({"sync_roles": value})
        assert provider._sync_roles == {"user", "assistant"}

    def test_unknown_only_preserves_default(self, provider):
        """Non-empty input with zero valid roles preserves default (typo safety)."""
        provider._apply_provider_config({"sync_roles": ["users", "system"]})
        assert provider._sync_roles == {"user", "assistant"}


class TestSyncRolesEnvVar:
    """Verify MNEMOSYNE_SYNC_ROLES env var support."""

    def test_env_var_user_only(self):
        from hermes_memory_provider import MnemosyneMemoryProvider
        with patch.dict(os.environ, {"MNEMOSYNE_SYNC_ROLES": "user"}):
            provider = MnemosyneMemoryProvider()
            assert provider._sync_roles == {"user"}

    def test_env_var_empty_disables(self):
        from hermes_memory_provider import MnemosyneMemoryProvider
        with patch.dict(os.environ, {"MNEMOSYNE_SYNC_ROLES": ""}):
            provider = MnemosyneMemoryProvider()
            assert provider._sync_roles == set()

    def test_env_var_not_set_uses_default(self):
        from hermes_memory_provider import MnemosyneMemoryProvider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MNEMOSYNE_SYNC_ROLES", None)
            provider = MnemosyneMemoryProvider()
            assert provider._sync_roles == {"user", "assistant"}
