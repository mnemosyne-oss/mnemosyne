"""Task 1 / Wave 1 P0: maintenance config via central hot-reload; Dream mode
disables sleep model-refresh auto-apply.

``model_refresh.auto_apply_enabled()`` must read through the central
hot-reload config (config.yaml > env > default) so a maintenance operation
observes one consistent value at its boundary, and must be forced off when
verified Dream mode is active (``dream_active``) so the sleep path does not
race Dream's canonical mutations.
"""

import pytest

from mnemosyne.core import config as config_module
from mnemosyne.core import model_refresh


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the central config at a throwable data dir and reset the singleton
    between tests so each case sees a clean config.yaml."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    yield
    config_module.MnemosyneConfig.reset_instance()


class TestAutoApplyCentralConfig:
    def test_reads_central_config_yaml_false(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "sleep_model_refresh_auto_apply: false\n", encoding="utf-8"
        )
        assert model_refresh.auto_apply_enabled() is False

    def test_reads_central_config_yaml_true(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "sleep_model_refresh_auto_apply: true\n", encoding="utf-8"
        )
        assert model_refresh.auto_apply_enabled() is True

    def test_yaml_overrides_env(self, tmp_path, monkeypatch):
        # config.yaml wins over env (config.yaml > env > default).
        (tmp_path / "config.yaml").write_text(
            "sleep_model_refresh_auto_apply: false\n", encoding="utf-8"
        )
        monkeypatch.setenv("MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY", "true")
        assert model_refresh.auto_apply_enabled() is False

    def test_env_used_when_no_yaml(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY", "false")
        assert model_refresh.auto_apply_enabled() is False

    def test_default_true_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY", raising=False)
        assert model_refresh.auto_apply_enabled() is True


class TestDreamModeGatesAutoApply:
    def test_dream_active_forces_auto_apply_off(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "sleep_model_refresh_auto_apply: true\ndream_active: true\n",
            encoding="utf-8",
        )
        assert model_refresh.auto_apply_enabled() is False

    def test_dream_inactive_keeps_auto_apply_on(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "sleep_model_refresh_auto_apply: true\ndream_active: false\n",
            encoding="utf-8",
        )
        assert model_refresh.auto_apply_enabled() is True

    def test_dream_active_env_flag_forces_off(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DREAM_ACTIVE", "true")
        assert model_refresh.auto_apply_enabled() is False

    def test_yaml_false_overrides_dream_active_env_true(self, tmp_path, monkeypatch):
        # config.yaml wins over env (config.yaml > env > default).
        (tmp_path / "config.yaml").write_text(
            "sleep_model_refresh_auto_apply: true\ndream_active: false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MNEMOSYNE_DREAM_ACTIVE", "true")
        assert model_refresh.auto_apply_enabled() is True

    def test_dream_active_hot_reload_observes_update(self, tmp_path):
        import os
        import time

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "sleep_model_refresh_auto_apply: true\ndream_active: false\n",
            encoding="utf-8",
        )
        assert model_refresh.auto_apply_enabled() is True

        time.sleep(0.05)
        cfg_file.write_text(
            "sleep_model_refresh_auto_apply: true\ndream_active: true\n",
            encoding="utf-8",
        )
        os.utime(str(cfg_file), (time.time() + 2, time.time() + 2))
        assert model_refresh.auto_apply_enabled() is False


class TestInferModelUpdateProposalsFallback:
    def test_host_llm_success_does_not_call_remote(self, monkeypatch):
        from mnemosyne.core import local_llm

        remote_called = False
        valid_json = (
            '[{"category": "model:user", "name": "editor", "confidence": 0.9, "body": "uses vim", "evidence_ids": ["m1"]}]'
        )

        def fake_host(prompt, max_tokens=2048, temperature=0.1):
            return True, valid_json

        def fake_remote(prompt, temperature=0.1):
            nonlocal remote_called
            remote_called = True
            return ""

        monkeypatch.setattr(local_llm, "_try_host_llm", fake_host)
        monkeypatch.setattr(local_llm, "_call_remote_llm", fake_remote)

        proposals = model_refresh.infer_model_update_proposals([{"content": "user likes vim"}])
        assert len(proposals) == 1
        assert proposals[0]["name"] == "editor"
        assert not remote_called

    def test_host_unattempted_falls_back_to_remote(self, monkeypatch):
        from mnemosyne.core import local_llm

        remote_called = False
        valid_json = (
            '[{"category": "model:user", "name": "shell", "confidence": 0.85, "body": "uses zsh", "evidence_ids": ["m2"]}]'
        )

        def fake_host(prompt, max_tokens=2048, temperature=0.1):
            return False, None

        def fake_remote(prompt, temperature=0.1):
            nonlocal remote_called
            remote_called = True
            return valid_json

        monkeypatch.setattr(local_llm, "_try_host_llm", fake_host)
        monkeypatch.setattr(local_llm, "_call_remote_llm", fake_remote)

        proposals = model_refresh.infer_model_update_proposals([{"content": "user likes vim"}])
        assert remote_called
        assert len(proposals) == 1
        assert proposals[0]["name"] == "shell"

    def test_host_attempted_and_failed_does_not_fall_back_to_remote(self, monkeypatch):
        from mnemosyne.core import local_llm

        remote_called = False

        def fake_host(prompt, max_tokens=2048, temperature=0.1):
            return True, None

        def fake_remote(prompt, temperature=0.1):
            nonlocal remote_called
            remote_called = True
            return '[{"category": "model:user", "name": "fail", "confidence": 0.9, "body": "x"}]'

        monkeypatch.setattr(local_llm, "_try_host_llm", fake_host)
        monkeypatch.setattr(local_llm, "_call_remote_llm", fake_remote)

        proposals = model_refresh.infer_model_update_proposals([{"content": "user likes vim"}])
        assert not remote_called
        assert proposals == []
