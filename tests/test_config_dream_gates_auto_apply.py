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
