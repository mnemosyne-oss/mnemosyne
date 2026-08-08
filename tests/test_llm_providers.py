"""Tests for named remote-LLM provider presets (mnemosyne.core.llm_providers)."""

import ast
import os
import subprocess
import sys

from mnemosyne.core import llm_providers as lp


class TestPresetRegistry:
    def test_minimax_preset_exists(self):
        preset = lp.get_provider_preset("minimax")
        assert preset is not None
        assert preset["name"] == "MiniMax"
        assert preset["default_model"] == "MiniMax-M3"
        assert preset["default_region"] == "global_en"

    def test_lookup_is_case_insensitive(self):
        assert lp.get_provider_preset("MiniMax") is lp.get_provider_preset("minimax")
        assert lp.get_provider_preset("  MINIMAX  ") is lp.get_provider_preset("minimax")

    def test_unknown_provider_returns_none(self):
        assert lp.get_provider_preset("does-not-exist") is None

    def test_list_providers_includes_minimax(self):
        assert "minimax" in lp.list_providers()


class TestRegionsAndEndpoints:
    def test_global_region_openai_and_anthropic_urls(self):
        assert lp.get_base_url("minimax", "global_en", api="openai") == "https://api.minimax.io/v1"
        assert (
            lp.get_base_url("minimax", "global_en", api="anthropic")
            == "https://api.minimax.io/anthropic"
        )

    def test_cn_region_openai_and_anthropic_urls(self):
        assert lp.get_base_url("minimax", "cn_zh", api="openai") == "https://api.minimaxi.com/v1"
        assert (
            lp.get_base_url("minimax", "cn_zh", api="anthropic")
            == "https://api.minimaxi.com/anthropic"
        )

    def test_empty_region_falls_back_to_default(self):
        assert lp.resolve_region("minimax", "") == "global_en"
        assert lp.get_base_url("minimax", "") == "https://api.minimax.io/v1"

    def test_unknown_region_falls_back_to_default(self):
        assert lp.resolve_region("minimax", "mars") == "global_en"
        assert lp.get_base_url("minimax", "mars") == "https://api.minimax.io/v1"

    def test_unknown_provider_urls_are_none(self):
        assert lp.get_base_url("nope", "global_en") is None
        assert lp.resolve_region("nope") is None


class TestModels:
    def test_model_ids(self):
        assert lp.list_models("minimax") == ["MiniMax-M3", "MiniMax-M2.7"]

    def test_m3_configuration(self):
        cfg = lp.get_model_config("minimax", "MiniMax-M3")
        assert cfg["context_window"] == 1_000_000
        assert cfg["pricing_usd_per_million_tokens"] == {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        }
        assert cfg["input_modalities"] == ["text", "image", "video"]
        assert cfg["thinking"] == ["adaptive", "disabled"]

    def test_m2_7_configuration(self):
        cfg = lp.get_model_config("minimax", "MiniMax-M2.7")
        assert cfg["context_window"] == 204_800
        assert cfg["pricing_usd_per_million_tokens"] == {
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        }
        assert cfg["input_modalities"] == ["text"]
        assert cfg["thinking"] == ["always_on"]

    def test_default_model(self):
        assert lp.default_model("minimax") == "MiniMax-M3"

    def test_unknown_model_config_is_none(self):
        assert lp.get_model_config("minimax", "MiniMax-X9") is None


class TestResolveProviderDefaults:
    def test_defaults_for_known_provider(self):
        base_url, model = lp.resolve_provider_defaults("minimax")
        assert base_url == "https://api.minimax.io/v1"
        assert model == "MiniMax-M3"

    def test_defaults_for_cn_region(self):
        base_url, model = lp.resolve_provider_defaults("minimax", "cn_zh")
        assert base_url == "https://api.minimaxi.com/v1"
        assert model == "MiniMax-M3"

    def test_defaults_for_unknown_provider(self):
        assert lp.resolve_provider_defaults("nope") == (None, None)


class TestLocalLLMWiring:
    """The remote client fills base URL / model from the preset at import time."""

    def _import_with_env(self, env_extra):
        env = os.environ.copy()
        for key in (
            "MNEMOSYNE_LLM_BASE_URL",
            "MNEMOSYNE_LLM_MODEL",
            "MNEMOSYNE_LLM_PROVIDER",
            "MNEMOSYNE_LLM_REGION",
        ):
            env.pop(key, None)
        env.update(env_extra)
        code = (
            "from mnemosyne.core import local_llm as m;"
            "print(repr(m.LLM_BASE_URL));print(repr(m.LLM_REMOTE_MODEL))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, check=True, env=env, text=True,
        )
        lines = out.stdout.splitlines()
        base_url = ast.literal_eval(lines[0])
        model = ast.literal_eval(lines[1])
        return base_url, model

    def test_provider_fills_base_url_and_model(self):
        base_url, model = self._import_with_env({"MNEMOSYNE_LLM_PROVIDER": "minimax"})
        assert base_url == "https://api.minimax.io/v1"
        assert model == "MiniMax-M3"

    def test_provider_region_selects_cn_endpoint(self):
        base_url, model = self._import_with_env(
            {"MNEMOSYNE_LLM_PROVIDER": "minimax", "MNEMOSYNE_LLM_REGION": "cn_zh"}
        )
        assert base_url == "https://api.minimaxi.com/v1"
        assert model == "MiniMax-M3"

    def test_explicit_base_url_and_model_win_over_preset(self):
        base_url, model = self._import_with_env(
            {
                "MNEMOSYNE_LLM_PROVIDER": "minimax",
                "MNEMOSYNE_LLM_BASE_URL": "http://localhost:8080/v1",
                "MNEMOSYNE_LLM_MODEL": "my-model",
            }
        )
        assert base_url == "http://localhost:8080/v1"
        assert model == "my-model"

    def test_no_provider_leaves_base_url_empty(self):
        base_url, model = self._import_with_env({})
        assert base_url == ""
        assert model == ""
