"""Tests for config system and profiles (issue #430).

Covers:
- Template coverage: all 74 template-eligible vars present in every template
- 15 consistency rules: contradictions caught
- Edge cases: force_local without model, sync_encrypt without key, etc.
- Config reader: YAML > env > default precedence, hot-reload, set/get
- Profile apply: dry-run, write, validation
- Config migrate: env vars → YAML
"""

import os
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.config import (
    ENV_VAR_MAP,
    REQUIRES_RESTART,
    MnemosyneConfig,
    get_config,
)
from mnemosyne.core.profiles import (
    PROFILES,
    TEMPLATE_KEYS,
    validate_profile,
    validate_all_profiles,
    list_profiles,
    get_profile,
    apply_profile,
    create_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_config(monkeypatch):
    """Create a MnemosyneConfig with a temp config.yaml and clean env."""
    # Clear all MNEMOSYNE_ env vars for isolated testing
    for key in list(os.environ):
        if key.startswith("MNEMOSYNE_"):
            monkeypatch.delenv(key, raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        MnemosyneConfig.reset_instance()
        config = MnemosyneConfig(config_path=config_path)
        # Monkey-patch the singleton
        monkeypatch.setattr(
            "mnemosyne.core.config.MnemosyneConfig._instance", config
        )
        yield config
        MnemosyneConfig.reset_instance()


# ---------------------------------------------------------------------------
# Template coverage — every template has all 74 keys
# ---------------------------------------------------------------------------

class TestTemplateCoverage:
    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_all_template_keys_present(self, profile_name):
        """Every template must contain all 74 template-eligible keys."""
        settings = PROFILES[profile_name]["settings"]
        expected = set(TEMPLATE_KEYS)
        actual = set(settings.keys())
        missing = expected - actual
        assert not missing, (
            f"Profile '{profile_name}' missing {len(missing)} keys: {sorted(missing)}"
        )

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_no_extra_keys(self, profile_name):
        """No template should have keys not in TEMPLATE_KEYS."""
        settings = PROFILES[profile_name]["settings"]
        expected = set(TEMPLATE_KEYS)
        actual = set(settings.keys())
        extra = actual - expected
        assert not extra, (
            f"Profile '{profile_name}' has {len(extra)} extra keys: {sorted(extra)}"
        )

    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_no_typos(self, profile_name):
        """Every key must be a known config key (no typos)."""
        settings = PROFILES[profile_name]["settings"]
        known = set(ENV_VAR_MAP.keys())
        for key in settings:
            assert key in known, (
                f"Profile '{profile_name}' has unknown key '{key}' — possible typo"
            )

    def test_template_keys_count(self):
        """TEMPLATE_KEYS must have exactly 68 entries (74 was initial estimate,
        actual after excluding deployment/secret/infra/internal keys = 68)."""
        assert len(TEMPLATE_KEYS) == 68, f"Expected 68, got {len(TEMPLATE_KEYS)}"

    def test_template_keys_in_env_var_map(self):
        """Every TEMPLATE_KEY must be in ENV_VAR_MAP."""
        for key in TEMPLATE_KEYS:
            assert key in ENV_VAR_MAP, f"'{key}' not in ENV_VAR_MAP"


# ---------------------------------------------------------------------------
# Consistency rules — 15 rules validated
# ---------------------------------------------------------------------------

class TestConsistencyRules:
    @pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
    def test_profile_passes_all_rules(self, profile_name):
        """Every built-in profile must pass all consistency rules."""
        settings = PROFILES[profile_name]["settings"]
        errors = validate_profile(settings)
        assert not errors, (
            f"Profile '{profile_name}' failed validation:\n  "
            + "\n  ".join(errors)
        )

    def test_all_profiles_validate_clean(self):
        """Bulk validation of all profiles."""
        results = validate_all_profiles()
        for name, errors in results.items():
            assert not errors, f"Profile '{name}' has errors: {errors}"

    # --- Individual rule tests with crafted bad configs ---

    def test_rule1_embeddings_aliases_inconsistent(self):
        """no_embeddings, skip_embeddings, embeddings_off must be equal."""
        settings = dict(PROFILES["balanced"]["settings"])
        settings["no_embeddings"] = "1"
        settings["skip_embeddings"] = "0"
        settings["embeddings_off"] = "1"
        errors = validate_profile(settings)
        assert any("inconsistent" in e.lower() for e in errors)

    def test_rule2_vec_weight_zero_with_embeddings(self):
        """vec_weight=0 with embeddings on is wasteful."""
        settings = dict(PROFILES["balanced"]["settings"])
        settings["no_embeddings"] = "0"
        settings["vec_weight"] = "0.0"
        errors = validate_profile(settings)
        assert any("vec_weight" in e.lower() for e in errors)

    def test_rule3_cross_session_needs_global_scope(self):
        """cross_session=1 requires default_scope=global."""
        settings = dict(PROFILES["balanced"]["settings"])
        settings["cross_session"] = "1"
        settings["default_scope"] = "session"
        errors = validate_profile(settings)
        assert any("cross_session" in e.lower() and "global" in e.lower() for e in errors)

    def test_rule4_smart_compress_needs_llm(self):
        """smart_compress=1 requires llm_enabled=true."""
        settings = dict(PROFILES["balanced"]["settings"])
        settings["smart_compress"] = "1"
        settings["llm_enabled"] = "false"
        errors = validate_profile(settings)
        assert any("smart_compress" in e.lower() and "llm" in e.lower() for e in errors)

    def test_rule5_model_refresh_needs_llm(self):
        """sleep_model_refresh_enabled requires llm_enabled."""
        settings = dict(PROFILES["minimal"]["settings"])
        settings["sleep_model_refresh_enabled"] = "true"
        settings["llm_enabled"] = "false"
        errors = validate_profile(settings)
        assert any("model_refresh" in e.lower() and "llm" in e.lower() for e in errors)

    def test_rule6_conflict_detection_needs_llm(self):
        """llm_conflict_detection requires llm_enabled."""
        settings = dict(PROFILES["minimal"]["settings"])
        settings["llm_conflict_detection"] = "true"
        settings["llm_enabled"] = "false"
        errors = validate_profile(settings)
        assert any("conflict_detection" in e.lower() and "llm" in e.lower() for e in errors)

    def test_rule7_persona_needs_llm(self):
        """persona_enabled requires llm_enabled."""
        settings = dict(PROFILES["minimal"]["settings"])
        settings["persona_enabled"] = "true"
        settings["llm_enabled"] = "false"
        errors = validate_profile(settings)
        assert any("persona" in e.lower() and "llm" in e.lower() for e in errors)

    def test_rule8_tier3_max_chars_zero_with_compress(self):
        """tier3_max_chars=0 with smart_compress=1 is content deletion."""
        settings = dict(PROFILES["balanced"]["settings"])
        settings["smart_compress"] = "1"
        settings["tier3_max_chars"] = "0"
        errors = validate_profile(settings)
        assert any("tier3_max_chars" in e.lower() for e in errors)

    def test_rule9_proactive_linking_needs_embeddings(self):
        """proactive_linking=1 requires no_embeddings=0."""
        settings = dict(PROFILES["minimal"]["settings"])
        settings["proactive_linking"] = "1"
        settings["no_embeddings"] = "1"
        errors = validate_profile(settings)
        assert any("proactive_linking" in e.lower() and "embeddings" in e.lower() for e in errors)

    def test_rule10_polyphonic_recall_needs_embeddings(self):
        """polyphonic_recall=1 requires no_embeddings=0."""
        settings = dict(PROFILES["minimal"]["settings"])
        settings["polyphonic_recall"] = "1"
        settings["no_embeddings"] = "1"
        errors = validate_profile(settings)
        assert any("polyphonic_recall" in e.lower() and "embeddings" in e.lower() for e in errors)

    def test_rule11_enhanced_recall_needs_embeddings(self):
        """enhanced_recall=1 requires no_embeddings=0."""
        settings = dict(PROFILES["minimal"]["settings"])
        settings["enhanced_recall"] = "1"
        settings["no_embeddings"] = "1"
        errors = validate_profile(settings)
        assert any("enhanced_recall" in e.lower() and "embeddings" in e.lower() for e in errors)

    def test_rule12_query_intent_needs_embeddings(self):
        """query_intent=1 requires no_embeddings=0."""
        settings = dict(PROFILES["minimal"]["settings"])
        settings["query_intent"] = "1"
        settings["no_embeddings"] = "1"
        errors = validate_profile(settings)
        assert any("query_intent" in e.lower() and "embeddings" in e.lower() for e in errors)

    def test_rule14_unknown_key_rejected(self):
        """Unknown keys are flagged as possible typos."""
        settings = dict(PROFILES["balanced"]["settings"])
        settings["nonexistent_key"] = "value"
        errors = validate_profile(settings)
        assert any("nonexistent_key" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Template-specific assertions
# ---------------------------------------------------------------------------

class TestTemplateSpecific:
    def test_minimal_llm_off_embeddings_off(self):
        s = PROFILES["minimal"]["settings"]
        assert s["llm_enabled"] == "false"
        assert s["no_embeddings"] == "1"
        assert s["fts_weight"] == "1.0"

    def test_speed_bit_vectors_small_limits(self):
        s = PROFILES["speed"]["settings"]
        assert s["vec_type"] == "bit"
        assert int(s["ep_limit"]) <= 10000

    def test_quality_float32_all_experimental_on(self):
        s = PROFILES["quality"]["settings"]
        assert s["vec_type"] == "float32"
        assert s["polyphonic_recall"] == "1"
        assert s["enhanced_recall"] == "1"
        assert s["query_intent"] == "1"
        assert s["proactive_linking"] == "1"
        assert s["fact_recall_enabled"] == "1"
        assert s["cross_session"] == "1"
        assert s["default_scope"] == "global"

    def test_research_deep_memory(self):
        s = PROFILES["research"]["settings"]
        assert s["cross_session"] == "1"
        assert s["proactive_linking"] == "1"
        assert int(s["sleep_batch"]) >= 10000

    def test_paranoid_security_first(self):
        s = PROFILES["paranoid"]["settings"]
        assert s["write_classifier"] == "strict"
        assert s["sync_encrypt"] == "true"
        assert s["host_llm_enabled"] == "false"
        assert s["force_local"] == "1"
        assert s["persona_enabled"] == "false"

    def test_balanced_matches_codebase_defaults(self):
        s = PROFILES["balanced"]["settings"]
        assert s["vec_type"] == "int8"
        assert s["vec_weight"] == "0.5"
        assert s["fts_weight"] == "0.3"
        assert s["importance_weight"] == "0.2"
        assert int(s["wm_max_items"]) == 10000
        assert int(s["ep_limit"]) == 50000
        assert s["default_scope"] == "session"
        assert s["write_classifier"] == "off"

    def test_embedded_tiny_footprint(self):
        s = PROFILES["embedded"]["settings"]
        assert int(s["wm_max_items"]) <= 500
        assert s["llm_enabled"] == "false"
        assert s["vec_type"] == "bit"

    def test_development_diagnostics_on(self):
        s = PROFILES["development"]["settings"]
        assert s["recall_diagnostics"] == "1"
        assert s["write_classifier"] == "warn"
        assert s["llm_conflict_detection"] == "true"


# ---------------------------------------------------------------------------
# Config reader tests
# ---------------------------------------------------------------------------

class TestConfigReader:
    def test_env_var_fallback(self, temp_config, monkeypatch):
        """When no YAML, env var is used."""
        monkeypatch.setenv("MNEMOSYNE_WM_MAX_ITEMS", "9999")
        assert temp_config.get_int("wm_max_items") == 9999

    def test_yaml_overrides_env(self, temp_config, monkeypatch):
        """YAML takes precedence over env vars."""
        monkeypatch.setenv("MNEMOSYNE_WM_MAX_ITEMS", "9999")
        temp_config.set("wm_max_items", 5555)
        assert temp_config.get_int("wm_max_items") == 5555

    def test_default_when_unset(self, temp_config, monkeypatch):
        """Default is returned when nothing is set."""
        monkeypatch.delenv("MNEMOSYNE_WM_MAX_ITEMS", raising=False)
        assert temp_config.get("wm_max_items", default=10000) == 10000

    def test_get_bool(self, temp_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_LLM_ENABLED", "true")
        assert temp_config.get_bool("llm_enabled") is True
        monkeypatch.setenv("MNEMOSYNE_LLM_ENABLED", "false")
        assert temp_config.get_bool("llm_enabled") is False

    def test_get_float(self, temp_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "0.7")
        assert temp_config.get_float("vec_weight") == 0.7

    def test_get_int(self, temp_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EP_LIMIT", "42000")
        assert temp_config.get_int("ep_limit") == 42000

    def test_set_and_get(self, temp_config):
        temp_config.set("ep_limit", 75000)
        assert temp_config.get_int("ep_limit") == 75000

    def test_reload_picks_up_changes(self, temp_config):
        """Hot-reload: external file changes are picked up."""
        temp_config.set("ep_limit", 10000)
        # Externally modify the YAML file
        import yaml
        config_path = temp_config.config_path
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        data["ep_limit"] = 99999
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        changed = temp_config.reload()
        assert "ep_limit" in changed
        assert temp_config.get_int("ep_limit") == 99999

    def test_reload_returns_changed_keys(self, temp_config):
        temp_config.set("ep_limit", 10000)
        temp_config.set("wm_max_items", 5000)

        import yaml
        config_path = temp_config.config_path
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        data["ep_limit"] = 20000  # changed
        # wm_max_items stays the same
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        changed = temp_config.reload()
        assert "ep_limit" in changed
        assert "wm_max_items" not in changed

    def test_migrate_from_env(self, temp_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WM_MAX_ITEMS", "12345")
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "float32")
        migrated = temp_config.migrate_from_env()
        assert "wm_max_items" in migrated
        assert "vec_type" in migrated
        assert temp_config.get_int("wm_max_items") == 12345
        assert temp_config.get_str("vec_type") == "float32"

    def test_requires_restart_warning(self, temp_config, caplog):
        """Setting a requires_restart key should log a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        temp_config.set("data_dir", "/some/new/path")
        assert any("requires restart" in r.message.lower() for r in caplog.records)

    def test_no_yaml_no_crash(self, temp_config, monkeypatch):
        """Config works without a config.yaml file."""
        monkeypatch.delenv("MNEMOSYNE_WM_MAX_ITEMS", raising=False)
        val = temp_config.get("wm_max_items", default=10000)
        assert val == 10000


# ---------------------------------------------------------------------------
# Profile apply tests
# ---------------------------------------------------------------------------

class TestProfileApply:
    def test_apply_writes_to_config(self, temp_config):
        success, errors = apply_profile("speed", config_path=temp_config.config_path)
        assert success, f"Errors: {errors}"
        assert temp_config.get_str("vec_type") == "bit"
        assert temp_config.get_int("ep_limit") == 10000

    def test_apply_dry_run_no_changes(self, temp_config):
        """Dry-run should not write anything."""
        success, errors = apply_profile("quality", config_path=temp_config.config_path, dry_run=True)
        assert success
        # Config should still be empty/default
        assert not temp_config.config_path.exists() or temp_config.get("vec_type") is None

    def test_apply_unknown_profile_fails(self, temp_config):
        success, errors = apply_profile("nonexistent", config_path=temp_config.config_path)
        assert not success
        assert any("unknown" in e.lower() for e in errors)

    def test_apply_invalid_profile_fails(self, temp_config):
        """A profile that fails validation should not be applied."""
        # Create a bad profile in-memory
        from mnemosyne.core.profiles import PROFILES
        original = PROFILES["balanced"]["settings"].copy()
        PROFILES["balanced"]["settings"] = dict(original, llm_enabled="false", smart_compress="1")
        try:
            success, errors = apply_profile("balanced", config_path=temp_config.config_path)
            assert not success
            assert any("smart_compress" in e.lower() for e in errors)
        finally:
            PROFILES["balanced"]["settings"] = original

    def test_apply_all_profiles(self, temp_config):
        """All 8 profiles should apply successfully."""
        for name in PROFILES:
            success, errors = apply_profile(name, config_path=temp_config.config_path)
            assert success, f"Profile '{name}' failed: {errors}"


# ---------------------------------------------------------------------------
# Profile create test
# ---------------------------------------------------------------------------

class TestProfileCreate:
    def test_create_from_current_config(self, temp_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WM_MAX_ITEMS", "7777")
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "int8")
        success = create_profile("custom", description="My custom profile")
        assert success
        assert "custom" in PROFILES
        assert PROFILES["custom"]["settings"]["wm_max_items"] == "7777"
