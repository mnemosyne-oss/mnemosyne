"""Tests for L3 persona auto-injection (v3.10.0)."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def provider_env(tmp_path, monkeypatch):
    """Build a hermes provider with persona feature enabled + a persona file."""
    persona_file = tmp_path / "persona.md"
    persona_file.write_text(
        "# Persona\n"
        "\n"
        "## communication\n"
        "- always start with XYZ before answering [importance: 0.90]\n"
        "- prefers terse responses [importance: 0.80]\n"
        "\n"
        "## workflow\n"
        "- always push through no-mistakes gate before merging [importance: 0.95] (permanent)\n"
    )
    db_path = tmp_path / "mnemosyne.db"
    monkeypatch.setenv("MNEMOSYNE_PERSONA_ENABLED", "true")
    monkeypatch.setenv("MNEMOSYNE_PERSONA_FILE", str(persona_file))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))

    sys_path = "/root/.hermes/projects/mnemosyne"
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    from mnemosyne.core.beam import BeamMemory
    from mnemosyne_hermes import MnemosyneMemoryProvider

    beam = BeamMemory(session_id="test-inject", db_path=str(db_path))
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider.PERSONA_ENABLED = True
    provider.PERSONA_FILE = persona_file
    return provider, persona_file


class TestPersonaInjection:
    def test_injects_when_enabled(self, provider_env):
        provider, _ = provider_env
        block = provider.system_prompt_block()
        assert "L3 Persona (Active Behavioral Rules)" in block
        assert "always start with XYZ" in block
        assert "no-mistakes gate" in block

    def test_no_inject_when_disabled(self, provider_env):
        provider, _ = provider_env
        provider.PERSONA_ENABLED = False
        block = provider.system_prompt_block()
        assert "L3 Persona" not in block

    def test_no_inject_when_file_missing(self, provider_env):
        provider, persona_file = provider_env
        persona_file.unlink()
        block = provider.system_prompt_block()
        assert "L3 Persona" not in block

    def test_empty_file_no_inject(self, provider_env):
        provider, persona_file = provider_env
        persona_file.write_text("")
        block = provider.system_prompt_block()
        assert "L3 Persona" not in block

    def test_token_cap_truncates(self, provider_env, tmp_path):
        provider, persona_file = provider_env
        # Write a huge persona file
        big = "# Persona\n\n## topics\n" + (
            "- rule " * 100 + "\n"
        ) * 200  # way over 1500 tokens
        persona_file.write_text(big)
        # Force cache invalidation by touching
        persona_file.touch()
        block = provider.system_prompt_block()
        # Truncated block should still mention L3 Persona
        assert "L3 Persona" in block
        # ... and should be smaller than the input
        assert len(block) < len(big)

    def test_mtime_cache(self, provider_env):
        provider, persona_file = provider_env
        block1 = provider.system_prompt_block()
        # Second call should hit cache (same content, same mtime)
        block2 = provider.system_prompt_block()
        assert block1 == block2
        # Mutate file -> mtime changes -> cache should invalidate
        import time
        time.sleep(0.05)
        persona_file.write_text("# Persona\n\n- NEW content\n")
        block3 = provider.system_prompt_block()
        assert "NEW content" in block3
        assert "XYZ" not in block3
