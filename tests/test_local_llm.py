import builtins
import importlib
import logging
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mnemosyne.core import local_llm
from mnemosyne.core.llm_backends import (
    CallableLLMBackend,
    set_host_llm_backend,
)

REAL_LOAD_LLM = local_llm._load_llm


class TestLocalModelDownloadNotice:
    def test_uncached_default_model_logs_notice_before_download(self, monkeypatch, tmp_path, caplog, capsys):
        """The default GGUF warning is logged before its network fetch, not printed."""
        cache_dir = tmp_path / "models"
        events = []
        download_calls = []

        def fake_download(**kwargs):
            events.append("download")
            download_calls.append(kwargs)
            return str(cache_dir / kwargs["filename"])

        monkeypatch.setattr(local_llm, "MODEL_CACHE_DIR", cache_dir)
        monkeypatch.setattr(local_llm, "DEFAULT_MODEL_REPO", "openbmb/MiniCPM5-1B-GGUF")
        monkeypatch.setattr(local_llm, "DEFAULT_MODEL_FILE", "MiniCPM5-1B-Q4_K_M.gguf")
        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
        caplog.set_level(logging.WARNING, logger=local_llm.__name__)
        original_warning = local_llm.logger.warning

        def record_warning(*args, **kwargs):
            events.append("warning")
            return original_warning(*args, **kwargs)

        monkeypatch.setattr(local_llm.logger, "warning", record_warning)
        local_llm._download_model()

        assert events == ["warning", "download"]
        assert download_calls == [{
            "repo_id": "openbmb/MiniCPM5-1B-GGUF",
            "filename": "MiniCPM5-1B-Q4_K_M.gguf",
            "local_dir": str(cache_dir),
            "local_dir_use_symlinks": False,
        }]
        assert capsys.readouterr().out == ""
        message = caplog.records[-1].getMessage()
        assert "MiniCPM5-1B-Q4_K_M.gguf" in message
        assert "openbmb/MiniCPM5-1B-GGUF" in message
        assert str(cache_dir) in message
        assert "approximately 656 MB" in message
        assert "current operation will block until the download completes" in message
        assert "MNEMOSYNE_LLM_ENABLED=false" in message
        assert "pre-cache" in message

    def test_cached_model_skips_notice_and_download(self, monkeypatch, tmp_path, caplog):
        """An already cached GGUF does not produce download-related output."""
        cache_dir = tmp_path / "models"
        cache_dir.mkdir()
        model_file = cache_dir / "cached.gguf"
        model_file.touch()
        monkeypatch.setattr(local_llm, "MODEL_CACHE_DIR", cache_dir)
        monkeypatch.setattr(local_llm, "DEFAULT_MODEL_FILE", model_file.name)
        caplog.set_level(logging.WARNING, logger=local_llm.__name__)

        with patch.dict(sys.modules, {"huggingface_hub": None}):
            assert local_llm._download_model() == model_file

        assert not caplog.records

    def test_overridden_model_omits_default_size_and_failed_download_falls_back(self, monkeypatch, tmp_path, caplog):
        """Custom artifacts avoid a misleading size claim and retain AAAK fallback."""
        cache_dir = tmp_path / "models"

        def failed_download(**kwargs):
            raise OSError("offline")

        monkeypatch.setattr(local_llm, "MODEL_CACHE_DIR", cache_dir)
        monkeypatch.setattr(local_llm, "DEFAULT_MODEL_REPO", "example/custom-gguf")
        monkeypatch.setattr(local_llm, "DEFAULT_MODEL_FILE", "custom.gguf")
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "_llm_instance", None)
        monkeypatch.setattr(local_llm, "_llm_available", None)
        monkeypatch.setattr(local_llm, "_load_llm", REAL_LOAD_LLM)
        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=failed_download))
        caplog.set_level(logging.WARNING, logger=local_llm.__name__)

        assert local_llm._load_llm() is None
        assert local_llm._llm_available is False
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "custom.gguf" in message
        assert "example/custom-gguf" in message
        assert "656 MB" not in message


class TestRemoteLLM:
    def test_llm_available_returns_true_when_base_url_set(self, monkeypatch):
        """BUG-2: llm_available() must report True when remote is configured."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://localhost:8080/v1")
        # Reset module-level cache
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setattr(local_llm, "_llm_available", None)
        monkeypatch.setattr(local_llm, "_llm_instance", None)

        assert local_llm.llm_available() is True

    def test_call_remote_llm_with_mock_response(self, monkeypatch):
        """BUG-2: _call_remote_llm parses OpenAI-compatible response correctly."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setenv("MNEMOSYNE_LLM_API_KEY", "sk-test")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setattr(local_llm, "LLM_API_KEY", "sk-test")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "test-model")
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 128)

        mock_response = {
            "choices": [
                {"message": {"content": "This is a test summary."}}
            ]
        }

        # Mock httpx by patching the import inside _call_remote_llm
        mock_client = MagicMock()
        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 200
        mock_response_obj.raise_for_status = lambda: None
        mock_response_obj.json.return_value = mock_response
        mock_client.post.return_value = mock_response_obj
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = lambda *args: None

        mock_httpx_module = MagicMock()
        mock_httpx_module.Client = MagicMock(return_value=mock_client)

        # Save original import to avoid recursion
        _orig_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                return mock_httpx_module
            return _orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", mock_import):
            result = local_llm._call_remote_llm("Test prompt")
            assert result == "This is a test summary."

            # Verify the call was made with correct payload
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://test-server/v1/chat/completions"
            payload = call_args[1]["json"]
            assert payload["model"] == "test-model"
            assert payload["messages"][0]["content"] == "Test prompt"
            assert "Authorization" in call_args[1]["headers"]

    def test_call_remote_llm_urllib_fallback(self, monkeypatch):
        """BUG-2: Falls back to urllib when httpx unavailable."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://test-server/v1")
        monkeypatch.setattr(local_llm, "LLM_API_KEY", "")
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 128)

        mock_response = {
            "choices": [
                {"message": {"content": "Fallback summary."}}
            ]
        }

        import json
        mock_data = json.dumps(mock_response).encode()

        class MockResponse:
            def read(self):
                return mock_data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        # Patch httpx import in local_llm module to simulate it not being installed
        with patch.dict("sys.modules", {"httpx": None}):
            with patch("urllib.request.urlopen", return_value=MockResponse()):
                result = local_llm._call_remote_llm("Test prompt")
                assert result == "Fallback summary."

    def test_summarize_memories_prefers_remote_over_local(self, monkeypatch):
        """BUG-2: summarize_memories() calls remote when BASE_URL is set."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "_llm_available", False)
        monkeypatch.setattr(local_llm, "_llm_instance", None)

        with patch.object(local_llm, "_call_remote_llm", return_value="Remote summary.") as mock_remote:
            result = local_llm.summarize_memories(["Memory one", "Memory two"])
            assert result == "Remote summary."
            mock_remote.assert_called_once()

    def test_summarize_memories_falls_back_local_when_remote_fails(self, monkeypatch):
        """BUG-2: When remote fails and local is unavailable, return None (aaak fallback)."""
        monkeypatch.setenv("MNEMOSYNE_LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")

        # Remote returns None (failure), local _load_llm returns None (unavailable)
        with patch.object(local_llm, "_call_remote_llm", return_value=None) as mock_remote:
            with patch.object(local_llm, "_load_llm", return_value=None) as mock_load:
                result = local_llm.summarize_memories(["Memory one"])
                # Should return None since both remote and local fail
                assert result is None
                mock_remote.assert_called_once()
                mock_load.assert_called_once()


class TestSleepPromptOverride:
    def test_build_prompt_uses_sleep_prompt_override(self, monkeypatch):
        """MNEMOSYNE_SLEEP_PROMPT can steer local consolidation language."""
        monkeypatch.setattr(
            local_llm,
            "SLEEP_PROMPT",
            "Fasse diese Erinnerungen auf Deutsch zusammen.\nQuelle: {source}\n{memories}\nAntwort:",
            raising=False,
        )

        prompt = local_llm._build_prompt(
            ["Ich mag Kaffee", "Berlin bleibt wichtig"],
            source="conversation",
        )

        assert "Fasse diese Erinnerungen auf Deutsch zusammen." in prompt
        assert "Quelle: conversation" in prompt
        assert "- Ich mag Kaffee" in prompt
        assert "- Berlin bleibt wichtig" in prompt
        assert "Summarize the following memories" not in prompt

    def test_build_host_prompt_uses_same_sleep_prompt_override(self, monkeypatch):
        """Host LLM consolidation gets the same language-controlled prompt."""
        monkeypatch.setattr(
            local_llm,
            "SLEEP_PROMPT",
            "Write in German. Source={source}. Memories:\n{memories}",
            raising=False,
        )

        prompt = local_llm._build_host_prompt(["User prefers tea"], source="profile")

        assert prompt == "Write in German. Source=profile. Memories:\n- User prefers tea"
        assert "<|user|>" not in prompt
        assert "</s>" not in prompt


class TestHostLLMBackend:
    """Tests for the host LLM adapter integration in summarize_memories()."""

    def test_host_llm_timeout_can_be_configured_from_env(self):
        """MNEMOSYNE_HOST_LLM_TIMEOUT overrides the host adapter timeout."""
        env = os.environ.copy()
        env["MNEMOSYNE_HOST_LLM_TIMEOUT"] = "120"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from mnemosyne.core import local_llm; print(local_llm.HOST_LLM_TIMEOUT)",
            ],
            capture_output=True,
            check=True,
            env=env,
            text=True,
        )

        assert result.stdout.strip() == "120.0"

    def _enable_host(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_PROVIDER", None)
        monkeypatch.setattr(local_llm, "HOST_LLM_MODEL", None)

    def test_summarize_memories_uses_host_when_enabled(self, monkeypatch):
        """Host backend is consulted before remote when enabled."""
        self._enable_host(monkeypatch)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 128)
        monkeypatch.setattr(local_llm, "HOST_LLM_PROVIDER", "openai-codex")
        monkeypatch.setattr(local_llm, "HOST_LLM_MODEL", "gpt-5.1-mini")

        captured = []

        def fake(prompt, *, max_tokens, temperature, timeout, provider=None, model=None):
            captured.append({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
                "provider": provider,
                "model": model,
            })
            return "Host summary."

        set_host_llm_backend(CallableLLMBackend("test", fake))
        with patch.object(local_llm, "_call_remote_llm") as mock_remote, \
             patch.object(local_llm, "_call_local_llm") as mock_local:
            assert local_llm.summarize_memories(["Memory one"]) == "Host summary."
            mock_remote.assert_not_called()
            mock_local.assert_not_called()
        assert captured
        assert captured[0]["max_tokens"] == 128
        assert captured[0]["temperature"] == 0.3
        assert captured[0]["timeout"] == local_llm.HOST_LLM_TIMEOUT
        assert captured[0]["provider"] == "openai-codex"
        assert captured[0]["model"] == "gpt-5.1-mini"
        # Host prompt MUST NOT contain TinyLlama chat-template tokens.
        assert "<|user|>" not in captured[0]["prompt"]
        assert "</s>" not in captured[0]["prompt"]
        assert "<|assistant|>" not in captured[0]["prompt"]

    def test_summarize_memories_skips_remote_on_host_miss(self, monkeypatch):
        """A3 contract: host enabled + host returns None → fall to local, NOT to remote."""
        self._enable_host(monkeypatch)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: None))
        with patch.object(local_llm, "_call_remote_llm", return_value="Remote summary.") as mock_remote, \
             patch.object(local_llm, "_call_local_llm", return_value="Local summary.") as mock_local:
            assert local_llm.summarize_memories(["Memory one"]) == "Local summary."
            mock_remote.assert_not_called()
            mock_local.assert_called_once()

    def test_summarize_memories_returns_none_when_host_and_local_both_fail(self, monkeypatch):
        """Host attempted + nothing + local fails → None (NOT remote)."""
        self._enable_host(monkeypatch)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: None))
        with patch.object(local_llm, "_call_remote_llm", return_value="Remote summary.") as mock_remote, \
             patch.object(local_llm, "_call_local_llm", return_value=None) as mock_local:
            assert local_llm.summarize_memories(["Memory one"]) is None
            mock_remote.assert_not_called()
            mock_local.assert_called_once()

    def test_summarize_memories_unchanged_when_HOST_LLM_ENABLED_false(self, monkeypatch):
        """REGRESSION: existing remote/local behavior is preserved when host is off."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", False)  # explicitly off
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        # Even with a backend registered, host is gated off.
        set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: "Host summary."))
        with patch.object(local_llm, "_call_remote_llm", return_value="Remote summary.") as mock_remote:
            assert local_llm.summarize_memories(["Memory one"]) == "Remote summary."
            mock_remote.assert_called_once()

    def test_summarize_memories_unchanged_when_LLM_ENABLED_false(self, monkeypatch):
        """A2 contract: MNEMOSYNE_LLM_ENABLED=false disables host and remote alike."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", False)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: "Host summary."))
        with patch.object(local_llm, "_call_remote_llm", return_value="Remote summary.") as mock_remote, \
             patch.object(local_llm, "_call_local_llm", return_value=None):
            # Host gated by LLM_ENABLED → not attempted; remote also gated → not called;
            # local: _call_local_llm internally checks via _load_llm() which itself
            # gates on LLM_ENABLED (preserving prior behavior). End result: None.
            assert local_llm.summarize_memories(["Memory one"]) is None
            mock_remote.assert_not_called()

    def test_summarize_memories_swallows_host_exception(self, monkeypatch):
        """Backend that raises is treated as host-attempted-with-no-output (A3 still applies)."""
        self._enable_host(monkeypatch)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")

        def boom(*a, **k):
            raise RuntimeError("provider exploded")

        set_host_llm_backend(CallableLLMBackend("test", boom))
        with patch.object(local_llm, "_call_remote_llm", return_value="Remote summary.") as mock_remote, \
             patch.object(local_llm, "_call_local_llm", return_value="Local summary.") as mock_local:
            assert local_llm.summarize_memories(["Memory one"]) == "Local summary."
            mock_remote.assert_not_called()
            mock_local.assert_called_once()


class TestLLMAvailable:
    """Tests for the host-aware llm_available() gate."""

    def test_llm_available_true_when_only_host_backend_registered(self, monkeypatch):
        """A5 contract: Hermes-only users (no remote URL, no GGUF) still report available."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "")
        monkeypatch.setattr(local_llm, "_llm_available", False)
        set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: "x"))
        assert local_llm.llm_available() is True

    def test_llm_available_false_when_host_enabled_but_no_backend(self, monkeypatch):
        """HOST_LLM_ENABLED=true with no backend registered must not fake availability."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "")
        monkeypatch.setattr(local_llm, "_llm_available", False)
        # No backend registered.
        assert local_llm.llm_available() is False

    def test_llm_available_false_when_LLM_ENABLED_false(self, monkeypatch):
        """A2 contract: MNEMOSYNE_LLM_ENABLED=false makes everything unavailable."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", False)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "_llm_available", False)
        set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: "x"))
        assert local_llm.llm_available() is False


class TestHostAwareChunking:
    """Tests for HOST_LLM_N_CTX-aware budgeting (decision C6)."""

    def test_prompt_token_budget_uses_host_n_ctx_when_host_will_handle(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "LLM_N_CTX", 2048)
        monkeypatch.setattr(local_llm, "HOST_LLM_N_CTX", 32000)
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 256)
        set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: "x"))

        host_budget = local_llm._prompt_token_budget()
        # Should be much larger than the TinyLlama-calibrated default budget.
        assert host_budget > 10_000

        # Same module without a host backend → falls back to LLM_N_CTX budget.
        set_host_llm_backend(None)
        local_budget = local_llm._prompt_token_budget()
        assert local_budget < host_budget


class TestThinkTagStripping:
    """Verify think tag removal from LLM output (closed tags only).

    Unclosed think tags are not stripped because there is no way to
    distinguish thinking content from the actual response when the
    closing tag is missing.
    """

    def test_clean_output_strips_closed_think_tags(self):
        raw = "<think>let me reason</think> The answer is 42."
        assert local_llm._clean_output(raw) == "The answer is 42."

    def test_clean_output_strips_multiline_closed_think_tags(self):
        raw = "<think>step 1\nstep 2</think>\nFinal answer."
        assert local_llm._clean_output(raw) == "Final answer."

    def test_clean_output_strips_multiple_think_blocks(self):
        raw = "<think>first</think>middle<think>second</think>end"
        assert local_llm._clean_output(raw) == "middleend"

    def test_clean_output_preserves_text_without_think_tags(self):
        raw = "Just a normal summary with no thinking."
        assert local_llm._clean_output(raw) == "Just a normal summary with no thinking."

    def test_clean_output_empty_after_stripping(self):
        raw = "<think>only thinking, no output</think>"
        assert local_llm._clean_output(raw) == ""

    def test_clean_output_rejects_nested_think_tags(self):
        assert local_llm._is_invalid_reasoning_output(
            local_llm._clean_output("<think>outer<think>inner</think>outer</think>")
        )

    def test_clean_output_rejects_unclosed_think_tag(self):
        raw = "middle<think>reasoning truncated at the token limit"
        assert local_llm._is_invalid_reasoning_output(local_llm._clean_output(raw))

    def test_clean_output_rejects_unmatched_closing_think_tag(self):
        assert local_llm._is_invalid_reasoning_output(
            local_llm._clean_output("safe prefix</think> malformed ordering")
        )

    def test_try_host_llm_strips_think_tags(self, monkeypatch):
        """Host LLM output with closed think tags should be cleaned."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_TIMEOUT", 5.0)
        monkeypatch.setattr(local_llm, "HOST_LLM_PROVIDER", None)
        monkeypatch.setattr(local_llm, "HOST_LLM_MODEL", None)
        set_host_llm_backend(CallableLLMBackend("test", lambda prompt, **kw: "<think>reasoning</think>Summary of memories."))

        attempted, text = local_llm._try_host_llm("test prompt", max_tokens=128, temperature=0.3)
        assert attempted is True
        assert text == "Summary of memories."

    def test_try_host_llm_rejects_unclosed_think_tag(self, monkeypatch):
        """Token-truncated host reasoning must not reach a persistence caller."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_TIMEOUT", 5.0)
        monkeypatch.setattr(local_llm, "HOST_LLM_PROVIDER", None)
        monkeypatch.setattr(local_llm, "HOST_LLM_MODEL", None)
        set_host_llm_backend(CallableLLMBackend("test", lambda prompt, **kw: "<think>reasoning\nActual output"))

        attempted, text = local_llm._try_host_llm("test prompt", max_tokens=128, temperature=0.3)
        assert attempted is True
        assert local_llm._is_invalid_reasoning_output(text)


class TestRemoteLLMFallback:
    """Tests for the LLM_FALLBACK_MODELS chain in _call_remote_llm()."""

    def _ok(self, text="ok"):
        return (text, 200, None)

    def _err(self, status):
        return (None, status, RuntimeError(f"http {status}"))

    def _connerr(self):
        return (None, None, ConnectionError("boom"))

    def test_is_retryable_status(self):
        assert local_llm._is_retryable_status(404) is True
        assert local_llm._is_retryable_status(400) is True
        assert local_llm._is_retryable_status(500) is True
        assert local_llm._is_retryable_status(502) is True
        assert local_llm._is_retryable_status(503) is True
        assert local_llm._is_retryable_status(401) is False
        assert local_llm._is_retryable_status(403) is False
        assert local_llm._is_retryable_status(429) is False
        assert local_llm._is_retryable_status(200) is False

    def test_primary_success_skips_fallback(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1", "fb2"])

        with patch.object(
            local_llm, "_call_remote_llm_with_model", return_value=self._ok("primary-out")
        ) as m:
            assert local_llm._call_remote_llm("p") == "primary-out"
            assert m.call_count == 1
            assert m.call_args.args[1] == "primary"

    def test_404_triggers_fallback(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1"])

        with patch.object(
            local_llm,
            "_call_remote_llm_with_model",
            side_effect=[self._err(404), self._ok("fb-out")],
        ) as m:
            assert local_llm._call_remote_llm("p") == "fb-out"
            assert m.call_count == 2
            assert m.call_args_list[0].args[1] == "primary"
            assert m.call_args_list[1].args[1] == "fb1"

    def test_5xx_triggers_fallback(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1"])

        with patch.object(
            local_llm,
            "_call_remote_llm_with_model",
            side_effect=[self._err(502), self._ok("fb-out")],
        ):
            assert local_llm._call_remote_llm("p") == "fb-out"

    def test_401_does_not_trigger_fallback(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1", "fb2"])

        with patch.object(
            local_llm, "_call_remote_llm_with_model", return_value=self._err(401)
        ) as m:
            assert local_llm._call_remote_llm("p") is None
            assert m.call_count == 1

    def test_429_does_not_trigger_fallback(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1"])

        with patch.object(
            local_llm, "_call_remote_llm_with_model", return_value=self._err(429)
        ) as m:
            assert local_llm._call_remote_llm("p") is None
            assert m.call_count == 1

    def test_connection_error_triggers_fallback(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1"])

        with patch.object(
            local_llm,
            "_call_remote_llm_with_model",
            side_effect=[self._connerr(), self._ok("fb-out")],
        ):
            assert local_llm._call_remote_llm("p") == "fb-out"

    def test_iterates_all_fallbacks_until_success(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1", "fb2", "fb3"])

        with patch.object(
            local_llm,
            "_call_remote_llm_with_model",
            side_effect=[
                self._err(404),
                self._err(503),
                self._ok("fb2-out"),
            ],
        ) as m:
            assert local_llm._call_remote_llm("p") == "fb2-out"
            assert m.call_count == 3

    def test_returns_none_when_all_fail(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1", "fb2"])

        with patch.object(
            local_llm,
            "_call_remote_llm_with_model",
            side_effect=[
                self._err(404),
                self._err(500),
                self._connerr(),
            ],
        ):
            assert local_llm._call_remote_llm("p") is None

    def test_empty_fallback_list_only_tries_primary(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", [])

        with patch.object(
            local_llm, "_call_remote_llm_with_model", return_value=self._ok("p")
        ) as m:
            assert local_llm._call_remote_llm("p") == "p"
            assert m.call_count == 1

    def test_primary_deduped_from_fallback_list(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://x/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["primary", "fb1"])

        with patch.object(
            local_llm, "_call_remote_llm_with_model", return_value=self._err(404)
        ) as m:
            assert local_llm._call_remote_llm("p") is None
            assert m.call_count == 2
            models = [call.args[1] for call in m.call_args_list]
            assert models == ["primary", "fb1"]

    def test_fallback_uses_overridden_base_url(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://primary/v1")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1"])
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_BASE_URL", "http://fb-host/v1")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_API_KEY", "fb-key")

        with patch.object(
            local_llm,
            "_call_remote_llm_with_model",
            side_effect=[self._err(404), self._ok("fb-out")],
        ) as m:
            assert local_llm._call_remote_llm("p") == "fb-out"
            primary_call = m.call_args_list[0]
            assert primary_call.kwargs["base_url"] == "http://primary/v1"
            fb_call = m.call_args_list[1]
            assert fb_call.kwargs["base_url"] == "http://fb-host/v1"
            assert fb_call.kwargs["api_key"] == "fb-key"

    def test_fallback_inherits_primary_url_when_no_override(self, monkeypatch):
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://primary/v1")
        monkeypatch.setattr(local_llm, "LLM_API_KEY", "primary-key")
        monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "primary")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["fb1"])
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_BASE_URL", "")
        monkeypatch.setattr(local_llm, "LLM_FALLBACK_API_KEY", "")

        with patch.object(
            local_llm,
            "_call_remote_llm_with_model",
            side_effect=[self._err(404), self._ok("fb-out")],
        ) as m:
            assert local_llm._call_remote_llm("p") == "fb-out"
            fb_call = m.call_args_list[1]
            assert fb_call.kwargs["base_url"] == "http://primary/v1"
            assert fb_call.kwargs["api_key"] == "primary-key"



@pytest.fixture
def restore_modes():
    """Undo restrictive chmods so pytest can clean up tmp_path.

    A 0o500 or 0o200 directory cannot be removed by the tmp_path teardown, which
    leaves garbage behind and emits a cleanup warning on every run.
    """
    touched = []
    yield touched
    for path in touched:
        try:
            path.chmod(0o700)
        except OSError:  # pragma: no cover - already gone
            pass


@contextmanager
def _model_cache_env(monkeypatch, value):
    """Reload local_llm under a patched MNEMOSYNE_MODEL_CACHE_DIR, then restore it.

    The first reload has to happen while the variable is set; the restoring
    reload has to happen after monkeypatch has unset it. Reloading inside a
    `finally` instead would run before fixture teardown and leave the module
    describing an environment that no longer exists.
    """
    try:
        with monkeypatch.context() as patched:
            if value is None:
                patched.delenv("MNEMOSYNE_MODEL_CACHE_DIR", raising=False)
            else:
                patched.setenv("MNEMOSYNE_MODEL_CACHE_DIR", value)
            yield importlib.reload(local_llm)
    finally:
        # Ordering matters: the inner context exits first, restoring the
        # environment, and only then is the module reloaded to match it. The
        # `finally` covers a raising body, which would otherwise skip the
        # restoring reload and leak the overridden path into later tests.
        importlib.reload(local_llm)


class TestModelCacheDirOverride:
    """MNEMOSYNE_MODEL_CACHE_DIR relocates the GGUF cache (#708).

    The override is authoritative when set: an unusable directory fails rather
    than falling back to `~/.hermes/mnemosyne/models`, since silently
    reinstating the location the user moved away from is the substitution the
    setting exists to prevent.
    """

    def _select(self, monkeypatch, path, *, from_env=True):
        monkeypatch.setattr(local_llm, "MODEL_CACHE_DIR", Path(path))
        monkeypatch.setattr(local_llm, "MODEL_CACHE_DIR_FROM_ENV", from_env, raising=False)

    _DEFAULT = Path.home() / ".hermes" / "mnemosyne" / "models"

    def test_unset_keeps_the_historical_location(self, monkeypatch):
        with _model_cache_env(monkeypatch, None) as module:
            assert module.MODEL_CACHE_DIR == self._DEFAULT
            assert module.MODEL_CACHE_DIR_FROM_ENV is False

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_value_keeps_the_historical_location(self, monkeypatch, blank):
        """Per spec, unset and empty behave identically."""
        with _model_cache_env(monkeypatch, blank) as module:
            assert module.MODEL_CACHE_DIR == self._DEFAULT
            assert module.MODEL_CACHE_DIR_FROM_ENV is False

    def test_set_value_relocates_and_expands_tilde(self, monkeypatch):
        with _model_cache_env(monkeypatch, "~/relocated-models") as module:
            assert module.MODEL_CACHE_DIR == Path.home() / "relocated-models"
            assert module.MODEL_CACHE_DIR_FROM_ENV is True

    def test_trailing_whitespace_in_the_path_is_preserved(self, monkeypatch):
        """A directory name may end in a space; stripping would select another."""
        with _model_cache_env(monkeypatch, "/tmp/models ") as module:
            assert str(module.MODEL_CACHE_DIR) == "/tmp/models "

    def test_module_state_is_restored_after_the_override_is_unset(self, monkeypatch):
        """The reload dance must not leak a stale path into later tests.

        Compared against the state captured before the context rather than the
        built-in default, so the assertion holds when the developer running the
        suite has MNEMOSYNE_MODEL_CACHE_DIR set in their own environment.
        """
        before = (local_llm.MODEL_CACHE_DIR, local_llm.MODEL_CACHE_DIR_FROM_ENV)

        with _model_cache_env(monkeypatch, "/tmp/leak-check") as module:
            assert module.MODEL_CACHE_DIR == Path("/tmp/leak-check")

        assert (local_llm.MODEL_CACHE_DIR, local_llm.MODEL_CACHE_DIR_FROM_ENV) == before

    def test_module_state_is_restored_when_the_body_raises(self, monkeypatch):
        """A failing assertion must not leak the overridden path either."""
        before = (local_llm.MODEL_CACHE_DIR, local_llm.MODEL_CACHE_DIR_FROM_ENV)

        with pytest.raises(RuntimeError, match="boom"):
            with _model_cache_env(monkeypatch, "/tmp/raises") as module:
                assert module.MODEL_CACHE_DIR == Path("/tmp/raises")
                raise RuntimeError("boom")

        assert (local_llm.MODEL_CACHE_DIR, local_llm.MODEL_CACHE_DIR_FROM_ENV) == before

    def test_resolution_is_import_time_not_call_time(self, monkeypatch, tmp_path):
        """Changing the variable after import must not move the cache.

        Without this, a call-time implementation that re-read os.environ on every
        access would satisfy every other test here, and the spec is explicit that
        resolution happens at import.
        """
        chosen = tmp_path / "at-import"

        with _model_cache_env(monkeypatch, str(chosen)) as module:
            assert module.MODEL_CACHE_DIR == chosen

            # Nested so the change is undone before the restoring reload runs.
            with monkeypatch.context() as later:
                later.setenv("MNEMOSYNE_MODEL_CACHE_DIR", str(tmp_path / "too-late"))
                assert module.MODEL_CACHE_DIR == chosen
                assert module._model_path() is None  # looks under chosen, not too-late
                chosen.mkdir(parents=True)
                (chosen / module.DEFAULT_MODEL_FILE).write_bytes(b"gguf")
                assert module._model_path() == chosen / module.DEFAULT_MODEL_FILE

    def test_lookup_and_download_both_use_the_override(
        self, monkeypatch, tmp_path, caplog
    ):
        """One directory for the cached-file check, the mkdir and hf_hub_download."""
        relocated = tmp_path / "elsewhere"
        self._select(monkeypatch, relocated)
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.write_bytes(b"gguf")
            return str(target)

        monkeypatch.setitem(
            sys.modules, "huggingface_hub", MagicMock(hf_hub_download=fake_download)
        )

        with caplog.at_level(logging.WARNING, logger=local_llm.__name__):
            got = local_llm._download_model()

        assert calls[0]["local_dir"] == str(relocated)
        assert got.parent == relocated
        assert relocated.is_dir()
        # The pre-download notice must name where the model is actually going.
        assert str(relocated) in caplog.records[0].getMessage()

        # The lookup agrees, and a second call short-circuits on the cached file
        # rather than downloading again. Calling _download_model() rather than
        # _model_path() is the point: it exercises the existing-file branch.
        assert local_llm._model_path() == relocated / local_llm.DEFAULT_MODEL_FILE
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=local_llm.__name__):
            again = local_llm._download_model()
        assert again == got
        assert len(calls) == 1
        assert caplog.records == []


    def test_uncreatable_override_fails_naming_the_variable_and_path(
        self, monkeypatch, tmp_path, caplog
    ):
        """A file where the directory should be: cannot be created."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("i am a file", encoding="utf-8")
        self._select(monkeypatch, blocker / "models")

        with caplog.at_level("ERROR", logger=local_llm.__name__):
            with pytest.raises(RuntimeError) as excinfo:
                local_llm._ensure_model_cache_dir()

        for expected in ("MNEMOSYNE_MODEL_CACHE_DIR", str(blocker / "models")):
            assert expected in str(excinfo.value)
        # Logged as well as raised: _load_llm swallows the exception, so the
        # raise alone would never reach the user. Both facts must survive the
        # trip through the logger, not just the variable name.
        logged = caplog.records[0].getMessage()
        assert "MNEMOSYNE_MODEL_CACHE_DIR" in logged
        assert str(blocker / "models") in logged

    def test_unwritable_override_fails_naming_the_variable_and_path(
        self, monkeypatch, tmp_path, restore_modes, caplog
    ):
        """Creatable is not the same as usable."""
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        restore_modes.append(readonly)
        self._select(monkeypatch, readonly)

        if os.access(readonly, os.W_OK):  # pragma: no cover - root ignores the bit
            pytest.skip("running as root; the unwritable case cannot be modelled")

        with caplog.at_level(logging.ERROR, logger=local_llm.__name__):
            with pytest.raises(RuntimeError) as excinfo:
                local_llm._ensure_model_cache_dir()

        for expected in ("MNEMOSYNE_MODEL_CACHE_DIR", str(readonly)):
            assert expected in str(excinfo.value)
            # Logged too: _load_llm swallows the exception, so the log is the
            # only channel that actually reaches the user.
            assert expected in caplog.records[0].getMessage()

    def test_writable_but_unsearchable_override_is_rejected(
        self, monkeypatch, tmp_path, restore_modes
    ):
        """Mode 0o200 passes W_OK, yet creating anything inside still fails.

        Traversing into a directory needs the execute bit, so W_OK alone would
        accept a directory the download cannot actually write into, and
        `_load_llm()` would swallow the resulting PermissionError.
        """
        unsearchable = tmp_path / "write-only"
        unsearchable.mkdir()
        unsearchable.chmod(0o200)
        restore_modes.append(unsearchable)

        if os.access(unsearchable, os.X_OK):  # pragma: no cover - root ignores the bit
            pytest.skip("running as root; the unsearchable case cannot be modelled")
        assert os.access(unsearchable, os.W_OK), "premise: W_OK alone would accept this"

        self._select(monkeypatch, unsearchable)

        def _must_not_download(**kwargs):
            raise AssertionError("must not download into an unusable directory")

        monkeypatch.setitem(
            sys.modules, "huggingface_hub", MagicMock(hf_hub_download=_must_not_download)
        )

        with pytest.raises(RuntimeError) as excinfo:
            local_llm._download_model()

        assert "MNEMOSYNE_MODEL_CACHE_DIR" in str(excinfo.value)
        assert str(unsearchable) in str(excinfo.value)

    def test_failure_never_falls_back_to_the_default_location(
        self, monkeypatch, tmp_path
    ):
        """The whole point: no silent return to ~/.hermes/mnemosyne/models."""
        blocker = tmp_path / "blocked"
        blocker.write_text("file", encoding="utf-8")
        self._select(monkeypatch, blocker / "models")
        default = Path.home() / ".hermes" / "mnemosyne" / "models"

        def _must_not_download(**kwargs):
            raise AssertionError("must not download after a cache-dir failure")

        monkeypatch.setitem(
            sys.modules, "huggingface_hub", MagicMock(hf_hub_download=_must_not_download)
        )

        with pytest.raises(RuntimeError) as excinfo:
            local_llm._download_model()

        assert str(default) not in str(excinfo.value)

    def test_unset_default_failure_suggests_the_variable(
        self, monkeypatch, tmp_path, caplog
    ):
        """With no override set, the error offers one instead of blaming it."""
        blocker = tmp_path / "plain"
        blocker.write_text("file", encoding="utf-8")
        self._select(monkeypatch, blocker / "models", from_env=False)

        with caplog.at_level(logging.ERROR, logger=local_llm.__name__):
            with pytest.raises(RuntimeError) as excinfo:
                local_llm._ensure_model_cache_dir()

        logged = caplog.records[0].getMessage()
        for message in (str(excinfo.value), logged):
            assert str(blocker / "models") in message
            assert "Set MNEMOSYNE_MODEL_CACHE_DIR to relocate it." in message
            # Must not blame a variable the user never set.
            assert "is set to" not in message
