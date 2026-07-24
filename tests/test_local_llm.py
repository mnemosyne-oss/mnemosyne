import builtins
import os
import subprocess
import sys
import pytest
from unittest.mock import patch, MagicMock

from mnemosyne.core import local_llm
from mnemosyne.core.llm_backends import (
    CallableLLMBackend,
    set_host_llm_backend,
)


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
             patch.object(local_llm, "_call_local_llm", return_value=None) as mock_local:
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
        raw = f"<think>let me reason</think> The answer is 42."
        assert local_llm._clean_output(raw) == "The answer is 42."

    def test_clean_output_strips_multiline_closed_think_tags(self):
        raw = f"<think>step 1\nstep 2</think>\nFinal answer."
        assert local_llm._clean_output(raw) == "Final answer."

    def test_clean_output_strips_multiple_think_blocks(self):
        raw = f"<think>first</think>middle<think>second</think>end"
        assert local_llm._clean_output(raw) == "middleend"

    def test_clean_output_preserves_text_without_think_tags(self):
        raw = "Just a normal summary with no thinking."
        assert local_llm._clean_output(raw) == "Just a normal summary with no thinking."

    def test_clean_output_empty_after_stripping(self):
        raw = f"<think>only thinking, no output</think>"
        assert local_llm._clean_output(raw) == ""

    def test_clean_output_does_not_strip_unclosed_think_tag(self):
        """Unclosed think tags are left as-is since we cannot determine
        where thinking ends and the response begins."""
        raw = f"middle<think>reasoning that never closes"
        assert local_llm._clean_output(raw) == raw

    def test_try_host_llm_strips_think_tags(self, monkeypatch):
        """Host LLM output with closed think tags should be cleaned."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_TIMEOUT", 5.0)
        monkeypatch.setattr(local_llm, "HOST_LLM_PROVIDER", None)
        monkeypatch.setattr(local_llm, "HOST_LLM_MODEL", None)
        set_host_llm_backend(CallableLLMBackend("test", lambda prompt, **kw: f"<think>reasoning</think>Summary of memories."))

        attempted, text = local_llm._try_host_llm("test prompt", max_tokens=128, temperature=0.3)
        assert attempted is True
        assert text == "Summary of memories."

    def test_try_host_llm_does_not_strip_unclosed_think_tag(self, monkeypatch):
        """Unclosed think tags in host output are left as-is."""
        monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
        monkeypatch.setattr(local_llm, "HOST_LLM_TIMEOUT", 5.0)
        monkeypatch.setattr(local_llm, "HOST_LLM_PROVIDER", None)
        monkeypatch.setattr(local_llm, "HOST_LLM_MODEL", None)
        set_host_llm_backend(CallableLLMBackend("test", lambda prompt, **kw: f"<think>reasoning\nActual output"))

        attempted, text = local_llm._try_host_llm("test prompt", max_tokens=128, temperature=0.3)
        assert attempted is True
        # Unclosed tag is not stripped - we can't tell where thinking ends
        assert "<think>" in text


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
