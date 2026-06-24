"""
Mnemosyne Local LLM Consolidation
=================================
Lightweight on-device summarization for the sleep/consolidation cycle.
Uses llama-cpp-python (ARM64 + x86_64 native) with ctransformers fallback.
Falls back to aaak encoding if the model is unavailable or inference fails.

Model cache: ~/.hermes/mnemosyne/models/
Default model: openbmb/MiniCPM5-1B-GGUF (Q4_K_M, ~656MB)
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional

# --- Config ------------------------------------------------------------------
DEFAULT_MODEL_REPO = "openbmb/MiniCPM5-1B-GGUF"
DEFAULT_MODEL_FILE = "MiniCPM5-1B-Q4_K_M.gguf"
MODEL_CACHE_DIR = Path.home() / ".hermes" / "mnemosyne" / "models"

LLM_ENABLED = os.environ.get("MNEMOSYNE_LLM_ENABLED", "true").lower() in ("1", "true", "yes")
LLM_MAX_TOKENS=int(os.environ.get("MNEMOSYNE_LLM_MAX_TOKENS", "2048") or "2048")
LLM_N_THREADS = int(os.environ.get("MNEMOSYNE_LLM_N_THREADS", "4"))
LLM_N_CTX = int(os.environ.get("MNEMOSYNE_LLM_N_CTX", "2048"))

# Override model via env
_env_repo = os.environ.get("MNEMOSYNE_LLM_REPO")
_env_file = os.environ.get("MNEMOSYNE_LLM_FILE")
if _env_repo and _env_file:
    DEFAULT_MODEL_REPO = _env_repo
    DEFAULT_MODEL_FILE = _env_file

# Remote API config
LLM_BASE_URL = os.environ.get("MNEMOSYNE_LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("MNEMOSYNE_LLM_API_KEY", "")
LLM_REMOTE_MODEL = os.environ.get("MNEMOSYNE_LLM_MODEL", "")
LLM_TIMEOUT = float(os.environ.get("MNEMOSYNE_LLM_TIMEOUT", "60"))

# Retryable errors only: 404/400 (model-not-found), 5xx, connection. Not 401/403/429.
LLM_FALLBACK_MODELS = [
    m.strip() for m in os.environ.get("MNEMOSYNE_LLM_FALLBACK_MODELS", "").split(",")
    if m.strip()
]
LLM_FALLBACK_BASE_URL = os.environ.get("MNEMOSYNE_LLM_FALLBACK_BASE_URL", "").rstrip("/") or LLM_BASE_URL
LLM_FALLBACK_API_KEY = os.environ.get("MNEMOSYNE_LLM_FALLBACK_API_KEY", "") or LLM_API_KEY

# Host LLM adapter (Hermes or another agent). Disabled by default to preserve
# existing standalone behavior. When MNEMOSYNE_HOST_LLM_ENABLED=true and a
# backend is registered via mnemosyne.core.llm_backends.set_host_llm_backend(),
# the host backend is consulted before the existing remote/local chain.
# See docs/hermes-llm-integration.md for the full behavior model.
HOST_LLM_ENABLED = os.environ.get("MNEMOSYNE_HOST_LLM_ENABLED", "false").lower() in ("1", "true", "yes")
HOST_LLM_PROVIDER = os.environ.get("MNEMOSYNE_HOST_LLM_PROVIDER", "").strip() or None
HOST_LLM_MODEL = os.environ.get("MNEMOSYNE_HOST_LLM_MODEL", "").strip() or None
HOST_LLM_TIMEOUT = 15.0  # Per-attempt safety cap; not user-facing.
# Host context window: local-model-calibrated LLM_N_CTX (2048) is too small for
# Codex/GPT-class aux models; use this larger budget when the host is the path.
HOST_LLM_N_CTX = int(os.environ.get("MNEMOSYNE_HOST_LLM_N_CTX", "32000"))

# Optional consolidation prompt override. Supports {source}, {memories}, and
# {memory_count}; unset keeps the historical built-in prompt.
SLEEP_PROMPT = os.environ.get("MNEMOSYNE_SLEEP_PROMPT", "").strip()

# --- Lazy singleton ----------------------------------------------------------
_llm_instance = None
_llm_backend = None  # "llamacpp", "ctransformers", or None
_llm_available = None  # None = not checked yet


def _ensure_sys_path():
    """Ensure /usr/local/lib/python3.11/site-packages is in sys.path
    so ctransformers is discoverable when Hermes runs in a venv."""
    sp = "/usr/local/lib/python3.11/site-packages"
    if sp not in sys.path and os.path.isdir(sp):
        sys.path.append(sp)


def _model_path() -> Optional[Path]:
    """Return path to the local GGUF model file, or None if not downloaded."""
    candidate = MODEL_CACHE_DIR / DEFAULT_MODEL_FILE
    return candidate if candidate.exists() else None


def _download_model() -> Path:
    """Download the GGUF model from HuggingFace if not present."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_CACHE_DIR / DEFAULT_MODEL_FILE
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface-hub"
        )

    downloaded = hf_hub_download(
        repo_id=DEFAULT_MODEL_REPO,
        filename=DEFAULT_MODEL_FILE,
        local_dir=str(MODEL_CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def _load_llm_llamacpp(model_path: Path):
    """Load the GGUF model via llama-cpp-python. Returns Llama instance or None."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return None

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=LLM_N_CTX,
            n_threads=LLM_N_THREADS,
            verbose=False,
        )
        return llm
    except Exception:
        return None


def _load_llm_ctransformers(model_path: Path):
    """Load the GGUF model via ctransformers (x86_64 only). Returns model or None."""
    _ensure_sys_path()

    try:
        from ctransformers import AutoModelForCausalLM
    except ImportError:
        return None

    try:
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            model_type="llama",
            max_new_tokens=LLM_MAX_TOKENS,
            threads=LLM_N_THREADS,
            context_length=LLM_N_CTX,
        )
    except Exception:
        return None


def _load_llm():
    """Lazy-load the best available local LLM backend.
    
    Priority: llama-cpp-python > ctransformers (x86_64 fallback).
    Returns the loaded model/LLM instance, or None if no backend works.
    """
    global _llm_instance, _llm_backend, _llm_available

    if _llm_instance is not None:
        return _llm_instance

    if not LLM_ENABLED:
        _llm_available = False
        return None

    # Get or download model file
    model_file = _model_path()
    if model_file is None:
        try:
            model_file = _download_model()
        except Exception:
            _llm_available = False
            return None

    # Try llama-cpp-python first (works on ARM64 + x86_64)
    llm = _load_llm_llamacpp(model_file)
    if llm is not None:
        _llm_instance = llm
        _llm_backend = "llamacpp"
        _llm_available = True
        return _llm_instance

    # Fall back to ctransformers (x86_64 only)
    llm = _load_llm_ctransformers(model_file)
    if llm is not None:
        _llm_instance = llm
        _llm_backend = "ctransformers"
        _llm_available = True
        return _llm_instance

    _llm_available = False
    return None


def _call_local_llm(prompt: str) -> Optional[str]:
    """Run inference on the local LLM using whichever backend is loaded."""
    llm = _load_llm()
    if llm is None:
        return None

    try:
        if _llm_backend == "llamacpp":
            # llama-cpp-python uses chat completion API
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                stop=["</s>", "<|user|>"],
                temperature=0.3,
            )
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
        else:
            # ctransformers uses direct callable
            return llm(prompt, max_new_tokens=LLM_MAX_TOKENS, stop=["</s>", "<|user|>"])
    except Exception:
        return None


def _memory_lines(memories: List[str]) -> str:
    """Format memories for insertion into consolidation prompts."""
    return "\n".join(f"- {m}" for m in memories if m)


def _format_sleep_prompt(memories: List[str], source: str = "") -> Optional[str]:
    """Render MNEMOSYNE_SLEEP_PROMPT when configured.

    Supported placeholders:
    - {source}: source label passed by sleep() / summarize_memories()
    - {memories}: newline-separated bullet list of source memories
    - {memory_count}: number of non-empty memories in this chunk
    """
    if not SLEEP_PROMPT:
        return None
    non_empty = [m for m in memories if m]
    return SLEEP_PROMPT.format(
        source=source,
        memories=_memory_lines(memories),
        memory_count=len(non_empty),
    )


def _build_prompt(memories: List[str], source: str = "") -> str:
    """Build a consolidation prompt from a list of memory strings.

    Uses a plain-text instruction format (no special model tokens)
    suitable for both local GGUF models and any LLM. For host LLM
    calls, use :func:`_build_host_prompt` instead.
    """
    custom = _format_sleep_prompt(memories, source=source)
    if custom is not None:
        return custom

    header = (
        "Summarize the following memories into 1-3 concise sentences. "
        "Preserve facts, names, preferences, and decisions. Discard fluff."
    )
    if source:
        header += f" Source: {source}."

    prompt = f"/no_think\n{header}\n\n{_memory_lines(memories)}\n\nSummary:"
    return prompt


def _build_host_prompt(memories: List[str], source: str = "") -> str:
    """Plain-text consolidation prompt for host LLMs (no local-model tokens).

    The host adapter wraps this string as the user-message content of a
    Chat Completions call; embedding local-model chat-template tokens here
    would degrade output quality on every modern aux provider.
    """
    custom = _format_sleep_prompt(memories, source=source)
    if custom is not None:
        return custom

    header = (
        "Summarize the following memories into 1-3 concise sentences. "
        "Preserve facts, names, preferences, and decisions. Discard fluff."
    )
    if source:
        header += f" Source: {source}."

    return f"{header}\n\n{_memory_lines(memories)}"


def _host_backend_will_handle_call() -> bool:
    """True iff the host backend will be the chosen path for an LLM call.

    Used to pick the right context budget at chunk time (HOST_LLM_N_CTX vs
    LLM_N_CTX) and to short-circuit llm_available() for Hermes-only users.
    """
    if not LLM_ENABLED or not HOST_LLM_ENABLED:
        return False
    try:
        from mnemosyne.core.llm_backends import get_host_llm_backend
        return get_host_llm_backend() is not None
    except Exception:
        return False


def _try_host_llm(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
):
    """Attempt the host LLM backend if enabled and registered.

    Returns ``(attempted, text)``:

    - ``(False, None)`` when host is disabled, MNEMOSYNE_LLM_ENABLED is false,
      or no backend is registered. Caller should proceed with the existing
      remote/local fallback chain.
    - ``(True, text-or-None)`` when the backend was called. The ``attempted``
      flag is the sentinel callers use to honor the precedence rule: when
      host is enabled and was attempted, the existing MNEMOSYNE_LLM_BASE_URL
      path MUST be skipped on failure; fall straight to local GGUF, then None.

    See ``docs/hermes-llm-integration.md`` for the full behavior model.
    """
    if not LLM_ENABLED or not HOST_LLM_ENABLED:
        return (False, None)
    try:
        from mnemosyne.core.llm_backends import call_host_llm, get_host_llm_backend
    except Exception:
        return (False, None)
    if get_host_llm_backend() is None:
        return (False, None)
    raw = call_host_llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=HOST_LLM_TIMEOUT,
        provider=HOST_LLM_PROVIDER,
        model=HOST_LLM_MODEL,
    )
    # NB: do NOT run host output through _clean_output(): that helper exists
    # to scrub local-model prompt-template echoes and bulleted prompt repeats
    # from local-model output. Host LLMs (Codex/GPT-class) don't echo our
    # prompt format, AND extract_facts() relies on `- bullet` lines surviving
    # so _parse_facts() can consume them. Just trim whitespace.
    text = raw.strip() if isinstance(raw, str) and raw.strip() else None
    if text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return (True, text)


def _clean_output(text: str) -> str:
    """Strip assistant tokens and extra whitespace from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<|assistant|>", "").replace("<|user|>", "")
    text = text.replace("</s>", "").strip()
    text = re.sub(r"^(Summarize the following memories.*?[.!?:]\s*)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(Preserve facts.*?[.!?:]\s*)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^Source:.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-*]\s.*\n", "", text, flags=re.MULTILINE)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English, with safety margin."""
    return max(1, len(text) // 4)


def _prompt_token_budget() -> int:
    """Return usable token budget for memory content (reserves overhead + output).

    Picks the larger HOST_LLM_N_CTX when the host backend will handle the
    call; otherwise the local-model-calibrated LLM_N_CTX. This avoids the
    multi-chunk-summary degradation on 128K-context aux providers.

    NOTE: output_reserve is capped at min(LLM_MAX_TOKENS, n_ctx // 4) so
    that LLM_MAX_TOKENS == n_ctx (the default for MiniCPM5-1B with 2048
    context) doesn't leave a negative budget for memory content. See
    BEAM-benchmark root-cause analysis (May 2026). Summarized output fits
    in 128-256 tokens for consolidation; reserving more than 1/4 of the
    context window for output starves the input side.
    """
    overhead = 80
    n_ctx = HOST_LLM_N_CTX if _host_backend_will_handle_call() else LLM_N_CTX
    output_reserve = min(LLM_MAX_TOKENS, max(128, n_ctx // 4))
    safety_margin = int(n_ctx * 0.2)
    return max(64, n_ctx - overhead - output_reserve - safety_margin)


def chunk_memories_by_budget(memories: List[str], source: str = "") -> List[List[str]]:
    """Split memories into chunks that fit within the LLM context window."""
    if not memories:
        return []

    budget = _prompt_token_budget()
    chunks = []
    current_chunk = []
    current_tokens = 0

    header = (
        "Summarize the following memories into 1-3 concise sentences. "
        "Preserve facts, names, preferences, and decisions. Discard fluff."
    )
    if source:
        header += f" Source: {source}."
    header_tokens = _estimate_tokens(header + "\n\n")

    format_overhead = _estimate_tokens("- \n")
    available = budget - header_tokens

    for memory in memories:
        mem_tokens = _estimate_tokens(memory) + format_overhead
        if mem_tokens > budget:
            continue
        if current_tokens + mem_tokens > available and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(memory)
        current_tokens += mem_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def llm_available() -> bool:
    """Check whether any LLM backend (host, remote, or local) is available.

    Returns True for Hermes-only users (no MNEMOSYNE_LLM_BASE_URL, no local
    GGUF) as long as a host backend is registered and enabled — otherwise
    sleep would skip ``summarize_memories()`` before the host path could run.
    """
    global _llm_available
    # 0. Host backend (if a host is registered and the user opted in).
    if _host_backend_will_handle_call():
        return True
    # 1. Remote API: only consider it when LLM is globally enabled.
    if LLM_ENABLED and LLM_BASE_URL:
        return True
    if _llm_available is not None:
        return _llm_available
    _load_llm()
    return bool(_llm_available)


def _is_retryable_status(status_code: int) -> bool:
    """Return True if an HTTP status is worth retrying against a fallback model.

    404/400: the requested model is missing or unrecognized on this endpoint —
    another model name on the same host may exist. 5xx: transient server-side
    failure. 401/403/429 are NOT retryable: a bad key or rate limit won't be
    fixed by swapping model names.
    """
    if status_code in (401, 403, 429):
        return False
    if status_code in (404, 400) or 500 <= status_code < 600:
        return True
    return False


def _call_remote_llm_with_model(
    prompt: str,
    model: str,
    temperature: float = 0.3,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = LLM_TIMEOUT,
) -> "tuple[Optional[str], Optional[int], Optional[Exception]]":
    """Call an OpenAI-compatible endpoint with a specific model name.

    Returns ``(text, status, exc)``:

    - ``(text, None, None)`` on success (text may be None for an empty body).
    - ``(None, status, exc)`` on HTTP/network failure. ``status`` is the HTTP
      status code when available, else None. ``exc`` is the underlying exception
      (HTTPStatusError, ConnectError, TimeoutException, etc.) or None.
    - ``(None, None, exc)`` for non-HTTP failures (JSON decode, malformed body).

    Callers (see ``_call_remote_llm``) use ``status`` to decide whether to
    retry against ``LLM_FALLBACK_MODELS``.
    """
    base_url = (base_url or LLM_BASE_URL).rstrip("/")
    if not base_url:
        return (None, None, RuntimeError("MNEMOSYNE_LLM_BASE_URL is empty"))
    api_key = api_key if api_key is not None else LLM_API_KEY

    import json

    try:
        import httpx
        has_httpx = True
    except ImportError:
        has_httpx = False

    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model or "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": temperature,
        "stop": ["</s>", "<|user|>"]
    }

    try:
        if has_httpx:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                status = response.status_code
                if status >= 400:
                    try:
                        response.raise_for_status()
                    except Exception as exc:
                        return (None, status, exc)
                try:
                    data = response.json()
                except Exception as exc:
                    return (None, status, exc)
        else:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status = getattr(resp, "status", 200)
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                return (None, exc.code, exc)
            except Exception as exc:
                return (None, None, exc)

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if choices and choices[0].get("message", {}).get("content"):
            return (choices[0]["message"]["content"], status, None)
        return (None, status, None)
    except Exception as exc:
        return (None, None, exc)


def _call_remote_llm(prompt: str, temperature: float = 0.3) -> Optional[str]:
    """Call an OpenAI-compatible remote endpoint for summarization.

    ``temperature`` defaults to 0.3 (paraphrase-safe for consolidation);
    callers that need deterministic output (e.g., fact extraction) can
    pass ``temperature=0.0``.

    Tries the primary ``LLM_REMOTE_MODEL`` first; on a retryable error
    (see ``_is_retryable_status``), iterates ``LLM_FALLBACK_MODELS`` in
    order. Returns the first successful response, or ``None`` if every
    model fails (caller falls through to local GGUF / None).
    """
    if not LLM_BASE_URL:
        return None

    primary = LLM_REMOTE_MODEL or "local"
    candidates: List[tuple[str, str, str]] = [(primary, LLM_BASE_URL, LLM_API_KEY)]
    for fb in LLM_FALLBACK_MODELS:
        if fb and fb != primary:
            candidates.append((fb, LLM_FALLBACK_BASE_URL or LLM_BASE_URL, LLM_FALLBACK_API_KEY or LLM_API_KEY))

    for model, base_url, api_key in candidates:
        text, status, exc = _call_remote_llm_with_model(
            prompt, model, temperature, base_url=base_url, api_key=api_key
        )
        if text:
            return text
        if status is None:
            continue
        if not _is_retryable_status(status):
            return None
    return None


def summarize_memories(memories: List[str], source: str = "") -> Optional[str]:
    """Summarize a batch of working-memory items into a single episodic string.

    Fallback chain:

    0. Host-provided LLM backend, only if MNEMOSYNE_HOST_LLM_ENABLED=true,
       MNEMOSYNE_LLM_ENABLED=true, AND a backend is registered. When this
       path is attempted but produces no usable text, the existing remote
       URL is **skipped** — falls through to local GGUF, then None. This
       prevents accidentally routing memory content to a stale
       MNEMOSYNE_LLM_BASE_URL the user forgot to clear.
    1. Remote OpenAI-compatible API (if MNEMOSYNE_LLM_BASE_URL is set
       AND MNEMOSYNE_LLM_ENABLED is not false).
    2. llama-cpp-python (ARM64 + x86_64 native).
    3. ctransformers (x86_64 only, legacy).
    4. Return None → caller falls back to AAAK encoding.
    """
    if not memories:
        return None

    # Chunk large memory lists to stay within context window limits.
    # chunk_memories_by_budget() respects LLM_N_CTX and safety margins.
    chunks = chunk_memories_by_budget(memories, source=source)

    def _summarize_chunk(chunk_memories: List[str], chunk_source: str = "") -> Optional[str]:
        """Summarize a single chunk of memories via the fallback chain."""
        host_prompt = _build_host_prompt(chunk_memories, source=chunk_source)
        prompt = _build_prompt(chunk_memories, source=chunk_source)

        # 0. Host backend.
        attempted, text = _try_host_llm(host_prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.3)
        if attempted:
            if text:
                return text
            raw = _call_local_llm(prompt)
            if raw:
                cleaned = _clean_output(raw)
                return cleaned if cleaned else None
            return None

        # 1. Remote API (skip if MNEMOSYNE_FORCE_LOCAL=1 or remote call fails).
        if LLM_ENABLED and LLM_BASE_URL and not os.environ.get("MNEMOSYNE_FORCE_LOCAL", "").lower() in ("1", "true", "yes"):
            raw = _call_remote_llm(prompt)
            if raw:
                cleaned = _clean_output(raw)
                return cleaned if cleaned else None

        # 2. Local LLM (llama-cpp-python or ctransformers fallback).
        raw = _call_local_llm(prompt)
        if raw:
            cleaned = _clean_output(raw)
            return cleaned if cleaned else None
        return None

    # Summarize each chunk individually.
    chunk_summaries = []
    for chunk in chunks:
        summary = _summarize_chunk(chunk, chunk_source=source)
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return None

    # If multiple chunks, do a second-pass summary to consolidate chunk summaries.
    if len(chunk_summaries) > 1:
        final = _summarize_chunk(chunk_summaries, chunk_source=f"{source} [chunked {len(chunks)} parts]")
        return final if final else chunk_summaries[0]

    return chunk_summaries[0]
