"""LLM client for fact extraction via OpenRouter.

Reuses the same patterns as _benchmarks/evaluate_beam_end_to_end.py LLMClient.
100% open source (MIT).
"""

import json as _json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_EXTRACTION_MODEL = os.environ.get(
    "MNEMOSYNE_EXTRACTION_MODEL",
    "google/gemini-2.5-flash",
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")
FALLBACK_MODELS = [
    "google/gemini-flash-latest",
     # Fallback: older
]


class ExtractionClient:
    """OpenAI-compatible API client for fact extraction via OpenRouter."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = model or DEFAULT_EXTRACTION_MODEL
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.call_count = 0

    def chat(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send chat completion with fallback and retry.

        Returns the response text, or empty string on total failure.

        [C13.b] Records the API-TRANSPORT outcome of the chat call.
        Does NOT record cloud-tier "extraction success" — that's only
        known after the caller parses the response as facts (see
        `extract_facts` below, which records the final outcome).

        Transport outcomes recorded:
          - record_attempt("cloud") — chat() entered
          - record_no_output("cloud") — API returned empty content
            (post-retry-and-fallback) OR all retries failed without
            a usable response. The latter also records a failure.
          - record_failure("cloud") — every model + retry combination
            raised an exception. Captures `last_exc` for the sample.

        /review caught the pre-fix behavior of recording success
        on non-empty HTTP responses — that conflated "API returned
        text" with "extraction yielded facts," producing
        success-AND-failure double-counting when `extract_facts()`
        couldn't parse the response.
        """
        from .diagnostics import get_diagnostics, _safe_for_log
        diag = get_diagnostics()
        diag.record_attempt("cloud")

        models_to_try = [self.model] + [
            m for m in FALLBACK_MODELS if m != self.model
        ]
        last_exc = None

        for model in models_to_try:
            for attempt in range(3):
                try:
                    result = self._call_api(
                        model, messages, temperature, max_tokens
                    )
                    if not result:
                        # API returned empty content. Record on the
                        # cloud-tier no_output counter; extract_facts
                        # will record the outer call outcome.
                        diag.record_no_output("cloud")
                    # Note: don't record success here. extract_facts()
                    # decides based on parseable output.
                    return result
                except Exception as e:
                    last_exc = e
                    msg = str(e)
                    if "429" in msg or "rate" in msg.lower():
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    else:
                        break  # Non-retryable, try next model
            # Brief pause between models
            time.sleep(1)

        # All models failed
        diag.record_failure(
            "cloud", exc=last_exc, reason="all_models_failed"
        )
        if last_exc is not None:
            logger.warning(
                "ExtractionClient.chat: all models failed; last error: %s",
                _safe_for_log(last_exc),
            )
        return ""

    def _call_api(
        self,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Single API call via urllib."""
        url = f"{self.base_url}/chat/completions"
        payload = _json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=payload, headers=headers)
        resp = urllib.request.urlopen(req, timeout=60)
        data = _json.loads(resp.read())
        self.call_count += 1
        return data["choices"][0]["message"]["content"]

    def extract_facts(self, messages: list) -> list:
        """Extract structured facts from a list of conversation messages.

        Args:
            messages: List of dicts with 'role' and 'content' keys.

        Returns:
            List of fact dicts (subject, predicate, object, etc.), or empty list on failure.
        """
        from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE

        # Build conversation text from messages
        conversation_text = ""
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                conversation_text += f"[{i}] [{role}]: {content}\n"

        if not conversation_text.strip():
            return []

        user_prompt = EXTRACTION_USER_TEMPLATE.format(
            conversation_text=conversation_text,
        )

        chat_messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # [C13.b] Outer-call accounting for the cloud entry point.
        # /review caught the pre-fix behavior: chat() recorded
        # transport success but `extract_facts` never called
        # record_call, so totals.success_rate excluded the cloud path
        # entirely. Now extract_facts owns the outer-call signal AND
        # the cloud-tier success/failure based on whether the response
        # actually parses into facts.
        from .diagnostics import get_diagnostics
        diag = get_diagnostics()

        response = self.chat(chat_messages, temperature=0.0, max_tokens=4096)

        if not response:
            # chat() already recorded transport-level failure /
            # no_output. Record the outer-call outcome at the totals
            # level for bird's-eye success-rate accounting.
            diag.record_call(succeeded=False, all_empty=True)
            return []

        # Parse JSON from response
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                facts = _json.loads(response[json_start:json_end])
                if isinstance(facts, list):
                    # Successful extraction: record cloud-tier
                    # success AND outer-call success.
                    diag.record_success("cloud", fact_count=len(facts))
                    diag.record_call(succeeded=True)
                    return facts
            # Response had brackets but contents didn't parse OR no
            # bracket at all OR parsed to non-list. Treat as parse
            # failure on the cloud tier so operators can spot the
            # model returning unusable output.
            diag.record_failure(
                "cloud", reason="no_facts_in_response"
            )
            diag.record_call(succeeded=False, all_empty=True)
        except (_json.JSONDecodeError, ValueError) as e:
            # [C13.b] Operator-visible signal: model returned text
            # but couldn't be parsed as a fact list. Distinguishes
            # "model has nothing to say" (no brackets — handled
            # above) from "model returned malformed JSON" (this
            # branch).
            diag.record_failure(
                "cloud", exc=e, reason="json_parse_failed"
            )
            diag.record_call(succeeded=False)
            logger.warning(
                "ExtractionClient.extract_facts: JSON parse failed on "
                "model response; %d chars returned",
                len(response),
            )

        return []
