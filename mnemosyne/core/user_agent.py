"""Canonical application User-Agent for outbound HTTP requests.

Some OpenAI-compatible providers reject the default library User-Agent
(``Python-urllib/3.x``, ``python-httpx/x.y``) outright, which surfaced as
embedding and remote-consolidation failures with no actionable diagnostic.
Send an explicit application identity instead.

The version is read from the installed package rather than hardcoded: a
literal string goes stale the moment ``mnemosyne/__init__.py`` bumps the
version, and the header would then advertise a release that no longer
matches the running code.
"""

from __future__ import annotations

_FALLBACK_VERSION = "0.0.0"


def application_user_agent() -> str:
    """Return the ``Mnemosyne/<version>`` User-Agent for remote API calls."""
    try:
        import mnemosyne

        version = getattr(mnemosyne, "__version__", "") or ""
    except Exception:  # pragma: no cover - defensive; import must never break a request
        version = ""
    return f"Mnemosyne/{str(version).strip() or _FALLBACK_VERSION}"
