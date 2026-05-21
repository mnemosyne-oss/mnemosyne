"""Shared surface memory classifier and helpers for cross-agent metadata.

Surface memory is for stable user/system/workflow metadata only.
Not for raw chat, task results, wiki/source contents, web/repo summaries,
sensitive data, or low-importance session details.
"""

from typing import Optional, Dict, Any
import re
import hashlib


# Secret detection patterns
SURFACE_SECRET_PATTERNS = [
    "api_key", "apikey", "secret", "password", "passwd", "bearer",
    "private key", "ssh-rsa", "-----begin", "credential", "cookie",
]

# Meta/path terms for location facts
SURFACE_META_PATH_TERMS = [
    "path", "repo", "repository", "project root", "project path", "vault", "wiki",
    "config", "database", "db", "bank", "directory", "folder", "lives at",
    "located at", "stored at", "is at", "under", "home", "port", "service",
]

# Preference terms
SURFACE_PREFERENCE_TERMS = [
    "prefers", "preference", "dislikes", "wants agents", "wants system",
    "always", "never", "do not", "don't", "workflow", "style", "when given",
]

# Source content terms (blocks external/research facts)
SURFACE_SOURCE_CONTENT_TERMS = [
    "added to", "summary of", "readme", "paper", "article", "blog post",
    "documentation says", "source says", "repo is", "repository is", "project is",
    "library", "framework", "implements", "written in", "built with",
    "https://", "http://",
]


def surface_hash(content: str) -> str:
    """Generate stable 24-char hash for surface memory deduplication."""
    normalized = " ".join(str(content).lower().split())
    return hashlib.sha256(f"surface:v1:{normalized}".encode("utf-8")).hexdigest()[:24]


def looks_secret_for_surface(content: str) -> bool:
    """Check if content looks like a secret/credential."""
    lowered = content.lower()
    if any(p in lowered for p in SURFACE_SECRET_PATTERNS):
        return True
    # Cheap high-entropy-ish guard for obvious credential assignments.
    if re.search(r"(?i)(key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}", content):
        return True
    return False


def classify_surface_candidate(
    content: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Return allowed surface kind, or None to keep memory private.

    Surface memory is only durable meta about user/system/workflow.
    Content learned from wiki/web/source material stays private by default.
    
    Returns one of: 'meta', 'preference', 'correction', 'identity', or None.
    """
    lowered = content.lower()
    meta = metadata or {}

    # Block if metadata indicates external source
    if meta.get("source_doc") or meta.get("url") or meta.get("page") or meta.get("source_url"):
        return None

    # Global source-content block: external/research facts are not shared surface,
    # even when they mention repo/project/library names.
    if any(term in lowered for term in SURFACE_SOURCE_CONTENT_TERMS):
        # Narrow exception: explicit location/config facts with actual local path
        # are meta, not source content.
        has_local_path = bool(re.search(r"(?:^|\s)(?:/home/|~/|~\.hermes/)", content))
        has_meta_term = any(term in lowered for term in SURFACE_META_PATH_TERMS)
        if not (has_local_path and has_meta_term and not re.search(r"https?://", content)):
            return None

    # Preference/correction/identity sources
    if source in {"preference", "correction", "identity", "builtin_memory_user"} or source.startswith("builtin_memory_"):
        if any(term in lowered for term in SURFACE_PREFERENCE_TERMS) or source in {"identity", "correction"}:
            return {
                "preference": "preference",
                "builtin_memory_user": "preference",
                "correction": "correction",
                "identity": "identity",
            }.get(source, "preference")

    # Local path + meta term = meta
    has_local_path = bool(re.search(r"(?:^|\s)(?:/home/|~/|~\.hermes/)", content))
    if has_local_path and any(term in lowered for term in SURFACE_META_PATH_TERMS):
        return "meta"

    # Port/service mentions
    if re.search(r"\b(port|service)\b.*\b\d{2,5}\b", lowered):
        return "meta"

    # Preference terms
    if any(term in lowered for term in SURFACE_PREFERENCE_TERMS):
        return "preference"

    return None


def label_surface_content(content: str, kind: str) -> str:
    """Add 'Surface <kind>:' prefix if not already present."""
    if content.lower().startswith((
        "surface fact:", "surface preference:", "surface correction:",
        "surface identity:", "surface meta:"
    )):
        return content
    
    label = {
        "preference": "Surface preference",
        "correction": "Surface correction",
        "identity": "Surface identity",
        "meta": "Surface meta",
    }.get(kind, "Surface meta")
    
    return f"{label}: {content}"
