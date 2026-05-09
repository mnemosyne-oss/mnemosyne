"""Chat text normalization for Mnemosyne ingestion (NAI-1).

Parsers like ClausIE/MinIE were built for Wikipedia, not chat logs.
This module provides aggressive regex normalization to make casual
messages parseable by structured extraction tools.

All algorithmic. Zero LLM calls. Zero new dependencies.
"""

import re
from typing import Optional

# ── Contraction expansion table ──────────────────────────────
# Matched against word boundaries, not spaces
_CONTRACTIONS: list[tuple[str, str]] = [
    (r"\bu\b", "you"),
    (r"\bur\b", "your"),
    (r"\bu're\b", "you are"),
    (r"\br\b", "are"),
    (r"\by\b", "why"),
    (r"\bb4\b", "before"),
    (r"\bbc\b", "because"),
    (r"\bcuz\b", "because"),
    (r"\bgonna\b", "going to"),
    (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "got to"),
    (r"\bkinda\b", "kind of"),
    (r"\bsorta\b", "sort of"),
    (r"\bdunno\b", "don't know"),
    (r"\blemme\b", "let me"),
    (r"\bgimme\b", "give me"),
    (r"\boutta\b", "out of"),
    (r"\bhafta\b", "have to"),
    (r"\bshoulda\b", "should have"),
    (r"\bwoulda\b", "would have"),
    (r"\bcoulda\b", "could have"),
]

# ── Filler / reaction words to strip ──────────────────────────
_FILLER_WORDS: set[str] = {
    "lol", "lmao", "lmaoo", "lmfao", "rofl", "omg", "omgg",
    "omggg", "brb", "idk", "idc", "tbh", "imo", "imho",
    "fwiw", "irl", "afaik", "iirc", "tldr", "nvm", "ikr",
    "wtf", "smh", "fr", "ngl", "istg", "w", "wdym",
}

# ── Fragment-starting verbs that need implicit subjects ───────
_FRAGMENT_STARTERS: set[str] = {
    "going", "coming", "thinking", "wondering",
    "feeling", "trying", "hoping", "planning",
    "working", "looking", "checking", "running",
    "testing", "building", "fixing", "deploying",
}


def normalize_chat(text: str, *, add_implicit_subjects: bool = True) -> Optional[str]:
    """Aggressive regex normalization for casual chat messages.

    Returns None if the message has no extractable meaning (too short,
    only filler/reactions).

    Processing order:
    1. Lowercase
    2. Expand contractions via word-boundary regex (u → you, gonna → going to)
    3. Strip filler/reaction words (lol, omg, brb, etc.)
    4. Collapse repeated characters (omgggg → omg)
    5. Remove emojis and non-ASCII
    6. Normalize whitespace
    7. Fragment detection: too short (<2 meaningful words) = None
    8. Implicit subject injection (going → i am going)

    Args:
        text: Raw chat message
        add_implicit_subjects: If True, prepend 'i am' to fragments
            that start with verbs

    Returns:
        Normalized text, or None if no extractable meaning remains.
    """
    if not text or not text.strip():
        return None

    # Step 1: Lowercase
    text = text.lower().strip()

    # Step 2: Expand contractions (word-boundary regex)
    for pattern, replacement in _CONTRACTIONS:
        text = re.sub(pattern, replacement, text)

    # Step 3: Strip filler/reaction words
    words = text.split()
    meaningful = [w for w in words if w.strip(".,!?;:'\"") not in _FILLER_WORDS]
    if not meaningful:
        return None
    text = " ".join(meaningful)

    # Step 4: Collapse repeated characters (omgggg → omg)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Step 5: Remove emojis and non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Step 6: Normalize whitespace
    text = " ".join(text.split())

    # Step 7: Fragment detection — need at least 2 meaningful words
    word_count = len(text.split())
    if word_count < 2:
        # Single long word might be a name/tool/endpoint
        if word_count == 1 and len(text.split()[0]) > 5:
            return text
        return None

    # Step 8: Implicit subject injection (only for true fragments: 2 words)
    if add_implicit_subjects and word_count == 2:
        first_word = text.split()[0] if text else ""
        if first_word in _FRAGMENT_STARTERS:
            text = "i am " + text

    return text


def normalize_batch(messages: list[str]) -> list[Optional[str]]:
    """Normalize a batch of messages. Returns None for unparseable ones.

    Useful for preprocessing entire conversations before entity extraction.
    """
    return [normalize_chat(msg) for msg in messages]


# ── Diagnostics: measure what fraction of messages survive normalization ──

def extraction_rate(messages: list[str]) -> dict:
    """Check how many messages survive normalization.

    Returns:
        Dict with total, survived, rate, and sample of dropped messages.
    """
    normalized = normalize_batch(messages)
    survived = [n for n in normalized if n is not None]
    dropped = [m for m, n in zip(messages, normalized) if n is None]

    return {
        "total": len(messages),
        "survived": len(survived),
        "dropped": len(dropped),
        "rate": round(len(survived) / len(messages), 3) if messages else 0.0,
        "dropped_samples": dropped[:5],
    }
