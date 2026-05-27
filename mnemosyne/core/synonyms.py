"""
Synonym Expansion for Mnemosyne Queries
========================================
Concept-group-based synonym expansion to improve recall quality.
Covers ~40 concept groups for Tier 1 exact match normalization.

Usage:
    from mnemosyne.core.synonyms import expand_query, normalize_query

    expanded = expand_query("what is the db password")
    # Returns: "what is the (database|db) (password|pass|pwd|credential)"
"""

# Concept groups — canonical form first, then synonyms
SYNONYM_GROUPS = {
    "database": ["db", "datastore", "data_store"],
    "password": ["pass", "pwd", "passwd", "credential", "secret", "token"],
    "config": ["configuration", "settings", "cfg", "setup"],
    "error": ["bug", "issue", "fault", "failure", "crash", "exception", "traceback"],
    "fix": ["repair", "resolve", "solve", "patch", "correct", "address"],
    "deploy": ["deployment", "release", "ship", "push", "rollout"],
    "server": ["host", "machine", "vm", "instance", "node", "vps"],
    "api": ["endpoint", "interface", "service"],
    "key": ["token", "credential", "secret", "api_key"],
    "user": ["account", "profile", "identity", "person"],
    "model": ["llm", "ai", "provider", "gpt", "claude", "gemini"],
    "speed": ["fast", "quick", "performance", "latency", "throughput"],
    "memory": ["recall", "remember", "storage", "retention"],
    "search": ["find", "lookup", "query", "retrieve", "locate"],
    "file": ["document", "doc", "text", "note"],
    "code": ["script", "program", "source", "implementation"],
    "test": ["verify", "check", "validate", "probe", "examine"],
    "backup": ["snapshot", "copy", "save", "archive"],
    "install": ["setup", "configure", "bootstrap", "init"],
    "update": ["upgrade", "refresh", "renew", "sync"],
    "delete": ["remove", "destroy", "purge", "clean", "wipe", "erase"],
    "list": ["show", "display", "enumerate", "catalog"],
    "time": ["date", "when", "timestamp", "schedule"],
    "url": ["link", "address", "uri", "path"],
    "health": ["status", "check", "pulse", "alive", "up"],
    "service": ["daemon", "process", "systemd", "worker"],
    "port": ["socket", "bind", "listen"],
    "network": ["internet", "connection", "connectivity", "dns"],
    "ssh": ["terminal", "shell", "remote", "connect"],
    "git": ["commit", "push", "pull", "repo", "repository", "branch"],
    "log": ["output", "stdout", "stderr", "trace", "debug"],
    "cron": ["schedule", "job", "task", "timer", "periodic"],
    "email": ["mail", "message", "inbox", "smtp"],
    "image": ["picture", "photo", "screenshot", "graphic"],
    "browser": ["web", "page", "site", "navigate", "chrome"],
    "monitor": ["watch", "observe", "track", "survey"],
    "alert": ["notify", "notification", "warning", "ping"],
    "migrate": ["transfer", "move", "relocate", "port"],
    "compare": ["diff", "versus", "vs", "contrast"],
    "save": ["store", "persist", "preserve", "keep"],
}

# Stop words to drop during normalization
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "mine", "yours", "hers", "ours", "theirs",
    "what", "which", "who", "whom", "where", "when", "why",
    "how", "this", "that", "these", "those", "of", "in", "to",
    "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below",
    "between", "under", "and", "but", "or", "nor", "not",
    "so", "than", "too", "very", "just", "about", "also",
    "really", "actually", "basically", "simply", "if", "then",
    "else", "while", "because", "though", "although",
}


def _build_reverse_map() -> dict:
    """Build word → canonical form mapping."""
    reverse = {}
    for canonical, synonyms in SYNONYM_GROUPS.items():
        reverse[canonical] = canonical
        for syn in synonyms:
            reverse[syn] = canonical
    return reverse


_WORD_TO_CANONICAL = _build_reverse_map()


def normalize_query(query: str) -> str:
    """
    Normalize a query for exact match caching (Tier 1).
    
    Steps:
    1. Lowercase
    2. Remove stop words
    3. Map synonyms to canonical forms
    4. Sort unique words
    5. Join
    
    Args:
        query: Raw query string
        
    Returns:
        Normalized query string for cache key
    """
    words = query.lower().split()
    canonical_words = []
    for word in words:
        if word in STOP_WORDS:
            continue
        canonical_words.append(_WORD_TO_CANONICAL.get(word, word))
    return " ".join(sorted(set(canonical_words)))


def expand_query(query: str) -> str:
    """
    Expand query with synonyms for broader FTS5/vector search.
    
    For each word that has a synonym group, expand to include
    all synonyms using OR grouping.
    
    Args:
        query: Raw query string
        
    Returns:
        Expanded query string with (word1|word2|...) groups
    """
    words = query.lower().split()
    expanded_parts = []
    for word in words:
        if word in STOP_WORDS:
            expanded_parts.append(word)
            continue
        canonical = _WORD_TO_CANONICAL.get(word)
        if canonical and canonical in SYNONYM_GROUPS:
            group = [canonical] + SYNONYM_GROUPS[canonical]
            if word not in group:
                group = [word] + group
            expanded_parts.append(f"({'|'.join(group)})")
        else:
            expanded_parts.append(word)
    return " ".join(expanded_parts)


def get_synonyms(word: str) -> list:
    """Get all synonyms for a word, including the canonical form."""
    word = word.lower()
    canonical = _WORD_TO_CANONICAL.get(word, word)
    if canonical in SYNONYM_GROUPS:
        return [canonical] + SYNONYM_GROUPS[canonical]
    return [word]
