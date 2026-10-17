"""
Entity Sketching System
Lightweight entity extraction and fuzzy matching without heavy NLP dependencies.

Uses regex patterns for entity extraction and pure Python Levenshtein distance
for fuzzy matching. No spaCy, no PyTorch, no external NLP libraries.

Storage: TripleStore triples (subject=memory_id, predicate="mentions", object="entity_name")
"""

import os
import re
from typing import List, Set, Tuple


# =============================================================================
# STOP WORDS — filtered from entity extraction
# =============================================================================

ENTITY_EXTRACTION_STOP_WORDS: Set[str] = {
    # --- upstream standard stop words ---
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "here", "there", "where",
    "when", "what", "which", "who", "whom", "whose", "how", "why",
    "assistant", "user", "skill", "review", "target", "class",
    "level", "signals", "phase", "api", "pi", "summary", "added",
    "active", "be", "not", "whether", "all", "no", "replying",
    "ai", "memory", "conversation", "fact",
    "false", "true", "none", "null", "signal",
    "hermes", "agent", "model", "system",
    "note", "task", "project", "result", "output", "input", "data",
    "step", "process", "point", "way", "thing", "time", "work",
    # --- expanded: capitalized English sentence-words (chat noise) ---
    "after", "again", "also", "although", "always", "another", "any",
    "anyone", "anything", "around", "away", "back", "because", "before",
    "being", "better", "both", "but", "can", "cannot", "certain", "could",
    "different", "does", "doing", "done", "down", "during", "each",
    "either", "else", "enough", "every", "everyone", "everything",
    "finally", "first", "following", "for", "from", "frustration",
    "further", "getting", "going", "good", "great", "has", "have",
    "having", "however", "if", "into", "just", "keep", "know", "last",
    "like", "look", "looking", "made", "make", "makes", "many", "maybe",
    "might", "more", "most", "much", "must", "need", "needs", "negative",
    "never", "next", "nothing", "off", "once", "one", "only",
    "onto", "other", "others", "otherwise", "out", "over", "part",
    "please", "possible", "rather", "really", "right", "same", "say",
    "see", "seen", "several", "should", "show", "shows", "since", "so",
    "something", "sometimes", "soon", "still", "such", "sure", "take",
    "than", "them", "then", "there", "these", "they", "thing", "things",
    "this", "those", "through", "thus", "time", "times", "today",
    "together", "too", "try", "under", "until", "updated", "upon",
    "used", "using", "very", "want", "wants", "was", "way", "ways",
    "well", "were", "what", "when", "where", "whether", "which",
    "while", "whole", "will", "with", "within", "without", "would",
    "yes", "yet", "you", "your",
    # --- code/UI tokens observed in real extraction noise ---
    "add", "app", "auth", "backup", "bank", "banks", "bootstrap", "bundled",
    "capture", "capturing", "clean", "cleared", "click", "code", "command",
    "commands", "confirmed", "convention", "create", "currently", "db",
    "design", "docs", "encode", "environment", "existing", "file", "files",
    "fixes", "full", "health", "hub", "ids", "image", "load", "loaded",
    "memories", "mode", "net", "nightly", "non", "okay", "operator",
    "option", "overlap", "patch", "pinned", "preference", "pr", "prompt",
    "protected", "read", "rebuild", "redundant", "repo", "research",
    "retire", "session", "shared", "skills", "support", "tonight",
    "todo", "umbrella", "unresolved", "update", "verified", "write",
    "writing",
    # observed high-frequency single-cap words in dev-chat (2nd wave)
    "action", "audit", "august", "check", "commit", "commits", "current",
    "live", "ok", "per", "provider", "run", "status", "two", "verification",
    "let", "acceptance", "account", "accounts", "act", "activities",
    "activity", "additional", "administration", "alternative", "applications",
    "applies", "approve", "archived", "assignment", "audience", "background",
    "blocked", "branch", "bridge", "build", "built", "cancel", "candidate",
    "canonical", "capability", "central", "chat", "cheap", "client",
    "close", "complete", "completed", "config", "confirm", "connected",
    "container", "contents", "convert", "core", "correct", "cost", "created",
    "creating", "credential", "credentials", "cron", "dashboard", "decision",
    "default", "deferred", "delete", "deletion", "deliver", "delivery",
    "deploy", "deployment", "description", "developer", "distinguishing",
    "documentation", "documented", "documenting", "domain", "draft", "drop",
    "durable", "dynamic", "earlier", "encrypt", "endpoints", "english",
    "europe", "even", "events", "evidence", "execute", "execution",
    "explicit", "explicitly", "export", "extended", "facts", "failure",
    "fetch", "filter", "final", "fine", "fix", "fixed", "focus", "follow",
    "fork", "found", "functional", "gaps", "goal", "group", "hard",
    "header", "history", "home", "hook", "housekeeping", "human", "humans",
    "hybrid", "identity", "implemented", "independent", "initial", "install",
    "installed", "instead", "interesting", "invalid", "inventory",
    "investigation", "isolation", "item", "job", "keeping", "kept", "key",
    "kind", "layer", "lifecycle", "likely", "limitation", "line", "list",
    "local", "log", "lost", "main", "maintaining", "mandatory", "mapping",
    "match", "meaning", "middle", "minor", "mixture", "monday", "mondays",
    "mounts", "multiple", "name", "notable", "noted", "notes", "object",
    "open", "operations", "optional", "options", "outcome", "parallel",
    "password", "paste", "pending", "persist", "pin", "plan", "plugin",
    "plus", "positioning", "post", "predicate", "prefer", "preserved",
    "primary", "private", "problem", "procedure", "proceed", "proof",
    "proposal", "proposed", "prose", "public", "pull", "purpose", "push",
    "pushed", "qualify", "quota", "rate", "reading", "reads", "ready",
    "real", "reasoning", "rebase", "receive", "recommended", "reconcile",
    "record", "recorded", "reference", "references", "register", "related",
    "relevant", "remaining", "remind", "remote", "remove", "renamed",
    "renaming", "reorganize", "replaced", "repository", "required",
    "requirement", "resume", "reviews", "revocation", "rewrite", "rewrote",
    "risks", "root", "rotate", "rotation", "router", "routing", "row",
    "runs", "runtime", "safety", "save", "saved", "saves", "schedule",
    "scheduled", "scope", "scripts", "second", "secret", "section",
    "security", "self", "separate", "separating", "server", "service",
    "settings", "shell", "short", "signed", "smoke", "source", "sources",
    "sovereignty", "split", "stable", "stale", "standing", "state",
    "subject", "summarize", "switch", "syntax", "technically", "tell",
    "temporary", "test", "text", "therefore", "tiny", "tomorrow", "tool",
    "tools", "topic", "total", "transformation", "transient", "transversal",
    "treat", "triggered", "trimmed", "triples", "unset", "unsigned",
    "updating", "upstream", "valid", "validate", "validation", "value",
    "verify", "vision", "wait", "watchdog", "websocket", "webhook",
    "welcome", "whenever", "working", "works", "writes", "zero",
}

# Backward compatibility alias
_STOP_WORDS = ENTITY_EXTRACTION_STOP_WORDS


# =============================================================================
# REGEX PATTERNS FOR ENTITY EXTRACTION
# =============================================================================

# Single capitalized word (fallback): Abdias, Python, John.
# Named so extract_entities_regex can tell this pattern's matches apart:
# they alone get the position drop below.
_SINGLE_CAP_WORD = re.compile(r'\b([A-Z][a-zA-Z]{1,20})\b')

# Opt-in position rule, read per call: MNEMOSYNE_ENTITY_POSITION_FILTER=1.
# In conversation text a single capitalized word is capitalized by where it
# stands far more often than because it is a name: a sentence opener
# ("Then", "So", "Okay"), a contraction stem, the first word of a quoted
# phrase. Off by default because short declarative facts open with their
# subject ("Jane is an architect."), and the rule would drop that name.
_POSITION_FILTER_ENV = "MNEMOSYNE_ENTITY_POSITION_FILTER"

# A word is sentence-initial at the start of the text, or when the nearest
# non-space character before it (closing quotes and brackets are skipped too,
# and the * closing a markdown emphasis span) ends a sentence, or when it is
# the first word of a stored turn: a speaker prefix such as `[USER] ` or
# `[ASSISTANT] `, a bracketed uppercase word opening its line and followed by
# a space, counts as a sentence start. Only that shape does: a bracket
# closing mid-sentence is skipped like any closer, and the name after it
# extracts.
_SENTENCE_TERMINATORS = frozenset(".!?:;\u2026\n\r")
_TRAILING_CLOSERS = frozenset("\"')]}*\u2019\u201d")
_SPEAKER_PREFIX = re.compile(r'\[[A-Z]+\]')

# A contraction stem is not a name: an apostrophe is a word boundary, so
# "Don't" matched "Don" and "Haven't" matched "Haven". A single word followed
# by an apostrophe and an unambiguous contraction tail is rejected. The 's
# tail is ambiguous ("Let's", "Alice's") and goes to the name: a possessive
# is the one form a name takes in a sentence about the person.
_CONTRACTION_TAIL = re.compile(r"['\u2019](?:t|ll|re|ve|d|m)\b", re.IGNORECASE)

# The word opening a quoted phrase is capitalized by position, like a
# sentence opener: the title case of a quoted heading ("Guarded canonical
# calls"), the first word of quoted speech ('Still us.'). Only the phrase
# shape is dropped, an opening quote right before the word and a lowercase
# word right after it. A quoted name standing alone ('Alice'), a quoted
# possessive ("Alice's notes") and a quoted list (['Alice', 'Maya']) keep
# their names.
_OPENING_QUOTES = frozenset("\"'\u201c\u2018")
_LOWERCASE_FOLLOWER = re.compile(r' [a-z]')


def _position_filter_enabled() -> bool:
    value = os.environ.get(_POSITION_FILTER_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _is_sentence_initial(text: str, pos: int) -> bool:
    i = pos - 1
    while i >= 0 and (text[i] in " \t" or text[i] in _TRAILING_CLOSERS):
        if text[i] == "]" and text[i + 1] in " \t":
            line_start = max(text.rfind("\n", 0, i), text.rfind("\r", 0, i)) + 1
            if _SPEAKER_PREFIX.fullmatch(text, line_start, i + 1):
                return True
        i -= 1
    return i < 0 or text[i] in _SENTENCE_TERMINATORS


def _opens_quoted_phrase(text: str, start: int, end: int) -> bool:
    if start == 0 or text[start - 1] not in _OPENING_QUOTES:
        return False
    if start > 1 and text[start - 2].isalnum():
        return False
    return _LOWERCASE_FOLLOWER.match(text, end) is not None


def _capitalized_by_position(text: str, start: int, end: int) -> bool:
    return (
        _is_sentence_initial(text, start)
        or _CONTRACTION_TAIL.match(text, end) is not None
        or _opens_quoted_phrase(text, start, end)
    )


_ENTITY_PATTERNS = [
    # @mentions: @username
    re.compile(r'@(\w{2,30})'),
    # Hashtags: #topic
    re.compile(r'#(\w{2,30})'),
    # There is deliberately no quoted-span pattern. A whole quoted span is
    # usually dialogue rather than a name ('Okay,', 'Talia pauses.'), and its
    # punctuation slips past the stop-word filter because 'okay,' != 'okay'.
    # A real name inside quotes still extracts from the capitalized patterns
    # below, which quotes do not block.
    # Capitalized word sequences (2-5 words): New York, Abdias J, San Francisco Bay Area
    re.compile(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){1,4})\b'),
    # Single capitalized word (fallback): Abdias, Python, John
    _SINGLE_CAP_WORD,
]


# =============================================================================
# 1. PURE PYTHON LEVENSHTEIN
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.
    Pure Python, zero dependencies.  O(len(s1) * len(s2)) time, O(min) space.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1  # ensure s2 is the shorter one

    if not s2:
        return len(s1)

    # Use two rows (current and previous) to keep space O(min(len1, len2))
    previous_row = list(range(len(s2) + 1))
    current_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row[0] = i + 1

        for j, c2 in enumerate(s2):
            # Cost: 0 if same character, 1 if different
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)

        # Swap rows
        previous_row, current_row = current_row, previous_row

    return previous_row[len(s2)]


def similarity(s1: str, s2: str) -> float:
    """
    Entity-aware similarity score: 1.0 = identical, 0.0 = completely different.

    Uses case-insensitive comparison with prefix/substring bonuses for
    entity name matching (e.g., "Abdias" vs "Abdias J" = 0.925).
    """
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    if s1_lower == s2_lower:
        return 1.0

    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 1.0

    # Prefix match bonus: 'Abdias' vs 'Abdias J'
    if s1_lower.startswith(s2_lower) or s2_lower.startswith(s1_lower):
        longer = max(len(s1_lower), len(s2_lower))
        shorter = min(len(s1_lower), len(s2_lower))
        # Require at least 30% length ratio to avoid short prefix noise
        # (e.g. "her" prefix-matching "Hermes" with only 3/7 ratio)
        if shorter / longer < 0.3:
            return 0.0
        return 0.7 + (shorter / longer) * 0.3  # 0.7 base + scaled bonus

    # Substring match: 'Mr. Smith' contains 'Smith'
    if s1_lower in s2_lower or s2_lower in s1_lower:
        longer = max(len(s1_lower), len(s2_lower))
        shorter = min(len(s1_lower), len(s2_lower))
        return 0.5 + (shorter / longer) * 0.3

    dist = levenshtein_distance(s1_lower, s2_lower)
    return 1.0 - (dist / max_len)


def extract_entities_regex(text: str) -> List[str]:
    """
    Extract entity candidates from text using regex patterns.

    Returns list of unique entity strings. No external dependencies.
    Filters out stop words, single lowercase words, and pure numbers.
    With MNEMOSYNE_ENTITY_POSITION_FILTER set, also drops a single
    capitalized word that is capitalized by its position: a sentence opener,
    a contraction stem, or the first word of a quoted phrase.
    """
    if not text or not isinstance(text, str):
        return []

    entities: Set[str] = set()
    position_filter = _position_filter_enabled()

    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            entity = match.group(1).strip()
            # Filter: must be at least 2 chars
            if len(entity) < 2:
                continue
            # Filter out stop words (single word only); case-insensitive
            words = entity.split()
            if len(words) == 1 and entity.lower() in _STOP_WORDS:
                # @mentions / hashtags are explicit entity signals — bypass
                # the stopword filter (mirrors the lowercase check below)
                match_start = match.start(1)
                prefix_char = text[match_start - 1] if match_start > 0 else ''
                if prefix_char not in ('@', '#'):
                    continue
            # Multi-word entities: drop when ANY word is a stopword
            # (upstream contract — "The Quick Brown Fox" drops on "the").
            if len(words) > 1 and any(w.lower() in _STOP_WORDS for w in words):
                continue
            # Multi-word entities must be proper-noun shaped: every word starts
            # uppercase, is a digit, or is a stopword. Keeps "PMO City",
            # rejects quoted sentence fragments ("reliable workflow").
            if len(words) > 1 and not all(
                w[0].isupper() or w[0].isdigit() or w.lower() in _STOP_WORDS
                for w in words
            ):
                continue
            # Filter out pure numbers
            if entity.replace('.', '').replace(',', '').isdigit():
                continue
            # Template/code junk
            if any(ch in entity for ch in ';/\\'):
                continue
            # Single-word ALL-CAPS tokens (emphasis/code, not names)
            if len(words) == 1 and entity.isupper():
                continue
            # Opt-in: drop a single capitalized word capitalized by position.
            # A name still extracts from any mid-sentence occurrence, from a
            # multi-word sequence, or with an @/# prefix (those come from the
            # other patterns, which skip this).
            if (
                position_filter
                and pattern is _SINGLE_CAP_WORD
                and _capitalized_by_position(text, match.start(1), match.end(1))
            ):
                continue
            # Filter out standalone lowercase words (unless @-mentioned/#-tagged)
            # But allow @mentions and hashtags which are lowercase by nature
            if len(words) == 1 and entity[0].islower() and not entity.startswith('@') and not entity.startswith('#'):
                # Check if this entity came from an @mention or #hashtag pattern
                # by looking at the original match position in the text
                match_start = match.start(1)  # start of group 1 (the captured entity)
                if match_start > 0:
                    prefix_char = text[match_start - 1] if match_start > 0 else ''
                    if prefix_char in ('@', '#'):
                        pass  # Allow @mentions and hashtags
                    else:
                        continue
                else:
                    continue
            entities.add(entity)

    # Post-process: merge adjacent capitalized words that appear together
    # e.g., if we have "New" and "York" separately, but "New York" also matched,
    # keep only the longest match
    result = sorted(list(entities))
    
    # Remove substrings that are part of longer entities
    # But only for word-like entities (not @mentions or hashtags)
    filtered: Set[str] = set()
    for entity in result:
        is_substring = False
        for other in result:
            if other != entity and entity in other:
                # Don't remove @mentions or hashtags that happen to be substrings
                if entity.startswith('@') or entity.startswith('#'):
                    continue
                # Don't remove if the containing entity starts with @ or #
                if other.startswith('@') or other.startswith('#'):
                    continue
                is_substring = True
                break
        if not is_substring:
            filtered.add(entity)

    return sorted(list(filtered))


def find_similar_entities(entity: str, known_entities: List[str], threshold: float = 0.8) -> List[Tuple[str, float]]:
    """
    Find known entities similar to the given entity.

    Returns list of (entity_name, similarity_score) tuples, sorted by score descending.
    """
    matches: List[Tuple[str, float]] = []
    entity_len = len(entity.strip())
    for known in known_entities:
        if known == entity:
            matches.append((known, 1.0))
            continue
        # Cheap upper bound before the O(n*m) levenshtein in similarity():
        # the edit-distance score is capped at min_len/max_len and the
        # prefix/substring bonuses at 0.7 + ratio*0.3, so when the length
        # ratio alone puts every branch below the threshold the pair can
        # never match. Critical when callers pass long free-text queries
        # against thousands of short entity names.
        known_len = len(known.strip())
        max_len = max(entity_len, known_len, 1)
        ratio = min(entity_len, known_len) / max_len
        if ratio < threshold and (0.7 + ratio * 0.3) < threshold:
            continue
        sim = similarity(entity, known)
        if sim >= threshold:
            matches.append((known, sim))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def entity_extraction_performance(text: str, iterations: int = 1000) -> float:
    """
    Measure entity extraction performance.
    Returns average time per extraction in milliseconds.
    """
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        extract_entities_regex(text)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000
