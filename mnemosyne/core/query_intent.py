"""
Query Intent Classification
=============================
Regex-based classification to adjust vector/FTS weights per query type.
Determines the user's search intent to optimize hybrid scoring.

Intents and their scoring adjustments:
- temporal: "what happened last week" → boost FTS, reduce vector
- factual: "what is the database password" → balanced
- entity: "what does Denis prefer" → boost entity matching
- preference: "what does Denis like" → boost importance, reduce recency
- procedural: "how do I deploy" → boost vector (semantic), reduce recency
- general: default weights

Usage:
    from mnemosyne.core.query_intent import classify_intent, adjust_weights
    
    intent = classify_intent("what happened last Monday")
    weights = adjust_weights(default_vw=0.5, default_fw=0.3, default_iw=0.2, intent=intent)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import re


@dataclass
class QueryIntent:
    """Classification result for a query."""
    category: str  # temporal, factual, entity, preference, procedural, general
    confidence: float  # 0.0 - 1.0
    signals: list = field(default_factory=list)  # which patterns matched
    
    # Weight adjustments (multipliers)
    vec_bias: float = 1.0
    fts_bias: float = 1.0
    importance_bias: float = 1.0


# Regex patterns per intent category
INTENT_PATTERNS = [
    # TEMPORAL — "when", "last week", "yesterday", dates, etc.
    ("temporal", [
        r"\b(when|last|yesterday|today|tomorrow|ago|before|after|since|until|during|recently|lately)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(this|next|last)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d+\s+(day|week|month|year|hour|minute)s?\s+(ago|from now|later|earlier)\b",
    ]),
    
    # FACTUAL — "what is", "who is", "where is", concrete facts
    ("factual", [
        r"\bwhat\s+is\b",
        r"\bwho\s+is\b",
        r"\bwhere\s+is\b",
        r"\b(definition|define|explain|meaning)\b",
        r"\bhow\s+(many|much|long|far)\b",
    ]),
    
    # ENTITY — seeking info about a person/place/thing
    ("entity", [
        r"\b(tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\b",
        r"\b(who\s+is|what\s+does)\s+[a-z]+\b",
        r"\b(about|regarding|concerning)\s+[a-z]+\b",
    ]),
    
    # PREFERENCE — likes, dislikes, preferences
    ("preference", [
        r"\b(prefer|like|dislike|want|hate|love|enjoy|favorite|best|worst)\b",
        r"\b(should\s+i|would\s+you|do\s+you\s+recommend)\b",
        r"\b(choose|pick|select|option|choice|decide)\b",
    ]),
    
    # PROCEDURAL — "how to", "how do I", steps/processes
    ("procedural", [
        r"\bhow\s+(to|do|can|should|would)\b",
        r"\b(step|process|procedure|workflow|guide|tutorial)\b",
        r"\b(setup|install|configure|build|deploy|run|execute|start|stop)\b",
    ]),
]


# Weight adjustments per intent (applied to default vec/fts/importance weights)
INTENT_WEIGHTS = {
    "temporal": {"vec_bias": 0.6, "fts_bias": 1.5, "importance_bias": 0.8},
    "factual": {"vec_bias": 1.0, "fts_bias": 1.2, "importance_bias": 0.9},
    "entity": {"vec_bias": 1.1, "fts_bias": 1.0, "importance_bias": 1.3},
    "preference": {"vec_bias": 0.9, "fts_bias": 0.8, "importance_bias": 1.5},
    "procedural": {"vec_bias": 1.3, "fts_bias": 0.9, "importance_bias": 0.7},
    "general": {"vec_bias": 1.0, "fts_bias": 1.0, "importance_bias": 1.0},
}


def classify_intent(query: str) -> QueryIntent:
    """
    Classify the search intent of a query.
    
    Args:
        query: The user's search query
        
    Returns:
        QueryIntent with category, confidence, and weight biases
    """
    query_lower = query.lower()
    best_intent = "general"
    best_score = 0.0
    all_signals = []
    
    for category, patterns in INTENT_PATTERNS:
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query_lower):
                matches += 1
                all_signals.append(category)
        
        if matches > 0:
            # Score: base 0.3 + 0.1 per match, max 1.0
            score = min(0.3 + matches * 0.15, 1.0)
            if score > best_score:
                best_score = score
                best_intent = category
    
    weights = INTENT_WEIGHTS.get(best_intent, INTENT_WEIGHTS["general"])
    
    return QueryIntent(
        category=best_intent,
        confidence=best_score,
        signals=all_signals,
        vec_bias=weights["vec_bias"],
        fts_bias=weights["fts_bias"],
        importance_bias=weights["importance_bias"],
    )


def adjust_weights(
    base_vec: float = 0.5,
    base_fts: float = 0.3,
    base_importance: float = 0.2,
    intent: Optional[QueryIntent] = None,
) -> Tuple[float, float, float]:
    """
    Adjust hybrid scoring weights based on query intent.
    
    Args:
        base_vec: Base vector weight
        base_fts: Base FTS5 weight
        base_importance: Base importance weight
        intent: Queried intent (None = general)
        
    Returns:
        (vec_weight, fts_weight, importance_weight) tuple, normalized to sum to 1.0
    """
    if intent is None:
        intent = QueryIntent(category="general", confidence=0.0)
    
    vw = base_vec * intent.vec_bias
    fw = base_fts * intent.fts_bias
    iw = base_importance * intent.importance_bias
    
    # Normalize
    total = vw + fw + iw
    if total > 0:
        vw, fw, iw = vw / total, fw / total, iw / total
    
    return (vw, fw, iw)
