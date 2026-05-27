"""
Weibull Decay Scoring for Mnemosyne
=====================================
Replaces uniform exponential temporal decay with memory-type-specific
Weibull distribution parameters (shape k, scale eta).

The Weibull distribution provides:
- Decreasing hazard rate (k < 1): memories decay slower over time (profiles, preferences)
- Constant hazard rate (k = 1): exponential, same as current behavior (general facts)
- Increasing hazard rate (k > 1): memories decay faster over time (events, requests)

Memory system compatibility: mirrors the per-type eta/k parameters from memory system's
Weibull decay system.

Usage:
    from mnemosyne.core.weibull import weibull_boost, WEIBULL_PARAMS
    
    boost = weibull_boost(timestamp, query_time, memory_type="preference")
"""

import math
from datetime import datetime, timezone
from typing import Optional, Union


# Per-memory-type Weibull parameters (k=shape, eta=scale in hours)
# Higher eta = slower decay, lower k = more long-term retention
WEIBULL_PARAMS = {
    # --- Long-term stable memories ---
    "profile":      {"k": 0.3, "eta": 8760.0},   # ~1 year scale, very slow decay
    "preference":   {"k": 0.4, "eta": 4380.0},   # ~6 months, slow decay
    "relationship": {"k": 0.35, "eta": 8760.0},  # ~1 year, people knowledge
    "learning":     {"k": 0.7, "eta": 1440.0},   # ~2 months
    
    # --- Medium-term working knowledge ---
    "fact":         {"k": 0.8, "eta": 720.0},    # ~1 month, near-exponential
    "entity":       {"k": 0.5, "eta": 4380.0},   # ~6 months, slow
    "setup":        {"k": 0.6, "eta": 2160.0},   # ~3 months, moderate decay
    "pattern":      {"k": 0.6, "eta": 1680.0},   # ~2.3 months
    "context":      {"k": 0.85, "eta": 360.0},   # ~15 days, session context
    "observation":  {"k": 0.9, "eta": 480.0},    # ~20 days
    "artifact":     {"k": 0.75, "eta": 2160.0},  # ~3 months, code/docs/files
    
    # --- Decaying / time-sensitive ---
    "project":      {"k": 0.85, "eta": 1080.0},  # ~45 days
    "goal":         {"k": 0.9, "eta": 720.0},    # ~1 month
    "decision":     {"k": 1.0, "eta": 336.0},    # ~2 weeks
    "commitment":   {"k": 1.0, "eta": 240.0},    # ~10 days, deadlines
    
    # --- Fast-decaying ---
    "event":        {"k": 1.2, "eta": 168.0},    # ~1 week, fast decay
    "instruction":  {"k": 0.9, "eta": 480.0},    # ~20 days, how-to knowledge
    "error":        {"k": 1.1, "eta": 336.0},    # ~2 weeks, medium-fast
    "issue":        {"k": 1.1, "eta": 336.0},    # ~2 weeks, medium-fast
    "request":      {"k": 1.5, "eta": 72.0},     # ~3 days, fastest
    
    # --- Default ---
    "general":      {"k": 1.0, "eta": 168.0},    # ~1 week, exponential
}


# Fallback to simple exponential if memory_type unknown
DEFAULT_HALFLIFE_HOURS = 168.0  # 1 week


def weibull_boost(
    timestamp: Optional[str],
    query_time: Optional[datetime] = None,
    memory_type: str = "general",
    halflife_hours: Optional[float] = None,
) -> float:
    """
    Compute temporal boost using Weibull decay.
    
    Args:
        timestamp: ISO timestamp of the memory
        query_time: Reference time for scoring (None = now UTC)
        memory_type: Memory classification (maps to Weibull params)
        halflife_hours: Override; force simple exponential if set
        
    Returns:
        Boost factor 0.0 - 1.0 (1.0 = most recent, decaying toward 0.0)
    """
    if timestamp is None:
        return 0.0
    
    if query_time is None:
        query_time = datetime.now(timezone.utc)
    
    # Parse timestamp
    try:
        # Try ISO format parsing
        if isinstance(timestamp, str):
            # Handle offset-aware timestamps
            ts = timestamp.replace("Z", "+00:00")
            # datetime.fromisoformat handles ISO 8601 in Python 3.7+
            try:
                mem_time = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                # Try common formats
                for fmt in [
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                ]:
                    try:
                        mem_time = datetime.strptime(ts[:26], fmt)
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    return 0.0
        elif isinstance(timestamp, datetime):
            mem_time = timestamp
        else:
            return 0.0
    except Exception:
        return 0.0
    
    # Normalize to UTC
    if hasattr(mem_time, 'tzinfo') and mem_time.tzinfo is not None:
        mem_time = mem_time.astimezone(timezone.utc).replace(tzinfo=None)
    if hasattr(query_time, 'tzinfo') and query_time.tzinfo is not None:
        query_time = query_time.astimezone(timezone.utc).replace(tzinfo=None)
    
    # Clamp: future timestamps get boost 1.0
    if mem_time > query_time:
        return 1.0
    
    # Compute age in hours
    delta = query_time - mem_time
    age_hours = delta.total_seconds() / 3600.0
    
    # Use Weibull if memory_type has params and no explicit halflife override
    if halflife_hours is not None:
        # Simple exponential fallback
        if halflife_hours <= 0:
            return 0.0
        return math.exp(-age_hours / halflife_hours)
    
    params = WEIBULL_PARAMS.get(memory_type)
    if params is None:
        # Unknown type → simple exponential with default halflife
        return math.exp(-age_hours / DEFAULT_HALFLIFE_HOURS)
    
    k = params["k"]
    eta = params["eta"]
    
    # Weibull survival function: exp(-(t/eta)^k)
    if eta <= 0:
        return 0.0
    
    return math.exp(-((age_hours / eta) ** k))


def weibull_decay_factor(
    age_hours: float,
    memory_type: str = "general",
) -> float:
    """
    Direct age-based Weibull computation (no timestamp parsing).
    
    Args:
        age_hours: Age in hours
        memory_type: Memory classification
        
    Returns:
        Decay factor 0.0 - 1.0
    """
    if age_hours <= 0:
        return 1.0
    
    params = WEIBULL_PARAMS.get(memory_type)
    if params is None:
        return math.exp(-age_hours / DEFAULT_HALFLIFE_HOURS)
    
    k = params["k"]
    eta = params["eta"]
    if eta <= 0:
        return 0.0
    
    return math.exp(-((age_hours / eta) ** k))
