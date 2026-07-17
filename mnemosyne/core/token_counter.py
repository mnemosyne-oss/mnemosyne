"""
Mnemosyne Token Counter
Lightweight token estimation for memory context benchmarking.
Uses tiktoken if available, falls back to chars/4 heuristic.
"""


# Try to import tiktoken, but don't fail if missing
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _ENCODING = None


def estimate_tokens(text: str, model: str = "default") -> int:
    """
    Estimate token count for a given text.
    
    Args:
        text: The text to estimate
        model: Optional model hint (claude, gpt4, etc.)
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    if _TIKTOKEN_AVAILABLE and _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    
    # Fallback: chars/4 is a decent approximation for English/Spanish
    return len(text) // 4


def estimate_cost(tokens: int, model: str = "claude-sonnet-4") -> dict:
    """
    Estimate API cost for injected prompt tokens.
    
    Args:
        tokens: Number of prompt tokens
        model: Model identifier for pricing lookup
        
    Returns:
        Dictionary with cost estimates for common models
    """
    # Pricing per 1M tokens (input) - update as needed
    PRICING = {
        "claude-sonnet-4": 3.00,
        "claude-haiku": 0.80,
        "gpt-4o": 2.50,
        "gpt-4o-mini": 0.15,
        "default": 3.00,
    }
    
    rate = PRICING.get(model, PRICING["default"])
    cost = (tokens / 1_000_000) * rate
    
    return {
        "tokens": tokens,
        "model": model,
        "cost_usd": round(cost, 6),
        "rate_per_1m": rate,
    }
