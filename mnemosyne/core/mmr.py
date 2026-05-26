"""
MMR (Maximal Marginal Relevance) Re-Ranking for Mnemosyne
==========================================================
Diversity re-ranking to reduce redundancy in recall results.

Algorithm: MMR balances relevance (high score) against novelty
(dissimilarity to already-selected results). This prevents the
top-k results from being near-duplicates of each other.

Parameters:
    lambda_param: 0.0 = pure diversity, 1.0 = pure relevance.
    Default 0.7 balances toward relevance with some diversity penalty.

Usage:
    from mnemosyne.core.mmr import mmr_rerank
    
    diverse_results = mmr_rerank(results, lambda_param=0.7, top_k=10)
"""

from typing import List, Dict, Callable, Optional
import math


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two text strings.
    Uses word-level overlap for speed.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    
    if not words_a or not words_b:
        return 0.0
    
    intersection = words_a & words_b
    union = words_a | words_b
    
    return len(intersection) / len(union)


def mmr_rerank(
    results: List[Dict],
    lambda_param: float = 0.7,
    top_k: int = 10,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> List[Dict]:
    """
    Re-rank results using MMR for diversity.
    
    Args:
        results: List of result dicts, each with 'content' and 'score' keys
        lambda_param: Relevance vs diversity tradeoff (0.0-1.0)
        top_k: Number of results to return
        similarity_fn: Custom similarity function (text_a, text_b) -> float.
                       Defaults to Jaccard word overlap.
                       
    Returns:
        Re-ranked list of results
    """
    if len(results) <= 1:
        return results[:top_k]
    
    if similarity_fn is None:
        similarity_fn = _jaccard_similarity
    
    # Sort by score initially
    sorted_results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
    
    selected = [sorted_results[0]]
    remaining = sorted_results[1:]
    
    while remaining and len(selected) < top_k:
        mmr_scores = []
        for candidate in remaining:
            relevance = candidate.get("score", 0)
            
            # Max similarity to any already-selected result
            max_sim = max(
                similarity_fn(candidate.get("content", ""), s.get("content", ""))
                for s in selected
            )
            
            # MMR formula: λ * relevance - (1-λ) * max_similarity
            mmr = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            mmr_scores.append(mmr)
        
        # Select best MMR-scored candidate
        best_idx = mmr_scores.index(max(mmr_scores))
        selected.append(remaining.pop(best_idx))
    
    # Add remaining if we couldn't fill top_k
    if len(selected) < top_k:
        selected.extend(remaining[:top_k - len(selected)])
    
    return selected
