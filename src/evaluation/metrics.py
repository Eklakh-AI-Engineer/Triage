"""Robustness metrics beyond the base-paper protocol.

The base paper  reports the 5-dimension scoring, the
8-criteria selection win-rate, and retrieval latency - with no significance
testing and no retrieval-quality measurement. These four metrics make the
evaluation strictly more rigorous while remaining directly comparable.
"""
from __future__ import annotations

import math
from math import comb

import numpy as np

def ndcg_at_k(ranked_relevance: list[int], k: int = 8) -> float:
    """NDCG@K over binary relevance judgments of the ranked retrieval list."""
    if not ranked_relevance:
        return 0.0
    rel = np.asarray(ranked_relevance[:k], dtype=np.float64)
    dcg = float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))
    ideal = np.sort(np.asarray(ranked_relevance, dtype=np.float64))[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_relevance: list[int], k: int = 8) -> float:
    total_relevant = sum(ranked_relevance)
    if total_relevant == 0:
        return 0.0
    return sum(ranked_relevance[:k]) / total_relevant

def mcnemar_test(b_correct: list[bool], a_correct: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (same questions).

    b = system B (e.g., HCAD), a = baseline. Returns the exact two-sided
    p-value - the significance test the base paper never runs.
    """
    if len(b_correct) != len(a_correct) or not b_correct:
        raise ValueError("McNemar requires paired, non-empty outcomes")
    n01 = sum(1 for b, a in zip(b_correct, a_correct) if b and not a)
    n10 = sum(1 for b, a in zip(b_correct, a_correct) if not b and a)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "significant_at_05": False}
    # exact two-sided binomial test on the discordant pairs
    p = sum(comb(n, i) for i in range(min(n01, n10) + 1)) / (2 ** (n - 1))
    p = min(1.0, 2 * p * 0.5)
    return {"n01": n01, "n10": n10, "p_value": round(p, 5),
            "significant_at_05": p < 0.05}

def citation_precision(answer: str, evidence_texts: list[str]) -> dict:
    """Citation precision: cited tags that point at real retrieved items.

    Directly measures the paper's stated goal (hallucination mitigation):
    what fraction of [Doc i]/[Evidence i] citations in the answer resolve to
    retrieved evidence items?
    """
    import re
    tags = re.findall(r"\[(?:Doc|Evidence|Context)\s*(\d+)\]", answer or "")
    if not tags:
        return {"n_citations": 0, "n_valid": 0, "precision": None}
    valid = sum(1 for t in tags if 1 <= int(t) <= len(evidence_texts))
    n = len(tags)
    return {"n_citations": n, "n_valid": valid,
            "precision": round(valid / n, 4) if n else None}
