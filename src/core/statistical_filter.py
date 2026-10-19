from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.config import FILTERING_K, MAX_EVIDENCE_ITEMS, MIN_PHI_SCORE

logger = logging.getLogger(__name__)

class StatisticalFilter:
    """Adaptive statistical Heaviside filter for context selection."""

    def __init__(self, k: float = FILTERING_K):
        self.k = k

    def compute_threshold(
        self, scores: np.ndarray | torch.Tensor, k: Optional[float] = None
    ) -> tuple[float, float, float]:
        """Compute adaptive threshold from similarity score distribution.

        Audit fix (item 9): pure NumPy - the previous version round-tripped
        tiny score arrays through CUDA on every call.
        """
        k = k if k is not None else self.k
        arr = np.asarray(scores, dtype=np.float64).ravel()
        if arr.size == 0:
            return 0.0, 0.0, 0.0
        if not np.isfinite(arr).all():
            raise ValueError("Non-finite scores supplied to the statistical filter")
        mu = float(arr.mean())
        sigma = float(arr.std())
        return mu + k * sigma, mu, sigma

    def heaviside_filter(
        self, scores: np.ndarray | torch.Tensor, k: Optional[float] = None
    ) -> np.ndarray:
        """Apply Heaviside step function: 1 if score >= threshold else 0."""
        threshold, _, _ = self.compute_threshold(scores, k=k)
        arr = np.asarray(scores, dtype=np.float64)
        return (arr >= threshold).astype(int)

    def filter_contexts(
        self,
        contexts: list[dict],
        scores: np.ndarray,
        k: Optional[float] = None,
        score_key: str = "phi_final",
        max_top_k: int = MAX_EVIDENCE_ITEMS,  # align with config K<=8
        min_score: float = MIN_PHI_SCORE,
    ) -> list[dict]:
        """Filter retrieved contexts using dual-gating.

        Gate 1: absolute quality floor - reject anything below min_score.
        Gate 2: relative Heaviside threshold on surviving candidates.
        """
        if not contexts:
            return []

        scores = np.asarray(scores, dtype=float)
        if len(contexts) != len(scores):
            raise ValueError(f"Mismatch: {len(contexts)} contexts vs {len(scores)} scores")

        # Gate 1: absolute quality floor
        above_floor = scores >= min_score
        if not above_floor.any():
            logger.info(
                "All %d candidates below absolute floor (%.4f); returning empty.",
                len(contexts), min_score,
            )
            return []

        gate1_contexts = [ctx for ctx, keep in zip(contexts, above_floor) if keep]
        gate1_scores = scores[above_floor]

        # Gate 2: relative Heaviside threshold on survivors
        mask = self.heaviside_filter(gate1_scores, k=k)
        filtered = [
            {**ctx, score_key: float(gate1_scores[i])}
            for i, ctx in enumerate(gate1_contexts)
            if mask[i] == 1
        ]

        if not filtered:
            logger.info("All candidates fell below Heaviside threshold; returning empty.")
            return []

        filtered.sort(key=lambda c: c[score_key], reverse=True)
        if max_top_k is not None:
            filtered = filtered[:max_top_k]

        return filtered

    def compute_retrieval_probability(self, scores: np.ndarray) -> np.ndarray:
        """Softmax normalization over candidate scores ."""
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return np.empty((0,), dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite scores supplied to softmax")
        shifted = scores - scores.max()
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum()

    def compute_context_quality(
        self,
        relevance: float,
        graph_consistency: float,
        predicate_accuracy: float,
        weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> float:
        """Weighted quality score across relevance, graph consistency, and predicates."""
        w1, w2, w3 = weights
        return w1 * relevance + w2 * graph_consistency + w3 * predicate_accuracy

    def hallucination_probability(self, context_quality: float) -> float:
        return max(0.0, 1.0 - context_quality)

    def score_variance(self, scores: np.ndarray) -> float:
        return float(np.var(np.asarray(scores, dtype=float)))
