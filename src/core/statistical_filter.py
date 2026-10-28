from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.config import FILTERING_K, MAX_EVIDENCE_ITEMS, MIN_PHI_SCORE, tau_sat as _tau_sat

logger = logging.getLogger(__name__)

class StatisticalFilter:
    """Adaptive statistical Heaviside filter for context selection."""

    def __init__(self, k: float = FILTERING_K):
        self.k = k

    def compute_threshold(
        self, scores: np.ndarray | torch.Tensor, k: Optional[float] = None,
        tau_sat: Optional[float] = None,
    ) -> tuple[float, float, float]:
        """Compute adaptive threshold from similarity score distribution.

        Audit fix (item 9): pure NumPy - the previous version round-tripped
        tiny score arrays through CUDA on every call.

        Deliverable-B saturation cap: when a calibrated ceiling tau_sat is
        available (25th percentile of gold development candidate scores),

            tau_eff = min(tau_sat, max(Phi_min, mu + k * sigma))

        so a homogeneous candidate pool cannot push the adaptive threshold
        past the calibrated ceiling and starve the prompt of evidence.
        """
        k = k if k is not None else self.k
        if tau_sat is None:
            tau_sat = _tau_sat()
        arr = np.asarray(scores, dtype=np.float64).ravel()
        if arr.size == 0:
            return 0.0, 0.0, 0.0
        if not np.isfinite(arr).all():
            raise ValueError("Non-finite scores supplied to the statistical filter")
        mu = float(arr.mean())
        sigma = float(arr.std())
        threshold = mu + k * sigma
        if tau_sat is not None:
            threshold = min(float(tau_sat), max(MIN_PHI_SCORE, threshold))
        return threshold, mu, sigma

    def heaviside_filter(
        self, scores: np.ndarray | torch.Tensor, k: Optional[float] = None,
        tau_sat: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Heaviside step function: 1 if score >= threshold else 0."""
        threshold, _, _ = self.compute_threshold(scores, k=k, tau_sat=tau_sat)
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

        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(contexts) != len(scores):
            raise ValueError(f"Mismatch: {len(contexts)} contexts vs {len(scores)} scores")
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite scores supplied to context filtering")
        if not np.isfinite(min_score):
            raise ValueError("min_score must be finite")
        if max_top_k is not None and max_top_k < 0:
            raise ValueError("max_top_k must be non-negative or None")

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

        # Gate 2: relative Heaviside threshold on survivors. Bound the
        # threshold by the best eligible score: mu + k*sigma can exceed
        # max(scores) for skewed/small pools, which otherwise rejects every
        # candidate even though at least one passed Gate 1. This is not a
        # forced admission of low-quality evidence: Gate 1 remains mandatory.
        threshold, mu, sigma = self.compute_threshold(gate1_scores, k=k)
        threshold = min(threshold, float(np.max(gate1_scores)))
        mask = gate1_scores >= threshold
        filtered = [
            {**ctx, score_key: float(gate1_scores[i])}
            for i, ctx in enumerate(gate1_contexts)
            if mask[i]
        ]

        logger.debug(
            "Dual gate: input=%d above_floor=%d kept=%d min=%.4f "
            "max=%.4f mean=%.4f std=%.4f threshold=%.4f",
            len(contexts), len(gate1_contexts), len(filtered),
            float(np.min(gate1_scores)), float(np.max(gate1_scores)),
            mu, sigma, threshold,
        )

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
